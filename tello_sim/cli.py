"""An interactive prompt for flying the simulated Tello by typing library calls.

The buttons in the 3D view are quicker, and teach you nothing about the API.
Here you type the same calls a flight script would contain --

    tello> takeoff()
    tello> move_forward(100)
    tello> rotate_clockwise(90)
    tello> get_battery()

-- watch them happen in the 3D view, and get the drone's real answer, including
its refusals. `help` prints the whole API with units and limits; `help <name>`
explains one call in detail.

The object at the prompt is a real `djitellopy.Tello` talking over real UDP,
unless you asked for the in-process path, in which case it is `SimTello`, which
has the same methods.  Either way, code that works here is code that works.

Three things can be typed at the prompt and the console works out which:

  * Python -- `move_forward(50)`, or a whole loop, or `for i in range(4): ...`
  * a raw SDK line -- `forward 50`, `cw 90`, `battery?`, exactly as it goes
    over the wire, which is worth seeing at least once
  * a console word -- `help`, `state`, `log`, `where`, `quit`
"""

from __future__ import annotations

import code
import os
import sys
import textwrap
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Optional

from . import cli_docs
from .cli_docs import BY_NAME, ENTRIES, GROUPS, SDK_VERBS, STANDALONE_SDK

HISTORY_FILE = Path.home() / ".tello_sim_history"

# Commands the firmware answers with silence rather than "ok".
NO_REPLY = ("rc", "emergency")


# ----------------------------------------------------------------------
# Colour, only when it will actually render
# ----------------------------------------------------------------------

_COLOUR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(text: str, code_: str) -> str:
    return f"\033[{code_}m{text}\033[0m" if _COLOUR else text


def dim(text: str) -> str:
    return _c(text, "2")


def bold(text: str) -> str:
    return _c(text, "1")


def green(text: str) -> str:
    return _c(text, "32")


def red(text: str) -> str:
    return _c(text, "31")


def cyan(text: str) -> str:
    return _c(text, "36")


def yellow(text: str) -> str:
    return _c(text, "33")


# ----------------------------------------------------------------------
# The connection
# ----------------------------------------------------------------------


class Link:
    """Whatever the console is flying, plus a log of what has been sent.

    Two things can sit behind this: a real `djitellopy.Tello` speaking UDP to
    the simulator, or a `SimTello` wired straight into it.  The console does not
    care which, and neither should anything you type -- that is the point.
    """

    def __init__(self, api: Any, kind: str, simulator=None, drone_name: str = "tello-1"):
        self.api = api
        self.kind = kind  # "udp" or "inproc"
        self.simulator = simulator
        self.drone_name = drone_name
        self.history: list[dict] = []
        self.echo: Optional[Callable[[dict], None]] = None
        self.names: list[str] = [drone_name]
        self._install_tap(api, drone_name)

    def also(self, api: Any, drone_name: str) -> None:
        """Log a second drone through the same console.

        A swarm command that only showed drone 1 answering would be worse than
        no log at all -- the interesting part of flying three is what the other
        two did.
        """
        self.names.append(drone_name)
        self._install_tap(api, drone_name)

    # -- logging -------------------------------------------------------

    def _record(self, command: str, response: str, elapsed: float, drone: str = "") -> None:
        entry = {
            "command": command,
            "response": response,
            "seconds": elapsed,
            "drone": drone or self.drone_name,
            "sim_time": None if self.simulator is None else round(self.simulator.time, 2),
        }
        self.history.append(entry)
        if self.echo is not None:
            self.echo(entry)

    def _install_tap(self, api: Any, name: str) -> None:
        """Log every command, whichever path it took to get out.

        Wrapped on the instance rather than the class, so nothing else in the
        process is affected and a second console is independent of this one.
        """
        if self.kind == "udp":
            original_return = api.send_command_with_return
            original_silent = api.send_command_without_return

            def with_return(command: str, *args, **kwargs):
                started = time.perf_counter()
                try:
                    response = original_return(command, *args, **kwargs)
                except Exception as exc:  # noqa: BLE001 - logged, then re-raised
                    self._record(command, f"raised {type(exc).__name__}",
                                 time.perf_counter() - started, name)
                    raise
                self._record(command, str(response), time.perf_counter() - started, name)
                return response

            def without_return(command: str):
                original_silent(command)
                self._record(command, "(no reply)", 0.0, name)

            api.send_command_with_return = with_return
            api.send_command_without_return = without_return
            return

        original_send = api._send
        original_rc = api.send_rc_control

        def send(command: str, *args, **kwargs):
            started = time.perf_counter()
            try:
                response = original_send(command, *args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                self._record(command, f"raised {type(exc).__name__}",
                             time.perf_counter() - started, name)
                raise
            self._record(command, str(response), time.perf_counter() - started, name)
            return response

        def rc(left_right: int, forward_back: int, up_down: int, yaw: int):
            original_rc(left_right, forward_back, up_down, yaw)
            self._record(f"rc {left_right} {forward_back} {up_down} {yaw}", "(no reply)", 0.0, name)

        api._send = send
        api.send_rc_control = rc

    # -- raw SDK -------------------------------------------------------

    def send_raw(self, line: str) -> str:
        """Send a line exactly as it goes over the wire."""
        verb = line.split()[0].lower()
        if self.kind == "udp":
            if verb in NO_REPLY:
                self.api.send_command_without_return(line)
                return "(no reply)"
            return str(self.api.send_command_with_return(line))

        if verb in NO_REPLY:
            self.api._drone.submit(line)
            self._record(line, "(no reply)", 0.0)
            return "(no reply)"
        return str(self.api._send(line))

    # -- telemetry -----------------------------------------------------

    def state(self) -> dict:
        return self.api.get_current_state()

    def true_pose(self):
        """Ground truth, when the console owns the simulation."""
        if hasattr(self.api, "true_position"):
            return self.api.true_position(), self.api.true_yaw_deg()
        if self.simulator is not None:
            import numpy as np

            drone = self.simulator.get(self.drone_name)
            return drone.state.position.copy(), float(np.rad2deg(drone.state.yaw))
        return None, None


# ----------------------------------------------------------------------
# Rendering the reference
# ----------------------------------------------------------------------


def _wrap(text: str, indent: str = "      ", width: int = 78) -> str:
    return textwrap.fill(
        " ".join(text.split()), width=width, initial_indent=indent, subsequent_indent=indent
    )


def render_index(api: Any = None) -> str:
    """Every call, grouped, one line each.

    Filtered to what `api` really provides: the two paths differ slightly (the
    query_* family is UDP-only, ground truth is simulator-only), and a menu
    offering something that raises NameError is worse than a shorter menu.
    """
    out = [
        "",
        bold("  The Tello API, as you would type it"),
        dim("  help <name> explains one call: units, limits, and what it becomes on the wire."),
        "",
    ]
    for key, title in GROUPS:
        entries = [
            entry
            for entry in cli_docs.entries_in(key)
            if api is None or _available(api, entry)
        ]
        if not entries:
            continue
        out.append(f"  {bold(title)}")
        for entry in entries:
            call = entry.call
            if len(call) <= 48:
                out.append(f"    {cyan(call)}{' ' * (48 - len(call))} {dim(entry.summary)}")
            else:
                # A long signature would push the summary off the edge; give it
                # its own line rather than wrapping into a ragged column.
                out.append(f"    {cyan(call)}")
                out.append(f"    {' ' * 48} {dim(entry.summary)}")
        out.append("")
    out.append(f"  {bold('Console words')}")
    for word, summary in CONSOLE_HELP:
        out.append(f"    {yellow(word)}{' ' * max(1, 48 - len(word))} {dim(summary)}")
    out.append("")
    out.append(dim("  Raw SDK lines work too: type  forward 50  or  battery?  to see the wire."))
    out.append("")
    return "\n".join(out)


def _available(api: Any, entry: cli_docs.Entry) -> bool:
    return any(hasattr(api, name) for name in (entry.name, *entry.aliases))


def render_entry(entry: cli_docs.Entry) -> str:
    out = ["", f"  {bold(cyan(entry.call))}", ""]
    out.append(_wrap(entry.summary, indent="  "))
    if entry.detail:
        out.append("")
        out.append(_wrap(entry.detail, indent="  "))
    if entry.limits:
        out.append("")
        out.append(f"  {bold('Limits')}")
        out.append(_wrap(entry.limits, indent="    "))
    if entry.hardware:
        out.append("")
        out.append(f"  {bold('On the real drone')}")
        out.append(_wrap(entry.hardware, indent="    "))
    if entry.sdk:
        out.append("")
        out.append(f"  {bold('On the wire')}   {dim(entry.sdk)}")
    if entry.example:
        out.append("")
        out.append(f"  {bold('Try')}           {green(entry.example)}")
    if entry.aliases:
        out.append("")
        out.append(f"  {bold('Also called')}   {dim(', '.join(entry.aliases))}")
    out.append("")
    return "\n".join(out)


def render_group(key: str, title: str, api: Any = None) -> str:
    out = ["", f"  {bold(title)}", ""]
    shown = 0
    for entry in cli_docs.entries_in(key):
        if api is not None and not _available(api, entry):
            continue
        shown += 1
        out.append(f"    {cyan(entry.call)}")
        out.append(_wrap(entry.summary, indent="      "))
        if entry.limits:
            out.append(_wrap(dim(entry.limits), indent="      "))
        out.append("")
    if not shown:
        name = type(api).__name__ if api is not None else "this object"
        out.append(
            _wrap(
                f"Nothing in this group is available on {name}. "
                "Ground truth and simulated sleep are SimTello's "
                "(run with --in-process); the console word `where` gives you "
                "the position either way.",
                indent="    ",
            )
        )
        out.append("")
    return "\n".join(out)


CONSOLE_HELP = [
    ("help / help <name>", "this list, or one call in detail"),
    ("state", "the drone's own telemetry, as it broadcasts it"),
    ("where", "true position in the room -- simulator only"),
    ("log [n]", "the last n commands, their answers and how long they took"),
    ("sdk", "the raw SDK verbs you can type directly"),
    ("demo", "a short flight to copy, one line at a time"),
    ("view", "the 3D view's address"),
    ("run <file>", "execute a file of console lines"),
    ("quit / exit", "land and stop"),
]


DEMO = """\
# A first flight. Type these one at a time and watch the 3D view.

takeoff()                    # up to about 80 cm, and hold
get_battery()                # how much is left
move_forward(100)            # a metre, in whatever direction the nose points
rotate_clockwise(90)         # turn right; 'forward' now means something else
move_forward(100)
get_distance_tof()           # what is underneath -- not the same as height
land()

# The same square, as a loop:
takeoff()
for i in range(4):
    move_forward(100)
    rotate_clockwise(90)

land()

# Where did it actually end up, versus where it started?
where
"""


# ----------------------------------------------------------------------
# The console
# ----------------------------------------------------------------------


class TelloConsole(code.InteractiveConsole):
    """A Python prompt that also understands raw SDK lines and console words."""

    def __init__(self, link: Link, extras: Optional[dict] = None, viewer_url: str = ""):
        self.link = link
        self.viewer_url = viewer_url
        self.show_traceback = False
        namespace = self._build_namespace(extras or {})
        super().__init__(locals=namespace)
        link.echo = self._echo_command
        self._setup_readline()

    # -- namespace -----------------------------------------------------

    def _build_namespace(self, extras: dict) -> dict:
        import numpy as np

        api = self.link.api
        namespace: dict[str, Any] = {
            "__name__": "__console__",
            "tello": api,
            "drone": api,
            "t": api,
            "np": np,
            "link": self.link,
        }
        namespace.update(extras)

        # Bare aliases, so `takeoff()` works as well as `tello.takeoff()`.
        # Only for names the object really has, so nothing is promised that the
        # underlying library does not provide.
        for entry in ENTRIES:
            for name in (entry.name, *entry.aliases):
                if hasattr(api, name) and name not in namespace:
                    namespace[name] = getattr(api, name)
        for name in dir(api):
            if name.startswith("query_") and name not in namespace:
                namespace[name] = getattr(api, name)
        return namespace

    # -- readline ------------------------------------------------------

    def _setup_readline(self) -> None:
        try:
            import readline
            import rlcompleter
        except ImportError:  # pragma: no cover - Windows without pyreadline
            return

        words = set(CONSOLE_WORDS) | set(BY_NAME) | set(SDK_VERBS)
        base = rlcompleter.Completer(self.locals)
        cache: list[str] = []

        def complete(text: str, state: int):
            if state == 0:
                buffer = readline.get_line_buffer().lstrip()
                if buffer.startswith(("help ", "?")):
                    found = {name for name in BY_NAME if name.startswith(text)}
                else:
                    found = {word for word in words if word.startswith(text)}
                    index = 0
                    while True:
                        item = base.complete(text, index)
                        if item is None:
                            break
                        found.add(item)
                        index += 1
                cache[:] = sorted(found)
            return cache[state] if state < len(cache) else None

        readline.set_completer(complete)
        readline.parse_and_bind("tab: complete")
        readline.set_completer_delims(" \t\n`~!@#$%^&*()=+[{]}\\|;:'\",<>/?")
        try:
            readline.read_history_file(HISTORY_FILE)
        except (OSError, FileNotFoundError):
            pass
        readline.set_history_length(2000)
        import atexit

        atexit.register(self._save_history)

    def _save_history(self) -> None:
        try:
            import readline

            readline.write_history_file(HISTORY_FILE)
        except Exception:  # noqa: BLE001 - history is a convenience, never fatal
            pass

    def write(self, data: str) -> None:
        """Everything the console says goes to stdout, in order.

        The base class writes to stderr, which interleaves badly with the
        values `displayhook` prints to stdout: a whole session comes out with
        the answers at the bottom.
        """
        sys.stdout.write(data)
        sys.stdout.flush()

    # -- command echo --------------------------------------------------

    def _echo_command(self, entry: dict) -> None:
        response = entry["response"]
        previous = self.link.history[-2] if len(self.link.history) > 1 else None
        if (
            previous is not None
            and previous["command"] == entry["command"]
            and previous["response"] == response
            and previous["drone"] == entry["drone"]
        ):
            # djitellopy retries a refused command three more times. The
            # refusal is worth seeing; four copies of it are not.
            return
        colour = red if response.startswith("error") or "raised" in response else green
        took = f"{entry['seconds']:.2f}s" if entry["seconds"] >= 0.005 else ""
        who = f"{entry['drone']} " if len(self.link.names) > 1 else ""
        self.write(
            f"  {dim('->')} {dim(who + entry['command']):<28} {colour(response):<22} {dim(took)}\n"
        )
        if response.startswith("error") and self.link.kind == "udp":
            self.write(dim("     djitellopy retries a refusal three more times before raising.\n"))

    # -- input dispatch ------------------------------------------------

    def push(self, line: str) -> bool:
        # Only intercept at the start of a statement; inside a block (a for
        # loop being typed) everything is Python.
        if not self.buffer:
            handled = self._intercept(line)
            if handled:
                return False
        return super().push(line)

    def _intercept(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            return bool(stripped)

        if stripped.startswith(":"):
            return self._console_word(stripped[1:].strip()) or True

        head = stripped.split()[0].lower()
        looks_pythonic = "(" in stripped or "=" in stripped

        if head in CONSOLE_WORDS and not looks_pythonic:
            return self._console_word(stripped)

        # A lone word. `takeoff` is a whole SDK line and means it; `move_forward`
        # is a function that was typed without its brackets, and printing the
        # bound method's repr helps nobody.
        if stripped == head and not looks_pythonic:
            if head in STANDALONE_SDK:
                self._raw(stripped)
                return True
            if head in BY_NAME:
                self._needs_brackets(BY_NAME[head])
                return True
            if head in SDK_VERBS:
                self.write(
                    f"  {dim('The SDK verb')} {bold(head)} {dim('takes an argument, e.g. ')}"
                    + green(f"{head} 50")
                    + "\n"
                )
                return True

        if head in SDK_VERBS and not looks_pythonic:
            self._raw(stripped)
            return True

        return False

    def _needs_brackets(self, entry: cli_docs.Entry) -> None:
        self.write(
            f"  {dim('That is a function -- call it:')} {green(entry.call)}\n"
            f"  {dim(entry.summary)}\n"
            f"  {dim('help ' + entry.name + ' for the detail.')}\n"
        )

    def _raw(self, line: str) -> None:
        try:
            response = self.link.send_raw(line)
        except Exception as exc:  # noqa: BLE001 - a wire error is not a crash
            self.write(f"  {red(type(exc).__name__)}: {exc}\n")
            return
        # The UDP path logs through the tap; the in-process path may not have,
        # so only echo here if nothing was recorded for this line.
        if not self.link.history or self.link.history[-1]["command"] != line:
            self.write(f"  {dim('->')} {dim(line):<28} {green(response)}\n")

    # -- console words -------------------------------------------------

    def _console_word(self, line: str) -> bool:
        parts = line.split()
        word, args = parts[0].lower(), parts[1:]

        if word in ("quit", "exit", "bye"):
            raise SystemExit(0)

        if word in ("help", "?"):
            self._help(args)
            return True

        if word == "state":
            self._state()
            return True

        if word == "where":
            self._where()
            return True

        if word == "log":
            count = int(args[0]) if args and args[0].isdigit() else 15
            self._log(count)
            return True

        if word == "sdk":
            self._sdk()
            return True

        if word == "demo":
            self.write("\n" + green(DEMO) + "\n")
            return True

        if word == "view":
            if self.viewer_url:
                self.write(f"\n  3D view: {cyan(self.viewer_url)}\n\n")
            else:
                self.write(
                    "\n  No 3D view in this session. Start one with "
                    + cyan("python run_sim3d.py")
                    + "\n  in another terminal, then rerun this with "
                    + cyan("--attach")
                    + ".\n\n"
                )
            return True

        if word == "run":
            if not args:
                self.write("  Usage: run <file>\n")
                return True
            self.run_file(Path(args[0]))
            return True

        if word == "traceback":
            self.show_traceback = not self.show_traceback
            self.write(f"  Full tracebacks {'on' if self.show_traceback else 'off'}.\n")
            return True

        if word == "clear":
            self.write("\033[2J\033[H" if _COLOUR else "\n" * 40)
            return True

        return False

    def _help(self, args: list[str]) -> None:
        if not args:
            self.write(render_index(self.link.api))
            return
        target = args[0].strip("()").lstrip("?")
        if target in BY_NAME:
            entry = BY_NAME[target]
            self.write(render_entry(entry))
            if not _available(self.link.api, entry):
                self.write(
                    dim(f"  Not available on {type(self.link.api).__name__}; "
                        "it is on the other path.\n\n")
                )
            return
        for key, title in GROUPS:
            if target.lower() == key:
                self.write(render_group(key, title, self.link.api))
                return
        near = [name for name in sorted(BY_NAME) if target.lower() in name.lower()]
        if near:
            self.write(f"\n  No entry called {target!r}. Close: {', '.join(near[:8])}\n\n")
        else:
            self.write(f"\n  No entry called {target!r}. Type {bold('help')} for the list.\n\n")

    def _state(self) -> None:
        try:
            state = self.link.state()
        except Exception as exc:  # noqa: BLE001
            self.write(f"  {red('no state yet')}: {exc}\n")
            return
        rows = [
            ("battery", f"{state.get('bat')} %"),
            ("height", f"{state.get('h')} cm"),
            ("tof (below)", f"{state.get('tof')} cm"),
            ("barometer", f"{state.get('baro')} cm"),
            ("yaw", f"{state.get('yaw')} deg"),
            ("pitch / roll", f"{state.get('pitch')} / {state.get('roll')} deg"),
            ("speed x/y/z", f"{state.get('vgx')} / {state.get('vgy')} / {state.get('vgz')} cm/s"),
            ("flight time", f"{state.get('time')} s"),
            ("mission pad", state.get("mid")),
        ]
        self.write("\n")
        for label, value in rows:
            self.write(f"  {label:<14} {bold(str(value))}\n")
        self.write(dim("\n  This is what the drone reports. `where` is what is actually true.\n\n"))

    def _where(self) -> None:
        if len(self.link.names) > 1 and self.link.simulator is not None:
            import numpy as np

            self.write("\n")
            for name in self.link.names:
                drone = self.link.simulator.get(name)
                position = drone.state.position
                yaw = float(np.rad2deg(drone.state.yaw))
                self.write(
                    f"  {name:<10} x {position[0]:+.2f}  y {position[1]:+.2f}  "
                    f"z {position[2]:.2f} m   yaw {yaw:+.1f} deg\n"
                )
            self.write(dim("\n  Metres, in the room's frame. No real drone knows this.\n\n"))
            return

        position, yaw = self.link.true_pose()
        if position is None:
            self.write(
                "\n  Ground truth is only available when this console owns the\n"
                "  simulation. Run without --attach to get it.\n\n"
            )
            return
        self.write(
            f"\n  true position   x {position[0]:+.2f} m   y {position[1]:+.2f} m   "
            f"z {position[2]:.2f} m\n"
            f"  true yaw        {yaw:+.1f} deg\n"
        )
        self.write(dim("\n  The real drone cannot tell you this. Score runs with it; never fly on it.\n\n"))

    def _log(self, count: int) -> None:
        history = self.link.history[-count:]
        if not history:
            self.write("\n  Nothing sent yet.\n\n")
            return
        many = len(self.link.names) > 1
        self.write(f"\n  {'sim t':>7}  {'command':<26} {'answer':<20} took\n")
        for entry in history:
            sim_time = "" if entry["sim_time"] is None else f"{entry['sim_time']:.1f}"
            took = f"{entry['seconds']:.2f}s" if entry["seconds"] >= 0.005 else ""
            who = f"{entry['drone']} " if many else ""
            self.write(
                f"  {sim_time:>7}  {who + entry['command']:<26} {entry['response']:<20} {took}\n"
            )
        self.write(
            dim("\n  The same lines go to logs/<run>/commands.jsonl, in the same\n"
                "  format a real flight produces, so the two can be compared.\n\n")
        )

    def _sdk(self) -> None:
        self.write(
            "\n  Raw SDK lines, typed exactly as they go over UDP to port 8889.\n"
            "  Every library call above turns into one of these:\n\n"
        )
        for line in textwrap.wrap(" ".join(sorted(SDK_VERBS)), width=70):
            self.write(f"    {dim(line)}\n")
        self.write(
            "\n  So  " + green("move_forward(50)") + "  and  " + green("forward 50")
            + "  are the same flight.\n\n"
        )

    # -- errors --------------------------------------------------------

    def showtraceback(self) -> None:
        exc_type, exc, tb = sys.exc_info()
        name = exc_type.__name__ if exc_type else ""

        if not self.show_traceback and name in ("TelloSimError", "TelloException"):
            self.write(f"  {red('drone refused it')}: {exc}\n")
            return

        if not self.show_traceback and name == "NameError":
            missing = str(exc).split("'")[1] if "'" in str(exc) else ""
            self.write(f"  {red('NameError')}: {exc}\n")
            if missing in BY_NAME:
                self.write(
                    f"  {dim('That one needs brackets: ')}"
                    f"{green(BY_NAME[missing].call)}\n"
                )
            elif missing in SDK_VERBS:
                self.write(dim("  That is a raw SDK verb; type it with its argument, e.g. ")
                           + green(f"{missing} 50") + "\n")
            else:
                self.write(dim("  Type ") + bold("help") + dim(" for the list of calls.\n"))
            return

        if not self.show_traceback:
            self.write(f"  {red(name)}: {exc}\n")
            self.write(dim("  Type ") + bold("traceback") + dim(" for the full stack.\n"))
            return

        traceback.print_exception(exc_type, exc, tb.tb_next if tb else None, file=sys.stderr)

    def showsyntaxerror(self, filename=None, **kwargs) -> None:
        exc_type, exc, _ = sys.exc_info()
        self.write(f"  {red('SyntaxError')}: {exc}\n")
        self.write(
            dim("  This prompt takes Python, raw SDK lines (")
            + green("forward 50")
            + dim(") and console words (")
            + bold("help")
            + dim(").\n")
        )

    # -- scripts -------------------------------------------------------

    def run_file(self, path: Path) -> None:
        """Run a file of console lines as though they had been typed."""
        if not path.exists():
            self.write(f"  {red('no such file')}: {path}\n")
            return
        self.write(dim(f"\n  running {path}\n\n"))
        for raw in path.read_text().splitlines():
            if raw.strip():
                self.write(f"{bold('tello>')} {raw}\n")
            more = self.push(raw)
            if more:
                continue
        self.push("")

    # -- banner --------------------------------------------------------

    def banner(self, world_name: str = "") -> str:
        api_name = type(self.link.api).__name__
        via = "real djitellopy over UDP" if self.link.kind == "udp" else "in-process, no network"
        lines = [
            "",
            bold("  Tello console") + dim(f"   —  {api_name}, {via}"),
            "",
        ]
        if world_name:
            lines.append(f"  room        {world_name}")
        if self.viewer_url:
            lines.append(f"  3D view     {cyan(self.viewer_url)}")
        lines += [
            "",
            "  Type " + bold("help") + " for every call with its units and limits,",
            "       " + bold("demo") + " for a first flight to copy,",
            "       " + bold("state") + " for telemetry, " + bold("log") + " for what you have sent,",
            "       " + bold("quit") + " to land and stop.",
            "",
            dim("  connect() has already been sent, so the drone is listening."),
            dim("  Start with:  ") + green("takeoff()"),
            "",
        ]
        return "\n".join(lines)


CONSOLE_WORDS = {
    "help", "?", "state", "where", "log", "sdk", "demo", "view", "run",
    "quit", "exit", "bye", "clear", "traceback",
}


def interact(
    link: Link,
    extras: Optional[dict] = None,
    viewer_url: str = "",
    world_name: str = "",
    script: Optional[Path] = None,
    stay: bool = True,
) -> None:
    """Run the prompt until the user leaves."""
    console = TelloConsole(link, extras=extras, viewer_url=viewer_url)
    print(console.banner(world_name))

    if script is not None:
        console.run_file(script)
        if not stay:
            return

    try:
        console.interact(banner="", exitmsg="")
    except SystemExit:
        pass
