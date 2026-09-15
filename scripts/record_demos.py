#!/usr/bin/env python3
"""Record the two demo videos: the GUI controller, and the console.

    python scripts/record_demos.py            both
    python scripts/record_demos.py cli        just the console one
    python scripts/record_demos.py controller just the GUI one

Nothing is staged.  Each video is a screen recording of the real programs --
a tkinter controller, or `run_cli.py` -- flying a real simulated drone, with
the 3D view open beside them.  The controller half needs a controller of your
own; see scripts/demo_drive_controller.py.  The button presses and the typing
come from the driver scripts next to this one, so the flight is repeatable and
the pacing is the drone's.

Captions are burned in afterwards from the timeline the driver wrote while it
ran, so they line up with what is happening rather than being guessed at.

Needs Screen Recording permission for whatever is running this.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "demos"
SCRATCH = OUT / ".raw"

FFMPEG = ""
FONT_REGULAR = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

PROFILE = "TelloDemoTmp"          # a throwaway Terminal profile, deleted after
CAPTURE_LATENCY = 0.7             # screencapture takes about this long to start


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------

def osa(script: str, timeout: float = 20.0) -> str:
    """Run one AppleScript and give back its result, or raise with the error."""
    wrapped = f"with timeout of {int(timeout)} seconds\n{script}\nend timeout"
    result = subprocess.run(
        ["osascript", "-"], input=wrapped, capture_output=True, text=True, timeout=timeout + 10
    )
    if result.returncode != 0:
        raise RuntimeError(f"AppleScript failed: {result.stderr.strip()}\n{script}")
    return result.stdout.strip()


def ffmpeg_path() -> str:
    global FFMPEG
    if not FFMPEG:
        import imageio_ffmpeg

        FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    return FFMPEG


def screen_size() -> tuple[int, int]:
    bounds = osa('tell application "Finder" to get bounds of window of desktop')
    _, _, width, height = [int(part) for part in bounds.split(", ")]
    return width, height


def wait_http(url: str, timeout: float = 30.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.5):
                return True
        except Exception:  # noqa: BLE001 - not up yet
            time.sleep(0.3)
    return False


def wait_for(path: Path, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists():
            return True
        time.sleep(0.4)
    return False


# ----------------------------------------------------------------------
# The pieces on screen
# ----------------------------------------------------------------------

class Chrome:
    """The 3D view, in a plain Chrome window with no tabs or address bar."""

    def __init__(self, url: str, box: tuple[int, int, int, int], profile_dir: Path):
        x, y, width, height = box
        self.process = subprocess.Popen(
            [
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                f"--user-data-dir={profile_dir}",
                f"--app={url}",
                f"--window-position={x},{y}",
                f"--window-size={width},{height}",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-session-crashed-bubble",
                "--disable-infobars",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def stop(self) -> None:
        self.process.terminate()
        try:
            self.process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            self.process.kill()


def make_profile(font_size: int, title: str) -> None:
    """A throwaway Terminal profile, so the user's own settings are untouched."""
    existing = osa('tell application "Terminal" to get name of every settings set')
    if PROFILE not in existing:
        osa(f'tell application "Terminal" to make new settings set '
            f'with properties {{name:"{PROFILE}"}}')
    osa(f'''tell application "Terminal"
  set s to settings set "{PROFILE}"
  set font name of s to "Menlo"
  set font size of s to {font_size}
  set background color of s to {{3500, 4600, 5600}}
  set normal text color of s to {{57000, 60000, 62000}}
  set bold text color of s to {{65535, 65535, 65535}}
  set cursor color of s to {{26000, 62000, 40000}}
  set title displays custom title of s to true
  set custom title of s to "{title}"
  set title displays device name of s to false
  set title displays shell path of s to false
  set title displays window size of s to false
end tell''')


def drop_profile() -> None:
    try:
        osa(f'tell application "Terminal" to delete settings set "{PROFILE}"')
    except Exception as exc:  # noqa: BLE001 - tidying up is not worth failing over
        print(f"[record] could not remove the temporary Terminal profile: {exc}")


def open_terminal(command: str, cols: int, rows: int) -> None:
    osa(f'tell application "Terminal" to do script {json.dumps(command)}')
    time.sleep(1.2)
    osa(f'''tell application "Terminal"
  set current settings of front window to settings set "{PROFILE}"
  set number of columns of front window to {cols}
  set number of rows of front window to {rows}
end tell''')


def terminal_bounds() -> tuple[int, int, int, int]:
    raw = osa('tell application "Terminal" to get bounds of front window')
    return tuple(int(part) for part in raw.split(", "))  # type: ignore[return-value]


def place_terminal(x: int, y: int) -> tuple[int, int, int, int]:
    left, top, right, bottom = terminal_bounds()
    width, height = right - left, bottom - top
    osa(f'tell application "Terminal" to set bounds of front window '
        f'to {{{x}, {y}, {x + width}, {y + height}}}')
    return x, y, width, height


def close_terminal() -> None:
    try:
        osa('tell application "Terminal" to close window 1 saving no', timeout=10)
    except Exception:  # noqa: BLE001
        pass


# ----------------------------------------------------------------------
# Recording
# ----------------------------------------------------------------------

def start_recording(rect: tuple[int, int, int, int], seconds: int, path: Path) -> tuple[subprocess.Popen, float]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    x, y, width, height = rect
    process = subprocess.Popen(
        ["/usr/sbin/screencapture", "-v", "-V", str(seconds), "-x",
         f"-R{x},{y},{width},{height}", str(path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return process, time.time()


# ----------------------------------------------------------------------
# Turning the raw capture into something to show people
# ----------------------------------------------------------------------

def title_card(path: Path, size: tuple[int, int], title: str, subtitle: str, footer: str) -> None:
    from PIL import Image, ImageDraw, ImageFont

    width, height = size
    image = Image.new("RGB", size, (14, 18, 24))
    draw = ImageDraw.Draw(image)
    big = ImageFont.truetype(FONT_BOLD, int(height * 0.075))
    medium = ImageFont.truetype(FONT_REGULAR, int(height * 0.036))
    small = ImageFont.truetype(FONT_REGULAR, int(height * 0.026))

    def centred(text, font, y, fill):
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        draw.text(((width - (right - left)) / 2 - left, y), text, font=font, fill=fill)
        return bottom - top

    y = height * 0.34
    y += centred(title, big, y, (233, 239, 245)) + height * 0.055
    y += centred(subtitle, medium, y, (128, 200, 160)) + height * 0.05
    centred(footer, small, y, (120, 132, 145))
    draw.line([(width * 0.42, height * 0.30), (width * 0.58, height * 0.30)], fill=(60, 120, 90), width=4)
    image.save(path)


def caption_filters(captions: list[dict], offset: float, width: int, video_height: int,
                    bar: int, scratch: Path) -> str:
    """One drawtext per caption, each switched on for the span it belongs to."""
    parts = []
    for index, entry in enumerate(captions):
        text = entry["text"].strip()
        if not text:
            continue
        start = max(0.0, entry["start"] + offset)
        end = max(start + 0.5, (entry["end"] or entry["start"] + 3.0) + offset)
        text_file = scratch / f"caption_{index:03d}.txt"
        text_file.write_text(text)
        size = 34 if len(text) < 78 else 29
        parts.append(
            f"drawtext=textfile={_escape(str(text_file))}:fontfile={_escape(FONT_REGULAR)}"
            f":fontcolor=0xE9EFF5:fontsize={size}:x=(w-text_w)/2"
            f":y={video_height + (bar - size) // 2 - 4}"
            f":enable='between(t\\,{start:.2f}\\,{end:.2f})'"
        )
    return ",".join(parts)


def _escape(path: str) -> str:
    return path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def build_video(raw: Path, captions_file: Path, record_started: float, out: Path,
                title: str, subtitle: str, footer: str, target_width: int = 1920) -> Path:
    payload = json.loads(captions_file.read_text())
    offset = payload["started"] - (record_started + CAPTURE_LATENCY)
    captions = payload["captions"]

    probe = subprocess.run(
        [ffmpeg_path(), "-hide_banner", "-i", str(raw)],
        capture_output=True, text=True,
    ).stderr
    source_width, source_height = 3024, 1768
    for line in probe.splitlines():
        if "Video:" in line:
            for token in line.split(","):
                token = token.strip().split(" ")[0]
                if "x" in token and token.replace("x", "").isdigit():
                    source_width, source_height = (int(v) for v in token.split("x"))
                    break
            break

    video_width = target_width
    video_height = int(round(source_height * target_width / source_width / 2)) * 2
    bar = 96
    total_height = video_height + bar

    SCRATCH.mkdir(parents=True, exist_ok=True)
    card = SCRATCH / f"{out.stem}_title.png"
    title_card(card, (video_width, total_height), title, subtitle, footer)

    drawtexts = caption_filters(captions, offset, video_width, video_height, bar, SCRATCH)
    tail = (payload["captions"][-1]["end"] or 0) + offset + 2.5

    chain = (
        f"[1:v]fps=30,scale={video_width}:{video_height},"
        f"pad={video_width}:{total_height}:0:0:color=0x0E1218"
    )
    if drawtexts:
        chain += "," + drawtexts
    chain += f",trim=start=0:end={tail:.2f},setpts=PTS-STARTPTS,setsar=1[main];"
    chain += f"[0:v]fps=30,setsar=1[card];[card][main]concat=n=2:v=1:a=0[out]"

    out.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "error",
        "-loop", "1", "-t", "3", "-i", str(card),
        "-i", str(raw),
        "-filter_complex", chain,
        "-map", "[out]",
        "-c:v", "libx264", "-preset", "slow", "-crf", "21",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(out),
    ]
    subprocess.run(command, check=True)
    return out


# ----------------------------------------------------------------------
# The two demos
# ----------------------------------------------------------------------

def check_display() -> None:
    """Fail early and clearly if there is no lit screen to record.

    With the lid shut and no external display, every capture comes back a
    uniform black rectangle -- and screencapture reports success, so the first
    sign of trouble would otherwise be a 90-second video of nothing.
    """
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        shot = Path(directory) / "probe.png"
        subprocess.run(
            ["/usr/sbin/screencapture", "-x", "-t", "png", "-R", "0,0,400,300", str(shot)],
            capture_output=True,
        )
        if not shot.exists():
            raise SystemExit(
                "The screen could not be captured at all. Grant Screen Recording "
                "permission to whatever is running this, and try again."
            )
        from PIL import Image

        if len(Image.open(shot).convert("L").getextrema()) and Image.open(shot).convert("L").getextrema() == (0, 0):
            raise SystemExit(
                "The screen is dark, so there is nothing to record.\n"
                "The lid is probably closed -- open it (or plug in a display), "
                "leave it awake, and run this again."
            )


def free_ports(ports: tuple[int, ...] = (8080, 8889)) -> None:
    """Refuse to start on top of a simulator someone left running."""
    import socket

    for port in ports:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM if port == 8080 else socket.SOCK_DGRAM)
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            raise SystemExit(
                f"port {port} is already in use -- something (run_sim3d.py? run_cli.py?) "
                "is still running. Stop it and try again."
            )
        finally:
            probe.close()


def record_cli(args) -> Path:
    """run_cli.py in a terminal, with the 3D view it is driving beside it."""
    check_display()
    free_ports()
    screen_w, _ = screen_size()
    top, bottom = args.top, args.bottom
    height = bottom - top

    captions = SCRATCH / "cli_captions.json"
    captions.unlink(missing_ok=True)
    raw = SCRATCH / "cli_raw.mov"

    make_profile(args.font_size, "tello console  —  python run_cli.py")
    command = (
        f"cd {shlex.quote(str(HERE))} && "
        f"source {shlex.quote(args.venv)}/bin/activate && "
        f"clear && python scripts/demo_drive_cli.py --captions {shlex.quote(str(captions))} "
        f"--rows {args.rows} --cols {args.cols}"
    )
    open_terminal(command, args.cols, args.rows)
    left, _, right, _ = terminal_bounds()
    term_width = right - left
    term_x = screen_w - term_width
    place_terminal(term_x, top)

    if not wait_http("http://127.0.0.1:8080/", timeout=45):
        raise SystemExit("the 3D view never came up on port 8080")

    chrome = Chrome("http://127.0.0.1:8080/", (0, top, max(360, term_x + 4), height), SCRATCH / "chrome")
    time.sleep(5)
    osa('tell application "Terminal" to activate')
    time.sleep(1.0)

    process, started = start_recording((0, top - 4, screen_w, height + 8), args.cli_seconds, raw)
    try:
        if not wait_for(captions, timeout=args.cli_seconds):
            print("[record] the driver did not finish in time; using what was captured")
        time.sleep(2.0)
    finally:
        process.wait()
        chrome.stop()
        close_terminal()
        drop_profile()

    return build_video(
        raw, captions, started, OUT / "tello_cli_demo.mp4",
        "Flying the Tello from the console",
        "python run_cli.py  —  the djitellopy API, typed",
        "Tello simulator  ·  real UDP on 127.0.0.1  ·  furnished lab, 1.0 m/s wind",
        target_width=args.width,
    )


def record_controller(args) -> Path:
    """A tkinter controller, pressing its own buttons, in the 3D view."""
    check_display()
    free_ports()
    screen_w, _ = screen_size()
    top, bottom = args.top, args.bottom
    height = bottom - top

    captions = SCRATCH / "controller_captions.json"
    captions.unlink(missing_ok=True)
    raw = SCRATCH / "controller_raw.mov"
    SCRATCH.mkdir(parents=True, exist_ok=True)

    simulator = subprocess.Popen(
        [args.python, "run_sim3d.py", "--no-browser", "--wind", "1.0", "--seed", "3"],
        cwd=str(HERE), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        if not wait_http("http://127.0.0.1:8080/", timeout=45):
            raise SystemExit("the 3D view never came up on port 8080")

        gui_width = 780
        gui_x = screen_w - gui_width
        chrome = Chrome("http://127.0.0.1:8080/", (0, top, max(360, gui_x + 4), height),
                        SCRATCH / "chrome")
        time.sleep(5)

        driver = subprocess.Popen(
            [args.python, "scripts/demo_drive_controller.py",
             "--captions", str(captions),
             "--geometry", f"{gui_width}x{height}+{gui_x}+{top}"],
            cwd=str(HERE), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(3.0)

        process, started = start_recording((0, top - 4, screen_w, height + 8), args.controller_seconds, raw)
        try:
            driver.wait(timeout=args.controller_seconds)
        except subprocess.TimeoutExpired:
            driver.kill()
        process.wait()
        chrome.stop()
    finally:
        simulator.send_signal(signal.SIGINT)
        try:
            simulator.wait(timeout=15)
        except subprocess.TimeoutExpired:
            simulator.kill()

    return build_video(
        raw, captions, started, OUT / "tello_controller_demo.mp4",
        "Flying the Tello from the GUI controller",
        "A GUI controller  —  unmodified, pointed at the simulator",
        "Tello simulator  ·  real UDP on 127.0.0.1  ·  furnished lab, 1.0 m/s wind",
        target_width=args.width,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("which", nargs="?", default="both",
                        choices=["both", "cli", "controller"])
    parser.add_argument("--venv", default=os.environ.get("VIRTUAL_ENV", str(Path(__file__).resolve().parent.parent / ".venv")))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--bottom", type=int, default=906)
    parser.add_argument("--cols", type=int, default=84)
    parser.add_argument("--rows", type=int, default=46)
    parser.add_argument("--font-size", type=int, default=14)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--controller-seconds", type=int, default=85,
                        help="how long to record the GUI demo for")
    parser.add_argument("--cli-seconds", type=int, default=125,
                        help="how long to record the console demo for")
    args = parser.parse_args()

    SCRATCH.mkdir(parents=True, exist_ok=True)
    awake = subprocess.Popen(["caffeinate", "-d"])  # no display sleep mid-take
    made = []
    if args.which in ("controller", "both"):
        made.append(record_controller(args))
    if args.which in ("cli", "both"):
        made.append(record_cli(args))

    awake.terminate()
    print()
    for path in made:
        size = path.stat().st_size / 1e6
        print(f"  {path}   {size:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
