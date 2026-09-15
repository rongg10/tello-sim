#!/usr/bin/env python3
"""Type a flight into run_cli.py, at human speed, for the CLI demo video.

The console is started on a pseudo-terminal and the characters are written to
it one at a time, so the terminal echoes them exactly as it would echo a person
typing.  Nothing is faked: the prompt, the completions, the timings and the
refusals in the recording are the console's own.

    python scripts/demo_drive_cli.py --captions out/captions.json

It expects a simulator on 127.0.0.1:8889 -- start one with run_sim3d.py.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import pty
import struct
import subprocess
import sys
import termios
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent

# (what to type, what to say about it, how long to hold afterwards)
SCRIPT: list[tuple[str, str, float]] = [
    ("", "A prompt instead of buttons \u2014 and what you type at it is the djitellopy API", 3.5),
    ("get_battery()", "The object at the prompt is a real djitellopy.Tello sending real UDP", 2.0),
    ("takeoff()", "Take off", 1.5),
    ("move_forward(100)", "Every call prints the SDK line it sent, the answer, and how long it took", 1.5),
    ("rotate_clockwise(90)", "Rotate", 1.2),
    ("forward 60", "The raw SDK line works too \u2014 the same flight, and what the library actually sends", 2.0),
    ("move_up(50)", "Climb", 1.2),
    ("move_forward(5)", "Ask for something the firmware refuses, and it is refused here as well", 3.5),
    ("state", "state \u2014 the telemetry the drone broadcasts, labelled and in units", 3.5),
    ("where", "where \u2014 where the drone actually is. Ground truth, which no real Tello can give you", 4.5),
    ("go_xyz_speed(120, 60, 0, 50)", "One diagonal move instead of three separate legs", 1.5),
    ("log", "log \u2014 every command, its answer and its duration, in the format a real flight produces", 4.5),
    ("land()", "Land", 2.0),
    ("", "Paste those lines into a .py file, change the IP back to 192.168.10.1, and it flies the drone", 4.5),
]


class Session:
    """A console on a pty, plus a typist."""

    def __init__(self, command: list[str], cwd: Path, rows: int, cols: int):
        self.master, slave = pty.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
        env = dict(os.environ, TERM="xterm-256color", PYTHONUNBUFFERED="1", COLUMNS=str(cols), LINES=str(rows))
        self.proc = subprocess.Popen(
            command, stdin=slave, stdout=slave, stderr=slave,
            cwd=str(cwd), env=env, close_fds=True, start_new_session=True,
        )
        os.close(slave)
        self.buffer = bytearray()
        self.last_output = time.time()
        self.alive = True
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        while self.alive:
            try:
                chunk = os.read(self.master, 8192)
            except OSError:
                break
            if not chunk:
                break
            self.buffer.extend(chunk)
            self.last_output = time.time()
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()

    # -- waiting -------------------------------------------------------

    def prompts(self) -> int:
        return self.buffer.count(b">>>")

    def wait_for_prompt(self, count: int, timeout: float = 60.0, idle: float = 0.45) -> None:
        """Wait until the console has printed its prompt again and gone quiet.

        Two conditions rather than one: the prompt alone can arrive while output
        is still streaming, and quiet alone happens while a long move is flying.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.prompts() >= count and time.time() - self.last_output > idle:
                return
            if self.proc.poll() is not None:
                return
            time.sleep(0.05)

    # -- typing --------------------------------------------------------

    def type(self, text: str, delay: float = 0.055) -> None:
        for character in text:
            os.write(self.master, character.encode())
            time.sleep(delay)
        time.sleep(0.25)
        os.write(self.master, b"\r")

    def close(self) -> None:
        self.alive = False
        try:
            self.proc.wait(timeout=8)
        except Exception:  # noqa: BLE001
            self.proc.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captions", type=Path)
    parser.add_argument("--rows", type=int, default=44)
    parser.add_argument("--cols", type=int, default=96)
    parser.add_argument("--start-delay", type=float, default=1.5)
    parser.add_argument(
        "--console-args", default="--no-browser --wind 1.0",
        help="arguments passed through to run_cli.py",
    )
    args = parser.parse_args()

    started = time.time()
    captions: list[dict] = []

    def say(text: str) -> None:
        now = time.time() - started
        if captions and captions[-1]["end"] is None:
            captions[-1]["end"] = now
        captions.append({"start": now, "end": None, "text": text})

    session = Session(
        [sys.executable, "run_cli.py", *args.console_args.split()],
        cwd=HERE, rows=args.rows, cols=args.cols,
    )

    expected = 1
    session.wait_for_prompt(expected, timeout=40.0)
    time.sleep(args.start_delay)

    for line, caption, hold in SCRIPT:
        if caption:
            say(caption)
        if line:
            session.type(line)
            expected += 1
            session.wait_for_prompt(expected, timeout=90.0)
        time.sleep(hold)

    session.type("exit()")
    time.sleep(3.0)
    session.close()

    if captions and captions[-1]["end"] is None:
        captions[-1]["end"] = time.time() - started
    if args.captions:
        args.captions.parent.mkdir(parents=True, exist_ok=True)
        args.captions.write_text(json.dumps({"started": started, "captions": captions}, indent=2))
    print(f"\n[demo] console session finished in {time.time() - started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
