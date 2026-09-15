#!/usr/bin/env python3
"""Fly the team's tkinter controller by pressing its own buttons, on a script.

This is the recording harness for the controller demo video.  It launches
`tello_controller.py` -- the real file -- against a simulator that is already
running, then presses the buttons in order with `invoke()`, showing each one
pressed as it goes.  Nothing about the controller is modified: the presses are
the presses a hand would make, so the flight in the 3D view is the flight the
GUI actually produces.

    python scripts/demo_drive_controller.py --captions out/captions.json

It expects a simulator on 127.0.0.1:8889 -- start one with run_sim3d.py.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

CONTROLLER = (
    HERE.parent / "Existing Work From the Group" / "tello_mac_controller" / "tello_controller.py"
)


def load_controller(path: Path):
    spec = importlib.util.spec_from_file_location("tello_controller", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["tello_controller"] = module
    spec.loader.exec_module(module)
    return module


class Director:
    """Walks the app through a flight, one button at a time.

    Every step waits for the controller's own command queue to go quiet before
    the next press, so the pacing is the drone's rather than a fixed timer --
    which is also what stops the queue filling with moves the video never shows
    finishing.
    """

    def __init__(self, app, root, steps, captions_path=None):
        self.app = app
        self.root = root
        self.steps = steps
        self.captions_path = captions_path
        self.captions: list[dict] = []
        self.index = 0
        self.settle_until = 0.0
        self.started = time.time()

    # -- helpers -------------------------------------------------------

    def quiet(self) -> bool:
        return (
            not self.app.command_running
            and not self.app.connecting
            and self.app.command_queue.empty()
            and time.time() >= self.settle_until
        )

    def press(self, button) -> None:
        """Invoke a button and draw it pressed, the way a click looks."""
        try:
            button.state(["pressed"])
            self.root.after(300, lambda: button.state(["!pressed"]))
        except Exception:  # noqa: BLE001 - a plain tk.Button has no .state
            pass
        button.invoke()

    def caption(self, text: str) -> None:
        now = time.time() - self.started
        if self.captions and self.captions[-1]["end"] is None:
            self.captions[-1]["end"] = now
        self.captions.append({"start": now, "end": None, "text": text})

    # -- the loop ------------------------------------------------------

    def tick(self) -> None:
        if self.index >= len(self.steps):
            # The last step's pause is part of the video: hold on it before
            # closing the caption and tearing the window down.
            if time.time() < self.settle_until:
                self.root.after(100, self.tick)
            else:
                self.finish()
            return

        step = self.steps[self.index]
        if not self.quiet() or not step.get("ready", lambda app: True)(self.app):
            self.root.after(100, self.tick)
            return

        self.index += 1
        if step.get("caption"):
            self.caption(step["caption"])
        action = step.get("do")
        if action is not None:
            try:
                action(self)
            except Exception as exc:  # noqa: BLE001 - a demo should not die silently
                print(f"[demo] step {self.index} failed: {exc}", flush=True)
        self.settle_until = time.time() + step.get("pause", 0.8)
        self.root.after(100, self.tick)

    def finish(self) -> None:
        if self.captions and self.captions[-1]["end"] is None:
            self.captions[-1]["end"] = time.time() - self.started
        if self.captions_path is not None:
            self.captions_path.parent.mkdir(parents=True, exist_ok=True)
            self.captions_path.write_text(
                json.dumps({"started": self.started, "captions": self.captions}, indent=2)
            )
        print(f"[demo] flight finished in {time.time() - self.started:.1f}s", flush=True)
        self.root.after(1500, self.root.destroy)


def build_steps(app):
    """The flight: what is said, what is pressed, and how long to hold after."""
    move = app.move_buttons  # Up, Forward, Left, Back, Right, Down, ccw, cw
    connected = lambda a: a.connected  # noqa: E731

    def press(button):
        return lambda d: d.press(button)

    def settings(**values):
        def apply(d):
            for name, value in values.items():
                getattr(d.app, name).set(value)
        return apply

    return [
        {"caption": "The team's own controller — the same file that flies the real Tello",
         "pause": 2.5},
        {"caption": "Only the IP changed: 127.0.0.1 instead of 192.168.10.1",
         "pause": 2.5},
        {"caption": "Connect — the same handshake the real drone answers",
         "do": press(app.connect_button), "pause": 2.0},
        {"caption": "Connected. Battery and height are live telemetry from the simulated drone",
         "ready": connected, "pause": 2.5},
        {"caption": "Take off", "ready": connected,
         "do": press(app.takeoff_button), "pause": 1.5},
        {"caption": "Distance 100 cm, speed 60 cm/s", "ready": connected,
         "do": settings(distance_cm=100, speed_cm_s=60), "pause": 1.2},
        {"caption": "Forward — the drone moves in the 3D view on the left",
         "ready": connected, "do": press(move[1]), "pause": 1.0},
        {"caption": "Rotation 90 degrees", "ready": connected,
         "do": settings(rotation_deg=90), "pause": 1.0},
        {"caption": "Rotate right", "ready": connected,
         "do": press(move[7]), "pause": 1.0},
        {"caption": "Forward again, now along the new heading", "ready": connected,
         "do": press(move[1]), "pause": 1.0},
        {"caption": "Distance 60 cm", "ready": connected,
         "do": settings(distance_cm=60), "pause": 1.0},
        {"caption": "Up — watch the height reading climb", "ready": connected,
         "do": press(move[0]), "pause": 1.0},
        {"caption": "Left", "ready": connected, "do": press(move[2]), "pause": 1.0},
        {"caption": "Rotate left", "ready": connected, "do": press(move[6]), "pause": 1.2},
        {"caption": "Every command is queued, logged and timed — the log matches a real flight's",
         "ready": connected, "pause": 3.0},
        {"caption": "Land — a priority command, so it jumps the queue",
         "ready": connected, "do": press(app.land_button), "pause": 2.0},
        {"caption": "It landed downwind of where it started: SDK moves are relative, "
                    "and nothing corrects for drift",
         "pause": 4.0},
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", type=Path, default=CONTROLLER)
    parser.add_argument("--geometry", default="746x876+762+30")
    parser.add_argument("--captions", type=Path)
    parser.add_argument("--start-delay", type=float, default=1.5)
    args = parser.parse_args()

    import tkinter as tk

    from tello_sim.patch import use_simulator

    use_simulator(verbose=False)
    controller = load_controller(args.controller)

    root = tk.Tk()
    app = controller.TelloControllerApp(root)
    app.drone_ip.set("127.0.0.1")
    root.title("Tello Queue Controller  —  SIMULATOR (127.0.0.1)")
    root.geometry(args.geometry)
    root.update_idletasks()
    root.lift()
    root.attributes("-topmost", True)
    root.after(500, lambda: root.attributes("-topmost", False))

    director = Director(app, root, build_steps(app), args.captions)
    root.after(int(args.start_delay * 1000), director.tick)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
