#!/usr/bin/env python3
"""Launch the team's existing tkinter controller against the simulator.

    python run_gui_sim.py

This starts a simulated drone, patches djitellopy so it can share this machine
with it, and then runs tello_controller.py exactly as written -- not a copy, not
a modified version, the same file that flies the real drone.  Click Connect and
fly.  The IP is filled in as 127.0.0.1 for you.

If the GUI works here and fails on the real Tello, the difference is the drone
or the network, not the code.  That is the whole point.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DEFAULT_CONTROLLER = (
    HERE.parent / "Existing Work From the Group" / "tello_mac_controller" / "tello_controller.py"
)


def require_tkinter() -> None:
    """The GUI needs a Python built with Tk. Say so clearly if this one is not.

    A pyenv or Homebrew Python is often built without it, and the failure is an
    obscure ImportError about `_tkinter`. The DJI environment does have it, so
    this almost always means the wrong interpreter is active.
    """
    try:
        import tkinter  # noqa: F401
    except ImportError:
        raise SystemExit(
            "This Python has no tkinter, so the GUI cannot start.\n"
            f"  interpreter: {sys.executable}\n\n"
            "The DJI environment has tkinter, so this is probably the wrong\n"
            "interpreter. Activate it and try again:\n\n"
            "    source ~/.pyenv/versions/DJI/bin/activate\n\n"
            "Everything else (run_sim3d.py, run_sim.py, run_scenario.py, the demo\n"
            "scripts) works on any Python 3.9 or newer, tkinter or not."
        )


def load_controller(path: Path):
    if not path.exists():
        raise SystemExit(
            f"Could not find the controller at:\n  {path}\n"
            "Pass its location with --controller /path/to/tello_controller.py"
        )
    spec = importlib.util.spec_from_file_location("tello_controller", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["tello_controller"] = module
    spec.loader.exec_module(module)
    return module


def run_attached(args) -> int:
    """Point the controller at a simulator someone else is already running.

    Use this with run_sim3d.py: that process owns the drone and the 3D view,
    this one is just another client pressing buttons at it. Commands sent from
    here show up in the 3D view immediately, because both are talking to the
    same drone.
    """
    import tkinter as tk

    from tello_sim.patch import use_simulator

    use_simulator()
    controller = load_controller(args.controller)

    root = tk.Tk()
    app = controller.TelloControllerApp(root)
    app.drone_ip.set("127.0.0.1")
    root.title("Tello Queue Controller  —  attached to a running simulator")
    print("\n[gui] Attached. Press Connect; the IP is already set to 127.0.0.1.")
    print("[gui] Nothing is logged here: the simulator process owns the run log.\n")

    root.mainloop()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--controller", type=Path, default=DEFAULT_CONTROLLER)
    parser.add_argument("--wind", type=float, default=0.0, help="steady wind speed in m/s")
    parser.add_argument("--gusty", action="store_true")
    parser.add_argument("--scenario", type=Path, help="take the world from a scenario file")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--attach",
        action="store_true",
        help="connect to a simulator that is already running (e.g. run_sim3d.py) "
             "instead of starting one here",
    )
    args = parser.parse_args()

    require_tkinter()

    if args.attach:
        return run_attached(args)

    # 1. Start the simulated drone on the network.
    import run_sim
    from tello_sim import Recorder, SimSpec, Simulator
    from tello_sim.render import dashboard
    from tello_sim.server import serve

    world = run_sim.build_world(
        argparse.Namespace(
            scenario=args.scenario, wind=args.wind, gusty=args.gusty,
            room=8.0, ceiling=2.5, seed=args.seed,
        )
    )
    recorder = Recorder("gui")
    simulator = Simulator(world=world, sim_spec=SimSpec(realtime=True, seed=args.seed), recorder=recorder)
    simulator.add_drone("tello-1")
    recorder.write_manifest(simulator.manifest())
    simulator.start()
    server = serve(simulator, verbose=True)

    # 2. Let djitellopy share this machine with the simulator.
    from tello_sim.patch import use_simulator

    use_simulator()

    # 3. Run the real controller, unmodified.
    import tkinter as tk

    controller = load_controller(args.controller)

    root = tk.Tk()
    app = controller.TelloControllerApp(root)
    app.drone_ip.set("127.0.0.1")
    root.title("Tello Queue Controller  —  SIMULATOR (127.0.0.1)")
    print("\n[gui] Controller is up. Press Connect; the IP is already set to 127.0.0.1.\n")

    try:
        root.mainloop()
    finally:
        server.stop()
        simulator.stop()
        recorder.close()
        print(f"\n[gui] log written to {recorder.directory}")
        try:
            print(f"[gui] dashboard: {dashboard(recorder.directory)}")
        except Exception as exc:  # noqa: BLE001
            print(f"[gui] no dashboard ({exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
