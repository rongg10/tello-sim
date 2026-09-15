#!/usr/bin/env python3
"""Start a simulated Tello with a live 3D view in the browser.

    python run_sim3d.py                                 still air
    python run_sim3d.py --wind 1.5 --gusty              a draught, plus gusts
    python run_sim3d.py --scenario scenarios/03_obstacles.yaml
    python run_sim3d.py --drones 3 --scenario scenarios/04_swarm_formation.yaml

A browser window opens showing the room and the drone. Fly it from the buttons
on the page, or with the keyboard (W A S D to move, R and F for height, Q and E
to turn, T to take off, L to land).

The UDP drone is running the whole time, so a tkinter controller can connect
to 127.0.0.1 at the same time and you will see it fly in the 3D view.
Both are ordinary clients; the drone does not care who is talking to it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_sim
from tello_sim import Recorder, SimSpec, Simulator
from tello_sim.render import dashboard
from tello_sim.server import serve
from tello_sim.webviewer import WebViewer


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--drones", type=int, default=1)
    parser.add_argument("--wind", type=float, default=0.0, help="steady wind speed in m/s")
    parser.add_argument("--gusty", action="store_true", help="add turbulence on top")
    parser.add_argument("--room", type=float, default=8.0, help="room size in metres")
    parser.add_argument("--ceiling", type=float, default=2.5)
    parser.add_argument(
        "--scenario", type=Path,
        help="fly in this scenario's room (default: scenarios/00_lab_room.yaml)",
    )
    parser.add_argument(
        "--empty", action="store_true",
        help="a bare room with nothing in it, instead of the default furnished one",
    )
    parser.add_argument("--list", action="store_true", help="list the scenarios and exit")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=8080, help="port for the 3D view")
    parser.add_argument("--udp-port", type=int, default=8889, help="drone command port")
    parser.add_argument("--no-udp", action="store_true", help="3D view only, no network drone")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser")
    parser.add_argument("--no-log", action="store_true")
    args = parser.parse_args()

    if args.list:
        return run_sim.list_worlds()

    world = run_sim.build_world(args)

    recorder = Recorder("live3d", enabled=not args.no_log)
    simulator = Simulator(
        world=world,
        sim_spec=SimSpec(realtime=True, seed=args.seed),
        recorder=None if args.no_log else recorder,
    )

    places = run_sim.starting_points(args, world, args.drones)
    for i, (position, yaw) in enumerate(places):
        simulator.add_drone(
            f"tello-{i + 1}", start_position=position, start_yaw_deg=yaw
        )

    if not args.no_log:
        recorder.write_manifest(simulator.manifest())

    simulator.start()

    viewer = WebViewer(simulator, port=args.port)
    url = viewer.start()

    server = None
    if not args.no_udp:
        server = serve(simulator, command_port=args.udp_port, verbose=True)

    print()
    print(f"  3D view:  {url}")
    if server is not None:
        print(f"  Drone on: 127.0.0.1:{args.udp_port}   (use this IP in any Tello code)")
        print()
        print("  To fly it from a GUI controller at the same time, in another terminal:")
        print("      python run_gui_sim.py --attach")
    print()
    print("  Ctrl-C to stop.")
    print()

    if not args.no_browser:
        viewer.open_browser()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[sim] stopping")
    finally:
        viewer.stop()
        if server is not None:
            server.stop()
        simulator.stop()
        if not args.no_log:
            recorder.close()
            print(f"[sim] log written to {recorder.directory}")
            try:
                print(f"[sim] dashboard: {dashboard(recorder.directory)}")
            except Exception as exc:  # noqa: BLE001
                print(f"[sim] no dashboard ({exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
