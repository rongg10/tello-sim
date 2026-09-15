#!/usr/bin/env python3
"""Fly the simulated Tello by typing library calls, with the 3D view watching.

    python run_cli.py                                   lab room, still air
    python run_cli.py --wind 1.5 --gusty                a draught, plus gusts
    python run_cli.py --scenario scenarios/07_apartment.yaml
    python run_cli.py --attach                          join a running run_sim3d.py

A browser window opens with the room, and a prompt appears in this terminal:

    tello> takeoff()
    tello> move_forward(100)
    tello> rotate_clockwise(90)
    tello> get_battery()

What you type is the real djitellopy API -- the same calls, the same argument
ranges, the same refusals -- so the flight you build here is a flight script,
not a simulator dialect.  `help` prints every call with its units and limits,
`help move_forward` explains one in detail, and raw SDK lines (`forward 100`)
work too, which is worth doing once to see what the library actually sends.

The buttons in the 3D view still work, and the team's tkinter controller can
attach at the same time.  The drone does not care who is talking to it.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_sim
from tello_sim import Recorder, SimSpec, Simulator
from tello_sim.cli import Link, interact
from tello_sim.render import dashboard


def build_arguments() -> argparse.Namespace:
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
    parser.add_argument("--host", default="127.0.0.1", help="drone address (real Tello: 192.168.10.1)")
    parser.add_argument(
        "--attach", action="store_true",
        help="talk to a simulator that is already running (run_sim3d.py or run_sim.py) "
             "instead of starting one here",
    )
    parser.add_argument(
        "--in-process", action="store_true",
        help="use SimTello instead of djitellopy: no sockets, and the only way "
             "to fly more than one drone from one console",
    )
    parser.add_argument("--no-view", action="store_true", help="no 3D view")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--no-connect", action="store_true", help="leave connect() for you to type")
    parser.add_argument("--script", type=Path, help="run a file of console lines first")
    parser.add_argument("--exit", action="store_true", help="with --script, do not stay at the prompt")
    return parser.parse_args()


def real_tello(host: str, port: int):
    """A genuine djitellopy.Tello, pointed at the simulator on this machine."""
    from tello_sim.patch import use_simulator

    use_simulator(verbose=False)

    import logging

    from djitellopy import Tello

    # The library narrates every command at INFO. The console prints the same
    # thing more compactly, so keep only the warnings.
    Tello.LOGGER.setLevel(logging.WARNING)

    drone = Tello(host=host)
    if port != Tello.CONTROL_UDP_PORT:
        # The library fixes the command port at class level. Retarget this one
        # instance rather than the class, so a second Tello in the same process
        # is unaffected.
        drone.address = (host, port)
    return drone


def attach(args) -> int:
    """Join a simulator someone else is running, as one more ordinary client."""
    drone = real_tello(args.host, args.udp_port)
    link = Link(drone, kind="udp")

    print(f"\n  Attaching to a drone at {args.host}:{args.udp_port} ...")
    if not args.no_connect:
        try:
            drone.connect()
        except Exception as exc:  # noqa: BLE001
            print(f"\n  Could not reach it: {exc}")
            print("  Is a simulator running? Start one with:  python run_sim3d.py\n")
            return 1

    try:
        interact(
            link,
            viewer_url=f"http://127.0.0.1:{args.port}/",
            world_name="(owned by the other process)",
            script=args.script,
            stay=not args.exit,
        )
    except KeyboardInterrupt:
        pass
    finally:
        # Land here rather than leaving it to djitellopy's __del__, which runs
        # at interpreter shutdown when the response thread may already be gone
        # -- and then the land times out with the drone still in the air.
        try:
            if drone.is_flying:
                drone.land()
        except Exception:  # noqa: BLE001 - shutting down
            pass
        drone.is_flying = False
    return 0


def own_simulation(args) -> int:
    """Start the world, the drone, the 3D view, and the prompt, all here."""
    world = run_sim.build_world(args)

    recorder = Recorder("console", enabled=not args.no_log)
    simulator = Simulator(
        world=world,
        sim_spec=SimSpec(realtime=True, seed=args.seed),
        recorder=None if args.no_log else recorder,
    )

    places = run_sim.starting_points(args, world, args.drones)
    for index, (position, yaw) in enumerate(places):
        simulator.add_drone(f"tello-{index + 1}", start_position=position, start_yaw_deg=yaw)

    if not args.no_log:
        recorder.write_manifest(simulator.manifest())

    simulator.start()

    viewer = None
    viewer_url = ""
    if not args.no_view:
        from tello_sim.webviewer import WebViewer

        viewer = WebViewer(simulator, port=args.port)
        viewer_url = viewer.start()

    # Two ways to reach the drone, and the choice matters.
    #
    # By default the console holds a real djitellopy.Tello and every call goes
    # out over UDP, which is the honest thing to practise against: the same
    # library, the same wire, the same timeouts.  It handles one drone, because
    # djitellopy tells drones apart by source IP.
    #
    # --in-process (and any swarm) uses SimTello instead: identical methods, no
    # sockets, and as many drones as you like.
    server = None
    extras: dict = {"sim": simulator, "world": world}

    if args.in_process or args.drones > 1:
        from tello_sim.client import SimSwarm, SimTello

        tellos = [
            SimTello(simulator.get(f"tello-{i + 1}"), simulator) for i in range(args.drones)
        ]
        api = tellos[0]
        extras["drones"] = tellos
        if len(tellos) > 1:
            extras["swarm"] = SimSwarm(tellos)
        link = Link(api, kind="inproc", simulator=simulator)
        for index, tello in enumerate(tellos[1:], start=2):
            link.also(tello, f"tello-{index}")
        if not args.no_connect:
            for tello in tellos:
                tello.connect()
    else:
        from tello_sim.server import serve

        server = serve(simulator, command_port=args.udp_port, verbose=False)
        api = real_tello(args.host, args.udp_port)
        link = Link(api, kind="udp", simulator=simulator)
        if not args.no_connect:
            try:
                api.connect()
            except Exception as exc:  # noqa: BLE001
                print(f"\n  The drone did not answer connect(): {exc}")
                print("  Type connect() at the prompt to try again.\n")

    if server is not None:
        print(f"\n  Drone on 127.0.0.1:{args.udp_port} — the tkinter GUI can attach too:")
        print("      python run_gui_sim.py --attach")
    if len(extras.get("drones", [])) > 1:
        print(f"\n  {args.drones} drones: `drones[0]` .. `drones[{args.drones - 1}]`, "
              "and `swarm` for all of them at once.")
        print("  `tello` is drones[0].")

    if viewer is not None and not args.no_browser:
        viewer.open_browser()

    try:
        interact(
            link,
            extras=extras,
            viewer_url=viewer_url,
            world_name=getattr(world, "name", "") or "",
            script=args.script,
            stay=not args.exit,
        )
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[cli] stopping")
        try:
            if link.api.is_flying:
                link.api.land()
        except Exception:  # noqa: BLE001 - shutting down; a failed land is not news
            pass
        time.sleep(0.2)
        if viewer is not None:
            viewer.stop()
        if server is not None:
            server.stop()
        simulator.stop()
        if not args.no_log:
            recorder.close()
            print(f"[cli] log written to {recorder.directory}")
            try:
                print(f"[cli] dashboard: {dashboard(recorder.directory)}")
            except Exception as exc:  # noqa: BLE001
                print(f"[cli] no dashboard ({exc})")
    return 0


def main() -> int:
    args = build_arguments()
    if args.list:
        return run_sim.list_worlds()
    if args.attach:
        return attach(args)
    return own_simulation(args)


if __name__ == "__main__":
    raise SystemExit(main())
