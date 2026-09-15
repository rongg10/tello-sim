#!/usr/bin/env python3
"""Start a simulated Tello on the network and leave it running.

    python run_sim.py                          still air, one drone
    python run_sim.py --wind 1.5               a 1.5 m/s draught
    python run_sim.py --wind 1.0 --gusty       draught plus turbulence
    python run_sim.py --scenario scenarios/03_obstacles.yaml

Then point any existing Tello code at 127.0.0.1 instead of 192.168.10.1.  Most
tkinter controllers already have an IP field, so nothing needs editing -- or
run `python run_gui_sim.py --controller ...` to launch one pre-pointed here.

Every command the simulated drone receives is printed here and written to
logs/<timestamp>_<name>/commands.jsonl.
"""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tello_sim import ConstantWind, Recorder, Simulator, SimSpec, SumOfWinds, TurbulentWind, World
from tello_sim import world as world_module
from tello_sim.render import dashboard
from tello_sim.scenarios import Scenario
from tello_sim.server import serve


HERE = Path(__file__).resolve().parent
DEFAULT_SCENARIO = HERE / "scenarios" / "00_lab_room.yaml"


def build_world(args) -> World:
    """Work out which room to fly in.

    The default is the furnished lab, not an empty box. An empty box hides
    everything worth seeing: nothing to fly around, nothing under the drone to
    confuse the downward sensor, no reason for the wind to be uneven. Pass
    --empty if a bare room is genuinely what you want.
    """
    import numpy as np

    rng = np.random.default_rng(args.seed)

    scenario_path = getattr(args, "scenario", None)
    if scenario_path is None and not getattr(args, "empty", False):
        if DEFAULT_SCENARIO.exists():
            scenario_path = DEFAULT_SCENARIO

    extra = []
    if args.wind > 0:
        extra.append(ConstantWind([args.wind, 0.0, 0.0]))
    if args.gusty:
        extra.append(TurbulentWind(sigma=0.6, theta=0.8, rng=rng))

    if scenario_path is not None:
        scenario = Scenario.load(scenario_path)
        world = world_module.from_config(scenario.world, rng)
        if extra:
            # Add the command-line weather on top of the scenario's own, rather
            # than silently ignoring it as the first version did.
            world.wind = SumOfWinds([world.wind] + extra)
        return world

    wind = extra[0] if len(extra) == 1 else (SumOfWinds(extra) if extra else None)
    return World(
        bounds_lower=(-args.room / 2, -args.room / 2, 0.0),
        bounds_upper=(args.room / 2, args.room / 2, args.ceiling),
        wind=wind,
        name="empty room",
    )


def starting_points(args, world, count: int) -> list:
    """Where to put the drones, as (position, yaw_deg) pairs.

    A scenario says where its drones start, and those places were chosen to be
    clear of the furniture. Ignoring them and dropping every drone in the middle
    of the room works fine in an empty box and puts the aircraft inside a wall
    the moment the room has any walls in it.
    """
    import numpy as np

    scenario_path = getattr(args, "scenario", None)
    if scenario_path is None and not getattr(args, "empty", False):
        if DEFAULT_SCENARIO.exists():
            scenario_path = DEFAULT_SCENARIO

    places = []
    if scenario_path is not None:
        entries = Scenario.load(scenario_path).drones or []
        places = [
            (list(entry.get("start", [0.0, 0.0, 0.0])), float(entry.get("yaw_deg", 0.0)))
            for entry in entries
        ]

    if not places:
        centre_x = float(world.bounds_lower[0] + world.bounds_upper[0]) / 2.0
        centre_y = float(world.bounds_lower[1] + world.bounds_upper[1]) / 2.0
        places = [([centre_x, centre_y, float(world.bounds_lower[2])], 0.0)]

    # Asking for more drones than the scenario defines: line the extras up
    # alongside the last one, and warn if that lands them somewhere solid.
    spacing = 1.2
    while len(places) < count:
        position, yaw = places[-1]
        places.append(([position[0], position[1] + spacing, position[2]], yaw))

    places = places[:count]

    radius = 0.13
    for index, (position, _) in enumerate(places):
        point = np.asarray(position, dtype=float)
        for obstacle in world.obstacles:
            if obstacle.penetration(point, radius) is not None:
                print(
                    f"[sim] warning: tello-{index + 1} starts inside "
                    f"'{obstacle.name}' at {position}"
                )
    return places


def list_worlds() -> int:
    """Print the available scenarios with a one-line summary of each."""
    from tello_sim import scenarios as scenario_module

    paths = scenario_module.list_scenarios(HERE / "scenarios")
    if not paths:
        print("No scenarios found.")
        return 1

    print(f"\n{len(paths)} scenarios in {HERE / 'scenarios'}:\n")
    for path in paths:
        scenario = scenario_module.Scenario.load(path)
        world = scenario.world or {}
        obstacles = len(world.get("obstacles") or [])
        pads = len(world.get("mission_pads") or [])
        drones = len(scenario.drones or []) or 1
        # First sentence of the description is the summary; trim on a word
        # boundary rather than mid-word.
        summary = " ".join((scenario.description or "").split())
        summary = summary.split(". ")[0].rstrip(".")
        if len(summary) > 76:
            summary = summary[:76].rsplit(" ", 1)[0] + " ..."
        print(f"  {path.name}")
        print(f"      {summary}")
        print(
            f"      {obstacles} obstacle(s), {pads} mission pad(s), {drones} drone(s)\n"
        )
    print("Use one with:  --scenario scenarios/<file>\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--drones", type=int, default=1, help="how many drones to serve")
    parser.add_argument("--wind", type=float, default=0.0, help="steady wind speed in m/s")
    parser.add_argument("--gusty", action="store_true", help="add turbulence on top")
    parser.add_argument("--room", type=float, default=8.0, help="room size in metres")
    parser.add_argument("--ceiling", type=float, default=2.5, help="ceiling height in metres")
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
    parser.add_argument("--port", type=int, default=8889, help="command port")
    parser.add_argument("--no-log", action="store_true", help="do not write a run log")
    parser.add_argument("--quiet", action="store_true", help="do not print each command")
    args = parser.parse_args()

    if args.list:
        return list_worlds()

    world = build_world(args)
    recorder = Recorder("live", enabled=not args.no_log)
    simulator = Simulator(
        world=world,
        sim_spec=SimSpec(realtime=True, seed=args.seed),
        recorder=recorder if not args.no_log else None,
    )

    places = starting_points(args, world, args.drones)
    for i, (position, yaw) in enumerate(places):
        simulator.add_drone(
            f"tello-{i + 1}", start_position=position, start_yaw_deg=yaw
        )

    if not args.no_log:
        recorder.write_manifest(simulator.manifest())

    simulator.start()
    server = serve(simulator, command_port=args.port, verbose=not args.quiet)

    print()
    print("  Simulated Tello is up. In your flight code, use:")
    print(f"      IP   127.0.0.1        (instead of 192.168.10.1)")
    print(f"      port {args.port}")
    print()
    if args.drones > 1:
        print("  Extra drones need loopback aliases before they can be reached:")
        for i in range(1, args.drones):
            print(f"      sudo ifconfig lo0 alias 127.0.0.{i + 1} up")
        print("  For swarm work the in-process path is usually easier: see scripts/demo_swarm.py")
        print()
    print("  Ctrl-C to stop.")
    print()

    def shutdown(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, shutdown)

    try:
        while True:
            signal.pause()
    except (KeyboardInterrupt, AttributeError):
        pass
    finally:
        print("\n[sim] stopping")
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
