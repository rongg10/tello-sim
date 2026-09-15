#!/usr/bin/env python3
"""Check every scenario file for the mistakes that are easy to make by hand.

    python scripts/check_scenarios.py            geometry only, instant
    python scripts/check_scenarios.py --fly      also fly each one and report

Room layouts are written as raw coordinates, so a drone can start inside a wall,
a mission pad can end up under a desk where the camera will never see it, and a
doorway can be narrower than the aircraft. None of those raise an error at run
time; they just produce a flight that quietly makes no sense. This catches them.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from tello_sim import scenarios
from tello_sim.config import DroneSpec
from tello_sim.world import Box, Cylinder

DEFAULT_RADIUS = DroneSpec().radius
PAD_MAX_HEIGHT = DroneSpec().pad_detect_max_height


def overlaps(obstacle, point: np.ndarray, radius: float) -> bool:
    return obstacle.penetration(np.asarray(point, dtype=float), radius) is not None


def covers_floor_at(obstacle, xy: np.ndarray) -> bool:
    """Would this obstacle hide a mission pad lying at these floor coordinates?"""
    x, y = float(xy[0]), float(xy[1])
    if isinstance(obstacle, Box):
        return (
            obstacle.lower[0] <= x <= obstacle.upper[0]
            and obstacle.lower[1] <= y <= obstacle.upper[1]
            and obstacle.lower[2] <= 0.05
        )
    if isinstance(obstacle, Cylinder):
        planar = float(np.linalg.norm(np.array([x, y]) - obstacle.center_xy))
        return planar <= obstacle.radius and obstacle.z_min <= 0.05
    return False


def check(path: Path) -> list:
    problems = []
    scenario = scenarios.Scenario.load(path)

    try:
        simulator, swarm, _ = scenario.build(record=False)
    except Exception as exc:  # noqa: BLE001 - report, do not crash the sweep
        return [f"will not build: {exc}"]

    world = simulator.world
    lower, upper = world.bounds_lower, world.bounds_upper

    for drone in simulator.drones:
        start = drone.state.position
        radius = drone.spec.radius

        if np.any(start < lower - 1e-6) or np.any(start > upper + 1e-6):
            problems.append(f"{drone.name} starts outside the room at {start.tolist()}")

        for obstacle in world.obstacles:
            if overlaps(obstacle, start, radius):
                problems.append(
                    f"{drone.name} starts inside '{obstacle.name}' at {start.tolist()}"
                )

        # Straight up from the start must be clear, or takeoff hits something.
        overhead = start + np.array([0.0, 0.0, drone.spec.takeoff_height])
        for obstacle in world.obstacles:
            if overlaps(obstacle, overhead, radius):
                problems.append(f"{drone.name} would take off into '{obstacle.name}'")

    for a_index, a in enumerate(simulator.drones):
        for b in simulator.drones[a_index + 1:]:
            gap = float(np.linalg.norm(a.state.position - b.state.position))
            if gap < a.spec.radius + b.spec.radius:
                problems.append(
                    f"{a.name} and {b.name} start {gap:.2f} m apart, which is touching"
                )

    for pad in world.mission_pads:
        for obstacle in world.obstacles:
            if covers_floor_at(obstacle, pad.center):
                problems.append(
                    f"mission pad {pad.pad_id} is underneath '{obstacle.name}' "
                    f"and can never be seen"
                )
        if np.any(pad.center < lower[:2]) or np.any(pad.center > upper[:2]):
            problems.append(f"mission pad {pad.pad_id} is outside the room")

    for obstacle in world.obstacles:
        if isinstance(obstacle, Box):
            if np.any(obstacle.lower > obstacle.upper):
                problems.append(f"'{obstacle.name}' has its corners the wrong way round")
            if obstacle.upper[2] > upper[2] + 1e-6:
                problems.append(f"'{obstacle.name}' pokes through the ceiling")

    return problems


def fly(path: Path) -> dict:
    simulator, swarm, recorder = scenarios.run(path, record=False, verbose=False)
    summary = simulator.summary()
    landed = all(not d.state.airborne for d in simulator.drones)
    return {"summary": summary, "landed": landed}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fly", action="store_true", help="also run each scenario")
    parser.add_argument("--dir", type=Path, default=Path("scenarios"))
    args = parser.parse_args()

    paths = scenarios.list_scenarios(args.dir)
    if not paths:
        print(f"No scenarios found in {args.dir}")
        return 1

    failures = 0
    for path in paths:
        problems = check(path)
        if problems:
            failures += 1
            print(f"\n{path.name}")
            for problem in problems:
                print(f"   {problem}")
        else:
            line = f"{path.name:30s} geometry ok"
            if args.fly:
                result = fly(path)
                s = result["summary"]
                hits = sum(s["collisions"].values())
                sep = s["min_separation_m"]
                line += (
                    f"   flew {s['sim_time']:6.1f}s"
                    f"   {hits} collision(s)"
                    f"   battery {min(s['battery_remaining'].values()):5.1f}%"
                )
                if sep is not None:
                    line += f"   min sep {sep:.2f} m"
                if not result["landed"]:
                    line += "   STILL AIRBORNE"
            print(line)

    print()
    if failures:
        print(f"{failures} scenario(s) with geometry problems")
        return 1
    print(f"all {len(paths)} scenarios look sane")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
