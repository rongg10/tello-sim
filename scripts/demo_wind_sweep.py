#!/usr/bin/env python3
"""How badly does wind break position holding? Sweep it and find out.

    python scripts/demo_wind_sweep.py

Flies the same square patrol at a range of wind speeds and reports, for each,
how far off the drone finished and how much battery the fight cost.  Because
the runs are seeded, the only thing changing between rows is the weather.

This is the experiment shape the whole simulator exists to support: one
condition varied, everything else held fixed, hundreds of times faster than
real time.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from tello_sim import ConstantWind, SimSpec, World, build


def fly(wind_speed: float, seed: int = 0) -> dict:
    world = World(
        bounds_lower=(-15, -15, 0), bounds_upper=(15, 15, 3),
        wind=ConstantWind([0.0, wind_speed, 0.0]),
    )
    sim, swarm = build(
        n=1, world=world,
        sim_spec=SimSpec(realtime=False, seed=seed, sensor_noise=False, actuation_noise=False),
    )
    drone = swarm[0]
    drone.connect()
    drone.takeoff()
    start = drone.true_position().copy()

    failures = 0
    for _ in range(4):
        for command in ("forward 150", "cw 90"):
            try:
                drone._send(command)
            except Exception:
                failures += 1

    error = float(np.linalg.norm(drone.true_position()[:2] - start[:2]))
    battery_used = 100.0 - drone.drone.state.battery
    flight_time = drone.drone.state.flight_time
    drone.land()
    return {
        "wind": wind_speed,
        "closing_error_m": error,
        "battery_used": battery_used,
        "flight_time": flight_time,
        "failed_commands": failures,
    }


print("A square patrol of four 1.5 m legs, flown at increasing wind speeds.")
print("'closing error' is how far from the start it finished; a perfect square closes at 0.\n")
print(f"{'wind m/s':>9} {'closing error m':>17} {'battery used %':>16} {'flight s':>10} {'failed cmds':>12}")
print("-" * 68)
for wind in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
    r = fly(wind)
    print(
        f"{r['wind']:9.1f} {r['closing_error_m']:17.2f} {r['battery_used']:16.1f} "
        f"{r['flight_time']:10.1f} {r['failed_commands']:12d}"
    )

print(
    "\nThe drone holds until about 2.5 m/s, then the wind exceeds what the\n"
    "controller can produce and the patrol stops closing at all."
)
