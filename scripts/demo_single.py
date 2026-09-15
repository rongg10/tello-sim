#!/usr/bin/env python3
"""The simplest possible use: one drone, no network, a short flight.

    python scripts/demo_single.py

Note the import at the top. Swapping that one line is the difference between
flying the simulator and flying the real drone -- everything below it is
ordinary djitellopy code.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from tello_sim import ConstantWind, Recorder, SimSpec, World, build
from tello_sim.render import dashboard

recorder = Recorder("demo_single")
world = World(
    bounds_lower=(-3, -3, 0), bounds_upper=(3, 3, 2.5),
    wind=ConstantWind([0.6, 0.0, 0.0]),
    name="demo: 0.6 m/s headwind",
)
sim, swarm = build(n=1, world=world, sim_spec=SimSpec(realtime=False, seed=1), recorder=recorder)
drone = swarm[0]

# ---- from here down this is just djitellopy ----
drone.connect()
print(f"battery {drone.get_battery()}%")

drone.takeoff()
drone.set_speed(40)

for _ in range(4):
    drone.move_forward(120)
    drone.rotate_clockwise(90)

drone.move_up(60)
drone.sleep(2.0)          # in a script, use drone.sleep, not time.sleep
drone.land()
# ---- end of ordinary flight code ----

recorder.close()
print(f"\nflew {sim.time:.1f}s of simulated time")
print(f"ended {np.linalg.norm(drone.true_position()[:2]):.2f} m from where it started "
      f"(a perfect square would end at 0.00)")
print(f"battery left: {drone.get_battery()}%")
print(f"log:       {recorder.directory}")
print(f"dashboard: {dashboard(recorder.directory)}")
