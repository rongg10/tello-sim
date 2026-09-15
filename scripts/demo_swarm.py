#!/usr/bin/env python3
"""Three drones in one sky, and the number that decides whether it works.

    python scripts/demo_swarm.py

Formation flight looks easy in still air: everyone is told the same thing and
everyone does it.  A gust that hits all three equally does not break it either
-- the whole formation translates and the gaps stay put.

What breaks it is wind that is *not* the same everywhere.  Here a draught
crosses only one side of the room, so the drone flying through it is pushed
toward its neighbour while the other two hold station.  No drone can see this
happening: each one knows only that it is holding position, and it is, as far
as it can tell.  The gap closes anyway.

That is the case a coordination protocol exists for, and minimum separation is
the number that measures it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from tello_sim import Recorder, SimSpec, SumOfWinds, TurbulentWind, WindTunnel, World, build
from tello_sim.render import dashboard

SPACING = 1.2

recorder = Recorder("demo_swarm")
world = World(
    bounds_lower=(-5, -5, 0), bounds_upper=(5, 5, 3),
    wind=SumOfWinds([
        # Light turbulence everywhere, so nothing is unrealistically clean.
        TurbulentWind(sigma=0.25, theta=0.9, rng=np.random.default_rng(4)),
        # A 2 m/s draught confined to y > 0.4: only the left-hand drone is in it.
        WindTunnel(velocity_mps=(0.0, -2.0, 0.0),
                   lower=(-10.0, 0.4, -1.0), upper=(10.0, 10.0, 5.0)),
    ]),
    name="three drones, draught on one side only",
)

sim, swarm = build(
    n=3, world=world,
    start_positions=[(0.0, -SPACING, 0.0), (0.0, 0.0, 0.0), (0.0, SPACING, 0.0)],
    sim_spec=SimSpec(realtime=False, seed=4),
    recorder=recorder,
)

swarm.connect()
swarm.takeoff()
swarm.set_speed(50)

# Every drone is given exactly the same instruction at the same instant.
swarm.parallel(lambda i, t: t.move_up(50))
swarm.sleep(3.0)
swarm.parallel(lambda i, t: t.move_forward(200))
swarm.sleep(6.0)
swarm.parallel(lambda i, t: t.move_forward(150))
swarm.sleep(3.0)
swarm.land()

recorder.close()

positions = np.array([t.true_position() for t in swarm])
names = [t.drone.name for t in swarm]
gaps = {
    f"{names[i]}-{names[j]}": float(np.linalg.norm(positions[i] - positions[j]))
    for i in range(len(positions)) for j in range(i + 1, len(positions))
}

print(f"\ncommanded separation:     {SPACING:.2f} m")
print(f"closest they ever came:   {sim.min_separation:.2f} m")
print(f"actual contacts:          {sim.summary()['near_misses']}")
print("\nfinal gaps:")
for pair, distance in gaps.items():
    print(f"  {pair:20s} {distance:.2f} m")
print("\nfinal y positions (the draught pushes toward -y):")
for name, position in zip(names, positions):
    print(f"  {name:10s} y = {position[1]:+.2f} m   (started {SPACING * (names.index(name) - 1):+.2f})")

print(f"\nlog:       {recorder.directory}")
print(f"dashboard: {dashboard(recorder.directory)}")
print(
    "\nEvery drone received identical commands, so the change in separation came\n"
    "entirely from the air. Neither drone in the closing pair could detect it\n"
    "alone: each believes it is holding station, and each is right."
)
