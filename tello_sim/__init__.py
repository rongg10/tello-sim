"""A DJI Tello simulator: same SDK, same wire protocol, plus wind and battery.

Two entry points, for two different jobs.

Checking that flight code works -- run the UDP server and point the existing
controller at 127.0.0.1.  Nothing in the flight code changes:

    from tello_sim import Simulator, World, TelloServer

Running experiments -- skip the network, go faster than real time, use as many
drones as you like:

    from tello_sim import build
    sim, swarm = build(n=3)
    swarm.connect(); swarm.takeoff()

Scoring a policy -- run a scenario across several seeds and get SR, OSR, SPL
and the energy it cost:

    from tello_sim import benchmark, scenarios
    report = benchmark.run(scenarios.load("scenarios/11_moving_door.yaml"))
    print(report.table())

Physics is MuJoCo. `pip install mujoco`, on Python 3.10 or newer.
"""

from .actions import Action, Hover, MoveTo, RCVelocity, RotateTo
from .client import SimSwarm, SimTello, TelloSimError, build
from .config import DEFAULT_DRONE, DEFAULT_SIM, DroneSpec, SimSpec
from .drone import SimDrone
from .dynamics import DroneState
from .protocol import parse_state
from .recorder import Recorder, latest_run, load_run
from .benchmark import Goal, Report, RunResult, Task
from .simulator import Simulator
from .wind import (
    BoundaryLayerWind,
    ConstantWind,
    GustBurst,
    NoWind,
    SumOfWinds,
    Transient,
    TurbulentWind,
    WindField,
    WindTunnel,
)
from .world import (
    Box,
    Cylinder,
    Dynamic,
    Fixed,
    MissionPad,
    Orbit,
    Placed,
    Sphere,
    Swing,
    Waypoints,
    World,
)

__all__ = [
    "Action", "Hover", "MoveTo", "RCVelocity", "RotateTo",
    "SimSwarm", "SimTello", "TelloSimError", "build",
    "DEFAULT_DRONE", "DEFAULT_SIM", "DroneSpec", "SimSpec",
    "SimDrone", "DroneState", "parse_state",
    "Recorder", "latest_run", "load_run",
    "Simulator",
    "BoundaryLayerWind", "ConstantWind", "GustBurst", "NoWind",
    "SumOfWinds", "Transient", "TurbulentWind", "WindField", "WindTunnel",
    "Box", "Cylinder", "Sphere", "MissionPad", "World",
    "Fixed", "Dynamic", "Placed", "Waypoints", "Orbit", "Swing",
    "Goal", "Task", "Report", "RunResult",
]

__version__ = "0.1.0"


def __getattr__(name: str):
    # The UDP server pulls in sockets and threads; keep the import lazy so that
    # research scripts that never touch the network do not pay for it.
    if name in ("TelloServer", "serve"):
        from . import server

        return getattr(server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
