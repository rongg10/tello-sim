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
"""

from .actions import Action, Hover, MoveTo, RCVelocity, RotateTo
from .client import SimSwarm, SimTello, TelloSimError, build
from .config import DEFAULT_DRONE, DEFAULT_SIM, DroneSpec, SimSpec
from .drone import SimDrone
from .dynamics import DroneState
from .protocol import parse_state
from .recorder import Recorder, latest_run, load_run
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
from .world import Box, Cylinder, MissionPad, World

__all__ = [
    "Action", "Hover", "MoveTo", "RCVelocity", "RotateTo",
    "SimSwarm", "SimTello", "TelloSimError", "build",
    "DEFAULT_DRONE", "DEFAULT_SIM", "DroneSpec", "SimSpec",
    "SimDrone", "DroneState", "parse_state",
    "Recorder", "latest_run", "load_run",
    "Simulator",
    "BoundaryLayerWind", "ConstantWind", "GustBurst", "NoWind",
    "SumOfWinds", "Transient", "TurbulentWind", "WindField", "WindTunnel",
    "Box", "Cylinder", "MissionPad", "World",
]

__version__ = "0.1.0"


def __getattr__(name: str):
    # The UDP server pulls in sockets and threads; keep the import lazy so that
    # research scripts that never touch the network do not pay for it.
    if name in ("TelloServer", "serve"):
        from . import server

        return getattr(server, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
