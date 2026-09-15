"""The space the drones fly in: room bounds, obstacles and mission pads.

Coordinate convention (right-handed, matching the Tello SDK):

    x  forward   y  left   z  up            all in metres
    yaw = rotation about z, 0 means the nose points along +x

The Tello's own `go x y z` command uses the same axes, expressed in the drone's
body frame and in centimetres, so translating between the two is only a yaw
rotation and a factor of 100.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .wind import NoWind, WindField


@dataclass
class Box:
    """An axis-aligned rectangular obstacle: a table, a pillar, a wall panel."""

    lower: np.ndarray
    upper: np.ndarray
    name: str = "box"

    def __post_init__(self) -> None:
        self.lower = np.asarray(self.lower, dtype=float)
        self.upper = np.asarray(self.upper, dtype=float)

    def closest_point(self, p: np.ndarray) -> np.ndarray:
        return np.clip(p, self.lower, self.upper)

    def penetration(self, p: np.ndarray, radius: float):
        """Return (depth, outward_normal) if a sphere at p overlaps, else None."""
        q = self.closest_point(p)
        delta = p - q
        dist = float(np.linalg.norm(delta))

        if dist > radius:
            return None

        if dist > 1e-9:
            return radius - dist, delta / dist

        # Centre is inside the box: push out through the nearest face.
        gaps_low = p - self.lower
        gaps_high = self.upper - p
        axis, sign, gap = 0, 1.0, float("inf")
        for a in range(3):
            if gaps_low[a] < gap:
                axis, sign, gap = a, -1.0, float(gaps_low[a])
            if gaps_high[a] < gap:
                axis, sign, gap = a, 1.0, float(gaps_high[a])
        normal = np.zeros(3)
        normal[axis] = sign
        return gap + radius, normal

    def describe(self) -> dict:
        return {
            "type": "box",
            "name": self.name,
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist(),
        }


@dataclass
class Cylinder:
    """A vertical cylinder: a column, a person, a chair leg."""

    center_xy: np.ndarray
    radius: float
    z_min: float = 0.0
    z_max: float = 3.0
    name: str = "cylinder"

    def __post_init__(self) -> None:
        self.center_xy = np.asarray(self.center_xy, dtype=float)

    def penetration(self, p: np.ndarray, radius: float):
        if p[2] < self.z_min - radius or p[2] > self.z_max + radius:
            return None
        delta = p[:2] - self.center_xy
        dist = float(np.linalg.norm(delta))
        reach = self.radius + radius
        if dist > reach:
            return None
        if dist < 1e-9:
            normal = np.array([1.0, 0.0, 0.0])
        else:
            normal = np.array([delta[0] / dist, delta[1] / dist, 0.0])
        return reach - dist, normal

    def describe(self) -> dict:
        return {
            "type": "cylinder",
            "name": self.name,
            "center_xy": self.center_xy.tolist(),
            "radius": self.radius,
            "z_min": self.z_min,
            "z_max": self.z_max,
        }


@dataclass
class MissionPad:
    """A Tello EDU mission pad lying flat on the floor.

    The downward camera only sees a pad from between 0.3 m and 1.2 m, so the
    simulator reproduces that window rather than pretending detection is
    perfect.  A pad the agent cannot see is as good as absent.
    """

    pad_id: int
    center: np.ndarray            # (x, y) in metres, pad sits on the floor
    yaw: float = 0.0              # radians, pad's own heading
    size: float = 0.18            # m, printed pads are 18 cm square

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=float)

    def describe(self) -> dict:
        return {
            "pad_id": self.pad_id,
            "center": self.center.tolist(),
            "yaw": self.yaw,
            "size": self.size,
        }


class World:
    """Everything outside the drones: the room, what is in it, and the air."""

    def __init__(
        self,
        bounds_lower: Sequence[float] = (-3.0, -3.0, 0.0),
        bounds_upper: Sequence[float] = (3.0, 3.0, 2.5),
        obstacles: Iterable = (),
        mission_pads: Iterable[MissionPad] = (),
        wind: WindField | None = None,
        name: str = "room",
    ):
        self.bounds_lower = np.asarray(bounds_lower, dtype=float)
        self.bounds_upper = np.asarray(bounds_upper, dtype=float)
        self.obstacles = list(obstacles)
        self.mission_pads = list(mission_pads)
        self.wind = wind if wind is not None else NoWind()
        self.name = name

    # -- air ---------------------------------------------------------------

    def step_wind(self, dt: float) -> None:
        self.wind.step(dt)

    def wind_at(self, position: np.ndarray, t: float) -> np.ndarray:
        return np.asarray(self.wind.velocity(position, t), dtype=float)

    # -- geometry ----------------------------------------------------------

    def resolve_collisions(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        radius: float,
        ground_clearance: float = 0.0,
    ):
        """Push a sphere out of anything it overlaps.

        Returns (position, velocity, hits) where hits lists what was struck.
        Velocity into a surface is removed, velocity along it is kept, so the
        drone slides along a wall rather than sticking to it.
        """
        p = position.copy()
        v = velocity.copy()
        hits: list[str] = []

        # Room boundary. The floor uses ground_clearance rather than the
        # collision radius, so a landed drone reports a height of zero the way
        # the real one does instead of hovering a body-radius above the ground.
        lo = self.bounds_lower + radius
        hi = self.bounds_upper - radius
        lo[2] = self.bounds_lower[2] + ground_clearance
        for axis, label in enumerate(("x", "y", "z")):
            if p[axis] < lo[axis]:
                p[axis] = lo[axis]
                if v[axis] < 0:
                    v[axis] = 0.0
                hits.append(f"bound_{label}_min")
            elif p[axis] > hi[axis]:
                p[axis] = hi[axis]
                if v[axis] > 0:
                    v[axis] = 0.0
                hits.append(f"bound_{label}_max")

        for obstacle in self.obstacles:
            result = obstacle.penetration(p, radius)
            if result is None:
                continue
            depth, normal = result
            p = p + normal * depth
            into = float(np.dot(v, normal))
            if into < 0.0:
                v = v - normal * into
            hits.append(obstacle.name)

        return p, v, hits

    def ground_height_at(self, position: np.ndarray) -> float:
        """Height of the highest surface directly below a point.

        The downward time-of-flight sensor measures to whatever is under the
        drone, not to the floor, so flying over a table makes the reading jump.
        Reproducing that stops an agent from trusting `tof` as absolute altitude.
        """
        ground = float(self.bounds_lower[2])
        x, y, z = float(position[0]), float(position[1]), float(position[2])
        for obstacle in self.obstacles:
            if isinstance(obstacle, Box):
                if (
                    obstacle.lower[0] <= x <= obstacle.upper[0]
                    and obstacle.lower[1] <= y <= obstacle.upper[1]
                    and obstacle.upper[2] <= z
                ):
                    ground = max(ground, float(obstacle.upper[2]))
            elif isinstance(obstacle, Cylinder):
                planar = float(np.linalg.norm(np.array([x, y]) - obstacle.center_xy))
                if planar <= obstacle.radius and obstacle.z_max <= z:
                    ground = max(ground, float(obstacle.z_max))
        return ground

    def mission_pad_under(self, position: np.ndarray, spec) -> MissionPad | None:
        """Which pad, if any, the downward camera can currently see."""
        h = float(position[2])
        if not (spec.pad_detect_min_height <= h <= spec.pad_detect_max_height):
            return None
        best, best_dist = None, float("inf")
        for pad in self.mission_pads:
            d = float(np.linalg.norm(position[:2] - pad.center))
            if d <= spec.pad_detect_radius and d < best_dist:
                best, best_dist = pad, d
        return best

    def describe(self) -> dict:
        return {
            "name": self.name,
            "bounds_lower": self.bounds_lower.tolist(),
            "bounds_upper": self.bounds_upper.tolist(),
            "obstacles": [o.describe() for o in self.obstacles],
            "mission_pads": [p.describe() for p in self.mission_pads],
            "wind": self.wind.describe(),
        }


def from_config(spec: dict | None, rng: np.random.Generator | None = None) -> World:
    """Build a world from a plain dict so scenarios can live in YAML."""
    from . import wind as wind_module

    spec = spec or {}
    obstacles: list = []
    for entry in spec.get("obstacles", []):
        kind = entry.get("type", "box")
        if kind == "box":
            obstacles.append(
                Box(entry["lower"], entry["upper"], entry.get("name", "box"))
            )
        elif kind == "cylinder":
            obstacles.append(
                Cylinder(
                    entry["center_xy"],
                    entry["radius"],
                    entry.get("z_min", 0.0),
                    entry.get("z_max", 3.0),
                    entry.get("name", "cylinder"),
                )
            )
        else:
            raise ValueError(f"Unknown obstacle type: {kind!r}")

    pads = [
        MissionPad(
            pad_id=p["pad_id"],
            center=p["center"],
            yaw=np.deg2rad(p.get("yaw_deg", 0.0)),
            size=p.get("size", 0.18),
        )
        for p in spec.get("mission_pads", [])
    ]

    return World(
        bounds_lower=spec.get("bounds_lower", (-3.0, -3.0, 0.0)),
        bounds_upper=spec.get("bounds_upper", (3.0, 3.0, 2.5)),
        obstacles=obstacles,
        mission_pads=pads,
        wind=wind_module.from_config(spec.get("wind"), rng),
        name=spec.get("name", "room"),
    )
