"""The space the drones fly in: room bounds, obstacles and mission pads.

Coordinate convention (right-handed, matching the Tello SDK):

    x  forward   y  left   z  up            all in metres
    yaw = rotation about z, 0 means the nose points along +x

The Tello's own `go x y z` command uses the same axes, expressed in the drone's
body frame and in centimetres, so translating between the two is only a yaw
rotation and a factor of 100.

Obstacles come in three flavours, and the difference is about who decides where
they go:

    static  compiled into the room and never moves. A pillar, a wall panel.
    mocap   moves exactly where its script says, and cannot be pushed. A door
            swinging shut, a person walking a fixed route, a lift platform.
            Infinitely heavy, in effect.
    free    has mass and obeys the physics. A cardboard box the drone can nudge
            off a table, a stack that falls over.

Collision geometry no longer lives here.  MuJoCo owns contacts now; this module
describes what exists and where it is meant to be, and `physics.py` turns that
into a model.  What stayed is the sensing geometry -- what the downward
time-of-flight sensor can see, which mission pad is in view -- because those are
questions about the Tello's sensors, not about rigid bodies.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from .wind import NoWind, WindField


# ----------------------------------------------------------------------
# How an obstacle moves
# ----------------------------------------------------------------------


class Motion:
    """Where a movable obstacle should be at a given time.

    Scripted motion is deliberately kinematic rather than forced.  A door that
    is *pushed* shut by a simulated actuator would need a motor model and a
    hinge, and would open again when a drone leaned on it.  A door that is
    *placed* shut closes on schedule, every run, which is what a repeatable
    benchmark needs.
    """

    kind = "mocap"

    def pose_at(self, t: float) -> tuple[np.ndarray, float]:
        """Return (position, yaw) at simulated time t."""
        raise NotImplementedError

    def describe(self) -> dict:
        raise NotImplementedError


class Fixed(Motion):
    """Does not move. The obstacle is baked into the room."""

    kind = "static"

    def pose_at(self, t: float):  # pragma: no cover - never called for static
        return None, None

    def describe(self) -> dict:
        return {"type": "fixed"}


class Dynamic(Motion):
    """Has mass and is left to the solver. Falls, topples, gets pushed."""

    kind = "free"

    def pose_at(self, t: float):  # pragma: no cover - physics decides, not us
        return None, None

    def describe(self) -> dict:
        return {"type": "dynamic"}


class Placed(Motion):
    """Movable, but positioned by hand rather than by a script.

    What a scripted obstacle becomes once something calls `move_obstacle` on it.
    Keeping the script would mean the placement was silently undone on the next
    step, which is indistinguishable from the call having failed.
    """

    kind = "mocap"

    def pose_at(self, t: float):
        return None, None

    def describe(self) -> dict:
        return {"type": "placed"}


@dataclass
class Waypoints(Motion):
    """Move through a list of points, one leg at a time.

    Times are cumulative seconds.  With `loop` the sequence restarts, which is
    how you get a door that opens and shuts all afternoon without writing out
    every cycle.
    """

    points: list
    times: list
    loop: bool = True
    yaws: list | None = None

    kind = "mocap"

    def __post_init__(self) -> None:
        self.points = [np.asarray(p, dtype=float) for p in self.points]
        self.times = [float(t) for t in self.times]
        if len(self.points) != len(self.times):
            raise ValueError(
                f"Waypoint motion needs one time per point: "
                f"got {len(self.points)} points and {len(self.times)} times."
            )
        if len(self.points) < 2:
            raise ValueError("Waypoint motion needs at least two points.")
        if any(b <= a for a, b in itertools.pairwise(self.times)):
            raise ValueError("Waypoint times must increase.")
        if self.yaws is not None:
            self.yaws = [float(np.deg2rad(y)) for y in self.yaws]
            if len(self.yaws) != len(self.points):
                raise ValueError("Waypoint motion needs one yaw per point, or none.")

    def pose_at(self, t: float) -> tuple[np.ndarray, float]:
        start, end = self.times[0], self.times[-1]
        span = end - start
        if self.loop and span > 0:
            t = start + (t - start) % span
        t = float(np.clip(t, start, end))

        index = int(np.searchsorted(self.times, t, side="right")) - 1
        index = int(np.clip(index, 0, len(self.points) - 2))
        t0, t1 = self.times[index], self.times[index + 1]
        alpha = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)

        position = (1 - alpha) * self.points[index] + alpha * self.points[index + 1]
        if self.yaws is None:
            yaw = 0.0
        else:
            y0, y1 = self.yaws[index], self.yaws[index + 1]
            delta = float((y1 - y0 + np.pi) % (2 * np.pi) - np.pi)
            yaw = y0 + alpha * delta
        return position, yaw

    def describe(self) -> dict:
        return {
            "type": "waypoints",
            "points": [p.tolist() for p in self.points],
            "times": list(self.times),
            "loop": self.loop,
            "yaws": None if self.yaws is None else [float(np.rad2deg(y)) for y in self.yaws],
        }


@dataclass
class Orbit(Motion):
    """Circle a point at constant rate. A person pacing, a drone-shaped hazard."""

    center: np.ndarray
    radius: float
    period: float = 10.0
    height: float = 1.0
    face_travel: bool = True

    kind = "mocap"

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=float)[:2]
        if self.period <= 0:
            raise ValueError("Orbit period must be positive.")

    def pose_at(self, t: float) -> tuple[np.ndarray, float]:
        angle = 2.0 * np.pi * (t / self.period)
        position = np.array(
            [
                self.center[0] + self.radius * np.cos(angle),
                self.center[1] + self.radius * np.sin(angle),
                self.height,
            ]
        )
        yaw = angle + np.pi / 2 if self.face_travel else 0.0
        return position, float(yaw)

    def describe(self) -> dict:
        return {
            "type": "orbit",
            "center": self.center.tolist(),
            "radius": self.radius,
            "period": self.period,
            "height": self.height,
            "face_travel": self.face_travel,
        }


@dataclass
class Swing(Motion):
    """Rotate about a fixed point between two angles. A door, a barrier arm."""

    pivot: np.ndarray
    from_deg: float = 0.0
    to_deg: float = 90.0
    period: float = 8.0
    hold: float = 0.0

    kind = "mocap"

    def __post_init__(self) -> None:
        self.pivot = np.asarray(self.pivot, dtype=float)
        if self.pivot.shape[0] == 2:
            self.pivot = np.array([self.pivot[0], self.pivot[1], 0.0])
        if self.period <= 0:
            raise ValueError("Swing period must be positive.")

    def pose_at(self, t: float) -> tuple[np.ndarray, float]:
        # A raised cosine rather than a triangle wave, so the door slows down as
        # it reaches each end instead of reversing instantly. An instant
        # reversal is a discontinuity the contact solver has to absorb.
        phase = (t % self.period) / self.period
        blend = 0.5 * (1.0 - np.cos(2.0 * np.pi * phase))
        if self.hold > 0:
            blend = float(np.clip(blend * (1.0 + 2.0 * self.hold) - self.hold, 0.0, 1.0))
        yaw = np.deg2rad(self.from_deg + blend * (self.to_deg - self.from_deg))
        return self.pivot.copy(), float(yaw)

    def describe(self) -> dict:
        return {
            "type": "swing",
            "pivot": self.pivot.tolist(),
            "from_deg": self.from_deg,
            "to_deg": self.to_deg,
            "period": self.period,
            "hold": self.hold,
        }


def motion_from_config(spec) -> Motion:
    """Build a Motion from the plain dict or string a scenario file holds."""
    if spec is None:
        return Fixed()
    if isinstance(spec, str):
        spec = {"type": spec}

    kind = spec.get("type", "fixed")
    if kind in ("fixed", "static"):
        return Fixed()
    if kind == "placed":
        return Placed()
    if kind in ("dynamic", "free"):
        return Dynamic()
    if kind == "waypoints":
        return Waypoints(
            points=spec["points"],
            times=spec["times"],
            loop=bool(spec.get("loop", True)),
            yaws=spec.get("yaws_deg"),
        )
    if kind == "orbit":
        return Orbit(
            center=spec["center"],
            radius=float(spec["radius"]),
            period=float(spec.get("period", 10.0)),
            height=float(spec.get("height", 1.0)),
            face_travel=bool(spec.get("face_travel", True)),
        )
    if kind == "swing":
        return Swing(
            pivot=spec["pivot"],
            from_deg=float(spec.get("from_deg", 0.0)),
            to_deg=float(spec.get("to_deg", 90.0)),
            period=float(spec.get("period", 8.0)),
            hold=float(spec.get("hold", 0.0)),
        )
    raise ValueError(
        f"Unknown motion type {kind!r}. "
        "Expected one of: fixed, dynamic, waypoints, orbit, swing."
    )


# ----------------------------------------------------------------------
# Shapes
# ----------------------------------------------------------------------


def _quat_matrix(q) -> np.ndarray:
    """Rotation matrix, columns are body axes in world coordinates, from w-first q."""
    w, x, y, z = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _yaw_matrix(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass
class _Obstacle:
    """Fields every obstacle shares.

    Two different things live on an obstacle and must not be confused:

    the shape    what the scenario defined: how big, and where it started.
                 Never changed after loading. Everything that builds a model or
                 draws a room reads this.
    the pose     where the physics engine has the obstacle's body right now.
                 Written every step for anything movable, None for static
                 obstacles and before the engine has run.

    They used to be one thing: the engine wrote each step's pose back into the
    shape. That quietly broke doors. A door's body sits on its hinge, not its
    middle, so every door ended up described -- and drawn, and measured for
    clearance -- half a metre from where it physically was.
    """

    name: str = "obstacle"
    motion: Motion = field(default_factory=Fixed)
    mass: float = 1.0
    rgba: tuple = (0.45, 0.50, 0.58, 1.0)
    yaw: float = 0.0

    live_position: np.ndarray | None = field(default=None, repr=False, compare=False)
    live_quat: np.ndarray | None = field(default=None, repr=False, compare=False)

    @property
    def motion_kind(self) -> str:
        return self.motion.kind

    @property
    def movable(self) -> bool:
        return self.motion.kind != "static"

    @property
    def rest_center(self) -> np.ndarray:
        """Centre of the shape as the scenario placed it."""
        raise NotImplementedError

    def body_origin(self) -> np.ndarray:
        """The point the engine's pose refers to, in the scenario's layout.

        The hinge for a swinging door, the shape's own centre for everything else.
        """
        if isinstance(self.motion, Swing):
            return np.asarray(self.motion.pivot, dtype=float)
        return self.rest_center

    def frame(self) -> tuple[np.ndarray, np.ndarray]:
        """Where the shape's centre is right now, and its rotation matrix.

        The same sum MuJoCo does for a geom hung off its body, so this agrees
        with the solver exactly rather than approximately.
        """
        origin = self.body_origin()
        offset = self.rest_center - origin
        if self.live_position is None or self.live_quat is None:
            rotation = _yaw_matrix(self.yaw)
            return origin + rotation @ offset, rotation
        rotation = _quat_matrix(self.live_quat)
        return np.asarray(self.live_position, dtype=float) + rotation @ offset, rotation


@dataclass
class Box(_Obstacle):
    """A rectangular obstacle: a table, a pillar, a wall panel, a door."""

    lower: np.ndarray = field(default_factory=lambda: np.zeros(3))
    upper: np.ndarray = field(default_factory=lambda: np.ones(3))

    def __post_init__(self) -> None:
        self.lower = np.asarray(self.lower, dtype=float)
        self.upper = np.asarray(self.upper, dtype=float)
        if np.any(self.upper < self.lower):
            raise ValueError(
                f"Box {self.name!r} has upper corner below lower: "
                f"{self.upper.tolist()} < {self.lower.tolist()}"
            )

    @property
    def center(self) -> np.ndarray:
        return 0.5 * (self.lower + self.upper)

    @property
    def rest_center(self) -> np.ndarray:
        return self.center

    @property
    def half_extents(self) -> np.ndarray:
        return 0.5 * (self.upper - self.lower)

    def top_at(self, x: float, y: float) -> float | None:
        centre, rotation = self.frame()
        # Bounds of the box as turned: exact while it is square to the room,
        # a little generous once it has swung or tipped. Good enough for what the
        # downward range sensor sees, which is all this is used for.
        reach = np.abs(rotation) @ self.half_extents
        if abs(x - centre[0]) <= reach[0] and abs(y - centre[1]) <= reach[1]:
            return float(centre[2] + reach[2])
        return None

    def describe(self) -> dict:
        return {
            "type": "box",
            "name": self.name,
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist(),
            "motion": self.motion.describe(),
            "motion_kind": self.motion_kind,
            "mass": self.mass,
            "rgba": list(self.rgba),
            "yaw": self.yaw,
        }


@dataclass
class Cylinder(_Obstacle):
    """A vertical cylinder: a column, a person, a chair leg."""

    center_xy: np.ndarray = field(default_factory=lambda: np.zeros(2))
    radius: float = 0.2
    z_min: float = 0.0
    z_max: float = 3.0

    def __post_init__(self) -> None:
        self.center_xy = np.asarray(self.center_xy, dtype=float)[:2]

    @property
    def center(self) -> np.ndarray:
        return np.array([self.center_xy[0], self.center_xy[1], 0.5 * (self.z_min + self.z_max)])

    @property
    def rest_center(self) -> np.ndarray:
        return self.center

    @property
    def half_height(self) -> float:
        return 0.5 * (self.z_max - self.z_min)

    def top_at(self, x: float, y: float) -> float | None:
        centre, rotation = self.frame()
        if abs(rotation[2, 2]) > 0.999:        # still standing up
            planar = float(np.linalg.norm(np.array([x, y]) - centre[:2]))
            return float(centre[2] + self.half_height) if planar <= self.radius else None
        reach = np.abs(rotation) @ np.array([self.radius, self.radius, self.half_height])
        if abs(x - centre[0]) <= reach[0] and abs(y - centre[1]) <= reach[1]:
            return float(centre[2] + reach[2])
        return None

    def describe(self) -> dict:
        return {
            "type": "cylinder",
            "name": self.name,
            "center_xy": self.center_xy.tolist(),
            "radius": self.radius,
            "z_min": self.z_min,
            "z_max": self.z_max,
            "motion": self.motion.describe(),
            "motion_kind": self.motion_kind,
            "mass": self.mass,
            "rgba": list(self.rgba),
            "yaw": self.yaw,
        }


@dataclass
class Sphere(_Obstacle):
    """A ball. Mostly useful as a light free body the drone can knock about."""

    center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    radius: float = 0.15

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=float)

    @property
    def rest_center(self) -> np.ndarray:
        return self.center

    def top_at(self, x: float, y: float) -> float | None:
        centre, _ = self.frame()
        planar = float(np.linalg.norm(np.array([x, y]) - centre[:2]))
        if planar > self.radius:
            return None
        # Height of the sphere's surface directly under the point, not its top,
        # so the time-of-flight reading curves the way the real surface does.
        return float(centre[2] + np.sqrt(max(self.radius**2 - planar**2, 0.0)))

    def describe(self) -> dict:
        return {
            "type": "sphere",
            "name": self.name,
            "center": self.center.tolist(),
            "radius": self.radius,
            "motion": self.motion.describe(),
            "motion_kind": self.motion_kind,
            "mass": self.mass,
            "rgba": list(self.rgba),
            "yaw": self.yaw,
        }


@dataclass
class MissionPad:
    """A Tello EDU mission pad lying flat on the floor.

    The downward camera only sees a pad from between 0.3 m and 1.2 m, so the
    simulator reproduces that window rather than pretending detection is
    perfect.  A pad the agent cannot see is as good as absent.
    """

    pad_id: int
    center: np.ndarray
    yaw: float = 0.0              # radians, pad's own heading
    size: float = 0.18            # m, printed pads are 18 cm square

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=float)[:2]

    def describe(self) -> dict:
        return {
            "pad_id": self.pad_id,
            "center": self.center.tolist(),
            "yaw": self.yaw,
            "size": self.size,
        }


# ----------------------------------------------------------------------
# The world
# ----------------------------------------------------------------------


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

        # Bumped whenever the set of bodies changes. The simulator watches this
        # to know it must recompile the MuJoCo model, which is the one change
        # that cannot be applied to a live one.
        self.revision = 0
        self._check_names()

    def _check_names(self) -> None:
        seen = set()
        for obstacle in self.obstacles:
            if obstacle.name in seen:
                raise ValueError(
                    f"Two obstacles are both named {obstacle.name!r}. "
                    "Names address obstacles at runtime, so they must be unique."
                )
            seen.add(obstacle.name)

    # -- air ---------------------------------------------------------------

    def step_wind(self, dt: float) -> None:
        self.wind.step(dt)

    def wind_at(self, position: np.ndarray, t: float) -> np.ndarray:
        return np.asarray(self.wind.velocity(position, t), dtype=float)

    # -- editing the room --------------------------------------------------

    def add_obstacle(self, obstacle) -> None:
        """Put something new in the room, mid-flight if you like.

        The model is recompiled on the next step. That costs a millisecond or
        two for a room this size, which is why this is allowed at all: the
        alternative, pre-allocating hidden bodies, caps how much a scenario can
        contain and leaves phantom geometry in every render.
        """
        if any(o.name == obstacle.name for o in self.obstacles):
            raise ValueError(f"An obstacle named {obstacle.name!r} is already here.")
        self.obstacles.append(obstacle)
        self.revision += 1

    def remove_obstacle(self, name: str) -> None:
        before = len(self.obstacles)
        self.obstacles = [o for o in self.obstacles if o.name != name]
        if len(self.obstacles) == before:
            raise KeyError(f"No obstacle named {name!r} to remove.")
        self.revision += 1

    def get_obstacle(self, name: str):
        for obstacle in self.obstacles:
            if obstacle.name == name:
                return obstacle
        raise KeyError(f"No obstacle named {name!r}.")

    def clear_obstacles(self) -> None:
        if self.obstacles:
            self.obstacles = []
            self.revision += 1

    def scripted_poses(self, t: float) -> dict:
        """Where every kinematically-driven obstacle should be at time t."""
        poses = {}
        for obstacle in self.obstacles:
            if obstacle.motion.kind != "mocap":
                continue
            position, yaw = obstacle.motion.pose_at(t)
            if position is not None:
                poses[obstacle.name] = (position, yaw)
        return poses

    # -- sensing geometry --------------------------------------------------

    def ground_height_at(self, position: np.ndarray) -> float:
        """Height of the highest surface directly below a point.

        The downward time-of-flight sensor measures to whatever is under the
        drone, not to the floor, so flying over a table makes the reading jump.
        Reproducing that stops an agent from trusting `tof` as absolute altitude.

        A moving obstacle is read at wherever it currently is, because the
        physics engine writes each obstacle's live pose back onto it every step.
        """
        ground = float(self.bounds_lower[2])
        x, y, z = float(position[0]), float(position[1]), float(position[2])
        for obstacle in self.obstacles:
            top = obstacle.top_at(x, y)
            if top is not None and top <= z:
                ground = max(ground, top)
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

    @staticmethod
    def distance_to(obstacle, position: np.ndarray) -> float:
        """Distance from a point to one obstacle's surface, zero if inside.

        Measured in the obstacle's own frame, wherever the engine has put it and
        however it is turned, so a swung door or a knocked-over brick is measured
        where it actually is. Exact for boxes, cylinders and spheres alike.
        """
        p = np.asarray(position, dtype=float)
        centre, rotation = obstacle.frame()
        if isinstance(obstacle, Sphere):
            return max(float(np.linalg.norm(p - centre)) - obstacle.radius, 0.0)
        local = rotation.T @ (p - centre)
        if isinstance(obstacle, Cylinder):
            radial = max(float(np.linalg.norm(local[:2])) - obstacle.radius, 0.0)
            vertical = max(abs(float(local[2])) - obstacle.half_height, 0.0)
            return float(np.hypot(radial, vertical))
        outside = np.maximum(np.abs(local) - obstacle.half_extents, 0.0)
        return float(np.linalg.norm(outside))

    def nearest_obstacle(self, position: np.ndarray):
        """The closest obstacle and how far away it is, ignoring the room itself."""
        best, best_distance = None, float("inf")
        for obstacle in self.obstacles:
            distance = self.distance_to(obstacle, position)
            if distance < best_distance:
                best, best_distance = obstacle, distance
        return best, best_distance

    def clearance_at(self, position: np.ndarray) -> float:
        """Distance from a point to the nearest surface, room walls included."""
        p = np.asarray(position, dtype=float)
        gaps = np.concatenate([p - self.bounds_lower, self.bounds_upper - p])
        return min(float(np.min(gaps)), self.nearest_obstacle(p)[1])

    def describe(self) -> dict:
        return {
            "name": self.name,
            "bounds_lower": self.bounds_lower.tolist(),
            "bounds_upper": self.bounds_upper.tolist(),
            "obstacles": [o.describe() for o in self.obstacles],
            "mission_pads": [p.describe() for p in self.mission_pads],
            "wind": self.wind.describe(),
        }


def obstacle_from_config(entry: dict):
    """Build one obstacle from the plain dict a scenario file holds."""
    kind = entry.get("type", "box")
    shared = dict(
        name=entry.get("name", kind),
        motion=motion_from_config(entry.get("motion")),
        mass=float(entry.get("mass", 1.0)),
        rgba=tuple(entry.get("rgba", (0.45, 0.50, 0.58, 1.0))),
        yaw=float(np.deg2rad(entry.get("yaw_deg", 0.0))),
    )

    if kind == "box":
        return Box(lower=entry["lower"], upper=entry["upper"], **shared)
    if kind == "cylinder":
        return Cylinder(
            center_xy=entry["center_xy"],
            radius=float(entry["radius"]),
            z_min=float(entry.get("z_min", 0.0)),
            z_max=float(entry.get("z_max", 3.0)),
            **shared,
        )
    if kind == "sphere":
        return Sphere(center=entry["center"], radius=float(entry["radius"]), **shared)
    raise ValueError(
        f"Unknown obstacle type: {kind!r}. Expected one of: box, cylinder, sphere."
    )


def from_config(spec: dict | None, rng: np.random.Generator | None = None) -> World:
    """Build a world from a plain dict so scenarios can live in YAML."""
    from . import wind as wind_module

    spec = spec or {}
    obstacles = [obstacle_from_config(entry) for entry in spec.get("obstacles", [])]

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
