"""Wind models.

A wind field answers one question: "what is the air doing at this point, right
now?"  It returns a velocity vector in m/s in world coordinates.  The drone
feels it through aerodynamic drag on its velocity *relative to the air*, which
is where the physics lives (see dynamics.py).

Fields may be stateful (turbulence evolves over time); the simulator calls
step(dt) once per tick before querying.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


class WindField:
    """Base class. A field with no wind at all."""

    name = "none"

    def step(self, dt: float) -> None:
        """Advance any internal state. Stateless fields ignore this."""

    def velocity(self, position: np.ndarray, t: float) -> np.ndarray:
        return np.zeros(3)

    def __add__(self, other: "WindField") -> "WindField":
        return SumOfWinds([self, other])

    def describe(self) -> dict:
        return {"type": self.name}


class NoWind(WindField):
    name = "none"


class ConstantWind(WindField):
    """Steady wind blowing in one direction, everywhere, forever.

    The simplest useful case: it produces a constant sideways force, so the
    drone settles at a fixed position offset from its target.  That offset is
    a clean, measurable signal of how much control authority is left.
    """

    name = "constant"

    def __init__(self, velocity_mps: Sequence[float]):
        self._v = np.asarray(velocity_mps, dtype=float)

    def velocity(self, position: np.ndarray, t: float) -> np.ndarray:
        return self._v

    def describe(self) -> dict:
        return {"type": self.name, "velocity": self._v.tolist()}


class BoundaryLayerWind(WindField):
    """Wind that gets stronger with height, as it does outdoors.

    Uses the standard logarithmic profile: near the ground friction slows the
    air, and speed grows with the log of altitude.  Relevant for an AVLN agent
    that chooses its cruising altitude, because climbing costs it stability.
    """

    name = "boundary_layer"

    def __init__(
        self,
        speed_at_reference: float,
        direction: Sequence[float] = (1.0, 0.0, 0.0),
        reference_height: float = 2.0,
        roughness: float = 0.05,
    ):
        d = np.asarray(direction, dtype=float)
        norm = np.linalg.norm(d)
        self._dir = d / norm if norm > 0 else np.array([1.0, 0.0, 0.0])
        self._speed_ref = float(speed_at_reference)
        self._h_ref = float(reference_height)
        self._z0 = float(roughness)

    def velocity(self, position: np.ndarray, t: float) -> np.ndarray:
        h = max(float(position[2]), self._z0 * 1.01)
        scale = np.log(h / self._z0) / np.log(self._h_ref / self._z0)
        return self._dir * (self._speed_ref * max(scale, 0.0))

    def describe(self) -> dict:
        return {
            "type": self.name,
            "speed_at_reference": self._speed_ref,
            "reference_height": self._h_ref,
            "direction": self._dir.tolist(),
            "roughness": self._z0,
        }


class TurbulentWind(WindField):
    """Random gusts that wander instead of jumping around.

    Implemented as an Ornstein-Uhlenbeck process: the wind is pulled back
    towards a mean value but is constantly kicked by noise.  `theta` sets how
    fast it returns to the mean (larger = twitchier), `sigma` how hard it is
    kicked.  Unlike white noise this produces gusts with realistic duration,
    which is what actually upsets a small airframe.
    """

    name = "turbulent"

    def __init__(
        self,
        mean: Sequence[float] = (0.0, 0.0, 0.0),
        sigma: float = 0.8,
        theta: float = 0.7,
        vertical_scale: float = 0.4,
        rng: np.random.Generator | None = None,
    ):
        self._mean = np.asarray(mean, dtype=float)
        self._sigma = np.array([sigma, sigma, sigma * vertical_scale])
        self._theta = float(theta)
        self._rng = rng if rng is not None else np.random.default_rng(0)
        self._v = self._mean.copy()

    def step(self, dt: float) -> None:
        drift = -self._theta * (self._v - self._mean) * dt
        kick = self._sigma * np.sqrt(dt) * self._rng.standard_normal(3)
        self._v = self._v + drift + kick

    def velocity(self, position: np.ndarray, t: float) -> np.ndarray:
        return self._v

    def describe(self) -> dict:
        return {
            "type": self.name,
            "mean": self._mean.tolist(),
            "sigma": float(self._sigma[0]),
            "theta": self._theta,
        }


class GustBurst(WindField):
    """A single strong gust that arrives at a known time and then dies away.

    Deterministic and repeatable, which makes it the right tool for a
    controlled experiment: every agent under test meets the same gust at the
    same moment, so the only variable is how they respond to it.
    """

    name = "gust_burst"

    def __init__(
        self,
        velocity_mps: Sequence[float],
        start: float,
        duration: float = 2.0,
        rise: float = 0.4,
    ):
        self._v = np.asarray(velocity_mps, dtype=float)
        self._start = float(start)
        self._duration = float(duration)
        self._rise = max(float(rise), 1e-6)

    def _envelope(self, t: float) -> float:
        dt = t - self._start
        if dt < 0.0 or dt > self._duration:
            return 0.0
        ramp_up = min(dt / self._rise, 1.0)
        ramp_down = min((self._duration - dt) / self._rise, 1.0)
        return float(min(ramp_up, ramp_down))

    def velocity(self, position: np.ndarray, t: float) -> np.ndarray:
        return self._v * self._envelope(t)

    def describe(self) -> dict:
        return {
            "type": self.name,
            "velocity": self._v.tolist(),
            "start": self._start,
            "duration": self._duration,
        }


class WindTunnel(WindField):
    """Wind confined to a slab of space, e.g. a draught across a doorway.

    Outside the slab the air is still.  Useful for testing whether an agent
    notices a hazard that only exists in part of the map.
    """

    name = "wind_tunnel"

    def __init__(
        self,
        velocity_mps: Sequence[float],
        lower: Sequence[float],
        upper: Sequence[float],
        softness: float = 0.25,
    ):
        self._v = np.asarray(velocity_mps, dtype=float)
        self._lo = np.asarray(lower, dtype=float)
        self._hi = np.asarray(upper, dtype=float)
        self._soft = max(float(softness), 1e-6)

    def velocity(self, position: np.ndarray, t: float) -> np.ndarray:
        # Smooth fade at the edges so the drone is not hit by a step function.
        inside = 1.0
        for axis in range(3):
            lo, hi, p = self._lo[axis], self._hi[axis], position[axis]
            edge = min(p - lo, hi - p)
            inside *= float(np.clip(edge / self._soft, 0.0, 1.0))
        return self._v * inside

    def describe(self) -> dict:
        return {
            "type": self.name,
            "velocity": self._v.tolist(),
            "lower": self._lo.tolist(),
            "upper": self._hi.tolist(),
        }


class Transient(WindField):
    """Switch any other field on for a while, then off again.

    GustBurst is a gust everywhere at once; WindTunnel is a draught in one part
    of the room that never stops.  Real hazards are usually both at once -- a
    door opens, air pours through that doorway for a few seconds, and it stops.
    Wrapping a spatial field in a time envelope gives that without a new field
    type for every combination:

        Transient(WindTunnel(...), start=10.0, duration=4.0)

    The envelope is deterministic, so every agent under test meets the same gust
    in the same place at the same moment.
    """

    name = "transient"

    def __init__(
        self,
        field: WindField,
        start: float,
        duration: float = 3.0,
        rise: float = 0.5,
    ):
        self._field = field
        self._start = float(start)
        self._duration = float(duration)
        self._rise = max(float(rise), 1e-6)

    def step(self, dt: float) -> None:
        self._field.step(dt)

    def _envelope(self, t: float) -> float:
        elapsed = t - self._start
        if elapsed < 0.0 or elapsed > self._duration:
            return 0.0
        return float(
            min(elapsed / self._rise, (self._duration - elapsed) / self._rise, 1.0)
        )

    def velocity(self, position: np.ndarray, t: float) -> np.ndarray:
        scale = self._envelope(t)
        if scale <= 0.0:
            return np.zeros(3)
        return self._field.velocity(position, t) * scale

    def describe(self) -> dict:
        return {
            "type": self.name,
            "start": self._start,
            "duration": self._duration,
            "field": self._field.describe(),
        }


class SumOfWinds(WindField):
    """Several fields acting at once, e.g. a steady breeze plus turbulence."""

    name = "sum"

    def __init__(self, fields: Sequence[WindField]):
        self._fields = list(fields)

    def step(self, dt: float) -> None:
        for f in self._fields:
            f.step(dt)

    def velocity(self, position: np.ndarray, t: float) -> np.ndarray:
        total = np.zeros(3)
        for f in self._fields:
            total = total + f.velocity(position, t)
        return total

    def describe(self) -> dict:
        return {"type": self.name, "fields": [f.describe() for f in self._fields]}


def from_config(spec: dict | None, rng: np.random.Generator | None = None) -> WindField:
    """Build a wind field from a plain dict, so scenarios can live in YAML."""
    if not spec:
        return NoWind()

    kind = spec.get("type", "none")
    if kind in ("none", None):
        return NoWind()
    if kind == "constant":
        return ConstantWind(spec["velocity"])
    if kind == "boundary_layer":
        return BoundaryLayerWind(
            speed_at_reference=spec["speed_at_reference"],
            direction=spec.get("direction", (1.0, 0.0, 0.0)),
            reference_height=spec.get("reference_height", 2.0),
            roughness=spec.get("roughness", 0.05),
        )
    if kind == "turbulent":
        return TurbulentWind(
            mean=spec.get("mean", (0.0, 0.0, 0.0)),
            sigma=spec.get("sigma", 0.8),
            theta=spec.get("theta", 0.7),
            rng=rng,
        )
    if kind == "gust_burst":
        return GustBurst(
            velocity_mps=spec["velocity"],
            start=spec["start"],
            duration=spec.get("duration", 2.0),
            rise=spec.get("rise", 0.4),
        )
    if kind == "wind_tunnel":
        return WindTunnel(
            spec["velocity"],
            spec["lower"],
            spec["upper"],
            spec.get("softness", 0.25),
        )
    if kind == "transient":
        return Transient(
            from_config(spec["field"], rng),
            start=spec["start"],
            duration=spec.get("duration", 3.0),
            rise=spec.get("rise", 0.5),
        )
    if kind == "sum":
        return SumOfWinds([from_config(f, rng) for f in spec["fields"]])

    raise ValueError(f"Unknown wind type: {kind!r}")
