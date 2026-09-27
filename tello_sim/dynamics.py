"""Drone state, and the small pieces of maths everything else shares.

This used to be the physics.  It integrated a point mass by hand and resolved
collisions by pushing a sphere out of whatever it overlapped.  MuJoCo does both
of those now, in `physics.py`, so what is left here is the state container, the
frame conversions the command layer needs, and the battery model.

The battery stayed because it is not rigid-body physics.  No engine models the
cost of holding a tilt against a crosswind, and that cost is one of the two
things this simulator exists to study.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

GRAVITY = 9.81  # m/s^2


def yaw_matrix(yaw: float) -> np.ndarray:
    """Rotation from body axes to world axes about the vertical."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def body_to_world(vector_body: np.ndarray, yaw: float) -> np.ndarray:
    return yaw_matrix(yaw) @ np.asarray(vector_body, dtype=float)


def world_to_body(vector_world: np.ndarray, yaw: float) -> np.ndarray:
    return yaw_matrix(yaw).T @ np.asarray(vector_world, dtype=float)


def wrap_angle(a: float) -> float:
    """Fold an angle into (-pi, pi] so yaw errors take the short way round."""
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass
class DroneState:
    """Everything that changes as the drone flies."""

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw: float = 0.0
    yaw_rate: float = 0.0

    # Attitude used to be back-computed from the commanded acceleration, because
    # a point mass has no attitude to report. It is now a real state that the
    # engine integrates, so a drone knocked sideways by a door stays knocked
    # sideways until it flies itself level again.
    pitch: float = 0.0
    roll: float = 0.0
    orientation: np.ndarray = field(default_factory=lambda: np.eye(3))

    battery: float = 100.0
    airborne: bool = False
    motors_on: bool = False

    # Bookkeeping the state packet reports back to the pilot.
    flight_time: float = 0.0
    last_accel_cmd: np.ndarray = field(default_factory=lambda: np.zeros(3))
    wind_seen: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def copy(self) -> "DroneState":
        return DroneState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            yaw=self.yaw,
            yaw_rate=self.yaw_rate,
            pitch=self.pitch,
            roll=self.roll,
            orientation=self.orientation.copy(),
            battery=self.battery,
            airborne=self.airborne,
            motors_on=self.motors_on,
            flight_time=self.flight_time,
            last_accel_cmd=self.last_accel_cmd.copy(),
            wind_seen=self.wind_seen.copy(),
        )


def attitude_basis(state: DroneState) -> np.ndarray:
    """The drone's body axes in world coordinates, as rows [forward, left, up].

    Read straight off the airframe's real orientation.  The previous version had
    to infer this from the acceleration the controller had asked for, because
    the point-mass model carried no attitude of its own; it therefore drew a
    drone that was always perfectly poised, even mid-crash.
    """
    return np.asarray(state.orientation, dtype=float).T.copy()


def drain_battery(state: DroneState, spec, dt: float) -> None:
    """Burn charge at a rate that depends on what the drone is actually doing.

    Hovering is not free, and fighting wind is more expensive than cruising in
    still air, because the controller has to hold a permanent tilt.  That makes
    energy a real constraint an agent can trade against -- which is the point,
    since no current AVLN benchmark constrains it at all.
    """
    if not state.motors_on:
        rate = spec.drain_idle
    else:
        effort = float(np.linalg.norm(state.last_accel_cmd))
        airspeed = float(np.linalg.norm(state.velocity - state.wind_seen))
        rate = (
            spec.drain_hover
            + spec.drain_per_accel * effort
            + spec.drain_per_speed * airspeed
        )
    state.battery = max(0.0, state.battery - rate * dt)
