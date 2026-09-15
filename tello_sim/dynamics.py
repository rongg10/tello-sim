"""Rigid-body motion for one drone.

Why a point mass and not four spinning rotors: the Tello SDK never exposes
motor commands.  You say "forward 30" and the drone's own flight controller
does the rest.  Simulating rotor aerodynamics would add parameters nobody can
measure through the SDK, and would change nothing you can observe.  What does
matter is how the airframe responds to a force it cannot cancel -- wind -- and
that is a mass, a drag coefficient and a limit on control authority.

The one equation that carries the model:

    m * dv/dt = F_control - c * (v - v_wind)

Drag acts on velocity *relative to the air*, so wind enters as a force that
pushes the drone downwind whenever it is not already moving with the air.  The
flight controller fights back, but only up to mass * max_accel.  Beyond that
the drone loses the argument and drifts, which is exactly what a real Tello
does in a corridor draught.
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

    # Attitude is a consequence of acceleration for a quadrotor, not an input.
    pitch: float = 0.0
    roll: float = 0.0

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
            battery=self.battery,
            airborne=self.airborne,
            motors_on=self.motors_on,
            flight_time=self.flight_time,
            last_accel_cmd=self.last_accel_cmd.copy(),
            wind_seen=self.wind_seen.copy(),
        )


def integrate(
    state: DroneState,
    spec,
    accel_cmd: np.ndarray,
    yaw_rate_cmd: float,
    wind: np.ndarray,
    dt: float,
    resting: bool = False,
) -> None:
    """Advance the state by one physics step, in place.

    accel_cmd is the acceleration the flight controller is asking for, in world
    axes, already saturated by the caller.  When the motors are off it is
    ignored and gravity takes over.

    `resting` means the airframe is sitting on the ground with the motors off.
    Friction against the floor is far stronger than the drag a draught can
    exert on 80 grams, so it is treated as absolute: the drone stays put.
    Leaving it out lets a landed drone creep downwind a fraction of a
    millimetre per tick, which over a few minutes silently walks it across the
    room and quietly ruins the takeoff reference.
    """
    wind = np.asarray(wind, dtype=float)
    state.wind_seen = wind

    if resting:
        state.velocity[:] = 0.0
        state.last_accel_cmd = np.zeros(3)
        state.yaw_rate = 0.0
        state.pitch = 0.0
        state.roll = 0.0
        return

    relative_velocity = state.velocity - wind
    drag_accel = -(spec.drag_coeff / spec.mass) * relative_velocity

    if state.motors_on:
        total_accel = np.asarray(accel_cmd, dtype=float) + drag_accel
    else:
        # Motors cut: only gravity and the air act on the airframe.
        total_accel = drag_accel - np.array([0.0, 0.0, GRAVITY])
        yaw_rate_cmd = 0.0

    state.last_accel_cmd = np.asarray(accel_cmd, dtype=float).copy()

    # Semi-implicit Euler: update velocity first, then use it for position.
    # Cheap, stable at 100 Hz, and conserves energy far better than plain Euler.
    state.velocity = state.velocity + total_accel * dt
    state.position = state.position + state.velocity * dt

    state.yaw_rate = float(yaw_rate_cmd)
    state.yaw = wrap_angle(state.yaw + state.yaw_rate * dt)

    # A quadrotor tilts in order to accelerate sideways; report that tilt so
    # the telemetry looks like the real thing.
    accel_body = world_to_body(state.last_accel_cmd, state.yaw)
    state.pitch = float(np.arctan2(accel_body[0], GRAVITY))
    state.roll = float(np.arctan2(-accel_body[1], GRAVITY))

    if state.airborne:
        state.flight_time += dt


def attitude_basis(state: DroneState) -> np.ndarray:
    """The drone's body axes in world coordinates, as rows [forward, left, up].

    Derived from physics rather than composed from Euler angles.  A quadrotor
    cannot choose its attitude freely: the rotors only push along one axis, so
    the airframe must tilt until that axis lines up with the total acceleration
    it needs -- thrust plus the gravity it is holding against.  The heading then
    fixes the one remaining degree of freedom.

    Stacking three Euler rotations would give the same answer only if the order
    and every sign happened to be right, and would be quietly wrong otherwise.
    """
    up = np.asarray(state.last_accel_cmd, dtype=float) + np.array([0.0, 0.0, GRAVITY])
    norm = float(np.linalg.norm(up))
    up = up / norm if norm > 1e-6 else np.array([0.0, 0.0, 1.0])

    heading = np.array([np.cos(state.yaw), np.sin(state.yaw), 0.0])
    left = np.cross(up, heading)
    norm = float(np.linalg.norm(left))
    if norm < 1e-6:                       # pointing straight up: any heading will do
        left = np.array([0.0, 1.0, 0.0])
    else:
        left = left / norm
    forward = np.cross(left, up)
    return np.array([forward, left, up])


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
