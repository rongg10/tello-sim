"""What the drone is currently trying to do.

Every SDK command becomes an Action: a small state machine that is asked, each
physics tick, "what acceleration do you want, and are you finished yet?".  The
network layer holds the caller until the action reports done, then replies
`ok` -- exactly the blocking behaviour of the real drone, including the way a
move that cannot be completed eventually times out and errors.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dynamics import DroneState, body_to_world, wrap_angle

# What an action returns each tick.
#   accel:    desired acceleration in world axes (m/s^2)
#   yaw_rate: desired yaw rate (rad/s)
#   done:     True when the command has been carried out
#   error:    a string if the command failed, else None
Outcome = tuple[np.ndarray, float, bool, "str | None"]


def _saturate_speed(v: np.ndarray, limit: float) -> np.ndarray:
    """Cap horizontal speed as a vector, vertical speed on its own axis."""
    out = np.array(v, dtype=float)
    horizontal = out[:2]
    speed = float(np.linalg.norm(horizontal))
    if speed > limit and speed > 1e-9:
        out[:2] = horizontal * (limit / speed)
    out[2] = float(np.clip(out[2], -limit, limit))
    return out


def position_controller(
    state: DroneState, spec, target: np.ndarray, speed_limit: float
) -> np.ndarray:
    """Cascade loop: position error -> desired velocity -> acceleration.

    Deliberately proportional only.  An integral term would erase the
    steady-state offset that constant wind produces, and that offset is one of
    the things worth measuring.
    """
    error = np.asarray(target, dtype=float) - state.position
    desired_velocity = _saturate_speed(spec.kp_pos * error, speed_limit)

    accel = spec.kv_vel * (desired_velocity - state.velocity)

    # Saturate authority.  This is the limit the wind competes against.
    horizontal = accel[:2]
    magnitude = float(np.linalg.norm(horizontal))
    if magnitude > spec.max_accel_xy and magnitude > 1e-9:
        accel[:2] = horizontal * (spec.max_accel_xy / magnitude)
    accel[2] = float(np.clip(accel[2], -spec.max_accel_z, spec.max_accel_z))
    return accel


def yaw_controller(state: DroneState, spec, target_yaw: float) -> float:
    error = wrap_angle(target_yaw - state.yaw)
    return float(np.clip(spec.kp_yaw * error, -spec.max_yaw_rate, spec.max_yaw_rate))


class StallDetector:
    """Notices when the drone has stopped getting any closer to its target.

    This is how a real Tello decides a move is over.  It has no idea where it
    truly is -- it integrates optical flow and stops when it believes it has
    gone far enough.  Push it with a steady wind and it settles a few
    centimetres downwind, reports `ok`, and carries on none the wiser.

    Judging completion on true position instead would make the simulator fail
    commands that the real drone completes happily, which is the wrong kind of
    wrong: flight code would be tuned against a drone stricter than the one it
    will actually fly.
    """

    def __init__(self, patience: float = 1.0, improvement: float = 0.005):
        self.patience = patience
        self.improvement = improvement
        self.best = float("inf")
        self.since_improvement = 0.0

    def update(self, error: float, dt: float) -> None:
        if error < self.best - self.improvement:
            self.best = error
            self.since_improvement = 0.0
        else:
            self.best = min(self.best, error)
            self.since_improvement += dt

    @property
    def stalled(self) -> bool:
        return self.since_improvement >= self.patience


class Action:
    """Base class."""

    name = "action"
    blocking = True     # does the caller wait for `ok`?

    def start(self, state: DroneState, spec) -> None:
        self.elapsed = 0.0

    def update(self, state: DroneState, spec, dt: float) -> Outcome:
        raise NotImplementedError

    def describe(self) -> dict:
        return {"action": self.name}


class Hover(Action):
    """Hold the position the drone was left at. The default between commands."""

    name = "hover"
    blocking = False

    def __init__(self, target: np.ndarray | None = None, target_yaw: float | None = None):
        self.target = None if target is None else np.asarray(target, dtype=float)
        self.target_yaw = target_yaw

    def start(self, state: DroneState, spec) -> None:
        self.elapsed = 0.0
        if self.target is None:
            self.target = state.position.copy()
        if self.target_yaw is None:
            self.target_yaw = state.yaw

    def update(self, state: DroneState, spec, dt: float) -> Outcome:
        self.elapsed += dt
        # Holding station uses the drone's full authority. The SDK `speed`
        # setting throttles commanded moves, not the stabiliser fighting wind.
        accel = position_controller(state, spec, self.target, spec.max_sdk_speed)
        return accel, yaw_controller(state, spec, self.target_yaw), False, None


class Grounded(Action):
    """Sitting on the floor with the motors off. Nothing to control."""

    name = "grounded"
    blocking = False

    def update(self, state: DroneState, spec, dt: float) -> Outcome:
        return np.zeros(3), 0.0, False, None


class TakeOff(Action):
    """Spin up, then climb to the height a Tello settles at (about 0.8 m)."""

    name = "takeoff"

    SPIN_UP = 1.0  # s of motor spin-up before the drone actually leaves the ground

    def start(self, state: DroneState, spec) -> None:
        self.elapsed = 0.0
        self.start_z = float(state.position[2])
        self.target_xy = state.position[:2].copy()
        self.target_yaw = state.yaw
        state.motors_on = True

    def update(self, state: DroneState, spec, dt: float) -> Outcome:
        self.elapsed += dt
        # Reserve time at the end for the drone to actually reach the height
        # the ramp is asking for, instead of reporting `ok` while still climbing.
        climb_time = max(spec.takeoff_duration - self.SPIN_UP - spec.settle_time, 0.1)
        progress = np.clip((self.elapsed - self.SPIN_UP) / climb_time, 0.0, 1.0)

        if self.elapsed >= self.SPIN_UP:
            state.airborne = True

        target_z = self.start_z + progress * (spec.takeoff_height - self.start_z)
        target = np.array([self.target_xy[0], self.target_xy[1], target_z])
        accel = position_controller(state, spec, target, spec.max_sdk_speed)
        yaw_rate = yaw_controller(state, spec, self.target_yaw)

        settled = abs(state.position[2] - spec.takeoff_height) < spec.position_tolerance
        done = self.elapsed >= spec.takeoff_duration and settled
        # Never hang: a drone pinned under a low ceiling still has to answer.
        done = done or self.elapsed > spec.takeoff_duration * 2.0
        return accel, yaw_rate, done, None


class Land(Action):
    """Descend to the floor and cut the motors."""

    name = "land"

    def start(self, state: DroneState, spec) -> None:
        self.elapsed = 0.0
        self.start_z = float(state.position[2])
        self.target_xy = state.position[:2].copy()
        self.target_yaw = state.yaw

    def update(self, state: DroneState, spec, dt: float) -> Outcome:
        self.elapsed += dt
        descent_time = max(spec.land_duration - spec.settle_time, 0.1)
        progress = np.clip(self.elapsed / descent_time, 0.0, 1.0)
        floor = spec.ground_clearance
        target_z = floor + (self.start_z - floor) * (1.0 - progress)
        target = np.array([self.target_xy[0], self.target_xy[1], target_z])
        accel = position_controller(state, spec, target, spec.max_sdk_speed)
        yaw_rate = yaw_controller(state, spec, self.target_yaw)

        touched = state.position[2] - floor < spec.position_tolerance
        # Gate on the calibrated duration too, so a landing reported by the
        # simulator takes as long as one measured on the real drone.
        done = (
            progress >= 1.0 and touched and self.elapsed >= spec.land_duration
        ) or self.elapsed > spec.land_duration * 2.5
        if done:
            state.airborne = False
            state.motors_on = False
            state.velocity[:] = 0.0
            state.position[2] = floor
        return accel, yaw_rate, done, None


class Emergency(Action):
    """Motors off, right now. The drone falls."""

    name = "emergency"
    blocking = False

    def start(self, state: DroneState, spec) -> None:
        self.elapsed = 0.0
        state.motors_on = False
        state.airborne = False

    def update(self, state: DroneState, spec, dt: float) -> Outcome:
        self.elapsed += dt
        return np.zeros(3), 0.0, False, None


class MoveTo(Action):
    """Fly to a point and stop there.

    Covers `forward`, `back`, `left`, `right`, `up`, `down` and `go`, which
    differ only in how the target is worked out before the action is created.

    The drone does not chase the destination directly.  It follows a reference
    point that slides from the start to the target at the commanded speed, and
    holds station on that moving point with its full authority.  The difference
    matters in wind: chasing the destination means the speed limit applies to
    the whole correction, so nearly all of it goes into travelling forward and
    almost none is left to resist being pushed sideways -- a drone that sails
    away downwind on every long leg.  Tracking a moving reference keeps the
    along-track speed at what was asked for while cross-track error is corrected
    at full strength, which is what a real flight controller does.
    """

    name = "move_to"

    def __init__(self, target: np.ndarray, speed: float, label: str = "move"):
        self.target = np.asarray(target, dtype=float)
        self.speed = float(speed)
        self.label = label

    def start(self, state: DroneState, spec) -> None:
        self.elapsed = 0.0
        self.target_yaw = state.yaw
        self.origin = state.position.copy()

        delta = self.target - self.origin
        self.distance = float(np.linalg.norm(delta))
        self.direction = delta / self.distance if self.distance > 1e-9 else np.zeros(3)

        self.travel_time = self.distance / max(self.speed, 1e-3)
        self.nominal = self.travel_time + spec.command_overhead
        self.timeout = spec.move_timeout_factor * self.nominal + 2.0
        self.stall = StallDetector()

    def reference(self) -> np.ndarray:
        travelled = min(self.elapsed * self.speed, self.distance)
        return self.origin + self.direction * travelled

    def update(self, state: DroneState, spec, dt: float) -> Outcome:
        self.elapsed += dt
        accel = position_controller(state, spec, self.reference(), spec.max_sdk_speed)
        yaw_rate = yaw_controller(state, spec, self.target_yaw)

        error = float(np.linalg.norm(self.target - state.position))
        speed = float(np.linalg.norm(state.velocity))
        self.stall.update(error, dt)

        if self.elapsed >= self.travel_time:
            if error < spec.position_tolerance and speed < 0.10:
                return accel, yaw_rate, True, None
            # Stopped making progress and no longer moving: as far as the drone
            # is concerned it has arrived, even if the wind says otherwise. This
            # is how the real one decides, having no idea where it truly is.
            if self.stall.stalled and speed < 0.08:
                return accel, yaw_rate, True, None

        if self.elapsed > self.timeout:
            # Still being carried somewhere it did not ask to go.
            return accel, yaw_rate, True, "error Motor stop"
        return accel, yaw_rate, False, None

    def describe(self) -> dict:
        return {"action": self.name, "label": self.label, "target": self.target.tolist()}


class RotateTo(Action):
    """Turn on the spot to a heading."""

    name = "rotate_to"

    def __init__(self, target_yaw: float):
        self.target_yaw = float(target_yaw)

    def start(self, state: DroneState, spec) -> None:
        self.elapsed = 0.0
        self.hold = state.position.copy()
        sweep = abs(wrap_angle(self.target_yaw - state.yaw))
        self.timeout = spec.move_timeout_factor * (sweep / spec.max_yaw_rate) + 2.0
        self.stall = StallDetector()

    def update(self, state: DroneState, spec, dt: float) -> Outcome:
        self.elapsed += dt
        accel = position_controller(state, spec, self.hold, spec.max_sdk_speed)
        yaw_rate = yaw_controller(state, spec, self.target_yaw)

        error = abs(wrap_angle(self.target_yaw - state.yaw))
        self.stall.update(error, dt)

        if error < spec.yaw_tolerance:
            return accel, 0.0, True, None
        if self.stall.stalled and abs(state.yaw_rate) < 0.05:
            return accel, yaw_rate, True, None
        if self.elapsed > self.timeout:
            return accel, yaw_rate, True, "error"
        return accel, yaw_rate, False, None


class CurveThrough(Action):
    """Arc through a waypoint to an end point, as `curve x1 y1 z1 x2 y2 z2 s`.

    A quadratic Bezier is used rather than the exact circular arc the firmware
    flies: it passes near the waypoint, ends exactly on target, and never
    produces the "out of range" surprises a circle fit does.
    """

    name = "curve"

    def __init__(self, start: np.ndarray, via: np.ndarray, end: np.ndarray, speed: float):
        self.p0 = np.asarray(start, dtype=float)
        self.p1 = np.asarray(via, dtype=float)
        self.p2 = np.asarray(end, dtype=float)
        self.speed = float(speed)

    def _point(self, s: float) -> np.ndarray:
        # Control point chosen so the curve actually passes through `via` at s=0.5.
        control = 2.0 * self.p1 - 0.5 * (self.p0 + self.p2)
        return (1 - s) ** 2 * self.p0 + 2 * (1 - s) * s * control + s**2 * self.p2

    def start(self, state: DroneState, spec) -> None:
        self.elapsed = 0.0
        self.target_yaw = state.yaw
        samples = [self._point(i / 40.0) for i in range(41)]
        self.length = float(sum(np.linalg.norm(b - a) for a, b in zip(samples, samples[1:])))
        self.duration = self.length / max(self.speed, 1e-3)
        self.timeout = spec.move_timeout_factor * self.duration + 3.0
        self.stall = StallDetector()

    def update(self, state: DroneState, spec, dt: float) -> Outcome:
        self.elapsed += dt
        s = float(np.clip(self.elapsed / max(self.duration, 1e-3), 0.0, 1.0))
        # Chase a point that slides along the curve: the drone lags slightly,
        # which is what following a path at speed actually looks like.
        target = self._point(min(s + 0.05, 1.0))
        accel = position_controller(state, spec, target, spec.max_sdk_speed)
        yaw_rate = yaw_controller(state, spec, self.target_yaw)

        error = float(np.linalg.norm(self.p2 - state.position))
        self.stall.update(error, dt)

        if s >= 1.0 and error < 0.15:
            return accel, yaw_rate, True, None
        if s >= 1.0 and self.stall.stalled and float(np.linalg.norm(state.velocity)) < 0.08:
            return accel, yaw_rate, True, None
        if self.elapsed > self.timeout:
            return accel, yaw_rate, True, "error"
        return accel, yaw_rate, False, None


class Flip(Action):
    """A barrel roll. Costs altitude and needs a healthy battery, as on the real drone."""

    name = "flip"

    DIRECTIONS = {
        "f": np.array([1.0, 0.0, 0.0]),
        "b": np.array([-1.0, 0.0, 0.0]),
        "l": np.array([0.0, 1.0, 0.0]),
        "r": np.array([0.0, -1.0, 0.0]),
    }

    def __init__(self, direction: str):
        self.direction = direction

    def start(self, state: DroneState, spec) -> None:
        self.elapsed = 0.0
        self.start_pos = state.position.copy()
        self.target_yaw = state.yaw
        offset_body = self.DIRECTIONS[self.direction] * 0.30
        self.offset_world = body_to_world(offset_body, state.yaw)

    def update(self, state: DroneState, spec, dt: float) -> Outcome:
        self.elapsed += dt
        s = float(np.clip(self.elapsed / max(spec.flip_duration, 0.1), 0.0, 1.0))
        # Out and back, dipping in the middle: a flip loses height.
        swing = np.sin(np.pi * s)
        target = self.start_pos + self.offset_world * swing
        target[2] -= 0.15 * swing
        accel = position_controller(state, spec, target, spec.max_sdk_speed)
        yaw_rate = yaw_controller(state, spec, self.target_yaw)
        return accel, yaw_rate, s >= 1.0, None


class RCVelocity(Action):
    """Continuous stick input, as `rc a b c d`.

    Unlike every other command this one never finishes: it stays in force until
    replaced.  The real drone behaves the same way, and stops if the stream of
    rc packets dries up -- which the watchdog in drone.py reproduces.
    """

    name = "rc"
    blocking = False

    def __init__(self, left_right: float, forward_back: float, up_down: float, yaw: float):
        # Each channel is -100..100 in the SDK.
        self.channels = np.array(
            [float(left_right), float(forward_back), float(up_down), float(yaw)]
        )

    def set_channels(self, left_right: float, forward_back: float, up_down: float, yaw: float) -> None:
        self.channels = np.array(
            [float(left_right), float(forward_back), float(up_down), float(yaw)]
        )

    def start(self, state: DroneState, spec) -> None:
        self.elapsed = 0.0

    def update(self, state: DroneState, spec, dt: float) -> Outcome:
        self.elapsed += dt
        lr, fb, ud, yw = self.channels / 100.0

        # Stick axes are body-relative: forward is wherever the nose points.
        desired_body = np.array([fb, -lr, ud]) * spec.max_rc_speed
        desired_world = body_to_world(desired_body, state.yaw)

        accel = spec.kv_vel * (desired_world - state.velocity)
        horizontal = accel[:2]
        magnitude = float(np.linalg.norm(horizontal))
        if magnitude > spec.max_accel_xy and magnitude > 1e-9:
            accel[:2] = horizontal * (spec.max_accel_xy / magnitude)
        accel[2] = float(np.clip(accel[2], -spec.max_accel_z, spec.max_accel_z))

        return accel, yw * spec.max_rc_yaw_rate, False, None
