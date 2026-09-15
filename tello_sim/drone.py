"""One simulated Tello: parses SDK commands, flies, and reports telemetry.

This is the part that has to be honest.  If the real drone rejects a 10 cm move
because the minimum is 20, so does this.  If the real drone answers `ok` only
once the move has finished, so does this.  Anything the simulator is more
permissive about is a bug that will only show up on the day you fly for real.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import actions as act
from .config import DroneSpec
from .dynamics import DroneState, body_to_world, drain_battery, integrate, world_to_body

CM = 0.01
OK = "ok"

# Commands that jump the queue: they are how a pilot regains control.
PREEMPTIVE = {"emergency", "stop", "rc"}

# The firmware sends nothing back for these, and djitellopy does not read a
# reply.  Answering anyway would leave a stray "ok" in the client's response
# buffer, which the *next* command would then consume -- every reply after that
# point belongs to the previous command.  A nasty, intermittent bug to inherit.
NO_REPLY = {"rc", "emergency"}

# Commands that need the drone to already be in the air.
NEEDS_FLIGHT = {
    "land", "up", "down", "left", "right", "forward", "back",
    "cw", "ccw", "go", "curve", "flip", "jump",
}


@dataclass
class PendingCommand:
    """A command in flight between the network thread and the simulation."""

    raw: str
    verb: str
    args: list
    submitted_at: float
    expects_response: bool = True
    event: threading.Event = field(default_factory=threading.Event)
    response: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    on_complete: Optional[callable] = None

    def complete(self, response: str, sim_time: float) -> None:
        self.response = response
        self.finished_at = sim_time
        self.event.set()
        if self.on_complete is not None:
            self.on_complete(self)

    def wait(self, timeout: float = 20.0) -> Optional[str]:
        return self.response if self.event.wait(timeout) else None


class CommandError(ValueError):
    """A command the real drone would reject outright."""

    def __init__(self, message: str = "error"):
        super().__init__(message)
        self.message = message


class SimDrone:
    """A single simulated aircraft."""

    def __init__(
        self,
        name: str = "tello-1",
        spec: DroneSpec | None = None,
        start_position=(0.0, 0.0, 0.0),
        start_yaw_deg: float = 0.0,
        serial: str | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.name = name
        self.spec = spec or DroneSpec()
        self.serial = serial or f"SIM{abs(hash(name)) % 10**9:09d}"
        self.rng = rng if rng is not None else np.random.default_rng(0)

        self.state = DroneState(
            position=np.asarray(start_position, dtype=float),
            yaw=float(np.deg2rad(start_yaw_deg)),
            battery=self.spec.battery_start,
        )
        self.home = self.state.position.copy()

        self.sdk_mode = False
        self.mission_pads_enabled = False
        self.mission_pad_direction = 0
        self.speed_setting = self.spec.default_speed
        self.detected_pad = None

        self._action: act.Action = act.Grounded()
        self._action.start(self.state, self.spec)
        self._current: Optional[PendingCommand] = None

        self._queue: deque[PendingCommand] = deque()
        self._immediate: deque[PendingCommand] = deque()
        self._lock = threading.Lock()

        self._rc_action: Optional[act.RCVelocity] = None
        self._last_rc_time = -1e9
        self.rc_watchdog = 2.0     # s without an rc packet before the drone hovers

        self.collisions: list[tuple[float, str]] = []
        self.events: list[tuple[float, str]] = []
        self.sim_time = 0.0

        # Set by the Simulator so every completed command reaches the recorder.
        self.on_command_finished = None

        # Which surfaces are being touched right now, so a drone resting
        # against a wall logs one collision rather than one per tick.
        self._active_hits: set[str] = set()

    # ------------------------------------------------------------------
    # Command entry point (called from network or script threads)
    # ------------------------------------------------------------------

    def submit(self, raw: str) -> PendingCommand:
        """Accept a raw SDK command string. Returns a handle to wait on.

        Validation and instant replies happen here, on the caller's thread;
        anything that takes flight time is queued for the simulation thread.
        """
        text = raw.strip()
        parts = text.split()
        verb = parts[0].lower() if parts else ""
        args = parts[1:]

        pending = PendingCommand(
            raw=text,
            verb=verb,
            args=args,
            submitted_at=self.sim_time,
            expects_response=verb not in NO_REPLY,
            on_complete=self.on_command_finished,
        )

        try:
            immediate = self._handle_immediately(verb, args, pending)
        except CommandError as exc:
            pending.complete(exc.message, self.sim_time)
            return pending

        if immediate is not None:
            pending.complete(immediate, self.sim_time)
            return pending

        with self._lock:
            if verb in PREEMPTIVE:
                self._immediate.append(pending)
            else:
                self._queue.append(pending)
        return pending

    def _handle_immediately(self, verb: str, args: list, pending: PendingCommand):
        """Commands the real drone answers without flying anywhere.

        Returns the response string, or None if the command must be queued.
        Raises CommandError for anything out of range.
        """
        if verb == "command":
            self.sdk_mode = True
            return OK

        if not self.sdk_mode and verb not in ("command",):
            # The real drone ignores everything until it is put in SDK mode.
            raise CommandError("error Not in SDK mode")

        # ---- read commands ----
        if verb == "battery?":
            return str(int(round(self.state.battery)))
        if verb == "speed?":
            return f"{self.speed_setting / CM:.0f}"
        if verb == "time?":
            return str(int(self.state.flight_time))
        if verb == "sn?":
            return self.serial
        if verb == "sdk?":
            return "20"
        if verb == "wifi?":
            return "90"
        if verb == "height?":
            return f"{self._height_cm()}dm"
        if verb == "tof?":
            return f"{self._tof_cm()}mm"
        if verb == "baro?":
            return f"{self._baro_cm() / 100.0:.2f}m"
        if verb == "temp?":
            return f"{self.spec.temperature_low}~{self.spec.temperature_high}C"
        if verb == "attitude?":
            return (
                f"pitch:{int(np.rad2deg(self.state.pitch))};"
                f"roll:{int(np.rad2deg(self.state.roll))};"
                f"yaw:{int(np.rad2deg(self.state.yaw))};"
            )
        if verb == "acceleration?":
            a = self.state.last_accel_cmd
            return f"agx:{a[0]*100:.2f};agy:{a[1]*100:.2f};agz:{a[2]*100:.2f};"

        # ---- set commands ----
        if verb == "speed":
            value = self._require_int(args, 0, 10, 100, "speed")
            self.speed_setting = value * CM
            return OK
        if verb == "mon":
            self.mission_pads_enabled = True
            return OK
        if verb == "moff":
            self.mission_pads_enabled = False
            self.detected_pad = None
            return OK
        if verb == "mdirection":
            self.mission_pad_direction = self._require_int(args, 0, 0, 2, "mdirection")
            return OK
        if verb in ("wifi", "ap", "port", "setfps", "setbitrate", "setresolution"):
            return OK
        if verb in ("streamon", "streamoff"):
            # Accepted so existing code runs unchanged; no video is produced.
            self.events.append((self.sim_time, f"{verb} (no camera in this simulator)"))
            return OK

        # ---- validation for queued commands ----
        self._validate(verb, args)
        return None

    # ------------------------------------------------------------------
    # Validation, mirroring the real firmware's limits
    # ------------------------------------------------------------------

    @staticmethod
    def _require_int(args, index, low, high, label) -> int:
        try:
            value = int(float(args[index]))
        except (IndexError, ValueError):
            raise CommandError(f"error {label} needs a number")
        if not low <= value <= high:
            raise CommandError(f"error {label} out of range {low}..{high}")
        return value

    def _validate(self, verb: str, args: list) -> None:
        # Range first, so a badly formed command reports what is actually wrong
        # with it rather than complaining that the drone is on the ground.
        if verb in ("up", "down", "left", "right", "forward", "back"):
            self._require_int(args, 0, 20, 500, verb)
        elif verb in ("cw", "ccw"):
            self._require_int(args, 0, 1, 360, verb)
        elif verb == "go":
            for i in range(3):
                self._require_int(args, i, -500, 500, "go")
            self._require_int(args, 3, 10, 100, "go speed")
            offsets = [int(float(a)) for a in args[:3]]
            if all(abs(v) <= 20 for v in offsets):
                # The firmware refuses a move that is inside the dead zone on
                # every axis at once, because it cannot tell it from noise.
                raise CommandError("error go distance too small")
            if len(args) >= 5:
                self._require_pad(args[4])
        elif verb == "curve":
            for i in range(6):
                self._require_int(args, i, -500, 500, "curve")
            self._require_int(args, 6, 10, 60, "curve speed")
            if len(args) >= 8:
                self._require_pad(args[7])
        elif verb == "jump":
            for i in range(3):
                self._require_int(args, i, -500, 500, "jump")
            self._require_int(args, 3, 10, 100, "jump speed")
            self._require_int(args, 4, -360, 360, "jump yaw")
            if len(args) >= 7:
                self._require_pad(args[5])
                self._require_pad(args[6])
        elif verb == "flip":
            if not args or args[0].lower() not in act.Flip.DIRECTIONS:
                raise CommandError("error flip needs l, r, f or b")
            if self.state.battery < 50.0:
                # The real Tello refuses to flip on a tired battery.
                raise CommandError("error Battery too low to flip")
        elif verb == "rc":
            for i in range(4):
                self._require_int(args, i, -100, 100, "rc")
        elif verb == "takeoff":
            if self.state.airborne:
                raise CommandError("error Already flying")
            if self.state.battery < 10.0:
                raise CommandError("error Battery too low")
        elif verb in ("land", "stop", "emergency"):
            pass
        else:
            raise CommandError("error unknown command")

        if verb in NEEDS_FLIGHT and not self.state.airborne and verb != "land":
            raise CommandError("error Not flying")

    def _require_pad(self, token) -> int:
        if not self.mission_pads_enabled:
            raise CommandError("error Mission pads not enabled")
        try:
            pad = int(str(token).lstrip("m"))
        except ValueError:
            raise CommandError("error bad mission pad id")
        if not 1 <= pad <= 8:
            raise CommandError("error mission pad id must be 1..8")
        return pad

    # ------------------------------------------------------------------
    # Turning a validated command into an Action (simulation thread)
    # ------------------------------------------------------------------

    def _drift(self) -> np.ndarray:
        """Where the drone will actually stop, versus where it was told to.

        Optical-flow position holding is good but not exact, so every relative
        move lands a few centimetres off. The error is independent per command,
        which means it accumulates over a long sequence -- the reason dead
        reckoning on a real Tello goes wrong after a dozen moves.
        """
        sigma = self.spec.move_error_std
        return self.rng.normal(0.0, sigma, 3) if sigma > 0 else np.zeros(3)

    def _yaw_drift(self) -> float:
        sigma = np.deg2rad(self.spec.yaw_error_std_deg)
        return float(self.rng.normal(0.0, sigma)) if sigma > 0 else 0.0

    def _build_action(self, pending: PendingCommand, world) -> act.Action:
        verb, args = pending.verb, pending.args
        pos, yaw = self.state.position, self.state.yaw
        speed = self.speed_setting

        if verb == "takeoff":
            return act.TakeOff()
        if verb == "land":
            return act.Land()
        if verb == "emergency":
            return act.Emergency()
        if verb == "stop":
            return act.Hover(target=pos.copy(), target_yaw=yaw)

        if verb in ("up", "down", "left", "right", "forward", "back"):
            distance = int(float(args[0])) * CM
            body = {
                "forward": np.array([distance, 0.0, 0.0]),
                "back": np.array([-distance, 0.0, 0.0]),
                "left": np.array([0.0, distance, 0.0]),
                "right": np.array([0.0, -distance, 0.0]),
                "up": np.array([0.0, 0.0, distance]),
                "down": np.array([0.0, 0.0, -distance]),
            }[verb]
            target = pos + body_to_world(body, yaw) + self._drift()
            return act.MoveTo(target, speed, label=pending.raw)

        if verb == "cw":
            return act.RotateTo(yaw - np.deg2rad(float(args[0])) + self._yaw_drift())
        if verb == "ccw":
            return act.RotateTo(yaw + np.deg2rad(float(args[0])) + self._yaw_drift())

        if verb == "go":
            offset = np.array([float(a) for a in args[:3]]) * CM
            move_speed = float(args[3]) * CM
            if len(args) >= 5:
                origin, pad_yaw = self._pad_frame(args[4], world)
                target = origin + body_to_world(offset, pad_yaw)
            else:
                target = pos + body_to_world(offset, yaw)
            return act.MoveTo(target + self._drift(), move_speed, label=pending.raw)

        if verb == "curve":
            via_body = np.array([float(a) for a in args[:3]]) * CM
            end_body = np.array([float(a) for a in args[3:6]]) * CM
            move_speed = float(args[6]) * CM
            if len(args) >= 8:
                origin, pad_yaw = self._pad_frame(args[7], world)
                via = origin + body_to_world(via_body, pad_yaw)
                end = origin + body_to_world(end_body, pad_yaw)
            else:
                via = pos + body_to_world(via_body, yaw)
                end = pos + body_to_world(end_body, yaw)
            return act.CurveThrough(pos.copy(), via, end, move_speed)

        if verb == "jump":
            offset = np.array([float(a) for a in args[:3]]) * CM
            move_speed = float(args[3]) * CM
            origin, pad_yaw = self._pad_frame(args[5], world)
            target = origin + body_to_world(offset, pad_yaw)
            return act.MoveTo(target, move_speed, label=pending.raw)

        if verb == "flip":
            return act.Flip(args[0].lower())

        if verb == "rc":
            channels = [float(a) for a in args[:4]]
            if self._rc_action is None:
                self._rc_action = act.RCVelocity(*channels)
            else:
                self._rc_action.set_channels(*channels)
            return self._rc_action

        raise CommandError("error unknown command")

    def _pad_frame(self, token, world):
        pad_id = int(str(token).lstrip("m"))
        for pad in world.mission_pads:
            if pad.pad_id == pad_id:
                return np.array([pad.center[0], pad.center[1], 0.0]), pad.yaw
        raise CommandError("error mission pad not found")

    # ------------------------------------------------------------------
    # Simulation step (simulation thread only)
    # ------------------------------------------------------------------

    def step(self, dt: float, sim_time: float, world) -> None:
        self.sim_time = sim_time

        self._service_queues(world)

        # A stream of rc packets that stops means the pilot let go of the sticks.
        if (
            isinstance(self._action, act.RCVelocity)
            and sim_time - self._last_rc_time > self.rc_watchdog
        ):
            self._start_action(act.Hover())
            self._rc_action = None
            self.events.append((sim_time, "rc timeout, holding position"))

        accel, yaw_rate, done, error = self._action.update(self.state, self.spec, dt)

        if not self.state.motors_on:
            accel = np.zeros(3)
            yaw_rate = 0.0

        wind = world.wind_at(self.state.position, sim_time)
        resting = (
            not self.state.motors_on
            and self.state.position[2] <= self.spec.ground_clearance + 1e-3
        )
        integrate(self.state, self.spec, accel, yaw_rate, wind, dt, resting=resting)
        drain_battery(self.state, self.spec, dt)

        # Remember how hard it was travelling before the floor stopped it: a
        # touchdown at landing speed is not a collision, a fall is.
        impact_speed = float(np.linalg.norm(self.state.velocity))

        position, velocity, hits = world.resolve_collisions(
            self.state.position,
            self.state.velocity,
            self.spec.radius,
            self.spec.ground_clearance,
        )
        self.state.position, self.state.velocity = position, velocity

        current_hits = {
            hit for hit in hits
            if hit != "bound_z_min" or impact_speed > 0.5
        }
        for hit in current_hits - self._active_hits:
            self.collisions.append((sim_time, hit))
            self.events.append((sim_time, f"collision with {hit}"))
        self._active_hits = current_hits

        if self.mission_pads_enabled:
            self.detected_pad = world.mission_pad_under(self.state.position, self.spec)
        else:
            self.detected_pad = None

        self._check_battery(sim_time)

        if done and self._current is not None:
            self._current.complete(error or OK, sim_time)
            self._current = None
            if error:
                self.events.append((sim_time, f"command failed: {error}"))
            self._start_action(act.Hover() if self.state.airborne else act.Grounded())

    def _service_queues(self, world) -> None:
        """Start the next command, letting preemptive ones cut in."""
        with self._lock:
            immediate = self._immediate.popleft() if self._immediate else None

        if immediate is not None:
            self._begin(immediate, world, preempt=True)
            return

        if self._current is not None or self._action.blocking:
            return

        with self._lock:
            nxt = self._queue.popleft() if self._queue else None
        if nxt is not None:
            self._begin(nxt, world, preempt=False)

    def _begin(self, pending: PendingCommand, world, preempt: bool) -> None:
        try:
            action = self._build_action(pending, world)
        except CommandError as exc:
            pending.complete(exc.message, self.sim_time)
            return

        if preempt and self._current is not None:
            # The interrupted command reports the interruption, as the real
            # drone does when you slam the sticks mid-move.
            self._current.complete("error interrupted", self.sim_time)
            self._current = None

        pending.started_at = self.sim_time
        if pending.verb == "rc":
            self._last_rc_time = self.sim_time

        self._start_action(action)

        if action.blocking:
            self._current = pending
        else:
            pending.complete(OK, self.sim_time)

    def _start_action(self, action: act.Action) -> None:
        if action is not self._action:
            action.start(self.state, self.spec)
        self._action = action

    def _check_battery(self, sim_time: float) -> None:
        spec = self.spec
        if self.state.battery <= spec.battery_forced_landing and self.state.airborne:
            if not isinstance(self._action, act.Land):
                self.events.append((sim_time, "battery critical, forced landing"))
                if self._current is not None:
                    self._current.complete("error Battery too low", sim_time)
                    self._current = None
                self._start_action(act.Land())

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def _noise(self, sigma: float) -> float:
        return float(self.rng.normal(0.0, sigma)) if sigma > 0 else 0.0

    def _height_cm(self) -> int:
        h = self.state.position[2] - self.home[2] + self._noise(self.spec.noise_height)
        return int(round(max(h, 0.0) / CM))

    def _tof_cm(self) -> int:
        clearance = self.state.position[2] - self._ground_below
        value = max(clearance + self._noise(self.spec.noise_height), 0.0) / CM
        # The real sensor reports a floor of 10 cm and saturates around 10 m.
        return int(round(min(max(value, 10.0), 1000.0)))

    def _baro_cm(self) -> float:
        return 100.0 * (self.state.position[2] / CM) / 100.0 + 5000.0

    def state_packet(self, world=None) -> str:
        """Build the SDK 2.0 state string, byte-for-byte shaped like the real one.

        Field units follow the firmware, not intuition: velocities are in
        decimetres per second, heights in centimetres, accelerations in
        thousandths of a g.  Getting these wrong in a simulator is a classic way
        to write control code that only works in simulation.
        """
        s = self.state
        self._ground_below = world.ground_height_at(s.position) if world is not None else 0.0

        pad_id = self.detected_pad.pad_id if self.detected_pad else -1
        if self.detected_pad is not None:
            relative = s.position - np.array(
                [self.detected_pad.center[0], self.detected_pad.center[1], 0.0]
            )
            px, py, pz = (relative / CM).astype(int)
        else:
            px = py = pz = 0

        velocity_body = world_to_body(s.velocity, s.yaw)
        noise_v = self.spec.noise_velocity if self.spec.noise_velocity else 0.0
        vgx, vgy, vgz = [
            int(round((v + self._noise(noise_v)) / 0.1)) for v in velocity_body
        ]

        pitch = int(round(np.rad2deg(s.pitch + self._noise(self.spec.noise_attitude))))
        roll = int(round(np.rad2deg(s.roll + self._noise(self.spec.noise_attitude))))
        yaw = int(round(np.rad2deg(s.yaw + self._noise(self.spec.noise_attitude))))

        # Specific force in milli-g, body axes, sign convention as logged by a
        # real Tello sitting on a table: agz reads about -1000.
        accel_body = world_to_body(s.last_accel_cmd, s.yaw)
        agx = accel_body[0] / 9.81 * 1000.0
        agy = accel_body[1] / 9.81 * 1000.0
        agz = -(1000.0 + accel_body[2] / 9.81 * 1000.0)

        fields = [
            f"mid:{pad_id}",
            f"x:{px}", f"y:{py}", f"z:{pz}",
            "mpry:0,0,0",
            f"pitch:{pitch}", f"roll:{roll}", f"yaw:{yaw}",
            f"vgx:{vgx}", f"vgy:{vgy}", f"vgz:{vgz}",
            f"templ:{self.spec.temperature_low}", f"temph:{self.spec.temperature_high}",
            f"tof:{self._tof_cm()}", f"h:{self._height_cm()}",
            f"bat:{int(round(s.battery))}",
            f"baro:{self._baro_cm():.2f}",
            f"time:{int(s.flight_time)}",
            f"agx:{agx:.2f}", f"agy:{agy:.2f}", f"agz:{agz:.2f}",
        ]
        return ";".join(fields) + ";\r\n"

    _ground_below = 0.0

    def snapshot(self) -> dict:
        """A flat record of everything worth logging this tick."""
        s = self.state
        return {
            "drone": self.name,
            "t": round(self.sim_time, 4),
            "x": float(s.position[0]),
            "y": float(s.position[1]),
            "z": float(s.position[2]),
            "vx": float(s.velocity[0]),
            "vy": float(s.velocity[1]),
            "vz": float(s.velocity[2]),
            "yaw_deg": float(np.rad2deg(s.yaw)),
            "pitch_deg": float(np.rad2deg(s.pitch)),
            "roll_deg": float(np.rad2deg(s.roll)),
            "battery": float(s.battery),
            "airborne": bool(s.airborne),
            "action": self._action.name,
            "wind_x": float(s.wind_seen[0]),
            "wind_y": float(s.wind_seen[1]),
            "wind_z": float(s.wind_seen[2]),
            "pad": self.detected_pad.pad_id if self.detected_pad else -1,
        }
