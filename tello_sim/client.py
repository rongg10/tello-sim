"""SimTello: the same API as djitellopy.Tello, but talking to the simulator.

Two ways to reach the simulator exist, and they are for different jobs:

  * the UDP server (server.py) is the honest one -- real sockets, real library,
    the existing GUI unchanged.  Use it to check that flight code works.
  * SimTello, here, skips the network.  Use it for experiments: hundreds of
    episodes, many drones, faster than real time, and perfectly repeatable.

Both drive the identical physics and the identical command parser, so a
behaviour seen in one appears in the other.

Timing note: in fast mode there is no background clock, so a blocking call
advances the simulation itself until the command finishes.  Scripts must use
`drone.sleep(seconds)` rather than `time.sleep`, or simulated time will never
move.
"""

from __future__ import annotations

import threading
from typing import Callable, Sequence

import numpy as np

from .drone import SimDrone
from .protocol import parse_state
from .simulator import Simulator


class TelloSimError(RuntimeError):
    """Raised when the drone answers with an error, as djitellopy would."""


def _sleep_in_sim(sim: Simulator, seconds: float) -> None:
    """Wait `seconds` of simulated time, whoever is driving the clock."""
    import time as _time

    if sim.is_running:
        _time.sleep(seconds)
        return

    if sim.may_drive():
        sim.run_for(seconds)
        return

    # Someone else advances the clock. Tell them we need it to keep going,
    # then watch it go past.
    target = sim.time + seconds
    token = sim.add_time_request(target)
    wall_deadline = _time.perf_counter() + max(seconds * 10.0, 30.0)
    try:
        while sim.time < target and _time.perf_counter() < wall_deadline:
            _time.sleep(0.0002)
    finally:
        sim.drop_time_request(token)


class SimTello:
    """A drop-in stand-in for djitellopy.Tello."""

    def __init__(self, drone: SimDrone, simulator: Simulator, raise_on_error: bool = True):
        self._drone = drone
        self._sim = simulator
        self.raise_on_error = raise_on_error
        self.is_flying = False
        self.stream_on = False

    # ------------------------------------------------------------------
    # Plumbing
    # ------------------------------------------------------------------

    @property
    def drone(self) -> SimDrone:
        return self._drone

    @property
    def simulator(self) -> Simulator:
        return self._sim

    def _send(self, command: str, timeout: float = 30.0) -> str:
        pending = self._drone.submit(command)

        if self._sim.may_drive():
            # Nobody else is turning the clock: do it here until this finishes.
            deadline = self._sim.time + timeout
            while not pending.event.is_set() and self._sim.time < deadline:
                self._sim.step_once()
            if not pending.event.is_set():
                raise TelloSimError(f"'{command}' timed out after {timeout}s of sim time")
            response = pending.response
        else:
            # Another thread owns the clock (a live simulation, or the swarm
            # pump). Just wait to be told the command finished.
            response = pending.wait(timeout=timeout)
            if response is None:
                raise TelloSimError(f"'{command}' timed out after {timeout}s")

        if self.raise_on_error and response is not None and response.startswith("error"):
            raise TelloSimError(f"'{command}' -> {response}")
        return response or ""

    def sleep(self, seconds: float) -> None:
        """Wait, in simulated time. Use this instead of time.sleep in scripts."""
        _sleep_in_sim(self._sim, seconds)

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self, wait_for_state: bool = True) -> None:
        self._send("command")
        if wait_for_state:
            self.sleep(0.2)

    def end(self) -> None:
        if self.is_flying:
            try:
                self.land()
            except TelloSimError:
                pass

    # ------------------------------------------------------------------
    # Flight
    # ------------------------------------------------------------------

    def takeoff(self) -> None:
        self._send("takeoff", timeout=30.0)
        self.is_flying = True

    def land(self) -> None:
        self._send("land", timeout=30.0)
        self.is_flying = False

    def emergency(self) -> None:
        self._send("emergency")
        self.is_flying = False

    def move(self, direction: str, x: int) -> None:
        self._send(f"{direction} {int(x)}")

    def move_forward(self, x: int) -> None:
        self.move("forward", x)

    def move_back(self, x: int) -> None:
        self.move("back", x)

    def move_left(self, x: int) -> None:
        self.move("left", x)

    def move_right(self, x: int) -> None:
        self.move("right", x)

    def move_up(self, x: int) -> None:
        self.move("up", x)

    def move_down(self, x: int) -> None:
        self.move("down", x)

    def rotate_clockwise(self, x: int) -> None:
        self._send(f"cw {int(x)}")

    def rotate_counter_clockwise(self, x: int) -> None:
        self._send(f"ccw {int(x)}")

    def flip(self, direction: str) -> None:
        self._send(f"flip {direction}")

    def flip_left(self) -> None:
        self.flip("l")

    def flip_right(self) -> None:
        self.flip("r")

    def flip_forward(self) -> None:
        self.flip("f")

    def flip_back(self) -> None:
        self.flip("b")

    def go_xyz_speed(self, x: int, y: int, z: int, speed: int) -> None:
        self._send(f"go {int(x)} {int(y)} {int(z)} {int(speed)}", timeout=60.0)

    def curve_xyz_speed(
        self, x1: int, y1: int, z1: int, x2: int, y2: int, z2: int, speed: int
    ) -> None:
        self._send(
            f"curve {int(x1)} {int(y1)} {int(z1)} {int(x2)} {int(y2)} {int(z2)} {int(speed)}",
            timeout=60.0,
        )

    def go_xyz_speed_mid(self, x: int, y: int, z: int, speed: int, mid: int) -> None:
        self._send(f"go {int(x)} {int(y)} {int(z)} {int(speed)} m{int(mid)}", timeout=60.0)

    def curve_xyz_speed_mid(
        self, x1: int, y1: int, z1: int, x2: int, y2: int, z2: int, speed: int, mid: int
    ) -> None:
        self._send(
            f"curve {int(x1)} {int(y1)} {int(z1)} {int(x2)} {int(y2)} {int(z2)} "
            f"{int(speed)} m{int(mid)}",
            timeout=60.0,
        )

    def go_xyz_speed_yaw_mid(
        self, x: int, y: int, z: int, speed: int, yaw: int, mid1: int, mid2: int
    ) -> None:
        self._send(
            f"jump {int(x)} {int(y)} {int(z)} {int(speed)} {int(yaw)} "
            f"m{int(mid1)} m{int(mid2)}",
            timeout=60.0,
        )

    def send_rc_control(
        self, left_right: int, forward_back: int, up_down: int, yaw: int
    ) -> None:
        """Stick input. Returns at once, like the real fire-and-forget command."""
        self._drone.submit(f"rc {int(left_right)} {int(forward_back)} {int(up_down)} {int(yaw)}")

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def set_speed(self, x: int) -> None:
        self._send(f"speed {int(x)}")

    def enable_mission_pads(self) -> None:
        self._send("mon")

    def disable_mission_pads(self) -> None:
        self._send("moff")

    def set_mission_pad_direction(self, x: int) -> None:
        self._send(f"mdirection {int(x)}")

    # djitellopy spells it out in full. Both names work here, so a script does
    # not have to know which path it is running on.
    set_mission_pad_detection_direction = set_mission_pad_direction

    def streamon(self) -> None:
        self.stream_on = True
        self._send("streamon")

    def streamoff(self) -> None:
        self.stream_on = False
        self._send("streamoff")

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def get_current_state(self) -> dict:
        return parse_state(self._drone.state_packet(self._sim.world))

    def get_state_field(self, key: str):
        state = self.get_current_state()
        if key not in state:
            raise TelloSimError(f"Could not get state property: {key}")
        return state[key]

    def get_battery(self) -> int:
        return self.get_state_field("bat")

    def get_height(self) -> int:
        return self.get_state_field("h")

    def get_distance_tof(self) -> int:
        return self.get_state_field("tof")

    def get_flight_time(self) -> int:
        return self.get_state_field("time")

    def get_barometer(self) -> float:
        return self.get_state_field("baro")

    def get_yaw(self) -> int:
        return self.get_state_field("yaw")

    def get_pitch(self) -> int:
        return self.get_state_field("pitch")

    def get_roll(self) -> int:
        return self.get_state_field("roll")

    def get_speed_x(self) -> int:
        return self.get_state_field("vgx")

    def get_speed_y(self) -> int:
        return self.get_state_field("vgy")

    def get_speed_z(self) -> int:
        return self.get_state_field("vgz")

    def get_temperature(self) -> float:
        state = self.get_current_state()
        return (state["templ"] + state["temph"]) / 2.0

    def get_mission_pad_id(self) -> int:
        return self.get_state_field("mid")

    # -- ground truth, which the real drone cannot give you -----------------

    def true_position(self) -> np.ndarray:
        """Exact position in world coordinates.

        Not available on hardware.  Use it to score a run, never to fly one --
        a controller that reads this will not survive contact with the drone.
        """
        return self._drone.state.position.copy()

    def true_yaw_deg(self) -> float:
        return float(np.rad2deg(self._drone.state.yaw))


class SimSwarm:
    """Several SimTellos treated as one, mirroring djitellopy's TelloSwarm.

    `parallel` is the interesting one: it starts a command on every drone at the
    same simulated instant and then advances one shared clock until all of them
    report done.  There is no way for one drone to run ahead of another, which
    is what makes formation and separation results trustworthy.
    """

    def __init__(self, tellos: Sequence[SimTello]):
        self.tellos = list(tellos)
        if not self.tellos:
            raise ValueError("A swarm needs at least one drone")
        self._sim = self.tellos[0].simulator

    def __len__(self) -> int:
        return len(self.tellos)

    def __iter__(self):
        return iter(self.tellos)

    def __getitem__(self, index: int) -> SimTello:
        return self.tellos[index]

    def parallel(self, func: Callable[[int, SimTello], None], timeout: float = 120.0) -> None:
        import time as _time

        errors: list[BaseException] = []

        def worker(index: int, tello: SimTello) -> None:
            try:
                func(index, tello)
            except BaseException as exc:  # noqa: BLE001 - re-raised after joining
                errors.append(exc)

        # Claim the clock before any worker exists. Do it afterwards and a
        # worker can find no driver, start stepping on its own, and end up
        # writing the telemetry log at the same time as this thread.
        driving = not self._sim.is_running
        if driving:
            self._sim.take_wheel()

        try:
            threads = [
                threading.Thread(target=worker, args=(i, t), daemon=True)
                for i, t in enumerate(self.tellos)
            ]
            for thread in threads:
                thread.start()

            if not driving:
                for thread in threads:
                    thread.join(timeout=timeout)
            else:
                # Drive the shared clock while the workers wait on it, stepping
                # only when a drone actually has work in progress or a script is
                # waiting for time to pass. That keeps two identical runs
                # identical, instead of letting thread scheduling leak in.
                deadline = self._sim.time + timeout
                while any(t.is_alive() for t in threads) and self._sim.time < deadline:
                    if self._sim.has_pending_work:
                        self._sim.step_once()
                    else:
                        _time.sleep(0.0002)
                for thread in threads:
                    thread.join(timeout=1.0)
        finally:
            if driving:
                self._sim.release_wheel()

        if errors:
            raise errors[0]

    def sequential(self, func: Callable[[int, SimTello], None]) -> None:
        for index, tello in enumerate(self.tellos):
            func(index, tello)

    # Convenience broadcasts, matching TelloSwarm's most-used methods.

    def connect(self) -> None:
        self.parallel(lambda i, t: t.connect())

    def takeoff(self) -> None:
        self.parallel(lambda i, t: t.takeoff())

    def land(self) -> None:
        self.parallel(lambda i, t: t.land())

    def set_speed(self, x: int) -> None:
        self.parallel(lambda i, t: t.set_speed(x))

    def sleep(self, seconds: float) -> None:
        _sleep_in_sim(self._sim, seconds)

    def end(self) -> None:
        self.parallel(lambda i, t: t.end())


def build(
    n: int = 1,
    world=None,
    sim_spec=None,
    recorder=None,
    start_positions: Sequence[Sequence[float]] | None = None,
    start_yaws_deg: Sequence[float] | None = None,
    names: Sequence[str] | None = None,
) -> tuple[Simulator, SimSwarm]:
    """Shortcut: make a simulator with n drones and return both halves.

    Drones are spaced along the y axis by default so they do not start on top
    of one another.
    """
    from .world import World

    sim = Simulator(world=world or World(), sim_spec=sim_spec, recorder=recorder)

    if start_positions is None:
        spacing = 0.8
        offset = (n - 1) * spacing / 2.0
        start_positions = [(0.0, i * spacing - offset, 0.0) for i in range(n)]

    tellos = []
    for i in range(n):
        drone = sim.add_drone(
            name=names[i] if names else f"tello-{i + 1}",
            start_position=start_positions[i],
            start_yaw_deg=start_yaws_deg[i] if start_yaws_deg else 0.0,
        )
        tellos.append(SimTello(drone, sim))

    if recorder is not None:
        recorder.write_manifest(sim.manifest())

    return sim, SimSwarm(tellos)
