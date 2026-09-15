"""The clock: steps every drone through the same world, in lockstep.

One simulation thread owns all the physics.  Network threads and user scripts
only ever hand commands in and read telemetry out, so there is exactly one place
where state changes and no locking puzzle to get wrong.

Two clock modes:

    realtime=True   sleeps to track the wall clock, so the existing GUI and the
                    real djitellopy library behave as they would against
                    hardware, timeouts and all.
    realtime=False  runs flat out.  A ten-minute flight takes a couple of
                    seconds, which is what makes sweeps over wind conditions or
                    hundreds of episodes practical.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Iterable, Sequence

import numpy as np

from .config import DEFAULT_SIM, DroneSpec, SimSpec
from .drone import SimDrone
from .recorder import Recorder
from .world import World


class Simulator:
    """Holds the world and every drone in it."""

    def __init__(
        self,
        world: World | None = None,
        drones: Iterable[SimDrone] = (),
        sim_spec: SimSpec | None = None,
        recorder: Recorder | None = None,
    ):
        self.world = world if world is not None else World()
        self.spec = sim_spec or SimSpec()
        self.drones: list[SimDrone] = list(drones)
        self.recorder = recorder
        self.rng = np.random.default_rng(self.spec.seed)

        self.time = 0.0
        self._telemetry_accumulator = 0.0
        self._telemetry_period = 1.0 / max(self.spec.telemetry_hz, 1e-6)

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        # In fast mode the clock is pumped by whichever thread is running the
        # script.  When several threads issue commands at once (a swarm), one of
        # them takes the wheel and the rest wait, so time never advances from
        # two places at once and a run stays reproducible.
        self._driver: threading.Thread | None = None

        # Threads that are waiting for simulated time to pass register here.
        # Without this the pump cannot tell the difference between "nothing to
        # do" and "somebody is waiting for the clock", and the two sides wait
        # for each other forever.
        self._time_requests: list[list[float]] = []
        self._requests_lock = threading.Lock()
        self._listeners: list[Callable[[SimDrone, float], None]] = []

        self.separation_events: list[tuple[float, str, str, float]] = []
        self.min_separation = 0.0

        for drone in self.drones:
            self._attach(drone)

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def add_drone(
        self,
        name: str | None = None,
        start_position: Sequence[float] = (0.0, 0.0, 0.0),
        start_yaw_deg: float = 0.0,
        spec: DroneSpec | None = None,
    ) -> SimDrone:
        name = name or f"tello-{len(self.drones) + 1}"
        drone = SimDrone(
            name=name,
            spec=spec or DroneSpec(),
            start_position=start_position,
            start_yaw_deg=start_yaw_deg,
            rng=np.random.default_rng(self.spec.seed + len(self.drones) + 1),
        )
        if not self.spec.sensor_noise:
            drone.spec.noise_height = 0.0
            drone.spec.noise_attitude = 0.0
            drone.spec.noise_velocity = 0.0
        if not self.spec.actuation_noise:
            drone.spec.move_error_std = 0.0
            drone.spec.yaw_error_std_deg = 0.0
        self.drones.append(drone)
        self._attach(drone)
        return drone

    def _attach(self, drone: SimDrone) -> None:
        if self.recorder is not None:
            drone.on_command_finished = (
                lambda pending, name=drone.name: self.recorder.log_command(name, pending)
            )

    def get(self, name: str) -> SimDrone:
        for drone in self.drones:
            if drone.name == name:
                return drone
        raise KeyError(f"No drone named {name!r}")

    def on_telemetry(self, callback: Callable[[SimDrone, float], None]) -> None:
        """Register something to be told about each drone at the telemetry rate.

        The UDP server uses this to broadcast state packets, which is why the
        packet rate in the simulator matches the real drone's 10 Hz rather than
        the 100 Hz physics rate.
        """
        self._listeners.append(callback)

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def step_once(self, dt: float | None = None) -> None:
        dt = self.spec.dt if dt is None else dt

        self.world.step_wind(dt)
        for drone in self.drones:
            drone.step(dt, self.time, self.world)

        self._resolve_drone_collisions()
        self._drain_events()

        self.time += dt

        self._telemetry_accumulator += dt
        if self._telemetry_accumulator >= self._telemetry_period:
            self._telemetry_accumulator -= self._telemetry_period
            self._emit_telemetry()

    def _resolve_drone_collisions(self) -> None:
        """Keep drones out of each other, and remember how close they got.

        Minimum separation is the number that matters for a swarm: it is the
        difference between a formation that works and one that only works in
        the absence of wind.
        """
        closest = float("inf")
        for i, a in enumerate(self.drones):
            for b in self.drones[i + 1:]:
                delta = a.state.position - b.state.position
                distance = float(np.linalg.norm(delta))
                closest = min(closest, distance)
                reach = a.spec.radius + b.spec.radius
                if distance >= reach:
                    continue

                normal = delta / distance if distance > 1e-9 else np.array([1.0, 0.0, 0.0])
                overlap = reach - distance
                a.state.position = a.state.position + normal * (overlap * 0.5)
                b.state.position = b.state.position - normal * (overlap * 0.5)

                closing = float(np.dot(a.state.velocity - b.state.velocity, normal))
                if closing < 0.0:
                    a.state.velocity = a.state.velocity - normal * closing * 0.5
                    b.state.velocity = b.state.velocity + normal * closing * 0.5

                self.separation_events.append((self.time, a.name, b.name, distance))
                a.events.append((self.time, f"contact with {b.name}"))
                b.events.append((self.time, f"contact with {a.name}"))

        if len(self.drones) > 1 and np.isfinite(closest):
            self.min_separation = (
                closest if self.min_separation == 0.0 else min(self.min_separation, closest)
            )

    def _drain_events(self) -> None:
        if self.recorder is None:
            for drone in self.drones:
                drone.events.clear()
            return
        for drone in self.drones:
            while drone.events:
                when, detail = drone.events.pop(0)
                kind = "collision" if detail.startswith(("collision", "contact")) else "note"
                self.recorder.log_event(when, drone.name, kind, detail)

    def _emit_telemetry(self) -> None:
        for drone in self.drones:
            # Build the packet first: it refreshes the cached ground clearance
            # that the snapshot's time-of-flight reading depends on.
            drone.state_packet(self.world)
            if self.recorder is not None:
                self.recorder.log_snapshot(drone.snapshot())
            for listener in self._listeners:
                listener(drone, self.time)

    # ------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------

    def run_for(self, seconds: float) -> None:
        """Advance the clock by this much. Honours realtime if it is set."""
        steps = int(round(seconds / self.spec.dt))
        started = time.perf_counter()
        for i in range(steps):
            self.step_once()
            if self.spec.realtime:
                target = started + (i + 1) * self.spec.dt
                delay = target - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)

    def start(self) -> None:
        """Run the clock on a background thread until stop() is called."""
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="tello-sim", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        started = time.perf_counter()
        step = 0
        while not self._stop.is_set():
            self.step_once()
            step += 1
            if self.spec.realtime:
                target = started + step * self.spec.dt
                delay = target - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -1.0:
                    # The machine cannot keep up; resynchronise rather than
                    # silently accumulating an ever-growing backlog.
                    started = time.perf_counter() - step * self.spec.dt

    # -- who is allowed to advance the clock ------------------------------

    def may_drive(self) -> bool:
        """True if the calling thread should step the simulation itself."""
        if self.is_running:
            return False
        return self._driver is None or self._driver is threading.current_thread()

    def take_wheel(self) -> None:
        self._driver = threading.current_thread()

    def release_wheel(self) -> None:
        if self._driver is threading.current_thread():
            self._driver = None

    def add_time_request(self, target: float) -> list[float]:
        """Ask the driver to keep the clock running until this instant."""
        token = [float(target)]
        with self._requests_lock:
            self._time_requests.append(token)
        return token

    def drop_time_request(self, token: list[float]) -> None:
        with self._requests_lock:
            try:
                self._time_requests.remove(token)
            except ValueError:
                pass

    @property
    def has_pending_work(self) -> bool:
        """Is there anything for the simulation to actually do right now?

        The swarm pump asks this before every step, so simulated time only moves
        while a drone has a command in progress or a script is waiting for the
        clock.  Idling in the air is deliberately *not* work: without that rule
        the clock would run on while the operating system got round to
        scheduling a worker thread, and two identical runs would diverge.
        """
        for drone in self.drones:
            if drone._current is not None or drone._queue or drone._immediate:
                return True
        with self._requests_lock:
            return any(token[0] > self.time for token in self._time_requests)

    @property
    def is_running(self) -> bool:
        """True when the clock is being driven by its own background thread."""
        return self._thread is not None and self._thread.is_alive()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # ------------------------------------------------------------------

    def manifest(self) -> dict:
        return {
            "world": self.world.describe(),
            "sim": {
                "dt": self.spec.dt,
                "telemetry_hz": self.spec.telemetry_hz,
                "realtime": self.spec.realtime,
                "seed": self.spec.seed,
                "sensor_noise": self.spec.sensor_noise,
            },
            "drones": [
                {
                    "name": d.name,
                    "serial": d.serial,
                    "home": d.home.tolist(),
                    "spec": d.spec.__dict__,
                }
                for d in self.drones
            ],
        }

    def summary(self) -> dict:
        """A few numbers worth printing at the end of every run."""
        return {
            "sim_time": round(self.time, 2),
            "drones": len(self.drones),
            "battery_remaining": {
                d.name: round(d.state.battery, 1) for d in self.drones
            },
            "collisions": {d.name: len(d.collisions) for d in self.drones},
            "min_separation_m": (
                round(self.min_separation, 3) if len(self.drones) > 1 else None
            ),
            "near_misses": len(self.separation_events),
        }

    def __enter__(self) -> "Simulator":
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
        if self.recorder is not None:
            self.recorder.close()
