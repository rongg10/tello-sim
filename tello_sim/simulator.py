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
from .physics import MuJoCoBackend
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

        # Called on every physics step, not at the telemetry rate. The benchmark
        # uses this for minimum-clearance: a drone at 1 m/s covers 10 cm between
        # telemetry packets, which is most of the gap you were trying to measure.
        self.step_probes: list[Callable[["Simulator"], None]] = []

        self.separation_events: list[tuple[float, str, str, float]] = []
        self.min_separation = 0.0

        # The physics engine is built on first use, not here, because a model
        # has to contain every drone and callers add drones after constructing
        # the simulator. `_world_revision` catches obstacles appearing later:
        # a compiled MuJoCo model cannot gain a body, so it has to be rebuilt.
        self._backend: MuJoCoBackend | None = None
        self._world_revision = -1

        # Held for every touch of the MuJoCo model. The clock thread steps it
        # while the 3D view reads obstacle poses and scripts add or move things,
        # and a rebuild that swaps the model out from under a running mj_step is
        # not a wrong answer but a segfault. Measured, not assumed: without this,
        # adding obstacles during a realtime run crashed the interpreter within
        # seconds. Reentrant, because a step emits telemetry that reads poses.
        self._physics_lock = threading.RLock()

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
        with self._physics_lock:
            self.drones.append(drone)
            self._attach(drone)
            self._backend = None      # a new airframe means a new model
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

    @property
    def physics(self) -> MuJoCoBackend:
        """The MuJoCo backend, compiled on demand and kept in step with the world."""
        with self._physics_lock:
            if self._backend is None:
                self._backend = MuJoCoBackend(
                    world=self.world,
                    drones=self.drones,
                    timestep=self.spec.dt,
                    collisions=self.spec.collisions,
                )
                self._world_revision = self.world.revision
            elif self.world.revision != self._world_revision:
                self._backend.rebuild(self.drones)
                self._world_revision = self.world.revision
            return self._backend

    def step_once(self, dt: float | None = None) -> None:
        with self._physics_lock:
            self._step_locked(self.spec.dt if dt is None else dt)

    def _step_locked(self, dt: float) -> None:
        backend = self.physics

        self.world.step_wind(dt)

        # Every drone decides what it wants before any of them move. They share
        # one solver, so advancing them one at a time would make a swarm's
        # contacts depend on list order.
        controls = {
            drone.name: drone.plan(dt, self.time, self.world) for drone in self.drones
        }
        hits = backend.step(controls, self.time, dt)
        for drone in self.drones:
            drone.settle(dt, self.time, self.world, hits.get(drone.name, []))

        self._track_separation()
        self._drain_events()
        for probe in self.step_probes:
            probe(self)

        self.time += dt

        self._telemetry_accumulator += dt
        if self._telemetry_accumulator >= self._telemetry_period:
            self._telemetry_accumulator -= self._telemetry_period
            self._emit_telemetry()

    def _track_separation(self) -> None:
        """Remember how close the drones got to each other.

        Keeping two airframes apart is the solver's job now.  Measuring how near
        they came is still ours, because minimum separation is the number that
        matters for a swarm: it is the difference between a formation that works
        and one that only works in the absence of wind.
        """
        if len(self.drones) < 2:
            return

        closest = float("inf")
        for i, a in enumerate(self.drones):
            for b in self.drones[i + 1:]:
                distance = float(np.linalg.norm(a.state.position - b.state.position))
                closest = min(closest, distance)
                if distance < a.spec.radius + b.spec.radius:
                    self.separation_events.append((self.time, a.name, b.name, distance))

        if np.isfinite(closest):
            self.min_separation = (
                closest if self.min_separation == 0.0 else min(self.min_separation, closest)
            )

    # ------------------------------------------------------------------
    # Editing the world while it runs
    # ------------------------------------------------------------------

    # All of these take the physics lock, so they are safe to call from any
    # thread while the clock runs on its own: a script, the 3D view, the CLI.

    def set_collisions(self, enabled: bool) -> None:
        """Turn contact response on or off for every drone, mid-flight if needed."""
        with self._physics_lock:
            self.spec.collisions = bool(enabled)
            self.physics.set_collisions(bool(enabled))

    def add_obstacle(self, obstacle):
        """Put something new in the room. The model recompiles on the next step."""
        with self._physics_lock:
            self.world.add_obstacle(obstacle)
        return obstacle

    def remove_obstacle(self, name: str) -> None:
        with self._physics_lock:
            self.world.remove_obstacle(name)

    def move_obstacle(self, name: str, position=None, yaw_deg: float | None = None) -> None:
        """Pick a movable obstacle up and put it somewhere else, right now.

        Only works on obstacles declared with a motion. A static obstacle is
        compiled into the room and has nothing to move.
        """
        yaw = None if yaw_deg is None else float(np.deg2rad(yaw_deg))
        with self._physics_lock:
            self.physics.move_obstacle(name, position=position, yaw=yaw)

    def obstacle_pose(self, name: str) -> tuple[np.ndarray, float]:
        """Where an obstacle currently is, after any scripted or physical motion."""
        with self._physics_lock:
            return self.physics.obstacle_pose(name)

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

        if self.recorder is not None:
            self.recorder.log_obstacles(self.time, self.moving_obstacle_poses())

    def moving_obstacle_poses(self) -> list:
        """Live poses of everything in the room that can move.

        Static geometry is left out: it is already in the manifest and has not
        changed, so re-recording it ten times a second would be most of the log.
        """
        poses = []
        with self._physics_lock:
            backend = self.physics
            for obstacle in self.world.obstacles:
                if not obstacle.movable:
                    continue
                position, yaw = backend.obstacle_pose(obstacle.name)
                poses.append(
                    {
                        "name": obstacle.name,
                        "p": [round(float(v), 4) for v in position],
                        "yaw": round(float(yaw), 4),
                        # w-first. Yaw stays for anything that only needs a
                        # heading; the quaternion is what shows a tumble.
                        "quat": [round(float(v), 5) for v in backend.obstacle_quat(obstacle.name)],
                    }
                )
        return poses

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
                # Recorded because a run with collisions off is not comparable
                # with one where they were on, and six months later the log is
                # the only thing left that remembers which it was.
                "collisions": self.spec.collisions,
                "engine": "mujoco",
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
