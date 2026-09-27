"""Scenario benchmarks: turning a flight into numbers you can compare.

A scenario already describes a room, a wind field and a script.  What it did not
describe was what *counts as doing well*, so every run ended with a human
squinting at a plot.  A benchmark adds the missing half: where the drone was
supposed to get to, how long it had, and what to measure on the way.

The metrics
-----------
The navigation four are the ones aerial vision-and-language navigation work
reports, and they are here under the same names so a number from this simulator
can sit in the same table as a number from a published benchmark:

    SR    success rate. Did it finish inside the goal radius?
    OSR   oracle success rate. Did it EVER pass inside the goal radius,
          even if it then wandered off?
    NE    navigation error. Distance from the goal at the end, in metres.
    SPL   success weighted by path length. One point for a perfect straight
          line, less for a success that took a scenic route, zero for a failure.

The gap between SR and OSR is the interesting one: it is the share of runs that
found the goal and then failed to stop, which is a different failure from never
finding it and wants a different fix.

Three more that the aerial benchmarks do not report, because their simulators
cannot:

    energy      battery percent consumed. Wind makes this move a lot.
    collisions  how many distinct things were hit.
    clearance   the smallest gap between the drone's cage and a wall, the
                ceiling or an obstacle, sampled every physics step. Zero or
                below means contact. The floor is left out, because every
                takeoff and landing touches it.

Why seeds matter
----------------
One run of one scenario is an anecdote.  Actuation noise, sensor noise and gust
timing all come off the seed, so a benchmark runs the same scenario across
several of them and reports the spread.  A policy that succeeds on seed 0 and
fails on seeds 1 through 4 has not succeeded.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml


# ----------------------------------------------------------------------
# What the drone was asked to do
# ----------------------------------------------------------------------


@dataclass
class Goal:
    """Somewhere the drone is supposed to end up.

    Distance is horizontal by default, and that is not a shortcut.  Every
    scenario ends by landing, so a goal set at flight height would be missed by
    the whole cruise altitude on every single run: a benchmark that always
    reports failure measures nothing.  Path length and SPL are planar for the
    same reason, since the climb is a fixed cost every route pays alike.

    Set `planar: false` when height is genuinely part of the task -- inspecting
    something at a particular level, holding station above a pad -- and make
    sure the script does not land before the run ends.
    """

    position: np.ndarray
    radius: float = 0.35
    name: str = "goal"
    planar: bool = True

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float)

    def distance_from(self, point: np.ndarray) -> float:
        delta = np.asarray(point, dtype=float) - self.position
        return float(np.linalg.norm(delta[:2] if self.planar else delta))

    def describe(self) -> dict:
        return {
            "position": self.position.tolist(),
            "radius": self.radius,
            "name": self.name,
            "planar": self.planar,
        }


@dataclass
class Task:
    """A goal, a deadline, and which drone is being judged.

    `time_limit` is in simulated seconds and is the honest way to fail a policy
    that would eventually arrive given a week.  `drone` names which aircraft the
    navigation metrics describe; the others still contribute collisions and
    separation, because a swarm that reaches the goal by shoving its neighbour
    into a wall has not solved the problem.
    """

    goal: Goal
    time_limit: float = 60.0
    drone: str | None = None
    name: str = "task"

    def describe(self) -> dict:
        return {
            "name": self.name,
            "goal": self.goal.describe(),
            "time_limit": self.time_limit,
            "drone": self.drone,
        }


# ----------------------------------------------------------------------
# Watching one run
# ----------------------------------------------------------------------


class RunMonitor:
    """Follows one flight and accumulates everything the metrics need.

    Attached as a step probe, so it sees every physics step.  Cheap on purpose:
    a few numpy operations per step, because at 200 Hz anything expensive here
    shows up directly in how long a sweep takes.
    """

    def __init__(self, task: Task, drone_name: str):
        self.task = task
        self.drone_name = drone_name

        self.path_length = 0.0
        self.min_clearance = float("inf")
        self.min_goal_distance = float("inf")
        self.reached_goal = False
        self.start_position: np.ndarray | None = None
        self.last_position: np.ndarray | None = None
        self.final_position = np.zeros(3)
        self.start_battery: float | None = None
        self.final_battery = 0.0
        self.elapsed = 0.0
        self.timed_out = False

    def __call__(self, sim) -> None:
        try:
            drone = sim.get(self.drone_name)
        except KeyError:  # the drone was removed mid-run; nothing left to watch
            return

        position = drone.state.position
        if self.start_position is None:
            self.start_position = position.copy()
            self.start_battery = float(drone.state.battery)
            self.last_position = position.copy()

        # Path length ignores the takeoff and landing climb, which every route
        # pays equally and which would otherwise swamp SPL on a short course.
        step = position - self.last_position
        self.path_length += float(np.linalg.norm(step[:2]))
        self.last_position = position.copy()

        gap = sim.world.clearance_at(position, include_floor=False) - drone.spec.radius
        self.min_clearance = min(self.min_clearance, gap)

        distance = self.task.goal.distance_from(position)
        self.min_goal_distance = min(self.min_goal_distance, distance)
        if distance <= self.task.goal.radius:
            self.reached_goal = True

        self.final_position = position.copy()
        self.final_battery = float(drone.state.battery)
        self.elapsed = float(sim.time)
        if self.elapsed > self.task.time_limit:
            self.timed_out = True

    # ------------------------------------------------------------------

    def result(self, sim, seed: int, label: str) -> "RunResult":
        drone = sim.get(self.drone_name)
        goal = self.task.goal

        final_distance = goal.distance_from(self.final_position)
        success = bool(final_distance <= goal.radius and not self.timed_out)

        # Shortest possible route, measured in the plane for the same reason the
        # path length is: the climb is not part of the navigation problem.
        start = self.start_position if self.start_position is not None else np.zeros(3)
        shortest = float(np.linalg.norm((goal.position - start)[:2]))
        travelled = max(self.path_length, 1e-6)
        spl = float(success) * shortest / max(shortest, travelled) if shortest > 1e-6 else float(success)

        collisions = sorted({name for _, name in drone.collisions})
        return RunResult(
            label=label,
            seed=seed,
            drone=self.drone_name,
            success=success,
            oracle_success=bool(self.reached_goal),
            navigation_error=final_distance,
            oracle_error=float(self.min_goal_distance),
            spl=float(spl),
            path_length=float(self.path_length),
            shortest_path=shortest,
            duration=self.elapsed,
            timed_out=self.timed_out,
            energy=float((self.start_battery or 0.0) - self.final_battery),
            battery_left=self.final_battery,
            collisions=len(collisions),
            collided_with=collisions,
            min_clearance=(
                float(self.min_clearance) if math.isfinite(self.min_clearance) else 0.0
            ),
            min_separation=float(sim.min_separation) if len(sim.drones) > 1 else None,
            final_position=self.final_position.tolist(),
        )


@dataclass
class RunResult:
    """One flight, scored."""

    label: str
    seed: int
    drone: str
    success: bool
    oracle_success: bool
    navigation_error: float
    oracle_error: float
    spl: float
    path_length: float
    shortest_path: float
    duration: float
    timed_out: bool
    energy: float
    battery_left: float
    collisions: int
    collided_with: list
    min_clearance: float
    min_separation: float | None
    final_position: list

    def as_dict(self) -> dict:
        return dict(self.__dict__)


# ----------------------------------------------------------------------
# Aggregating across seeds
# ----------------------------------------------------------------------


def _mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _stdev(values: Sequence[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


@dataclass
class Report:
    """Every run of one benchmark, and the summary across them."""

    name: str
    runs: list = field(default_factory=list)

    def summary(self) -> dict:
        if not self.runs:
            return {"name": self.name, "runs": 0}

        successes = [r for r in self.runs if r.success]
        return {
            "name": self.name,
            "runs": len(self.runs),
            "SR": _mean([float(r.success) for r in self.runs]),
            "OSR": _mean([float(r.oracle_success) for r in self.runs]),
            "SPL": _mean([r.spl for r in self.runs]),
            "NE": _mean([r.navigation_error for r in self.runs]),
            "NE_sd": _stdev([r.navigation_error for r in self.runs]),
            "energy": _mean([r.energy for r in self.runs]),
            "energy_sd": _stdev([r.energy for r in self.runs]),
            "collisions": _mean([float(r.collisions) for r in self.runs]),
            "crash_rate": _mean([float(r.collisions > 0) for r in self.runs]),
            "min_clearance": min(r.min_clearance for r in self.runs),
            "duration": _mean([r.duration for r in self.runs]),
            "timeouts": sum(1 for r in self.runs if r.timed_out),
            # Only meaningful over the runs that actually succeeded: the path
            # length of a failure says nothing about how efficient success is.
            "path_ratio": (
                _mean([r.path_length / max(r.shortest_path, 1e-6) for r in successes])
                if successes else 0.0
            ),
        }

    def to_dict(self) -> dict:
        return {"summary": self.summary(), "runs": [r.as_dict() for r in self.runs]}

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    def table(self) -> str:
        """The summary as one line, for printing a sweep as it runs."""
        s = self.summary()
        if not s.get("runs"):
            return f"{self.name}: no runs"
        return (
            f"{s['name']:<26} n={s['runs']:<3} "
            f"SR={s['SR']:.2f}  OSR={s['OSR']:.2f}  SPL={s['SPL']:.2f}  "
            f"NE={s['NE']:.2f}±{s['NE_sd']:.2f}m  "
            f"energy={s['energy']:.1f}%  "
            f"crash={s['crash_rate']:.2f}  "
            f"clear={s['min_clearance']:.2f}m"
        )


def compare(reports: Sequence[Report]) -> str:
    """Several reports side by side, which is the point of running a sweep."""
    if not reports:
        return "nothing to compare"
    header = (
        f"{'condition':<26} {'n':>3} {'SR':>5} {'OSR':>5} {'SPL':>5} "
        f"{'NE(m)':>8} {'energy%':>8} {'crash':>6} {'clear(m)':>9}"
    )
    lines = [header, "-" * len(header)]
    for report in reports:
        s = report.summary()
        if not s.get("runs"):
            lines.append(f"{report.name:<26} {'-':>3}")
            continue
        lines.append(
            f"{s['name']:<26} {s['runs']:>3} {s['SR']:>5.2f} {s['OSR']:>5.2f} "
            f"{s['SPL']:>5.2f} {s['NE']:>8.2f} {s['energy']:>8.1f} "
            f"{s['crash_rate']:>6.2f} {s['min_clearance']:>9.2f}"
        )
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Running one
# ----------------------------------------------------------------------


def task_from_config(spec: dict) -> Task:
    """Build a Task from the `benchmark:` block of a scenario file."""
    if "goal" not in spec:
        raise ValueError(
            "A benchmark needs a goal. Add one to the scenario's benchmark block:\n"
            "  benchmark:\n"
            "    goal: [2.0, 1.0, 1.0]\n"
            "    success_radius: 0.35"
        )
    return Task(
        goal=Goal(
            position=spec["goal"],
            radius=float(spec.get("success_radius", 0.35)),
            name=spec.get("goal_name", "goal"),
            planar=bool(spec.get("planar", True)),
        ),
        time_limit=float(spec.get("time_limit", 60.0)),
        drone=spec.get("drone"),
        name=spec.get("name", "task"),
    )


def run_once(
    scenario,
    task: Task,
    seed: int,
    label: str,
    collisions: bool | None = None,
    log_dir: str | Path = "logs",
    record: bool = False,
    verbose: bool = False,
) -> RunResult:
    """Fly one scenario once, under one seed, and score it.

    The scenario is rebuilt from scratch for every seed rather than reset,
    because a reset that forgets one field is a bug that shows up as a
    mysteriously good result on the second run and nowhere else.
    """
    from .scenarios import Scenario

    spec = dict(scenario.sim)
    spec["seed"] = seed
    if collisions is not None:
        spec["collisions"] = bool(collisions)

    replica = Scenario(
        name=scenario.name,
        description=scenario.description,
        sim=spec,
        world=scenario.world,
        drones=scenario.drones,
        drone_spec=scenario.drone_spec,
        script=scenario.script,
        source=scenario.source,
    )

    simulator, swarm, recorder = replica.build(
        record=record, log_dir=log_dir, realtime=False
    )

    drone_name = task.drone or simulator.drones[0].name
    monitor = RunMonitor(task, drone_name)
    simulator.step_probes.append(monitor)

    try:
        swarm.connect()
        for step in replica.script:
            if monitor.timed_out:
                break
            replica._execute(step, swarm, {t.drone.name: t for t in swarm}, simulator, verbose)
    finally:
        for tello in swarm:
            if tello.drone.state.airborne:
                try:
                    tello.land()
                except Exception:
                    # A drone that cannot land is usually one that already
                    # crashed. The metrics for the flight still stand.
                    pass
        if recorder is not None:
            recorder.close()

    return monitor.result(simulator, seed, label)


def run(
    scenario,
    task: Task | None = None,
    seeds: Sequence[int] = (0, 1, 2),
    label: str | None = None,
    collisions: bool | None = None,
    log_dir: str | Path = "logs",
    record: bool = False,
    verbose: bool = False,
) -> Report:
    """Fly one scenario across several seeds and collect the results."""
    if task is None:
        if not scenario.benchmark:
            raise ValueError(
                f"Scenario {scenario.name!r} has no benchmark block, so there is "
                "nothing to score it against. Add one, or pass a Task."
            )
        task = task_from_config(scenario.benchmark)

    label = label or scenario.name
    report = Report(name=label)
    for seed in seeds:
        result = run_once(
            scenario, task, seed, label,
            collisions=collisions, log_dir=log_dir, record=record, verbose=verbose,
        )
        report.runs.append(result)
        if verbose:
            mark = "ok  " if result.success else "FAIL"
            print(
                f"  [{mark}] seed {seed}: NE={result.navigation_error:.2f}m "
                f"SPL={result.spl:.2f} energy={result.energy:.1f}% "
                f"hits={result.collisions}"
            )
    return report


def ablate_collisions(
    scenario,
    task: Task | None = None,
    seeds: Sequence[int] = (0, 1, 2),
    **kwargs: Any,
) -> list[Report]:
    """Run the same scenario with contacts on and off.

    The pair is the measurement.  With contacts on you learn whether the policy
    survives; with them off you learn where it was actually trying to go, which
    a policy that crashes in the first three seconds never gets to show you.
    A large gap between the two means the route was only ever working by luck.
    """
    return [
        run(scenario, task, seeds, label=f"{scenario.name} [solid]",
            collisions=True, **kwargs),
        run(scenario, task, seeds, label=f"{scenario.name} [ghost]",
            collisions=False, **kwargs),
    ]


def load_suite(directory: str | Path = "scenarios") -> list:
    """Every scenario in a directory that carries a benchmark block."""
    from .scenarios import Scenario

    directory = Path(directory)
    if not directory.exists():
        return []
    out = []
    for path in sorted(directory.glob("*.yaml")):
        scenario = Scenario.load(path)
        if scenario.benchmark:
            out.append(scenario)
    return out
