"""Scenarios: a flight, its world and its weather, written down in one file.

The reason for putting these in YAML rather than in Python is repeatability.  A
scenario file is a complete description of an experiment -- room, obstacles,
wind, how many drones, what they were told to do -- so a result can be handed to
someone else as a single artefact, and re-run months later without depending on
whatever a script happened to contain that day.

Every run copies the scenario into its log directory, so a recorded flight can
always be traced back to the conditions that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from . import world as world_module
from .client import SimSwarm, SimTello, TelloSimError
from .config import DroneSpec, SimSpec
from .recorder import Recorder
from .simulator import Simulator


@dataclass
class Scenario:
    name: str
    description: str = ""
    sim: dict = field(default_factory=dict)
    world: dict = field(default_factory=dict)
    drones: list = field(default_factory=list)
    drone_spec: dict = field(default_factory=dict)
    script: list = field(default_factory=list)
    source: Path | None = None

    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "Scenario":
        path = Path(path)
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls(
            name=data.get("name", path.stem),
            description=data.get("description", ""),
            sim=data.get("sim", {}) or {},
            world=data.get("world", {}) or {},
            drones=data.get("drones", []) or [],
            drone_spec=data.get("drone_spec", {}) or {},
            script=data.get("script", []) or [],
            source=path,
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "sim": self.sim,
            "world": self.world,
            "drones": self.drones,
            "drone_spec": self.drone_spec,
            "script": self.script,
            "source": str(self.source) if self.source else None,
        }

    # ------------------------------------------------------------------

    def build(
        self,
        record: bool = True,
        log_dir: str | Path = "logs",
        realtime: bool | None = None,
    ) -> tuple[Simulator, SimSwarm, Recorder | None]:
        """Create the simulator, the drones and the recorder for this scenario."""
        sim_spec = SimSpec(**self.sim)
        if realtime is not None:
            sim_spec.realtime = realtime

        rng = np.random.default_rng(sim_spec.seed)
        world = world_module.from_config(self.world, rng)

        recorder = Recorder(self.name, log_dir=log_dir, enabled=record) if record else None
        simulator = Simulator(world=world, sim_spec=sim_spec, recorder=recorder)

        entries = self.drones or [{"name": "tello-1", "start": [0.0, 0.0, 0.0]}]
        tellos = []
        for index, entry in enumerate(entries):
            spec = DroneSpec(**{**self.drone_spec, **entry.get("spec", {})})
            drone = simulator.add_drone(
                name=entry.get("name", f"tello-{index + 1}"),
                start_position=entry.get("start", [0.0, index * 0.8, 0.0]),
                start_yaw_deg=entry.get("yaw_deg", 0.0),
                spec=spec,
            )
            tellos.append(SimTello(drone, simulator))

        if recorder is not None:
            manifest = simulator.manifest()
            manifest["scenario"] = self.to_dict()
            recorder.write_manifest(manifest)

        return simulator, SimSwarm(tellos), recorder

    # ------------------------------------------------------------------

    def run(
        self,
        record: bool = True,
        log_dir: str | Path = "logs",
        realtime: bool = False,
        verbose: bool = True,
    ) -> tuple[Simulator, SimSwarm, Recorder | None]:
        """Build the scenario and carry out its script."""
        simulator, swarm, recorder = self.build(record, log_dir, realtime)
        by_name = {t.drone.name: t for t in swarm}

        if verbose:
            print(f"[scenario] {self.name}: {self.description or 'no description'}")

        try:
            # Put every drone into SDK mode before anything else, the way a
            # pilot always would. Scenarios describe flights, not handshakes.
            swarm.connect()
            for step in self.script:
                self._execute(step, swarm, by_name, simulator, verbose)
        finally:
            # Never leave a drone in the air at the end of a run: an unfinished
            # flight makes the battery and collision numbers meaningless.
            for tello in swarm:
                if tello.drone.state.airborne:
                    try:
                        tello.land()
                    except Exception:
                        pass
            if recorder is not None:
                recorder.close()

        if verbose:
            print(f"[scenario] done: {simulator.summary()}")
        return simulator, swarm, recorder

    def _execute(self, step: Any, swarm, by_name, simulator, verbose: bool) -> None:
        if isinstance(step, str):
            step = {"all": step}

        if "wait" in step:
            swarm.sleep(float(step["wait"]))
            return

        if "repeat" in step:
            block = step["repeat"]
            floor = block.get("while_battery_above")
            for iteration in range(int(block.get("times", 1))):
                if floor is not None:
                    lowest = min(t.drone.state.battery for t in swarm)
                    if lowest <= float(floor):
                        if verbose:
                            print(f"  [repeat] stopping at battery {lowest:.1f}%")
                        return
                if verbose:
                    print(f"  [repeat] lap {iteration + 1}")
                for inner in block.get("steps", []):
                    self._execute(inner, swarm, by_name, simulator, verbose)
            return

        if "parallel" in step:
            actions = step["parallel"]

            def worker(index: int, tello) -> None:
                for entry in actions:
                    if entry.get("drone") == tello.drone.name:
                        self._send(tello, entry["cmd"], verbose)

            swarm.parallel(worker)
            return

        if "all" in step:
            command = step["all"]
            swarm.parallel(lambda i, t: self._send(t, command, verbose))
            return

        if "drone" in step:
            self._send(by_name[step["drone"]], step["cmd"], verbose)
            return

        raise ValueError(f"Cannot interpret scenario step: {step!r}")

    @staticmethod
    def _send(tello: SimTello, command: str, verbose: bool) -> None:
        """Send one raw SDK command and report what came back.

        Goes through SimTello so the blocking behaviour is identical whether the
        clock is running in real time or being pumped as fast as it will go.
        A rejected command is reported and the script carries on: the recorder
        has it, and stopping the whole run over one bad move would throw away
        everything that came after it.
        """
        try:
            result = tello._send(command)
        except TelloSimError as exc:
            result = str(exc)
        if verbose:
            print(f"  {tello.drone.name}: {command} -> {result}")


def load(path: str | Path) -> Scenario:
    return Scenario.load(path)


def run(path: str | Path, **kwargs):
    return Scenario.load(path).run(**kwargs)


def list_scenarios(directory: str | Path = "scenarios") -> list[Path]:
    directory = Path(directory)
    return sorted(directory.glob("*.yaml")) if directory.exists() else []
