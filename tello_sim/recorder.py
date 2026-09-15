"""Recording a run: every command received, every position, every event.

The point of writing all three is that a simulated run and a real flight can be
recorded in the same format and then compared directly.  Divergence between the
two is the calibration signal -- it tells you which constant in config.py is
wrong, instead of leaving you to guess.

Files written per run, under logs/<timestamp>_<name>/:

    commands.jsonl   what was asked, what was answered, and how long it took
    telemetry.csv    position, velocity, battery and wind, ten times a second
    events.jsonl     collisions, timeouts, battery warnings
    run.json         the scenario and every constant used, for reproducibility
"""

from __future__ import annotations

import csv
import json
import threading
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

TELEMETRY_COLUMNS = [
    "t", "drone", "x", "y", "z", "vx", "vy", "vz",
    "yaw_deg", "pitch_deg", "roll_deg",
    "battery", "airborne", "action",
    "wind_x", "wind_y", "wind_z", "pad",
]


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


class Recorder:
    """Writes one run to disk. Safe to use from the simulation thread."""

    def __init__(self, name: str = "run", log_dir: str | Path = "logs", enabled: bool = True):
        self.enabled = enabled
        self.name = name
        # Commands answered instantly (`command`, `battery?`, `speed`) complete
        # on the calling thread, so several drones in a swarm can be writing
        # here at once. Without this the files interleave mid-line.
        self._write_lock = threading.Lock()
        if not enabled:
            self.directory = None
            return

        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.directory = Path(log_dir) / f"{stamp}_{name}"
        self.directory.mkdir(parents=True, exist_ok=True)

        self._commands = (self.directory / "commands.jsonl").open("w", encoding="utf-8")
        self._events = (self.directory / "events.jsonl").open("w", encoding="utf-8")
        self._telemetry_file = (self.directory / "telemetry.csv").open("w", newline="", encoding="utf-8")
        self._telemetry = csv.DictWriter(self._telemetry_file, fieldnames=TELEMETRY_COLUMNS)
        self._telemetry.writeheader()
        self._closed = False

    # -- writing -----------------------------------------------------------

    def log_command(self, drone_name: str, pending, wall_clock: float | None = None) -> None:
        """Record one command exchange, including how long the drone took."""
        if not self.enabled:
            return
        record = {
            "wall_clock": wall_clock if wall_clock is not None else time.time(),
            "drone": drone_name,
            "command": pending.raw,
            "verb": pending.verb,
            "args": pending.args,
            "response": pending.response,
            "t_submitted": pending.submitted_at,
            "t_started": pending.started_at,
            "t_finished": pending.finished_at,
            "duration": (
                None
                if pending.finished_at is None
                else round(pending.finished_at - pending.submitted_at, 4)
            ),
        }
        with self._write_lock:
            self._commands.write(json.dumps(record) + "\n")
            self._commands.flush()

    def log_snapshot(self, snapshot: dict) -> None:
        if not self.enabled:
            return
        with self._write_lock:
            self._telemetry.writerow({k: snapshot.get(k) for k in TELEMETRY_COLUMNS})

    def log_event(self, sim_time: float, drone_name: str, kind: str, detail: str = "") -> None:
        if not self.enabled:
            return
        with self._write_lock:
            self._events.write(
                json.dumps(
                    {"t": round(sim_time, 4), "drone": drone_name, "kind": kind, "detail": detail}
                )
                + "\n"
            )
            self._events.flush()

    def write_manifest(self, payload: dict) -> None:
        """Store the scenario and the physical constants that produced this run."""
        if not self.enabled:
            return
        with (self.directory / "run.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=_jsonable)

    def close(self) -> None:
        if not self.enabled or getattr(self, "_closed", True):
            return
        self._telemetry_file.flush()
        self._telemetry_file.close()
        self._commands.close()
        self._events.close()
        self._closed = True

    def __enter__(self) -> "Recorder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def load_run(directory: str | Path) -> dict:
    """Read a recorded run back, for plotting or comparing against real flight."""
    directory = Path(directory)

    telemetry: list[dict] = []
    telemetry_path = directory / "telemetry.csv"
    if telemetry_path.exists():
        with telemetry_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                parsed = dict(row)
                for key in TELEMETRY_COLUMNS:
                    if key in ("drone", "action", "airborne"):
                        continue
                    try:
                        parsed[key] = float(row[key])
                    except (TypeError, ValueError):
                        parsed[key] = None
                parsed["airborne"] = row.get("airborne") == "True"
                telemetry.append(parsed)

    def read_jsonl(path: Path) -> list[dict]:
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    manifest_path = directory / "run.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    return {
        "directory": directory,
        "telemetry": telemetry,
        "commands": read_jsonl(directory / "commands.jsonl"),
        "events": read_jsonl(directory / "events.jsonl"),
        "manifest": manifest,
    }


def latest_run(log_dir: str | Path = "logs") -> Path | None:
    """The most recent run directory, so scripts can say 'plot what I just did'."""
    log_dir = Path(log_dir)
    if not log_dir.exists():
        return None
    runs = sorted((p for p in log_dir.iterdir() if p.is_dir()), key=lambda p: p.name)
    return runs[-1] if runs else None
