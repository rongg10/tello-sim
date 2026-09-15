"""A live 3D view of the simulation, served to a browser.

Why a browser rather than a desktop 3D window: the obvious choice on this
machine would have been PyBullet, but it has no prebuilt wheel for Apple
silicon and its window insists on owning the main thread, which fights the
tkinter controller.  A page served over localhost has neither problem, needs
nothing compiled, renders as well as anything, and can be screen-recorded or
shown on someone else's laptop without installing a thing.

The page is both a view and a controller.  It sends ordinary SDK commands, so
it is doing exactly what the tkinter GUI does -- and both can be connected at
once, because the drone does not care who is talking to it.

Nothing here affects the physics.  The viewer only reads state and submits
commands, the same as any other client.
"""

from __future__ import annotations

import json
import mimetypes
import threading
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

import numpy as np

from .dynamics import attitude_basis
from .simulator import Simulator

VIEWER_ROOT = Path(__file__).resolve().parent / "viewer"

# How finely the wind field is sampled for display. Small enough to send at
# frame rate, dense enough to make a draught that covers half the room obvious.
WIND_GRID = (4, 4, 2)


class ViewerState:
    """Everything the page needs, gathered on demand from the simulation."""

    def __init__(self, simulator: Simulator, log_size: int = 200):
        self.simulator = simulator
        self.log: deque[dict] = deque(maxlen=log_size)
        self._pending: deque = deque()
        self._lock = threading.Lock()
        self._log_seq = 0

    # -- commands ------------------------------------------------------

    def submit(self, drone_name: str, command: str) -> dict:
        drone = self.simulator.get(drone_name)
        pending = drone.submit(command)
        with self._lock:
            self._pending.append((drone_name, pending))
        return {"ok": True, "command": command, "drone": drone_name}

    def _drain_pending(self) -> None:
        """Move finished commands into the log the page displays."""
        with self._lock:
            still_running = deque()
            for name, pending in self._pending:
                if pending.event.is_set():
                    self._log_seq += 1
                    took = (
                        None
                        if pending.finished_at is None
                        else round(pending.finished_at - pending.submitted_at, 2)
                    )
                    self.log.append(
                        {
                            "seq": self._log_seq,
                            "t": round(self.simulator.time, 2),
                            "drone": name,
                            "command": pending.raw,
                            "response": pending.response,
                            "took": took,
                        }
                    )
                else:
                    still_running.append((name, pending))
            self._pending = still_running

    def pending_count(self, drone_name: str) -> int:
        with self._lock:
            return sum(1 for name, _ in self._pending if name == drone_name)

    # -- world ---------------------------------------------------------

    def world_payload(self) -> dict:
        world = self.simulator.world
        return {
            "world": world.describe(),
            "drones": [
                {"name": d.name, "serial": d.serial, "radius": d.spec.radius}
                for d in self.simulator.drones
            ],
            "wind_grid": WIND_GRID,
        }

    # -- per-frame state ----------------------------------------------

    def state_payload(self) -> dict:
        self._drain_pending()
        simulator = self.simulator
        world = simulator.world
        now = simulator.time

        drones = []
        for drone in simulator.drones:
            state = drone.state
            basis = attitude_basis(state)
            # The rotors have to work harder to hold a tilt or to climb; showing
            # that in the spin rate makes the effort visible rather than implied.
            effort = float(np.linalg.norm(state.last_accel_cmd + np.array([0.0, 0.0, 9.81])))
            drones.append(
                {
                    "name": drone.name,
                    "p": [round(float(v), 4) for v in state.position],
                    "v": [round(float(v), 3) for v in state.velocity],
                    "forward": [round(float(v), 4) for v in basis[0]],
                    "left": [round(float(v), 4) for v in basis[1]],
                    "up": [round(float(v), 4) for v in basis[2]],
                    "yaw_deg": round(float(np.degrees(state.yaw)), 1),
                    "battery": round(float(state.battery), 1),
                    "airborne": bool(state.airborne),
                    "motors": bool(state.motors_on),
                    "rotor": round(effort / 9.81, 3) if state.motors_on else 0.0,
                    "action": drone._action.name,
                    "height_cm": drone._height_cm(),
                    "tof_cm": drone._tof_cm(),
                    "wind": [round(float(v), 3) for v in state.wind_seen],
                    "pad": drone.detected_pad.pad_id if drone.detected_pad else -1,
                    "pending": self.pending_count(drone.name),
                }
            )

        return {
            "t": round(now, 2),
            "drones": drones,
            "wind": self._wind_samples(now),
            "log": list(self.log)[-40:],
            "min_separation": (
                round(simulator.min_separation, 3) if len(simulator.drones) > 1 else None
            ),
        }

    def _wind_samples(self, now: float) -> list:
        """Sample the wind on a coarse grid so the page can draw the field.

        Worth the bytes: a draught that covers only part of the room is
        invisible in a single number, and it is exactly the condition that makes
        a formation fail.
        """
        world = self.simulator.world
        lower, upper = world.bounds_lower, world.bounds_upper
        nx, ny, nz = WIND_GRID
        samples = []
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    point = np.array(
                        [
                            lower[0] + (upper[0] - lower[0]) * (i + 0.5) / nx,
                            lower[1] + (upper[1] - lower[1]) * (j + 0.5) / ny,
                            lower[2] + (upper[2] - lower[2]) * (k + 0.5) / nz,
                        ]
                    )
                    vector = world.wind_at(point, now)
                    if float(np.linalg.norm(vector)) < 0.02:
                        continue
                    samples.append(
                        {
                            "p": [round(float(v), 3) for v in point],
                            "v": [round(float(v), 3) for v in vector],
                        }
                    )
        return samples


class _Handler(BaseHTTPRequestHandler):
    viewer: ViewerState = None          # set by the server factory
    protocol_version = "HTTP/1.1"

    # -- plumbing ------------------------------------------------------

    def log_message(self, *args) -> None:
        """Silence the per-request logging; the page polls many times a second."""

    def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_json(self, payload: dict, status: int = 200) -> None:
        self._send(json.dumps(payload).encode("utf-8"), "application/json", status)

    def _serve_file(self, relative: str) -> None:
        # Resolve inside the viewer directory only: this server is bound to
        # localhost, but a path-traversal bug is not worth leaving lying around.
        target = (VIEWER_ROOT / relative.lstrip("/")).resolve()
        try:
            target.relative_to(VIEWER_ROOT.resolve())
        except ValueError:
            self._send_json({"error": "not found"}, 404)
            return
        if not target.is_file():
            self._send_json({"error": f"not found: {relative}"}, 404)
            return
        kind = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self._send(target.read_bytes(), kind)

    # -- routes --------------------------------------------------------

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._serve_file("index.html")
        elif path == "/api/world":
            self._send_json(self.viewer.world_payload())
        elif path == "/api/state":
            self._send_json(self.viewer.state_payload())
        else:
            self._serve_file(path)

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] != "/api/command":
            self._send_json({"error": "not found"}, 404)
            return

        length = int(self.headers.get("Content-Length", 0) or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send_json({"error": "bad json"}, 400)
            return

        command = str(body.get("command", "")).strip()
        if not command:
            self._send_json({"error": "no command"}, 400)
            return

        drone_name = body.get("drone") or self.viewer.simulator.drones[0].name
        try:
            self._send_json(self.viewer.submit(drone_name, command))
        except KeyError:
            self._send_json({"error": f"no drone named {drone_name}"}, 404)


class WebViewer:
    """Serves the live 3D view. Runs on its own thread; never blocks the clock."""

    def __init__(self, simulator: Simulator, host: str = "127.0.0.1", port: int = 8080):
        self.viewer = ViewerState(simulator)
        self.host = host
        self.port = port
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def start(self) -> str:
        handler = type("BoundHandler", (_Handler,), {"viewer": self.viewer})
        # Port 0 lets the OS pick a free one if the requested port is taken.
        for candidate in (self.port, 0):
            try:
                self._server = ThreadingHTTPServer((self.host, candidate), handler)
                break
            except OSError:
                if candidate == 0:
                    raise
        self.port = self._server.server_address[1]
        self._server.daemon_threads = True

        self._thread = threading.Thread(
            target=self._server.serve_forever, name="tello-webviewer", daemon=True
        )
        self._thread.start()
        return self.url

    def open_browser(self) -> None:
        webbrowser.open(self.url)

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def __enter__(self) -> "WebViewer":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
