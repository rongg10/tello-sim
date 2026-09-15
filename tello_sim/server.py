"""A fake Tello on the network.

The real drone is a UDP endpoint that reads plain-text commands on port 8889,
answers `ok`, and shouts its state at port 8890 ten times a second.  That is the
entire interface.  Reproduce it faithfully and every piece of existing flight
code -- a tkinter controller, a notebook, djitellopy itself -- runs against
the simulator with no modification beyond typing a different IP address.

Running the two on one machine has one wrinkle: djitellopy binds port 8889
locally as well as sending to it, so it would collide with this server.
patch.py fixes that in three lines without touching any flight code.

For several drones at once each needs its own address, because djitellopy tells
drones apart by source IP.  On macOS:

    sudo ifconfig lo0 alias 127.0.0.2 up
    sudo ifconfig lo0 alias 127.0.0.3 up

For swarm work that is usually more trouble than it is worth -- use the
in-process SimTello path instead, which has no such limit.
"""

from __future__ import annotations

import queue
import socket
import threading
import time
from dataclasses import dataclass
from typing import Sequence

from .drone import SimDrone
from .protocol import CONTROL_UDP_PORT, STATE_UDP_PORT
from .simulator import Simulator


@dataclass
class Endpoint:
    """One simulated drone's presence on the network."""

    drone: SimDrone
    bind_host: str = "0.0.0.0"
    command_port: int = CONTROL_UDP_PORT
    state_port: int = STATE_UDP_PORT


class TelloServer:
    """Serves one or more simulated drones over UDP."""

    def __init__(
        self,
        simulator: Simulator,
        endpoints: Sequence[Endpoint] | None = None,
        state_targets: Sequence[str] = ("127.0.0.1",),
        verbose: bool = True,
    ):
        self.simulator = simulator
        self.verbose = verbose
        self.endpoints = list(endpoints) if endpoints else [
            Endpoint(drone) for drone in simulator.drones[:1]
        ]

        self._sockets: dict[str, socket.socket] = {}
        self._state_sockets: dict[str, socket.socket] = {}
        self._reply_queues: dict[str, queue.Queue] = {}
        self._clients: dict[str, set[str]] = {}
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()

        # Where to send state before any client has said hello.
        self._default_targets = list(state_targets)

        simulator.on_telemetry(self._broadcast_state)

    # ------------------------------------------------------------------

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    def start(self) -> None:
        bound = []
        for endpoint in self.endpoints:
            name = endpoint.drone.name

            command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            command_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                command_socket.bind((endpoint.bind_host, endpoint.command_port))
            except OSError as exc:
                # Almost always a loopback alias that has not been created. Not
                # a reason to bring the whole session down: the drone still
                # exists and still flies, it just has no address on the network.
                # The 3D viewer does not use UDP at all.
                command_socket.close()
                self._log(
                    f"[sim] {name} has no network address "
                    f"({endpoint.bind_host}:{endpoint.command_port}: {exc.strerror}).\n"
                    f"       It still flies, and the 3D view still shows it. To reach it "
                    f"over UDP:\n"
                    f"           sudo ifconfig lo0 alias {endpoint.bind_host} up"
                )
                continue
            command_socket.settimeout(0.5)
            self._sockets[name] = command_socket
            bound.append(endpoint)

            state_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            state_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._state_sockets[name] = state_socket

            self._reply_queues[name] = queue.Queue()
            self._clients[name] = set()

            receiver = threading.Thread(
                target=self._receive_loop, args=(endpoint,), daemon=True,
                name=f"recv-{name}",
            )
            responder = threading.Thread(
                target=self._reply_loop, args=(endpoint,), daemon=True,
                name=f"reply-{name}",
            )
            receiver.start()
            responder.start()
            self._threads += [receiver, responder]

            self._log(
                f"[sim] {name} listening on {endpoint.bind_host}:{endpoint.command_port}, "
                f"state to :{endpoint.state_port}"
            )

        # Only the endpoints that actually got an address should be broadcasting
        # state, or _broadcast_state would look up sockets that were never made.
        self.endpoints = bound

    def stop(self) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=1.5)
        for sock in list(self._sockets.values()) + list(self._state_sockets.values()):
            try:
                sock.close()
            except OSError:
                pass
        self._threads.clear()

    # ------------------------------------------------------------------

    def _receive_loop(self, endpoint: Endpoint) -> None:
        """Read commands and hand them straight to the simulation.

        Deliberately does not wait for the command to finish: the real drone can
        accept an emergency stop while it is still flying a move, and so must
        this.  Replies are sent from a separate thread, in arrival order.
        """
        name = endpoint.drone.name
        sock = self._sockets[name]

        while not self._stop.is_set():
            try:
                data, address = sock.recvfrom(1518)
            except socket.timeout:
                continue
            except OSError:
                break

            try:
                text = data.decode("utf-8").strip()
            except UnicodeDecodeError:
                continue
            if not text:
                continue

            self._clients[name].add(address[0])
            pending = endpoint.drone.submit(text)

            if pending.expects_response:
                self._reply_queues[name].put((pending, address))
            elif self.verbose:
                self._log(f"[{name}] <- {text}   (no reply, as on the real drone)")

    def _reply_loop(self, endpoint: Endpoint) -> None:
        name = endpoint.drone.name
        sock = self._sockets[name]
        replies = self._reply_queues[name]

        while not self._stop.is_set():
            try:
                pending, address = replies.get(timeout=0.5)
            except queue.Empty:
                continue

            # Wait in wall-clock terms; the simulation is running in realtime
            # here, so a long move blocks the caller exactly as hardware does.
            if not pending.event.wait(timeout=60.0):
                continue

            response = pending.response or "error timeout"
            try:
                sock.sendto(response.encode("utf-8"), address)
            except OSError:
                continue

            if self.verbose:
                took = (
                    ""
                    if pending.finished_at is None
                    else f"  [{pending.finished_at - pending.submitted_at:.2f}s]"
                )
                self._log(f"[{name}] <- {pending.raw}   -> {response}{took}")

    def _broadcast_state(self, drone: SimDrone, sim_time: float) -> None:
        """Push a state packet to whoever is talking to this drone."""
        endpoint = next((e for e in self.endpoints if e.drone is drone), None)
        if endpoint is None:
            return

        sock = self._state_sockets.get(drone.name)
        if sock is None:
            return

        targets = self._clients.get(drone.name) or self._default_targets
        packet = drone.state_packet(self.simulator.world).encode("utf-8")
        for host in targets:
            try:
                sock.sendto(packet, (host, endpoint.state_port))
            except OSError:
                pass

    # ------------------------------------------------------------------

    def serve_forever(self) -> None:
        """Run until interrupted, printing a line whenever anything happens."""
        self.start()
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            self._log("\n[sim] stopping")
        finally:
            self.stop()


def serve(
    simulator: Simulator,
    hosts: Sequence[str] | None = None,
    command_port: int = CONTROL_UDP_PORT,
    state_port: int = STATE_UDP_PORT,
    verbose: bool = True,
) -> TelloServer:
    """Put every drone in the simulator on the network and start serving.

    With one drone the default bind of 0.0.0.0:8889 is all you need.  With more,
    pass one host per drone (see the note at the top of this file).
    """
    drones = simulator.drones
    if hosts is None:
        hosts = ["0.0.0.0"] + [f"127.0.0.{i + 2}" for i in range(len(drones) - 1)]
    if len(hosts) < len(drones):
        raise ValueError(
            f"{len(drones)} drones need {len(drones)} bind addresses, got {len(hosts)}"
        )

    endpoints = [
        Endpoint(drone, bind_host=host, command_port=command_port, state_port=state_port)
        for drone, host in zip(drones, hosts)
    ]
    server = TelloServer(simulator, endpoints, verbose=verbose)
    server.start()
    return server
