"""Let the real djitellopy library talk to the simulator on this machine.

The problem, in one sentence: djitellopy binds UDP port 8889 locally *and*
sends to port 8889 on the drone, which is fine when the drone is a separate
device on the network and a collision when it is another process on the same
laptop.

The fix is to hand djitellopy a socket on an ephemeral port before it creates
one of its own.  Replies come back to whatever port the command was sent from,
so nothing else has to change: the library still demultiplexes by source IP,
still receives state on 8890, still times out the way it always did.

Usage, before any Tello object is constructed:

    from tello_sim.patch import use_simulator
    use_simulator()

    from djitellopy import Tello
    drone = Tello(host="127.0.0.1")
    drone.connect()

Or, without editing a single line of existing code, run the launcher:

    python run_gui_sim.py
"""

from __future__ import annotations

import socket
import threading

_patched = False


def use_simulator(verbose: bool = True) -> None:
    """Prepare djitellopy to share this machine with a simulated drone.

    Safe to call more than once. Has no effect on how the library behaves when
    talking to real hardware, so a script that calls this still flies a real
    Tello if you point it at 192.168.10.1.
    """
    global _patched
    if _patched:
        return

    try:
        import djitellopy.tello as dt
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "djitellopy is not installed. Run: pip install -r requirements.txt"
        ) from exc

    if dt.threads_initialized:
        raise RuntimeError(
            "use_simulator() must be called before the first Tello object is created."
        )

    # An ephemeral local port instead of 8889, leaving 8889 free for the
    # simulator to listen on.
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    client_socket.bind(("", 0))
    dt.client_socket = client_socket

    threading.Thread(
        target=dt.Tello.udp_response_receiver, daemon=True, name="tello-responses"
    ).start()
    threading.Thread(
        target=dt.Tello.udp_state_receiver, daemon=True, name="tello-state"
    ).start()

    # Tell the library its sockets are already up, so __init__ does not make
    # its own and collide with the simulator.
    dt.threads_initialized = True
    _patched = True

    if verbose:
        port = client_socket.getsockname()[1]
        print(
            f"[patch] djitellopy will send from local port {port} "
            f"instead of binding 8889, leaving it free for the simulator.",
            flush=True,
        )
