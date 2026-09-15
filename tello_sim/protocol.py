"""The Tello SDK 2.0 wire format, in one place.

Kept free of any dependency on djitellopy so that batch research runs need
nothing but numpy, while the UDP path stays byte-compatible with the real
library.  Both paths parse state with the function below, so a bug in the
packet format shows up identically in both.
"""

from __future__ import annotations

from typing import Dict, Union

# Field types, matching the real firmware and djitellopy's own converters.
INT_STATE_FIELDS = (
    "mid", "x", "y", "z",
    "pitch", "roll", "yaw",
    "vgx", "vgy", "vgz",
    "templ", "temph",
    "tof", "h", "bat", "time",
)
FLOAT_STATE_FIELDS = ("baro", "agx", "agy", "agz")

STATE_FIELD_CONVERTERS: Dict[str, type] = {k: int for k in INT_STATE_FIELDS}
STATE_FIELD_CONVERTERS.update({k: float for k in FLOAT_STATE_FIELDS})

CONTROL_UDP_PORT = 8889
STATE_UDP_PORT = 8890
VIDEO_UDP_PORT = 11111


def parse_state(state: str) -> Dict[str, Union[int, float, str]]:
    """Turn a raw state string into a dictionary.

    Deliberately identical in behaviour to djitellopy.Tello.parse_state,
    including skipping fields it cannot convert rather than raising.
    """
    state = state.strip()
    if state == "ok":
        return {}

    parsed: Dict[str, Union[int, float, str]] = {}
    for field in state.split(";"):
        split = field.split(":")
        if len(split) < 2:
            continue
        key, raw = split[0], split[1]
        converter = STATE_FIELD_CONVERTERS.get(key)
        if converter is not None:
            try:
                parsed[key] = converter(raw)
            except ValueError:
                continue
        else:
            parsed[key] = raw
    return parsed
