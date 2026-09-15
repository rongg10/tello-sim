"""The API reference the interactive console prints.

One entry per method you can call on a Tello, written for someone who has not
used the library before: what it does, what units its arguments are in, what
the firmware will refuse, and what raw SDK line it turns into on the wire.

Kept as data rather than as docstrings because docstrings answer "what does
this function do" and the useful question at the prompt is "what can I type
next, and what will the drone refuse".  The ranges here are the same ones
`drone.py` enforces, which are the same ones the real firmware enforces -- so
an error you see in the simulator is an error you would have seen in the air.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Entry:
    """One callable, documented."""

    name: str
    signature: str
    group: str
    summary: str
    detail: str = ""
    sdk: str = ""
    example: str = ""
    limits: str = ""
    hardware: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)

    @property
    def call(self) -> str:
        return f"{self.name}{self.signature}"


GROUPS = [
    ("connect", "Getting the drone listening"),
    ("flight", "Taking off and landing"),
    ("move", "Straight-line moves"),
    ("rotate", "Turning on the spot"),
    ("flip", "Flips"),
    ("position", "Moving to a point"),
    ("rc", "Stick input"),
    ("pads", "Mission pads (Tello EDU)"),
    ("settings", "Settings"),
    ("read", "Reading telemetry"),
    ("simonly", "Things only the simulator can tell you"),
]


ENTRIES: list[Entry] = [
    # ------------------------------------------------------------------
    # connect
    # ------------------------------------------------------------------
    Entry(
        name="connect",
        signature="()",
        group="connect",
        summary="Put the drone into SDK mode. Nothing else works until this runs.",
        detail=(
            "Sends the single word `command`. The Tello ignores every other "
            "instruction until it has seen it, and answers `error Not in SDK "
            "mode` if you try. The console runs this for you at startup; you "
            "only need it again after `end()`."
        ),
        sdk="command",
        example="connect()",
    ),
    Entry(
        name="end",
        signature="()",
        group="connect",
        summary="Land if still flying, then close the connection.",
        detail=(
            "The tidy way to finish. On real hardware it also stops the video "
            "stream and the background threads."
        ),
        example="end()",
    ),
    # ------------------------------------------------------------------
    # flight
    # ------------------------------------------------------------------
    Entry(
        name="takeoff",
        signature="()",
        group="flight",
        summary="Take off and hover at about 80 cm.",
        detail=(
            "Blocks until the drone is up -- roughly 3 seconds. The height it "
            "settles at is fixed by the firmware and is not a parameter; climb "
            "afterwards with move_up() if you want to be higher."
        ),
        sdk="takeoff",
        example="takeoff()",
        limits="Refused below 10% battery, or if already flying.",
    ),
    Entry(
        name="land",
        signature="()",
        group="flight",
        summary="Descend and switch the motors off.",
        detail="Blocks until the drone is down. Safe to call when already landed.",
        sdk="land",
        example="land()",
    ),
    Entry(
        name="emergency",
        signature="()",
        group="flight",
        summary="Cut the motors immediately. The drone falls.",
        detail=(
            "Not a landing -- power to the rotors stops where it is. Use it "
            "when something has gone wrong and you would rather have a broken "
            "propeller than a broken anything else. Like the real firmware, "
            "this command sends no reply."
        ),
        sdk="emergency",
        example="emergency()",
    ),
    # ------------------------------------------------------------------
    # move
    # ------------------------------------------------------------------
    Entry(
        name="move_forward",
        signature="(x)",
        group="move",
        summary="Fly forward x centimetres, in the direction the nose is pointing.",
        detail=(
            "Every move is *relative* to where the drone is now and which way "
            "it is facing -- there is no 'go to this spot in the room' command "
            "in the SDK. That is the single most important fact about this API: "
            "if wind pushes the drone half a metre sideways, every later move "
            "starts from the new place and the offset is never recovered."
        ),
        sdk="forward <x>",
        example="move_forward(50)",
        limits="20 to 500 cm. Anything outside that is refused, not clamped.",
        hardware="Lands a few centimetres off, and the error accumulates over a run.",
    ),
    Entry(
        name="move_back",
        signature="(x)",
        group="move",
        summary="Fly backwards x centimetres. The drone does not turn round.",
        sdk="back <x>",
        example="move_back(40)",
        limits="20 to 500 cm.",
    ),
    Entry(
        name="move_left",
        signature="(x)",
        group="move",
        summary="Slide left x centimetres, nose unchanged.",
        sdk="left <x>",
        example="move_left(30)",
        limits="20 to 500 cm.",
    ),
    Entry(
        name="move_right",
        signature="(x)",
        group="move",
        summary="Slide right x centimetres, nose unchanged.",
        sdk="right <x>",
        example="move_right(30)",
        limits="20 to 500 cm.",
    ),
    Entry(
        name="move_up",
        signature="(x)",
        group="move",
        summary="Climb x centimetres.",
        sdk="up <x>",
        example="move_up(50)",
        limits="20 to 500 cm, and the ceiling stops you before 500 does.",
    ),
    Entry(
        name="move_down",
        signature="(x)",
        group="move",
        summary="Descend x centimetres.",
        sdk="down <x>",
        example="move_down(30)",
        limits="20 to 500 cm.",
    ),
    Entry(
        name="move",
        signature="(direction, x)",
        group="move",
        summary="The six moves above, with the direction as a string.",
        detail=(
            "direction is one of 'forward', 'back', 'left', 'right', 'up', "
            "'down'. Useful when the direction is coming from a variable or a "
            "policy rather than being written out."
        ),
        sdk="<direction> <x>",
        example="move('forward', 50)",
        limits="20 to 500 cm.",
    ),
    # ------------------------------------------------------------------
    # rotate
    # ------------------------------------------------------------------
    Entry(
        name="rotate_clockwise",
        signature="(x)",
        group="rotate",
        summary="Turn x degrees clockwise, seen from above.",
        detail=(
            "Turning changes what 'forward' means for every later move, which "
            "is the usual source of a flight going somewhere unexpected."
        ),
        sdk="cw <x>",
        example="rotate_clockwise(90)",
        limits="1 to 360 degrees.",
        hardware="A degree or two of error each time, and it accumulates.",
    ),
    Entry(
        name="rotate_counter_clockwise",
        signature="(x)",
        group="rotate",
        summary="Turn x degrees anticlockwise.",
        sdk="ccw <x>",
        example="rotate_counter_clockwise(45)",
        limits="1 to 360 degrees.",
    ),
    # ------------------------------------------------------------------
    # flip
    # ------------------------------------------------------------------
    Entry(
        name="flip",
        signature="(direction)",
        group="flip",
        summary="Barrel roll. direction is 'l', 'r', 'f' or 'b'.",
        detail=(
            "Costs height and a noticeable slice of battery, and needs clear "
            "air on every side."
        ),
        sdk="flip <l|r|f|b>",
        example="flip('l')",
        limits="Refused below 50% battery -- the real firmware does the same.",
    ),
    Entry(
        name="flip_left",
        signature="()",
        group="flip",
        summary="Flip to the left.",
        sdk="flip l",
        example="flip_left()",
        limits="Needs 50% battery.",
    ),
    Entry(
        name="flip_right",
        signature="()",
        group="flip",
        summary="Flip to the right.",
        sdk="flip r",
        example="flip_right()",
        limits="Needs 50% battery.",
    ),
    Entry(
        name="flip_forward",
        signature="()",
        group="flip",
        summary="Flip forwards.",
        sdk="flip f",
        example="flip_forward()",
        limits="Needs 50% battery.",
    ),
    Entry(
        name="flip_back",
        signature="()",
        group="flip",
        summary="Flip backwards.",
        sdk="flip b",
        example="flip_back()",
        limits="Needs 50% battery.",
    ),
    # ------------------------------------------------------------------
    # position
    # ------------------------------------------------------------------
    Entry(
        name="go_xyz_speed",
        signature="(x, y, z, speed)",
        group="position",
        summary="Fly to an offset from here, all three axes at once.",
        detail=(
            "Axes are the drone's own, in centimetres: x forward, y left, "
            "z up. Negative values go the other way. speed is in cm/s. One "
            "diagonal move instead of three separate legs, which is both "
            "quicker and less error than chaining move_forward/left/up."
        ),
        sdk="go <x> <y> <z> <speed>",
        example="go_xyz_speed(100, 50, 0, 40)",
        limits=(
            "-500 to 500 cm per axis, speed 10 to 100 cm/s. Refused when all "
            "three offsets are within 20 cm of zero -- the firmware cannot "
            "tell a move that small from noise."
        ),
    ),
    Entry(
        name="curve_xyz_speed",
        signature="(x1, y1, z1, x2, y2, z2, speed)",
        group="position",
        summary="Fly an arc through one point and on to a second.",
        detail=(
            "Both points are offsets from where the drone is now, same axes as "
            "go_xyz_speed. The firmware fits an arc through the three points, "
            "and rejects the request if that arc is too tight."
        ),
        sdk="curve <x1> <y1> <z1> <x2> <y2> <z2> <speed>",
        example="curve_xyz_speed(50, 50, 0, 100, 0, 0, 30)",
        limits="-500 to 500 cm per axis, speed 10 to 60 cm/s.",
    ),
    # ------------------------------------------------------------------
    # rc
    # ------------------------------------------------------------------
    Entry(
        name="send_rc_control",
        signature="(left_right, forward_back, up_down, yaw)",
        group="rc",
        summary="Hold the sticks at these positions. Returns immediately.",
        detail=(
            "Four channels, each -100 to 100, meaning a fraction of full stick "
            "rather than any distance. Unlike every other command this one does "
            "not block and does not stop: the drone keeps doing it until you "
            "send different numbers, so a control loop must send this "
            "repeatedly and send zeros to stop. This is the command an "
            "autonomous controller actually uses; the move_* family is for "
            "scripted flight."
        ),
        sdk="rc <lr> <fb> <ud> <yaw>",
        example="send_rc_control(0, 50, 0, 0)   # then later: send_rc_control(0, 0, 0, 0)",
        limits="-100 to 100 on each channel. No reply is sent, matching the firmware.",
    ),
    # ------------------------------------------------------------------
    # mission pads
    # ------------------------------------------------------------------
    Entry(
        name="enable_mission_pads",
        signature="()",
        group="pads",
        summary="Start looking for mission pads with the downward camera.",
        detail=(
            "Mission pads are the printed mats that come with the Tello EDU. "
            "They are the only way this drone gets an absolute position fix -- "
            "everything else in the SDK is relative and drifts."
        ),
        sdk="mon",
        example="enable_mission_pads()",
    ),
    Entry(
        name="disable_mission_pads",
        signature="()",
        group="pads",
        summary="Stop looking for pads.",
        sdk="moff",
        example="disable_mission_pads()",
    ),
    Entry(
        name="set_mission_pad_detection_direction",
        signature="(x)",
        group="pads",
        summary="Which camera looks for pads: 0 down, 1 forward, 2 both.",
        detail="Only the downward camera is simulated, so 0 is the honest setting.",
        sdk="mdirection <x>",
        example="set_mission_pad_detection_direction(0)",
        limits="0, 1 or 2. Mission pads must be enabled first.",
        aliases=("set_mission_pad_direction",),
    ),
    Entry(
        name="go_xyz_speed_mid",
        signature="(x, y, z, speed, mid)",
        group="pads",
        summary="Fly to a point measured from pad `mid`, not from here.",
        detail=(
            "The one absolute move in the whole SDK. The offset is measured in "
            "the pad's own frame, so repeating this command puts the drone in "
            "the same place every time however far it has drifted -- which is "
            "the fix for accumulated error."
        ),
        sdk="go <x> <y> <z> <speed> m<mid>",
        example="go_xyz_speed_mid(0, 0, 100, 40, 1)",
        limits=(
            "Pad ids 1 to 8. The pad has to be visible: between 30 cm and "
            "120 cm above it, as on the real hardware."
        ),
    ),
    Entry(
        name="curve_xyz_speed_mid",
        signature="(x1, y1, z1, x2, y2, z2, speed, mid)",
        group="pads",
        summary="An arc, with both points measured from a pad.",
        sdk="curve <x1> <y1> <z1> <x2> <y2> <z2> <speed> m<mid>",
        example="curve_xyz_speed_mid(50, 0, 0, 100, 50, 0, 30, 1)",
        limits="Speed 10 to 60 cm/s, pad ids 1 to 8.",
    ),
    Entry(
        name="go_xyz_speed_yaw_mid",
        signature="(x, y, z, speed, yaw, mid1, mid2)",
        group="pads",
        summary="Fly to a point over pad mid2, then turn to face `yaw` in its frame.",
        detail=(
            "The `jump` command: leave one pad, arrive over another, ending at "
            "a known position *and* a known heading. Two of these in a row keep "
            "a long flight from drifting at all."
        ),
        sdk="jump <x> <y> <z> <speed> <yaw> m<mid1> m<mid2>",
        example="go_xyz_speed_yaw_mid(0, 0, 100, 40, 0, 1, 2)",
        limits="Yaw -360 to 360, pad ids 1 to 8.",
    ),
    # ------------------------------------------------------------------
    # settings
    # ------------------------------------------------------------------
    Entry(
        name="set_speed",
        signature="(x)",
        group="settings",
        summary="Set the cruise speed used by every move_* command, in cm/s.",
        detail=(
            "go_xyz_speed and curve_xyz_speed carry their own speed and ignore "
            "this. Faster is not free: the drone overshoots more, and battery "
            "goes down with manoeuvring rather than with time alone."
        ),
        sdk="speed <x>",
        example="set_speed(60)",
        limits="10 to 100 cm/s.",
    ),
    Entry(
        name="streamon",
        signature="()",
        group="settings",
        summary="Start the video stream. Accepted here, but no frames are produced.",
        detail=(
            "The simulator has no camera: the command answers `ok` so existing "
            "code runs unchanged, and get_frame_read() will not give you "
            "anything. The 3D view does have a drone's-eye camera; wiring it "
            "back through the SDK is the piece that is missing."
        ),
        sdk="streamon",
        example="streamon()",
    ),
    Entry(
        name="streamoff",
        signature="()",
        group="settings",
        summary="Stop the video stream.",
        sdk="streamoff",
        example="streamoff()",
    ),
    # ------------------------------------------------------------------
    # read
    # ------------------------------------------------------------------
    Entry(
        name="get_battery",
        signature="()",
        group="read",
        summary="Battery percentage, 0 to 100.",
        detail=(
            "Read from the 10 Hz state broadcast, so it costs nothing and does "
            "not wait for the drone. Drains faster when manoeuvring, and faster "
            "still when holding a tilt against wind. Below 5% the drone lands "
            "itself whatever you have asked for."
        ),
        example="get_battery()",
    ),
    Entry(
        name="get_height",
        signature="()",
        group="read",
        summary="Height above the take-off point, in centimetres.",
        detail="Coarse: the firmware reports it in decimetres, so it moves in steps of 10.",
        example="get_height()",
    ),
    Entry(
        name="get_distance_tof",
        signature="()",
        group="read",
        summary="Downward time-of-flight range in cm -- distance to whatever is underneath.",
        detail=(
            "Not height above the floor. Fly over a desk and this jumps, "
            "because it measures to the desk. Worth watching in "
            "scenarios/03_obstacles.yaml before any controller trusts it."
        ),
        example="get_distance_tof()",
    ),
    Entry(
        name="get_flight_time",
        signature="()",
        group="read",
        summary="Seconds the motors have been running.",
        example="get_flight_time()",
    ),
    Entry(
        name="get_barometer",
        signature="()",
        group="read",
        summary="Barometric altitude in cm. Noisier than get_height and drifts.",
        example="get_barometer()",
    ),
    Entry(
        name="get_yaw",
        signature="()",
        group="read",
        summary="Heading in degrees, relative to where the drone was switched on.",
        detail="Not a compass bearing -- it is relative to power-on, and it drifts.",
        example="get_yaw()",
    ),
    Entry(
        name="get_pitch",
        signature="()",
        group="read",
        summary="Nose-up angle in degrees.",
        detail=(
            "Tilt is how a quadrotor accelerates, so a non-zero pitch in level "
            "flight means it is pushing against something -- usually wind."
        ),
        example="get_pitch()",
    ),
    Entry(
        name="get_roll",
        signature="()",
        group="read",
        summary="Bank angle in degrees.",
        example="get_roll()",
    ),
    Entry(
        name="get_speed_x",
        signature="()",
        group="read",
        summary="Ground speed along x, in cm/s.",
        example="get_speed_x()",
    ),
    Entry(
        name="get_speed_y",
        signature="()",
        group="read",
        summary="Ground speed along y, in cm/s.",
        example="get_speed_y()",
    ),
    Entry(
        name="get_speed_z",
        signature="()",
        group="read",
        summary="Vertical speed, in cm/s.",
        example="get_speed_z()",
    ),
    Entry(
        name="get_temperature",
        signature="()",
        group="read",
        summary="Mean of the two onboard temperature sensors, in Celsius.",
        example="get_temperature()",
    ),
    Entry(
        name="get_mission_pad_id",
        signature="()",
        group="read",
        summary="Id of the pad underneath, or -1 for none.",
        detail="Needs enable_mission_pads() and a height between 30 and 120 cm.",
        example="get_mission_pad_id()",
    ),
    Entry(
        name="get_current_state",
        signature="()",
        group="read",
        summary="The whole state broadcast as a dict, all fields at once.",
        detail=(
            "Keys are the raw SDK names: mid, x, y, z, pitch, roll, yaw, vgx, "
            "vgy, vgz, templ, temph, tof, h, bat, baro, time, agx, agy, agz. "
            "Every get_* above is a lookup in here."
        ),
        example="get_current_state()",
    ),
    Entry(
        name="get_state_field",
        signature="(key)",
        group="read",
        summary="One field of the state broadcast, by its raw SDK name.",
        example="get_state_field('bat')",
    ),
    Entry(
        name="query_battery",
        signature="()",
        group="read",
        summary="Ask the drone for the battery, instead of reading the broadcast.",
        detail=(
            "The query_* family sends a real command and waits for the answer, "
            "so it costs a round trip and can time out. The get_* family reads "
            "the state the drone is already broadcasting and cannot fail. "
            "Prefer get_*; query_* exists because some fields are only "
            "available that way. Also: query_attitude, query_barometer, "
            "query_distance_tof, query_flight_time, query_height, query_speed, "
            "query_wifi_signal_noise_ratio, query_sdk_version, "
            "query_serial_number."
        ),
        sdk="battery?",
        example="query_battery()",
    ),
    # ------------------------------------------------------------------
    # simulator only
    # ------------------------------------------------------------------
    Entry(
        name="true_position",
        signature="()",
        group="simonly",
        summary="Exact position in the room, in metres. Not available on hardware.",
        detail=(
            "Use it to score a run -- how far did the drone end up from where "
            "it was told to go -- never to fly one. A controller that reads "
            "this works perfectly here and not at all on the real drone."
        ),
        example="true_position()",
    ),
    Entry(
        name="true_yaw_deg",
        signature="()",
        group="simonly",
        summary="Exact heading in degrees. Not available on hardware.",
        example="true_yaw_deg()",
    ),
    Entry(
        name="sleep",
        signature="(seconds)",
        group="simonly",
        summary="Wait, in simulated time. Use this rather than time.sleep.",
        detail=(
            "In a real-time run the two are the same thing. In a fast run "
            "time.sleep just makes you wait while the simulation stands still."
        ),
        example="sleep(2)",
    ),
]


BY_NAME: dict[str, Entry] = {}
for _entry in ENTRIES:
    BY_NAME[_entry.name] = _entry
    for _alias in _entry.aliases:
        BY_NAME[_alias] = _entry


# The raw SDK verbs, for the console's "you typed a wire command" path.
SDK_VERBS = {
    "command", "takeoff", "land", "emergency", "stop",
    "up", "down", "left", "right", "forward", "back",
    "cw", "ccw", "flip", "go", "curve", "jump", "rc",
    "speed", "mon", "moff", "mdirection",
    "streamon", "streamoff", "wifi", "ap", "port",
    "setfps", "setbitrate", "setresolution",
    "battery?", "speed?", "time?", "sn?", "sdk?", "wifi?",
    "height?", "tof?", "baro?", "temp?", "attitude?", "acceleration?",
}


# SDK verbs that are a complete line on their own. Everything else needs an
# argument, so a bare `forward` is a typo rather than a command.
STANDALONE_SDK = {
    "command", "takeoff", "land", "emergency", "stop",
    "mon", "moff", "streamon", "streamoff",
} | {verb for verb in SDK_VERBS if verb.endswith("?")}


def entries_in(group: str) -> list[Entry]:
    return [entry for entry in ENTRIES if entry.group == group]
