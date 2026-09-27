"""MuJoCo as the simulator's physics engine.

What changed and why
--------------------
The simulator used to integrate a point mass by hand: one linear ODE, a
semi-implicit Euler step, and collisions resolved by pushing a sphere out of
whatever it overlapped.  That was enough to study wind, which is the thing the
research needs, but it could not answer anything about *contact* -- what a
bump does to attitude, what happens when a drone clips a doorframe, how a
swarm behaves when two airframes actually touch -- and it could not carry a
moving obstacle at all.

MuJoCo replaces both halves of that.  It owns the rigid-body state and the
contact solver; this module owns the translation between the Tello SDK's idea
of flight and MuJoCo's idea of forces.

The model is a quadrotor, not a dot
-----------------------------------
A real quadrotor cannot push sideways.  Its rotors only push along the body's
own up-axis, so to accelerate sideways it must first *tilt*, and the tilt takes
time.  The old point-mass model skipped that: it applied the commanded
acceleration directly and reported a cosmetic tilt afterwards, computed from
the acceleration it had already applied.

Here the tilt is real and causal:

    thrust  = |m * (a_cmd + g)|      applied along the body's ACTUAL up-axis
    torque  = geometric attitude controller driving that axis toward
              the direction the thrust was supposed to point

So a commanded sideways move now produces a tilt, and the tilt produces the
sideways acceleration, in that order.  Attitude is a state with dynamics rather
than a number derived for the telemetry packet.  A collision that knocks the
airframe over is something the drone has to recover from.

What is deliberately NOT modelled
---------------------------------
Four individual rotors.  The Tello SDK never exposes motor commands, so rotor
thrust curves, blade inertia and motor time constants are parameters nobody can
measure through the interface we actually have.  Total thrust plus body torque
has the same observable consequences and every constant in it can be calibrated
from a flight log.

Coordinate conventions
----------------------
Shared with the rest of the package: x forward, y left, z up, metres, radians,
yaw zero means the nose points along +x.  MuJoCo uses the same right-handed
convention, so no axis juggling is needed anywhere in this file.

One gotcha worth writing down: for a free joint MuJoCo reports *linear* velocity
in world axes but *angular* velocity in body axes.  Mixing those up produces a
drone that yaws the wrong way only while it is tilted, which is a miserable bug
to find.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np

from .dynamics import GRAVITY, DroneState, wrap_angle

if TYPE_CHECKING:  # pragma: no cover - import cost only matters at runtime
    from .drone import SimDrone
    from .world import World

try:
    import mujoco
except ImportError as exc:  # pragma: no cover - environment problem, not logic
    raise ImportError(
        "tello_sim now uses MuJoCo for physics.\n"
        "Install it with:  pip install mujoco\n"
        "\n"
        "MuJoCo publishes arm64 wheels for Python 3.10 and newer. If pip tries to\n"
        "build from source and complains about MUJOCO_PATH, the interpreter is too\n"
        "old -- check `python -V`, and rebuild the virtual environment on 3.12."
    ) from exc


# ----------------------------------------------------------------------
# Quaternion helpers
#
# MuJoCo stores quaternions w-first. numpy has no quaternion type and pulling
# in scipy for four lines of algebra is not worth the dependency.
# ----------------------------------------------------------------------


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix whose COLUMNS are the body axes in world coordinates.

    Only needed where no matrix is already to hand.  Inside the step loop, read
    `data.xmat` instead: MuJoCo has computed it anyway, and recomputing it here
    was costing more than the entire physics step.
    """
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def yaw_from_matrix(rotation: np.ndarray) -> float:
    """Heading of the body's forward axis, projected onto the ground plane.

    Taken from where the nose actually points rather than from a Z-Y-X Euler
    decomposition, which goes singular when the airframe is tipped on its side
    -- exactly the case a contact simulation is there to produce.
    """
    return float(np.arctan2(rotation[1, 0], rotation[0, 0]))


def yaw_from_quat(q: np.ndarray) -> float:
    """Heading from a quaternion, for callers holding one rather than a matrix."""
    return yaw_from_matrix(quat_to_matrix(q))


def yaw_quat(yaw: float) -> np.ndarray:
    half = 0.5 * float(yaw)
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])


def vee(m: np.ndarray) -> np.ndarray:
    """Inverse of the skew-symmetric hat map."""
    return np.array([m[2, 1], m[0, 2], m[1, 0]])


# ----------------------------------------------------------------------
# Model building
# ----------------------------------------------------------------------


# MuJoCo decides whether two geoms may touch with a bitmask test:
#
#     contact  <=>  (contype1 & conaffinity2)  or  (contype2 & conaffinity1)
#
# The masks below exist so that switching collisions off makes a drone a ghost
# to the walls, the obstacles and the other drones -- but NOT to the floor. A
# drone that falls through the floor the moment you disable collisions is
# useless for an ablation, because it never gets as far as taking off.
_MASK_WORLD = 1          # walls, ceiling, obstacles
_MASK_FLOOR_TYPE = 2     # the floor's own type bit
_MASK_FLOOR_AFF = 5      # the floor accepts world bodies and ghosts alike
_MASK_DRONE_SOLID = (5, 3)   # (contype, conaffinity): touches everything
_MASK_DRONE_GHOST = (4, 2)   # touches the floor only, not even other drones


def _vec(values: Sequence[float]) -> str:
    return " ".join(f"{float(v):.6g}" for v in values)


def _name(raw: str) -> str:
    """Make a string safe to use as an MJCF element name.

    Scenario files are written by people, so obstacle names contain spaces,
    slashes and the occasional apostrophe. MuJoCo needs those to be unique
    tokens, and a name collision is reported as a compile error a long way from
    the YAML line that caused it.
    """
    cleaned = "".join(c if c.isalnum() or c in "_-" else "_" for c in raw)
    return cleaned or "unnamed"


class _NameAllocator:
    """Hands out unique MJCF names, remembering what each one came from."""

    def __init__(self) -> None:
        self._taken: set[str] = set()

    def take(self, raw: str) -> str:
        base = _name(raw)
        candidate = base
        suffix = 2
        while candidate in self._taken:
            candidate = f"{base}_{suffix}"
            suffix += 1
        self._taken.add(candidate)
        return candidate


def build_mjcf(world: "World", drones: Sequence["SimDrone"], timestep: float) -> tuple[str, dict]:
    """Write the MJCF XML for a world and the drones flying in it.

    Returns the XML and a map from our names to MJCF names, because the two
    differ whenever a scenario uses a name MuJoCo will not accept.

    Room walls are real geoms rather than a clamp on position.  The difference
    shows up the moment wind pins a drone against one: clamping silently
    deletes the momentum, while a contact lets the airframe scrape along the
    surface and tells the benchmark it happened.
    """
    names = _NameAllocator()
    mapping: dict[str, str] = {}

    lower = np.asarray(world.bounds_lower, dtype=float)
    upper = np.asarray(world.bounds_upper, dtype=float)
    centre = 0.5 * (lower + upper)
    half = 0.5 * (upper - lower)
    # Walls need thickness; a zero-thickness plane lets a fast body tunnel.
    thickness = 0.05

    parts: list[str] = []
    parts.append('<mujoco model="tello_sim">')
    # implicitfast, not RK4. RK4 is the better integrator for smooth flight but
    # it evaluates the dynamics four times a step and handles contact badly, and
    # contact is the entire reason this simulator now has an engine under it.
    parts.append(
        f'  <option timestep="{timestep:.6g}" gravity="0 0 {-GRAVITY:.6g}" '
        f'integrator="implicitfast" cone="elliptic"/>'
    )
    parts.append('  <compiler angle="radian" autolimits="true"/>')
    # A contact must not be stiffer than the timestep can resolve, or the solver
    # answers a hard tap with a launch across the room. Tying the time constant
    # to dt means changing the step rate cannot silently break the physics.
    solref = max(2.0 * timestep, 0.004)
    parts.append("  <default>")
    parts.append(
        f'    <geom solref="{solref:.6g} 1" solimp="0.9 0.98 0.001" '
        f'friction="0.9 0.02 0.002" condim="4" '
        f'contype="{_MASK_WORLD}" conaffinity="{_MASK_WORLD}"/>'
    )
    parts.append("  </default>")
    parts.append("  <worldbody>")

    # --- The room ---------------------------------------------------------
    parts.append(
        f'    <geom name="floor" type="box" '
        f'pos="{_vec([centre[0], centre[1], lower[2] - thickness])}" '
        f'size="{_vec([half[0] + thickness, half[1] + thickness, thickness])}" '
        f'contype="{_MASK_FLOOR_TYPE}" conaffinity="{_MASK_FLOOR_AFF}" '
        f'rgba="0.82 0.84 0.86 1"/>'
    )
    parts.append(
        f'    <geom name="ceiling" type="box" '
        f'pos="{_vec([centre[0], centre[1], upper[2] + thickness])}" '
        f'size="{_vec([half[0] + thickness, half[1] + thickness, thickness])}" '
        f'rgba="0.82 0.84 0.86 0.25"/>'
    )
    wall_specs = [
        ("wall_x_min", [lower[0] - thickness, centre[1], centre[2]], [thickness, half[1], half[2]]),
        ("wall_x_max", [upper[0] + thickness, centre[1], centre[2]], [thickness, half[1], half[2]]),
        ("wall_y_min", [centre[0], lower[1] - thickness, centre[2]], [half[0], thickness, half[2]]),
        ("wall_y_max", [centre[0], upper[1] + thickness, centre[2]], [half[0], thickness, half[2]]),
    ]
    for wall_name, pos, size in wall_specs:
        names.take(wall_name)
        parts.append(
            f'    <geom name="{wall_name}" type="box" pos="{_vec(pos)}" '
            f'size="{_vec(size)}" rgba="0.75 0.78 0.82 0.18"/>'
        )

    # --- Mission pads -----------------------------------------------------
    # Visual only, and explicitly non-colliding: a printed pad is a sheet of
    # paper on the floor, not a kerb to trip over.
    for pad in world.mission_pads:
        pad_name = names.take(f"pad_{pad.pad_id}")
        parts.append(
            f'    <geom name="{pad_name}" type="box" '
            f'contype="0" conaffinity="0" '
            f'pos="{_vec([pad.center[0], pad.center[1], lower[2] + 0.001])}" '
            f'quat="{_vec(yaw_quat(pad.yaw))}" '
            f'size="{_vec([pad.size / 2, pad.size / 2, 0.001])}" '
            f'rgba="0.95 0.75 0.15 1"/>'
        )

    # --- Obstacles --------------------------------------------------------
    for obstacle in world.obstacles:
        mjcf_name = names.take(obstacle.name)
        mapping[obstacle.name] = mjcf_name
        parts.extend(_obstacle_xml(obstacle, mjcf_name))

    # --- Drones -----------------------------------------------------------
    for drone in drones:
        body_name = names.take(drone.name)
        mapping[drone.name] = body_name
        spec = drone.spec
        start = np.asarray(drone.state.position, dtype=float)
        # Spawn resting on its own legs rather than with the body centre on the
        # floor, otherwise the first solver step has to resolve a half-buried
        # airframe and launches it.
        start_z = max(float(start[2]), float(lower[2]) + spec.body_half_height)
        ixx, iyy, izz = spec.inertia_diag()
        parts.append(
            f'    <body name="{body_name}" '
            f'pos="{_vec([start[0], start[1], start_z])}" '
            f'quat="{_vec(yaw_quat(drone.state.yaw))}">'
        )
        parts.append(f'      <freejoint name="{body_name}_free"/>')
        parts.append(
            f'      <inertial pos="0 0 0" mass="{spec.mass:.6g}" '
            f'diaginertia="{_vec([ixx, iyy, izz])}"/>'
        )
        # A flat cylinder, not a sphere. A Tello with prop guards is a disc:
        # it is the disc that catches a doorframe, and a disc resting on the
        # floor puts its centre 2 cm up rather than a whole body-radius, which
        # is what lets a landed drone report a height of zero.
        parts.append(
            f'      <geom name="{body_name}_hull" type="cylinder" '
            f'size="{spec.radius:.6g} {spec.body_half_height:.6g}" '
            f'rgba="0.15 0.16 0.18 1"/>'
        )
        parts.append("    </body>")

    parts.append("  </worldbody>")
    parts.append("</mujoco>")
    return "\n".join(parts), mapping


def _obstacle_xml(obstacle, mjcf_name: str) -> list[str]:
    """One obstacle, as either a fixed geom, a mocap body or a free body.

    Three kinds, because they answer different questions:

    static  a pillar. Compiled straight into the world; costs nothing.
    mocap   a door swinging, a person walking a fixed route. Driven
            kinematically, so it moves exactly as scripted and the drone cannot
            shove it aside. Infinitely heavy, in effect.
    free    a cardboard box, a stack of cups. Has mass and falls over. This is
            the one that needs a real engine to be worth anything.
    """
    from .world import Swing

    kind = obstacle.motion_kind
    body = obstacle.describe()
    rgba = body.get("rgba") or [0.45, 0.5, 0.58, 1.0]

    if body["type"] == "box":
        lower = np.asarray(body["lower"], dtype=float)
        upper = np.asarray(body["upper"], dtype=float)
        centre = 0.5 * (lower + upper)
        size = np.maximum(0.5 * (upper - lower), 1e-4)
        shape = f'type="box" size="{_vec(size)}"'
    elif body["type"] == "cylinder":
        z_min, z_max = float(body["z_min"]), float(body["z_max"])
        centre = np.array([body["center_xy"][0], body["center_xy"][1], 0.5 * (z_min + z_max)])
        half_height = max(0.5 * (z_max - z_min), 1e-4)
        shape = f'type="cylinder" size="{float(body["radius"]):.6g} {half_height:.6g}"'
    elif body["type"] == "sphere":
        centre = np.asarray(body["center"], dtype=float)
        shape = f'type="sphere" size="{float(body["radius"]):.6g}"'
    else:  # pragma: no cover - obstacle_from_config rejects unknown types first
        raise ValueError(f"Cannot build MJCF for obstacle type {body['type']!r}")

    quat = yaw_quat(float(body.get("yaw", 0.0)))

    if kind == "static":
        return [
            f'    <geom name="{mjcf_name}" {shape} pos="{_vec(centre)}" '
            f'quat="{_vec(quat)}" rgba="{_vec(rgba)}"/>'
        ]

    # A swinging door turns about its hinge, not about its own middle, so the
    # body frame goes on the pivot and the panel hangs off to one side of it.
    # Get this wrong and the door spins in place like a revolving sign.
    if isinstance(obstacle.motion, Swing):
        body_pos = np.asarray(obstacle.motion.pivot, dtype=float)
        geom_pos = centre - body_pos
    else:
        body_pos = centre
        geom_pos = np.zeros(3)

    opening = f'    <body name="{mjcf_name}" pos="{_vec(body_pos)}" quat="{_vec(quat)}"'
    lines = [opening + (' mocap="true">' if kind == "mocap" else ">")]
    if kind == "free":
        lines.append(f'      <freejoint name="{mjcf_name}_free"/>')
        mass = max(float(obstacle.mass), 1e-4)
        # Inertia of a solid of roughly this size, rather than MuJoCo's default
        # guess from the geom, so a light prop and a heavy crate topple at
        # visibly different rates.
        extent = float(np.max(np.abs(geom_pos)) + 0.1)
        moment = mass * extent * extent / 6.0
        lines.append(
            f'      <inertial pos="{_vec(geom_pos)}" mass="{mass:.6g}" '
            f'diaginertia="{_vec([moment, moment, moment])}"/>'
        )
    lines.append(
        f'      <geom name="{mjcf_name}_geom" {shape} pos="{_vec(geom_pos)}" '
        f'rgba="{_vec(rgba)}"/>'
    )
    lines.append("    </body>")
    return lines


# ----------------------------------------------------------------------
# The backend
# ----------------------------------------------------------------------


@dataclass
class _DroneHandle:
    """Everything needed to talk to one drone inside the MuJoCo model."""

    drone: "SimDrone"
    body_id: int
    qpos_adr: int
    qvel_adr: int
    geom_ids: np.ndarray
    yaw_target: float = 0.0
    thrust: float = 0.0


class MuJoCoBackend:
    """Owns the MuJoCo model and translates SDK-level flight into forces.

    One model holds the room, every obstacle and every drone, so drone-to-drone
    contact and drone-to-obstacle contact come from the same solver and need no
    special-casing anywhere else in the package.
    """

    def __init__(
        self,
        world: "World",
        drones: Sequence["SimDrone"],
        timestep: float,
        collisions: bool = True,
    ):
        self.world = world
        self.timestep = float(timestep)
        self._collisions = bool(collisions)
        self.model = None
        self.data = None
        self.handles: dict[str, _DroneHandle] = {}
        self._mocap_index: dict[str, int] = {}
        self._obstacle_body_id: dict[str, int] = {}
        self._drone_geom_ids: set[int] = set()
        self.build(drones)

    # -- construction --------------------------------------------------

    def build(self, drones: Sequence["SimDrone"]) -> None:
        """Compile the model. Called again whenever the scene gains or loses a body."""
        xml, mapping = build_mjcf(self.world, drones, self.timestep)
        self._xml = xml
        try:
            self.model = mujoco.MjModel.from_xml_string(xml)
        except ValueError as exc:
            raise ValueError(
                f"MuJoCo rejected the generated model: {exc}\n"
                "This is a bug in tello_sim's MJCF writer, not in your scenario."
            ) from exc
        self.data = mujoco.MjData(self.model)
        self.mapping = mapping

        self.handles.clear()
        self._drone_geom_ids.clear()
        for drone in drones:
            body_name = mapping[drone.name]
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{body_name}_free"
            )
            start = self.model.body_geomadr[body_id]
            count = self.model.body_geomnum[body_id]
            geom_ids = np.arange(start, start + count)
            self._drone_geom_ids.update(int(g) for g in geom_ids)
            self.handles[drone.name] = _DroneHandle(
                drone=drone,
                body_id=body_id,
                qpos_adr=int(self.model.jnt_qposadr[joint_id]),
                qvel_adr=int(self.model.jnt_dofadr[joint_id]),
                geom_ids=geom_ids,
                yaw_target=float(drone.state.yaw),
            )
            self._write_state(drone)

        self._mocap_index.clear()
        self._obstacle_body_id.clear()
        for obstacle in self.world.obstacles:
            mjcf_name = mapping.get(obstacle.name)
            if mjcf_name is None:
                continue
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, mjcf_name)
            if body_id < 0:
                continue  # a static obstacle is a bare geom, it has no body
            self._obstacle_body_id[obstacle.name] = int(body_id)
            mocap_id = int(self.model.body_mocapid[body_id])
            if mocap_id >= 0:
                self._mocap_index[obstacle.name] = mocap_id

        self._cache_geom_labels()
        self.set_collisions(self._collisions)
        self._restore_obstacle_poses()
        mujoco.mj_forward(self.model, self.data)

    def rebuild(self, drones: Sequence["SimDrone"]) -> None:
        """Recompile after the obstacle set changed, preserving flight state.

        MuJoCo models are immutable once compiled, so adding an obstacle mid-run
        means a new model.  Cheap for rooms of this size, and the alternative --
        pre-allocating a pool of hidden bodies -- puts a hard cap on how much a
        scenario can put in the room and leaves phantom geoms in every render.
        """
        saved = {name: h.drone.state.copy() for name, h in self.handles.items()}
        saved_yaw = {name: h.yaw_target for name, h in self.handles.items()}
        self.build(drones)
        for name, state in saved.items():
            handle = self.handles.get(name)
            if handle is None:
                continue
            handle.drone.state = state
            handle.yaw_target = saved_yaw.get(name, state.yaw)
            self._write_state(handle.drone)
        mujoco.mj_forward(self.model, self.data)

    # -- collision switch ----------------------------------------------

    @property
    def collisions(self) -> bool:
        return self._collisions

    def set_collisions(self, enabled: bool) -> None:
        """Turn contact response on or off for the drones.

        Off means the airframes become ghosts: they fly through walls,
        obstacles and each other.  The room, the wind and the battery all still
        apply.

        This is for ablations.  A navigation policy that is allowed to clip a
        corner tells you how much of its route quality came from the planner and
        how much came from the physics refusing to let it cheat.  It is also the
        only honest way to measure "how often would this policy have crashed",
        because a policy that crashes at second three never reaches the part of
        the route you wanted to measure.

        Obstacles keep colliding with each other and with the floor, so a stack
        of boxes still stands up while the drone passes straight through it.

        The floor is the one exception: a ghost still lands on it.  Switching
        contacts off entirely drops a parked drone through the ground before it
        ever gets to take off, which answers no question anybody asked.
        """
        self._collisions = bool(enabled)
        contype, conaffinity = _MASK_DRONE_SOLID if enabled else _MASK_DRONE_GHOST
        for geom_id in self._drone_geom_ids:
            self.model.geom_contype[geom_id] = contype
            self.model.geom_conaffinity[geom_id] = conaffinity

    # -- obstacle motion -----------------------------------------------

    def move_obstacle(
        self,
        name: str,
        position: Sequence[float] | None = None,
        yaw: float | None = None,
    ) -> None:
        """Place a movable obstacle. Works on mocap and free bodies alike.

        Moving an obstacle that is following a script detaches the script.  The
        alternative is worse: the script would rewrite the pose on the very next
        step, so the call would appear to do nothing at all, and the caller would
        have no way to tell placement from a no-op.
        """
        from .world import Placed

        obstacle = self.world.get_obstacle(name)
        if obstacle.motion.kind == "mocap" and not isinstance(obstacle.motion, Placed):
            obstacle.motion = Placed()

        mocap_id = self._mocap_index.get(name)
        if mocap_id is not None:
            if position is not None:
                self.data.mocap_pos[mocap_id] = np.asarray(position, dtype=float)
            if yaw is not None:
                self.data.mocap_quat[mocap_id] = yaw_quat(yaw)
            return

        body_id = self._obstacle_body_id.get(name)
        if body_id is None:
            raise KeyError(
                f"Obstacle {name!r} cannot be moved. Static obstacles are compiled "
                "into the world; give it motion: mocap or motion: free to move it."
            )
        joint_id = int(self.model.body_jntadr[body_id])
        if joint_id < 0:
            raise KeyError(f"Obstacle {name!r} has no joint to move.")
        adr = int(self.model.jnt_qposadr[joint_id])
        if position is not None:
            self.data.qpos[adr : adr + 3] = np.asarray(position, dtype=float)
        if yaw is not None:
            self.data.qpos[adr + 3 : adr + 7] = yaw_quat(yaw)
        dof = int(self.model.jnt_dofadr[joint_id])
        self.data.qvel[dof : dof + 6] = 0.0

    def obstacle_pose(self, name: str) -> tuple[np.ndarray, float]:
        """Where an obstacle actually is now, after being moved or knocked over."""
        body_id = self._obstacle_body_id.get(name)
        if body_id is None:
            obstacle = self.world.get_obstacle(name)  # static: it never left
            return np.asarray(obstacle.rest_center, dtype=float), float(obstacle.yaw)
        return self.data.xpos[body_id].copy(), yaw_from_quat(self.data.xquat[body_id])

    def obstacle_quat(self, name: str) -> np.ndarray:
        """Full orientation of an obstacle, w-first.

        Yaw is enough for a door or a walker, which only ever turn about the
        vertical. It is not enough for a block that has been knocked over: drawn
        from yaw alone, a tumbling block falls to the floor perfectly upright,
        which looks like anything but a physics engine.
        """
        body_id = self._obstacle_body_id.get(name)
        if body_id is None:
            return yaw_quat(float(self.world.get_obstacle(name).yaw))
        return self.data.xquat[body_id].copy()

    # -- state transfer ------------------------------------------------

    def _write_state(self, drone: "SimDrone") -> None:
        """Push our DroneState into MuJoCo. Used at build and after a teleport.

        The height is lifted clear of the floor first.  A scenario says the
        drone starts at z=0, meaning "on the ground", but the airframe's centre
        cannot be on the ground: it sits half a body-height above it.  Writing
        the literal zero buries the disc in the floor, and the solver answers a
        buried body by firing it into the air at two and a half metres a second,
        which looks exactly like a mysterious takeoff bug.
        """
        handle = self.handles[drone.name]
        state = drone.state
        adr, dof = handle.qpos_adr, handle.qvel_adr

        floor = float(self.world.bounds_lower[2]) + drone.spec.body_half_height
        if state.position[2] < floor:
            state.position = np.array(
                [state.position[0], state.position[1], floor], dtype=float
            )

        self.data.qpos[adr : adr + 3] = state.position
        self.data.qpos[adr + 3 : adr + 7] = yaw_quat(state.yaw)
        self.data.qvel[dof : dof + 3] = state.velocity
        self.data.qvel[dof + 3 : dof + 6] = 0.0

    def _read_state(self, handle: _DroneHandle) -> None:
        """Pull MuJoCo's answer back into the DroneState the rest of the code reads."""
        state = handle.drone.state
        adr, dof = handle.qpos_adr, handle.qvel_adr
        state.position = self.data.qpos[adr : adr + 3].copy()
        state.velocity = self.data.qvel[dof : dof + 3].copy()

        # xmat is the same rotation the quaternion encodes, already computed by
        # the forward kinematics that mj_step just ran.
        rotation = self.data.xmat[handle.body_id].reshape(3, 3).copy()
        state.yaw = yaw_from_matrix(rotation)
        state.orientation = rotation

        # Angular velocity comes out of a free joint in BODY axes.
        omega_body = self.data.qvel[dof + 3 : dof + 6]
        state.yaw_rate = float((rotation @ omega_body)[2])

        # Pitch and roll now describe the airframe's real attitude rather than
        # being back-computed from the commanded acceleration.
        up = rotation[:, 2]
        forward = rotation[:, 0]
        state.pitch = float(np.arcsin(np.clip(-forward[2], -1.0, 1.0)))
        state.roll = float(np.arctan2(up[1] * np.cos(state.yaw) - up[0] * np.sin(state.yaw),
                                      up[2]))

    # -- the step ------------------------------------------------------

    def apply_control(
        self,
        handle: _DroneHandle,
        accel_cmd: np.ndarray,
        yaw_rate_cmd: float,
        wind: np.ndarray,
        dt: float,
    ) -> None:
        """Turn a desired acceleration into the force and torque MuJoCo needs.

        Wind enters here, as drag on velocity relative to the air, which is the
        one piece of the old model that carries over unchanged because it was
        never the part that needed an engine.
        """
        state = handle.drone.state
        spec = handle.drone.spec
        body_id = handle.body_id

        rotation = self.data.xmat[body_id].reshape(3, 3)
        body_up = rotation[:, 2]

        velocity = self.data.qvel[handle.qvel_adr : handle.qvel_adr + 3]
        relative_velocity = velocity - np.asarray(wind, dtype=float)
        drag = -spec.drag_coeff * relative_velocity
        state.wind_seen = np.asarray(wind, dtype=float)

        if not state.motors_on:
            # Rotors stopped: gravity is MuJoCo's job, drag is ours, and the
            # airframe tumbles however the contacts say it should.
            self.data.xfrc_applied[body_id, :3] = drag
            self.data.xfrc_applied[body_id, 3:] = 0.0
            handle.thrust = 0.0
            state.last_accel_cmd = np.zeros(3)
            return

        accel_cmd = np.asarray(accel_cmd, dtype=float)
        state.last_accel_cmd = accel_cmd.copy()

        # The force the flight controller wishes it could apply, including the
        # part that merely holds the aircraft up.
        desired_force = spec.mass * (accel_cmd + np.array([0.0, 0.0, GRAVITY]))
        magnitude = float(np.linalg.norm(desired_force))
        if magnitude < 1e-9:
            desired_up = np.array([0.0, 0.0, 1.0])
        else:
            desired_up = desired_force / magnitude

        # Thrust acts along the axis the rotors actually point down, not the one
        # we wish they pointed down. Until the attitude controller has finished
        # tilting the airframe, some of the commanded acceleration simply does
        # not happen -- which is the whole reason for modelling attitude.
        thrust = float(np.clip(magnitude, 0.0, spec.max_thrust(GRAVITY)))
        handle.thrust = thrust
        self.data.xfrc_applied[body_id, :3] = thrust * body_up + drag

        # --- attitude ---
        handle.yaw_target = wrap_angle(handle.yaw_target + float(yaw_rate_cmd) * dt)
        heading = np.array([np.cos(handle.yaw_target), np.sin(handle.yaw_target), 0.0])

        left = np.cross(desired_up, heading)
        norm = float(np.linalg.norm(left))
        if norm < 1e-6:  # thrust straight along the heading: pick any consistent frame
            left = np.array([0.0, 1.0, 0.0])
        else:
            left = left / norm
        forward = np.cross(left, desired_up)
        desired_rotation = np.column_stack([forward, left, desired_up])

        # Geometric attitude error (Lee et al.), which stays well behaved at the
        # large tilt angles a collision can produce. Euler-angle errors do not.
        error_matrix = desired_rotation.T @ rotation - rotation.T @ desired_rotation
        attitude_error = 0.5 * vee(error_matrix)

        omega_body = self.data.qvel[handle.qvel_adr + 3 : handle.qvel_adr + 6].copy()
        desired_omega = rotation.T @ np.array([0.0, 0.0, float(yaw_rate_cmd)])
        rate_error = omega_body - desired_omega

        inertia = np.asarray(spec.inertia_diag(), dtype=float)
        torque_body = (
            -spec.kR_attitude * inertia * attitude_error
            - spec.kW_attitude * inertia * rate_error
            + np.cross(omega_body, inertia * omega_body)
        )
        limit = spec.max_torque()
        torque_body = np.clip(torque_body, -limit, limit)
        self.data.xfrc_applied[body_id, 3:] = rotation @ torque_body

    def step(
        self,
        controls: dict[str, tuple[np.ndarray, float]],
        sim_time: float,
        dt: float,
    ) -> dict[str, list[str]]:
        """Advance every drone one physics step. Returns what each one touched."""
        self._drive_scripted_obstacles(sim_time)

        for name, handle in self.handles.items():
            accel_cmd, yaw_rate_cmd = controls.get(name, (np.zeros(3), 0.0))
            wind = self.world.wind_at(handle.drone.state.position, sim_time)
            self.apply_control(handle, accel_cmd, yaw_rate_cmd, wind, dt)

        mujoco.mj_step(self.model, self.data)

        for handle in self.handles.values():
            self._read_state(handle)
        self._write_back_obstacle_poses()

        if self._collisions:
            return self._collect_contacts()
        return self._ghost_contacts()

    def _drive_scripted_obstacles(self, sim_time: float) -> None:
        """Put every kinematically-driven obstacle where its script says."""
        for name, (position, yaw) in self.world.scripted_poses(sim_time).items():
            mocap_id = self._mocap_index.get(name)
            if mocap_id is None:
                continue
            self.data.mocap_pos[mocap_id] = position
            self.data.mocap_quat[mocap_id] = yaw_quat(yaw)

    def _write_back_obstacle_poses(self) -> None:
        """Record each movable obstacle's live body pose on the obstacle.

        Without this the sensing geometry in `world.py` would keep answering
        from wherever the scenario file first put the box, so a drone could fly
        under a table that had already been pushed aside and still read its
        height off the tabletop.

        Only the pose is written, never the shape. The shape used to be moved
        to the body's position instead, which is wrong for a door: its body sits
        on the hinge, so the door was then described half a metre off.
        """
        for obstacle in self.world.obstacles:
            body_id = self._obstacle_body_id.get(obstacle.name)
            if body_id is None:
                continue
            obstacle.live_position = self.data.xpos[body_id].copy()
            obstacle.live_quat = self.data.xquat[body_id].copy()

    def _restore_obstacle_poses(self) -> None:
        """Put movable obstacles back where the last model had them.

        A model is compiled from the scenario's layout, so every rebuild --
        and adding anything to the room causes one -- would otherwise return
        each door, walker and knocked-over brick to where it started.
        """
        for obstacle in self.world.obstacles:
            if obstacle.live_position is None or obstacle.live_quat is None:
                continue
            body_id = self._obstacle_body_id.get(obstacle.name)
            if body_id is None:
                continue
            mocap_id = int(self.model.body_mocapid[body_id])
            if mocap_id >= 0:
                self.data.mocap_pos[mocap_id] = obstacle.live_position
                self.data.mocap_quat[mocap_id] = obstacle.live_quat
                continue
            joint_id = int(self.model.body_jntadr[body_id])
            if joint_id >= 0:
                adr = int(self.model.jnt_qposadr[joint_id])
                self.data.qpos[adr : adr + 3] = obstacle.live_position
                self.data.qpos[adr + 3 : adr + 7] = obstacle.live_quat

    def _ghost_contacts(self) -> dict[str, list[str]]:
        """What each drone WOULD have hit, with collisions turned off.

        The solver generates no contacts for a ghost, so overlap is measured
        directly against each obstacle's bounding form.  Approximate near a
        corner, and deliberately biased toward reporting a hit: a benchmark that
        under-counts crashes is worse than one that over-counts them.
        """
        hits: dict[str, list[str]] = {name: [] for name in self.handles}
        if not self.world or not self.handles:
            return hits

        lower = np.asarray(self.world.bounds_lower, dtype=float)
        upper = np.asarray(self.world.bounds_upper, dtype=float)

        positions = {
            name: handle.drone.state.position for name, handle in self.handles.items()
        }
        for name, handle in self.handles.items():
            p = positions[name]
            spec = handle.drone.spec
            radius = spec.radius

            for axis, label in enumerate("xy"):
                if p[axis] - radius < lower[axis]:
                    hits[name].append(f"wall_{label}_min")
                elif p[axis] + radius > upper[axis]:
                    hits[name].append(f"wall_{label}_max")

            # Vertically the airframe is a flat disc, not a ball. Measuring the
            # floor against its radius would report a drone standing peacefully
            # on the ground as permanently colliding with it.
            if p[2] - spec.body_half_height < lower[2] - 1e-4:
                hits[name].append("floor")
            elif p[2] + spec.body_half_height > upper[2]:
                hits[name].append("ceiling")

            # Obstacles are asked about separately from the room, or a drone
            # resting on the floor -- zero clearance, by definition -- would be
            # reported as colliding with whichever obstacle happened to be
            # nearest, however far away that was.
            nearest, distance = self.world.nearest_obstacle(p)
            if nearest is not None and distance < radius:
                hits[name].append(nearest.name)

            for other, other_p in positions.items():
                if other == name:
                    continue
                reach = radius + self.handles[other].drone.spec.radius
                if float(np.linalg.norm(p - other_p)) < reach:
                    hits[name].append(other)
        return hits

    def _cache_geom_labels(self) -> None:
        """Name every geom once, at build time.

        Looking these up per contact per step meant four `mj_id2name` calls a
        tick for a drone merely resting on the floor, which cost more than the
        entire physics step it was reporting on.
        """
        self._geom_owner: dict[int, str] = {}
        for name, handle in self.handles.items():
            for geom_id in handle.geom_ids:
                self._geom_owner[int(geom_id)] = name

        labels: list[str] = []
        for geom_id in range(self.model.ngeom):
            raw = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if raw is None:
                labels.append(f"geom_{geom_id}")
            else:
                labels.append(raw[:-5] if raw.endswith("_geom") else raw)
        self._geom_labels = labels

    def _collect_contacts(self) -> dict[str, list[str]]:
        """Name whatever each drone is touching this step.

        Read from the solver rather than recomputed, so the list is exactly what
        the physics acted on.
        """
        hits: dict[str, list[str]] = {name: [] for name in self.handles}
        owner_of = self._geom_owner
        labels = self._geom_labels

        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            owner1, owner2 = owner_of.get(g1), owner_of.get(g2)
            if owner1 is None and owner2 is None:
                continue  # two obstacles bumping each other; not our business
            if owner1 is not None:
                hits[owner1].append(owner2 or labels[g2])
            if owner2 is not None:
                hits[owner2].append(owner1 or labels[g1])
        return hits

    # -- introspection for the viewer ----------------------------------

    def scene_description(self) -> list[dict]:
        """Current pose of every obstacle, for the 3D view to draw.

        The viewer used to read obstacle positions straight from the scenario,
        which was fine when nothing moved.  Now that a box can be pushed over,
        the only truthful source is the solver.
        """
        out = []
        for obstacle in self.world.obstacles:
            described = obstacle.describe()
            body_id = self._obstacle_body_id.get(obstacle.name)
            if body_id is not None:
                described["pos"] = self.data.xpos[body_id].tolist()
                described["yaw"] = yaw_from_quat(self.data.xquat[body_id])
                described["moved"] = True
            out.append(described)
        return out

    @property
    def xml(self) -> str:
        """The generated MJCF, for debugging a scenario that will not compile."""
        return self._xml
