"""Physical and behavioural constants for the Tello simulator.

Everything a real flight can teach us lives here, in one place, so the model can
be calibrated against measured data instead of guessed at forever.

The values below are starting points: DJI's published specs where they exist,
and reasonable estimates where they do not.  Fields marked CALIBRATE should be
replaced with numbers measured from the real drone (see docs in README).

Units are SI internally (metres, seconds, kilograms, radians).  The Tello SDK
speaks centimetres and degrees; conversion happens only at the protocol edge.
"""

from __future__ import annotations

from dataclasses import dataclass, field

CM = 0.01  # multiply a centimetre value by this to get metres


@dataclass
class DroneSpec:
    """Physical properties of one airframe."""

    # --- Mass and size -----------------------------------------------------
    mass: float = 0.080          # kg, Tello with battery and propeller guards
    radius: float = 0.13         # m, collision disc. Bare Tello ~0.10, caged ~0.13

    # The airframe is a flat disc, not a ball. Half-height matters because it
    # decides where the body centre sits when the drone is on the floor, and
    # therefore what height a landed drone reports.
    body_half_height: float = 0.02   # m, a Tello is about 4 cm tall

    # --- Aerodynamics ------------------------------------------------------
    # Linear drag acting on velocity *relative to the air*.  Chosen so an
    # uncontrolled drone reaches ~80% of the wind speed in about one second,
    # which is roughly how a 80 g airframe behaves.
    drag_coeff: float = 0.11     # N per (m/s)

    # --- Control authority -------------------------------------------------
    # How hard the onboard flight controller can push.  This is the single most
    # important parameter for wind studies: when the wind force exceeds
    # mass * max_accel, the drone can no longer hold position and drifts away.
    max_accel_xy: float = 3.5    # m/s^2   CALIBRATE
    max_accel_z: float = 2.5     # m/s^2   CALIBRATE
    max_yaw_rate: float = 1.75   # rad/s  (~100 deg/s)  CALIBRATE
    yaw_accel: float = 6.0       # rad/s^2

    # --- Inner loop gains --------------------------------------------------
    # Position -> desired velocity, then desired velocity -> acceleration.
    # Proportional only, so constant wind leaves a steady position offset, the
    # way a real Tello sags downwind rather than holding perfectly.
    #
    # Together with max_accel_xy these gains decide the wind speed at which the
    # drone stops coping.  Holding still against wind w needs an acceleration of
    # drag_coeff * w / mass, so the default numbers give up at roughly
    #     w = max_accel_xy * mass / drag_coeff = 3.5 * 0.080 / 0.11 = 2.5 m/s
    # which matches a Tello being an indoor aircraft that dislikes any breeze.
    kp_pos: float = 4.0          # 1/s
    kv_vel: float = 5.0          # 1/s
    kp_yaw: float = 4.0          # 1/s

    # --- Attitude loop (MuJoCo backend) ------------------------------------
    # A quadrotor accelerates sideways by tilting, so the rate at which it can
    # tilt is a hard limit on how fast it can dodge. These gains set that rate.
    # They are stiff because an 80 g airframe has almost no rotational inertia:
    # a Tello snaps to a new attitude in well under a tenth of a second.
    #
    # Raise them and the drone tracks commands more crisply but the solver needs
    # smaller steps to stay stable. Lower them and it wallows.
    kR_attitude: float = 220.0   # 1/s^2, attitude error -> angular accel
    kW_attitude: float = 28.0    # 1/s,   angular rate error -> angular accel
    max_torque_ratio: float = 0.55  # fraction of (m*g*radius) available as torque

    # Moment of inertia, modelled as a uniform disc of the airframe's own size.
    # A real Tello is not uniform, but nothing in the SDK can tell the
    # difference, and this puts the number in the right order of magnitude.
    inertia_scale: float = 1.0   # CALIBRATE, multiply the disc estimate

    # How much more than its own weight the rotors can lift. A Tello is a gentle
    # aircraft: it will not do a 3 g pull-up, and capping thrust here is what
    # makes it lose an argument with a strong downdraught.
    thrust_to_weight: float = 2.2    # CALIBRATE

    # --- Speed limits ------------------------------------------------------
    default_speed: float = 0.30      # m/s, matches `speed 30`
    min_sdk_speed: float = 0.10      # m/s, SDK allows 10-100 cm/s
    max_sdk_speed: float = 1.00      # m/s
    max_rc_speed: float = 1.00       # m/s at full stick deflection
    max_rc_yaw_rate: float = 1.75    # rad/s at full stick

    # --- Timings (seconds) -------------------------------------------------
    # Measured on the real drone with time.perf_counter() around each command.
    takeoff_duration: float = 5.0    # CALIBRATE
    land_duration: float = 4.0       # CALIBRATE
    takeoff_height: float = 0.80     # m, where a Tello settles after takeoff
    flip_duration: float = 1.4
    command_overhead: float = 0.15   # s of latency before a move starts moving
    connect_delay: float = 0.20      # s for the `command` handshake

    # --- Command acceptance ------------------------------------------------
    position_tolerance: float = 0.02     # m, "arrived" threshold
    yaw_tolerance: float = 0.012        # rad (~0.7 deg)
    move_timeout_factor: float = 3.0    # give up after this * nominal duration
    # A scrape along a wall makes and breaks contact many times a second.
    # A surface must come clear for this long before it counts as a new hit.
    collision_cooldown: float = 0.5     # s
    ground_clearance: float = 0.0       # m, height of the body centre when landed
    settle_time: float = 1.5            # s reserved at the end of takeoff/landing

    # --- Actuation error ---------------------------------------------------
    # A real Tello does not travel exactly 100 cm when told to. Optical flow
    # drift and an imperfect ground-speed estimate leave a few centimetres of
    # error, and it accumulates over a long sequence of relative moves. This is
    # separate from sensor noise: it changes where the drone actually goes.
    move_error_std: float = 0.03        # m per axis      CALIBRATE
    yaw_error_std_deg: float = 1.5      # deg per turn    CALIBRATE

    # --- Battery -----------------------------------------------------------
    # A healthy Tello hovers for about 13 minutes.  Manoeuvring costs more.
    battery_start: float = 100.0
    drain_idle: float = 0.010        # %/s, powered on but landed
    drain_hover: float = 0.128       # %/s, 100% / (13 * 60 s)   CALIBRATE
    drain_per_accel: float = 0.020   # %/s per (m/s^2) of commanded acceleration
    drain_per_speed: float = 0.015   # %/s per (m/s) of airspeed
    battery_low_warning: float = 20.0
    battery_forced_landing: float = 5.0

    # --- Sensors -----------------------------------------------------------
    noise_height: float = 0.004      # m, std dev on barometer/ToF height
    noise_attitude: float = 0.004    # rad, std dev on pitch/roll/yaw report
    noise_velocity: float = 0.02     # m/s, std dev on reported velocity
    temperature_low: int = 60        # deg C, what a warm Tello reports
    temperature_high: int = 63

    # --- Mission pads (Tello EDU only) -------------------------------------
    pad_detect_min_height: float = 0.30   # m
    pad_detect_max_height: float = 1.20   # m
    pad_detect_radius: float = 0.35       # m, horizontal capture radius

    # --- Derived quantities the physics engine needs -----------------------

    def inertia_diag(self) -> tuple[float, float, float]:
        """Principal moments, treating the airframe as a uniform flat disc.

        For a disc of mass m and radius r spun about its own axis the moment is
        m*r^2/2, and about a diameter it is m*r^2/4. The drone is wider than it
        is tall, so yaw is the stiff axis and pitch/roll are the nimble ones,
        which is the right qualitative behaviour even before calibration.
        """
        base = self.mass * self.radius * self.radius * self.inertia_scale
        return (base / 4.0, base / 4.0, base / 2.0)

    def max_thrust(self, gravity: float = 9.81) -> float:
        """Total rotor force available, in newtons."""
        return self.thrust_to_weight * self.mass * gravity

    def max_torque(self, gravity: float = 9.81) -> float:
        """Torque available about any body axis, in newton-metres.

        Scaled from weight and arm length rather than set as a free number, so
        changing the drone's mass or size keeps its agility plausible.
        """
        return self.max_torque_ratio * self.mass * gravity * self.radius


@dataclass
class SimSpec:
    """Simulator-level settings."""

    # Physics step. 200 Hz by default, which is a measured compromise rather
    # than a round number:
    #
    #   dt      realtime factor    4 m/s fall onto a 4 cm shelf
    #   0.002       ~31x           caught, 0.3 cm penetration
    #   0.005       ~76x           caught, 2.0 cm penetration
    #   0.010      ~150x           caught, 1.3 cm penetration
    #   0.020      ~300x           FELL STRAIGHT THROUGH
    #
    # Free-flight trajectories agree to a millimetre across all of these, so the
    # only thing dt buys is contact fidelity. 0.02 tunnels, and tunnelling is a
    # silent failure that quietly corrupts a whole sweep, so the default keeps a
    # margin. Set dt=0.01 to double throughput when a scenario has no thin
    # obstacles and nothing falls from height.
    dt: float = 0.005                # s, physics step (200 Hz)
    telemetry_hz: float = 10.0       # Hz, matches the real Tello state broadcast
    realtime: bool = True            # False runs as fast as the CPU allows
    seed: int = 0                    # deterministic noise and gusts
    log_dir: str = "logs"
    sensor_noise: bool = True
    actuation_noise: bool = True
    crash_on_collision: bool = False  # True = a bump ends the flight

    # --- Collisions --------------------------------------------------------
    # False makes the drones ghosts: they pass through walls, obstacles and
    # each other, while wind, gravity and the battery all still apply.
    #
    # This exists for ablations. A policy that crashes at second three never
    # reaches the part of the route you wanted to measure, so "how good is the
    # route" and "how often does it crash" have to be asked separately.
    # `report_ghost_contacts` keeps counting what the drone would have hit, so
    # a ghost run still produces a crash rate.
    collisions: bool = True
    report_ghost_contacts: bool = True


DEFAULT_DRONE = DroneSpec()
DEFAULT_SIM = SimSpec()
