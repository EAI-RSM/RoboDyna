from ._base_task import Base_Task
from .utils import *
import numpy as np
import sapien
import sapien.physx
import transforms3d as t3d


class catch_valley_ball_v1(Base_Task):
    """Place a bowl to catch a red ball leaving a down-then-up valley ramp.

    The ramp, ball, and robot use the standard Base_Task table/arm setup.  The
    ball follows a deterministic, step-driven drop and roll while the robot
    moves the bowl.  At the upward ramp's edge it becomes a real dynamic body,
    so the final flight and catch are resolved by PhysX.

    Options (independent toggles; CLI via ``--task-arg`` or legacy ``--option``):
      - Default — red ball travels straight toward the exit edge.
      - Option 1 — ``wall_bounce_enabled``: red ball rebounds from a side rail
        so its heading changes mid-run.
        CLI: ``--task-arg wall_bounce_enabled=true`` or ``--option 1``.
      - Option 2 — ``enable_distractor``: black distractor ball on a randomized
        lateral lane (above or below the red ball). Exit points are separated by at
        least one ball diameter. Red/black contact uses an equal-mass bounce (no
        pass-through). Catching the black ball fails the episode.
        CLI: ``--task-arg enable_distractor=true`` or ``--option 2``.
      Options 1 and 2 may be combined; exit separation still applies.
      Both balls sample a random lane y each episode.

      Layout mirror (x → −x) — ``random_mirror`` / ``mirrored``:
        Default layout exits toward +x (right arm places the bowl). Mirrored layout
        exits toward −x (left arm places the bowl). With ``random_mirror: true``
        (default) the side is chosen per episode when ``mirrored`` is unset.
        CLI: ``--task-arg mirrored=true`` or ``--task-arg random_mirror=true``.
    """

    SIM_HZ = 250.0
    GRAVITY = 9.81

    RAMP_HALF_WIDTH_DEFAULT = 0.125
    RAMP_THICKNESS_DEFAULT = 0.012
    RAMP_CENTER_Y_DEFAULT = 0.08
    # Valley shifted left so the exit stays near arm reach on the catch side.
    RAMP_VALLEY_X_DEFAULT = -0.14
    RAMP_VALLEY_HEIGHT_DEFAULT = 0.025
    # Horizontal run means (episode samples ±length_jitter). Catching side is shorter.
    DOWN_RUN_DEFAULT = 0.324
    DOWN_RISE_DEFAULT = 0.094
    UP_RUN_DEFAULT = 0.2592  # catching-side run = 0.324 * 0.8
    UP_RISE_DEFAULT = 0.085
    PARAM_JITTER_DEFAULT = 0.10  # ±10% around non-length means
    LENGTH_JITTER_DEFAULT = 0.05  # ±5% around down_run / up_run
    # Kept for config compat; length uses length_jitter, not a global scale.
    PLATFORM_LENGTH_SCALE_MIN_DEFAULT = 1.0
    PLATFORM_LENGTH_SCALE_MAX_DEFAULT = 1.0
    CURVE_SEGMENTS_DEFAULT = 16
    RAIL_HEIGHT_DEFAULT = 0.055
    RAIL_THICKNESS_DEFAULT = 0.006
    RANDOM_MIRROR_DEFAULT = True
    MIRRORED_DEFAULT = None  # None → sample when random_mirror, else use explicit bool
    # When mirrored (exit toward −x / left), nudge the whole fixture toward +x.
    MIRROR_X_SHIFT = 0.05

    BALL_RADIUS_DEFAULT = 0.018
    BALL_MASS_DEFAULT = 0.50  # kg (500 g)
    DROP_HEIGHT_DEFAULT = 0.14
    DROP_TIME_DEFAULT = 0.60
    DROP_WALL_ANGLE_MIN_DEFAULT = 15.0
    DROP_WALL_ANGLE_MAX_DEFAULT = 25.0
    WALL_BOUNCE_ENABLED_DEFAULT = False
    ENABLE_DISTRACTOR_DEFAULT = False
    DISTRACTOR_COLOR_DEFAULT = [0.05, 0.05, 0.05]
    BALL_PATH_MODE_DEFAULT = "straight"
    DROP_FORWARD_ANGLE_MIN_DEFAULT = -8.0
    DROP_FORWARD_ANGLE_MAX_DEFAULT = 12.0
    INITIAL_FORWARD_SPEED_DEFAULT = 0.105
    ROLL_TIME_MIN_DEFAULT = 4.0
    ROLL_TIME_MAX_DEFAULT = 7.0
    LAUNCH_SPEED_DEFAULT = 0.385
    ROLL_ACCELERATION_DEFAULT = 0.25
    PHYSICS_MAX_STEPS_DEFAULT = 1200
    RED_LINE_GAP_DEFAULT = 0.05
    LANDING_GAP_DEFAULT = 0.15  # ~15 cm past red line
    IDLE_TIME_MIN_DEFAULT = 1.0
    IDLE_TIME_MAX_DEFAULT = 2.0
    SETTLE_STEPS_DEFAULT = 150

    CATCHER_MODEL_DEFAULT = "021_cup"
    BOWL_ID_DEFAULT = 1  # 021_cup instance base1
    BOWL_SCALE_MULT_DEFAULT = 1.0
    BOWL_INNER_RADIUS_DEFAULT = 0.042  # catch success half-width
    BOWL_OUTER_RADIUS_DEFAULT = 0.052
    BOWL_HEIGHT_DEFAULT = 0.080
    BOWL_PLACE_Z_OFFSET = -0.010  # place height relative to table_top
    BOWL_MASS_DEFAULT = 0.25

    def setup_demo(self, **kwags):
        self._cfg = kwags.get("task_args", {}).get("catch_valley_ball_v1", {})

        # The environment object is reused between episodes. load_camera can
        # invoke the per-step hook before the new actors have been constructed.
        self._loaded = False
        self._ball_phase = None
        self._distractor_phase = None
        self._expert_demo = False
        self._bowl_ready = False
        self._bowl_welded = False
        self._arm_ball_contact = False
        self.distractor = None
        self._distractor_rigid = None
        self.enable_distractor = False
        self.wall_bounce_enabled = False
        self.mirrored = False
        self.side = 1.0
        super()._init_task_env_(**kwags)

        # Evaluation does not call play_once, so start the self-contained ball
        # motion after initialization. Expert collection resets the same state.
        self._start_ball_motion(expert_demo=False)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _sample_pm(mean, jitter=0.10):
        """Sample uniformly in ``[mean*(1-jitter), mean*(1+jitter)]``."""
        mean = float(mean)
        jitter = float(max(jitter, 0.0))
        lo = mean * (1.0 - jitter)
        hi = mean * (1.0 + jitter)
        if lo > hi:
            lo, hi = hi, lo
        if abs(hi - lo) < 1e-12:
            return mean
        return float(np.random.uniform(lo, hi))

    @staticmethod
    def _as_bool(value, default: bool) -> bool:
        if value is None:
            return bool(default)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        s = str(value).strip().lower()
        if s in ("1", "true", "yes", "on"):
            return True
        if s in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"catch_valley_ball_v1 expected a boolean, got {value!r}")

    def _parse_wall_bounce_enabled(self, c) -> bool:
        """Option 1: wall rebound path (preferred) or legacy ``option: 1``."""
        wall = c.get("wall_bounce_enabled", c.get("opt1", None))
        legacy = c.get("option", None)
        if legacy is not None and wall is None:
            if legacy in (1, "1", "wall_bounce", "wall_bounce_enabled", "angled"):
                wall = True
            elif legacy in (2, "2", "enable_distractor", "distractor"):
                wall = False
            else:
                raise ValueError(
                    "catch_valley_ball_v1 option must be 1/wall_bounce_enabled or "
                    "2/enable_distractor (or set the booleans directly)"
                )
        if wall is not None:
            return self._as_bool(wall, self.WALL_BOUNCE_ENABLED_DEFAULT)

        # Backward-compatible path-mode override when the toggle is unset.
        mode = str(c.get("ball_path_mode", self.BALL_PATH_MODE_DEFAULT)).strip().lower()
        if mode == "angled":
            return True
        if mode == "random":
            return bool(np.random.choice([False, True]))
        return bool(self.WALL_BOUNCE_ENABLED_DEFAULT)

    def _parse_enable_distractor(self, c) -> bool:
        """Option 2: black distractor ball (preferred) or legacy ``option: 2``."""
        distractor = c.get("enable_distractor", c.get("opt2", None))
        legacy = c.get("option", None)
        if legacy is not None and distractor is None:
            if legacy in (2, "2", "enable_distractor", "distractor"):
                distractor = True
            elif legacy in (1, "1", "wall_bounce", "wall_bounce_enabled", "angled"):
                distractor = False
            else:
                raise ValueError(
                    "catch_valley_ball_v1 option must be 1/wall_bounce_enabled or "
                    "2/enable_distractor (or set the booleans directly)"
                )
        return self._as_bool(distractor, self.ENABLE_DISTRACTOR_DEFAULT)

    def _parse_mirrored(self, c) -> bool:
        """Layout mirror (x → −x): explicit ``mirrored``, else random when ``random_mirror``."""
        random_mirror = self._as_bool(
            c.get("random_mirror", self.RANDOM_MIRROR_DEFAULT),
            self.RANDOM_MIRROR_DEFAULT,
        )
        mirror_cfg = c.get("mirrored", self.MIRRORED_DEFAULT)
        if mirror_cfg is None:
            return bool(random_mirror and (np.random.rand() < 0.5))
        return self._as_bool(mirror_cfg, False)

    def _option_label(self) -> str:
        parts = []
        if getattr(self, "wall_bounce_enabled", False):
            parts.append("option 1")
        if getattr(self, "enable_distractor", False):
            parts.append("option 2")
        if getattr(self, "mirrored", False):
            parts.append("mirrored")
        return ", ".join(parts) if parts else "baseline"

    def _past_red_line_x(self, x, margin=0.0):
        """True if ``x`` is strictly on the catch side of the red line.

        - Right layout (``side > 0``): need ``x > red_line_x + margin`` (+x catch).
        - Left / mirrored (``side < 0``): need ``x < red_line_x - margin`` (−x catch).

        Bowl on or overlapping the line fails when ``margin`` includes the outer
        radius (near rim must clear the line).
        """
        x = float(x)
        margin = float(margin)
        red = float(self.red_line_x)
        if float(self.side) < 0.0:
            return bool(x < red - margin)
        return bool(x > red + margin)

    def _catch_target_x(self, landing_x):
        """Bowl place-x: past the red line in the travel direction, at/beyond landing."""
        # Extra margin so the rim clears the line even with placement slip.
        # Mirrored (−x): required is more negative than the line; else more positive.
        clearance = self.bowl_outer_radius + 0.025
        if float(self.side) < 0.0:
            required = float(self.red_line_x) - clearance
            return float(min(float(landing_x), required))
        required = float(self.red_line_x) + clearance
        return float(max(float(landing_x), required))

    # ---------------------------------------------------------------- actors
    def load_actors(self):
        c = self._cfg
        self.table_top = 0.74 + self.table_z_bias

        # Mirror flips the whole fixture across the table midline (x → −x).
        # mirrored → exit on −x → left arm; else exit on +x → right arm.
        self.mirrored = self._parse_mirrored(c)
        self.side = -1.0 if self.mirrored else 1.0

        jitter = float(c.get("param_jitter", self.PARAM_JITTER_DEFAULT))
        self.param_jitter = float(np.clip(jitter, 0.0, 0.5))
        length_jitter = float(c.get("length_jitter", self.LENGTH_JITTER_DEFAULT))
        self.length_jitter = float(np.clip(length_jitter, 0.0, 0.5))
        pm = lambda mean: self._sample_pm(mean, self.param_jitter)
        pm_len = lambda mean: self._sample_pm(mean, self.length_jitter)

        # Ramp shape / curve — each mean ±param_jitter (default ±10%).
        self.ramp_half_width = pm(c.get("ramp_half_width", self.RAMP_HALF_WIDTH_DEFAULT))
        self.ramp_thickness = pm(c.get("ramp_thickness", self.RAMP_THICKNESS_DEFAULT))
        self.ramp_center_y = float(c.get("ramp_center_y", self.RAMP_CENTER_Y_DEFAULT))
        self.valley_x = pm(c.get("ramp_valley_x", self.RAMP_VALLEY_X_DEFAULT))
        self.valley_height = pm(c.get("ramp_valley_height", self.RAMP_VALLEY_HEIGHT_DEFAULT))
        # Entry vs catching-side run lengths: independent means ±length_jitter (±5%).
        down_mean = float(c.get("down_run", c.get("ramp_run", self.DOWN_RUN_DEFAULT)))
        up_mean = float(c.get("up_run", self.UP_RUN_DEFAULT))
        self.down_run = pm_len(down_mean)
        self.up_run = pm_len(up_mean)
        self.down_rise = pm(c.get("down_rise", self.DOWN_RISE_DEFAULT))
        self.up_rise = pm(c.get("up_rise", self.UP_RISE_DEFAULT))
        self.platform_length_scale = 1.0
        curve_mean = float(c.get("curve_segments", self.CURVE_SEGMENTS_DEFAULT))
        self.curve_segments = int(round(pm(curve_mean)))
        self.rail_height = pm(c.get("rail_height", self.RAIL_HEIGHT_DEFAULT))
        self.rail_thickness = pm(c.get("rail_thickness", self.RAIL_THICKNESS_DEFAULT))

        self.ball_radius = float(c.get("ball_radius", self.BALL_RADIUS_DEFAULT))
        self.ball_mass = float(c.get("ball_mass", self.BALL_MASS_DEFAULT))
        self.drop_height = pm(c.get("drop_height", self.DROP_HEIGHT_DEFAULT))
        self.drop_time = pm(c.get("drop_time", self.DROP_TIME_DEFAULT))
        self.wall_bounce_enabled = self._parse_wall_bounce_enabled(c)
        self.enable_distractor = self._parse_enable_distractor(c)
        self.distractor_color = list(c.get("distractor_color", self.DISTRACTOR_COLOR_DEFAULT))
        self.ball_path_mode = "angled" if self.wall_bounce_enabled else "straight"
        wall_angle_min = float(c.get(
            "angled_path_min_deg",
            c.get("drop_wall_angle_min_deg", self.DROP_WALL_ANGLE_MIN_DEFAULT),
        ))
        wall_angle_max = float(c.get(
            "angled_path_max_deg",
            c.get("drop_wall_angle_max_deg", self.DROP_WALL_ANGLE_MAX_DEFAULT),
        ))
        forward_angle_min = float(c.get("drop_forward_angle_min_deg", self.DROP_FORWARD_ANGLE_MIN_DEFAULT))
        forward_angle_max = float(c.get("drop_forward_angle_max_deg", self.DROP_FORWARD_ANGLE_MAX_DEFAULT))
        # Speed / accel means ±jitter. Compat: mid of old min/max if only those exist.
        if "initial_forward_speed" in c:
            speed_mean = float(c["initial_forward_speed"])
        elif "initial_forward_speed_min" in c or "initial_forward_speed_max" in c:
            speed_mean = 0.5 * (
                float(c.get("initial_forward_speed_min", self.INITIAL_FORWARD_SPEED_DEFAULT))
                + float(c.get("initial_forward_speed_max", self.INITIAL_FORWARD_SPEED_DEFAULT))
            )
        else:
            speed_mean = float(self.INITIAL_FORWARD_SPEED_DEFAULT)
        self.initial_forward_speed = pm(speed_mean)
        self.roll_time = float(np.random.uniform(
            c.get("roll_time_min", self.ROLL_TIME_MIN_DEFAULT),
            c.get("roll_time_max", self.ROLL_TIME_MAX_DEFAULT),
        ))
        self.launch_speed = pm(c.get("launch_speed", self.LAUNCH_SPEED_DEFAULT))
        self.roll_acceleration = pm(c.get("roll_acceleration", self.ROLL_ACCELERATION_DEFAULT))
        self.physics_max_steps = int(c.get("physics_max_steps", self.PHYSICS_MAX_STEPS_DEFAULT))
        self.red_line_gap = pm(c.get("red_line_gap", self.RED_LINE_GAP_DEFAULT))
        if "landing_gap" in c:
            landing_mean = float(c["landing_gap"])
        elif "landing_gap_min" in c or "landing_gap_max" in c:
            landing_mean = 0.5 * (
                float(c.get("landing_gap_min", self.LANDING_GAP_DEFAULT))
                + float(c.get("landing_gap_max", self.LANDING_GAP_DEFAULT))
            )
        else:
            landing_mean = float(self.LANDING_GAP_DEFAULT)
        self.landing_gap = pm(landing_mean)
        self.idle_time = float(np.random.uniform(
            c.get("idle_time_min", self.IDLE_TIME_MIN_DEFAULT),
            c.get("idle_time_max", self.IDLE_TIME_MAX_DEFAULT),
        ))
        self.settle_steps = int(c.get("settle_steps", self.SETTLE_STEPS_DEFAULT))

        self.catcher_model = str(c.get("catcher_model", self.CATCHER_MODEL_DEFAULT))
        self.bowl_id = int(c.get("bowl_id", self.BOWL_ID_DEFAULT))
        self.bowl_scale_mult = float(c.get("bowl_scale_mult", self.BOWL_SCALE_MULT_DEFAULT))
        self.bowl_inner_radius = float(c.get("bowl_inner_radius", self.BOWL_INNER_RADIUS_DEFAULT))
        self.bowl_outer_radius = float(c.get("bowl_outer_radius", self.BOWL_OUTER_RADIUS_DEFAULT))
        self.bowl_height = float(c.get("bowl_height", self.BOWL_HEIGHT_DEFAULT))
        self.bowl_mass = float(c.get("bowl_mass", self.BOWL_MASS_DEFAULT))

        # Keep malformed parameter sweeps from creating inverted or degenerate
        # collision geometry.
        self.ramp_half_width = max(self.ramp_half_width, 0.045)
        self.ramp_thickness = max(self.ramp_thickness, 0.004)
        self.down_run = max(self.down_run, 0.05)
        self.up_run = max(self.up_run, 0.05)
        self.down_rise = max(self.down_rise, 0.01)
        self.up_rise = max(self.up_rise, 0.01)
        self.curve_segments = max(3, self.curve_segments)
        self.ball_radius = float(np.clip(self.ball_radius, 0.008, 0.03))
        self.ball_mass = max(float(self.ball_mass), 0.001)
        self.drop_time = max(self.drop_time, 0.05)
        self.roll_time = max(self.roll_time, 0.2)
        self.launch_speed = max(self.launch_speed, 0.05)
        self.initial_forward_speed = max(self.initial_forward_speed, 0.02)
        self.roll_acceleration = float(np.clip(self.roll_acceleration, 0.0, 0.8))
        self.physics_max_steps = max(self.physics_max_steps, 100)
        self.red_line_gap = max(self.red_line_gap, 0.0)
        self.landing_gap = max(self.landing_gap, 0.02)
        self.bowl_scale_mult = max(self.bowl_scale_mult, 0.25)
        self.bowl_mass = max(float(self.bowl_mass), 0.05)

        # Default: straight to the edge. Opt 1: reflected lateral path that
        # rebounds from one or both rails mid-run.
        if not self.wall_bounce_enabled:
            self.drop_wall_angle_deg = 0.0
        else:
            wall_angle_abs = float(np.random.uniform(
                min(wall_angle_min, wall_angle_max),
                max(wall_angle_min, wall_angle_max),
            ))
            self.drop_wall_angle_deg = float(
                np.random.choice([-1.0, 1.0]) * wall_angle_abs
            )
        self.drop_forward_angle_deg = float(np.random.uniform(
            min(forward_angle_min, forward_angle_max),
            max(forward_angle_min, forward_angle_max),
        ))
        # initial_forward_speed already sampled ±jitter above.
        self.initial_lateral_speed = float(
            self.initial_forward_speed * np.tan(np.deg2rad(self.drop_wall_angle_deg))
        )
        self.initial_lateral_speed = float(np.clip(
            self.initial_lateral_speed,
            -0.28,
            0.28,
        ))
        self.initial_vertical_speed = float(
            self.initial_forward_speed * np.tan(np.deg2rad(self.drop_forward_angle_deg))
        )
        self.initial_ball_velocity = np.array([
            self.initial_forward_speed,
            self.initial_lateral_speed,
            self.initial_vertical_speed,
        ])
        lane_half = max(
            0.005,
            self.ramp_half_width
            - 0.5 * self.rail_thickness
            - self.ball_radius
            - 0.001,
        )
        self._lane_min = self.ramp_center_y - lane_half
        self._lane_max = self.ramp_center_y + lane_half
        self._drop_lateral_travel = 0.0
        self._drop_forward_travel = float(
            self.drop_height * np.tan(np.deg2rad(self.drop_forward_angle_deg))
        )
        self.min_exit_separation = 2.0 * self.ball_radius  # one diameter

        surface_points = self._make_curved_surface()
        if self.mirrored:
            surface_points = surface_points.copy()
            surface_points[:, 0] *= -1.0
            surface_points[:, 0] += float(self.MIRROR_X_SHIFT)
        tangents = np.gradient(surface_points, axis=0)
        # Rotate tangent 90° in the XZ plane; keep the branch that points upward.
        # (A plain x-mirror makes tangent_x flip sign, which would otherwise put
        # the "up" normal into the table and build an upside-down ramp.)
        normals = np.column_stack([
            -tangents[:, 2],
            np.zeros(len(tangents)),
            tangents[:, 0],
        ])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
        normals[normals[:, 2] < 0.0] *= -1.0
        # Centerline surface offset by radius; lateral Y is baked below so
        # wall / ball-ball bounces are deterministic across plan+render passes.
        self._surface_ball_path = surface_points + self.ball_radius * normals
        self.ball_path = self._surface_ball_path.copy()
        self.distractor_path = None
        self.ball_ball_bounces = 0
        self.drop_wall_bounces = 0
        self.exit_separation = 0.0
        self._bake_ball_lanes()

        path_steps = np.linalg.norm(np.diff(self.ball_path, axis=0), axis=1)
        self.ball_path_cumulative = np.concatenate([[0.0], np.cumsum(path_steps)])
        self.ball_path_length = float(self.ball_path_cumulative[-1])
        # On-ramp roll is kinematic (posed along the path), so mass has no effect.
        # Average along-track speed is ``initial_forward_speed``:
        #   roll_time = path_length / speed
        self.roll_time = float(
            self.ball_path_length / max(float(self.initial_forward_speed), 1e-3)
        )
        self.ball_start = self.ball_path[0].copy()
        self.ball_valley = self.ball_path[self.curve_segments].copy()
        self.ball_exit = self.ball_path[-1].copy()
        self.ball_y = float(self.ball_start[1])
        # Drop is upstream of the entry (opposite the travel direction).
        self.ball_drop = self.ball_start + np.array([
            -self.side * self._drop_forward_travel,
            0.0,
            self.drop_height,
        ])
        exit_tangent = self.ball_path[-1] - self.ball_path[-2]
        # Elevation only (use |dx| so mirrored −x exits keep a positive climb angle).
        self.up_angle = float(np.arctan2(exit_tangent[2], abs(exit_tangent[0])))
        exit_horizontal = exit_tangent[:2]
        self.release_direction_xy = (
            exit_horizontal / max(np.linalg.norm(exit_horizontal), 1e-6)
        )
        start_tangent = self.ball_path[1] - self.ball_path[0]
        self.down_angle = float(abs(np.arctan2(start_tangent[2], start_tangent[0])))

        self.ramp_parts = []
        for index in range(len(surface_points) - 1):
            self._build_ramp_segment(
                name=f"curved_ramp_{index}",
                point_a=surface_points[index],
                point_b=surface_points[index + 1],
            )
        self._build_support("start_support", surface_points[0])
        self._build_support("valley_support", surface_points[self.curve_segments])
        self._build_support("exit_support", surface_points[-1])
        self.ramp_exit_x = float(surface_points[-1, 0])
        self.red_line_x = self.ramp_exit_x + self.side * self.red_line_gap
        self.red_line = create_box(
            self.scene,
            sapien.Pose([
                self.red_line_x,
                self.ramp_center_y,
                self.table_top + 0.001,
            ]),
            half_size=[0.004, self.ramp_half_width, 0.001],
            color=[0.92, 0.05, 0.04],
            is_static=True,
            name="catch_boundary_line",
        )
        self.ramp_parts.append(self.red_line)

        self.ball = create_sphere(
            self.scene,
            sapien.Pose(self.ball_drop.tolist()),
            radius=self.ball_radius,
            color=[0.88, 0.16, 0.12],
            is_static=False,
            name="valley_ball",
        )
        self._ball_rigid = self._get_rigid(self.ball)
        if self._ball_rigid is not None:
            self._ball_rigid.set_mass(self.ball_mass)
            self._ball_rigid.set_disable_gravity(True)
            self._ball_rigid.set_kinematic(True)
            self._ball_rigid.set_kinematic_target(sapien.Pose(self.ball_drop.tolist()))
            material = sapien.physx.PhysxMaterial(
                static_friction=0.12,
                dynamic_friction=0.08,
                restitution=0.22,
            )
            for shape in self._ball_rigid.get_collision_shapes():
                shape.set_physical_material(material)

        self.distractor = None
        self._distractor_rigid = None
        self._distractor_phase = None
        self.distractor_landing = None
        self.distractor_release_velocity = None
        # Lane bake already filled distractor_path / exit / cumulative / start.
        if self.enable_distractor:
            self._spawn_distractor()

        # Spawn catcher on the catch side, already past the red-line x so the
        # subsequent place mainly adjusts y toward the predicted landing.
        bowl_x = float(self.side * np.random.uniform(0.30, 0.34))
        if self.mirrored:
            bowl_x += float(self.MIRROR_X_SHIFT)
        bowl_pose = rand_pose(
            xlim=[bowl_x, bowl_x],
            ylim=[-0.22, -0.16],
            zlim=[self.table_top],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=False,
        )
        self.bowl = create_actor(
            self,
            pose=bowl_pose,
            modelname=self.catcher_model,
            model_id=self.bowl_id,
            convex=True,
            is_static=False,
            scale_mult=self.bowl_scale_mult,
        )
        self.bowl.set_mass(self.bowl_mass)
        bowl_rigid = self._get_rigid(self.bowl)
        if bowl_rigid is not None:
            bowl_material = sapien.physx.PhysxMaterial(
                static_friction=0.9,
                dynamic_friction=0.9,
                restitution=0.0,
            )
            for shape in bowl_rigid.get_collision_shapes():
                shape.set_physical_material(bowl_material)

        # Aim PhysX exit so the ball lands ~landing_gap past the red line
        # (default band centers near 15 cm), varying with exit lip / speed.
        # Clamp to arm reach on the catch half (~0.36 m from center).
        reach_lim = 0.36
        target_landing_x = float(self.red_line_x + self.side * self.landing_gap)
        if float(self.side) > 0.0:
            target_landing_x = min(target_landing_x, reach_lim)
        else:
            target_landing_x = max(target_landing_x, -reach_lim)
        self._tune_launch_to_landing(target_landing_x)
        if self.enable_distractor:
            self._update_distractor_release_velocity()
            self._compute_distractor_landing()
        self.physics_run_steps = self.physics_max_steps

        for part in self.ramp_parts:
            self.add_prohibit_area(part, padding=0.015)
        self.add_prohibit_area(self.bowl, padding=0.05)

        self.drop_steps = max(1, int(round(self.drop_time * self.SIM_HZ)))
        self.roll_steps = max(1, int(round(self.roll_time * self.SIM_HZ)))
        self.idle_steps = max(0, int(round(self.idle_time * self.SIM_HZ)))
        self.flight_steps = self.physics_max_steps
        self._robot_link_names = self._collect_robot_link_names()
        self._bowl_welded = False
        self._bowl_arm = None
        self._bowl_ee_offset = None
        self._loaded = True
        self._ball_phase = "frozen"
        if self.enable_distractor:
            self._distractor_phase = "frozen"

    def _sample_lane_y(self):
        """Uniform random y inside the playable lane."""
        return float(np.random.uniform(self._lane_min, self._lane_max))

    def _sample_distractor_start_y(self, red_y):
        """Sample black-ball y above or below the red ball with ≥1-diameter gap."""
        min_sep = self.min_exit_separation
        candidates = []
        if red_y + min_sep <= self._lane_max + 1e-9:
            candidates.append((red_y + min_sep, self._lane_max))
        if red_y - min_sep >= self._lane_min - 1e-9:
            candidates.append((self._lane_min, red_y - min_sep))
        if not candidates:
            # Degenerate narrow lane: push to the farther rail.
            if abs(self._lane_max - red_y) >= abs(red_y - self._lane_min):
                return float(self._lane_max)
            return float(self._lane_min)
        low, high = candidates[int(np.random.randint(len(candidates)))]
        if high < low:
            low, high = high, low
        if high - low < 1e-6:
            return float(low)
        return float(np.random.uniform(low, high))

    def _step_lateral_y(self, y, slope, dx):
        """Integrate lateral y; reflect slope at the rails."""
        y = float(y + slope * dx)
        hit = False
        for _ in range(6):
            if y < self._lane_min:
                y = self._lane_min + (self._lane_min - y)
                slope = -slope
                hit = True
            elif y > self._lane_max:
                y = self._lane_max - (y - self._lane_max)
                slope = -slope
                hit = True
            else:
                break
        y = float(np.clip(y, self._lane_min, self._lane_max))
        return y, float(slope), hit

    def _resolve_ball_ball_y(self, y_r, y_d, slope_r, slope_d, y_r_prev, y_d_prev):
        """Equal-mass 1D elastic bounce on Y when centers would overlap."""
        min_sep = self.min_exit_separation
        if abs(y_r - y_d) >= min_sep - 1e-9:
            return y_r, y_d, slope_r, slope_d, False

        # Approaching if relative y motion closes the gap.
        closing = (y_r - y_d) * ((y_r - y_r_prev) - (y_d - y_d_prev)) < 0.0
        overlapping = abs(y_r - y_d) < min_sep
        if not (closing or overlapping):
            return y_r, y_d, slope_r, slope_d, False

        # Swap lateral slopes (equal mass) and separate centers by one diameter.
        slope_r, slope_d = float(slope_d), float(slope_r)
        mid = 0.5 * (y_r + y_d)
        half = 0.5 * min_sep
        if y_r <= y_d:
            y_r = mid - half
            y_d = mid + half
        else:
            y_r = mid + half
            y_d = mid - half
        y_r = float(np.clip(y_r, self._lane_min, self._lane_max))
        y_d = float(np.clip(y_d, self._lane_min, self._lane_max))
        # If clamping collapsed the gap, push toward opposite rails.
        if abs(y_r - y_d) < min_sep - 1e-9:
            if y_r <= y_d:
                y_r = float(self._lane_min)
                y_d = float(min(self._lane_max, y_r + min_sep))
            else:
                y_r = float(self._lane_max)
                y_d = float(max(self._lane_min, y_r - min_sep))
        return y_r, y_d, slope_r, slope_d, True

    def _bake_ball_lanes(self):
        """Randomize start Ys and bake lateral paths with rail + ball-ball bounce."""
        path = self._surface_ball_path
        n = len(path)
        min_sep = self.min_exit_separation
        lateral_slope0 = float(np.tan(np.deg2rad(self.drop_wall_angle_deg)))

        best = None
        for _ in range(32):
            y_r0 = self._sample_lane_y()
            y_d0 = (
                self._sample_distractor_start_y(y_r0)
                if self.enable_distractor
                else None
            )
            y_r = np.zeros(n, dtype=np.float64)
            y_d = np.zeros(n, dtype=np.float64) if self.enable_distractor else None
            y_r[0] = y_r0
            slope_r = lateral_slope0
            slope_d = 0.0
            rail_hits = 0
            ball_hits = 0
            if self.enable_distractor:
                y_d[0] = y_d0

            for i in range(n - 1):
                dx = float(path[i + 1, 0] - path[i, 0])
                y_r_prev = float(y_r[i])
                y_r[i + 1], slope_r, hit_r = self._step_lateral_y(y_r[i], slope_r, dx)
                if hit_r:
                    rail_hits += 1
                if self.enable_distractor:
                    y_d_prev = float(y_d[i])
                    y_d[i + 1], slope_d, hit_d = self._step_lateral_y(
                        y_d[i], slope_d, dx
                    )
                    if hit_d:
                        rail_hits += 1
                    y_r[i + 1], y_d[i + 1], slope_r, slope_d, bounced = (
                        self._resolve_ball_ball_y(
                            float(y_r[i + 1]),
                            float(y_d[i + 1]),
                            slope_r,
                            slope_d,
                            y_r_prev,
                            y_d_prev,
                        )
                    )
                    if bounced:
                        ball_hits += 1

            exit_sep = (
                float(abs(y_r[-1] - y_d[-1]))
                if self.enable_distractor
                else min_sep
            )
            candidate = (y_r, y_d, rail_hits, ball_hits, exit_sep, y_r0, y_d0)
            if best is None or exit_sep > best[4]:
                best = candidate
            if (not self.enable_distractor) or exit_sep >= min_sep - 1e-6:
                break

        y_r, y_d, rail_hits, ball_hits, exit_sep, y_r0, y_d0 = best
        self.ball_start_y = float(y_r0)
        self.ball_path = path.copy()
        self.ball_path[:, 1] = y_r
        self.drop_wall_bounces = int(rail_hits)
        self.ball_ball_bounces = int(ball_hits)
        self.exit_separation = float(exit_sep)

        if self.enable_distractor:
            self.distractor_start_y = float(y_d0)
            self.distractor_path = path.copy()
            self.distractor_path[:, 1] = y_d
            d_steps = np.linalg.norm(np.diff(self.distractor_path, axis=0), axis=1)
            self.distractor_path_cumulative = np.concatenate(
                [[0.0], np.cumsum(d_steps)]
            )
            self.distractor_path_length = float(self.distractor_path_cumulative[-1])
            self.distractor_start = self.distractor_path[0].copy()
            self.distractor_exit = self.distractor_path[-1].copy()
            d_exit_tangent = self.distractor_path[-1] - self.distractor_path[-2]
            d_horizontal = d_exit_tangent[:2]
            self.distractor_release_direction_xy = (
                d_horizontal / max(np.linalg.norm(d_horizontal), 1e-6)
            )
            self.exit_separation = float(abs(y_r[-1] - y_d[-1]))
        else:
            self.distractor_start_y = None
            self.distractor_path = None
            self.distractor_path_cumulative = None
            self.distractor_path_length = 0.0
            self.distractor_start = None
            self.distractor_exit = None
            self.distractor_release_direction_xy = None

    def _spawn_distractor(self):
        """Create the black distractor at its baked drop pose."""
        self.distractor_drop = self.distractor_start + np.array([
            -self.side * self._drop_forward_travel,
            0.0,
            self.drop_height,
        ])
        self.distractor = create_sphere(
            self.scene,
            sapien.Pose(self.distractor_drop.tolist()),
            radius=self.ball_radius,
            color=self.distractor_color,
            is_static=False,
            name="valley_distractor_ball",
        )
        self._distractor_rigid = self._get_rigid(self.distractor)
        if self._distractor_rigid is not None:
            self._distractor_rigid.set_mass(self.ball_mass)
            self._distractor_rigid.set_disable_gravity(True)
            self._distractor_rigid.set_kinematic(True)
            self._distractor_rigid.set_kinematic_target(
                sapien.Pose(self.distractor_drop.tolist())
            )
            material = sapien.physx.PhysxMaterial(
                static_friction=0.12,
                dynamic_friction=0.08,
                restitution=0.22,
            )
            for shape in self._distractor_rigid.get_collision_shapes():
                shape.set_physical_material(material)

    def _reflect_lane(self, raw_y):
        """Reflect an unbounded lateral coordinate between the two rails."""
        width = self._lane_max - self._lane_min
        if width <= 1e-9:
            return float(self.ramp_center_y)
        phase = (float(raw_y) - self._lane_min) % (2.0 * width)
        if phase <= width:
            return float(self._lane_min + phase)
        return float(self._lane_max - (phase - width))

    def _make_curved_surface(self):
        """Create a smooth U profile with a horizontal tangent at the valley."""
        valley_z = self.table_top + self.valley_height
        points = []
        for index in range(self.curve_segments + 1):
            u = index / float(self.curve_segments)
            points.append([
                self.valley_x - self.down_run * (1.0 - u),
                self.ramp_center_y,
                valley_z + self.down_rise * (1.0 - u) ** 2,
            ])
        for index in range(1, self.curve_segments + 1):
            u = index / float(self.curve_segments)
            points.append([
                self.valley_x + self.up_run * u,
                self.ramp_center_y,
                valley_z + self.up_rise * u ** 2,
            ])
        return np.asarray(points, dtype=np.float64)

    def _ramp_segment_quat(self, point_a, point_b):
        """Board pose: local +X along the segment, local +Z outward/up."""
        delta = np.asarray(point_b, dtype=np.float64) - np.asarray(point_a, dtype=np.float64)
        length = float(np.linalg.norm(delta))
        if length < 1e-9:
            return t3d.euler.euler2quat(0.0, 0.0, 0.0)
        tangent = delta / length
        # 90° XZ rotation of the tangent; keep the upward branch so a mirrored
        # (−x) layout does not invert the board/rails into the table.
        normal = np.array([-tangent[2], 0.0, tangent[0]], dtype=np.float64)
        if normal[2] < 0.0:
            normal = -normal
        nlen = float(np.linalg.norm(normal))
        normal = (
            np.array([0.0, 0.0, 1.0], dtype=np.float64)
            if nlen < 1e-9
            else normal / nlen
        )
        binormal = np.cross(normal, tangent)
        blen = float(np.linalg.norm(binormal))
        binormal = (
            np.array([0.0, 1.0, 0.0], dtype=np.float64)
            if blen < 1e-9
            else binormal / blen
        )
        # Prefer width along +Y without flipping the upward normal: reverse
        # tangent+binormal together (keeps a right-handed frame).
        if binormal[1] < 0.0:
            tangent = -tangent
            binormal = -binormal
        rot = np.column_stack([tangent, binormal, normal])
        if float(np.linalg.det(rot)) < 0.0:
            binormal = -binormal
            rot = np.column_stack([tangent, binormal, normal])
        return t3d.quaternions.mat2quat(rot)

    def _build_ramp_segment(self, name, point_a, point_b):
        center = 0.5 * (np.asarray(point_a, dtype=np.float64) + np.asarray(point_b, dtype=np.float64))
        length = float(np.linalg.norm(np.asarray(point_b) - np.asarray(point_a)))
        quat = self._ramp_segment_quat(point_a, point_b)
        board = create_box(
            self.scene,
            sapien.Pose(center.tolist(), quat.tolist()),
            half_size=[length * 0.5 + 0.0002, self.ramp_half_width, self.ramp_thickness * 0.5],
            color=[0.68, 0.58, 0.43],
            is_static=True,
            name=name,
        )
        self.ramp_parts.append(board)

        rotation = t3d.quaternions.quat2mat(quat)
        for side, suffix in ((-1.0, "left"), (1.0, "right")):
            local_offset = np.array([
                0.0,
                side * (self.ramp_half_width - self.rail_thickness * 0.5),
                0.5 * (self.ramp_thickness + self.rail_height),
            ])
            rail_center = center + rotation @ local_offset
            rail = create_box(
                self.scene,
                sapien.Pose(rail_center.tolist(), quat.tolist()),
                half_size=[length * 0.5 + 0.0002, self.rail_thickness * 0.5, self.rail_height * 0.5],
                color=[0.42, 0.32, 0.22],
                is_static=True,
                name=f"{name}_{suffix}_rail",
            )
            self.ramp_parts.append(rail)

    def _build_support(self, name, surface_point):
        underside_z = float(surface_point[2] - self.ramp_thickness)
        height = max(0.01, underside_z - self.table_top)
        support = create_box(
            self.scene,
            sapien.Pose([
                float(surface_point[0]),
                self.ramp_center_y,
                self.table_top + 0.5 * height,
            ]),
            half_size=[0.018, 0.035, 0.5 * height],
            color=[0.38, 0.32, 0.27],
            is_static=True,
            name=name,
        )
        self.ramp_parts.append(support)

    def _get_rigid(self, actor):
        entity = actor.actor if hasattr(actor, "actor") else actor
        for component in entity.get_components():
            if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
                return component
        return None

    def _collect_robot_link_names(self):
        names = set()
        for articulation in (self.robot.left_entity, self.robot.right_entity):
            if articulation is None:
                continue
            for link in articulation.get_links():
                names.add(link.get_name())
        return names

    def _end_effector_position(self, arm_tag):
        if arm_tag == "right":
            return np.asarray(self.robot.get_right_ee_pose()[:3])
        return np.asarray(self.robot.get_left_ee_pose()[:3])

    def _weld_bowl_to_end_effector(self, arm_tag):
        """Make the rim grasp deterministic while preserving the arm motion."""
        self._bowl_arm = str(arm_tag)
        self._bowl_ee_offset = (
            np.asarray(self.bowl.get_pose().p)
            - self._end_effector_position(self._bowl_arm)
        )
        rigid = self._get_rigid(self.bowl)
        if rigid is not None:
            rigid.set_disable_gravity(True)
            rigid.set_kinematic(True)
            rigid.set_linear_velocity(np.zeros(3))
            rigid.set_angular_velocity(np.zeros(3))
            rigid.set_kinematic_target(self.bowl.get_pose())
        self._bowl_welded = True

    def _update_welded_bowl(self):
        if not self._bowl_welded:
            return
        rigid = self._get_rigid(self.bowl)
        target_position = (
            self._end_effector_position(self._bowl_arm)
            + self._bowl_ee_offset
        )
        target_position[2] = max(float(target_position[2]), self.table_top)
        target = sapien.Pose(target_position.tolist(), self.bowl.get_pose().q)
        # Apply the pose immediately for rendering and also set the PhysX
        # target for the next step. A target alone can visibly lag behind fast
        # arm motion and briefly tunnel the small bowl through the tabletop.
        self.bowl.actor.set_pose(target)
        if rigid is not None:
            rigid.set_kinematic_target(target)

    def _unweld_bowl(self):
        if not self._bowl_welded:
            return
        rigid = self._get_rigid(self.bowl)
        if rigid is not None:
            # Release at the arm's actual drop pose and let gravity settle the
            # bowl onto the tabletop. No pose correction or teleport is used.
            rigid.set_kinematic(False)
            rigid.set_disable_gravity(False)
            rigid.set_linear_velocity(np.zeros(3))
            rigid.set_angular_velocity(np.zeros(3))
        self._bowl_welded = False

    def _fix_bowl_at_placed_pose(self):
        """Lock the bowl only after the arm has released it on the table."""
        rigid = self._get_rigid(self.bowl)
        if rigid is None:
            return
        placed_pose = self.bowl.get_pose()
        rigid.set_linear_velocity(np.zeros(3))
        rigid.set_angular_velocity(np.zeros(3))
        rigid.set_disable_gravity(True)
        rigid.set_kinematic(True)
        rigid.set_kinematic_target(placed_pose)

    def _update_release_velocity(self):
        horizontal_speed = self.launch_speed * np.cos(self.up_angle)
        self.release_velocity = np.array([
            horizontal_speed * self.release_direction_xy[0],
            horizontal_speed * self.release_direction_xy[1],
            self.launch_speed * np.sin(self.up_angle),
        ])

    def _update_distractor_release_velocity(self):
        if not self.enable_distractor:
            return
        horizontal_speed = self.launch_speed * np.cos(self.up_angle)
        self.distractor_release_velocity = np.array([
            horizontal_speed * self.distractor_release_direction_xy[0],
            horizontal_speed * self.distractor_release_direction_xy[1],
            self.launch_speed * np.sin(self.up_angle),
        ])

    def _compute_landing(self):
        # Solve z(t) = bowl-rest height for the positive projectile root.
        rest_z = self.table_top + self.ball_radius + 0.012
        z0 = float(self.ball_exit[2])
        vz = float(self.release_velocity[2])
        discriminant = max(0.0, vz * vz + 2.0 * self.GRAVITY * (z0 - rest_z))
        self.flight_time = (vz + np.sqrt(discriminant)) / self.GRAVITY
        self.landing = self.ball_exit[:2] + self.release_velocity[:2] * self.flight_time

    def _tune_launch_to_landing(self, target_x, tol=0.012, max_iters=45):
        """Nudge ``launch_speed`` so predicted landing x matches ``target_x``."""
        target_x = float(target_x)
        for _ in range(int(max_iters)):
            self._update_release_velocity()
            self._compute_landing()
            err = float(self.side) * (float(self.landing[0]) - target_x)
            if abs(err) <= float(tol):
                break
            if err < 0.0:
                self.launch_speed *= 1.06
            else:
                self.launch_speed *= 0.94
            self.launch_speed = float(np.clip(self.launch_speed, 0.12, 2.8))
        self._update_release_velocity()
        self._compute_landing()

    def _compute_distractor_landing(self):
        if not self.enable_distractor or self.distractor_exit is None:
            return
        rest_z = self.table_top + self.ball_radius + 0.012
        z0 = float(self.distractor_exit[2])
        vz = float(self.distractor_release_velocity[2])
        discriminant = max(0.0, vz * vz + 2.0 * self.GRAVITY * (z0 - rest_z))
        flight_time = (vz + np.sqrt(discriminant)) / self.GRAVITY
        self.distractor_landing = (
            self.distractor_exit[:2]
            + self.distractor_release_velocity[:2] * flight_time
        )

    # ---------------------------------------------------------- ball motion
    def _freeze_ball(self):
        self.ball.set_pose(sapien.Pose(self.ball_drop.tolist()))
        if self._ball_rigid is not None:
            self._ball_rigid.set_linear_velocity(np.zeros(3))
            self._ball_rigid.set_angular_velocity(np.zeros(3))
            self._ball_rigid.set_disable_gravity(True)
            self._ball_rigid.set_kinematic(True)
            self._ball_rigid.set_kinematic_target(sapien.Pose(self.ball_drop.tolist()))
        self._ball_phase = "frozen"
        if self.enable_distractor and self.distractor is not None:
            self.distractor.set_pose(sapien.Pose(self.distractor_drop.tolist()))
            if self._distractor_rigid is not None:
                self._distractor_rigid.set_linear_velocity(np.zeros(3))
                self._distractor_rigid.set_angular_velocity(np.zeros(3))
                self._distractor_rigid.set_disable_gravity(True)
                self._distractor_rigid.set_kinematic(True)
                self._distractor_rigid.set_kinematic_target(
                    sapien.Pose(self.distractor_drop.tolist())
                )
            self._distractor_phase = "frozen"

    def _start_ball_motion(self, expert_demo):
        if not getattr(self, "_loaded", False):
            return
        self._expert_demo = bool(expert_demo)
        self._bowl_ready = False
        self._arm_ball_contact = False
        self._ball_phase = "dropping"
        self._drop_i = 0
        self._roll_i = 0
        self._roll_distance = 0.0
        self._distractor_roll_distance = 0.0
        self._physics_step_count = 0
        self.ball.set_pose(sapien.Pose(self.ball_drop.tolist()))
        if self._ball_rigid is not None:
            self._ball_rigid.set_disable_gravity(True)
            self._ball_rigid.set_kinematic(True)
            self._ball_rigid.set_linear_velocity(np.zeros(3))
            self._ball_rigid.set_angular_velocity(np.zeros(3))
            self._ball_rigid.set_kinematic_target(sapien.Pose(self.ball_drop.tolist()))
        if self.enable_distractor and self.distractor is not None:
            self._distractor_phase = "dropping"
            self.distractor.set_pose(sapien.Pose(self.distractor_drop.tolist()))
            if self._distractor_rigid is not None:
                self._distractor_rigid.set_disable_gravity(True)
                self._distractor_rigid.set_kinematic(True)
                self._distractor_rigid.set_linear_velocity(np.zeros(3))
                self._distractor_rigid.set_angular_velocity(np.zeros(3))
                self._distractor_rigid.set_kinematic_target(
                    sapien.Pose(self.distractor_drop.tolist())
                )

    def _ball_touching_rail(self):
        ball_name = self.ball.get_name()
        for contact in self.scene.get_contacts():
            name0 = contact.bodies[0].entity.name
            name1 = contact.bodies[1].entity.name
            if ball_name not in (name0, name1):
                continue
            other = name1 if name0 == ball_name else name0
            if other.endswith("_left_rail") or other.endswith("_right_rail"):
                return True
        return False

    def _simulate_ball_landing(self):
        """Run and reset a physical rollout to predict the table impact point."""
        required_x = self.red_line_x + self.side * (self.bowl_outer_radius + 0.005)
        best_landing = self.landing.copy()
        best_steps = self.physics_max_steps
        best_bounces = 0

        for _ in range(5):
            self._start_ball_motion(expert_demo=False)
            touching_rail = False
            bounce_count = 0
            landing = None
            landing_step = self.physics_max_steps
            for step in range(self.physics_max_steps):
                self.scene.step()
                rail_contact = self._ball_touching_rail()
                if rail_contact and not touching_rail:
                    bounce_count += 1
                touching_rail = rail_contact
                position = np.asarray(self.ball.get_pose().p)
                left_ramp = (
                    self.side * position[0]
                    > self.side * self.ramp_exit_x + self.ball_radius
                )
                reached_table = position[2] <= self.table_top + self.ball_radius + 0.006
                if left_ramp and reached_table:
                    landing = position[:2].copy()
                    landing_step = step + 1
                    break
                if position[2] < self.table_top - 0.20:
                    break

            if landing is not None:
                best_landing = landing
                best_steps = landing_step
                best_bounces = bounce_count
                landing_in_lane = (
                    abs(landing[1] - self.ramp_center_y)
                    <= self.ramp_half_width - self.bowl_outer_radius
                )
                if self.side * landing[0] >= self.side * required_x and landing_in_lane:
                    break

            # Preserve the sampled angles but add speed if the physical ball
            # does not clear the mandatory catch boundary.
            if landing is None or self.side * landing[0] < self.side * required_x:
                self.initial_forward_speed *= 1.15
            if (
                landing is None
                or abs(landing[1] - self.ramp_center_y)
                > self.ramp_half_width - self.bowl_outer_radius
            ):
                self.initial_lateral_speed *= 0.5
            self.initial_vertical_speed = (
                self.initial_forward_speed
                * np.tan(np.deg2rad(self.drop_forward_angle_deg))
            )
            self.initial_ball_velocity = np.array([
                self.initial_forward_speed,
                self.initial_lateral_speed,
                self.initial_vertical_speed,
            ])
            self._freeze_ball()
            self.scene.step()

        self.landing = np.asarray(best_landing)
        self.physics_run_steps = min(
            self.physics_max_steps,
            best_steps + self.settle_steps,
        )
        self.drop_wall_bounces = int(best_bounces)
        self._freeze_ball()
        self.scene.step()
        self._arm_ball_contact = False

    def _set_kinematic_ball_pose(self, position, roll_distance=0.0, actor=None, rigid=None):
        # A sphere moving toward +x rolls around the negative y axis.
        roll_angle = -float(roll_distance) / max(self.ball_radius, 1e-6)
        quat = t3d.quaternions.axangle2quat([0.0, 1.0, 0.0], roll_angle)
        pose = sapien.Pose(np.asarray(position).tolist(), quat.tolist())
        target_rigid = rigid if rigid is not None else self._ball_rigid
        target_actor = actor if actor is not None else self.ball
        if target_rigid is not None:
            target_rigid.set_kinematic_target(pose)
        else:
            target_actor.set_pose(pose)

    def _advance_ball(self):
        if self._ball_phase == "dropping":
            self._drop_i += 1
            fraction = min(1.0, self._drop_i / float(self.drop_steps))
            vertical_fraction = fraction * fraction
            position = np.array([
                self.ball_drop[0] + (self.ball_start[0] - self.ball_drop[0]) * fraction,
                self.ball_start[1],
                self.ball_drop[2] + (self.ball_start[2] - self.ball_drop[2]) * vertical_fraction,
            ])
            self._set_kinematic_ball_pose(position)
            if self.enable_distractor and self._distractor_phase == "dropping":
                distractor_position = np.array([
                    self.distractor_drop[0]
                    + (self.distractor_start[0] - self.distractor_drop[0]) * fraction,
                    self.distractor_start[1],
                    self.distractor_drop[2]
                    + (self.distractor_start[2] - self.distractor_drop[2]) * vertical_fraction,
                ])
                self._set_kinematic_ball_pose(
                    distractor_position,
                    actor=self.distractor,
                    rigid=self._distractor_rigid,
                )
            if fraction >= 1.0:
                self._ball_phase = "rolling"
                if self.enable_distractor:
                    self._distractor_phase = "rolling"

        elif self._ball_phase == "rolling":
            self._roll_i += 1
            fraction = min(1.0, self._roll_i / float(self.roll_steps))
            accelerated_fraction = (
                (1.0 - self.roll_acceleration) * fraction
                + self.roll_acceleration * fraction * fraction
            )
            distance = accelerated_fraction * self.ball_path_length
            segment = int(np.searchsorted(self.ball_path_cumulative, distance, side="right") - 1)
            segment = int(np.clip(segment, 0, len(self.ball_path) - 2))
            segment_start = self.ball_path_cumulative[segment]
            segment_length = (
                self.ball_path_cumulative[segment + 1] - segment_start
            )
            local = (distance - segment_start) / max(segment_length, 1e-6)
            position = (
                self.ball_path[segment]
                + (self.ball_path[segment + 1] - self.ball_path[segment]) * local
            )
            self._roll_distance = distance
            self._set_kinematic_ball_pose(position, distance)

            if self.enable_distractor and self._distractor_phase == "rolling":
                d_distance = accelerated_fraction * self.distractor_path_length
                d_segment = int(
                    np.searchsorted(self.distractor_path_cumulative, d_distance, side="right") - 1
                )
                d_segment = int(np.clip(d_segment, 0, len(self.distractor_path) - 2))
                d_start = self.distractor_path_cumulative[d_segment]
                d_len = self.distractor_path_cumulative[d_segment + 1] - d_start
                d_local = (d_distance - d_start) / max(d_len, 1e-6)
                d_position = (
                    self.distractor_path[d_segment]
                    + (self.distractor_path[d_segment + 1] - self.distractor_path[d_segment])
                    * d_local
                )
                self._distractor_roll_distance = d_distance
                self._set_kinematic_ball_pose(
                    d_position,
                    d_distance,
                    actor=self.distractor,
                    rigid=self._distractor_rigid,
                )

            if fraction >= 1.0 and (self._bowl_ready or not self._expert_demo):
                self._release_ball()

    def _release_ball(self):
        if self._ball_phase == "released":
            return
        self._ball_phase = "released"
        self.ball.set_pose(sapien.Pose(self.ball_exit.tolist()))
        if self._ball_rigid is not None:
            self._ball_rigid.set_kinematic(False)
            self._ball_rigid.set_disable_gravity(False)
            self._ball_rigid.set_linear_velocity(self.release_velocity)
            self._ball_rigid.set_angular_velocity([
                0.0,
                -self.side * self.launch_speed / max(self.ball_radius, 1e-6),
                0.0,
            ])
            self._ball_rigid.set_linear_damping(0.08)
            self._ball_rigid.set_angular_damping(0.5)

        if self.enable_distractor and self.distractor is not None:
            self._distractor_phase = "released"
            self.distractor.set_pose(sapien.Pose(self.distractor_exit.tolist()))
            if self._distractor_rigid is not None:
                self._distractor_rigid.set_kinematic(False)
                self._distractor_rigid.set_disable_gravity(False)
                self._distractor_rigid.set_linear_velocity(self.distractor_release_velocity)
                self._distractor_rigid.set_angular_velocity([
                    0.0,
                    -self.side * self.launch_speed / max(self.ball_radius, 1e-6),
                    0.0,
                ])
                self._distractor_rigid.set_linear_damping(0.08)
                self._distractor_rigid.set_angular_damping(0.5)

    def _check_arm_ball_contact(self):
        if self._arm_ball_contact or not getattr(self, "_loaded", False):
            return
        # After the bowl is placed, arm retreat can brush the vessel/ball and
        # must not count as an illegal arm catch.
        if getattr(self, "_bowl_ready", False):
            return
        ball_name = self.ball.get_name()
        for contact in self.scene.get_contacts():
            name0 = contact.bodies[0].entity.name
            name1 = contact.bodies[1].entity.name
            if (
                (name0 == ball_name and name1 in self._robot_link_names)
                or (name1 == ball_name and name0 in self._robot_link_names)
            ):
                self._arm_ball_contact = True
                return

    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
        if not getattr(self, "_loaded", False):
            return
        if self._ball_phase == "released":
            self._physics_step_count += 1
        self._check_arm_ball_contact()
        self._update_welded_bowl()
        self._advance_ball()

    def _dwell(self, steps):
        for i in range(max(0, int(steps))):
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and i % self.save_freq == 0:
                self._take_picture()

    # --------------------------------------------------------------- policy
    def play_once(self):
        arm_tag = ArmTag("left" if self.mirrored else "right")

        # Start the ball before the first robot command. Its drop and
        # accelerated roll advance concurrently with all arm actions.
        self._start_ball_motion(expert_demo=True)
        self.move(self.grasp_actor(self.bowl, arm_tag=arm_tag, pre_grasp_dis=0.10))
        self._weld_bowl_to_end_effector(arm_tag)
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.12, move_axis="arm"))

        bowl_now = np.asarray(self.bowl.get_pose().p)
        target = np.array([
            self._catch_target_x(self.landing[0]),
            self.landing[1],
            self.table_top + self.BOWL_PLACE_Z_OFFSET,
        ])
        displacement = target - bowl_now
        self.move(self.move_by_displacement(
            arm_tag=arm_tag,
            x=float(displacement[0]),
            y=float(displacement[1]),
            z=float(displacement[2]),
            move_axis="world",
        ))
        # Detach at table height first; the bowl then rests on the table while
        # the fingers open instead of following a moving gripper and dropping.
        self._unweld_bowl()
        self.move(self.open_gripper(arm_tag))
        self._dwell(20)
        self._fix_bowl_at_placed_pose()
        self.move(self.back_to_origin(arm_tag))

        self._bowl_ready = True
        remaining_drop = max(0, self.drop_steps - self._drop_i)
        remaining_roll = max(0, self.roll_steps - self._roll_i)
        self._dwell(
            remaining_drop
            + remaining_roll
            + self.flight_steps
            + self.settle_steps
        )
        self._check_arm_ball_contact()

        self.info["info"] = {
            "{A}": "valley ball",
            "{B}": f"{self.catcher_model}/base{self.bowl_id}",
            "{a}": str(arm_tag),
            "{opt}": self._option_label(),
            "{flip}": "mirrored" if self.mirrored else "default",
        }
        return self.info

    # -------------------------------------------------------------- success
    def _ball_in_bowl(self, ball_actor):
        if ball_actor is None:
            return False
        ball_position = np.asarray(ball_actor.get_pose().p)
        bowl_position = np.asarray(self.bowl.get_pose().p)
        horizontal_offset = float(np.linalg.norm(ball_position[:2] - bowl_position[:2]))
        in_height = (
            bowl_position[2] - 0.01
            <= ball_position[2]
            <= bowl_position[2] + self.bowl_height + 2.0 * self.ball_radius
        )
        return bool(horizontal_offset < self.bowl_inner_radius and in_height)

    def _catch_state(self):
        ball_position = np.asarray(self.ball.get_pose().p)
        bowl_position = np.asarray(self.bowl.get_pose().p)
        horizontal_offset = float(np.linalg.norm(ball_position[:2] - bowl_position[:2]))
        in_height = (
            bowl_position[2] - 0.01
            <= ball_position[2]
            <= bowl_position[2] + self.bowl_height + 2.0 * self.ball_radius
        )
        in_bowl = bool(horizontal_offset < self.bowl_inner_radius and in_height)
        # X-only: fully clear of the red line on the catch side of this layout.
        # Left/mirrored → further −x; right → further +x. On the line = fail.
        line_clearance = self.bowl_outer_radius + 0.008
        behind_line = bool(
            self._past_red_line_x(bowl_position[0], margin=line_clearance)
        )
        distractor_in_bowl = bool(
            self.enable_distractor and self._ball_in_bowl(self.distractor)
        )
        return (
            horizontal_offset,
            in_bowl,
            behind_line,
            ball_position,
            bowl_position,
            distractor_in_bowl,
        )

    def check_success(self):
        if not getattr(self, "_loaded", False) or self._ball_phase != "released":
            self._last_fail_reason = "ball_not_released"
            return False
        self._check_arm_ball_contact()
        _, in_bowl, behind_line, _, bowl_position, distractor_in_bowl = self._catch_state()
        # Catching the black distractor is an explicit failure.
        if distractor_in_bowl:
            self._last_fail_reason = "distractor_in_bowl"
            return False
        if self._arm_ball_contact:
            self._last_fail_reason = "arm_ball_contact"
            return False
        if not in_bowl:
            self._last_fail_reason = "ball_not_in_bowl"
            return False
        if not behind_line:
            side = "left/−x" if float(self.side) < 0.0 else "right/+x"
            self._last_fail_reason = (
                f"bowl_not_past_red_line({side}: bowl_x={float(bowl_position[0]):.3f}, "
                f"red_line_x={float(self.red_line_x):.3f})"
            )
            return False
        self._last_fail_reason = ""
        return True

    def get_obs(self):
        obs = super().get_obs()
        try:
            (
                offset,
                in_bowl,
                behind_line,
                ball_position,
                bowl_position,
                distractor_in_bowl,
            ) = self._catch_state()
            distractor_position = (
                list(map(float, self.distractor.get_pose().p))
                if self.enable_distractor and self.distractor is not None
                else [0.0, 0.0, 0.0]
            )
            obs["valley_catch"] = {
                "ball_position": list(map(float, ball_position)),
                "bowl_position": list(map(float, bowl_position)),
                "predicted_landing": list(map(float, self.landing)),
                "landing_gap": float(getattr(self, "landing_gap", 0.0)),
                "param_jitter": float(getattr(self, "param_jitter", 0.0)),
                "horizontal_offset": float(offset),
                "in_bowl": float(in_bowl),
                "bowl_behind_line": float(behind_line),
                "red_line_x": float(self.red_line_x),
                "arm_ball_contact": float(self._arm_ball_contact),
                "phase": str(self._ball_phase),
                "down_run": float(self.down_run),
                "down_rise": float(self.down_rise),
                "up_run": float(self.up_run),
                "up_rise": float(self.up_rise),
                "platform_length_scale": float(self.platform_length_scale),
                "drop_time": float(self.drop_time),
                "ball_path_mode": str(self.ball_path_mode),
                "wall_bounce_enabled": float(self.wall_bounce_enabled),
                "enable_distractor": float(self.enable_distractor),
                "drop_wall_angle_deg": float(self.drop_wall_angle_deg),
                "drop_forward_angle_deg": float(self.drop_forward_angle_deg),
                "drop_wall_bounces": int(self.drop_wall_bounces),
                "ball_ball_bounces": int(getattr(self, "ball_ball_bounces", 0)),
                "ball_start_y": float(getattr(self, "ball_start_y", self.ramp_center_y)),
                "distractor_start_y": float(
                    getattr(self, "distractor_start_y", 0.0) or 0.0
                ),
                "roll_time": float(self.roll_time),
                "launch_speed": float(self.launch_speed),
                "initial_forward_speed": float(self.initial_forward_speed),
                "curve_segments": int(self.curve_segments),
                "catcher_model": str(getattr(self, "catcher_model", "002_bowl")),
                "bowl_scale_mult": float(self.bowl_scale_mult),
                "distractor_position": distractor_position,
                "distractor_in_bowl": float(distractor_in_bowl),
                "exit_separation": float(getattr(self, "exit_separation", 0.0)),
                "mirrored": float(getattr(self, "mirrored", False)),
                "side": float(getattr(self, "side", 1.0)),
                "option_label": self._option_label(),
            }
        except Exception:
            pass
        return obs
