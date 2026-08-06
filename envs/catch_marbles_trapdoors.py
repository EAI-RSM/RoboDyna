from ._base_task import Base_Task
from .utils import *
import sapien
import sapien.physx
import numpy as np
import transforms3d as t3d
import pickle
import os
from .utils.dynamic_utils import DynamicMotionHelper, StepCounter
from .utils.save_file import save_pkl


class catch_marbles_trapdoors(Base_Task):
    """Press a colored button to open the matching trapdoor and catch the target marble.

    A shallow catch bowl sits under a trapdoor platform at the table centre (corner posts bridge
    the gap). Four colored button cubes sit in front of the fixture. The upper floor has four
    middle trapdoor tiles matching the button colors. Pressing a button opens the matching door.
    A settled marble sticks partly out of the bowl rim (``marble_protrude_frac``).

    A colored target marble travels left-to-right in the upper box. Success requires dropping it
    only through the trapdoor whose color matches the marble (below the trapdoor level, inside
    the lower box — settle/speed not required).

    Config options (task_args.catch_marbles_trapdoors):
      default — doors can be opened repeatedly with no limit (auto-close, then reopen).
      door_open_once / opt1 — each door may be opened only once, then stays locked closed.
      enable_distractor / opt2 — spawn a black distractor marble from the opposite direction
          at the same height as the target, offset on Y so the lanes do not collide
          (pass-through by default).
      target_color — null/random to pick 1 of the 4 trapdoor colors each episode, or a specific
          color name (red|yellow|blue|green) / index (0..3). The expert presses with the left arm
          when that trapdoor is on the left half, right arm when on the right half.
      shuffle_colors — if true (default), randomize left→right order of the four colors for both
          trapdoors and matching keys each episode.
    """

    N_BUTTONS_DEFAULT = 4
    BUTTON_X_DEFAULT = [-0.15, -0.05, 0.05, 0.15]
    BUTTON_Y_DEFAULT = -0.18
    # Colored keycap (pressed by the robot).
    BUTTON_HALF_DEFAULT = [0.020, 0.020, 0.014]
    # Larger, thinner black base under each key for a more realistic key look.
    KEY_BASE_HALF_DEFAULT = [0.032, 0.032, 0.005]
    KEY_BASE_COLOR_DEFAULT = [0.08, 0.08, 0.08]
    BUTTON_PRESS_DEPTH_DEFAULT = 0.03
    PRESS_HOLD_STEPS_DEFAULT = 20
    POST_PRESS_DWELL_DEFAULT = 10
    FINAL_DWELL_STEPS_DEFAULT = 80
    SHUFFLE_COLORS_DEFAULT = True
    # Reactive key — same spring proxy as fill_coffee_jar (FORCE_* defaults).
    BUTTON_TOUCH_XY_TOL = 0.055
    BUTTON_VISUAL_STEP = 0.0007
    BUTTON_FORCE_ENGAGE_SLACK = 0.05
    BUTTON_EE_TO_TCP = 0.12

    BOX_Y_DEFAULT = -0.02
    BOX_HALF_W_DEFAULT = 0.30
    BOX_HALF_D_DEFAULT = 0.055
    BOX_WALL_H_DEFAULT = 0.04
    BOX_WALL_T_DEFAULT = 0.006
    STACK_GAP_DEFAULT = None  # None = auto-clear a protruding marble under the upper floor
    # Fraction of marble diameter that sticks out above the lower catch-bowl rim.
    MARBLE_PROTRUDE_FRAC_DEFAULT = 0.40

    FLOOR_TILE_COUNT_DEFAULT = 6
    DOOR_WIDTH_DEFAULT = 0.10
    DOOR_OPEN_ANGLE_DEG_DEFAULT = 95.0
    DOOR_OPEN_SPEED_DEG_DEFAULT = 220.0
    DOOR_OPEN_DURATION_SEC_DEFAULT = 0.5
    DOOR_OPEN_ONCE_DEFAULT = False

    BALL_RADIUS_DEFAULT = 0.012
    BALL_SPEED_DEFAULT = 0.18
    BALL_SPEED_SCALE_MIN_DEFAULT = 0.8   # sample marble speeds in [scale_min, scale_max] * ball_speed
    BALL_SPEED_SCALE_MAX_DEFAULT = 1.2
    BALL_X_MARGIN_DEFAULT = 0.02
    BALL_BOUNCE_HEIGHT_DEFAULT = 0.012
    BALL_BOUNCE_FREQ_DEFAULT = 8.0
    BALL_Y_OFFSET_DEFAULT = 0.0
    BALL_DROP_SETTLE_STEPS_DEFAULT = 80

    ENABLE_DISTRACTOR_DEFAULT = False
    DISTRACTOR_COLOR_DEFAULT = [0.05, 0.05, 0.05]
    DISTRACTOR_COLLIDE_DEFAULT = False  # default pass-through; Y lanes keep marbles from overlapping
    DISTRACTOR_SPEED_SCALE_MIN_DEFAULT = 0.8
    DISTRACTOR_SPEED_SCALE_MAX_DEFAULT = 1.2
    # Same Z as the target (0). Separate lanes on Y so paths do not collide.
    DISTRACTOR_HEIGHT_OFFSET_DEFAULT = 0.0
    DISTRACTOR_Y_OFFSET_DEFAULT = 0.030

    COLOR_NAME_ALIASES = {
        "red": "red",
        "yellow": "yellow",
        "orange": "yellow",
        "blue": "blue",
        "green": "green",
    }

    def setup_demo(self, **kwags):
        self._cfg = kwags.get("task_args", {}).get("catch_marbles_trapdoors", {})
        self.buttons = []
        self.button_bases = []
        self.door_tiles = []
        self._button_pressed = []
        self._door_open = []
        self._door_open_with_ball_over = []
        self._door_angle_deg = []
        self._door_target_angle_deg = []
        self._door_open_time_left = []
        self._door_locked_closed = []
        self._door_x_bounds = []
        self._door_hinge_y = 0.0
        self._door_half_y = 0.0
        self._door_half_x = 0.0
        self._door_floor_z = 0.0
        self.door_width = float(self.DOOR_WIDTH_DEFAULT)
        self.door_open_angle_deg = float(self.DOOR_OPEN_ANGLE_DEG_DEFAULT)
        self.door_open_speed_deg = float(self.DOOR_OPEN_SPEED_DEG_DEFAULT)
        self.door_open_duration_sec = float(self.DOOR_OPEN_DURATION_SEC_DEFAULT)
        self.door_open_once = bool(self.DOOR_OPEN_ONCE_DEFAULT)
        self.button_colors = []
        self.button_color_names = []
        self.target_button_idx = -1
        self._buttons_held = set()
        self._reactive_buttons = None
        self._mutex_violation = False
        self.shuffle_colors = bool(self.SHUFFLE_COLORS_DEFAULT)
        self.key_base_half = list(self.KEY_BASE_HALF_DEFAULT)
        self.key_base_color = list(self.KEY_BASE_COLOR_DEFAULT)
        self.ball = None
        self._ball_rigid = None
        self._ball_dir = 1.0
        self._ball_phase = 0.0
        self._ball_mode = "track"
        self._ball_x_min = 0.0
        self._ball_x_max = 0.0
        self._ball_y = 0.0
        self._ball_z_base = 0.0
        self.ball_speed = float(self.BALL_SPEED_DEFAULT)
        self.ball_bounce_height = float(self.BALL_BOUNCE_HEIGHT_DEFAULT)
        self.ball_bounce_freq = float(self.BALL_BOUNCE_FREQ_DEFAULT)
        self.ball_drop_settle_steps = int(self.BALL_DROP_SETTLE_STEPS_DEFAULT)
        self.lower_box_wall_h = float(self.BOX_WALL_H_DEFAULT)
        self.upper_box_wall_h = float(self.BOX_WALL_H_DEFAULT)
        self._press_lead_steps = 0
        self._lower_box_floor_z = 0.0
        self._lower_box_top_z = 0.0
        self._upper_box_floor_z = 0.0
        self._upper_box_top_z = 0.0
        self._ball_drop_door_idx = -1
        self.enable_distractor = bool(self.ENABLE_DISTRACTOR_DEFAULT)
        self.distractor = None
        self._distractor_rigid = None
        self._distractor_dir = -1.0
        self._distractor_phase = 0.0
        self._distractor_mode = "track"
        self._distractor_speed = float(self.BALL_SPEED_DEFAULT)
        self._distractor_drop_door_idx = -1
        self._distractor_z_base = 0.0
        self._distractor_y = 0.0
        self.distractor_height_offset = float(self.DISTRACTOR_HEIGHT_OFFSET_DEFAULT)
        self.distractor_y_offset = float(self.DISTRACTOR_Y_OFFSET_DEFAULT)
        self.distractor_collide = False
        super()._init_task_env_(**kwags)

    # ------------------------------------------------------------------ actors
    def load_actors(self):
        c = self._cfg
        self.n_buttons = int(c.get("n_buttons", self.N_BUTTONS_DEFAULT))
        self.button_y = float(c.get("button_y", self.BUTTON_Y_DEFAULT))
        self.button_half = list(c.get("button_half", self.BUTTON_HALF_DEFAULT))
        self.key_base_half = list(c.get("key_base_half", self.KEY_BASE_HALF_DEFAULT))
        self.key_base_color = list(c.get("key_base_color", self.KEY_BASE_COLOR_DEFAULT))
        self.button_press_depth = float(c.get("button_press_depth", self.BUTTON_PRESS_DEPTH_DEFAULT))
        self.press_hold_steps = int(c.get("press_hold_steps", self.PRESS_HOLD_STEPS_DEFAULT))
        self.post_press_dwell = int(c.get("post_press_dwell", self.POST_PRESS_DWELL_DEFAULT))
        self.final_dwell_steps = int(c.get("final_dwell_steps", self.FINAL_DWELL_STEPS_DEFAULT))
        self.shuffle_colors = self._parse_bool(
            c.get("shuffle_colors", self.SHUFFLE_COLORS_DEFAULT),
            default=self.SHUFFLE_COLORS_DEFAULT,
        )

        self.box_y = float(c.get("box_y", self.BOX_Y_DEFAULT))
        self.box_half_w = float(c.get("box_half_w", self.BOX_HALF_W_DEFAULT))
        self.box_half_d = float(c.get("box_half_d", self.BOX_HALF_D_DEFAULT))
        self.box_wall_h = float(c.get("box_wall_h", self.BOX_WALL_H_DEFAULT))
        self.box_wall_t = float(c.get("box_wall_t", self.BOX_WALL_T_DEFAULT))
        self.upper_box_wall_h = float(c.get("upper_box_wall_h", self.box_wall_h))

        self.floor_tile_count = int(c.get("floor_tile_count", self.FLOOR_TILE_COUNT_DEFAULT))
        self.door_width = float(c.get("door_width", self.DOOR_WIDTH_DEFAULT))
        self.door_open_angle_deg = float(c.get("door_open_angle_deg", self.DOOR_OPEN_ANGLE_DEG_DEFAULT))
        self.door_open_speed_deg = float(c.get("door_open_speed_deg", self.DOOR_OPEN_SPEED_DEG_DEFAULT))
        self.door_open_duration_sec = float(c.get("door_open_duration_sec", self.DOOR_OPEN_DURATION_SEC_DEFAULT))
        self.door_open_once = self._parse_bool(
            c.get("door_open_once", c.get("opt1", self.DOOR_OPEN_ONCE_DEFAULT)),
            default=self.DOOR_OPEN_ONCE_DEFAULT,
        )
        button_x_cfg = c.get("button_x", None)
        self.button_x = list(button_x_cfg) if button_x_cfg is not None else self._aligned_button_x_positions()

        self.ball_radius = float(np.clip(float(c.get("ball_radius", self.BALL_RADIUS_DEFAULT)), 0.008, 0.03))
        # Shallow catch bowl: rim low enough that part of a settled marble sticks out.
        protrude = float(np.clip(
            float(c.get("marble_protrude_frac", self.MARBLE_PROTRUDE_FRAC_DEFAULT)),
            0.15,
            0.60,
        ))
        self.marble_protrude_frac = protrude
        default_lower_h = float(
            self.box_wall_t / 2.0 + 2.0 * self.ball_radius * (1.0 - protrude)
        )
        self.lower_box_wall_h = float(c.get("lower_box_wall_h", default_lower_h))
        # Keep the trapdoor floor just above a protruding marble (unless overridden).
        marble_top_from_floor_z = float(self.box_wall_t / 2.0 + 2.0 * self.ball_radius)
        auto_stack_gap = max(
            0.0,
            marble_top_from_floor_z + 0.008 - self.lower_box_wall_h - self.box_wall_t / 2.0,
        )
        stack_gap_cfg = c.get("stack_gap", self.STACK_GAP_DEFAULT)
        self.stack_gap = float(auto_stack_gap if stack_gap_cfg is None else stack_gap_cfg)
        # ball_speed is the nominal default; per-episode speeds sample ±20% around it.
        self.ball_speed_default = float(c.get("ball_speed", self.BALL_SPEED_DEFAULT))
        self.ball_speed_scale_min = float(c.get("ball_speed_scale_min", self.BALL_SPEED_SCALE_MIN_DEFAULT))
        self.ball_speed_scale_max = float(c.get("ball_speed_scale_max", self.BALL_SPEED_SCALE_MAX_DEFAULT))
        self.ball_speed = self._sample_speed_from_default(
            self.ball_speed_default,
            self.ball_speed_scale_min,
            self.ball_speed_scale_max,
        )
        self.ball_x_margin = float(c.get("ball_x_margin", self.BALL_X_MARGIN_DEFAULT))
        self.ball_bounce_height = float(c.get("ball_bounce_height", self.BALL_BOUNCE_HEIGHT_DEFAULT))
        self.ball_bounce_freq = float(c.get("ball_bounce_freq", self.BALL_BOUNCE_FREQ_DEFAULT))
        self.ball_y_offset = float(c.get("ball_y_offset", self.BALL_Y_OFFSET_DEFAULT))
        self.ball_drop_settle_steps = int(c.get("ball_drop_settle_steps", self.BALL_DROP_SETTLE_STEPS_DEFAULT))
        self.enable_distractor = self._parse_bool(
            c.get("enable_distractor", c.get("opt2", self.ENABLE_DISTRACTOR_DEFAULT)),
            default=self.ENABLE_DISTRACTOR_DEFAULT,
        )
        self.distractor_color = list(c.get("distractor_color", self.DISTRACTOR_COLOR_DEFAULT))
        self.distractor_collide = self._parse_distractor_collide(
            c.get("distractor_collide", self.DISTRACTOR_COLLIDE_DEFAULT)
        )
        self.distractor_height_offset = float(
            c.get("distractor_height_offset", self.DISTRACTOR_HEIGHT_OFFSET_DEFAULT)
        )
        self.distractor_y_offset = float(
            c.get("distractor_y_offset", self.DISTRACTOR_Y_OFFSET_DEFAULT)
        )
        self.table_z = 0.74 + self.table_z_bias
        self.box_center = np.array([0.0, self.box_y], dtype=np.float64)

        lower_floor_z = self.table_z + 0.001
        self.lower_box = self._build_open_box("lower_box", lower_floor_z, wall_h=self.lower_box_wall_h)
        self._lower_box_floor_z = lower_floor_z + self.box_wall_t / 2.0
        self._lower_box_top_z = float(self.lower_box["top_z"])
        upper_floor_z = lower_floor_z + self.lower_box_wall_h + self.stack_gap + self.box_wall_t / 2.0
        self.upper_box = self._build_open_box("upper_box", upper_floor_z, with_floor=False, wall_h=self.upper_box_wall_h)
        self._upper_box_floor_z = upper_floor_z + self.box_wall_t / 2.0
        self._upper_box_top_z = float(self.upper_box["top_z"])
        self._build_stack_posts(self._lower_box_top_z, upper_floor_z)

        base_button_colors = [
            [0.85, 0.20, 0.20],
            [0.90, 0.60, 0.15],
            [0.20, 0.55, 0.85],
            [0.30, 0.70, 0.30],
        ]
        base_button_color_names = ["red", "yellow", "blue", "green"]
        # Randomize left→right order of colors; keys and trapdoors share the same order.
        n = self.n_buttons
        order = list(range(len(base_button_color_names)))
        if self.shuffle_colors:
            np.random.shuffle(order)
        order = [order[i % len(order)] for i in range(n)]
        self.button_colors = [list(base_button_colors[i]) for i in order]
        self.button_color_names = [base_button_color_names[i] for i in order]
        self.color_order = list(self.button_color_names)
        self.target_button_idx = self._resolve_target_button_idx(c.get("target_color", None))

        self.buttons = []
        self.button_bases = []
        button_homes = []
        button_tops = []
        for i, bx in enumerate(self.button_x[:self.n_buttons]):
            cap_hz = float(self.button_half[2])
            # Keycap on the table inside a hollow bezel (not a solid under-cube).
            cap_z = self.table_z + cap_hz
            home = sapien.Pose([bx, self.button_y, cap_z])
            walls = add_key_base_border(
                self,
                float(bx),
                float(self.button_y),
                float(self.table_z),
                self.button_half,
                color=list(self.key_base_color),
                name_prefix=f"panel_key_base_{i}",
            )
            btn = create_box(
                self,
                pose=home,
                half_size=list(self.button_half),
                color=self.button_colors[i],
                is_static=True,
                name=f"panel_button_{i}",
            )
            self.button_bases.extend(walls)
            self.buttons.append(btn)
            # World pose after create_box (includes table_z_bias) — spring math
            # and set_pose must share this frame (see fill_coffee_jar / ReactivePushButtons).
            world_home = btn.get_pose()
            button_homes.append(world_home)
            button_tops.append(float(world_home.p[2]) + cap_hz)

        self._reactive_buttons = ReactivePushButtons(
            self,
            actors=self.buttons,
            home_poses=button_homes,
            max_depth=float(self.button_half[2]),
            xy_tol=float(self.BUTTON_TOUCH_XY_TOL),
            visual_step=float(self.BUTTON_VISUAL_STEP),
            force_engage_slack=float(self.BUTTON_FORCE_ENGAGE_SLACK),
            ee_to_tcp=float(self.BUTTON_EE_TO_TCP),
        )
        self._reactive_buttons.set_tops_z(button_tops)

        self._button_pressed = [False] * self.n_buttons
        self._door_open = [False] * self.n_buttons
        self._door_open_with_ball_over = [False] * self.n_buttons
        self._door_angle_deg = [0.0] * self.n_buttons
        self._door_target_angle_deg = [0.0] * self.n_buttons
        self._door_open_time_left = [0.0] * self.n_buttons
        self._door_locked_closed = [False] * self.n_buttons
        self.door_tiles = []
        self._door_x_bounds = []
        self._build_upper_box_floor_tiles(upper_floor_z, self.button_colors)

        self.ball_radius = float(np.clip(self.ball_radius, 0.008, 0.03))
        self.ball_x_margin = float(max(self.ball_x_margin, 0.005))
        self._ball_x_min = (
            self.box_center[0] - self.box_half_w + self.box_wall_t + self.ball_radius + self.ball_x_margin
        )
        self._ball_x_max = (
            self.box_center[0] + self.box_half_w - self.box_wall_t - self.ball_radius - self.ball_x_margin
        )
        self._ball_y = float(self.box_y + self.ball_y_offset)
        self._ball_z_base = float(self._upper_box_floor_z + self.ball_radius + 0.002)
        self._ball_dir = float(np.random.choice([-1.0, 1.0]))
        self._ball_phase = float(np.random.uniform(0.0, 2.0 * np.pi))
        # Randomize where the target marble appears and which way it starts moving.
        ball_x0 = float(np.random.uniform(self._ball_x_min, self._ball_x_max))
        ball_color = self.button_colors[self.target_button_idx] if self.target_button_idx >= 0 else [0.92, 0.92, 0.95]
        self.ball = create_sphere(
            self,
            pose=sapien.Pose([ball_x0, self._ball_y, self._ball_z_base]),
            radius=self.ball_radius,
            color=ball_color,
            is_static=False,
            name="bouncing_ball",
        )
        self._ball_rigid = self._get_rigid(self.ball)
        self._configure_kinematic_marble(self._ball_rigid)

        self.distractor = None
        self._distractor_rigid = None
        self._distractor_dir = -self._ball_dir
        self._distractor_phase = float(np.random.uniform(0.0, 2.0 * np.pi))
        self._distractor_mode = "track"
        self._distractor_drop_door_idx = -1
        self._distractor_speed = self._resolve_distractor_speed(c)
        # Same height as the target; offset on Y so the two lanes do not collide.
        self._distractor_z_base = float(self._ball_z_base + max(0.0, self.distractor_height_offset))
        self._distractor_y = self._resolve_distractor_y()
        if self.enable_distractor:
            # Opposite travel direction: start near the wall it will leave from.
            sep = 2.5 * self.ball_radius
            if self._distractor_dir < 0:
                lo = max(ball_x0 + sep, self._ball_x_min)
                hi = self._ball_x_max
                dist_x0 = float(np.random.uniform(lo, hi)) if hi > lo + 1e-6 else float(self._ball_x_max)
            else:
                lo = self._ball_x_min
                hi = min(ball_x0 - sep, self._ball_x_max)
                dist_x0 = float(np.random.uniform(lo, hi)) if hi > lo + 1e-6 else float(self._ball_x_min)
            if abs(dist_x0 - ball_x0) < 2.2 * self.ball_radius:
                dist_x0 = self._ball_x_max if self._distractor_dir < 0 else self._ball_x_min
            self.distractor = create_sphere(
                self,
                pose=sapien.Pose([dist_x0, self._distractor_y, self._distractor_z_base]),
                radius=self.ball_radius,
                color=self.distractor_color,
                is_static=False,
                name="distractor_ball",
            )
            self._distractor_rigid = self._get_rigid(self.distractor)
            self._configure_kinematic_marble(self._distractor_rigid)

        self._buttons_held = set()
        self._mutex_violation = False
        self._ball_mode = "track"
        self._ball_drop_door_idx = -1
        self._press_lead_steps = 0

        for btn in self.buttons:
            self.add_prohibit_area(btn, padding=0.03)
        for base in self.button_bases:
            self.add_prohibit_area(base, padding=0.02)
        self.add_prohibit_area([0.0, self.box_y, self.table_z, 1, 0, 0, 0], padding=0.04)

    @staticmethod
    def _parse_bool(value, default=False):
        if value is None:
            return bool(default)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in ("1", "true", "yes", "y", "on"):
            return True
        if text in ("0", "false", "no", "n", "off"):
            return False
        return bool(default)

    def _parse_distractor_collide(self, value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value).strip().lower()
        if text in ("1", "true", "yes", "y", "on", "collide", "bounce"):
            return True
        if text in ("0", "false", "no", "n", "off", "pass", "passthrough", "pass_through"):
            return False
        # default / "random": choose per episode
        return bool(np.random.rand() < 0.5)

    def _resolve_target_button_idx(self, target_color):
        """Pick trapdoor/button index for the target marble color.

        target_color:
          null / "random" — sample uniformly among the 4 trapdoor colors
          "red"|"yellow"|"blue"|"green" (or 0..3) — use that specific color
        """
        if self.n_buttons <= 0:
            return -1
        if target_color is None or target_color == "" or str(target_color).strip().lower() in ("random", "none", "null"):
            return int(np.random.randint(self.n_buttons))
        if isinstance(target_color, (int, np.integer)):
            idx = int(target_color)
            if 0 <= idx < self.n_buttons:
                return idx
            raise ValueError(f"target_color index {idx} out of range [0, {self.n_buttons})")
        text = str(target_color).strip().lower()
        if text.isdigit():
            idx = int(text)
            if 0 <= idx < self.n_buttons:
                return idx
            raise ValueError(f"target_color index {idx} out of range [0, {self.n_buttons})")
        name = self.COLOR_NAME_ALIASES.get(text, text)
        for i, cname in enumerate(self.button_color_names):
            if cname == name:
                return i
        raise ValueError(
            f"target_color must be one of {self.button_color_names} "
            f"(or index 0..{self.n_buttons - 1}, or random); got {target_color!r}"
        )

    def _door_center_x(self, door_idx: int) -> float:
        if door_idx < 0 or door_idx >= len(self._door_x_bounds):
            if 0 <= door_idx < len(self.button_x):
                return float(self.button_x[door_idx])
            return 0.0
        x0, x1 = self._door_x_bounds[door_idx]
        return float(0.5 * (x0 + x1))

    def _arm_for_door(self, door_idx: int) -> ArmTag:
        """Left arm for left-half trapdoors, right arm for right-half trapdoors."""
        return ArmTag("left" if self._door_center_x(door_idx) < 0.0 else "right")

    def _sample_speed_from_default(self, default_speed: float, scale_min: float, scale_max: float) -> float:
        lo = float(min(scale_min, scale_max))
        hi = float(max(scale_min, scale_max))
        scale = float(np.random.uniform(lo, hi))
        return float(abs(default_speed) * scale)

    def _resolve_distractor_speed(self, c):
        explicit = c.get("distractor_speed", None)
        if explicit is not None:
            return float(abs(explicit))
        # Same default base as the target; independent ±20% sample (unless overridden).
        scale_min = float(c.get("distractor_speed_scale_min", c.get("ball_speed_scale_min", self.DISTRACTOR_SPEED_SCALE_MIN_DEFAULT)))
        scale_max = float(c.get("distractor_speed_scale_max", c.get("ball_speed_scale_max", self.DISTRACTOR_SPEED_SCALE_MAX_DEFAULT)))
        return self._sample_speed_from_default(self.ball_speed_default, scale_min, scale_max)

    def _configure_kinematic_marble(self, rigid):
        if rigid is None:
            return
        try:
            rigid.set_disable_gravity(True)
            rigid.set_kinematic(True)
            inelastic = sapien.physx.PhysxMaterial(
                static_friction=0.9,
                dynamic_friction=0.9,
                restitution=0.0,
            )
            for shape in rigid.get_collision_shapes():
                shape.set_physical_material(inelastic)
        except Exception:
            pass

    def _build_open_box(self, prefix: str, floor_z: float, with_floor: bool = True, wall_h: float | None = None):
        wall_h = float(self.box_wall_h if wall_h is None else wall_h)
        floor = None
        if with_floor:
            floor = create_box(
                self,
                pose=sapien.Pose([self.box_center[0], self.box_center[1], floor_z]),
                half_size=[self.box_half_w, self.box_half_d, self.box_wall_t / 2.0],
                color=[0.55, 0.40, 0.25],
                is_static=True,
                name=f"{prefix}_floor",
            )
        wall_cz = floor_z + wall_h / 2.0
        walls = [
            create_box(
                self,
                pose=sapien.Pose([self.box_center[0], self.box_center[1] - self.box_half_d, wall_cz]),
                half_size=[self.box_half_w, self.box_wall_t / 2.0, wall_h / 2.0],
                color=[0.55, 0.40, 0.25],
                is_static=True,
                name=f"{prefix}_front",
            ),
            create_box(
                self,
                pose=sapien.Pose([self.box_center[0], self.box_center[1] + self.box_half_d, wall_cz]),
                half_size=[self.box_half_w, self.box_wall_t / 2.0, wall_h / 2.0],
                color=[0.55, 0.40, 0.25],
                is_static=True,
                name=f"{prefix}_back",
            ),
            create_box(
                self,
                pose=sapien.Pose([self.box_center[0] - self.box_half_w, self.box_center[1], wall_cz]),
                half_size=[self.box_wall_t / 2.0, self.box_half_d, wall_h / 2.0],
                color=[0.55, 0.40, 0.25],
                is_static=True,
                name=f"{prefix}_left",
            ),
            create_box(
                self,
                pose=sapien.Pose([self.box_center[0] + self.box_half_w, self.box_center[1], wall_cz]),
                half_size=[self.box_wall_t / 2.0, self.box_half_d, wall_h / 2.0],
                color=[0.55, 0.40, 0.25],
                is_static=True,
                name=f"{prefix}_right",
            ),
        ]
        return {
            "floor": floor,
            "walls": walls,
            "floor_z": floor_z,
            "top_z": floor_z + wall_h,
        }

    def _build_stack_posts(self, lower_top_z: float, upper_floor_z: float):
        """Corner posts bridging the catch bowl rim to the trapdoor floor."""
        post_h = float(upper_floor_z - lower_top_z)
        if post_h < 0.004:
            return
        half = max(self.box_wall_t / 2.0, 0.003)
        cx = float(self.box_center[0])
        cy = float(self.box_center[1])
        hw = float(self.box_half_w)
        hd = float(self.box_half_d)
        cz = float(lower_top_z + 0.5 * post_h)
        for name, x, y in (
            ("fl", cx - hw, cy - hd),
            ("fr", cx + hw, cy - hd),
            ("bl", cx - hw, cy + hd),
            ("br", cx + hw, cy + hd),
        ):
            create_box(
                self,
                pose=sapien.Pose([x, y, cz]),
                half_size=[half, half, 0.5 * post_h],
                color=[0.50, 0.36, 0.22],
                is_static=True,
                name=f"stack_post_{name}",
            )

    # --------------------------------------------------------- kinematic scene motion
    def _aligned_button_x_positions(self):
        tile_count = max(6, int(self.floor_tile_count))
        lane_half_x = self.box_half_w / tile_count
        middle_start = (tile_count - self.n_buttons) // 2
        return [
            -self.box_half_w + lane_half_x + (middle_start + i) * (2.0 * lane_half_x)
            for i in range(self.n_buttons)
        ]

    def _build_upper_box_floor_tiles(self, floor_z: float, button_colors):
        tile_count = max(6, int(self.floor_tile_count))
        lane_half_x = self.box_half_w / tile_count
        tile_half_x = min(0.5 * max(self.door_width, 0.01), lane_half_x)
        inner_half_y = max(self.box_half_d - self.box_wall_t / 2.0, 0.01)
        tile_half_y = min(tile_half_x, inner_half_y)
        filler_half_x = max(lane_half_x, tile_half_x)
        tile_half_z = self.box_wall_t / 2.0
        self._door_half_x = tile_half_x
        self._door_half_y = tile_half_y
        self._door_hinge_y = self.box_y + tile_half_y
        self._door_floor_z = floor_z
        neutral_color = [0.42, 0.34, 0.28]

        middle_start = (tile_count - self.n_buttons) // 2
        middle_stop = middle_start + self.n_buttons
        for tile_idx in range(tile_count):
            cx = -self.box_half_w + lane_half_x + tile_idx * (2.0 * lane_half_x)
            if middle_start <= tile_idx < middle_stop:
                btn_idx = tile_idx - middle_start
                door = create_box(
                    self,
                    pose=sapien.Pose([cx, self.box_y, floor_z]),
                    half_size=[tile_half_x, tile_half_y, tile_half_z],
                    color=button_colors[btn_idx % len(button_colors)],
                    is_static=False,
                    name=f"trapdoor_tile_{btn_idx}",
                )
                rigid = self._get_rigid(door)
                if rigid is not None:
                    try:
                        rigid.set_disable_gravity(True)
                        rigid.set_kinematic(True)
                    except Exception:
                        pass
                self.door_tiles.append(door)
                self._door_x_bounds.append((cx - tile_half_x, cx + tile_half_x))
                self._set_door_pose(btn_idx, 0.0)
            else:
                create_box(
                    self,
                    pose=sapien.Pose([cx, self.box_y, floor_z]),
                    half_size=[filler_half_x, tile_half_y, tile_half_z],
                    color=neutral_color,
                    is_static=True,
                    name=f"upper_floor_tile_{tile_idx}",
                )

    def _get_rigid(self, entity):
        base_entity = entity.actor if hasattr(entity, "actor") else entity
        for component in base_entity.get_components():
            if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
                return component
        return None

    def _set_door_pose(self, idx: int, angle_deg: float):
        if idx < 0 or idx >= len(self.door_tiles):
            return
        door = self.door_tiles[idx]
        cx = 0.5 * (self._door_x_bounds[idx][0] + self._door_x_bounds[idx][1])
        theta = float(np.deg2rad(angle_deg))
        offset = np.array([0.0, -self._door_half_y, 0.0], dtype=np.float64)
        rot = t3d.axangles.axangle2mat([1.0, 0.0, 0.0], theta)
        local_center = rot @ offset
        hinge = np.array([cx, self._door_hinge_y, self._door_floor_z], dtype=np.float64)
        center = hinge + local_center
        quat = t3d.quaternions.axangle2quat([1.0, 0.0, 0.0], theta)
        door.actor.set_pose(sapien.Pose(center.tolist(), quat.tolist()))
        self._door_angle_deg[idx] = float(angle_deg)

    def _marble_in_box(self, marble, z_lo: float, z_hi: float):
        if marble is None:
            return False
        p = np.array(marble.get_pose().p, dtype=np.float64)
        in_x = abs(p[0] - self.box_center[0]) <= (self.box_half_w - self.box_wall_t)
        in_y = abs(p[1] - self.box_center[1]) <= (self.box_half_d - self.box_wall_t)
        in_z = z_lo <= p[2] <= z_hi
        return bool(in_x and in_y and in_z)

    def _lower_box_z_bounds(self):
        """Z range for 'inside the lower catch box' — must not overlap the upper floor lane.

        Allows marble centers above the shallow bowl rim (part sticks out) while
        still rejecting anything still riding on the upper trapdoor floor.
        """
        z_lo = float(self._lower_box_floor_z - 0.01)
        z_hi = float(min(
            self._lower_box_top_z + self.ball_radius + 0.01,
            self._upper_box_floor_z - self.ball_radius - 0.002,
        ))
        if z_hi <= z_lo:
            z_hi = float(self._lower_box_floor_z + 2.0 * self.ball_radius)
        return z_lo, z_hi

    def _ball_in_upper_box(self):
        return self._marble_in_box(
            self.ball,
            self._upper_box_floor_z - 0.01,
            self._upper_box_top_z + 0.02,
        )

    def _ball_in_lower_box(self):
        """Target catch: inside the box XY and below the closed trapdoor floor level.

        No settle / low-speed requirement — dropping below the original upper-floor
        plane into the catch volume is enough.
        """
        z_lo, z_hi = self._lower_box_z_bounds()
        return self._marble_in_box(self.ball, z_lo, z_hi)

    def _distractor_in_lower_box(self):
        # Any presence in the lower volume counts as a distractor failure (including mid-fall).
        z_lo, z_hi = self._lower_box_z_bounds()
        return self._marble_in_box(self.distractor, z_lo, z_hi)

    def _advance_doors(self):
        dt = float(self.scene.get_timestep())
        step = abs(self.door_open_speed_deg) * dt
        for idx in range(len(self.door_tiles)):
            if self._door_open[idx] and not self._door_locked_closed[idx]:
                self._door_open_time_left[idx] = max(0.0, float(self._door_open_time_left[idx]) - dt)
                if self._door_open_time_left[idx] <= 1e-9:
                    self._door_target_angle_deg[idx] = 0.0
                    if self.door_open_once:
                        self._door_locked_closed[idx] = True
            cur = float(self._door_angle_deg[idx])
            tgt = float(self._door_target_angle_deg[idx])
            if abs(cur - tgt) <= 1e-3:
                # Fully closed again: allow reopening unless once-only locked.
                if (
                    not self.door_open_once
                    and self._door_open[idx]
                    and abs(cur) <= 1e-3
                    and abs(tgt) <= 1e-3
                    and not self._door_locked_closed[idx]
                ):
                    self._door_open[idx] = False
                    self._button_pressed[idx] = False
                continue
            if cur < tgt:
                cur = min(cur + step, tgt)
            else:
                cur = max(cur - step, tgt)
            self._set_door_pose(idx, cur)

    def _marble_over_door(self, marble, mode: str, door_idx: int) -> bool:
        if (
            marble is None
            or mode != "track"
            or door_idx < 0
            or door_idx >= len(self._door_x_bounds)
        ):
            return False
        p = np.array(marble.get_pose().p, dtype=np.float64)
        x0, x1 = self._door_x_bounds[door_idx]
        in_x = x0 <= p[0] <= x1
        in_y = abs(p[1] - self.box_y) <= self._door_half_y
        # Allow the elevated distractor lane as well as the target lane.
        z_hi = max(
            self._upper_box_top_z + 0.02,
            float(getattr(self, "_distractor_z_base", self._ball_z_base))
            + abs(self.ball_bounce_height)
            + 0.02,
        )
        in_upper_z = (self._upper_box_floor_z - 0.01) <= p[2] <= z_hi
        return bool(in_x and in_y and in_upper_z)

    def _ball_over_door(self, door_idx: int) -> bool:
        return self._marble_over_door(self.ball, self._ball_mode, door_idx)

    def _distractor_over_door(self, door_idx: int) -> bool:
        return self._marble_over_door(self.distractor, self._distractor_mode, door_idx)

    def _release_marble(self, which: str, door_idx: int):
        if which == "ball":
            marble, rigid, speed, direction, mode_attr, drop_attr = (
                self.ball,
                self._ball_rigid,
                self.ball_speed,
                self._ball_dir,
                "_ball_mode",
                "_ball_drop_door_idx",
            )
        else:
            marble, rigid, speed, direction, mode_attr, drop_attr = (
                self.distractor,
                self._distractor_rigid,
                self._distractor_speed,
                self._distractor_dir,
                "_distractor_mode",
                "_distractor_drop_door_idx",
            )
        if marble is None or rigid is None or getattr(self, mode_attr) != "track":
            return
        setattr(self, drop_attr, int(door_idx))
        try:
            rigid.set_kinematic(False)
            rigid.set_disable_gravity(False)
            rigid.set_linear_velocity(np.array([direction * abs(speed), 0.0, 0.0]))
            rigid.set_angular_velocity(np.zeros(3))
        except Exception:
            pass
        setattr(self, mode_attr, "dropped")

    def _release_ball_from_upper_box(self, door_idx: int):
        self._release_marble("ball", door_idx)

    def _apply_marble_wall_bounce(self, next_x: float, direction: float):
        if next_x > self._ball_x_max:
            return self._ball_x_max, -1.0
        if next_x < self._ball_x_min:
            return self._ball_x_min, 1.0
        return next_x, direction

    def _resolve_marble_collision(self, x_ball: float, x_dist: float):
        """1D elastic bounce when marbles approach within contact distance."""
        if not self.distractor_collide:
            return self._ball_dir, self._distractor_dir
        if self._ball_mode != "track" or self._distractor_mode != "track":
            return self._ball_dir, self._distractor_dir
        sep = abs(x_ball - x_dist)
        if sep >= 2.0 * self.ball_radius:
            return self._ball_dir, self._distractor_dir
        approaching = (x_ball - x_dist) * (self._ball_dir - self._distractor_dir) < 0.0
        if not approaching and sep > 1.05 * 2.0 * self.ball_radius:
            return self._ball_dir, self._distractor_dir
        # Exchange 1D velocities (equal mass) => swap directions; also nudge apart.
        return float(self._distractor_dir), float(self._ball_dir)

    def _set_marble_pose(self, marble, rigid, pose: sapien.Pose):
        if marble is None:
            return
        if rigid is not None:
            try:
                rigid.set_kinematic_target(pose)
                return
            except Exception:
                pass
        marble.set_pose(pose)

    def _advance_ball_motion(self):
        dt = float(self.scene.get_timestep())
        ball_pose = self.ball.get_pose() if self.ball is not None else None
        dist_pose = self.distractor.get_pose() if self.distractor is not None else None

        next_ball_x = None
        next_dist_x = None
        if self.ball is not None and self._ball_mode == "track":
            next_ball_x = float(ball_pose.p[0] + self._ball_dir * abs(self.ball_speed) * dt)
            next_ball_x, self._ball_dir = self._apply_marble_wall_bounce(next_ball_x, self._ball_dir)
        if self.distractor is not None and self._distractor_mode == "track":
            next_dist_x = float(dist_pose.p[0] + self._distractor_dir * abs(self._distractor_speed) * dt)
            next_dist_x, self._distractor_dir = self._apply_marble_wall_bounce(next_dist_x, self._distractor_dir)

        if next_ball_x is not None and next_dist_x is not None:
            self._ball_dir, self._distractor_dir = self._resolve_marble_collision(next_ball_x, next_dist_x)
            # Recompute one step with possibly swapped directions, then separate if overlapping.
            next_ball_x = float(ball_pose.p[0] + self._ball_dir * abs(self.ball_speed) * dt)
            next_dist_x = float(dist_pose.p[0] + self._distractor_dir * abs(self._distractor_speed) * dt)
            next_ball_x, self._ball_dir = self._apply_marble_wall_bounce(next_ball_x, self._ball_dir)
            next_dist_x, self._distractor_dir = self._apply_marble_wall_bounce(next_dist_x, self._distractor_dir)
            if self.distractor_collide and abs(next_ball_x - next_dist_x) < 2.0 * self.ball_radius:
                mid = 0.5 * (next_ball_x + next_dist_x)
                if next_ball_x <= next_dist_x:
                    next_ball_x = mid - self.ball_radius
                    next_dist_x = mid + self.ball_radius
                else:
                    next_dist_x = mid - self.ball_radius
                    next_ball_x = mid + self.ball_radius
                next_ball_x = float(np.clip(next_ball_x, self._ball_x_min, self._ball_x_max))
                next_dist_x = float(np.clip(next_dist_x, self._ball_x_min, self._ball_x_max))

        # Only commit a drop when the marble is near the floor. Releasing at bounce apex
        # with horizontal speed lets it clear the opening even though the door is open.
        drop_z_slack = max(0.003, 0.35 * abs(self.ball_bounce_height))

        if next_ball_x is not None:
            self._ball_phase += abs(self.ball_bounce_freq) * dt
            next_z = float(self._ball_z_base + abs(self.ball_bounce_height) * abs(np.sin(self._ball_phase)))
            next_pose = sapien.Pose([next_ball_x, self._ball_y, next_z], ball_pose.q)
            dropped = False
            near_floor = next_z <= (self._ball_z_base + drop_z_slack)
            for idx, is_open in enumerate(self._door_open):
                if not is_open or self._door_angle_deg[idx] < 15.0:
                    continue
                x0, x1 = self._door_x_bounds[idx]
                if (
                    near_floor
                    and x0 <= next_ball_x <= x1
                    and abs(self._ball_y - self.box_y) <= self._door_half_y
                ):
                    self._set_marble_pose(self.ball, self._ball_rigid, next_pose)
                    self._release_marble("ball", idx)
                    dropped = True
                    break
            if not dropped:
                self._set_marble_pose(self.ball, self._ball_rigid, next_pose)

        if next_dist_x is not None:
            self._distractor_phase += abs(self.ball_bounce_freq) * dt
            next_z = float(
                self._distractor_z_base + abs(self.ball_bounce_height) * abs(np.sin(self._distractor_phase))
            )
            next_pose = sapien.Pose([next_dist_x, self._distractor_y, next_z], dist_pose.q)
            dropped = False
            near_floor = next_z <= (self._distractor_z_base + drop_z_slack)
            for idx, is_open in enumerate(self._door_open):
                if not is_open or self._door_angle_deg[idx] < 15.0:
                    continue
                x0, x1 = self._door_x_bounds[idx]
                if (
                    near_floor
                    and x0 <= next_dist_x <= x1
                    and abs(self._distractor_y - self.box_y) <= self._door_half_y
                ):
                    self._set_marble_pose(self.distractor, self._distractor_rigid, next_pose)
                    self._release_marble("distractor", idx)
                    dropped = True
                    break
            if not dropped:
                self._set_marble_pose(self.distractor, self._distractor_rigid, next_pose)

    def _resolve_distractor_y(self) -> float:
        """Same height as target; pick a parallel Y lane that stays inside the upper box."""
        y_lim = float(self.box_half_d - self.box_wall_t - self.ball_radius - 0.002)
        y_lim = max(0.0, y_lim)
        sep = max(abs(float(self.distractor_y_offset)), 2.2 * float(self.ball_radius))
        # Prefer the side opposite the target's ball_y_offset so both stay near center.
        sign = -1.0 if self._ball_y >= self.box_y else 1.0
        cand = float(self._ball_y + sign * sep)
        if abs(cand - self.box_y) > y_lim + 1e-9:
            cand = float(self._ball_y - sign * sep)
        return float(np.clip(cand, self.box_y - y_lim, self.box_y + y_lim))

    def _update_reactive_buttons(self):
        """Depress keycaps under the gripper; open the matching door on press edge.

        Holding a key does **not** keep the door open — each door uses
        ``door_open_duration_sec`` then auto-closes.  A new open requires a
        full key release (spring back) and then another press edge.
        """
        bank = getattr(self, "_reactive_buttons", None)
        if bank is None:
            return
        triggered = bank.update()
        interactive = bool(
            getattr(self, "_interactive_universal_controls", False)
            or getattr(self, "_interactive_robot_mode", False)
        )
        # Always animate keycaps; auto-open doors only in interactive teleop
        # (expert demos call ``_open_door_direct`` after the timed press).
        if interactive:
            for idx in triggered:
                opened = self._open_door_direct(int(idx))
                if opened:
                    color = (
                        self.button_color_names[idx]
                        if 0 <= idx < len(self.button_color_names)
                        else "?"
                    )
                    print(
                        f"[catch_marbles_trapdoors] pressed button {idx} ({color}); "
                        f"door open ≤ {self.door_open_duration_sec:.2f}s "
                        f"(release key fully to press again)"
                    )
        # Mirror held set from visual state so release clears the latch.
        self._buttons_held = {i for i, on in enumerate(bank.held_mask()) if on}

    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
        self._update_reactive_buttons()
        if len(self._buttons_held) > 1:
            self._mutex_violation = True
        self._advance_doors()
        self._advance_ball_motion()

    # --------------------------------------------------------- press helpers
    def _dwell(self, steps: int):
        for i in range(max(0, int(steps))):
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (i % self.save_freq == 0):
                self._take_picture()

    def _snapshot_marble_state(self):
        return {
            "ball_dir": float(self._ball_dir),
            "ball_phase": float(self._ball_phase),
            "ball_mode": str(self._ball_mode),
            "ball_drop_door_idx": int(self._ball_drop_door_idx),
            "distractor_dir": float(self._distractor_dir),
            "distractor_phase": float(self._distractor_phase),
            "distractor_mode": str(self._distractor_mode),
            "distractor_drop_door_idx": int(self._distractor_drop_door_idx),
        }

    def _restore_marble_state(self, state):
        self._ball_dir = state["ball_dir"]
        self._ball_phase = state["ball_phase"]
        self._ball_mode = state["ball_mode"]
        self._ball_drop_door_idx = state["ball_drop_door_idx"]
        self._distractor_dir = state["distractor_dir"]
        self._distractor_phase = state["distractor_phase"]
        self._distractor_mode = state["distractor_mode"]
        self._distractor_drop_door_idx = state["distractor_drop_door_idx"]

    def _estimate_press_lead_steps(self, arm_tag: ArmTag) -> int:
        if self._press_lead_steps > 0:
            return self._press_lead_steps
        # During trajectory replay, never dry-run a press — that can desync joint paths.
        if not getattr(self, "need_plan", True):
            dt = float(self.scene.get_timestep())
            self._press_lead_steps = max(8, int(round(0.12 / max(dt, 1e-6))))
            return self._press_lead_steps
        robot_state = DynamicMotionHelper.save_robot_state(self.robot)
        actors = [self.ball]
        if self.distractor is not None:
            actors.append(self.distractor)
        actors_state = DynamicMotionHelper.save_actors_state(actors, self._get_rigid)
        original_save_freq = self.save_freq
        original_plan_success = self.plan_success
        original_left_cnt = self.left_cnt
        original_right_cnt = self.right_cnt
        original_left_path_len = len(getattr(self, "left_joint_path", []) or [])
        original_right_path_len = len(getattr(self, "right_joint_path", []) or [])
        marble_state = self._snapshot_marble_state()
        self.save_freq = None
        self.plan_success = True
        try:
            with StepCounter(self.scene) as counter:
                self.move(self.move_by_displacement(arm_tag, z=-self.button_press_depth))
            lead_steps = max(1, counter.get_count())
        finally:
            DynamicMotionHelper.restore_robot_state(self.robot, robot_state, stabilization_steps=0, scene=None)
            DynamicMotionHelper.restore_actors_state(actors_state, stabilization_steps=0, scene=None)
            self.save_freq = original_save_freq
            self.plan_success = original_plan_success
            self.left_cnt = original_left_cnt
            self.right_cnt = original_right_cnt
            # Drop any joint segments appended by the dry-run press.
            if hasattr(self, "left_joint_path") and self.left_joint_path is not None:
                self.left_joint_path = self.left_joint_path[:original_left_path_len]
            if hasattr(self, "right_joint_path") and self.right_joint_path is not None:
                self.right_joint_path = self.right_joint_path[:original_right_path_len]
            self._restore_marble_state(marble_state)
        self._press_lead_steps = int(lead_steps)
        return self._press_lead_steps

    def save_traj_data(self, idx):
        super().save_traj_data(idx)
        # Persist press lead so replay waits for the same marble intercept window.
        file_path = os.path.join(self.save_dir, "_traj_data", f"episode{idx}.pkl")
        with open(file_path, "rb") as f:
            traj_data = pickle.load(f)
        traj_data["press_lead_steps"] = int(getattr(self, "_press_lead_steps", 0) or 0)
        save_pkl(file_path, traj_data)

    def load_tran_data(self, idx):
        traj_data = super().load_tran_data(idx)
        lead = int(traj_data.get("press_lead_steps", 0) or 0)
        if lead > 0:
            self._press_lead_steps = lead
        return traj_data

    def _wait_until_press_window(self, arm_tag: ArmTag, door_idx: int, max_steps: int | None = None) -> bool:
        if door_idx < 0 or door_idx >= len(self._door_x_bounds) or self.ball is None:
            return False
        lead_steps = self._estimate_press_lead_steps(arm_tag)
        dt = float(self.scene.get_timestep())
        lead_dist = abs(self.ball_speed) * lead_steps * dt
        x0, x1 = self._door_x_bounds[door_idx]
        door_pad = max(2.0 * self.ball_radius, 0.25 * (x1 - x0))
        if max_steps is None:
            sweep_span = max(self._ball_x_max - self._ball_x_min, 1e-6)
            sweep_time = (2.0 * sweep_span) / max(abs(self.ball_speed), 1e-6)
            # Allow a couple full L↔R sweeps when a distractor must clear the door.
            sweeps = 3.0 if self.distractor is not None else 1.0
            max_steps = max(
                self.final_dwell_steps,
                int(np.ceil(sweeps * sweep_time / max(dt, 1e-6))) + 5,
            )
        for i in range(max(0, int(max_steps))):
            pose = self.ball.get_pose()
            x_now = float(pose.p[0])
            if self._ball_dir >= 0:
                trigger_x = x0 - lead_dist
                ready = x_now >= trigger_x
            else:
                trigger_x = x1 + lead_dist
                ready = x_now <= trigger_x
            distractor_clear = True
            if self.distractor is not None and self._distractor_mode == "track":
                dx = float(self.distractor.get_pose().p[0])
                distractor_clear = not (x0 - door_pad <= dx <= x1 + door_pad)
            if ready and self._ball_mode == "track" and distractor_clear:
                return True
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (i % self.save_freq == 0):
                self._take_picture()
            if self._ball_mode != "track":
                break
        return False

    def _open_door_direct(self, btn_idx: int) -> bool:
        """Open trapdoor ``btn_idx`` for ``door_open_duration_sec`` (press edge).

        Holding the key does not extend the open window.  Another open needs a
        fresh press edge after the key has fully sprung back (enforced by
        ``ReactivePushButtons``).  While the door is already open, a new edge
        restarts the timer; ``door_open_once`` still locks after the first open.
        """
        if btn_idx < 0 or btn_idx >= self.n_buttons:
            return False
        if self._door_locked_closed[btn_idx]:
            return False
        if self.door_open_once and self._button_pressed[btn_idx]:
            return False
        opened_on_time = self._ball_over_door(btn_idx)
        self._button_pressed[btn_idx] = True
        self._door_open[btn_idx] = True
        self._door_open_with_ball_over[btn_idx] = (
            self._door_open_with_ball_over[btn_idx] or opened_on_time
        )
        # Fresh edge → fresh open window (hold does not keep refreshing this).
        self._door_open_time_left[btn_idx] = max(0.0, float(self.door_open_duration_sec))
        self._door_target_angle_deg[btn_idx] = self.door_open_angle_deg
        return True

    def _press_button(self, arm_tag: ArmTag, btn_idx: int):
        if not self.plan_success:
            return
        if self._door_locked_closed[btn_idx]:
            return
        if self._door_open[btn_idx] and self._door_angle_deg[btn_idx] > 5.0:
            return
        if self.door_open_once and self._button_pressed[btn_idx]:
            return
        btn = self.buttons[btn_idx]
        self.move(
            self.grasp_actor(
                btn,
                arm_tag=arm_tag,
                pre_grasp_dis=0.09,
                grasp_dis=0.09,
                contact_point_id=0,
                gripper_pos=0.5,
            )
        )
        if not self.plan_success:
            return
        if not self._wait_until_press_window(arm_tag, btn_idx):
            return
        self._buttons_held.add(btn_idx)
        self.move(self.move_by_displacement(arm_tag, z=-self.button_press_depth))
        if not self._open_door_direct(btn_idx):
            # Door may already be open / locked; still finish the press motion.
            pass
        self._dwell(self.press_hold_steps)
        self._buttons_held.discard(btn_idx)
        self.move(self.move_by_displacement(arm_tag, z=self.button_press_depth + 0.01))

    def _wait_for_ball_drop(self):
        sweep_span = max(self._ball_x_max - self._ball_x_min, 1e-6)
        sweep_time = (2.0 * sweep_span) / max(abs(self.ball_speed), 1e-6)
        wait_steps = max(
            self.final_dwell_steps,
            int(np.ceil(sweep_time / max(self.scene.get_timestep(), 1e-6))) + self.press_hold_steps,
        )
        for i in range(wait_steps):
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (i % self.save_freq == 0):
                self._take_picture()
            if self._ball_mode != "track":
                self._dwell(self.ball_drop_settle_steps)
                return
        self._dwell(self.final_dwell_steps)

    # ------------------------------------------------------------- policy
    def play_once(self):
        arm_tag = None
        if 0 <= self.target_button_idx < self.n_buttons:
            btn_idx = self.target_button_idx
            # Arm follows the matching trapdoor's side (left half → left arm, right → right).
            arm_tag = self._arm_for_door(btn_idx)
            self._press_button(arm_tag, btn_idx)
            if self.plan_success:
                self.move(self.back_to_origin(arm_tag))
                self._dwell(self.post_press_dwell)
        self._wait_for_ball_drop()

        arm_name = str(arm_tag) if arm_tag is not None else "left"
        self.info["info"] = {
            "{A}": f"{self.button_color_names[self.target_button_idx]} ball" if self.target_button_idx >= 0 else "colored ball",
            "{B}": "matching trapdoor",
            "{a}": "left",
            "{b}": "right",
        }
        self.info["operating_arm"] = arm_name
        self.info["target_door_x"] = float(self._door_center_x(self.target_button_idx))
        self.info["color_order"] = list(getattr(self, "color_order", self.button_color_names))
        return self.info

    # ----------------------------------------------------------- metric/obs
    def check_success(self):
        """Success iff the target marble fell through the matching door into the box.

        Required:
          - target was released through the matching-color trapdoor
          - target is inside the box and below the original trapdoor floor level

        Opening the matching door under the marble is NOT enough — the marble must
        actually drop below the upper-floor plane. Settle / low speed is not required.
        ``target_opened_when_ball_over`` is logged for diagnostics only.

        Failures:
          - target still on the upper surface / never dropped / sailed over the opening
          - target went through a differently colored trapdoor
          - distractor (if present) went through any trapdoor into the lower box
        """
        target_valid = 0 <= self.target_button_idx < len(self._door_open)
        ball_dropped = bool(self._ball_mode != "track")
        used_matching_door = bool(target_valid and self._ball_drop_door_idx == self.target_button_idx)
        used_wrong_door = bool(ball_dropped and self._ball_drop_door_idx >= 0 and not used_matching_door)
        ball_inside = self._ball_in_lower_box()
        ball_still_on_top = bool(self._ball_in_upper_box() and not ball_inside)
        distractor_dropped = bool(
            self.enable_distractor
            and self.distractor is not None
            and self._distractor_mode != "track"
        )
        distractor_inside = bool(self._distractor_in_lower_box())
        distractor_through_any = bool(
            distractor_dropped and (self._distractor_drop_door_idx >= 0 or distractor_inside)
        ) or bool(self.enable_distractor and distractor_inside)

        opened_door_indices = [int(i) for i, is_open in enumerate(self._door_open) if is_open]
        wrong_door_opened = bool(
            target_valid and any(i != self.target_button_idx for i in opened_door_indices)
        )
        target_door_opened = bool(target_valid and (
            self._door_open[self.target_button_idx]
            or self._button_pressed[self.target_button_idx]
            or self._ball_drop_door_idx == self.target_button_idx
        ))
        opened_when_ball_over = bool(target_valid and self._door_open_with_ball_over[self.target_button_idx])

        self.info["buttons_pressed"] = int(sum(self._button_pressed))
        self.info["target_button_idx"] = int(self.target_button_idx)
        self.info["target_color"] = (
            self.button_color_names[self.target_button_idx]
            if target_valid and self.target_button_idx < len(self.button_color_names)
            else ""
        )
        self.info["color_order"] = list(getattr(self, "color_order", self.button_color_names))
        self.info["opened_door_indices"] = opened_door_indices
        self.info["door_open_with_ball_over"] = list(self._door_open_with_ball_over)
        self.info["ball_drop_door_idx"] = int(self._ball_drop_door_idx)
        self.info["distractor_drop_door_idx"] = int(self._distractor_drop_door_idx)
        self.info["only_target_door_open"] = bool(target_door_opened and not wrong_door_opened)
        self.info["target_opened_when_ball_over"] = opened_when_ball_over
        self.info["ball_dropped"] = ball_dropped
        self.info["used_matching_door"] = used_matching_door
        self.info["used_wrong_door"] = used_wrong_door
        self.info["ball_in_lower_box"] = ball_inside
        self.info["ball_still_on_top"] = ball_still_on_top
        self.info["wrong_door_opened"] = wrong_door_opened
        self.info["distractor_through_any"] = distractor_through_any
        self.info["distractor_in_lower_box"] = distractor_inside
        self.info["door_open_once"] = bool(self.door_open_once)
        self.info["enable_distractor"] = bool(self.enable_distractor)
        self.info["distractor_collide"] = bool(self.distractor_collide)
        self.info["mutex_violation"] = bool(self._mutex_violation)

        # Deliberately ignore opened_when_ball_over / target_door_opened here: door timing
        # alone must not succeed if the marble never drops below the trapdoor level.
        return bool(
            target_valid
            and ball_dropped
            and used_matching_door
            and ball_inside
            and not ball_still_on_top
            and not used_wrong_door
            and not distractor_through_any
            and not distractor_inside
            and not self._mutex_violation
        )

    def get_obs(self):
        obs = super().get_obs()
        held_mask = [bool(i in self._buttons_held) for i in range(self.n_buttons)]
        obs["button_box"] = {
            "button_pressed": list(self._button_pressed),
            "door_open": list(self._door_open),
            "door_open_with_ball_over": list(self._door_open_with_ball_over),
            "door_angle_deg": list(self._door_angle_deg),
            "door_open_time_left": list(map(float, self._door_open_time_left)),
            "door_locked_closed": list(self._door_locked_closed),
            "door_open_once": bool(self.door_open_once),
            "buttons_held": held_mask,
            "buttons_held_count": int(sum(held_mask)),
            "mutex_violation": bool(self._mutex_violation),
            "target_button_idx": int(self.target_button_idx),
            "target_color": (
                self.button_color_names[self.target_button_idx]
                if 0 <= self.target_button_idx < len(self.button_color_names)
                else ""
            ),
            "color_order": list(getattr(self, "color_order", self.button_color_names)),
            "shuffle_colors": bool(self.shuffle_colors),
            "ball_drop_door_idx": int(self._ball_drop_door_idx),
            "ball_position": list(map(float, self.ball.get_pose().p)) if self.ball is not None else [0.0, 0.0, 0.0],
            "ball_direction": float(self._ball_dir),
            "ball_mode": str(self._ball_mode),
            "ball_in_upper_box": bool(self._ball_in_upper_box()),
            "ball_in_lower_box": bool(self._ball_in_lower_box()),
            "enable_distractor": bool(self.enable_distractor),
            "distractor_collide": bool(self.distractor_collide),
            "distractor_speed": float(self._distractor_speed),
            "distractor_drop_door_idx": int(self._distractor_drop_door_idx),
            "distractor_position": (
                list(map(float, self.distractor.get_pose().p))
                if self.distractor is not None
                else [0.0, 0.0, 0.0]
            ),
            "distractor_direction": float(self._distractor_dir),
            "distractor_mode": str(self._distractor_mode),
            "distractor_in_lower_box": bool(self._distractor_in_lower_box()),
        }
        return obs
