from ._base_task import Base_Task
from .utils import *
import sapien
import sapien.physx
import numpy as np
import transforms3d as t3d
from .utils.dynamic_utils import DynamicMotionHelper, StepCounter


class catch_marbles_trapdoors(Base_Task):
    """Press four colored buttons to open matching trapdoors on a stacked box fixture.

    Two identical open-top boxes are stacked vertically at the table centre. Four colored button
    cubes sit in front of the fixture in the near zone. The bottom of the upper box is split into
    six square floor tiles; the four middle tiles match the button colors and act as trapdoors.
    When a button is pressed, the corresponding colored bottom tile rotates downward like a door
    opening.

    A small colored ball continuously bounces left-to-right inside the upper box. The correct
    solution is to press only the button whose color matches the ball, and to start opening that
    trapdoor while the ball is directly above it.
    """

    N_BUTTONS_DEFAULT = 4
    BUTTON_X_DEFAULT = [-0.15, -0.05, 0.05, 0.15]
    BUTTON_Y_DEFAULT = -0.18
    BUTTON_HALF_DEFAULT = [0.022, 0.022, 0.018]
    BUTTON_PRESS_DEPTH_DEFAULT = 0.03
    PRESS_HOLD_STEPS_DEFAULT = 20
    POST_PRESS_DWELL_DEFAULT = 10
    FINAL_DWELL_STEPS_DEFAULT = 80

    BOX_Y_DEFAULT = -0.02
    BOX_HALF_W_DEFAULT = 0.30
    BOX_HALF_D_DEFAULT = 0.055
    BOX_WALL_H_DEFAULT = 0.04
    BOX_WALL_T_DEFAULT = 0.006
    STACK_GAP_DEFAULT = 0.0

    FLOOR_TILE_COUNT_DEFAULT = 6
    DOOR_WIDTH_DEFAULT = 0.10
    DOOR_OPEN_ANGLE_DEG_DEFAULT = 95.0
    DOOR_OPEN_SPEED_DEG_DEFAULT = 220.0

    BALL_RADIUS_DEFAULT = 0.012
    BALL_SPEED_DEFAULT = 0.18
    BALL_X_MARGIN_DEFAULT = 0.02
    BALL_BOUNCE_HEIGHT_DEFAULT = 0.012
    BALL_BOUNCE_FREQ_DEFAULT = 8.0
    BALL_Y_OFFSET_DEFAULT = 0.0
    BALL_DROP_SETTLE_STEPS_DEFAULT = 80

    def setup_demo(self, **kwags):
        self._cfg = kwags.get("task_args", {}).get("catch_marbles_trapdoors", {})
        self.buttons = []
        self.door_tiles = []
        self._button_pressed = []
        self._door_open = []
        self._door_open_with_ball_over = []
        self._door_angle_deg = []
        self._door_target_angle_deg = []
        self._door_x_bounds = []
        self._door_hinge_y = 0.0
        self._door_half_y = 0.0
        self._door_half_x = 0.0
        self._door_floor_z = 0.0
        self.door_width = float(self.DOOR_WIDTH_DEFAULT)
        self.door_open_angle_deg = float(self.DOOR_OPEN_ANGLE_DEG_DEFAULT)
        self.door_open_speed_deg = float(self.DOOR_OPEN_SPEED_DEG_DEFAULT)
        self.button_colors = []
        self.button_color_names = []
        self.target_button_idx = -1
        self._buttons_held = set()
        self._mutex_violation = False
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
        super()._init_task_env_(**kwags)

    # ------------------------------------------------------------------ actors
    def load_actors(self):
        c = self._cfg
        self.n_buttons = int(c.get("n_buttons", self.N_BUTTONS_DEFAULT))
        self.button_y = float(c.get("button_y", self.BUTTON_Y_DEFAULT))
        self.button_half = list(c.get("button_half", self.BUTTON_HALF_DEFAULT))
        self.button_press_depth = float(c.get("button_press_depth", self.BUTTON_PRESS_DEPTH_DEFAULT))
        self.press_hold_steps = int(c.get("press_hold_steps", self.PRESS_HOLD_STEPS_DEFAULT))
        self.post_press_dwell = int(c.get("post_press_dwell", self.POST_PRESS_DWELL_DEFAULT))
        self.final_dwell_steps = int(c.get("final_dwell_steps", self.FINAL_DWELL_STEPS_DEFAULT))

        self.box_y = float(c.get("box_y", self.BOX_Y_DEFAULT))
        self.box_half_w = float(c.get("box_half_w", self.BOX_HALF_W_DEFAULT))
        self.box_half_d = float(c.get("box_half_d", self.BOX_HALF_D_DEFAULT))
        self.box_wall_h = float(c.get("box_wall_h", self.BOX_WALL_H_DEFAULT))
        self.box_wall_t = float(c.get("box_wall_t", self.BOX_WALL_T_DEFAULT))
        self.stack_gap = float(c.get("stack_gap", self.STACK_GAP_DEFAULT))
        self.lower_box_wall_h = 2.0 * self.box_half_d
        self.upper_box_wall_h = float(c.get("upper_box_wall_h", self.box_wall_h))

        self.floor_tile_count = int(c.get("floor_tile_count", self.FLOOR_TILE_COUNT_DEFAULT))
        self.door_width = float(c.get("door_width", self.DOOR_WIDTH_DEFAULT))
        self.door_open_angle_deg = float(c.get("door_open_angle_deg", self.DOOR_OPEN_ANGLE_DEG_DEFAULT))
        self.door_open_speed_deg = float(c.get("door_open_speed_deg", self.DOOR_OPEN_SPEED_DEG_DEFAULT))
        button_x_cfg = c.get("button_x", None)
        self.button_x = list(button_x_cfg) if button_x_cfg is not None else self._aligned_button_x_positions()

        self.ball_radius = float(c.get("ball_radius", self.BALL_RADIUS_DEFAULT))
        self.ball_speed = float(c.get("ball_speed", self.BALL_SPEED_DEFAULT))
        self.ball_x_margin = float(c.get("ball_x_margin", self.BALL_X_MARGIN_DEFAULT))
        self.ball_bounce_height = float(c.get("ball_bounce_height", self.BALL_BOUNCE_HEIGHT_DEFAULT))
        self.ball_bounce_freq = float(c.get("ball_bounce_freq", self.BALL_BOUNCE_FREQ_DEFAULT))
        self.ball_y_offset = float(c.get("ball_y_offset", self.BALL_Y_OFFSET_DEFAULT))
        self.ball_drop_settle_steps = int(c.get("ball_drop_settle_steps", self.BALL_DROP_SETTLE_STEPS_DEFAULT))

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

        base_button_colors = [
            [0.85, 0.20, 0.20],
            [0.90, 0.60, 0.15],
            [0.20, 0.55, 0.85],
            [0.30, 0.70, 0.30],
        ]
        base_button_color_names = ["red", "yellow", "blue", "green"]
        self.button_colors = [base_button_colors[i % len(base_button_colors)] for i in range(self.n_buttons)]
        self.button_color_names = [base_button_color_names[i % len(base_button_color_names)] for i in range(self.n_buttons)]
        self.target_button_idx = int(np.random.randint(self.n_buttons)) if self.n_buttons > 0 else -1

        self.buttons = []
        for i, bx in enumerate(self.button_x[:self.n_buttons]):
            bz = self.table_z + self.button_half[2]
            btn = create_box(
                self,
                pose=sapien.Pose([bx, self.button_y, bz]),
                half_size=list(self.button_half),
                color=self.button_colors[i],
                is_static=True,
                name=f"panel_button_{i}",
            )
            self.buttons.append(btn)

        self._button_pressed = [False] * self.n_buttons
        self._door_open = [False] * self.n_buttons
        self._door_open_with_ball_over = [False] * self.n_buttons
        self._door_angle_deg = [0.0] * self.n_buttons
        self._door_target_angle_deg = [0.0] * self.n_buttons
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
        ball_x0 = self._ball_x_min if self._ball_dir > 0 else self._ball_x_max
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
        if self._ball_rigid is not None:
            try:
                self._ball_rigid.set_disable_gravity(True)
                self._ball_rigid.set_kinematic(True)
                inelastic = sapien.physx.PhysxMaterial(
                    static_friction=0.9,
                    dynamic_friction=0.9,
                    restitution=0.0,
                )
                for shape in self._ball_rigid.get_collision_shapes():
                    shape.set_physical_material(inelastic)
            except Exception:
                pass

        self._buttons_held = set()
        self._mutex_violation = False
        self._ball_mode = "track"
        self._ball_drop_door_idx = -1
        self._press_lead_steps = 0

        for btn in self.buttons:
            self.add_prohibit_area(btn, padding=0.03)
        self.add_prohibit_area([0.0, self.box_y, self.table_z, 1, 0, 0, 0], padding=0.04)

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

    def _ball_in_upper_box(self):
        if self.ball is None:
            return False
        p = np.array(self.ball.get_pose().p, dtype=np.float64)
        in_x = abs(p[0] - self.box_center[0]) <= (self.box_half_w - self.box_wall_t)
        in_y = abs(p[1] - self.box_center[1]) <= (self.box_half_d - self.box_wall_t)
        in_z = (self._upper_box_floor_z - 0.01) <= p[2] <= (self._upper_box_top_z + 0.02)
        return bool(in_x and in_y and in_z)

    def _ball_in_lower_box(self):
        if self.ball is None:
            return False
        p = np.array(self.ball.get_pose().p, dtype=np.float64)
        in_x = abs(p[0] - self.box_center[0]) <= (self.box_half_w - self.box_wall_t)
        in_y = abs(p[1] - self.box_center[1]) <= (self.box_half_d - self.box_wall_t)
        in_z = (self._lower_box_floor_z - 0.01) <= p[2] <= (self._lower_box_top_z + 0.03)
        return bool(in_x and in_y and in_z)

    def _advance_doors(self):
        dt = float(self.scene.get_timestep())
        step = abs(self.door_open_speed_deg) * dt
        for idx in range(len(self.door_tiles)):
            cur = float(self._door_angle_deg[idx])
            tgt = float(self._door_target_angle_deg[idx])
            if abs(cur - tgt) <= 1e-3:
                continue
            if cur < tgt:
                cur = min(cur + step, tgt)
            else:
                cur = max(cur - step, tgt)
            self._set_door_pose(idx, cur)

    def _ball_over_door(self, door_idx: int) -> bool:
        if (
            self.ball is None
            or self._ball_mode != "track"
            or door_idx < 0
            or door_idx >= len(self._door_x_bounds)
        ):
            return False
        p = np.array(self.ball.get_pose().p, dtype=np.float64)
        x0, x1 = self._door_x_bounds[door_idx]
        in_x = x0 <= p[0] <= x1
        in_y = abs(p[1] - self.box_y) <= self._door_half_y
        in_upper_z = (self._upper_box_floor_z - 0.01) <= p[2] <= (self._upper_box_top_z + 0.02)
        return bool(in_x and in_y and in_upper_z)

    def _release_ball_from_upper_box(self, door_idx: int):
        if self.ball is None or self._ball_rigid is None or self._ball_mode != "track":
            return
        self._ball_drop_door_idx = int(door_idx)
        try:
            self._ball_rigid.set_kinematic(False)
            self._ball_rigid.set_disable_gravity(False)
            self._ball_rigid.set_linear_velocity(np.array([self._ball_dir * abs(self.ball_speed), 0.0, 0.0]))
            self._ball_rigid.set_angular_velocity(np.zeros(3))
        except Exception:
            pass
        self._ball_mode = "dropped"

    def _advance_ball_motion(self):
        if self.ball is None:
            return
        if self._ball_mode != "track":
            return
        dt = float(self.scene.get_timestep())
        pose = self.ball.get_pose()
        next_x = float(pose.p[0] + self._ball_dir * abs(self.ball_speed) * dt)
        if next_x > self._ball_x_max:
            next_x = self._ball_x_max
            self._ball_dir = -1.0
        elif next_x < self._ball_x_min:
            next_x = self._ball_x_min
            self._ball_dir = 1.0
        self._ball_phase += abs(self.ball_bounce_freq) * dt
        next_z = float(self._ball_z_base + abs(self.ball_bounce_height) * abs(np.sin(self._ball_phase)))
        next_pose = sapien.Pose([next_x, self._ball_y, next_z], pose.q)
        for idx, is_open in enumerate(self._door_open):
            if not is_open or self._door_angle_deg[idx] < 15.0:
                continue
            x0, x1 = self._door_x_bounds[idx]
            if x0 <= next_x <= x1:
                if self._ball_rigid is not None:
                    self.ball.set_pose(next_pose)
                self._release_ball_from_upper_box(idx)
                return
        if self._ball_rigid is not None:
            try:
                self._ball_rigid.set_kinematic_target(next_pose)
                return
            except Exception:
                pass
        self.ball.set_pose(next_pose)

    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
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

    def _estimate_press_lead_steps(self, arm_tag: ArmTag) -> int:
        if self._press_lead_steps > 0:
            return self._press_lead_steps
        robot_state = DynamicMotionHelper.save_robot_state(self.robot)
        actors_state = DynamicMotionHelper.save_actors_state([self.ball], self._get_rigid)
        original_save_freq = self.save_freq
        original_plan_success = self.plan_success
        original_left_cnt = self.left_cnt
        original_right_cnt = self.right_cnt
        ball_dir = float(self._ball_dir)
        ball_phase = float(self._ball_phase)
        ball_mode = str(self._ball_mode)
        ball_drop_door_idx = int(self._ball_drop_door_idx)
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
            self._ball_dir = ball_dir
            self._ball_phase = ball_phase
            self._ball_mode = ball_mode
            self._ball_drop_door_idx = ball_drop_door_idx
        self._press_lead_steps = int(lead_steps)
        return self._press_lead_steps

    def _wait_until_press_window(self, arm_tag: ArmTag, door_idx: int, max_steps: int | None = None) -> bool:
        if door_idx < 0 or door_idx >= len(self._door_x_bounds) or self.ball is None:
            return False
        lead_steps = self._estimate_press_lead_steps(arm_tag)
        dt = float(self.scene.get_timestep())
        lead_dist = abs(self.ball_speed) * lead_steps * dt
        x0, x1 = self._door_x_bounds[door_idx]
        if max_steps is None:
            sweep_span = max(self._ball_x_max - self._ball_x_min, 1e-6)
            sweep_time = (2.0 * sweep_span) / max(abs(self.ball_speed), 1e-6)
            max_steps = max(
                self.final_dwell_steps,
                int(np.ceil(sweep_time / max(dt, 1e-6))) + 5,
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
            if ready and self._ball_mode == "track":
                return True
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (i % self.save_freq == 0):
                self._take_picture()
            if self._ball_mode != "track":
                break
        return False

    def _press_button(self, arm_tag: ArmTag, btn_idx: int):
        if not self.plan_success or self._button_pressed[btn_idx]:
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
        opened_on_time = self._ball_over_door(btn_idx)
        self._button_pressed[btn_idx] = True
        self._door_open[btn_idx] = True
        self._door_open_with_ball_over[btn_idx] = opened_on_time
        self._door_target_angle_deg[btn_idx] = self.door_open_angle_deg
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
        if 0 <= self.target_button_idx < len(self.button_x):
            btn_idx = self.target_button_idx
            bx = self.button_x[btn_idx]
            arm_tag = ArmTag("left" if bx < 0 else "right")
            self._press_button(arm_tag, btn_idx)
            if self.plan_success:
                self.move(self.back_to_origin(arm_tag))
                self._dwell(self.post_press_dwell)
        self._wait_for_ball_drop()

        self.info["info"] = {
            "{A}": f"{self.button_color_names[self.target_button_idx]} ball" if self.target_button_idx >= 0 else "colored ball",
            "{B}": "bottom trapdoors",
            "{C}": "stacked boxes",
            "{D}": "colored button cubes",
            "{a}": "left arm",
            "{b}": "right arm",
        }
        return self.info

    # ----------------------------------------------------------- metric/obs
    def check_success(self):
        target_valid = 0 <= self.target_button_idx < len(self._door_open)
        only_target_open = bool(
            target_valid
            and self._door_open[self.target_button_idx]
            and int(sum(self._door_open)) == 1
        )
        opened_when_ball_over = bool(
            target_valid
            and self._door_open_with_ball_over[self.target_button_idx]
        )
        used_matching_door = bool(target_valid and self._ball_drop_door_idx == self.target_button_idx)
        ball_inside = self._ball_in_lower_box()
        self.info["buttons_pressed"] = int(sum(self._button_pressed))
        self.info["target_button_idx"] = int(self.target_button_idx)
        self.info["target_color"] = (
            self.button_color_names[self.target_button_idx]
            if target_valid and self.target_button_idx < len(self.button_color_names)
            else ""
        )
        self.info["opened_door_indices"] = [int(i) for i, is_open in enumerate(self._door_open) if is_open]
        self.info["door_open_with_ball_over"] = list(self._door_open_with_ball_over)
        self.info["ball_drop_door_idx"] = int(self._ball_drop_door_idx)
        self.info["only_target_door_open"] = only_target_open
        self.info["target_opened_when_ball_over"] = opened_when_ball_over
        self.info["used_matching_door"] = used_matching_door
        self.info["ball_in_lower_box"] = ball_inside
        self.info["mutex_violation"] = bool(self._mutex_violation)
        return bool(only_target_open and opened_when_ball_over and not self._mutex_violation)

    def get_obs(self):
        obs = super().get_obs()
        held_mask = [bool(i in self._buttons_held) for i in range(self.n_buttons)]
        obs["button_box"] = {
            "button_pressed": list(self._button_pressed),
            "door_open": list(self._door_open),
            "door_open_with_ball_over": list(self._door_open_with_ball_over),
            "door_angle_deg": list(self._door_angle_deg),
            "buttons_held": held_mask,
            "buttons_held_count": int(sum(held_mask)),
            "mutex_violation": bool(self._mutex_violation),
            "target_button_idx": int(self.target_button_idx),
            "target_color": (
                self.button_color_names[self.target_button_idx]
                if 0 <= self.target_button_idx < len(self.button_color_names)
                else ""
            ),
            "ball_drop_door_idx": int(self._ball_drop_door_idx),
            "ball_position": list(map(float, self.ball.get_pose().p)) if self.ball is not None else [0.0, 0.0, 0.0],
            "ball_direction": float(self._ball_dir),
            "ball_mode": str(self._ball_mode),
            "ball_in_upper_box": bool(self._ball_in_upper_box()),
            "ball_in_lower_box": bool(self._ball_in_lower_box()),
        }
        return obs
