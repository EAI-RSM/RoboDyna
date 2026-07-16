from ._base_task import Base_Task
from .utils import *
import sapien
import sapien.physx
import numpy as np


class goalkeeper(Base_Task):
    """Keep a moving ball out of the goal by placing a square blocker in time.

    A ball travels left-to-right or right-to-left across the table toward a goal on one side.
    A red deadline line marks the last moment the robot is allowed to reposition the goalkeeper.
    A green placement area sits directly in front of the goal. The robot must grasp the square
    goalkeeper, place it fully inside that green area while the ball is still behind the red line,
    then release it before the ball reaches the goal mouth.
    """

    BALL_RADIUS_DEFAULT = 0.018
    BALL_SPEED_MIN_DEFAULT = 0.07
    BALL_SPEED_MAX_DEFAULT = 0.09
    BALL_START_X_DEFAULT = 0.24
    BALL_START_Y_JITTER_DEFAULT = 0.02
    BALL_GOAL_END_X_OFFSET_DEFAULT = 0.08
    BALL_TARGET_Y_MARGIN_DEFAULT = 0.02
    BALL_ANGLE_DEG_MIN_DEFAULT = -10.0
    BALL_ANGLE_DEG_MAX_DEFAULT = 10.0

    GOAL_X_DEFAULT = 0.20
    GOAL_CENTER_Y_DEFAULT = 0.10
    GOAL_CENTER_Y_JITTER_DEFAULT = 0.0
    GOAL_HALF_W_DEFAULT = 0.11
    GOAL_POST_T_DEFAULT = 0.01
    GOAL_POST_H_DEFAULT = 0.12
    GOAL_BAR_T_DEFAULT = 0.01
    GREEN_AREA_X_LEN_DEFAULT = 0.10
    GREEN_AREA_Y_EXTRA_DEFAULT = 0.05
    RED_LINE_X_DEFAULT = 0.2

    KEEPER_X_DEFAULT = 0.16
    KEEPER_SPAWN_X_DEFAULT = 0.10
    KEEPER_SPAWN_Y_DEFAULT = -0.12
    KEEPER_GOAL_CLEARANCE_DEFAULT = 0.05
    KEEPER_POSE_TOL_DEFAULT = 0.03

    BALL_SETTLE_STEPS_DEFAULT = 120

    def setup_demo(self, **kwags):
        self._cfg = kwags.get("task_args", {}).get("goalkeeper", {})
        self._loaded = False
        self._ball_motion_active = False
        self._ball_step = 0
        self._ball_blocked = False
        self._goal_conceded = False
        self._late_failure = False
        self.goalkeeper = None
        self.ball = None
        self._ball_rigid = None
        self.goalkeeper_target_pose = None
        self.ball_start_pose = None
        self.ball_target_pose = None
        self._ball_crossed_goal = False
        self.green_area_x_min = 0.0
        self.green_area_x_max = 0.0
        self.green_area_y_min = 0.0
        self.green_area_y_max = 0.0
        super()._init_task_env_(**kwags)
        self._ball_motion_active = True

    # ------------------------------------------------------------------ helpers
    def _get_rigid(self, entity):
        obj = entity.actor if hasattr(entity, "actor") else entity
        for comp in obj.get_components():
            if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent):
                return comp
        return None

    def _keeper_xy_bounds(self):
        if getattr(self, "goalkeeper", None) is None:
            return None
        pose_m = self.goalkeeper.get_pose().to_transformation_matrix()
        local_corners = np.array([
            [-self.keeper_half_x, -self.keeper_half_y, 0.0, 1.0],
            [-self.keeper_half_x,  self.keeper_half_y, 0.0, 1.0],
            [ self.keeper_half_x, -self.keeper_half_y, 0.0, 1.0],
            [ self.keeper_half_x,  self.keeper_half_y, 0.0, 1.0],
        ], dtype=np.float64)
        world_corners = (pose_m @ local_corners.T).T
        return (
            float(np.min(world_corners[:, 0])),
            float(np.max(world_corners[:, 0])),
            float(np.min(world_corners[:, 1])),
            float(np.max(world_corners[:, 1])),
        )

    def _ball_path_y_at_x(self, x: float) -> float:
        start_x = float(self.ball_start_pose[0])
        end_x = float(self.ball_target_pose[0])
        if abs(end_x - start_x) < 1e-8:
            return float(self.ball_start_pose[1])
        t = float((x - start_x) / (end_x - start_x))
        return float(self.ball_start_pose[1] + t * (self.ball_target_pose[1] - self.ball_start_pose[1]))

    def _keeper_in_zone(self):
        if getattr(self, "goalkeeper", None) is None:
            return False
        bounds = self._keeper_xy_bounds()
        if bounds is None:
            return False
        x_min, x_max, y_min, y_max = bounds
        keeper_z = float(self.goalkeeper.get_pose().p[2])
        target_z = float(self.table_top_z + self.keeper_half_z)
        z_ok = abs(keeper_z - target_z) <= 0.02
        return bool(
            x_min >= (self.green_area_x_min - 1e-4)
            and x_max <= (self.green_area_x_max + 1e-4)
            and y_min >= (self.green_area_y_min - 1e-4)
            and y_max <= (self.green_area_y_max + 1e-4)
            and z_ok
        )

    def _wait_for_outcome(self):
        max_steps = int(self.ball_total_steps + self.ball_settle_steps)
        for i in range(max(0, max_steps)):
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (i % self.save_freq == 0):
                self._take_picture()

    def _place_keeper_from_top(self, arm_tag: ArmTag):
        if getattr(self, "goalkeeper", None) is None or self.goalkeeper_target_pose is None:
            return

        target_p = np.asarray(self.goalkeeper_target_pose.p, dtype=np.float64)
        frame_clearance_z = self.table_top_z + max(
            self.goal_post_h + self.keeper_half_z + 0.08,
            self.keeper_half_z + 0.18,
        )

        keeper_p = np.asarray(self.goalkeeper.get_pose().p, dtype=np.float64)
        lift_dz = float(frame_clearance_z - keeper_p[2])
        if lift_dz > 1e-4:
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=lift_dz))

        keeper_p = np.asarray(self.goalkeeper.get_pose().p, dtype=np.float64)
        goal_over_x = float(self.goal_x - self.travel_dir * (self.keeper_half_x + 0.01))
        goal_over_y = float(self.goal_center_y)
        dx_goal = float(goal_over_x - keeper_p[0])
        dy_goal = float(goal_over_y - keeper_p[1])
        if abs(dx_goal) > 1e-4 or abs(dy_goal) > 1e-4:
            self.move(self.move_by_displacement(arm_tag=arm_tag, x=dx_goal, y=dy_goal))

        keeper_p = np.asarray(self.goalkeeper.get_pose().p, dtype=np.float64)
        dx = float(target_p[0] - keeper_p[0])
        dy = float(target_p[1] - keeper_p[1])
        if abs(dx) > 1e-4 or abs(dy) > 1e-4:
            self.move(self.move_by_displacement(arm_tag=arm_tag, x=dx, y=dy))

        keeper_p = np.asarray(self.goalkeeper.get_pose().p, dtype=np.float64)
        place_dz = float(target_p[2] - keeper_p[2])
        if abs(place_dz) > 1e-4:
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=place_dz))

        self.move(self.open_gripper(arm_tag))

    # ------------------------------------------------------------------ actors
    def load_actors(self):
        c = self._cfg
        self.table_top_z = 0.74 + self.table_z_bias

        self.goal_x_abs = float(c.get("goal_x", c.get("goal_y", self.GOAL_X_DEFAULT)))
        self.goal_center_y = float(c.get("goal_center_y", self.GOAL_CENTER_Y_DEFAULT))
        goal_center_y_jitter = float(c.get("goal_center_y_jitter", self.GOAL_CENTER_Y_JITTER_DEFAULT))
        self.goal_center_y = float(np.random.uniform(
            self.goal_center_y - goal_center_y_jitter,
            self.goal_center_y + goal_center_y_jitter,
        ))
        self.goal_half_w = float(c.get("goal_half_w", self.GOAL_HALF_W_DEFAULT))
        self.goal_post_t = float(c.get("goal_post_t", self.GOAL_POST_T_DEFAULT))
        self.goal_post_h = float(c.get("goal_post_h", self.GOAL_POST_H_DEFAULT))
        self.goal_bar_t = float(c.get("goal_bar_t", self.GOAL_BAR_T_DEFAULT))
        self.green_area_x_len = float(c.get("green_area_x_len", self.GREEN_AREA_X_LEN_DEFAULT))
        self.green_area_y_extra = float(c.get("green_area_y_extra", self.GREEN_AREA_Y_EXTRA_DEFAULT))
        self.red_line_goal_offset = float(c.get("red_line_x", c.get("red_line_y", self.RED_LINE_X_DEFAULT)))

        self.keeper_x_abs = float(c.get("keeper_x", c.get("keeper_y", self.KEEPER_X_DEFAULT)))
        self.keeper_pose_tol = float(c.get("keeper_pose_tol", self.KEEPER_POSE_TOL_DEFAULT))
        self.keeper_spawn_x = float(c.get("keeper_spawn_x", self.KEEPER_SPAWN_X_DEFAULT))
        self.keeper_spawn_y = float(c.get("keeper_spawn_y", self.KEEPER_SPAWN_Y_DEFAULT))
        self.keeper_goal_clearance = float(c.get("keeper_goal_clearance", self.KEEPER_GOAL_CLEARANCE_DEFAULT))

        self.ball_radius = float(c.get("ball_radius", self.BALL_RADIUS_DEFAULT))
        self.ball_speed_min = float(c.get("ball_speed_min", self.BALL_SPEED_MIN_DEFAULT))
        self.ball_speed_max = float(c.get("ball_speed_max", self.BALL_SPEED_MAX_DEFAULT))
        self.ball_start_x_abs = float(c.get("ball_start_x", self.BALL_START_X_DEFAULT))
        self.ball_start_y_jitter = float(c.get("ball_start_y_jitter", c.get("ball_start_x_jitter", self.BALL_START_Y_JITTER_DEFAULT)))
        self.ball_goal_end_x_offset = float(c.get("ball_goal_end_x_offset", c.get("ball_goal_end_y_offset", self.BALL_GOAL_END_X_OFFSET_DEFAULT)))
        self.ball_target_y_margin = float(c.get("ball_target_y_margin", c.get("ball_target_x_margin", self.BALL_TARGET_Y_MARGIN_DEFAULT)))
        self.ball_angle_deg_min = float(c.get("ball_angle_deg_min", self.BALL_ANGLE_DEG_MIN_DEFAULT))
        self.ball_angle_deg_max = float(c.get("ball_angle_deg_max", self.BALL_ANGLE_DEG_MAX_DEFAULT))
        self.ball_settle_steps = int(c.get("ball_settle_steps", self.BALL_SETTLE_STEPS_DEFAULT))

        self.travel_dir = float(np.random.choice([-1.0, 1.0]))
        self.goal_x = float(self.travel_dir * abs(self.goal_x_abs))
        self.red_line_x = float(self.goal_x - self.travel_dir * abs(self.red_line_goal_offset))
        self.ball_speed = float(np.random.uniform(self.ball_speed_min, self.ball_speed_max))
        self.keeper_half_x = self.ball_radius
        self.keeper_half_y = self.ball_radius
        self.keeper_half_z = self.ball_radius
        green_half_x = 0.5 * self.green_area_x_len
        green_half_y = self.goal_half_w + 0.5 * self.green_area_y_extra
        self.green_area_center_x = float(self.goal_x - self.travel_dir * green_half_x)
        self.green_area_center_y = float(self.goal_center_y)
        self.green_area_x_min = float(min(self.goal_x, self.goal_x - self.travel_dir * self.green_area_x_len))
        self.green_area_x_max = float(max(self.goal_x, self.goal_x - self.travel_dir * self.green_area_x_len))
        self.green_area_y_min = float(self.goal_center_y - green_half_y)
        self.green_area_y_max = float(self.goal_center_y + green_half_y)

        goal_end_x = float(self.goal_x + self.travel_dir * abs(self.ball_goal_end_x_offset))
        dt = float(self.scene.get_timestep())
        for _ in range(64):
            start_y = float(np.random.uniform(
                self.goal_center_y - self.ball_start_y_jitter,
                self.goal_center_y + self.ball_start_y_jitter,
            ))
            launch_angle_deg = float(np.random.uniform(self.ball_angle_deg_min, self.ball_angle_deg_max))
            angle_rad = np.deg2rad(launch_angle_deg)
            end_y = float(start_y + np.tan(angle_rad) * (goal_end_x - (-self.travel_dir * self.ball_start_x_abs)))
            if abs(end_y - self.goal_center_y) <= (self.goal_half_w - self.ball_target_y_margin):
                break
        else:
            end_y = float(np.clip(
                end_y,
                self.goal_center_y - self.goal_half_w + self.ball_target_y_margin,
                self.goal_center_y + self.goal_half_w - self.ball_target_y_margin,
            ))

        start_x = float(-self.travel_dir * abs(self.ball_start_x_abs))
        self.launch_angle_deg = float(np.degrees(np.arctan2(end_y - start_y, goal_end_x - start_x)))
        self.ball_start_pose = np.array(
            [start_x, start_y, self.table_top_z + self.ball_radius],
            dtype=np.float64,
        )
        self.ball_target_pose = np.array(
            [goal_end_x, end_y, self.table_top_z + self.ball_radius],
            dtype=np.float64,
        )
        ball_vec = self.ball_target_pose - self.ball_start_pose
        ball_dist = float(np.linalg.norm(ball_vec))
        self.ball_dir = ball_vec / max(ball_dist, 1e-8)
        self.ball_total_steps = max(1, int(np.ceil(ball_dist / max(self.ball_speed * dt, 1e-8))))

        keeper_x_min = float(self.green_area_x_min + self.keeper_half_x)
        keeper_x_max = float(self.green_area_x_max - self.keeper_half_x)
        self.goal_intersection_y = self._ball_path_y_at_x(self.goal_x)
        preferred_keeper_x = float(self.goal_x - self.travel_dir * (self.keeper_half_x + 0.002))
        self.keeper_x = float(np.clip(preferred_keeper_x, keeper_x_min, keeper_x_max))
        keeper_y = float(self._ball_path_y_at_x(self.keeper_x))
        keeper_y = float(np.clip(
            keeper_y,
            self.green_area_y_min + self.keeper_half_y,
            self.green_area_y_max - self.keeper_half_y,
        ))
        self.goalkeeper_target_pose = sapien.Pose(
            [self.keeper_x, keeper_y, self.table_top_z + self.keeper_half_z],
            [1, 0, 0, 0],
        )

        goal_color = (0.92, 0.92, 0.94)
        post_half = [self.goal_post_t * 0.5, self.goal_post_t * 0.5, self.goal_post_h * 0.5]
        self.goal_left_post = create_box(
            self,
            pose=sapien.Pose([self.goal_x, self.goal_center_y - self.goal_half_w, self.table_top_z + self.goal_post_h * 0.5], [1, 0, 0, 0]),
            half_size=post_half,
            color=goal_color,
            is_static=True,
            name="goal_left_post",
        )
        self.goal_right_post = create_box(
            self,
            pose=sapien.Pose([self.goal_x, self.goal_center_y + self.goal_half_w, self.table_top_z + self.goal_post_h * 0.5], [1, 0, 0, 0]),
            half_size=post_half,
            color=goal_color,
            is_static=True,
            name="goal_right_post",
        )
        self.goal_bar = create_box(
            self,
            pose=sapien.Pose([self.goal_x, self.goal_center_y, self.table_top_z + self.goal_post_h - self.goal_bar_t * 0.5], [1, 0, 0, 0]),
            half_size=[self.goal_post_t * 0.5, self.goal_half_w + self.goal_post_t, self.goal_bar_t * 0.5],
            color=goal_color,
            is_static=True,
            name="goal_bar",
        )
        create_visual_box(
            self,
            pose=sapien.Pose([self.red_line_x, self.goal_center_y, self.table_top_z + 0.001], [1, 0, 0, 0]),
            half_size=[0.002, self.goal_half_w + 0.10, 0.001],
            color=(0.95, 0.12, 0.12),
            name="red_line",
        )
        create_visual_box(
            self,
            pose=sapien.Pose([self.green_area_center_x, self.green_area_center_y, self.table_top_z + 0.001], [1, 0, 0, 0]),
            half_size=[green_half_x, green_half_y, 0.001],
            color=(0.18, 0.72, 0.25),
            name="goal_green_area",
        )

        keeper_x0 = float(self.goal_x)
        goal_lower_y = float(self.goal_center_y - self.goal_half_w)
        keeper_y0 = float(goal_lower_y - self.keeper_goal_clearance - self.keeper_half_y)
        self.goalkeeper = create_box(
            self,
            pose=sapien.Pose([keeper_x0, keeper_y0, self.table_top_z + self.keeper_half_z], [1, 0, 0, 0]),
            half_size=[self.keeper_half_x, self.keeper_half_y, self.keeper_half_z],
            color=(0.90, 0.70, 0.12),
            name="goalkeeper_square",
            is_static=False,
        )
        self.goalkeeper.set_mass(0.08)
        for comp in self.goalkeeper.actor.get_components():
            if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent):
                try:
                    comp.set_linear_damping(10.0)
                    comp.set_angular_damping(10.0)
                except Exception:
                    pass

        self.ball = create_sphere(
            self.scene,
            pose=sapien.Pose(self.ball_start_pose.tolist(), [1, 0, 0, 0]),
            radius=self.ball_radius,
            color=(0.18, 0.56, 0.90),
            is_static=False,
            name="goal_ball",
        )
        self._ball_rigid = self._get_rigid(self.ball)
        if self._ball_rigid is not None:
            try:
                self._ball_rigid.set_disable_gravity(True)
                self._ball_rigid.set_kinematic(True)
                self._ball_rigid.set_linear_velocity(np.zeros(3))
                self._ball_rigid.set_angular_velocity(np.zeros(3))
            except Exception:
                pass

        self.add_prohibit_area(self.goalkeeper, padding=0.02)
        self._loaded = True

    # ------------------------------------------------------------- motion / checks
    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
        if not getattr(self, "_loaded", False):
            return
        if not getattr(self, "_ball_motion_active", False):
            return
        if self._ball_blocked:
            return
        if self._ball_rigid is None or getattr(self, "ball", None) is None:
            return

        self._ball_step += 1
        progress = min(1.0, self._ball_step / float(self.ball_total_steps))
        next_p = self.ball_start_pose + (self.ball_target_pose - self.ball_start_pose) * progress

        keeper_ok = self._keeper_in_zone()
        if (not self._late_failure) and (self.travel_dir * next_p[0] >= self.travel_dir * self.red_line_x) and (not keeper_ok):
            self._late_failure = True

        if keeper_ok:
            x_min, x_max, y_min, y_max = self._keeper_xy_bounds()
            face_x = float((x_min - self.ball_radius - 0.001) if self.travel_dir > 0 else (x_max + self.ball_radius + 0.001))
            if (
                self.travel_dir * next_p[0] >= self.travel_dir * face_x
                and (y_min - self.ball_radius) <= float(next_p[1]) <= (y_max + self.ball_radius)
            ):
                next_p[0] = face_x
                self._ball_blocked = True
                self._ball_motion_active = False
                pose = sapien.Pose(next_p.tolist(), [1, 0, 0, 0])
                self.ball.set_pose(pose)
                try:
                    self._ball_rigid.set_kinematic_target(pose)
                except Exception:
                    pass
                return

        if self.travel_dir * next_p[0] >= self.travel_dir * self.goal_x:
            self._ball_crossed_goal = True
        if (
            (not self._goal_conceded)
            and self.travel_dir * next_p[0] >= self.travel_dir * self.goal_x
            and abs(float(next_p[1] - self.goal_center_y)) <= self.goal_half_w
        ):
            self._goal_conceded = True

        pose = sapien.Pose(next_p.tolist(), [1, 0, 0, 0])
        self.ball.set_pose(pose)
        try:
            self._ball_rigid.set_kinematic_target(pose)
        except Exception:
            pass
        if progress >= 1.0:
            self._ball_motion_active = False

    # ----------------------------------------------------------------- policy
    def play_once(self):
        arm_tag = ArmTag("right" if self.goalkeeper.get_pose().p[0] > 0 else "left")
        grasp_contact_id = [0, 1, 2, 3]

        self.move(self.close_gripper(arm_tag, pos=0.6))
        self.move(
            self.grasp_actor(
                self.goalkeeper,
                arm_tag=arm_tag,
                pre_grasp_dis=0.10,
                grasp_dis=0.0,
                contact_point_id=grasp_contact_id,
            )
        )
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.12, move_axis="arm"))
        self._place_keeper_from_top(arm_tag)
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.08, move_axis="arm"))
        self.move(self.back_to_origin(arm_tag))

        self._wait_for_outcome()

        self.info["info"] = {
            "{A}": "goalkeeper_square",
            "{B}": "goal_frame",
            "{a}": str(arm_tag),
        }
        return self.info

    # ---------------------------------------------------------------- success / obs
    def check_success(self):
        keeper_ok = self._keeper_in_zone()
        success = bool(
            keeper_ok
            and self._ball_blocked
            and (not self._late_failure)
            and (not self._goal_conceded)
            and self.is_left_gripper_open()
            and self.is_right_gripper_open()
        )
        self.info["goalkeeper"] = {
            "keeper_in_zone": bool(keeper_ok),
            "ball_blocked": bool(self._ball_blocked),
            "ball_crossed_goal": bool(self._ball_crossed_goal),
            "late_failure": bool(self._late_failure),
            "goal_conceded": bool(self._goal_conceded),
        }
        return success

    def get_obs(self):
        obs = super().get_obs()
        obs["goalkeeper"] = {
            "ball_pos": self.ball.get_pose().p.tolist() if getattr(self, "ball", None) is not None else [0.0, 0.0, 0.0],
            "keeper_pos": self.goalkeeper.get_pose().p.tolist() if getattr(self, "goalkeeper", None) is not None else [0.0, 0.0, 0.0],
            "keeper_target": self.goalkeeper_target_pose.p.tolist() if self.goalkeeper_target_pose is not None else [0.0, 0.0, 0.0],
            "keeper_in_zone": bool(self._keeper_in_zone()),
            "ball_blocked": bool(self._ball_blocked),
            "ball_crossed_goal": bool(self._ball_crossed_goal),
            "late_failure": bool(self._late_failure),
            "goal_conceded": bool(self._goal_conceded),
            "ball_speed": float(getattr(self, "ball_speed", 0.0)),
            "launch_angle_deg": float(getattr(self, "launch_angle_deg", 0.0)),
            "red_line_x": float(getattr(self, "red_line_x", 0.0)),
            "goal_x": float(getattr(self, "goal_x", 0.0)),
            "goal_center_y": float(getattr(self, "goal_center_y", 0.0)),
            "goal_intersection_y": float(getattr(self, "goal_intersection_y", 0.0)),
            "green_area_center": [float(getattr(self, "green_area_center_x", 0.0)), float(getattr(self, "green_area_center_y", 0.0))],
            "green_area_bounds": [
                float(getattr(self, "green_area_x_min", 0.0)),
                float(getattr(self, "green_area_x_max", 0.0)),
                float(getattr(self, "green_area_y_min", 0.0)),
                float(getattr(self, "green_area_y_max", 0.0)),
            ],
            "travel_dir": float(getattr(self, "travel_dir", 0.0)),
        }
        return obs
