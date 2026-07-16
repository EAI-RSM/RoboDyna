from ._base_task import Base_Task
from .utils import *
import numpy as np
import sapien
import sapien.physx
import transforms3d as t3d


class catch_valley_ball(Base_Task):
    """Place a bowl to catch a ball leaving a down-then-up valley ramp.

    The ramp, ball, and robot use the standard Base_Task table/arm setup.  The
    ball follows a deterministic, step-driven drop and roll while the robot
    moves the bowl.  At the upward ramp's edge it becomes a real dynamic body,
    so the final flight and catch are resolved by PhysX.
    """

    SIM_HZ = 250.0
    GRAVITY = 9.81

    RAMP_HALF_WIDTH_DEFAULT = 0.125
    RAMP_THICKNESS_DEFAULT = 0.012
    RAMP_CENTER_Y_DEFAULT = 0.08
    RAMP_VALLEY_X_DEFAULT = -0.02
    RAMP_VALLEY_HEIGHT_DEFAULT = 0.025
    DOWN_RUN_DEFAULT = 0.27
    DOWN_RISE_DEFAULT = 0.13
    UP_RUN_DEFAULT = 0.20
    UP_RISE_DEFAULT = 0.055
    CURVE_SEGMENTS_DEFAULT = 16
    RAIL_HEIGHT_DEFAULT = 0.055
    RAIL_THICKNESS_DEFAULT = 0.006

    BALL_RADIUS_DEFAULT = 0.018
    DROP_HEIGHT_DEFAULT = 0.14
    DROP_TIME_DEFAULT = 0.60
    DROP_WALL_ANGLE_MIN_DEFAULT = 15.0
    DROP_WALL_ANGLE_MAX_DEFAULT = 25.0
    BALL_PATH_MODE_DEFAULT = "random"
    DROP_FORWARD_ANGLE_MIN_DEFAULT = -8.0
    DROP_FORWARD_ANGLE_MAX_DEFAULT = 12.0
    INITIAL_FORWARD_SPEED_MIN_DEFAULT = 0.90
    INITIAL_FORWARD_SPEED_MAX_DEFAULT = 1.10
    ROLL_TIME_MIN_DEFAULT = 4.0
    ROLL_TIME_MAX_DEFAULT = 7.0
    LAUNCH_SPEED_DEFAULT = 0.72
    ROLL_ACCELERATION_DEFAULT = 0.35
    PHYSICS_MAX_STEPS_DEFAULT = 1200
    RED_LINE_GAP_DEFAULT = 0.05
    IDLE_TIME_MIN_DEFAULT = 1.0
    IDLE_TIME_MAX_DEFAULT = 2.0
    SETTLE_STEPS_DEFAULT = 150

    BOWL_ID_DEFAULT = 1
    BOWL_SCALE_MULT_DEFAULT = 0.65
    BOWL_INNER_RADIUS_DEFAULT = 0.028
    BOWL_OUTER_RADIUS_DEFAULT = 0.037
    BOWL_HEIGHT_DEFAULT = 0.040

    def setup_demo(self, **kwags):
        self._cfg = kwags.get("task_args", {}).get("catch_valley_ball", {})

        # The environment object is reused between episodes. load_camera can
        # invoke the per-step hook before the new actors have been constructed.
        self._loaded = False
        self._ball_phase = None
        self._expert_demo = False
        self._bowl_ready = False
        self._bowl_welded = False
        self._arm_ball_contact = False
        super()._init_task_env_(**kwags)

        # Evaluation does not call play_once, so start the self-contained ball
        # motion after initialization. Expert collection resets the same state.
        self._start_ball_motion(expert_demo=False)

    # ---------------------------------------------------------------- actors
    def load_actors(self):
        c = self._cfg
        self.table_top = 0.74 + self.table_z_bias

        self.ramp_half_width = float(c.get("ramp_half_width", self.RAMP_HALF_WIDTH_DEFAULT))
        self.ramp_thickness = float(c.get("ramp_thickness", self.RAMP_THICKNESS_DEFAULT))
        self.ramp_center_y = float(c.get("ramp_center_y", self.RAMP_CENTER_Y_DEFAULT))
        self.valley_x = float(c.get("ramp_valley_x", self.RAMP_VALLEY_X_DEFAULT))
        self.valley_height = float(c.get("ramp_valley_height", self.RAMP_VALLEY_HEIGHT_DEFAULT))
        self.down_run = float(c.get("down_run", self.DOWN_RUN_DEFAULT))
        self.down_rise = float(c.get("down_rise", self.DOWN_RISE_DEFAULT))
        self.up_run = float(c.get("up_run", self.UP_RUN_DEFAULT))
        self.up_rise = float(c.get("up_rise", self.UP_RISE_DEFAULT))
        self.curve_segments = int(c.get("curve_segments", self.CURVE_SEGMENTS_DEFAULT))
        self.rail_height = float(c.get("rail_height", self.RAIL_HEIGHT_DEFAULT))
        self.rail_thickness = float(c.get("rail_thickness", self.RAIL_THICKNESS_DEFAULT))

        self.ball_radius = float(c.get("ball_radius", self.BALL_RADIUS_DEFAULT))
        self.drop_height = float(c.get("drop_height", self.DROP_HEIGHT_DEFAULT))
        self.drop_time = float(c.get("drop_time", self.DROP_TIME_DEFAULT))
        requested_path_mode = str(
            c.get("ball_path_mode", self.BALL_PATH_MODE_DEFAULT)
        ).strip().lower()
        if requested_path_mode not in {"straight", "angled", "random"}:
            requested_path_mode = self.BALL_PATH_MODE_DEFAULT
        self.ball_path_mode = (
            str(np.random.choice(["straight", "angled"]))
            if requested_path_mode == "random"
            else requested_path_mode
        )
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
        initial_speed_min = float(c.get(
            "initial_forward_speed_min",
            self.INITIAL_FORWARD_SPEED_MIN_DEFAULT,
        ))
        initial_speed_max = float(c.get(
            "initial_forward_speed_max",
            self.INITIAL_FORWARD_SPEED_MAX_DEFAULT,
        ))
        self.roll_time = float(np.random.uniform(
            c.get("roll_time_min", self.ROLL_TIME_MIN_DEFAULT),
            c.get("roll_time_max", self.ROLL_TIME_MAX_DEFAULT),
        ))
        self.launch_speed = float(c.get("launch_speed", self.LAUNCH_SPEED_DEFAULT))
        self.roll_acceleration = float(c.get(
            "roll_acceleration",
            self.ROLL_ACCELERATION_DEFAULT,
        ))
        self.physics_max_steps = int(c.get("physics_max_steps", self.PHYSICS_MAX_STEPS_DEFAULT))
        self.red_line_gap = float(c.get("red_line_gap", self.RED_LINE_GAP_DEFAULT))
        self.idle_time = float(np.random.uniform(
            c.get("idle_time_min", self.IDLE_TIME_MIN_DEFAULT),
            c.get("idle_time_max", self.IDLE_TIME_MAX_DEFAULT),
        ))
        self.settle_steps = int(c.get("settle_steps", self.SETTLE_STEPS_DEFAULT))

        self.bowl_id = int(c.get("bowl_id", self.BOWL_ID_DEFAULT))
        self.bowl_scale_mult = float(c.get("bowl_scale_mult", self.BOWL_SCALE_MULT_DEFAULT))
        self.bowl_inner_radius = float(c.get("bowl_inner_radius", self.BOWL_INNER_RADIUS_DEFAULT))
        self.bowl_outer_radius = float(c.get("bowl_outer_radius", self.BOWL_OUTER_RADIUS_DEFAULT))
        self.bowl_height = float(c.get("bowl_height", self.BOWL_HEIGHT_DEFAULT))

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
        self.drop_time = max(self.drop_time, 0.05)
        self.roll_time = max(self.roll_time, 0.2)
        self.launch_speed = max(self.launch_speed, 0.05)
        self.roll_acceleration = float(np.clip(self.roll_acceleration, 0.0, 0.8))
        self.physics_max_steps = max(self.physics_max_steps, 100)
        self.red_line_gap = max(self.red_line_gap, 0.0)
        self.bowl_scale_mult = max(self.bowl_scale_mult, 0.25)

        # Straight episodes stay on the centerline. Angled episodes follow a
        # reflected lateral path and visibly rebound from one or both rails.
        if self.ball_path_mode == "straight":
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
        self.initial_forward_speed = float(np.random.uniform(
            min(initial_speed_min, initial_speed_max),
            max(initial_speed_min, initial_speed_max),
        ))
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
        self.ball_y = self.ramp_center_y

        surface_points = self._make_curved_surface()
        tangents = np.gradient(surface_points, axis=0)
        normals = np.column_stack([
            -tangents[:, 2],
            np.zeros(len(tangents)),
            tangents[:, 0],
        ])
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        self.ball_path = surface_points + self.ball_radius * normals
        lateral_slope = np.tan(np.deg2rad(self.drop_wall_angle_deg))
        raw_lateral = (
            self.ramp_center_y
            + (self.ball_path[:, 0] - self.ball_path[0, 0]) * lateral_slope
        )
        self.ball_path[:, 1] = [
            self._reflect_lane(y) for y in raw_lateral
        ]
        total_lateral_travel = float(raw_lateral[-1] - raw_lateral[0])
        lane_width = self._lane_max - self._lane_min
        self.drop_wall_bounces = max(
            0,
            int(np.floor(
                (abs(total_lateral_travel) + 0.5 * lane_width)
                / lane_width
            )),
        )
        path_steps = np.linalg.norm(np.diff(self.ball_path, axis=0), axis=1)
        self.ball_path_cumulative = np.concatenate([[0.0], np.cumsum(path_steps)])
        self.ball_path_length = float(self.ball_path_cumulative[-1])
        self.ball_start = self.ball_path[0].copy()
        self.ball_valley = self.ball_path[self.curve_segments].copy()
        self.ball_exit = self.ball_path[-1].copy()
        self.ball_drop = self.ball_start + np.array([
            -self._drop_forward_travel,
            0.0,
            self.drop_height,
        ])
        exit_tangent = self.ball_path[-1] - self.ball_path[-2]
        self.up_angle = float(np.arctan2(exit_tangent[2], exit_tangent[0]))
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
        self.red_line_x = self.ramp_exit_x + self.red_line_gap
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
            self._ball_rigid.set_disable_gravity(True)
            self._ball_rigid.set_kinematic(True)
            self._ball_rigid.set_kinematic_target(sapien.Pose(self.ball_drop.tolist()))
            material = sapien.physx.PhysxMaterial(
                static_friction=0.12,
                dynamic_friction=0.08,
                restitution=0.18,
            )
            for shape in self._ball_rigid.get_collision_shapes():
                shape.set_physical_material(material)

        bowl_x = float(np.random.uniform(0.24, 0.28))
        bowl_pose = rand_pose(
            xlim=[bowl_x, bowl_x],
            ylim=[-0.20, -0.16],
            zlim=[self.table_top],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=False,
        )
        self.bowl = create_actor(
            self,
            pose=bowl_pose,
            modelname="002_bowl",
            model_id=self.bowl_id,
            convex=True,
            is_static=False,
            scale_mult=self.bowl_scale_mult,
        )
        self.bowl.set_mass(0.08)
        bowl_rigid = self._get_rigid(self.bowl)
        if bowl_rigid is not None:
            bowl_material = sapien.physx.PhysxMaterial(
                static_friction=0.9,
                dynamic_friction=0.9,
                restitution=0.0,
            )
            for shape in bowl_rigid.get_collision_shapes():
                shape.set_physical_material(bowl_material)

        self._update_release_velocity()
        self._compute_landing()
        required_landing_x = self.red_line_x + self.bowl_outer_radius + 0.005
        while self.landing[0] < required_landing_x:
            self.launch_speed *= 1.05
            self._update_release_velocity()
            self._compute_landing()
        self.physics_run_steps = self.physics_max_steps

        for part in self.ramp_parts:
            self.add_prohibit_area(part, padding=0.015)
        self.add_prohibit_area(self.bowl, padding=0.04)

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

    def _build_ramp_segment(self, name, point_a, point_b):
        center = 0.5 * (point_a + point_b)
        length = float(np.linalg.norm(point_b - point_a))
        delta = point_b - point_a
        angle = -float(np.arctan2(delta[2], delta[0]))
        quat = t3d.euler.euler2quat(0.0, angle, 0.0)
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

    def _update_release_velocity(self):
        horizontal_speed = self.launch_speed * np.cos(self.up_angle)
        self.release_velocity = np.array([
            horizontal_speed * self.release_direction_xy[0],
            horizontal_speed * self.release_direction_xy[1],
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
        self._physics_step_count = 0
        self.ball.set_pose(sapien.Pose(self.ball_drop.tolist()))
        if self._ball_rigid is not None:
            self._ball_rigid.set_disable_gravity(True)
            self._ball_rigid.set_kinematic(True)
            self._ball_rigid.set_linear_velocity(np.zeros(3))
            self._ball_rigid.set_angular_velocity(np.zeros(3))
            self._ball_rigid.set_kinematic_target(sapien.Pose(self.ball_drop.tolist()))

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
        required_x = self.red_line_x + self.bowl_outer_radius + 0.005
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
                left_ramp = position[0] > self.ramp_exit_x + self.ball_radius
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
                if landing[0] >= required_x and landing_in_lane:
                    break

            # Preserve the sampled angles but add speed if the physical ball
            # does not clear the mandatory catch boundary.
            if landing is None or landing[0] < required_x:
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

    def _set_kinematic_ball_pose(self, position, roll_distance=0.0):
        # A sphere moving toward +x rolls around the negative y axis.
        roll_angle = -float(roll_distance) / max(self.ball_radius, 1e-6)
        quat = t3d.quaternions.axangle2quat([0.0, 1.0, 0.0], roll_angle)
        pose = sapien.Pose(np.asarray(position).tolist(), quat.tolist())
        if self._ball_rigid is not None:
            self._ball_rigid.set_kinematic_target(pose)
        else:
            self.ball.set_pose(pose)

    def _advance_ball(self):
        if self._ball_phase == "dropping":
            self._drop_i += 1
            fraction = min(1.0, self._drop_i / float(self.drop_steps))
            vertical_fraction = fraction * fraction
            raw_y = self.ramp_center_y + self._drop_lateral_travel * fraction
            position = np.array([
                self.ball_drop[0] + (self.ball_start[0] - self.ball_drop[0]) * fraction,
                self._reflect_lane(raw_y),
                self.ball_drop[2] + (self.ball_start[2] - self.ball_drop[2]) * vertical_fraction,
            ])
            self._set_kinematic_ball_pose(position)
            if fraction >= 1.0:
                self._ball_phase = "rolling"

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
                -self.launch_speed / max(self.ball_radius, 1e-6),
                0.0,
            ])
            self._ball_rigid.set_linear_damping(0.08)
            self._ball_rigid.set_angular_damping(0.5)

    def _check_arm_ball_contact(self):
        if self._arm_ball_contact or not getattr(self, "_loaded", False):
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
        arm_tag = ArmTag("right")

        # Start the ball before the first robot command. Its drop and
        # accelerated roll advance concurrently with all arm actions.
        self._start_ball_motion(expert_demo=True)
        self.move(self.grasp_actor(self.bowl, arm_tag=arm_tag, pre_grasp_dis=0.10))
        self._weld_bowl_to_end_effector(arm_tag)
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.12, move_axis="arm"))

        bowl_now = np.asarray(self.bowl.get_pose().p)
        target = np.array([
            max(
                self.landing[0],
                self.red_line_x + self.bowl_outer_radius + 0.005,
            ),
            self.landing[1],
            self.table_top,
        ])
        displacement = target - bowl_now
        self.move(self.move_by_displacement(
            arm_tag=arm_tag,
            x=float(displacement[0]),
            y=float(displacement[1]),
            z=float(displacement[2]),
            move_axis="world",
        ))
        self.move(self.open_gripper(arm_tag))
        self._unweld_bowl()
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
            "{B}": f"002_bowl/base{self.bowl_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    # -------------------------------------------------------------- success
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
        behind_line = bool(
            bowl_position[0] - self.bowl_outer_radius
            >= self.red_line_x
            and abs(bowl_position[1] - self.ramp_center_y)
            <= self.ramp_half_width - self.bowl_outer_radius
        )
        return horizontal_offset, in_bowl, behind_line, ball_position, bowl_position

    def check_success(self):
        if not getattr(self, "_loaded", False) or self._ball_phase != "released":
            return False
        self._check_arm_ball_contact()
        _, in_bowl, behind_line, _, _ = self._catch_state()
        return bool(in_bowl and behind_line and not self._arm_ball_contact)

    def get_obs(self):
        obs = super().get_obs()
        try:
            offset, in_bowl, behind_line, ball_position, bowl_position = self._catch_state()
            obs["valley_catch"] = {
                "ball_position": list(map(float, ball_position)),
                "bowl_position": list(map(float, bowl_position)),
                "predicted_landing": list(map(float, self.landing)),
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
                "drop_time": float(self.drop_time),
                "ball_path_mode": str(self.ball_path_mode),
                "drop_wall_angle_deg": float(self.drop_wall_angle_deg),
                "drop_forward_angle_deg": float(self.drop_forward_angle_deg),
                "drop_wall_bounces": int(self.drop_wall_bounces),
                "roll_time": float(self.roll_time),
                "launch_speed": float(self.launch_speed),
                "curve_segments": int(self.curve_segments),
                "bowl_scale_mult": float(self.bowl_scale_mult),
            }
        except Exception:
            pass
        return obs
