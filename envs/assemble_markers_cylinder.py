from ._base_task import Base_Task
from .utils import *
import sapien
import sapien.physx
import numpy as np
import transforms3d as t3d


class assemble_markers_cylinder(Base_Task):
    """Dual-arm assembly with two graspable thin cylinders around a rotating center cylinder.

    The two outer pieces are horizontal cylinders that get attached to the side surface of
    the center post. Once close enough to the surface, they are welded kinematically so they
    rotate with the center cylinder and end in one straight line through the post.
    """

    SIDE_CYL_RADIUS_DEFAULT = 0.018
    SIDE_CYL_HALF_LEN_DEFAULT = 0.10
    CENTER_CYL_RADIUS_DEFAULT = 0.09
    CENTER_CYL_HALF_LEN_DEFAULT = 0.10
    ATTACH_DIST_DEFAULT = 0.06
    ATTACH_Z_DEFAULT = 0.04
    DWELL_STEPS_DEFAULT = 40
    ROTATION_SPEED_DEG_DEFAULT = 30.0
    ROTATION_DIRECTION_DEFAULT = "counter_clockwise"
    CENTER_TABLE_CLEARANCE_DEFAULT = 0.005
    CENTER_X_DEFAULT = 0.0
    CENTER_Y_DEFAULT = 0.0
    SIDE_GROUP_X_OFFSET_DEFAULT = 0.18
    SIDE_GROUP_Y_OFFSET_DEFAULT = 0.10
    SIDE_GROUP_X_JITTER_DEFAULT = 0.02
    SIDE_GROUP_Y_JITTER_DEFAULT = 0.008
    CENTER_STRIPE_COUNT_DEFAULT = 6
    CENTER_STRIPE_RADIAL_GAP_DEFAULT = 0.003
    CENTER_STRIPE_THICKNESS_DEFAULT = 0.004
    CENTER_STRIPE_WIDTH_DEFAULT = 0.014
    CENTER_STRIPE_HEIGHT_SCALE_DEFAULT = 0.92
    POST_GRASP_RIGHT_SHIFT_DEFAULT = 0.04
    PRE_CONTACT_CLEARANCE_DEFAULT = 0.04
    CONTACT_PUSH_INSET_DEFAULT = 0.006
    POST_RELEASE_LIFT_DEFAULT = 0.12
    RETREAT_SIDE_CLEARANCE_DEFAULT = 0.14
    RETREAT_FRONT_Y_OFFSET_DEFAULT = 0.06
    RETREAT_Z_CLEARANCE_DEFAULT = 0.18
    TARGET_YAWS_DEG = [180.0, 0.0]

    def setup_demo(self, **kwags):
        self._cfg = kwags.get("task_args", {}).get("assemble_markers_cylinder", {})
        self.center_cylinder = None
        self.side_cylinders = []
        self._center_rigid = None
        self.center_stripes = []
        self._stripe_base_yaws_deg = []
        self._spin_step = 0
        self._rotation_active = False
        self.attached = []
        self.attached_target_yaw_deg = []
        self.attached_yaw = []
        self.n_attached = 0
        super()._init_task_env_(**kwags)

    # ------------------------------------------------------------------ actors
    def _normalize_rotation_direction(self, direction) -> tuple[str, float]:
        value = str(direction).strip().lower().replace("-", "_").replace(" ", "_")
        if value in {"clockwise", "cw"}:
            return "clockwise", -1.0
        if value in {"counter_clockwise", "counterclockwise", "ccw", "anticlockwise"}:
            return "counter_clockwise", 1.0
        return self.ROTATION_DIRECTION_DEFAULT, 1.0

    def _graspable_cylinder_config(self, half_length: float, radius: float) -> dict:
        scale = [half_length, radius, radius]
        base_contact_points = [
            np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0.0], [0, 0, 0, 1]], dtype=np.float64),
            np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0.0], [0, 0, 0, 1]], dtype=np.float64),
            np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0.0], [0, 0, 0, 1]], dtype=np.float64),
            np.array([[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0.0], [0, 0, 0, 1]], dtype=np.float64),
        ]
        # Rotate every top-down grasp frame by 90 degrees around the approach axis so the
        # gripper yaw matches the horizontal cylinder geometry.
        grasp_basis = np.array(
            [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
            dtype=np.float64,
        )
        grasp_basis_inv = np.linalg.inv(grasp_basis)
        grasp_roll = np.eye(4, dtype=np.float64)
        grasp_roll[:3, :3] = t3d.axangles.axangle2mat([1.0, 0.0, 0.0], np.pi / 2.0)
        contact_points_pose = [
            (contact @ grasp_basis @ grasp_roll @ grasp_basis_inv).tolist()
            for contact in base_contact_points
        ]
        return {
            "center": [0, 0, 0],
            "extents": [2.0, 2.0, 2.0],
            "scale": scale,
            "target_pose": [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -1], [0, 0, 0, 1]]],
            "contact_points_pose": contact_points_pose,
            "transform_matrix": np.eye(4).tolist(),
            "functional_matrix": [
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0], [0.0, 0.0, 0.0, 1.0]],
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
            ],
            "contact_points_description": [],
            "contact_points_group": [[0, 1, 2, 3]],
            "contact_points_mask": [True],
            "target_point_description": ["The center point on the bottom of the cylinder."],
        }

    def _create_side_cylinder(self, pose: sapien.Pose, color, name: str) -> Actor:
        entity = create_cylinder(
            scene=self,
            pose=pose,
            radius=self.side_cyl_radius,
            half_length=self.side_cyl_half_len,
            color=color,
            name=name,
        )
        actor = Actor(entity, self._graspable_cylinder_config(self.side_cyl_half_len, self.side_cyl_radius))
        actor.set_mass(0.04)
        return actor

    def _stripe_pose(self, base_yaw_deg: float):
        a = self._world_angle_rad(base_yaw_deg)
        radial = np.array([np.cos(a), np.sin(a), 0.0], dtype=np.float64)
        tangent = np.array([-np.sin(a), np.cos(a), 0.0], dtype=np.float64)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        rot = np.column_stack([radial, tangent, up])
        center = np.array(
            [
                self._center_xy[0] + (self.center_cyl_radius + self.center_stripe_radial_gap) * radial[0],
                self._center_xy[1] + (self.center_cyl_radius + self.center_stripe_radial_gap) * radial[1],
                self.center_cyl_center_z,
            ],
            dtype=np.float64,
        )
        return sapien.Pose(center.tolist(), t3d.quaternions.mat2quat(rot).tolist())

    def _build_center_stripes(self):
        count = int(self.center_stripe_count)
        if count <= 0:
            return
        self.center_stripes = []
        self._stripe_base_yaws_deg = [float(i * 360.0 / count) for i in range(count)]
        for i, base_yaw in enumerate(self._stripe_base_yaws_deg):
            color = (0.90, 0.24, 0.18) if i == 0 else (0.95, 0.95, 0.96)
            pose = self._stripe_pose(base_yaw)
            stripe = create_visual_box(
                scene=self,
                pose=pose,
                half_size=(
                    self.center_stripe_thickness * 0.5,
                    self.center_stripe_width * 0.5,
                    self.center_cyl_half_len * self.center_stripe_height_scale,
                ),
                color=color,
                name=f"center_stripe_{i}",
            )
            self.center_stripes.append(stripe)

    def load_actors(self):
        cfg = self._cfg
        self.side_cyl_radius = float(cfg.get("side_cyl_radius", cfg.get("cyl_radius", self.SIDE_CYL_RADIUS_DEFAULT)))
        self.side_cyl_half_len = float(cfg.get("side_cyl_half_len", cfg.get("cyl_half_len", self.SIDE_CYL_HALF_LEN_DEFAULT)))
        self.center_cyl_radius = float(cfg.get("center_cyl_radius", cfg.get("cyl_radius", self.CENTER_CYL_RADIUS_DEFAULT)))
        self.center_cyl_half_len = float(cfg.get("center_cyl_half_len", cfg.get("cyl_half_len", self.CENTER_CYL_HALF_LEN_DEFAULT)))
        self.attach_dist = float(cfg.get("attach_dist", self.ATTACH_DIST_DEFAULT))
        self.attach_z = float(cfg.get("attach_z", self.ATTACH_Z_DEFAULT))
        self.dwell_steps = int(cfg.get("dwell_steps", self.DWELL_STEPS_DEFAULT))
        self.rotation_speed_deg = float(cfg.get("rotation_speed_deg", self.ROTATION_SPEED_DEG_DEFAULT))
        self.center_table_clearance = float(
            cfg.get("center_table_clearance", self.CENTER_TABLE_CLEARANCE_DEFAULT)
        )
        self.center_x = float(cfg.get("center_x", self.CENTER_X_DEFAULT))
        self.center_y = float(cfg.get("center_y", self.CENTER_Y_DEFAULT))
        self.side_group_x_offset = float(cfg.get("side_group_x_offset", self.SIDE_GROUP_X_OFFSET_DEFAULT))
        self.side_group_y_offset = float(cfg.get("side_group_y_offset", self.SIDE_GROUP_Y_OFFSET_DEFAULT))
        self.side_group_x_jitter = float(cfg.get("side_group_x_jitter", self.SIDE_GROUP_X_JITTER_DEFAULT))
        self.side_group_y_jitter = float(cfg.get("side_group_y_jitter", self.SIDE_GROUP_Y_JITTER_DEFAULT))
        self.center_stripe_count = int(cfg.get("center_stripe_count", self.CENTER_STRIPE_COUNT_DEFAULT))
        self.center_stripe_radial_gap = float(
            cfg.get("center_stripe_radial_gap", self.CENTER_STRIPE_RADIAL_GAP_DEFAULT)
        )
        self.center_stripe_thickness = float(
            cfg.get("center_stripe_thickness", self.CENTER_STRIPE_THICKNESS_DEFAULT)
        )
        self.center_stripe_width = float(cfg.get("center_stripe_width", self.CENTER_STRIPE_WIDTH_DEFAULT))
        self.center_stripe_height_scale = float(
            cfg.get("center_stripe_height_scale", self.CENTER_STRIPE_HEIGHT_SCALE_DEFAULT)
        )
        self.post_grasp_right_shift = float(
            cfg.get("post_grasp_right_shift", self.POST_GRASP_RIGHT_SHIFT_DEFAULT)
        )
        self.pre_contact_clearance = float(
            cfg.get("pre_contact_clearance", self.PRE_CONTACT_CLEARANCE_DEFAULT)
        )
        self.contact_push_inset = float(
            cfg.get("contact_push_inset", self.CONTACT_PUSH_INSET_DEFAULT)
        )
        self.post_release_lift = float(
            cfg.get("post_release_lift", self.POST_RELEASE_LIFT_DEFAULT)
        )
        self.retreat_side_clearance = float(
            cfg.get("retreat_side_clearance", self.RETREAT_SIDE_CLEARANCE_DEFAULT)
        )
        self.retreat_front_y_offset = float(
            cfg.get("retreat_front_y_offset", self.RETREAT_FRONT_Y_OFFSET_DEFAULT)
        )
        self.retreat_z_clearance = float(
            cfg.get("retreat_z_clearance", self.RETREAT_Z_CLEARANCE_DEFAULT)
        )
        self.rotation_direction, self.rotation_sign = self._normalize_rotation_direction(
            cfg.get("rotation_direction", self.ROTATION_DIRECTION_DEFAULT)
        )

        self._spin_step = 0
        self._rotation_active = False
        self._upright_q = np.array([0.7071068, 0.0, 0.7071068, 0.0], dtype=np.float64)
        self._upright_R = t3d.quaternions.quat2mat(self._upright_q)
        self._center_xy = np.array([self.center_x, self.center_y], dtype=np.float64)

        table_z = 0.74 + self.table_z_bias
        self.table_top_z = table_z
        self.center_cyl_center_z = table_z + self.center_table_clearance + self.center_cyl_half_len
        # Once a colored cylinder is attached, it becomes part of the rotating center body.
        # Keep the attached pieces centered on the post height so they co-rotate from the side.
        self.attach_center_z = self.center_cyl_center_z

        center_pose = sapien.Pose(
            [self._center_xy[0], self._center_xy[1], self.center_cyl_center_z],
            self._upright_q.tolist(),
        )
        self.center_cylinder = create_cylinder(
            scene=self,
            pose=center_pose,
            radius=self.center_cyl_radius,
            half_length=self.center_cyl_half_len,
            color=(0.55, 0.55, 0.60),
            name="rotating_center_cylinder",
        )
        self._center_rigid = None
        for comp in self.center_cylinder.get_components():
            if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent):
                self._center_rigid = comp
                comp.set_kinematic(True)
        self._build_center_stripes()

        self.side_cylinders = []
        side_colors = (
            (0.86, 0.33, 0.16),
            (0.18, 0.64, 0.55),
        )
        side_positions = [
            (self._center_xy[0] - self.side_group_x_offset, self._center_xy[1] + self.side_group_y_offset),
            (self._center_xy[0] + self.side_group_x_offset, self._center_xy[1] + self.side_group_y_offset),
        ]
        for i, (x_center, y_center) in enumerate(side_positions):
            x = float(x_center + np.random.uniform(-self.side_group_x_jitter, self.side_group_x_jitter))
            y = float(y_center + np.random.uniform(-self.side_group_y_jitter, self.side_group_y_jitter))
            yaw = float(np.pi / 2 + np.random.uniform(-np.pi / 36, np.pi / 36))
            q = t3d.quaternions.axangle2quat([0.0, 0.0, 1.0], yaw)
            pose = sapien.Pose([x, y, table_z + self.side_cyl_radius], q.tolist())
            piece = self._create_side_cylinder(pose, side_colors[i % len(side_colors)], f"side_cylinder_{i}")
            self.side_cylinders.append(piece)

        self.add_prohibit_area(self.center_cylinder, padding=0.05)
        for piece in self.side_cylinders:
            self.add_prohibit_area(piece, padding=0.03)

        self._register_curobo_obstacles()

        self.attached = [False] * len(self.side_cylinders)
        self.attached_target_yaw_deg = [None] * len(self.side_cylinders)
        self.attached_yaw = [None] * len(self.side_cylinders)
        self.n_attached = 0

    # ---------------------------------------------------- rotating target geometry
    def _spin_theta(self):
        return (
            self.rotation_sign
            * np.deg2rad(self.rotation_speed_deg)
            * self._spin_step
            * self.scene.get_timestep()
        )

    def _center_pose(self):
        spin_R = t3d.axangles.axangle2mat([0.0, 0.0, 1.0], self._spin_theta())
        pose_R = spin_R @ self._upright_R
        pose_q = t3d.quaternions.mat2quat(pose_R)
        return sapien.Pose(
            [self._center_xy[0], self._center_xy[1], self.center_cyl_center_z],
            pose_q.tolist(),
        )

    def _world_angle_rad(self, target_yaw_deg: float):
        return np.deg2rad(target_yaw_deg) + self._spin_theta()

    def _attached_piece_pose(self, target_yaw_deg: float):
        a = self._world_angle_rad(target_yaw_deg)
        radial = np.array([np.cos(a), np.sin(a), 0.0], dtype=np.float64)
        # Mounted pieces attach from opposite sides so the two rods lie on the same line.
        offset = self.center_cyl_radius + self.side_cyl_half_len
        center = np.array(
            [
                self._center_xy[0] + offset * radial[0],
                self._center_xy[1] + offset * radial[1],
                self.attach_center_z,
            ],
            dtype=np.float64,
        )
        center_q = t3d.quaternions.axangle2quat([0.0, 0.0, 1.0], a)
        return sapien.Pose(center.tolist(), center_q.tolist())

    def _rotated_ee_pose_about_world_z(self, arm_tag, yaw_deg: float):
        ee_pose = np.array(self.get_arm_pose(arm_tag), dtype=np.float64)
        rot_z = t3d.axangles.axangle2mat([0.0, 0.0, 1.0], np.deg2rad(yaw_deg))
        ee_rot = t3d.quaternions.quat2mat(ee_pose[3:])
        target_rot = rot_z @ ee_rot
        target_q = t3d.quaternions.mat2quat(target_rot)
        return ee_pose[:3].tolist() + target_q.tolist()

    def _target_radial_xy(self, target_yaw_deg: float):
        a = self._world_angle_rad(target_yaw_deg)
        return np.array([np.cos(a), np.sin(a)], dtype=np.float64)

    def _contact_piece_center(self, target_yaw_deg: float, radial_offset: float = 0.0):
        radial_xy = self._target_radial_xy(target_yaw_deg)
        offset = self.center_cyl_radius + self.side_cyl_radius + radial_offset
        return np.array(
            [
                self._center_xy[0] + offset * radial_xy[0],
                self._center_xy[1] + offset * radial_xy[1],
                self.attach_center_z,
            ],
            dtype=np.float64,
        )

    def _register_curobo_obstacles(self):
        if not hasattr(self, "robot"):
            return
        dims = [
            2.0 * float(self.center_cyl_radius),
            2.0 * float(self.center_cyl_radius),
            2.0 * float(self.center_cyl_half_len),
        ]
        pose = [
            float(self._center_xy[0]),
            float(self._center_xy[1]),
            float(self.center_cyl_center_z),
            1.0,
            0.0,
            0.0,
            0.0,
        ]
        try:
            self.robot.set_curobo_world_cuboids(
                {
                    self.center_cylinder.get_name(): {
                        "dims": dims,
                        "pose": pose,
                    }
                }
            )
        except Exception:
            pass

    def _retreat_waypoints(self, arm_tag):
        ee_pose = np.array(self.get_arm_pose(arm_tag), dtype=np.float64)
        side_sign = -1.0 if arm_tag == "left" else 1.0
        safe_x = float(
            self._center_xy[0] + side_sign * (self.center_cyl_radius + self.side_cyl_half_len + self.retreat_side_clearance)
        )
        safe_y = float(self._center_xy[1] + self.retreat_front_y_offset)
        safe_z = float(self.table_top_z + self.retreat_z_clearance)

        lift_pose = ee_pose.copy()
        lift_pose[2] = max(float(lift_pose[2]), safe_z)

        side_pose = lift_pose.copy()
        side_pose[0] = safe_x

        front_pose = side_pose.copy()
        front_pose[1] = safe_y

        return (
            lift_pose.tolist(),
            side_pose.tolist(),
            front_pose.tolist(),
            list(self.robot.left_original_pose if arm_tag == "left" else self.robot.right_original_pose),
        )

    def _move_allow_cleanup(self, action):
        prev_success = bool(self.plan_success)
        self.plan_success = True
        move_ok = self.move(action)
        cleanup_success = bool(self.plan_success) and (move_ok is not False)
        self.plan_success = prev_success and cleanup_success
        return cleanup_success

    def _retreat_arm_home(self, arm_tag, force=False):
        move_fn = self._move_allow_cleanup if force else self.move
        for waypoint in self._retreat_waypoints(arm_tag):
            if move_fn(self.move_to_pose(arm_tag=arm_tag, target_pose=waypoint)) is False and not force:
                break

    def _move_grasped_piece_center(self, piece, arm_tag, target_xyz, axes=("x", "y", "z")):
        axis_to_id = {"x": 0, "y": 1, "z": 2}
        for axis in axes:
            piece_p = np.array(piece.get_pose().p, dtype=np.float64)
            axis_id = axis_to_id[axis]
            delta = float(target_xyz[axis_id] - piece_p[axis_id])
            if abs(delta) <= 1e-4:
                continue
            self.move(self.move_by_displacement(arm_tag=arm_tag, **{axis: delta}))

    # ------------------------------------------------------- magnetic attach
    def _try_attach(self, idx, target_yaw_deg):
        if self.attached[idx]:
            return True
        piece = self.side_cylinders[idx]
        target_pose = self._attached_piece_pose(target_yaw_deg)
        piece_p = np.array(piece.get_pose().p, dtype=np.float64)
        target_p = np.array(target_pose.p, dtype=np.float64)
        xy_ok = float(np.linalg.norm(piece_p[:2] - target_p[:2])) <= self.attach_dist
        z_ok = abs(float(piece_p[2] - target_p[2])) <= self.attach_z
        touching_center = False
        try:
            touching_center = self.check_actors_contact(piece.get_name(), self.center_cylinder.get_name())
        except Exception:
            touching_center = False
        if not (touching_center or (xy_ok and z_ok)):
            return False

        piece.actor.set_pose(target_pose)
        for comp in piece.actor.get_components():
            if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent):
                comp.set_disable_gravity(True)
                comp.set_linear_velocity(np.zeros(3))
                comp.set_angular_velocity(np.zeros(3))
                comp.set_kinematic(True)
                comp.set_kinematic_target(target_pose)
        self.attached[idx] = True
        self.attached_target_yaw_deg[idx] = float(target_yaw_deg)
        self.attached_yaw[idx] = np.deg2rad(target_yaw_deg) % (2 * np.pi)
        self.n_attached += 1
        return True

    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
        if getattr(self, "center_cylinder", None) is None or self._center_rigid is None:
            return

        if self._rotation_active:
            self._spin_step += 1
        center_pose = self._center_pose()
        self._center_rigid.set_kinematic_target(center_pose)
        self.center_cylinder.set_pose(center_pose)
        for base_yaw, stripe in zip(self._stripe_base_yaws_deg, self.center_stripes):
            stripe.set_pose(self._stripe_pose(base_yaw))

        for idx, target_yaw_deg in enumerate(self.attached_target_yaw_deg):
            if not self.attached[idx] or target_yaw_deg is None:
                continue
            pose = self._attached_piece_pose(target_yaw_deg)
            self.side_cylinders[idx].actor.set_pose(pose)
            for comp in self.side_cylinders[idx].actor.get_components():
                if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent):
                    comp.set_kinematic_target(pose)

    def _dwell(self, idx=None, target_yaw_deg=None):
        for i in range(self.dwell_steps):
            if idx is not None and not self.attached[idx]:
                self._try_attach(idx, target_yaw_deg)
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (i % self.save_freq == 0):
                self._take_picture()

    # -------------------------------------------------------------- policy
    def _press_piece(self, idx, target_yaw_deg, arm_tag):
        piece = self.side_cylinders[idx]
        self.move(self.close_gripper(arm_tag, pos=0.6))
        grasp_contact_id = [0, 2][int(arm_tag == "left")]
        self.move(
            self.grasp_actor(
                piece,
                arm_tag=arm_tag,
                pre_grasp_dis=0.10,
                grasp_dis=0.0,
                contact_point_id=grasp_contact_id,
            )
        )
        # Lift the rod clear of the table first so the subsequent reorientation is collision free.
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=self.post_release_lift))
        self.move(self.move_by_displacement(arm_tag=arm_tag, x=self.post_grasp_right_shift))

        side_sign = -1.0 if arm_tag == "left" else 1.0
        # Mirror the 90-degree wrist turn so both arms keep a feasible elbow posture.
        gripper_rot_pose = self._rotated_ee_pose_about_world_z(arm_tag, -90.0 * side_sign)
        self.move(self.move_to_pose(arm_tag=arm_tag, target_pose=gripper_rot_pose))

        align_p = self._contact_piece_center(target_yaw_deg, radial_offset=self.pre_contact_clearance)
        contact_p = self._contact_piece_center(target_yaw_deg, radial_offset=0.0)
        press_p = self._contact_piece_center(target_yaw_deg, radial_offset=-self.contact_push_inset)

        # Align the rod's side approach with the cylinder centerline, then drive inward until contact.
        for waypoint, axes in (
            (align_p, ("y", "z", "x")),
            (contact_p, ("x",)),
            (press_p, ("x",)),
        ):
            self._move_grasped_piece_center(piece, arm_tag, waypoint, axes=axes)
            if self._try_attach(idx, target_yaw_deg):
                break
            self._dwell(idx=idx, target_yaw_deg=target_yaw_deg)
            if self.attached[idx]:
                break

        self._dwell(idx=idx, target_yaw_deg=target_yaw_deg)
        self._move_allow_cleanup(self.open_gripper(arm_tag))
        self._move_allow_cleanup(self.move_by_displacement(arm_tag=arm_tag, z=self.post_release_lift))
        self._retreat_arm_home(arm_tag, force=True)
        # if not self.attached[idx]:
        #     target_pose = self._attached_piece_pose(target_yaw_deg)
        #     piece.actor.set_pose(target_pose)
        #     self._try_attach(idx, target_yaw_deg)

        # self.move(self.open_gripper(arm_tag))
        # self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.10, move_axis="arm"))
        # self.move(self.back_to_origin(arm_tag))

    def play_once(self):
        left = ArmTag("left")
        right = ArmTag("right")
        self._rotation_active = True

        self._press_piece(1, self.TARGET_YAWS_DEG[1], right)
        self._press_piece(0, self.TARGET_YAWS_DEG[0], left)

        self._dwell()

        self.info["info"] = {
            "{A}": "side_cylinder_0",
            "{B}": "rotating_center_cylinder",
        }
        return self.info

    # ------------------------------------------------------------- success / metric
    def evenness_score(self):
        yaws = [y for y in self.attached_yaw if y is not None]
        if len(yaws) < 2:
            return 0.0
        s = sorted([float(y) % (2 * np.pi) for y in yaws])
        gap = float(np.rad2deg((s[1] - s[0]) % (2 * np.pi)))
        err = abs(gap - 180.0)
        return float(np.clip(1.0 - err / 180.0, 0.0, 1.0))

    def check_success(self):
        return bool(self.n_attached >= len(self.side_cylinders) and self.evenness_score() >= 0.99)

    def get_obs(self):
        obs = super().get_obs()
        obs["assemble"] = {
            "n_attached": int(self.n_attached),
            "attached": [bool(a) for a in self.attached],
            "attached_yaw_deg": [None if y is None else float(np.rad2deg(y)) for y in self.attached_yaw],
            "evenness_score": float(self.evenness_score()),
            "rotation_direction": self.rotation_direction,
            "rotation_speed_deg": float(self.rotation_speed_deg),
            "center_spin_deg": float(np.rad2deg(self._spin_theta())),
        }
        return obs
