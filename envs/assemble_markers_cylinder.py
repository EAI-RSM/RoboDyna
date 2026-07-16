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
    SIDE_GROUP_X_OFFSET_DEFAULT = 0.18
    SIDE_GROUP_Y_OFFSET_DEFAULT = 0.13
    TARGET_YAWS_DEG = [180.0, 0.0]

    def setup_demo(self, **kwags):
        self._cfg = kwags.get("task_args", {}).get("assemble_markers_cylinder", {})
        self.center_cylinder = None
        self.side_cylinders = []
        self._center_rigid = None
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
        self.side_group_x_offset = float(cfg.get("side_group_x_offset", self.SIDE_GROUP_X_OFFSET_DEFAULT))
        self.side_group_y_offset = float(cfg.get("side_group_y_offset", self.SIDE_GROUP_Y_OFFSET_DEFAULT))
        self.rotation_direction, self.rotation_sign = self._normalize_rotation_direction(
            cfg.get("rotation_direction", self.ROTATION_DIRECTION_DEFAULT)
        )

        self._spin_step = 0
        self._rotation_active = False
        self._upright_q = np.array([0.7071068, 0.0, 0.7071068, 0.0], dtype=np.float64)
        self._upright_R = t3d.quaternions.quat2mat(self._upright_q)
        self._center_xy = np.array([0.0, 0.0], dtype=np.float64)

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

        self.side_cylinders = []
        side_colors = (
            (0.86, 0.33, 0.16),
            (0.18, 0.64, 0.55),
        )
        side_positions = [
            (-self.side_group_x_offset, self.side_group_y_offset),
            (self.side_group_x_offset, self.side_group_y_offset),
        ]
        for i, (x_center, y_center) in enumerate(side_positions):
            x = float(x_center + np.random.uniform(-0.008, 0.008))
            y = float(y_center + np.random.uniform(-0.008, 0.008))
            yaw = float(np.pi / 2 + np.random.uniform(-np.pi / 36, np.pi / 36))
            q = t3d.quaternions.axangle2quat([0.0, 0.0, 1.0], yaw)
            pose = sapien.Pose([x, y, table_z + self.side_cyl_radius], q.tolist())
            piece = self._create_side_cylinder(pose, side_colors[i % len(side_colors)], f"side_cylinder_{i}")
            self.side_cylinders.append(piece)

        self.add_prohibit_area(self.center_cylinder, padding=0.05)
        for piece in self.side_cylinders:
            self.add_prohibit_area(piece, padding=0.03)

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
        if not (xy_ok and z_ok):
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
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.12, move_axis="arm"))

        target_pose = self._attached_piece_pose(target_yaw_deg)
        target_xy = np.array(target_pose.p[:2], dtype=np.float64)
        side_sign = -1.0 if arm_tag == "left" else 1.0
        side_stage_x = float(
            self._center_xy[0] + side_sign * (self.center_cyl_radius + self.side_cyl_half_len + 0.12)
        )
        approach_x = float(target_xy[0] + side_sign * 0.08)

        cur = np.array(
            self.robot.get_left_ee_pose() if arm_tag == "left" else self.robot.get_right_ee_pose(),
            dtype=np.float64,
        )

        # Keep each arm on its own side of the center post instead of taking a diagonal path
        # that can swing behind the cylinder and get stuck.
        dx = float(side_stage_x - cur[0])
        if abs(dx) > 1e-4:
            self.move(self.move_by_displacement(arm_tag=arm_tag, x=dx))

        cur = np.array(
            self.robot.get_left_ee_pose() if arm_tag == "left" else self.robot.get_right_ee_pose(),
            dtype=np.float64,
        )
        dy = float(target_xy[1] - cur[1])
        if abs(dy) > 1e-4:
            self.move(self.move_by_displacement(arm_tag=arm_tag, y=dy))

        dz = float(target_pose.p[2] - piece.get_pose().p[2])
        if abs(dz) > 1e-4:
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=dz, move_axis="arm"))

        cur = np.array(
            self.robot.get_left_ee_pose() if arm_tag == "left" else self.robot.get_right_ee_pose(),
            dtype=np.float64,
        )
        dx = float(approach_x - cur[0])
        if abs(dx) > 1e-4:
            self.move(self.move_by_displacement(arm_tag=arm_tag, x=dx))

        target_pose = self._attached_piece_pose(target_yaw_deg)
        piece_p = np.array(piece.get_pose().p, dtype=np.float64)
        target_p = np.array(target_pose.p, dtype=np.float64)
        corr_dx = float(target_p[0] - piece_p[0])
        corr_dy = float(target_p[1] - piece_p[1])
        if abs(corr_dy) > 1e-4:
            self.move(self.move_by_displacement(arm_tag=arm_tag, y=corr_dy))
        if abs(corr_dx) > 1e-4:
            self.move(self.move_by_displacement(arm_tag=arm_tag, x=corr_dx))
        self._dwell(idx=idx, target_yaw_deg=target_yaw_deg)
        self.move(self.open_gripper(arm_tag))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.10, move_axis="arm"))
        self.move(self.back_to_origin(arm_tag))
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
