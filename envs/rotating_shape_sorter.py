from ._base_task import Base_Task
from .utils import *
import sapien
import sapien.physx
import sapien.render
import numpy as np
import transforms3d as t3d


class rotating_shape_sorter(Base_Task):
    """Pick a ball and drop it through a rotating corner hole into the box below.

    The cap spins deterministically every physics step so plan and render passes stay in lock-step.
    A colored sphere spawns on either the left or right side of the central box. The corresponding
    arm grasps the ball, moves it to a same-side release station over the rotating platform, waits
    for the circular corner hole to pass underneath, and releases so the ball falls into the box.
    """

    # ----- tunable params (CLASS DEFAULTS; overridable via task_args.rotating_shape_sorter) -----
    SPIN_SPEED_DEFAULT = 0.9          # cap angular speed (rad / sim-step-unit baseline scaling)
    CAP_HOLE_RADIUS_DEFAULT = 0.03    # radius of the circular opening at the cap center
    CAP_HOLE_DIAMETER_DEFAULT = 2.0 * CAP_HOLE_RADIUS_DEFAULT
    CAP_HOLE_CORNER_MARGIN_DEFAULT = 0.012  # clearance between the hole and the square edges
    CAP_HOLE_BANDS_DEFAULT = 40       # more bands => smoother approximation of the circular cutout
    BALL_RADIUS_DEFAULT = 0.025
    BALL_SIDE_CLEARANCE_DEFAULT = 0.14
    BALL_Y_JITTER_DEFAULT = 0.045
    TRANSPORT_CLEARANCE_Z_DEFAULT = 0.08
    RELEASE_CLEARANCE_Z_DEFAULT = 0.004
    HOLE_ALIGN_TOL_DEFAULT = 0.015
    ALIGN_SEARCH_STEPS_DEFAULT = 1200
    POST_RELEASE_STEPS_DEFAULT = 220
    RELEASE_OPEN_STEPS_DEFAULT = 200
    HOLE_DROP_INSET_DEFAULT = 0.006

    def setup_demo(self, **kwags):
        self._cfg = kwags.get("task_args", {}).get("rotating_shape_sorter", {})
        self._cap_step = 0
        self._cap_angle = 0.0
        self._cap_tracking = False
        self.ball_in_box = False
        self.ball_released = False
        self.selected_arm = None
        self.bucket_floor_z = 0.0
        self.hole_orbit_radius = 0.0
        self._drop_target_xy = np.zeros(2, dtype=np.float64)
        self._release_open_steps = self.RELEASE_OPEN_STEPS_DEFAULT
        super()._init_task_env_(**kwags)

    # ----------------------------------------------------------------- actors
    def load_actors(self):
        cfg = self._cfg
        self.spin_speed = float(cfg.get("spin_speed", self.SPIN_SPEED_DEFAULT))
        hole_diameter = cfg.get("cap_hole_diameter", None)
        if hole_diameter is None:
            self.cap_hole_radius = float(cfg.get("cap_hole_radius", self.CAP_HOLE_RADIUS_DEFAULT))
        else:
            self.cap_hole_radius = 0.5 * float(hole_diameter)
        self.cap_hole_corner_margin = float(
            cfg.get("cap_hole_corner_margin", self.CAP_HOLE_CORNER_MARGIN_DEFAULT)
        )
        self.cap_hole_bands = int(cfg.get("cap_hole_bands", self.CAP_HOLE_BANDS_DEFAULT))
        self.ball_radius = float(cfg.get("ball_radius", self.BALL_RADIUS_DEFAULT))
        self.ball_side_clearance = float(cfg.get("ball_side_clearance", self.BALL_SIDE_CLEARANCE_DEFAULT))
        self.ball_y_jitter = float(cfg.get("ball_y_jitter", self.BALL_Y_JITTER_DEFAULT))
        self.transport_clearance_z = float(
            cfg.get("transport_clearance_z", self.TRANSPORT_CLEARANCE_Z_DEFAULT)
        )
        self.release_clearance_z = float(
            cfg.get("release_clearance_z", self.RELEASE_CLEARANCE_Z_DEFAULT)
        )
        self.hole_align_tol = float(cfg.get("hole_align_tol", self.HOLE_ALIGN_TOL_DEFAULT))
        self.align_search_steps = int(cfg.get("align_search_steps", self.ALIGN_SEARCH_STEPS_DEFAULT))
        self.post_release_steps = int(
            cfg.get("post_release_steps", cfg.get("preview_steps", self.POST_RELEASE_STEPS_DEFAULT))
        )
        self.release_open_steps = int(cfg.get("release_open_steps", self.RELEASE_OPEN_STEPS_DEFAULT))
        self.hole_drop_inset = float(cfg.get("hole_drop_inset", self.HOLE_DROP_INSET_DEFAULT))

        # randomized spin direction
        self.spin_dir = float(np.random.choice([-1.0, 1.0]))
        # randomized speed jitter
        self.spin_omega = self.spin_dir * self.spin_speed * float(np.random.uniform(0.7, 1.3))

        z0 = 0.74 + self.table_z_bias
        self.table_top_z = z0

        # ---- bucket: a wide, SHALLOW open-top tray at table center-mid ----
        # Wide so each arm's drop spot stays over the cavity; shallow (low walls) so the tall walls
        # don't block the arm reaching over the centre (the deep-bucket version was unplannable).
        self.bucket_center = np.array([0.0, -0.02])     # x, y at center-mid
        self.bucket_half = 0.12                         # inner half-extent (xy) -- wide tray
        self.bucket_h = 0.15                            # support box / rim height
        wall_t = 0.010
        floor_z = z0
        bc = self.bucket_center
        # floor
        self.bucket_floor = create_box(
            scene=self, pose=sapien.Pose([bc[0], bc[1], floor_z + 0.006], [1, 0, 0, 0]),
            half_size=(self.bucket_half + wall_t, self.bucket_half + wall_t, 0.006),
            color=(0.55, 0.4, 0.25), name="bucket_floor", is_static=True,
        )
        self.bucket_floor_z = floor_z + 0.012
        cz = floor_z + 0.012 + self.bucket_h / 2.0
        walls = [
            ((bc[0], bc[1] + self.bucket_half + wall_t / 2), (self.bucket_half + wall_t, wall_t / 2)),
            ((bc[0], bc[1] - self.bucket_half - wall_t / 2), (self.bucket_half + wall_t, wall_t / 2)),
            ((bc[0] + self.bucket_half + wall_t / 2, bc[1]), (wall_t / 2, self.bucket_half + wall_t)),
            ((bc[0] - self.bucket_half - wall_t / 2, bc[1]), (wall_t / 2, self.bucket_half + wall_t)),
        ]
        self.bucket_walls = []
        for i, ((wx, wy), (hx, hy)) in enumerate(walls):
            w = create_box(
                scene=self, pose=sapien.Pose([wx, wy, cz], [1, 0, 0, 0]),
                half_size=(hx, hy, self.bucket_h / 2.0),
                color=(0.6, 0.45, 0.3), name=f"bucket_wall{i}", is_static=True,
            )
            self.bucket_walls.append(w)

        self.ball_radius = float(np.clip(self.ball_radius, 0.01, 0.04))
        self.ball_side_clearance = float(max(self.ball_side_clearance, 0.01))
        self.ball_y_jitter = float(np.clip(self.ball_y_jitter, 0.0, self.bucket_half * 0.85))
        self.ball_side = "left" if int(np.random.randint(0, 2)) == 0 else "right"
        side_sign = -1.0 if self.ball_side == "left" else 1.0
        # Spawn the ball outside the rotating cap footprint, not just outside the box wall,
        # so the grasp approach stays clear of the platform edge.
        platform_outer_half = self.bucket_half + 0.012
        ball_x = float(
            bc[0]
            + side_sign * (platform_outer_half + self.ball_radius + self.ball_side_clearance)
        )
        ball_y = float(bc[1] + np.random.uniform(-self.ball_y_jitter, self.ball_y_jitter))
        self.ball_color = np.random.uniform(0.15, 0.95, size=3).tolist()
        self.ball = self._build_ball(
            pose=sapien.Pose([ball_x, ball_y, floor_z + self.ball_radius], [1, 0, 0, 0]),
            color=self.ball_color,
        )

        # height of the cap plane (just above the bucket rim)
        self.cap_z = floor_z + 0.012 + self.bucket_h + 0.015
        self.cap_center = np.array([bc[0], bc[1], self.cap_z])

        # ---- rotating cap: a square collidable platform with a real circular corner hole ----
        self.cap_half_extent = self.bucket_half + 0.012
        self.cap_hole_radius = float(np.clip(
            self.cap_hole_radius,
            0.02,
            self.cap_half_extent - 0.02,
        ))
        self.cap_hole_diameter = 2.0 * self.cap_hole_radius
        self.cap_hole_corner_margin = float(np.clip(
            self.cap_hole_corner_margin,
            0.004,
            self.cap_half_extent - self.cap_hole_radius - 0.002,
        ))
        self.cap_thickness = 0.006
        self._cap_base_q = np.array([1.0, 0.0, 0.0, 0.0])
        self._cap_hole_local_xy = np.array([
            self.cap_half_extent - self.cap_hole_radius - self.cap_hole_corner_margin,
            self.cap_half_extent - self.cap_hole_radius - self.cap_hole_corner_margin,
        ], dtype=np.float64)
        self.hole_orbit_radius = float(np.linalg.norm(self._cap_hole_local_xy))
        self.cap_entity = self._build_cap()
        self._cap_rigid = self._get_rigid(self.cap_entity)
        if self._cap_rigid is not None:
            self._cap_rigid.set_kinematic(True)
        self._place_cap(0.0)
        radial_dir = np.array([side_sign, 0.0], dtype=np.float64)
        self._drop_target_xy = np.array(
            self.cap_center[:2] + radial_dir * max(self.hole_orbit_radius - self.hole_drop_inset, 0.0),
            dtype=np.float64,
        )

        # keep clutter / spawn collisions away
        self.add_prohibit_area(self.bucket_floor, padding=0.05)
        self.add_prohibit_area(self.ball, padding=0.03)

    # ----------------------------------------------------- rotating-cap state
    def _get_rigid(self, entity):
        for component in entity.get_components():
            if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
                return component
        return None

    def _add_cap_box(self, builder, x0, x1, y0, y1, material, visual_material):
        if x1 - x0 <= 1e-4 or y1 - y0 <= 1e-4:
            return
        pose = sapien.Pose(
            [0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.0],
            [1.0, 0.0, 0.0, 0.0],
        )
        half_size = [0.5 * (x1 - x0), 0.5 * (y1 - y0), self.cap_thickness]
        builder.add_box_collision(
            pose=pose,
            half_size=half_size,
            material=material,
        )
        builder.add_box_visual(
            pose=pose,
            half_size=half_size,
            material=visual_material,
        )

    def _build_ball(self, pose, color):
        entity = create_sphere(
            self,
            pose=pose,
            radius=self.ball_radius,
            color=color,
            is_static=False,
            name="sorter_ball",
        )
        z_off = 0.35 * self.ball_radius
        data = {
            "scale": [1.0, 1.0, 1.0],
            "center": [0.0, 0.0, 0.0],
            "extents": [2.0 * self.ball_radius, 2.0 * self.ball_radius, 2.0 * self.ball_radius],
            "transform_matrix": np.eye(4).tolist(),
            "target_pose": [np.eye(4).tolist()],
            "contact_points_pose": [
                [[0, 0, 1, 0.0], [1, 0, 0, 0.0], [0, 1, 0, z_off], [0, 0, 0, 1]],
                [[1, 0, 0, 0.0], [0, 0, -1, 0.0], [0, 1, 0, z_off], [0, 0, 0, 1]],
                [[-1, 0, 0, 0.0], [0, 0, 1, 0.0], [0, 1, 0, z_off], [0, 0, 0, 1]],
                [[0, 0, -1, 0.0], [-1, 0, 0, 0.0], [0, 1, 0, z_off], [0, 0, 0, 1]],
            ],
            "functional_matrix": [],
        }
        ball_actor = Actor(entity, data, mass=0.03)
        rigid = self._get_rigid(entity)
        if rigid is not None:
            try:
                inelastic = sapien.physx.PhysxMaterial(
                    static_friction=0.9,
                    dynamic_friction=0.9,
                    restitution=0.0,
                )
                for shape in rigid.get_collision_shapes():
                    shape.set_physical_material(inelastic)
            except Exception:
                pass
        return ball_actor

    def _build_cap(self):
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("dynamic")

        hole_bands = max(12, int(self.cap_hole_bands))
        cap_span = 2.0 * self.cap_half_extent
        band_h = cap_span / hole_bands
        hole_cx = float(self._cap_hole_local_xy[0])
        hole_cy = float(self._cap_hole_local_xy[1])
        physical_material = self.scene.default_physical_material
        visual_material = sapien.render.RenderMaterial(base_color=[0.30, 0.30, 0.34, 1.0])

        for band_idx in range(hole_bands):
            y0 = -self.cap_half_extent + band_idx * band_h
            y1 = min(self.cap_half_extent, y0 + band_h)
            band_cy = 0.5 * (y0 + y1)
            dy = band_cy - hole_cy

            if abs(dy) >= self.cap_hole_radius:
                self._add_cap_box(
                    builder,
                    -self.cap_half_extent,
                    self.cap_half_extent,
                    y0,
                    y1,
                    physical_material,
                    visual_material,
                )
                continue

            hole_dx = float(np.sqrt(max(self.cap_hole_radius ** 2 - dy ** 2, 0.0)))
            self._add_cap_box(
                builder,
                -self.cap_half_extent,
                hole_cx - hole_dx,
                y0,
                y1,
                physical_material,
                visual_material,
            )
            self._add_cap_box(
                builder,
                hole_cx + hole_dx,
                self.cap_half_extent,
                y0,
                y1,
                physical_material,
                visual_material,
            )

        builder.set_initial_pose(sapien.Pose(self.cap_center.tolist(), [1, 0, 0, 0]))
        return builder.build(name="sorter_cap")

    def _place_cap(self, angle):
        """Pose the rotating square platform with its corner pass-through hole."""
        self._cap_angle = float(angle)
        qz = t3d.quaternions.axangle2quat([0, 0, 1], angle)
        q = t3d.quaternions.qmult(qz, self._cap_base_q)
        pose = sapien.Pose(self.cap_center.tolist(), q.tolist())
        if getattr(self, "_cap_rigid", None) is not None and getattr(self, "_cap_tracking", False):
            self._cap_rigid.set_kinematic_target(pose)
        else:
            self.cap_entity.set_pose(pose)

    def _hole_world_xy_at_step(self, step):
        angle = self._cap_angle_at_step(step)
        rot = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ], dtype=np.float64)
        return self.cap_center[:2] + rot @ self._cap_hole_local_xy

    def _steps_until_hole_alignment(self, target_xy, max_steps=None, tol=None, lead_steps=0):
        target_xy = np.array(target_xy, dtype=np.float64)
        max_steps = int(self.align_search_steps if max_steps is None else max_steps)
        tol = float(self.hole_align_tol if tol is None else tol)
        lead_steps = max(0, int(lead_steps))
        best_steps = 0
        best_err = float("inf")
        for wait_steps in range(max_steps + 1):
            hole_xy = self._hole_world_xy_at_step(self._cap_step + wait_steps + lead_steps)
            err = float(np.linalg.norm(hole_xy - target_xy))
            if err < best_err:
                best_err = err
                best_steps = wait_steps
            if err <= tol:
                return wait_steps
        return best_steps

    def _move_ball_to_height(self, arm_tag: ArmTag, target_z: float):
        if getattr(self, "ball", None) is None:
            return
        dz = float(target_z) - float(self.ball.get_pose().p[2])
        if abs(dz) > 1e-4:
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=dz, move_axis="world"))

    def _dwell(self, steps: int):
        for i in range(max(0, int(steps))):
            self._update_kinematic_tasks()
            self.scene.step()
            self.ball_in_box = self._ball_in_box()
            if self.save_freq and (i % self.save_freq == 0):
                self._take_picture()

    def _ball_in_box(self):
        if getattr(self, "ball", None) is None:
            return False
        p = np.array(self.ball.get_pose().p, dtype=np.float64)
        in_x = abs(p[0] - self.bucket_center[0]) <= (self.bucket_half - 0.15 * self.ball_radius)
        in_y = abs(p[1] - self.bucket_center[1]) <= (self.bucket_half - 0.15 * self.ball_radius)
        above_floor = p[2] >= (self.bucket_floor_z + 0.2 * self.ball_radius)
        below_platform = p[2] <= (self.cap_z - self.cap_thickness - 0.01)
        return bool(in_x and in_y and above_floor and below_platform)

    def _update_kinematic_tasks(self):
        # base hook drives DOMINO's dynamic object motion; runs every physics step
        super()._update_kinematic_tasks()
        if getattr(self, "_cap_tracking", False):
            self._cap_step += 1
            # step-driven angle => identical in plan & render passes
            angle = self.spin_omega * (self._cap_step * 0.01)
            self._place_cap(angle)
        self.ball_in_box = self._ball_in_box()

    def _cap_angle_at_step(self, step):
        return self.spin_omega * (step * 0.01)

    # ----------------------------------------------------------------- policy
    def play_once(self):
        arm_name = "left" if self.ball_side == "left" else "right"
        arm_tag = ArmTag(arm_name)
        self.selected_arm = arm_name
        self._cap_tracking = True
        self.move(self.grasp_actor(self.ball, arm_tag=arm_tag, pre_grasp_dis=0.08))

        transport_z = float(
            self.cap_z + self.cap_thickness + self.ball_radius + self.transport_clearance_z
        )
        release_z = float(
            self.cap_z + self.cap_thickness + self.ball_radius + self.release_clearance_z
        )
        self._move_ball_to_height(arm_tag=arm_tag, target_z=transport_z)

        ball_xy = np.array(self.ball.get_pose().p[:2], dtype=np.float64)
        delta_xy = self._drop_target_xy - ball_xy
        if np.linalg.norm(delta_xy) > 1e-4:
            self.move(self.move_by_displacement(
                arm_tag=arm_tag,
                x=float(delta_xy[0]),
                y=float(delta_xy[1]),
            ))

        self._move_ball_to_height(arm_tag=arm_tag, target_z=release_z)
        release_target_xy = np.array(self.ball.get_pose().p[:2], dtype=np.float64)
        wait_steps = self._steps_until_hole_alignment(
            release_target_xy,
            lead_steps=self.release_open_steps,
        )
        self._dwell(wait_steps)
        self.ball_released = True
        self.move(self.open_gripper(arm_tag))
        self._dwell(20)
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.08, move_axis="arm"))
        self.move(self.back_to_origin(arm_tag))
        self._dwell(self.post_release_steps)
        self._cap_tracking = False
        self.ball_in_box = self._ball_in_box()

        self.info["info"] = {
            "{A}": "rotating sorter platform",
            "{B}": "circular hole",
            "{C}": "bucket",
            "{D}": "sorter_ball",
            "{a}": arm_name,
        }
        return self.info

    def check_success(self):
        self.ball_in_box = self._ball_in_box()
        self.info["ball_side"] = str(getattr(self, "ball_side", "left"))
        self.info["selected_arm"] = str(getattr(self, "selected_arm", "left"))
        self.info["ball_in_box"] = bool(self.ball_in_box)
        return bool(self.ball_in_box)

    # ----------------------------------------------------------------- obs
    def get_obs(self):
        obs = super().get_obs()
        obs["sorter"] = {
            "cap_angle": float(getattr(self, "_cap_angle", 0.0)),
            "spin_omega": float(getattr(self, "spin_omega", 0.0)),
            "cap_hole_radius": float(getattr(self, "cap_hole_radius", self.CAP_HOLE_RADIUS_DEFAULT)),
            "cap_hole_diameter": float(getattr(self, "cap_hole_diameter", self.CAP_HOLE_DIAMETER_DEFAULT)),
            "ball_radius": float(getattr(self, "ball_radius", self.BALL_RADIUS_DEFAULT)),
            "ball_side": str(getattr(self, "ball_side", "left")),
            "ball_color": list(getattr(self, "ball_color", [0.0, 0.0, 0.0])),
            "ball_center": self.ball.get_pose().p.tolist() if getattr(self, "ball", None) is not None else [0.0, 0.0, 0.0],
            "drop_target_xy": self._drop_target_xy.tolist() if hasattr(self, "_drop_target_xy") else [0.0, 0.0],
            "ball_in_box": bool(getattr(self, "ball_in_box", False)),
        }
        return obs
