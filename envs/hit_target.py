from ._base_task import Base_Task
from .utils import *
from .utils.actor_utils import Actor
import sapien
import sapien.render
import sapien.physx
import numpy as np


class hit_target(Base_Task):
    """Single-arm (right) dynamic-intercept task.

    A round target board sways/translates across the far-mid of the table.
    An optional blocker can be spawned in front of the target; it may stay fixed or
    sway left/right similarly to the target.
    A dart (a small elongated primitive with a 'tip' point) spawns in the near zone within
    the RIGHT arm's reach. The right arm grasps the dart, leads the target's motion, and
    drives the dart TIP into the yellow center circle. On a tip/target contact a stick
    (fixed) constraint forms (the dart is frozen relative to the target). The task succeeds
    only when the first stick lands within the yellow center circle; any other board contact
    is a failure.

    No external asset: both the dart and the target board are built from SAPIEN primitives at
    runtime. Reserved-range asset ids [340,349] are documented in the NOTICE below; nothing is
    written to assets/objects (pure primitives).
    """

    # ----- target geometry (a flat round board standing up, facing -Y toward the robot/near zone)
    BOARD_RADIUS_DEFAULT = 0.075
    CENTER_RADIUS_DEFAULT = 0.018
    N_RINGS_DEFAULT = 4
    BOARD_THICKNESS = 0.012          # half-thickness of the backing board (m)
    FACE_ROT_Q = [np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]  # cylinder axis +X -> +Y
    RING_COLORS = [
        [0.85, 0.10, 0.10],          # outer red
        [0.95, 0.95, 0.95],          # white
        [0.15, 0.35, 0.80],          # blue
        [0.90, 0.80, 0.20],          # yellow center
    ]
    BLOCKER_ENABLED_DEFAULT = False
    BLOCKER_DYNAMIC_DEFAULT = False
    BLOCKER_RADIUS_DEFAULT = 0.040
    BLOCKER_THICKNESS_DEFAULT = 0.002
    BLOCKER_Y_GAP_DEFAULT = 0.050
    BLOCKER_Z_OFFSET_DEFAULT = 0.0
    BLOCKER_X_OFFSET_DEFAULT = 0.0
    BLOCKER_SPEED_DEFAULT = 1.15
    BLOCKER_SPEED_MIN_DEFAULT = 0.70
    BLOCKER_SPEED_MAX_DEFAULT = 1.45
    BLOCKER_SPEED_MIN_DELTA_DEFAULT = 0.10
    BLOCKER_COLOR = [0.32, 0.32, 0.36]
    BLOCKER_CLEARANCE_Z = 0.020
    TARGET_Y_OFFSET_DEFAULT = 0.050

    # ----- dart geometry (scale=[1,1,1]; matrices are in meters)
    DART_SHAFT_HALF = [0.045, 0.008, 0.008]   # half extents of the grip shaft (long axis = local X)
    DART_TIP_HALF = [0.012, 0.004, 0.004]     # half extents of the pointed tip
    DART_COLOR = [0.20, 0.85, 0.55]

    # ----- target motion (step-driven, applied in _update_kinematic_tasks)
    SWAY_AMP_DEFAULT = 0.10          # sway amplitude in x (m)
    SWAY_PERIOD_DEFAULT = 900        # sim steps per full sway cycle
    TARGET_SPEED_DEFAULT = 1.0       # multiplier on sway rate
    MOTION_X_MIN_DEFAULT = 0.03
    MOTION_X_MAX_DEFAULT = 0.24
    TARGET_CENTER_X_MIN_DEFAULT = 0.11
    TARGET_CENTER_X_MAX_DEFAULT = 0.15

    # ----- contact / stick
    STICK_DIST = 0.035               # tip-to-board-plane distance (m) that triggers the stick

    def setup_demo(self, **kwags):
        self._cfg = kwags.get("task_args", {}).get("hit_target", {})
        # Initialize per-step state BEFORE base setup, because _init_task_env_ calls
        # _update_kinematic_tasks() during scene construction (before load_actors runs).
        self._step_count = 0
        self._stuck = False
        self._hit_center = False
        self._hit_planar_offset = None
        self._hit_radial_offset = None
        self.hit_score = 0.0
        self._target_rigid = None
        self._dart_rigid = None
        self._blocker_rigid = None
        super()._init_task_env_(**kwags)

    # ------------------------------------------------------------------ actors
    def load_actors(self):
        cfg = self._cfg
        self.board_radius = float(
            cfg.get("board_radius", cfg.get("board_half_size", cfg.get("r_outer", self.BOARD_RADIUS_DEFAULT)))
        )
        self.center_radius = float(
            cfg.get("center_radius", cfg.get("center_half_size", self.CENTER_RADIUS_DEFAULT))
        )
        self.center_radius = min(self.center_radius, self.board_radius * 0.25)
        self.n_rings = max(1, int(cfg.get("n_rings", self.N_RINGS_DEFAULT)))
        self.sway_amp = float(cfg.get("sway_amp", self.SWAY_AMP_DEFAULT))
        self.sway_period = int(cfg.get("sway_period", self.SWAY_PERIOD_DEFAULT))
        self.target_speed = float(cfg.get("target_speed", self.TARGET_SPEED_DEFAULT))
        self.motion_x_min = float(cfg.get("motion_x_min", self.MOTION_X_MIN_DEFAULT))
        self.motion_x_max = float(cfg.get("motion_x_max", self.MOTION_X_MAX_DEFAULT))
        if self.motion_x_min > self.motion_x_max:
            self.motion_x_min, self.motion_x_max = self.motion_x_max, self.motion_x_min
        self.target_center_x_min = float(cfg.get("target_center_x_min", self.TARGET_CENTER_X_MIN_DEFAULT))
        self.target_center_x_max = float(cfg.get("target_center_x_max", self.TARGET_CENTER_X_MAX_DEFAULT))
        if self.target_center_x_min > self.target_center_x_max:
            self.target_center_x_min, self.target_center_x_max = (
                self.target_center_x_max,
                self.target_center_x_min,
            )
        self.blocker_enabled = bool(cfg.get("blocker_enabled", cfg.get("use_blocker", self.BLOCKER_ENABLED_DEFAULT)))
        self.blocker_dynamic = bool(cfg.get("blocker_dynamic", self.BLOCKER_DYNAMIC_DEFAULT))
        self.blocker_radius = float(cfg.get("blocker_radius", self.BLOCKER_RADIUS_DEFAULT))
        self.blocker_radius = min(self.blocker_radius, self.board_radius * 0.95)
        self.blocker_thickness = float(cfg.get("blocker_thickness", self.BLOCKER_THICKNESS_DEFAULT))
        self.blocker_y_gap = abs(float(cfg.get("blocker_y_gap", self.BLOCKER_Y_GAP_DEFAULT)))
        self.blocker_z_offset = float(cfg.get("blocker_z_offset", self.BLOCKER_Z_OFFSET_DEFAULT))
        self.blocker_x_offset = float(cfg.get("blocker_x_offset", self.BLOCKER_X_OFFSET_DEFAULT))
        self.target_y_offset = float(cfg.get("target_y_offset", self.TARGET_Y_OFFSET_DEFAULT))
        self.blocker_speed = float(cfg.get("blocker_speed", self.BLOCKER_SPEED_DEFAULT))
        self.blocker_speed_min = float(cfg.get("blocker_speed_min", self.BLOCKER_SPEED_MIN_DEFAULT))
        self.blocker_speed_max = float(cfg.get("blocker_speed_max", self.BLOCKER_SPEED_MAX_DEFAULT))
        self.blocker_speed_min_delta = abs(
            float(cfg.get("blocker_speed_min_delta", self.BLOCKER_SPEED_MIN_DELTA_DEFAULT))
        )

        # per-episode randomization of the target path
        self.sway_amp *= float(np.random.uniform(0.7, 1.0))
        self.sway_period = int(self.sway_period * float(np.random.uniform(0.85, 1.25)))
        self.sway_dir = float(np.random.choice([-1.0, 1.0]))
        self.sway_phase0 = float(np.random.uniform(0.0, 2.0 * np.pi))
        if self.blocker_dynamic:
            lo = min(self.blocker_speed_min, self.blocker_speed_max)
            hi = max(self.blocker_speed_min, self.blocker_speed_max)
            blocker_speed = float(np.random.uniform(lo, hi))
            if hi - lo > 1e-6 and abs(blocker_speed - self.target_speed) < self.blocker_speed_min_delta:
                for _ in range(12):
                    blocker_speed = float(np.random.uniform(lo, hi))
                    if abs(blocker_speed - self.target_speed) >= self.blocker_speed_min_delta:
                        break
                else:
                    if blocker_speed >= self.target_speed:
                        blocker_speed = min(hi, self.target_speed + self.blocker_speed_min_delta)
                    else:
                        blocker_speed = max(lo, self.target_speed - self.blocker_speed_min_delta)
            self.blocker_speed = blocker_speed

        # ---- dart: near zone, RIGHT arm's reach (x>0). laid flat, long axis along +X.
        # qpos identity -> shaft long axis is local X = world X (points toward +X, the right).
        # We grasp the shaft from the top; the tip is the +X end.
        dart_x = float(np.random.uniform(0.10, 0.22))
        dart_y = float(np.random.uniform(-0.12, -0.04))
        dart_pose = sapien.Pose(
            [dart_x, dart_y, 0.74 + self.table_z_bias + self.DART_SHAFT_HALF[2]],
            [1, 0, 0, 0],
        )
        self.dart = self._build_dart(dart_pose)
        self.dart.set_mass(0.02)

        # ---- target board: mid zone, standing up facing the robot (-Y normal). It must stay
        # WHOLLY within the RIGHT arm's reachable workspace: x>0 (right half) with the sway kept
        # so the center circle never crosses the centerline, a modest forward y, and a low standing
        # height so the absolute intercept pose is plannable. (Diagnosis: a board at y~0.18 / z~0.86
        # whose sway reached x<0 made the very first targeting move unplannable every seed.)
        # Keep the board far enough FORWARD that it does not crowd the dart's near-zone grasp
        # (a board at y~0.10 blocked the top-down grasp), but with its x sway clamped to the
        # right half so the intercept stays reachable.
        self.target_y = 0.08 + self.target_y_offset + float(np.random.uniform(-0.02, 0.02))
        self.target_center_x = float(np.random.uniform(self.target_center_x_min, self.target_center_x_max))
        self.target_z = 0.77 + self.table_z_bias + self.board_radius
        # Clamp sway to the configured shared travel band so the target and blocker follow
        # the same widened left-right range while staying in the reachable workspace.
        self.sway_amp = float(
            max(
                0.0,
                min(
                    self.sway_amp,
                    self.target_center_x - self.motion_x_min,
                    self.motion_x_max - self.target_center_x,
                ),
            )
        )
        board_pose = sapien.Pose(
            [self.target_center_x, self.target_y, self.target_z],
            [1, 0, 0, 0],
        )
        self.target = self._build_target(board_pose)
        # make the board a kinematic body so set_kinematic_target drives its motion
        self._target_rigid = self._get_rigid(self.target)
        if self._target_rigid is not None:
            self._target_rigid.set_kinematic(True)

        self.blocker = None
        if self.blocker_enabled:
            self.blocker_y = self.target_y - self.blocker_y_gap
            self.blocker_z = self.target_z + self.blocker_z_offset
            blocker_pose = sapien.Pose(
                [self._blocker_x_at(0), self.blocker_y, self.blocker_z],
                [1, 0, 0, 0],
            )
            self.blocker = self._build_blocker(blocker_pose)
            self._blocker_rigid = self._get_rigid(self.blocker)
            if self._blocker_rigid is not None:
                self._blocker_rigid.set_kinematic(True)
            self.add_prohibit_area(self.blocker, padding=0.03)

        self.add_prohibit_area(self.dart, padding=0.04)

    # --------------------------------------------------------------- dart build
    def _build_dart(self, pose):
        builder = self.scene.create_actor_builder()
        sh = self.DART_SHAFT_HALF
        tp = self.DART_TIP_HALF
        tip_cx = sh[0] + tp[0]   # tip box centered just beyond the +X end of the shaft
        # collision
        builder.add_box_collision(pose=sapien.Pose([0, 0, 0]), half_size=sh,
                                  material=self.scene.default_physical_material)
        builder.add_box_collision(pose=sapien.Pose([tip_cx, 0, 0]), half_size=tp,
                                  material=self.scene.default_physical_material)
        # visual
        shaft_mat = sapien.render.RenderMaterial(base_color=[*self.DART_COLOR, 1.0])
        tip_mat = sapien.render.RenderMaterial(base_color=[0.95, 0.95, 0.30, 1.0])
        builder.add_box_visual(pose=sapien.Pose([0, 0, 0]), half_size=sh, material=shaft_mat)
        builder.add_box_visual(pose=sapien.Pose([tip_cx, 0, 0]), half_size=tp, material=tip_mat)
        builder.set_initial_pose(pose)
        entity = builder.build(name="dart")

        tip_x = sh[0] + 2 * tp[0]   # local x of the very tip apex
        data = {
            "scale": [1.0, 1.0, 1.0],
            "center": [0, 0, 0],
            "extents": [2 * tip_x, 2 * sh[1], 2 * sh[2]],
            "transform_matrix": np.eye(4).tolist(),
            "target_pose": [np.eye(4).tolist()],
            # grasp the shaft from straight above (top-down). This contact frame, fed through the
            # grasp builder (contact @ [[0,0,1],[-1,0,0],[0,-1,0]]), yields a gripper whose forward
            # (tool) axis points -Z (down) and a pre-grasp point directly above the shaft. Verified
            # numerically: gripper fwd = [0,0,-1], pre-grasp 0.2 m above.
            "contact_points_pose": [
                [[1, 0, 0, -0.01],
                 [0, 0, -1, 0.0],
                 [0, 1, 0, 0.0],
                 [0, 0, 0, 1]],
                [[1, 0, 0, 0.01],
                 [0, 0, -1, 0.0],
                 [0, 1, 0, 0.0],
                 [0, 0, 0, 1]],
            ],
            # functional point 0 = the dart TIP apex (local +X end), orientation matching the shaft
            "functional_matrix": [
                [[1, 0, 0, tip_x],
                 [0, 1, 0, 0.0],
                 [0, 0, 1, 0.0],
                 [0, 0, 0, 1]],
            ],
        }
        return Actor(entity, data, mass=0.02)

    # ------------------------------------------------------------- target build
    def _build_target(self, pose):
        builder = self.scene.create_actor_builder()
        th = self.BOARD_THICKNESS
        board_radius = self.board_radius
        face_rot = sapien.Pose([0, 0, 0], self.FACE_ROT_Q)
        # backing board: a thin round slab. Cylinders are X-axis-aligned by default, so rotate
        # them so the thickness runs along local Y and the visible disc lies in the X-Z plane.
        builder.add_cylinder_collision(
            pose=face_rot,
            radius=board_radius,
            half_length=th,
            material=self.scene.default_physical_material,
        )
        back_mat = sapien.render.RenderMaterial(base_color=[0.12, 0.12, 0.12, 1.0])
        builder.add_cylinder_visual(
            pose=face_rot,
            radius=board_radius * 1.15,
            half_length=th,
            material=back_mat,
        )

        # Round target pattern: red outer ring, then white, blue, and the yellow center circle.
        face_y = -(th + 0.001)
        for k in range(self.n_rings):
            frac = (self.n_rings - k) / self.n_rings
            radius = board_radius * frac
            if k == self.n_rings - 1:
                radius = min(radius, self.center_radius if self.center_radius > 0 else radius)
            col = self.RING_COLORS[min(k, len(self.RING_COLORS) - 1)]
            yk = face_y - 0.0008 * k
            mat = sapien.render.RenderMaterial(base_color=[*col, 1.0])
            builder.add_cylinder_visual(
                pose=sapien.Pose([0, yk, 0], self.FACE_ROT_Q),
                radius=radius,
                half_length=0.001,
                material=mat,
            )

        builder.set_initial_pose(pose)
        entity = builder.build(name="target_board")

        # functional point 0 = center of the yellow circle, on the front face (local -Y).
        data = {
            "scale": [1.0, 1.0, 1.0],
            "center": [0, 0, 0],
            "extents": [2 * board_radius, 2 * th, 2 * board_radius],
            "transform_matrix": np.eye(4).tolist(),
            "target_pose": [np.eye(4).tolist()],
            "contact_points_pose": [],
            "functional_matrix": [
                [[1, 0, 0, 0.0],
                 [0, 1, 0, face_y],
                 [0, 0, 1, 0.0],
                 [0, 0, 0, 1]],
            ],
        }
        return Actor(entity, data, mass=1.0)

    def _build_blocker(self, pose):
        builder = self.scene.create_actor_builder()
        th = self.blocker_thickness
        face_rot = sapien.Pose([0, 0, 0], self.FACE_ROT_Q)
        builder.add_cylinder_collision(
            pose=face_rot,
            radius=self.blocker_radius,
            half_length=th,
            material=self.scene.default_physical_material,
        )
        blocker_mat = sapien.render.RenderMaterial(base_color=[*self.BLOCKER_COLOR, 1.0])
        builder.add_cylinder_visual(
            pose=face_rot,
            radius=self.blocker_radius,
            half_length=th,
            material=blocker_mat,
        )

        builder.set_initial_pose(pose)
        entity = builder.build(name="target_blocker")

        data = {
            "scale": [1.0, 1.0, 1.0],
            "center": [0, 0, 0],
            "extents": [2 * self.blocker_radius, 2 * th, 2 * self.blocker_radius],
            "transform_matrix": np.eye(4).tolist(),
            "target_pose": [np.eye(4).tolist()],
            "contact_points_pose": [],
            "functional_matrix": [
                [[1, 0, 0, 0.0],
                 [0, 1, 0, -(th + 0.001)],
                 [0, 0, 1, 0.0],
                 [0, 0, 0, 1]],
            ],
        }
        return Actor(entity, data, mass=1.0)

    # ------------------------------------------------------------- helpers
    def _get_rigid(self, actor):
        for c in actor.actor.get_components():
            if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                return c
        return None

    def _target_x_at(self, step):
        ph = self.sway_phase0 + 2.0 * np.pi * self.target_speed * step / max(1, self.sway_period)
        return self.target_center_x + self.sway_dir * self.sway_amp * np.sin(ph)

    def _blocker_x_at(self, step):
        if self.blocker_dynamic:
            ph = self.sway_phase0 + 2.0 * np.pi * self.blocker_speed * step / max(1, self.sway_period)
            base_x = self.target_center_x + self.sway_dir * self.sway_amp * np.sin(ph)
        else:
            base_x = self.target_center_x
        return base_x + self.blocker_x_offset

    def _target_center_world(self):
        """World position of the yellow-center circle (functional point 0 of the board)."""
        return np.array(self.target.get_functional_point(0, "list")[:3])

    # ---------------------------------------------------- per-step kinematic motion
    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
        self._step_count += 1

        if self._target_rigid is not None and not self._stuck:
            x = self._target_x_at(self._step_count)
            cur = self._target_rigid.entity.get_pose()
            tgt = sapien.Pose([x, self.target_y, self.target_z], cur.q)
            self._target_rigid.set_kinematic_target(tgt)

        if self._blocker_rigid is not None and self.blocker_enabled and self.blocker_dynamic:
            x = self._blocker_x_at(self._step_count)
            cur = self._blocker_rigid.entity.get_pose()
            tgt = sapien.Pose([x, self.blocker_y, self.blocker_z], cur.q)
            self._blocker_rigid.set_kinematic_target(tgt)

        # while stuck, keep the dart welded to the board (a fixed/stick constraint emulation)
        if self._stuck and self._dart_rigid is not None and self._target_rigid is not None:
            board_pose = self._target_rigid.entity.get_pose()
            tip_world = board_pose.to_transformation_matrix() @ np.append(self._stick_local, 1.0)
            dart_pose = sapien.Pose(tip_world[:3] - self._tip_offset_world, self._stick_dart_q)
            self._dart_rigid.set_kinematic_target(dart_pose)

    def _try_form_stick(self):
        """Stick the dart on any board hit; success is only the yellow center circle."""
        tip = np.array(self.dart.get_functional_point(0, "list")[:3])
        target_center = self._target_center_world()
        planar_offset = tip[[0, 2]] - target_center[[0, 2]]
        radial_offset = float(np.linalg.norm(planar_offset))
        plane_dist = abs(float(tip[1] - target_center[1]))           # along the board normal (y)
        within_board = radial_offset <= self.board_radius
        if plane_dist <= self.STICK_DIST and within_board:
            self._dart_rigid = self._get_rigid(self.dart)
            if self._dart_rigid is None:
                return False
            self._dart_rigid.set_kinematic(True)
            # cache the tip location in the board's LOCAL frame so the weld tracks the board
            board_pose = self._target_rigid.entity.get_pose()
            inv = np.linalg.inv(board_pose.to_transformation_matrix())
            self._stick_local = (inv @ np.append(tip, 1.0))[:3]
            self._tip_offset_world = tip - np.array(self.dart.get_pose().p)
            self._stick_dart_q = self.dart.get_pose().q
            self._stuck = True
            self._hit_planar_offset = planar_offset.astype(float)
            self._hit_radial_offset = radial_offset
            self._hit_center = radial_offset <= self.center_radius
            self.hit_score = 1.0 if self._hit_center else 0.0
            return True
        return False

    def _dwell(self, steps):
        """Advance the sim (target keeps moving), recording frames, attempting the stick each step."""
        for i in range(int(steps)):
            self._update_kinematic_tasks()
            self.scene.step()
            if not self._stuck:
                self._try_form_stick()
            if self.save_freq and (i % self.save_freq == 0):
                self._take_picture()
            if self._stuck:
                break

    def _dbg(self, tag):
        import os
        if os.environ.get("HIT_TARGET_DEBUG") or os.environ.get("STAB_DEBUG"):
            tip = np.array(self.dart.get_functional_point(0, "list")[:3])
            target_center = self._target_center_world()
            blocker_msg = ""
            if getattr(self, "blocker", None) is not None:
                blocker_msg = f" blocker={np.round(self.blocker.get_pose().p,3).tolist()}"
            print(f"[HIT_TARGET] {tag}: plan_success={self.plan_success} "
                  f"tip={np.round(tip,3).tolist()} center={np.round(target_center,3).tolist()} "
                  f"stuck={self._stuck} center_hit={self._hit_center} step={self._step_count}{blocker_msg}",
                  flush=True)

    # ------------------------------------------------------------------ policy
    def play_once(self):
        arm = ArmTag("right")   # dart always spawns on the right (x>0)
        self._dbg("start")

        # 1) grasp the dart by its shaft
        self.move(self.grasp_actor(self.dart, arm_tag=arm, pre_grasp_dis=0.08, contact_point_id=0))
        self._dbg("after grasp")

        # lift, and at the same time YAW the gripper +90deg about world Z so the dart shaft/tip
        # (which lay along world +X after the top-down grasp) swings to point toward +Y -- i.e. at
        # the board. This lets the EE stay in its comfortable near-zone reach while the ~0.10 m tip
        # extension does the forward reaching into the board (a pure +Y EE push to the board's depth
        # was unplannable). The new EE quat = world_yaw(+90) composed with the current EE quat.
        import transforms3d as _t3d
        cur_q = np.array(self.get_arm_pose(str(arm))[3:])
        yaw = _t3d.quaternions.axangle2quat([0, 0, 1], np.pi / 2)
        new_q = _t3d.quaternions.qmult(yaw, cur_q)
        self.move(self.move_by_displacement(arm_tag=arm, z=0.10, quat=list(new_q), move_axis="world"))
        self._dbg("after lift+yaw")

        # 2) lead the target: predict where the center circle will be a short horizon ahead. With the tip
        #    now pointing +Y, align the EE laterally (x) and in height (z) to the center, then push
        #    the EE forward (+Y) so the tip drives into the board. Separate small relative moves keep
        #    each IK sub-goal reachable.
        lead = 60
        x_lead = self._target_x_at(self._step_count + lead)
        target_center_now = self._target_center_world()

        # lateral align (x) toward the predicted center column
        tip = np.array(self.dart.get_functional_point(0, "list")[:3])
        self.move(self.move_by_displacement(arm_tag=arm, x=float(x_lead - tip[0]), move_axis="world"))
        self._dbg("after align x")

        if self.blocker_enabled:
            tip = np.array(self.dart.get_functional_point(0, "list")[:3])
            blocker_top = self.blocker_z + self.blocker_radius + self.BLOCKER_CLEARANCE_Z
            lift_clear = max(0.0, float(blocker_top - tip[2]))
            if lift_clear > 1e-4:
                self.move(self.move_by_displacement(arm_tag=arm, z=lift_clear, move_axis="world"))
            self._dbg("after blocker lift")

            tip = np.array(self.dart.get_functional_point(0, "list")[:3])
            pass_gap = max(0.0, float(self.target_y - tip[1] - 0.03))
            if pass_gap > 1e-4:
                self.move(self.move_by_displacement(arm_tag=arm, y=pass_gap, move_axis="world"))
            self._dbg("after blocker pass")

            x_lead = self._target_x_at(self._step_count + 20)
            tip = np.array(self.dart.get_functional_point(0, "list")[:3])
            self.move(self.move_by_displacement(arm_tag=arm, x=float(x_lead - tip[0]), move_axis="world"))
            self._dbg("after blocker realign x")

        # height align (z) to the center-circle level
        target_center_now = self._target_center_world()
        tip = np.array(self.dart.get_functional_point(0, "list")[:3])
        self.move(self.move_by_displacement(arm_tag=arm, z=float(target_center_now[2] - tip[2]), move_axis="world"))
        self._dbg("after align z")

        # 3) drive forward into the board (+Y) in increments, dwelling so the moving target can be
        #    intercepted and the stick constraint can form. The tip leads the EE by ~0.10 m in +Y.
        tip = np.array(self.dart.get_functional_point(0, "list")[:3])
        gap = float(self.target_y - tip[1])   # close most of the remaining tip->board distance
        self.move(self.move_by_displacement(arm_tag=arm, y=max(0.0, gap - 0.02), move_axis="world"))
        self._dbg("after push")
        self._dwell(400)
        self._dbg("after dwell1")

        # if not yet stuck, nudge further toward the board and dwell again
        for _ in range(3):
            if self._stuck:
                break
            self.move(self.move_by_displacement(arm_tag=arm, y=0.05, move_axis="world"))
            self._dwell(250)

        self.info["info"] = {
            "{A}": "dart/base340",
            "{B}": "moving_target/base341",
            "{a}": str(arm),
        }
        return self.info

    # ------------------------------------------------------------------ success
    def check_success(self):
        return bool(self._stuck and self._hit_center)

    # record the target pose / center hit / hit score per frame
    def get_obs(self):
        obs = super().get_obs()
        if getattr(self, "target", None) is None or self._target_rigid is None:
            return obs
        target_center = self._target_center_world().tolist()
        tip = np.array(self.dart.get_functional_point(0, "list")[:3]).tolist()
        planar_offset = (
            self._hit_planar_offset.tolist() if self._hit_planar_offset is not None else [-1.0, -1.0]
        )
        obs["hit_target"] = {
            "target_center_world": target_center,
            "dart_tip_world": tip,
            "target_x": float(self._target_x_at(self._step_count)),
            "stuck": bool(self._stuck),
            "center_hit": bool(self._hit_center),
            "planar_offset": planar_offset,
            "radial_offset": float(self._hit_radial_offset) if self._hit_radial_offset is not None else -1.0,
            "hit_score": float(self.hit_score),
            "board_radius": float(self.board_radius),
            "center_radius": float(self.center_radius),
            "board_half_size": float(self.board_radius),
            "center_half_size": float(self.center_radius),
            "blocker_enabled": bool(self.blocker_enabled),
            "blocker_dynamic": bool(self.blocker_dynamic),
            "blocker_radius": float(self.blocker_radius) if self.blocker_enabled else 0.0,
            "blocker_speed": float(self.blocker_speed) if self.blocker_enabled else 0.0,
            "blocker_world": self.blocker.get_pose().p.tolist() if getattr(self, "blocker", None) is not None else [0.0, 0.0, 0.0],
        }
        return obs
