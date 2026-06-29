from ._base_task import Base_Task
from .utils import *
from .utils.actor_utils import Actor
import sapien
import sapien.render
import sapien.physx
import numpy as np


class stab_moving_target(Base_Task):
    """Single-arm (right) dynamic-intercept task.

    A concentric-ring target board sways/translates across the far-mid of the table.
    A dart (a small elongated primitive with a 'tip' point) spawns in the near zone within
    the RIGHT arm's reach. The right arm grasps the dart, leads the target's motion, and
    drives the dart TIP into the bullseye. On a tip/target contact a stick (fixed) constraint
    forms (the dart is frozen relative to the target) and the radial offset from the bullseye
    center is measured. hit_score = clamp(1 - radial_offset / R_outer, 0, 1).

    No external asset: both the dart and the ring board are built from SAPIEN primitives at
    runtime. Reserved-range asset ids [340,349] are documented in the NOTICE below; nothing is
    written to assets/objects (pure primitives).
    """

    # ----- target geometry (a flat board standing up, facing -Y toward the robot/near zone)
    R_OUTER_DEFAULT = 0.075          # outer ring radius (m); hit_score denominator
    N_RINGS_DEFAULT = 4              # number of concentric rings
    BOARD_THICKNESS = 0.012          # half-thickness of the backing board (m)
    # ring colors from bullseye (center) outward: red, white, blue, yellow-ish
    RING_COLORS = [
        [0.85, 0.10, 0.10],          # bullseye
        [0.95, 0.95, 0.95],
        [0.15, 0.35, 0.80],
        [0.90, 0.80, 0.20],
    ]

    # ----- dart geometry (scale=[1,1,1]; matrices are in meters)
    DART_SHAFT_HALF = [0.045, 0.008, 0.008]   # half extents of the grip shaft (long axis = local X)
    DART_TIP_HALF = [0.012, 0.004, 0.004]     # half extents of the pointed tip
    DART_COLOR = [0.20, 0.85, 0.55]

    # ----- target motion (step-driven, applied in _update_kinematic_tasks)
    SWAY_AMP_DEFAULT = 0.10          # sway amplitude in x (m)
    SWAY_PERIOD_DEFAULT = 900        # sim steps per full sway cycle
    TARGET_SPEED_DEFAULT = 1.0       # multiplier on sway rate

    # ----- contact / stick
    STICK_DIST = 0.035               # tip-to-board-plane distance (m) that triggers the stick

    def setup_demo(self, **kwags):
        self._cfg = kwags.get("task_args", {}).get("stab_moving_target", {})
        # Initialize per-step state BEFORE base setup, because _init_task_env_ calls
        # _update_kinematic_tasks() during scene construction (before load_actors runs).
        self._step_count = 0
        self._stuck = False
        self._hit_radial_offset = None
        self.hit_score = 0.0
        self._target_rigid = None
        self._dart_rigid = None
        super()._init_task_env_(**kwags)

    # ------------------------------------------------------------------ actors
    def load_actors(self):
        cfg = self._cfg
        self.R_outer = float(cfg.get("r_outer", self.R_OUTER_DEFAULT))
        self.n_rings = int(cfg.get("n_rings", self.N_RINGS_DEFAULT))
        self.sway_amp = float(cfg.get("sway_amp", self.SWAY_AMP_DEFAULT))
        self.sway_period = int(cfg.get("sway_period", self.SWAY_PERIOD_DEFAULT))
        self.target_speed = float(cfg.get("target_speed", self.TARGET_SPEED_DEFAULT))

        # per-episode randomization of the target path
        self.sway_amp *= float(np.random.uniform(0.7, 1.0))
        self.sway_period = int(self.sway_period * float(np.random.uniform(0.85, 1.25)))
        self.sway_dir = float(np.random.choice([-1.0, 1.0]))
        self.sway_phase0 = float(np.random.uniform(0.0, 2.0 * np.pi))

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
        # so the bullseye never crosses the centerline, a modest forward y, and a low standing
        # height so the absolute intercept pose is plannable. (Diagnosis: a board at y~0.18 / z~0.86
        # whose sway reached x<0 made the very first targeting move unplannable every seed.)
        # Keep the board far enough FORWARD that it does not crowd the dart's near-zone grasp
        # (a board at y~0.10 blocked the top-down grasp), but with its x sway clamped to the
        # right half so the intercept stays reachable.
        self.target_y = 0.08 + float(np.random.uniform(-0.02, 0.02))
        self.target_center_x = float(np.random.uniform(0.11, 0.15))
        self.target_z = 0.77 + self.table_z_bias + self.R_outer
        # clamp sway so the bullseye stays in roughly x in [0.06, 0.20] (right-arm reachable)
        self.sway_amp = float(min(self.sway_amp, self.target_center_x - 0.06, 0.20 - self.target_center_x))
        board_pose = sapien.Pose(
            [self.target_center_x, self.target_y, self.target_z],
            [1, 0, 0, 0],
        )
        self.target = self._build_target(board_pose)
        # make the board a kinematic body so set_kinematic_target drives its motion
        self._target_rigid = self._get_rigid(self.target)
        if self._target_rigid is not None:
            self._target_rigid.set_kinematic(True)

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
        R = self.R_outer
        # backing board: a thin square slab. Local frame: board face normal is local +Y
        # (we set its world pose facing -Y toward the robot below). We model the slab thickness
        # along local Y, the disc in the local X-Z plane.
        back_half = [R * 1.15, th, R * 1.15]
        builder.add_box_collision(pose=sapien.Pose([0, 0, 0]), half_size=back_half,
                                  material=self.scene.default_physical_material)
        back_mat = sapien.render.RenderMaterial(base_color=[0.12, 0.12, 0.12, 1.0])
        builder.add_box_visual(pose=sapien.Pose([0, 0, 0]), half_size=back_half, material=back_mat)

        # concentric rings: stacked thin discs of decreasing radius, on the front face (local -Y).
        # Approximate each ring as a flat cylinder (RenderShapeCylinder axis = local X by default,
        # so rotate it to point along local Y). We use boxes->cylinders via builder.add_cylinder?
        # builder lacks cylinder visuals here, so render rings as thin square plates of decreasing
        # size layered slightly forward; visually reads as concentric rings.
        face_y = -(th + 0.001)
        for k in range(self.n_rings):
            frac = (self.n_rings - k) / self.n_rings   # outer ring largest
            r = R * frac
            col = self.RING_COLORS[k % len(self.RING_COLORS)]
            plate_half = [r, 0.001, r]
            # push each successively-smaller ring a hair more forward so it isn't z-fought
            yk = face_y - 0.0008 * k
            mat = sapien.render.RenderMaterial(base_color=[*col, 1.0])
            builder.add_box_visual(pose=sapien.Pose([0, yk, 0]), half_size=plate_half, material=mat)

        builder.set_initial_pose(pose)
        entity = builder.build(name="target_board")

        # functional point 0 = bullseye center, on the front face (local -Y), with +Z (the dart
        # tip should approach along the board's outward normal = local -Y / world toward robot).
        data = {
            "scale": [1.0, 1.0, 1.0],
            "center": [0, 0, 0],
            "extents": [2 * R, 2 * th, 2 * R],
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

    # ------------------------------------------------------------- helpers
    def _get_rigid(self, actor):
        for c in actor.actor.get_components():
            if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                return c
        return None

    def _target_x_at(self, step):
        ph = self.sway_phase0 + 2.0 * np.pi * self.target_speed * step / max(1, self.sway_period)
        return self.target_center_x + self.sway_dir * self.sway_amp * np.sin(ph)

    def _bullseye_world(self):
        """World position of the bullseye center (functional point 0 of the board)."""
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

        # while stuck, keep the dart welded to the board (a fixed/stick constraint emulation)
        if self._stuck and self._dart_rigid is not None and self._target_rigid is not None:
            board_pose = self._target_rigid.entity.get_pose()
            tip_world = board_pose.to_transformation_matrix() @ np.append(self._stick_local, 1.0)
            dart_pose = sapien.Pose(tip_world[:3] - self._tip_offset_world, self._stick_dart_q)
            self._dart_rigid.set_kinematic_target(dart_pose)

    def _try_form_stick(self):
        """If the dart tip is within STICK_DIST of the board face plane and inside the board
        footprint, freeze the dart relative to the board (stick constraint) and record offset."""
        tip = np.array(self.dart.get_functional_point(0, "list")[:3])
        bull = self._bullseye_world()
        radial = float(np.linalg.norm(tip[[0, 2]] - bull[[0, 2]]))   # offset in board plane (x,z)
        plane_dist = abs(float(tip[1] - bull[1]))                    # along the board normal (y)
        if plane_dist <= self.STICK_DIST and radial <= self.R_outer * 1.2:
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
            self._hit_radial_offset = radial
            self.hit_score = float(np.clip(1.0 - radial / self.R_outer, 0.0, 1.0))
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
        if os.environ.get("STAB_DEBUG"):
            tip = np.array(self.dart.get_functional_point(0, "list")[:3])
            bull = self._bullseye_world()
            print(f"[STAB] {tag}: plan_success={self.plan_success} "
                  f"tip={np.round(tip,3).tolist()} bull={np.round(bull,3).tolist()} "
                  f"stuck={self._stuck} step={self._step_count}", flush=True)

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

        # 2) lead the target: predict where the bullseye will be a short horizon ahead. With the tip
        #    now pointing +Y, align the EE laterally (x) and in height (z) to the bullseye, then push
        #    the EE forward (+Y) so the tip drives into the board. Separate small relative moves keep
        #    each IK sub-goal reachable.
        lead = 60
        x_lead = self._target_x_at(self._step_count + lead)
        bull_now = self._bullseye_world()

        # lateral align (x) toward the predicted bullseye column
        tip = np.array(self.dart.get_functional_point(0, "list")[:3])
        self.move(self.move_by_displacement(arm_tag=arm, x=float(x_lead - tip[0]), move_axis="world"))
        self._dbg("after align x")

        # height align (z) to the bullseye level
        tip = np.array(self.dart.get_functional_point(0, "list")[:3])
        self.move(self.move_by_displacement(arm_tag=arm, z=float(bull_now[2] - tip[2]), move_axis="world"))
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
            "{B}": "target_board/base341",
            "{a}": str(arm),
        }
        return self.info

    # ------------------------------------------------------------------ success
    def check_success(self):
        if not self._stuck or self._hit_radial_offset is None:
            return False
        # success = dart embedded in the target (a stick formed within the outer ring)
        return bool(self._hit_radial_offset <= self.R_outer)

    # record the target pose / radial offset / hit score per frame
    def get_obs(self):
        obs = super().get_obs()
        if getattr(self, "target", None) is None or self._target_rigid is None:
            return obs
        bull = self._bullseye_world().tolist()
        tip = np.array(self.dart.get_functional_point(0, "list")[:3]).tolist()
        obs["stab"] = {
            "bullseye_world": bull,
            "dart_tip_world": tip,
            "target_x": float(self._target_x_at(self._step_count)),
            "stuck": bool(self._stuck),
            "radial_offset": float(self._hit_radial_offset) if self._hit_radial_offset is not None else -1.0,
            "hit_score": float(self.hit_score),
            "r_outer": float(self.R_outer),
        }
        return obs
