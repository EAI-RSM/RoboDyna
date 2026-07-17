from ._base_task import Base_Task
from .utils import *
import sapien
import sapien.render
import numpy as np


class whack_a_mole(Base_Task):
    """Whack-a-mole. A fixed green board with a grid of holes spans both arms' reach.
    Up to N moles (default 4) continuously bob out of / back into their holes. Each
    arm starts with a blue cube gripped between its fingers (slightly larger than a
    hole) so the gripper cannot enter a hole. A hit requires physical mesh contact
    between a held cube and a mole that is above the board surface; hovering or
    buried moles do not count. Hit moles turn green and stay down.

    Optional rabbit distractors (num_distractors, up to M) occupy other holes and
    also bob. Touching a rabbit turns it red and fails the episode.

    difficulty=easy  — hit one mole at a time with the arm on that side.
    difficulty=hard  — hit two non-adjacent opposite-side moles with both arms at once.

    Success = every mole has been touched from above at least once, and no rabbit
    has been touched.
    """

    NUM_MOLES_DEFAULT = 4
    NUM_DISTRACTORS_DEFAULT = 0
    NUM_DISTRACTORS_MAX = 4       # hard cap M on rabbit distractors
    DIFFICULTY_DEFAULT = "easy"
    POP_STEPS_DEFAULT = 90          # sim steps each mole holds at the top of its cycle
    PRE_POP_STEPS_DEFAULT = 16
    TOUCH_TOL_DEFAULT = 0.04
    HOLE_COUNT_DEFAULT = 9
    HOLE_SIZE_DEFAULT = 0.055
    HOLE_BAR_THICKNESS_DEFAULT = 0.02
    CUBE_HOLE_SCALE_DEFAULT = 1.08  # cube side / hole_size (>1 so it can't enter a hole)

    BOARD_HALF = [0.30, 0.13, 0.048]
    BOARD_COLOR = [0.22, 0.62, 0.28]  # green box
    HIDE_DEPTH = 0.080
    MOLE_MODEL = "221_mole"
    RABBIT_MODEL = "222_rabbit"
    # Mesh is Y-up; rotate so height aligns with world Z.
    MOLE_Q = [0.70710678, 0.70710678, 0.0, 0.0]
    MOLE_COLOR = [0.45, 0.32, 0.22]
    MOLE_TOUCHED_COLOR = [0.20, 0.85, 0.28]  # green when hit
    RABBIT_COLOR = [0.98, 0.82, 0.05]         # yellow (distinct from brown moles)
    RABBIT_TOUCHED_COLOR = [0.92, 0.12, 0.10]  # red on illegal touch
    CUBE_COLOR = [0.15, 0.45, 0.95]            # blue — held mallet cubes
    MOLE_HEIGHT = 0.0585          # authored world height after scale
    RABBIT_HEIGHT = 0.0585        # match mole size
    POP_SPEED = 0.08              # m/s while rising / falling

    def setup_demo(self, **kwags):
        self._cfg = kwags.get("task_args", {}).get("whack_a_mole", {})
        # init kinematic bookkeeping before base init (may call _update_kinematic_tasks)
        self._global_step = 0
        self.moles = []
        self._mole_rigids = []
        self._mole_shapes = []
        self._mole_state = []
        self.rabbits = []
        self._rabbit_rigids = []
        self._rabbit_shapes = []
        self._rabbit_state = []
        self.distractor_hit = False
        self.holes = []
        self.hammer_cubes = {}
        self._cube_comps = {}
        self._cube_weld = {}
        super()._init_task_env_(**kwags)

    # ---------------------------------------------------------------- board
    def create_board(self):
        self.hole_size = float(self._cfg.get("hole_size", self.HOLE_SIZE_DEFAULT))
        self.hole_bar_thickness = float(
            self._cfg.get("hole_bar_thickness", self.HOLE_BAR_THICKNESS_DEFAULT))
        self.hole_count = int(self._cfg.get("hole_count", self.HOLE_COUNT_DEFAULT))
        board_cy = float(np.random.uniform(0.0, 0.05))
        self.board_center = np.array(
            [0.0, board_cy, self.table_top + self.BOARD_HALF[2]], dtype=float)
        board_color = self._cfg.get("board_color", self.BOARD_COLOR)
        self.board = create_hollow_box_with_holes(
            self.scene,
            sapien.Pose(p=self.board_center.tolist()),
            half_size=self.BOARD_HALF,
            color=list(board_color),
            is_static=True,
            name="hole_board",
            hole_count=self.hole_count,
            hole_size=self.hole_size,
            wall_thickness=0.02,
            top_thickness=0.02,
            bar_thickness=self.hole_bar_thickness,
        )
        self.board_top_z = float(self.board_center[2] + self.BOARD_HALF[2])

        self.hole_rows = int(np.floor(np.sqrt(self.hole_count)))
        self.hole_cols = int(np.ceil(self.hole_count / self.hole_rows))
        x_half, y_half = self.BOARD_HALF[0], self.BOARD_HALF[1]
        gap_x = (2 * x_half - self.hole_cols * self.hole_size) / (self.hole_cols + 1)
        gap_y = (2 * y_half - self.hole_rows * self.hole_size) / (self.hole_rows + 1)
        if gap_x < self.hole_bar_thickness or gap_y < self.hole_bar_thickness:
            raise ValueError("Requested hole_size is too large for the board top")
        x_centers = np.linspace(
            -x_half + gap_x + self.hole_size / 2.0,
            x_half - gap_x - self.hole_size / 2.0,
            self.hole_cols,
        )
        y_centers = np.linspace(
            -y_half + gap_y + self.hole_size / 2.0,
            y_half - gap_y - self.hole_size / 2.0,
            self.hole_rows,
        )
        # row-major: (r,c) with r along y, c along x
        self.holes = []
        self.hole_rc = []
        for r, dy in enumerate(y_centers):
            for c, cx in enumerate(x_centers):
                if len(self.holes) >= self.hole_count:
                    break
                self.holes.append(np.array(
                    [self.board_center[0] + cx, self.board_center[1] + dy], dtype=float))
                self.hole_rc.append((r, c))
        self.num_holes = len(self.holes)

    # ---------------------------------------------------------------- actors
    def load_actors(self):
        self._global_step = 0
        self.moles = []
        self._mole_rigids = []
        self._mole_shapes = []
        self._mole_state = []
        self.rabbits = []
        self._rabbit_rigids = []
        self._rabbit_shapes = []
        self._rabbit_state = []
        self.distractor_hit = False
        self.holes = []
        self.hole_rc = []
        self.touched = []
        self.schedule = []
        self._schedule_i = 0
        self.hammer_cubes = {}
        self._cube_comps = {}
        self._cube_weld = {}
        self._cubes_ready = False

        self.num_moles = int(self._cfg.get("num_moles", self.NUM_MOLES_DEFAULT))
        self.num_distractors = int(
            self._cfg.get("num_distractors", self.NUM_DISTRACTORS_DEFAULT))
        self.num_distractors = max(0, min(self.num_distractors, self.NUM_DISTRACTORS_MAX))
        self.difficulty = str(self._cfg.get("difficulty", self.DIFFICULTY_DEFAULT)).lower()
        if self.difficulty not in ("easy", "hard"):
            self.difficulty = self.DIFFICULTY_DEFAULT
        self.pop_steps = int(self._cfg.get("pop_steps", self.POP_STEPS_DEFAULT))
        self.pre_pop_steps = int(self._cfg.get("pre_pop_steps", self.PRE_POP_STEPS_DEFAULT))
        self.touch_tol = float(self._cfg.get("touch_tol", self.TOUCH_TOL_DEFAULT))
        self.mole_height = float(self._cfg.get("mole_height", self.MOLE_HEIGHT))
        self.rabbit_height = float(self._cfg.get("rabbit_height", self.RABBIT_HEIGHT))
        self.cube_hole_scale = float(
            self._cfg.get("cube_hole_scale", self.CUBE_HOLE_SCALE_DEFAULT))
        self.table_top = 0.74 + self.table_z_bias

        self.create_board()
        # Hide yellow wrist mounts + gray finger-pad meshes so only the blue
        # mallet cube reads as the in-hand object (pads otherwise look like a
        # gray cube covering the painted mallet).
        self._hide_wrist_camera_mounts()
        self._gray_out_arm_yellow_parts()
        self._hide_finger_pad_visuals()
        # Mallet cube: slightly larger than a hole so it cannot fall in.
        self.cube_side = float(self.hole_size) * self.cube_hole_scale
        self.cube_half = self.cube_side * 0.5
        if self.num_moles > self.num_holes:
            raise ValueError(f"num_moles ({self.num_moles}) > hole_count ({self.num_holes})")
        max_distractors = min(
            self.NUM_DISTRACTORS_MAX, self.num_holes - self.num_moles)
        if self.num_distractors > max_distractors:
            self.num_distractors = max_distractors

        hole_ids = self._assign_holes(self.num_moles)
        self.mole_holes = list(hole_ids)
        self._spawn_moles()

        rabbit_holes = self._assign_holes(
            self.num_distractors, exclude=self.mole_holes)
        self.rabbit_holes = list(rabbit_holes)
        self._spawn_rabbits()

        self.touched = [False] * self.num_moles
        self.schedule = self._build_schedule()
        self._schedule_i = 0

        self.prohibited_area.append([
            self.board_center[0] - self.BOARD_HALF[0] - 0.03,
            self.board_center[1] - self.BOARD_HALF[1] - 0.03,
            self.board_center[0] + self.BOARD_HALF[0] + 0.03,
            self.board_center[1] + self.BOARD_HALF[1] + 0.03,
        ])

    def _spawn_poppable(
        self,
        hole_idx,
        name,
        modelname,
        height,
        color,
        phase,
        actors,
        rigids,
        shapes_out,
        states,
    ):
        """Spawn one bobbing critter (mole or rabbit) in a hole."""
        pose_p = self._critter_pose_p(hole_idx, raised=False, height=height)
        actor = create_actor(
            self.scene,
            pose=sapien.Pose(p=pose_p.tolist(), q=self.MOLE_Q),
            modelname=modelname,
            model_id=0,
            convex=True,
            is_static=False,
        )
        actor.set_name(name)
        actor.set_mass(0.05)
        rigid = None
        for c in actor.actor.get_components():
            if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                rigid = c
                try:
                    c.set_linear_damping(50.0)
                    c.set_angular_damping(50.0)
                except Exception:
                    pass
                c.set_kinematic(False)
                c.set_disable_gravity(True)
                try:
                    for shape in c.get_collision_shapes():
                        shape.set_collision_groups([2, 2, 0, 0])
                except Exception:
                    pass
        shapes = []
        for c in actor.actor.get_components():
            if isinstance(c, sapien.render.RenderBodyComponent):
                shapes = list(c.render_shapes)
        hidden_z = float(pose_p[2])
        raised_z = float(
            self._critter_pose_p(hole_idx, raised=True, height=height)[2])
        if phase == 0:
            motion, z0, raised0, hold_left = "rising", hidden_z, False, 0
        elif phase == 1:
            motion, z0, raised0, hold_left = "hold", raised_z, True, self.pop_steps // 2
        else:
            motion, z0, raised0, hold_left = "falling", raised_z, True, 0
        actors.append(actor)
        rigids.append(rigid)
        shapes_out.append(shapes)
        states.append({
            "hole": int(hole_idx),
            "raised": bool(raised0),
            "touched": False,
            "motion": motion,
            "target_z": float(raised_z if motion == "rising" else hidden_z),
            "hidden_z": hidden_z,
            "raised_z": raised_z,
            "hold_left": int(hold_left),
            "height": float(height),
        })
        idx = len(actors) - 1
        self._set_critter_color(shapes, color)
        self._set_critter_pose(actors, rigids, states, idx, raised=raised0, z=z0)

    def _spawn_moles(self):
        for i, hole_idx in enumerate(self.mole_holes):
            self._spawn_poppable(
                hole_idx=hole_idx,
                name=f"mole_{i}",
                modelname=self.MOLE_MODEL,
                height=self.mole_height,
                color=self.MOLE_COLOR,
                phase=i % 3,
                actors=self.moles,
                rigids=self._mole_rigids,
                shapes_out=self._mole_shapes,
                states=self._mole_state,
            )

    def _spawn_rabbits(self):
        for i, hole_idx in enumerate(self.rabbit_holes):
            # offset phase so rabbits are not synced with moles
            self._spawn_poppable(
                hole_idx=hole_idx,
                name=f"rabbit_{i}",
                modelname=self.RABBIT_MODEL,
                height=self.rabbit_height,
                color=self.RABBIT_COLOR,
                phase=(i + 1) % 3,
                actors=self.rabbits,
                rigids=self._rabbit_rigids,
                shapes_out=self._rabbit_shapes,
                states=self._rabbit_state,
            )

    def _assign_holes(self, n, exclude=None):
        """Pick n distinct holes, preferring a left/right mix for hard mode."""
        if n <= 0:
            return []
        exclude = set(exclude or [])
        left = [i for i, h in enumerate(self.holes)
                if h[0] < -0.02 and i not in exclude]
        right = [i for i, h in enumerate(self.holes)
                 if h[0] > 0.02 and i not in exclude]
        center = [i for i, h in enumerate(self.holes)
                  if abs(h[0]) <= 0.02 and i not in exclude]
        chosen = []
        # take alternately from left/right so hard-mode pairing is feasible
        pools = [list(left), list(right)]
        for p in pools:
            np.random.shuffle(p)
        while len(chosen) < n and (pools[0] or pools[1]):
            for p in pools:
                if len(chosen) >= n:
                    break
                if p:
                    chosen.append(p.pop())
        leftover = [i for i in center + left + right if i not in chosen]
        np.random.shuffle(leftover)
        while len(chosen) < n and leftover:
            chosen.append(leftover.pop())
        return chosen[:n]

    def _holes_too_close(self, h1, h2):
        r1, c1 = self.hole_rc[h1]
        r2, c2 = self.hole_rc[h2]
        # chebyshev <= 1 => adjacent (incl. diagonal) — leave gripper clearance
        return max(abs(r1 - r2), abs(c1 - c2)) <= 1

    def _build_schedule(self):
        remaining = list(range(self.num_moles))
        np.random.shuffle(remaining)
        schedule = []
        if self.difficulty == "easy":
            for i in remaining:
                schedule.append([i])
            return schedule

        # hard: pair moles on opposite sides in non-adjacent holes
        while len(remaining) >= 2:
            pair = self._pick_hard_pair(remaining)
            if pair is None:
                break
            schedule.append(pair)
            for p in pair:
                remaining.remove(p)
        for i in remaining:
            schedule.append([i])
        return schedule

    def _pick_hard_pair(self, candidates):
        left, right = [], []
        for i in candidates:
            x = float(self.holes[self.mole_holes[i]][0])
            (left if x < 0 else right).append(i)
        np.random.shuffle(left)
        np.random.shuffle(right)
        options = []
        for a in left:
            for b in right:
                if not self._holes_too_close(self.mole_holes[a], self.mole_holes[b]):
                    options.append([a, b])
        if not options:
            # fallback: any non-adjacent pair (even same side)
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    a, b = candidates[i], candidates[j]
                    if not self._holes_too_close(self.mole_holes[a], self.mole_holes[b]):
                        options.append([a, b])
        if not options:
            return None
        return list(options[int(np.random.randint(0, len(options)))])

    # ---------------------------------------------------------- mole / rabbit kinematics
    def _critter_pose_p(self, hole_idx, raised, height):
        h = self.holes[hole_idx]
        if raised:
            z = self.board_top_z + height * 0.5 + 1e-3
        else:
            z = self.board_top_z - self.HIDE_DEPTH
        return np.array([h[0], h[1], z], dtype=float)

    def _mole_pose_p(self, hole_idx, raised):
        return self._critter_pose_p(hole_idx, raised, self.mole_height)

    def _set_critter_pose(self, actors, rigids, states, idx, raised=None, z=None):
        st = states[idx]
        hole = st["hole"]
        height = float(st.get("height", self.mole_height))
        if raised is None:
            raised = st["raised"]
        p = self._critter_pose_p(hole, raised, height)
        if z is not None:
            p[2] = float(z)
        pose = sapien.Pose(p=p.tolist(), q=self.MOLE_Q)
        actors[idx].actor.set_pose(pose)
        rigid = rigids[idx]
        if rigid is not None:
            try:
                rigid.set_linear_velocity([0.0, 0.0, 0.0])
                rigid.set_angular_velocity([0.0, 0.0, 0.0])
            except Exception:
                pass
        st["raised"] = bool(raised)

    def _set_mole_pose(self, idx, raised=None, z=None):
        self._set_critter_pose(
            self.moles, self._mole_rigids, self._mole_state, idx,
            raised=raised, z=z)

    def _set_rabbit_pose(self, idx, raised=None, z=None):
        self._set_critter_pose(
            self.rabbits, self._rabbit_rigids, self._rabbit_state, idx,
            raised=raised, z=z)

    @staticmethod
    def _set_critter_color(shapes, rgb):
        col = list(rgb)[:3] + [1.0]
        for s in shapes:
            try:
                s.material.set_base_color(col)
            except Exception:
                pass

    def _set_mole_color(self, idx, rgb):
        self._set_critter_color(self._mole_shapes[idx], rgb)

    def _set_rabbit_color(self, idx, rgb):
        self._set_critter_color(self._rabbit_shapes[idx], rgb)

    # ---------------------------------------------------------- hammer cubes
    @staticmethod
    def _pose7_to_mat(pose7):
        p = np.asarray(pose7[:3], dtype=float)
        q = np.asarray(pose7[3:], dtype=float)  # [w,x,y,z]
        from transforms3d.quaternions import quat2mat
        T = np.eye(4)
        T[:3, :3] = quat2mat(q)
        T[:3, 3] = p
        return T

    @staticmethod
    def _mat_to_pose(T):
        from transforms3d.quaternions import mat2quat
        q = mat2quat(T[:3, :3])  # [w,x,y,z]
        return sapien.Pose(T[:3, 3].tolist(), q.tolist())

    def _disable_link_render(self, entity, link_name):
        """Disable all render bodies on a named robot link (collision kept)."""
        try:
            link = entity.find_link_by_name(link_name)
        except Exception:
            link = None
        if link is None:
            return
        ent = link.entity if hasattr(link, "entity") else link
        for comp in ent.get_components():
            if not isinstance(comp, sapien.render.RenderBodyComponent):
                continue
            try:
                comp.disable()
            except Exception:
                pass
            try:
                comp.visibility = 0.0
            except Exception:
                pass

    def _hide_wrist_camera_mounts(self):
        """Disable any remaining camera-mount / camera meshes on each wrist."""
        robot = getattr(self, "robot", None)
        if robot is None:
            return
        for entity in (robot.left_entity, robot.right_entity):
            if entity is None:
                continue
            for link_name in ("camera_base", "camera"):
                self._disable_link_render(entity, link_name)

    def _hide_finger_pad_visuals(self):
        """Hide gray WSG finger / guide meshes that look like a held gray cube."""
        robot = getattr(self, "robot", None)
        if robot is None:
            return
        for entity in (robot.left_entity, robot.right_entity):
            if entity is None:
                continue
            for link_name in (
                    "finger_left", "finger_right",
                    "gripper_left", "gripper_right"):
                self._disable_link_render(entity, link_name)

    @staticmethod
    def _is_yellowish(rgba):
        if rgba is None or len(rgba) < 3:
            return False
        r, g, b = float(rgba[0]), float(rgba[1]), float(rgba[2])
        return r > 0.55 and g > 0.35 and b < 0.45 and (r + g) > (b + 0.55)

    def _gray_out_arm_yellow_parts(self):
        """Recolor yellow UR wrist/forearm plastic so it no longer looks like a cube."""
        robot = getattr(self, "robot", None)
        if robot is None:
            return
        gray = [0.45, 0.45, 0.47, 1.0]
        link_names = (
            "wrist_1_link", "wrist_2_link", "wrist_3_link",
            "forearm_link", "upper_arm_link", "ee_link",
            "wsg_50_base_link",
        )
        for entity in (robot.left_entity, robot.right_entity):
            if entity is None:
                continue
            for link_name in link_names:
                try:
                    link = entity.find_link_by_name(link_name)
                except Exception:
                    link = None
                if link is None:
                    continue
                ent = link.entity if hasattr(link, "entity") else link
                for comp in ent.get_components():
                    if not isinstance(comp, sapien.render.RenderBodyComponent):
                        continue
                    for shape in comp.render_shapes:
                        parts = []
                        try:
                            parts = list(shape.get_parts())
                        except Exception:
                            parts = []
                        mats = []
                        if parts:
                            for part in parts:
                                try:
                                    mats.append(part.get_material())
                                except Exception:
                                    try:
                                        mats.append(part.material)
                                    except Exception:
                                        pass
                        else:
                            try:
                                mats.append(shape.get_material())
                            except Exception:
                                try:
                                    mats.append(shape.material)
                                except Exception:
                                    pass
                        for mat in mats:
                            if mat is None:
                                continue
                            try:
                                bc = list(mat.base_color)
                            except Exception:
                                try:
                                    bc = list(mat.get_base_color())
                                except Exception:
                                    bc = None
                            if not self._is_yellowish(bc):
                                continue
                            try:
                                mat.set_base_color(gray)
                                mat.base_color = gray
                                mat.set_metallic(0.1)
                                mat.set_roughness(0.6)
                                mat.set_emission([0.0, 0.0, 0.0, 1.0])
                            except Exception:
                                try:
                                    mat.set_base_color(gray)
                                except Exception:
                                    pass

    def _paint_cube(self, cube):
        """Force the held mallet cube to the configured blue in RGB renders."""
        rgba = list(self.CUBE_COLOR)[:3] + [1.0]
        for c in cube.actor.get_components():
            if not isinstance(c, sapien.render.RenderBodyComponent):
                continue
            try:
                c.visibility = 1.0
                if not c.is_enabled:
                    c.enable()
            except Exception:
                pass
            for s in c.render_shapes:
                try:
                    mat = s.get_material()
                except Exception:
                    mat = s.material
                try:
                    mat.set_base_color(rgba)
                    mat.base_color = rgba
                    mat.set_metallic(0.0)
                    mat.set_roughness(0.35)
                    mat.set_specular(0.1)
                    mat.set_emission([0.0, 0.0, 0.0, 1.0])
                except Exception:
                    try:
                        mat.set_base_color(rgba)
                    except Exception:
                        pass

    def _finger_midpoint_world(self, arm_tag):
        """World position in the grasp aperture between the two finger pads."""
        entity = (self.robot.left_entity if str(arm_tag) == "left"
                  else self.robot.right_entity)
        fl = entity.find_link_by_name("finger_left")
        fr = entity.find_link_by_name("finger_right")
        base = entity.find_link_by_name("wsg_50_base_link")
        if fl is None or fr is None:
            return None
        p0 = np.array(fl.entity.get_pose().p, dtype=float)
        p1 = np.array(fr.entity.get_pose().p, dtype=float)
        mid = 0.5 * (p0 + p1)
        # Push from the gripper body toward the finger tips (not sideways).
        if base is not None:
            base_p = np.array(base.entity.get_pose().p, dtype=float)
            tip_dir = mid - base_p
            n = float(np.linalg.norm(tip_dir))
            if n > 1e-6:
                mid = mid + (0.035 / n) * tip_dir
        return mid

    def _grasp_local_T_for_arm(self, arm_tag):
        """EE-local transform that seats the cube between the finger pads."""
        ee = np.array(self.get_arm_pose(arm_tag), dtype=float)
        ee_T = self._pose7_to_mat(ee)
        mid = self._finger_midpoint_world(arm_tag)
        local = np.eye(4)
        if mid is None:
            # fallback: EE +Z points back to the wrist
            local[2, 3] = -0.045
            return local
        # keep EE orientation; only translate to the finger midpoint
        ee_R_inv = ee_T[:3, :3].T
        local[:3, 3] = ee_R_inv @ (mid - ee_T[:3, 3])
        return local

    def _spawn_and_grip_cubes(self):
        """Seat a cube between each gripper's fingers and close the jaws on it.

        The cube lives in the grasp aperture (between fingers), not as an EE tip
        attachment. Collision stays on so the oversized face blocks hole entry.
        """
        if self._cubes_ready:
            return
        left, right = ArmTag("left"), ArmTag("right")
        half = float(self.cube_half)

        prev_plan = self.plan_success
        self.move(self.open_gripper(left, pos=1.0), self.open_gripper(right, pos=1.0))
        if not self.plan_success:
            self.plan_success = prev_plan

        for arm in (left, right):
            local_T = self._grasp_local_T_for_arm(arm)
            ee = np.array(self.get_arm_pose(arm), dtype=float)
            pose = self._mat_to_pose(self._pose7_to_mat(ee) @ local_T)
            cube = create_box(
                self.scene,
                pose,
                half_size=[half, half, half],
                color=list(self.CUBE_COLOR),
                is_static=False,
                name=f"hammer_cube_{arm}",
            )
            cube.set_mass(0.03)
            self._paint_cube(cube)
            rigid = None
            for c in cube.actor.get_components():
                if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                    rigid = c
                    try:
                        c.set_linear_damping(20.0)
                        c.set_angular_damping(20.0)
                    except Exception:
                        pass
                    # Kinematic seat in the jaw aperture (between fingers).
                    # Collision only with moles (group bit 2) so PhysX can report
                    # real cube–mole mesh contact without fighting the robot/board.
                    c.set_kinematic(True)
                    c.set_disable_gravity(True)
                    c.set_kinematic_target(pose)
                    try:
                        for shape in c.get_collision_shapes():
                            shape.set_collision_groups([2, 2, 0, 0])
                    except Exception:
                        pass
            self.hammer_cubes[str(arm)] = cube
            self._cube_comps[str(arm)] = rigid
            self._cube_weld[str(arm)] = local_T.copy()

        # close jaws around the cube (visual "gripped between fingers")
        prev_plan = self.plan_success
        self.move(self.close_gripper(left, pos=0.0), self.close_gripper(right, pos=0.0))
        if not self.plan_success:
            self.plan_success = prev_plan

        # re-seat at the closed-jaw finger midpoint and lock the weld
        for arm in ("left", "right"):
            local_T = self._grasp_local_T_for_arm(ArmTag(arm))
            ee = np.array(self.get_arm_pose(ArmTag(arm)), dtype=float)
            pose = self._mat_to_pose(self._pose7_to_mat(ee) @ local_T)
            self.hammer_cubes[arm].actor.set_pose(pose)
            self._paint_cube(self.hammer_cubes[arm])
            rigid = self._cube_comps.get(arm)
            if rigid is not None:
                try:
                    rigid.set_kinematic_target(pose)
                except Exception:
                    pass
            self._cube_weld[arm] = local_T.copy()
        self._cubes_ready = True

    def _update_hammer_cubes(self):
        """Keep each cube seated in its gripper's jaw aperture (between fingers)."""
        if not getattr(self, "_cubes_ready", False):
            return
        for arm, local_T in self._cube_weld.items():
            ee = np.array(self.get_arm_pose(ArmTag(arm)), dtype=float)
            pose = self._mat_to_pose(self._pose7_to_mat(ee) @ local_T)
            self.hammer_cubes[arm].actor.set_pose(pose)
            # keep blue material applied (some passes reset render state)
            if self._global_step % 30 == 0:
                self._paint_cube(self.hammer_cubes[arm])
            rigid = self._cube_comps.get(arm)
            if rigid is not None:
                try:
                    rigid.set_kinematic_target(pose)
                except Exception:
                    pass

    def _advance_pop_cycle(self, actors, rigids, states, set_pose):
        """Advance one bobbing group (moles or rabbits) for a single sim step."""
        dt = float(self.scene.get_timestep())
        for idx, st in enumerate(states):
            rigid = rigids[idx]
            if rigid is None:
                continue

            # Hit critters stay pinned down (pose only if they drifted).
            if st["touched"]:
                st["motion"] = None
                st["raised"] = False
                cur_z = float(rigid.entity.get_pose().p[2])
                if abs(cur_z - st["hidden_z"]) > 1e-4:
                    set_pose(idx, raised=False)
                continue

            motion = st.get("motion")
            if motion is None:
                st["motion"] = "rising"
                st["target_z"] = st["raised_z"]
                motion = "rising"

            if motion == "hold":
                st["hold_left"] = int(st.get("hold_left", 0)) - 1
                st["raised"] = True
                # Do not set_pose every hold frame — leave the body still so
                # cube–mesh contacts can persist across steps.
                if st["hold_left"] <= 0:
                    st["motion"] = "falling"
                    st["target_z"] = st["hidden_z"]
                continue

            cur = np.array(rigid.entity.get_pose().p, dtype=float)
            if motion == "rising":
                next_z = cur[2] + self.POP_SPEED * dt
                reached = next_z >= st["raised_z"]
                if reached:
                    next_z = st["raised_z"]
                    st["motion"] = "hold"
                    st["hold_left"] = int(self.pop_steps)
                    st["raised"] = True
                else:
                    st["raised"] = False
                set_pose(idx, raised=st["raised"], z=next_z)
            elif motion == "falling":
                next_z = cur[2] - self.POP_SPEED * dt
                reached = next_z <= st["hidden_z"]
                if reached:
                    next_z = st["hidden_z"]
                    st["motion"] = "rising"
                    st["target_z"] = st["raised_z"]
                    st["raised"] = False
                else:
                    st["raised"] = True
                set_pose(idx, raised=st["raised"], z=next_z)

    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
        self._global_step = getattr(self, "_global_step", 0) + 1
        self._update_hammer_cubes()
        if not getattr(self, "_mole_state", None) and not getattr(self, "_rabbit_state", None):
            return

        # Poll BEFORE teleporting: set_pose on a dynamic body clears the
        # PhysX contact manifold from the previous scene.step().
        self._poll_mole_hits()
        self._poll_rabbit_hits()

        if getattr(self, "_mole_state", None):
            self._advance_pop_cycle(
                self.moles, self._mole_rigids, self._mole_state, self._set_mole_pose)
        if getattr(self, "_rabbit_state", None):
            self._advance_pop_cycle(
                self.rabbits, self._rabbit_rigids, self._rabbit_state,
                self._set_rabbit_pose)

    def _dwell(self, steps):
        for _ in range(int(steps)):
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (self._global_step % self.save_freq == 0):
                self._take_picture()

    def _sync_rise(self, idxs):
        """Align listed unhit moles onto the same rising edge (for dual-arm presses)."""
        for idx in idxs:
            st = self._mole_state[idx]
            if st["touched"]:
                continue
            st["motion"] = "rising"
            st["target_z"] = st["raised_z"]
            st["hold_left"] = 0
            st["raised"] = False
            self._set_mole_pose(idx, raised=False)

    def _pin_raised(self, idxs):
        """Keep target moles at the top until hit so they don't retreat mid-press."""
        for idx in idxs:
            st = self._mole_state[idx]
            if st["touched"]:
                continue
            st["motion"] = "hold"
            st["hold_left"] = 10**9
            st["raised"] = True
            self._set_mole_pose(idx, raised=True)

    def _wait_until_raised(self, idxs, max_steps=900, sync=False):
        """Wait until every listed (unhit) mole is at the top of its pop cycle."""
        if sync and len(idxs) > 1:
            self._sync_rise(idxs)
        for _ in range(int(max_steps)):
            ready = all(
                (self._mole_state[i]["touched"]
                 or (self._mole_state[i]["raised"]
                     and self._mole_state[i].get("motion") == "hold"))
                for i in idxs
            )
            if ready:
                # pin so the hold phase cannot expire during the press motion
                self._pin_raised(idxs)
                return True
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (self._global_step % self.save_freq == 0):
                self._take_picture()
        return False

    def _wait_until_down(self, idxs, max_steps=400):
        for _ in range(int(max_steps)):
            down = all(
                (not self._mole_state[i]["raised"])
                and self._mole_state[i].get("motion") is None
                for i in idxs
            )
            if down:
                return True
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (self._global_step % self.save_freq == 0):
                self._take_picture()
        return False

    # ------------------------------------------------------------- touch
    def _arm_for_hole(self, hole_idx):
        return ArmTag("right" if self.holes[hole_idx][0] > 0 else "left")

    def _mark_touched(self, idx):
        """Instant hit: recolor green and snap the mole fully down."""
        if self.touched[idx]:
            return
        self.touched[idx] = True
        st = self._mole_state[idx]
        st["touched"] = True
        st["motion"] = None
        st["hold_left"] = 0
        st["raised"] = False
        self._set_mole_color(idx, self.MOLE_TOUCHED_COLOR)
        self._set_mole_pose(idx, raised=False)

    def _critter_above_surface(self, actors, states, idx):
        """True iff the critter is not completely below the board top."""
        p = np.array(actors[idx].get_pose().p, dtype=float)
        height = float(states[idx].get("height", self.mole_height))
        top_z = float(p[2] + height * 0.5)
        return top_z > self.board_top_z + 1e-4

    def _mole_above_surface(self, idx):
        return self._critter_above_surface(self.moles, self._mole_state, idx)

    def _cube_actor_mesh_contact(self, actor_name):
        """True iff a held cube has PhysX mesh contact with the named actor."""
        cube_names = {f"hammer_cube_{arm}" for arm in getattr(self, "hammer_cubes", {})}
        if not cube_names:
            return False
        for cube_name in cube_names:
            try:
                if self.check_actors_contact(actor_name, cube_name):
                    return True
            except Exception:
                continue
        return False

    def _cube_mole_mesh_contact(self, idx):
        return self._cube_actor_mesh_contact(self.moles[idx].get_name())

    def _poll_mole_hits(self):
        """Every physics step: cube–mole mesh contact above surface -> green + snap down."""
        for idx, st in enumerate(self._mole_state):
            if st["touched"]:
                continue
            if not self._mole_above_surface(idx):
                continue
            if self._cube_mole_mesh_contact(idx):
                self._mark_touched(idx)

    def _mark_rabbit_touched(self, idx):
        """Illegal hit: recolor rabbit red, snap down, fail the episode."""
        st = self._rabbit_state[idx]
        if st["touched"]:
            return
        st["touched"] = True
        st["motion"] = None
        st["hold_left"] = 0
        st["raised"] = False
        self._set_rabbit_color(idx, self.RABBIT_TOUCHED_COLOR)
        self._set_rabbit_pose(idx, raised=False)
        self.distractor_hit = True
        # stop the expert — touching a distractor is an immediate failure
        self.plan_success = False

    def _poll_rabbit_hits(self):
        """Cube–rabbit contact above surface -> red + episode fail."""
        if not getattr(self, "_rabbit_state", None):
            return
        for idx, st in enumerate(self._rabbit_state):
            if st["touched"]:
                continue
            if not self._critter_above_surface(self.rabbits, self._rabbit_state, idx):
                continue
            if self._cube_actor_mesh_contact(self.rabbits[idx].get_name()):
                self._mark_rabbit_touched(idx)

    def _cube_bottom_z(self, arm_tag):
        """World Z of the lowest point of the held cube."""
        arm = str(arm_tag)
        if arm in getattr(self, "hammer_cubes", {}):
            cp = np.array(self.hammer_cubes[arm].get_pose().p, dtype=float)
            return float(cp[2] - self.cube_half)
        local_T = self._cube_weld.get(arm)
        if local_T is None:
            local_T = self._grasp_local_T_for_arm(arm_tag)
        ee = np.array(self.get_arm_pose(arm_tag), dtype=float)
        cube_T = self._pose7_to_mat(ee) @ local_T
        return float(cube_T[2, 3] - self.cube_half)

    def _ee_z_for_cube_bottom(self, arm_tag, cube_bottom_z):
        """EE world Z that places the held cube's underside at cube_bottom_z."""
        arm = str(arm_tag)
        local_T = self._cube_weld.get(arm)
        if local_T is None:
            local_T = self._grasp_local_T_for_arm(arm_tag)
        # cube_z ≈ ee_z + R_ee @ local_t; for near-vertical EE use local z
        local_z = float(local_T[2, 3])
        return float(cube_bottom_z + self.cube_half - local_z)

    def _hover_ee_z(self, arm_tag):
        target_bottom = float(self.board_top_z + self.mole_height + 0.06)
        return self._ee_z_for_cube_bottom(arm_tag, target_bottom)

    def _cube_xy_err(self, idx, arm_tag):
        hole = self.holes[self.mole_holes[idx]]
        cube_p = np.array(self.hammer_cubes[str(arm_tag)].get_pose().p, dtype=float)
        return float(np.linalg.norm(cube_p[:2] - hole[:2]))

    def _approach_hole(self, idx, arm_tag):
        """Rise, then slide XY at safe height, then fine-align the cube over the hole."""
        hole = self.holes[self.mole_holes[idx]]
        hover_z = self._hover_ee_z(arm_tag)

        cur = np.array(self.get_arm_pose(arm_tag), dtype=float)
        if cur[2] < hover_z - 0.01:
            self.move(self.move_by_displacement(
                arm_tag=arm_tag, z=float(hover_z - cur[2]), move_axis="world"))

        cur = np.array(self.get_arm_pose(arm_tag), dtype=float)
        self.move(self.move_by_displacement(
            arm_tag=arm_tag,
            x=float(hole[0] - cur[0]),
            y=float(hole[1] - cur[1]),
            z=float(hover_z - cur[2]),
            move_axis="world",
        ))

        # correct residual XY error measured at the cube (not the EE flange)
        for _ in range(3):
            if self._cube_xy_err(idx, arm_tag) < 0.02:
                break
            cube_p = np.array(
                self.hammer_cubes[str(arm_tag)].get_pose().p, dtype=float)
            self.move(self.move_by_displacement(
                arm_tag=arm_tag,
                x=float(hole[0] - cube_p[0]),
                y=float(hole[1] - cube_p[1]),
                move_axis="world",
            ))

    def _press_down(self, arm_tag, depth=None):
        """Press until the cube underside reaches just above the board top.

        Raised moles stick above the board, so the descending cube's mesh must
        physically contact them before the underside hits the surface.
        """
        if depth is None:
            cube_bottom = self._cube_bottom_z(arm_tag)
            # stop with the cube face barely above the board (never into a hole)
            target_bottom = self.board_top_z + 0.002
            depth = max(0.01, float(cube_bottom - target_bottom))
        return self.move_by_displacement(
            arm_tag=arm_tag, z=-float(depth), move_axis="world")

    def _press_up(self, arm_tag, depth=None):
        if depth is None:
            ee = np.array(self.get_arm_pose(arm_tag), dtype=float)
            hover_ee_z = self._hover_ee_z(arm_tag)
            depth = max(0.02, float(hover_ee_z - ee[2]))
        return self.move_by_displacement(
            arm_tag=arm_tag, z=float(depth), move_axis="world")

    # ------------------------------------------------------------- policy
    def play_once(self):
        # start with a cube gripped between each hand's fingers (blocks hole entry)
        self._spawn_and_grip_cubes()

        for group in self.schedule:
            if not self.plan_success:
                break
            group = [i for i in group if not self.touched[i]]
            if not group:
                continue
            if len(group) == 1:
                self._play_single(group[0])
            else:
                self._play_pair(group[0], group[1])

        arms_used = sorted({
            str(self._arm_for_hole(self.mole_holes[i])) for i in range(self.num_moles)
        })
        self.info["info"] = {
            "{A}": f"{self.MOLE_MODEL}/base0",
            "{B}": "hole_board",
            "{a}": arms_used[0] if len(arms_used) == 1 else "both arms",
            "{n}": str(self.num_moles),
            "{m}": self.difficulty,
            "{d}": str(self.num_distractors),
            "{R}": f"{self.RABBIT_MODEL}/base0",
        }
        return self.info

    def _play_single(self, idx):
        arm = self._arm_for_hole(self.mole_holes[idx])
        # keep gripper closed so the cube stays between the fingers
        self._approach_hole(idx, arm)
        if self._cube_xy_err(idx, arm) > 0.03:
            # one more full approach if still misaligned
            self._approach_hole(idx, arm)
        self._dwell(self.pre_pop_steps)

        self._wait_until_raised([idx])
        if self.touched[idx]:
            return
        if self._cube_xy_err(idx, arm) > 0.03:
            self._approach_hole(idx, arm)
            self._pin_raised([idx])
        self.move(self._press_down(arm))
        # hold the press so PhysX can register cube–mole mesh contact
        self._dwell(30)
        if not self.touched[idx]:
            # small extra push if the first press missed the contact window
            self.move(self.move_by_displacement(
                arm_tag=arm, z=-0.008, move_axis="world"))
            self._dwell(20)
        self.move(self._press_up(arm))

    def _play_pair(self, i, j):
        if self.holes[self.mole_holes[i]][0] > self.holes[self.mole_holes[j]][0]:
            i, j = j, i
        arm_i = self._arm_for_hole(self.mole_holes[i])
        arm_j = self._arm_for_hole(self.mole_holes[j])
        if arm_i == arm_j:
            self._play_single(i)
            self._play_single(j)
            return

        self._approach_hole(i, arm_i)
        self._approach_hole(j, arm_j)
        self._dwell(self.pre_pop_steps)

        self._wait_until_raised([i, j], sync=True)
        if self.touched[i] and self.touched[j]:
            return
        self.move(self._press_down(arm_i), self._press_down(arm_j))
        self._dwell(30)
        self.move(self._press_up(arm_i), self._press_up(arm_j))

    # ------------------------------------------------------------- success
    def check_success(self):
        if getattr(self, "distractor_hit", False):
            return False
        if not getattr(self, "touched", None):
            return False
        return bool(all(self.touched) and len(self.touched) == self.num_moles)

    def get_obs(self):
        obs = super().get_obs()
        obs["whack_a_mole"] = {
            "difficulty": str(getattr(self, "difficulty", "easy")),
            "num_moles": int(getattr(self, "num_moles", 0)),
            "num_distractors": int(getattr(self, "num_distractors", 0)),
            "touched": [bool(t) for t in getattr(self, "touched", [])],
            "distractor_hit": bool(getattr(self, "distractor_hit", False)),
            "raised": [bool(st.get("raised", False)) for st in getattr(self, "_mole_state", [])],
            "rabbit_raised": [
                bool(st.get("raised", False)) for st in getattr(self, "_rabbit_state", [])
            ],
            "holes": [int(h) for h in getattr(self, "mole_holes", [])],
            "rabbit_holes": [int(h) for h in getattr(self, "rabbit_holes", [])],
        }
        return obs
