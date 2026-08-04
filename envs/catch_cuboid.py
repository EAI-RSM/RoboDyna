from ._base_task import Base_Task
from .utils import *
import sapien
import sapien.render
import numpy as np
from scene_utils import print_c, get_quat_euler

class catch_rat(Base_Task):
    """Catch a rat that pops up from holes on a mid-table board spanning both arms.

    Config options under task_args.catch_rat (independent booleans):

      catch_two_mice (option 1, default false)
          Two mice pop from opposite-side holes at once; both arms must catch
          them. Success requires BOTH mice held. When false: single mouse.

      opaque_surface (option 2, default false)
          Opaque colored top lattice (current solid board color). When false:
          glass / see-through tinted top lattice.

    The rat(s) are kinematic actors: while retracted they sit hidden below the
    board top, and during the pop window a step-driven schedule raises them to a
    grippable height, then drops them back down.
    """

    # ---- task params (class defaults; override via task_args.catch_rat in the config) ----
    CATCH_TWO_MICE_DEFAULT = False     # option 1
    OPAQUE_SURFACE_DEFAULT = False     # option 2 (false => glass top)
    NUM_APPEARANCES_DEFAULT = 3        # how many times a rat pops up over the episode
    POP_STEPS_DEFAULT = 90             # sim steps the rat stays raised per appearance (its window)
    PRE_POP_STEPS_DEFAULT = 12         # short settle before each pop while the arm pre-positions
    GRASP_TOL_DEFAULT = 0.01         # horizontal tolerance (m) used to normalize the catch offset

    RAT_HALF = [0.020, 0.026, 0.035]  # rat body half-extents (small graspable box)
    BOARD_HALF = [0.30, 0.13, 0.048]  # hole board half-extents (wide span, thin)
    POP_HEIGHT = 0.055                # how far above the board top the rat rises when popped
    HIDE_DEPTH = 0.075                # how far below the board top the rat hides when retracted
    # Per-episode rise/fall speed (m/s) is sampled uniformly from [min, max].
    RAT_MOVE_SPEED_MIN_DEFAULT = 0.04
    RAT_MOVE_SPEED_MAX_DEFAULT = 0.10


    def setup_demo(self, **kwags):
        # capture task-scoped params BEFORE init (kwags isn't stored on self otherwise)
        self._cfg = kwags.get("task_args", {}).get("catch_rat", {})
        self._parse_option()
        # init bookkeeping before base init (may call _update_kinematic_tasks)
        self._global_step = 0
        self.rats = []
        self._rat_rigids = []
        self._rat_holes = []
        self._rat_names = []
        self._rat_raised = []
        self._rat_auto_motion = []
        self._rat_pop_target_z = []
        self._rat_hidden_z = []
        super()._init_task_env_(**kwags)
        self._configure_observer_camera()

    def _as_bool(self, value, default):
        if value is None:
            return bool(default)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        s = str(value).strip().lower()
        if s in ("1", "true", "yes", "on"):
            return True
        if s in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"catch_rat expected a boolean, got {value!r}")

    def _parse_option(self):
        """Read option-1 / option-2 config flags (independent).

        Preferred keys:
          catch_two_mice   — option 1
          opaque_surface   — option 2

        Legacy ``option`` / record_demo ``--option`` convenience:
          1 / catch_two_mice / dual_catch  -> enable option 1
          2 / opaque_surface / opaque      -> enable option 2
        """
        cfg = self._cfg
        catch_two = cfg.get("catch_two_mice", None)
        opaque = cfg.get("opaque_surface", None)

        # Legacy single ``option`` value from older configs / --option CLI.
        legacy = cfg.get("option", None)
        if legacy is not None:
            aliases = {
                1: "catch_two_mice",
                2: "opaque_surface",
                "1": "catch_two_mice",
                "2": "opaque_surface",
                "catch_two_mice": "catch_two_mice",
                "dual_catch": "catch_two_mice",
                "opaque_surface": "opaque_surface",
                "opaque": "opaque_surface",
                # old names (mapped to the new flags they roughly meant)
                "transparent_grid": None,  # both flags false
            }
            if legacy not in aliases:
                raise ValueError(
                    "catch_rat option must be 1/catch_two_mice or "
                    "2/opaque_surface (or set catch_two_mice / opaque_surface booleans)")
            key = aliases[legacy]
            if key == "catch_two_mice" and catch_two is None:
                catch_two = True
            elif key == "opaque_surface" and opaque is None:
                opaque = True

        self.catch_two_mice = self._as_bool(catch_two, self.CATCH_TWO_MICE_DEFAULT)
        self.opaque_surface = self._as_bool(opaque, self.OPAQUE_SURFACE_DEFAULT)
        # aliases used elsewhere in the task
        self.dual_catch = self.catch_two_mice
        self.transparent_grid = not self.opaque_surface

    def _configure_observer_camera(self):
        """Frame the hole board from the table's upper-right corner."""
        camera = getattr(getattr(self, "cameras", None), "observer_camera", None)
        if camera is None:
            return
        camera_pos = np.array([0.38, 0.52, 1.45], dtype=np.float64)
        look_at = np.array([0.0, -0.05, 0.95], dtype=np.float64)
        forward = look_at - camera_pos
        forward /= np.linalg.norm(forward)
        left = np.cross(np.array([0.0, 0.0, 1.0]), forward)
        left /= np.linalg.norm(left)
        up = np.cross(forward, left)
        m = np.eye(4)
        m[:3, :3] = np.stack([forward, left, up], axis=1)
        m[:3, 3] = camera_pos
        camera.entity.set_pose(sapien.Pose(m))

    def create_board(self):
         # --- hole board: fixed, static, spanning the mid zone across both arms' reach ---
        self.hole_size = float(self._cfg.get("hole_size", 0.03))
        self.hole_bar_thickness = float(self._cfg.get("hole_bar_thickness", 0.02))
        self.hole_count = int(self._cfg.get("hole_count", 9))
        board_cy = float(np.random.uniform(0, 0.05))
        self.board_center = np.array([0.0, board_cy, self.table_top + self.BOARD_HALF[2]])
        self.board = create_hollow_box_with_holes(
            self.scene,
            sapien.Pose(p=self.board_center.tolist()),
            half_size=self.BOARD_HALF,
            color=[0.45, 0.32, 0.18],
            is_static=True,
            name="hole_board",
            hole_count=self.hole_count,
            hole_size=self.hole_size,
            wall_thickness=0.02,
            top_thickness=0.02,
            bar_thickness=self.hole_bar_thickness,
            top_transparent=self.transparent_grid,
        )
        self.board_top_z = self.board_center[2] + self.BOARD_HALF[2]

        # --- hole grid: calculate positions for N holes based on the chosen square size.
        self.hole_rows = int(np.floor(np.sqrt(self.hole_count)))
        self.hole_cols = int(np.ceil(self.hole_count / self.hole_rows))
        print_c(
            f"Hole grid: {self.hole_rows} x {self.hole_cols} = "
            f"{self.hole_rows * self.hole_cols}",
            color="blue",
        )

        x_half = self.BOARD_HALF[0]
        y_half = self.BOARD_HALF[1]
        gap_x = (2 * x_half - self.hole_cols * self.hole_size) / (self.hole_cols + 1)
        gap_y = (2 * y_half - self.hole_rows * self.hole_size) / (self.hole_rows + 1)
        if gap_x < self.hole_bar_thickness or gap_y < self.hole_bar_thickness:
            raise ValueError("Requested hole_size is too large for the board top")
        x_centers = np.linspace(
            -x_half + gap_x + self.hole_size / 2,
            x_half - gap_x - self.hole_size / 2,
            self.hole_cols,
        )
        y_centers = np.linspace(
            -y_half + gap_y + self.hole_size / 2,
            y_half - gap_y - self.hole_size / 2,
            self.hole_rows,
        )
        # row-major (r along y, c along x) so dual-gap checks use grid cells
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

    def _holes_too_close(self, h1, h2):
        """True if holes are adjacent (incl. diagonal): no empty cell between them."""
        r1, c1 = self.hole_rc[h1]
        r2, c2 = self.hole_rc[h2]
        # Chebyshev <= 1 => neighbors; require >= 1 cell gap => distance >= 2
        return max(abs(r1 - r2), abs(c1 - c2)) <= 1

    def _pick_dual_holes(self):
        """Pick left/right holes with >=1 cell gap; prefer similar rows.

        Far diagonal corners are hard for simultaneous grasps, so same-row /
        adjacent-row pairs are preferred when available.
        """
        left = [i for i, h in enumerate(self.holes) if h[0] < -0.02]
        right = [i for i, h in enumerate(self.holes) if h[0] > 0.02]
        if not left or not right:
            raise ValueError("catch_rat catch_two_mice needs holes on both table sides")
        pairs = [
            (a, b) for a in left for b in right
            if not self._holes_too_close(a, b)
        ]
        if not pairs:
            raise ValueError(
                "catch_rat catch_two_mice: no left/right hole pair with a "
                "one-cell gap (arms would collide)")
        preferred = [
            (a, b) for a, b in pairs
            if abs(self.hole_rc[a][0] - self.hole_rc[b][0]) <= 1
        ]
        pool = preferred if preferred else pairs
        a, b = pool[int(np.random.randint(0, len(pool)))]
        return int(a), int(b)

    def _spawn_rat(self, hole_idx, name):
        """Create one kinematic rat body under the given hole."""
        rat = create_box(
            self.scene,
            sapien.Pose(p=self._rat_pose_p(hole_idx, raised=False).tolist()),
            half_size=self.rat_half,
            color=[0.40, 0.40, 0.42],
            is_static=False,
            name=name,
        )
        rat.set_mass(0.02)
        rigid = None
        for c in rat.actor.get_components():
            if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                rigid = c
        if rigid is not None:
            rigid.set_kinematic(True)
            hidden_pose = self._rat_pose_p(hole_idx, raised=False)
            raised_z = self.board_top_z + self.rat_half[2] + 1e-4
            rigid.set_kinematic_target(sapien.Pose(p=hidden_pose.tolist()))
            self._rat_auto_motion.append("rising")
            self._rat_pop_target_z.append(float(raised_z))
            self._rat_hidden_z.append(float(hidden_pose[2]))
        else:
            self._rat_auto_motion.append(False)
            self._rat_pop_target_z.append(0.0)
            self._rat_hidden_z.append(0.0)
        self.rats.append(rat)
        self._rat_rigids.append(rigid)
        self._rat_holes.append(int(hole_idx))
        self._rat_names.append(name)
        self._rat_raised.append(False)
        print_c(f"{name} at {rat.get_pose().p} (hole {hole_idx})", color="blue")
        return rat

    # ---------------------------------------------------------------- actors
    def load_actors(self):
        # init per-step / bookkeeping state FIRST (base setup may call _update_kinematic_tasks)
        self._global_step = 0
        self.rats = []
        self._rat_rigids = []
        self._rat_holes = []
        self._rat_names = []
        self._rat_raised = []
        self._rat_auto_motion = []
        self._rat_pop_target_z = []
        self._rat_hidden_z = []
        self.catches = 0
        self.appearances_done = 0
        self._grab_offsets = []

        self.num_appearances = int(self._cfg.get("num_appearances", self.NUM_APPEARANCES_DEFAULT))
        self.pop_steps = int(self._cfg.get("pop_steps", self.POP_STEPS_DEFAULT))
        self.pre_pop_steps = int(self._cfg.get("pre_pop_steps", self.PRE_POP_STEPS_DEFAULT))
        self.grasp_tol = float(self._cfg.get("grasp_tol", self.GRASP_TOL_DEFAULT))
        self.rat_move_speed_min = float(
            self._cfg.get("rat_move_speed_min", self.RAT_MOVE_SPEED_MIN_DEFAULT))
        self.rat_move_speed_max = float(
            self._cfg.get("rat_move_speed_max", self.RAT_MOVE_SPEED_MAX_DEFAULT))
        if self.rat_move_speed_min <= 0 or self.rat_move_speed_max <= 0:
            raise ValueError("catch_rat rat_move_speed_min/max must be > 0")
        if self.rat_move_speed_min > self.rat_move_speed_max:
            raise ValueError("catch_rat rat_move_speed_min must be <= rat_move_speed_max")

        self.table_top = 0.74 + self.table_z_bias
        self.create_board()

        # Get the size of the rat based on the hole size, ensuring it fits within the hole.
        rat_size = float(self.hole_size) - 0.004
        if rat_size <= 0:
            raise ValueError("hole_size must be larger than 4 mm to fit the rat")
        rat_half_xy = rat_size / 2.0
        self.rat_half = [rat_half_xy, rat_half_xy, self.RAT_HALF[2]]
        # One shared speed per episode so dual rats stay synchronized.
        self._rat_pop_speed = float(np.random.uniform(
            self.rat_move_speed_min, self.rat_move_speed_max))
        print_c(
            f"rat move speed: {self._rat_pop_speed:.4f} m/s "
            f"(range [{self.rat_move_speed_min:.4f}, {self.rat_move_speed_max:.4f}])",
            color="blue",
        )

        if self.dual_catch:
            h_left, h_right = self._pick_dual_holes()
            self._spawn_rat(h_left, "rat_body_left")
            self._spawn_rat(h_right, "rat_body_right")
        else:
            hole = int(np.random.randint(0, self.num_holes))
            self._spawn_rat(hole, "rat_body")

        # convenience aliases used by the single-rat expert / obs
        self.rat = self.rats[0]
        self._rat_rigid = self._rat_rigids[0]
        self._active_hole = self._rat_holes[0]

        # keep clutter off the board
        self.prohibited_area.append([
            self.board_center[0] - self.BOARD_HALF[0] - 0.03,
            self.board_center[1] - self.BOARD_HALF[1] - 0.03,
            self.board_center[0] + self.BOARD_HALF[0] + 0.03,
            self.board_center[1] + self.BOARD_HALF[1] + 0.03,
        ])
        self.faling_once = False

    # ---------------------------------------------------------- rat kinematics
    def _rat_pose_p(self, hole_idx, raised):
        h = self.holes[hole_idx]
        if raised:
            # center the rat so its bottom is just above the board top (no overlap)
            z = self.board_top_z + self.rat_half[2] + 1e-4
        else:
            z = self.board_top_z - self.HIDE_DEPTH
        return np.array([h[0], h[1], z])

    def _set_rat(self, hole_idx, raised, rat_idx=0):
        self._rat_holes[rat_idx] = hole_idx
        self._rat_raised[rat_idx] = raised
        if rat_idx == 0:
            self._active_hole = hole_idx
        rigid = self._rat_rigids[rat_idx]
        if rigid is not None:
            rigid.set_kinematic_target(
                sapien.Pose(p=self._rat_pose_p(hole_idx, raised).tolist()))

    def _step_one_rat(self, i):
        rigid = self._rat_rigids[i]
        if rigid is None:
            return
        motion = self._rat_auto_motion[i]
        if motion:
            dt = float(self.scene.get_timestep())
            current_pose = rigid.entity.get_pose()
            cur_z = float(current_pose.p[2])
            if motion == "rising":
                next_z = cur_z + self._rat_pop_speed * dt
                reached = next_z >= self._rat_pop_target_z[i]
                if reached:
                    next_z = self._rat_pop_target_z[i]
                target_p = np.array(current_pose.p)
                target_p[2] = next_z
                rigid.set_kinematic_target(sapien.Pose(p=target_p, q=current_pose.q))
                if reached:
                    self._rat_auto_motion[i] = "falling"
                    self._rat_raised[i] = True
                return
            if motion == "falling":
                next_z = cur_z - self._rat_pop_speed * dt
                reached = next_z <= self._rat_hidden_z[i]
                if reached:
                    next_z = self._rat_hidden_z[i]
                target_p = np.array(current_pose.p)
                target_p[2] = next_z
                rigid.set_kinematic_target(sapien.Pose(p=target_p, q=current_pose.q))
                if reached:
                    self._rat_auto_motion[i] = "rising"
                    self._rat_raised[i] = False
                return
            print_c(f"rat auto motion[{i}]: {motion}", "red")
            return

        # pin kinematic rats that aren't auto-moving
        if rigid.get_kinematic():
            rigid.set_kinematic_target(
                sapien.Pose(p=self._rat_pose_p(self._rat_holes[i], self._rat_raised[i]).tolist()))

    def _update_kinematic_tasks(self):
        # base hook drives DOMINO dynamic-object motion; runs every physics step.
        super()._update_kinematic_tasks()
        self._global_step = getattr(self, "_global_step", 0) + 1
        if not getattr(self, "_rat_rigids", None) or not getattr(self, "holes", None):
            return
        for i in range(len(self._rat_rigids)):
            self._step_one_rat(i)

    # ------------------------------------------------------------- dwell
    def _dwell(self, steps):
        """Advance sim `steps`, driving kinematics and recording frames periodically."""
        
        for i in range(int(steps)):
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (self._global_step % self.save_freq == 0):
                self._take_picture()

    def _release_rat(self, rat_idx=0):
        # turn the rat into a free dynamic body so the gripper can carry it off
        rigid = self._rat_rigids[rat_idx]
        if rigid is not None:
            try:
                print_c(
                    f"{self._rat_names[rat_idx]} released at {rigid.get_pose().p}",
                    color="red",
                )
                rigid.set_kinematic(False)
                self._rat_rigids[rat_idx] = None
                self._rat_auto_motion[rat_idx] = False
                if rat_idx == 0:
                    self._rat_rigid = None
            except Exception:
                pass

    def _arm_for_hole(self, hole_idx):
        return ArmTag("right" if self.holes[hole_idx][0] > 0 else "left")

    def _approach_rat(self, rat_idx, arm_tag):
        """Move the arm above the rat's hole, preserving EE orientation."""
        current = np.array(self.get_arm_pose(arm_tag), dtype=float)
        cur_pos = current[:3]
        rat_pos = np.array(self.rats[rat_idx].get_pose().p)
        target = np.array(
            [float(rat_pos[0]), float(rat_pos[1]), float(self.board_top_z + 0.16)])
        dx = float(target[0] - cur_pos[0])
        dy = float(target[1] - cur_pos[1])
        dz = float(target[2] - cur_pos[2])
        return self.move_by_displacement(
            arm_tag=arm_tag, x=dx, y=dy, z=dz, move_axis="world")

    def _wait_for_rising(self, rat_indices, max_iters=200):
        """Hold still until listed rats are rising and clear of the board top."""
        obj_poses = {i: [] for i in rat_indices}
        for _ in range(max_iters):
            hold = [
                self.move_by_displacement(arm_tag=self._arm_for_hole(self._rat_holes[i]),
                                          z=0.0, move_axis="arm")
                for i in rat_indices
            ]
            # one arm hold is enough to tick the sim when dual; move both if dual
            if len(hold) == 1:
                self.move(hold[0])
            else:
                self.move(hold[0], hold[1])
            ready = True
            for i in rat_indices:
                rat_top = float(self.rats[i].get_pose().p[2]) + self.rat_half[2]
                hist = obj_poses[i]
                hist.append(rat_top)
                if len(hist) > 2:
                    hist.pop(0)
                if len(hist) < 2:
                    ready = False
                    continue
                if not (hist[0] < hist[1] and rat_top >= self.board_top_z + 0.01):
                    ready = False
            if ready:
                return True
        return False

    def _try_catch(self, rat_idx, arm_tag):
        raised_p = self.rats[rat_idx].get_pose().p
        ee = np.array(self.get_arm_pose(arm_tag)[:3])
        offset = float(np.linalg.norm(ee[:2] - raised_p[:2]))
        gripper_closed = (self.is_left_gripper_close() if arm_tag == "left"
                          else self.is_right_gripper_close())
        caught = bool(gripper_closed and offset < self.grasp_tol * 2.5)
        if caught:
            self.catches += 1
            self._grab_offsets.append(min(offset, self.grasp_tol))
        return caught, offset

    # ------------------------------------------------------------- policy
    def play_once(self):
        if self.dual_catch:
            return self._play_dual()
        return self._play_single()

    def _play_single(self):
        arm_tag = ArmTag("right" if self.rat.get_pose().p[0] > 0 else "left")
        self.move(self._approach_rat(0, arm_tag))

        for _ in range(10):
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.0, move_axis="arm"))
        self._wait_for_rising([0])

        self.move(self.close_gripper(arm_tag=arm_tag))
        caught, _ = self._try_catch(0, arm_tag)
        if caught:
            self._release_rat(0)
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.12, move_axis="arm"))

        self.info["info"] = {
            "{A}": "rat_body",
            "{B}": "hole_board",
            "{a}": str(arm_tag),
            "{o}": self._option_label(),
        }
        return self.info

    def _option_label(self):
        return (
            f"catch_two_mice={str(self.catch_two_mice).lower()},"
            f"opaque_surface={str(self.opaque_surface).lower()}"
        )

    def _play_dual(self):
        """Both arms approach opposite-side rats and close simultaneously."""
        left = ArmTag("left")
        right = ArmTag("right")
        # rats were spawned left then right
        idx_left, idx_right = 0, 1

        self.move(
            self._approach_rat(idx_left, left),
            self._approach_rat(idx_right, right),
        )
        for _ in range(10):
            self.move(
                self.move_by_displacement(arm_tag=left, z=0.0, move_axis="arm"),
                self.move_by_displacement(arm_tag=right, z=0.0, move_axis="arm"),
            )
        self._wait_for_rising([idx_left, idx_right])

        # Drop a bit closer so both pinches seat before closing.
        self.move(
            self.move_by_displacement(arm_tag=left, z=-0.04, move_axis="world"),
            self.move_by_displacement(arm_tag=right, z=-0.04, move_axis="world"),
        )
        self.move(
            self.close_gripper(arm_tag=left),
            self.close_gripper(arm_tag=right),
        )
        # Let PhysX register mesh contacts before judging the grasps.
        self._dwell(20)

        caught_l, _ = self._try_catch(idx_left, left)
        caught_r, _ = self._try_catch(idx_right, right)
        held_l = self._rat_in_gripper(self._rat_names[idx_left])
        held_r = self._rat_in_gripper(self._rat_names[idx_right])
        # Dual success requires BOTH mice; only release/lift if both are held.
        if caught_l and caught_r and held_l and held_r:
            self._release_rat(idx_left)
            self._release_rat(idx_right)
            self.move(
                self.move_by_displacement(arm_tag=left, z=0.12, move_axis="arm"),
                self.move_by_displacement(arm_tag=right, z=0.12, move_axis="arm"),
            )

        self.info["info"] = {
            "{A}": "rat_body_left",
            "{C}": "rat_body_right",
            "{B}": "hole_board",
            "{a}": "both arms",
            "{o}": self._option_label(),
        }
        return self.info

    # --------------------------------------------------------- metric
    def _catch_score(self):
        if not self._grab_offsets:
            return 0.0
        vals = [float(np.clip(1.0 - o / self.grasp_tol, 0.0, 1.0)) for o in self._grab_offsets]
        return float(np.mean(vals))

    def _rat_in_gripper(self, name):
        try:
            return bool(self.get_gripper_actor_contact_position(name))
        except Exception:
            return False

    def _rats_held(self):
        """Per-rat gripper-contact flags (same order as self._rat_names)."""
        return [self._rat_in_gripper(n) for n in getattr(self, "_rat_names", [])]

    # ------------------------------------------------------------- success
    def check_success(self):
        held = self._rats_held()
        if self.catch_two_mice:
            # Option 1: both mice must be picked up; one catch is not enough.
            return bool(len(held) == 2 and all(held) and self.catches >= 2)
        return bool(held and held[0])

    # record per-frame whack-a-mole state into the trajectory
    def get_obs(self):
        obs = super().get_obs()
        held = self._rats_held()
        obs["catch_rat"] = {
            "catch_two_mice": bool(getattr(self, "catch_two_mice", False)),
            "opaque_surface": bool(getattr(self, "opaque_surface", False)),
            "rat_move_speed": float(getattr(self, "_rat_pop_speed", 0.0)),
            "active_hole": int(self._rat_holes[0]) if self._rat_holes else -1,
            "holes": [int(h) for h in getattr(self, "_rat_holes", [])],
            "rat_raised": [bool(r) for r in getattr(self, "_rat_raised", [])],
            "rats_held": [bool(h) for h in held],
            "catches": int(getattr(self, "catches", 0)),
            "appearances": int(getattr(self, "appearances_done", 0)),
        }
        return obs
