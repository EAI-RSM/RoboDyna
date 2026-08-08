from ._base_task import Base_Task
from .utils import *
import sapien
import sapien.render
import numpy as np
from scene_utils import print_c, get_quat_euler

class catch_cuboid(Base_Task):
    """Catch a cuboid that pops up from holes on a mid-table board spanning both arms.

    Config options under task_args.catch_cuboid (independent booleans):

      catch_two_cuboids (option 1, default false)
          Two cuboids pop from opposite-side holes at once; both arms must catch
          them. Success requires BOTH cuboids held. When false: single cuboid.

      opaque_surface (option 2, default false)
          Opaque colored top lattice (current solid board color). When false:
          glass / see-through tinted top lattice.

    The cuboid(s) are kinematic actors: while retracted they sit hidden below the
    board top, and during the pop window a step-driven schedule raises them to a
    grippable height, then drops them back down.
    """

    # ---- task params (class defaults; override via task_args.catch_cuboid in the config) ----
    CATCH_TWO_CUBOIDS_DEFAULT = False     # option 1
    OPAQUE_SURFACE_DEFAULT = False     # option 2 (false => glass top)
    NUM_APPEARANCES_DEFAULT = 5        # how many times a cuboid pops up over the episode
    POP_STEPS_DEFAULT = 90             # sim steps the cuboid stays raised per appearance (its window)
    PRE_POP_STEPS_DEFAULT = 12         # short settle before each pop while the arm pre-positions
    GRASP_TOL_DEFAULT = 0.01         # horizontal tolerance (m) used to normalize the catch offset

    CUBOID_HALF = [0.020, 0.026, 0.035]  # cuboid body half-extents (small graspable box)
    BOARD_HALF = [0.30, 0.13, 0.060]  # hole board half-extents (open bottom, closed sides)
    BOARD_PANEL_THICKNESS = 0.02      # wooden top lattice thickness
    POP_HEIGHT = 0.055                # how far above the board top the cuboid rises when popped
    HIDE_DEPTH = 0.070                # how far below the board top the cuboid hides when retracted
    # Per-episode rise/fall speed (m/s) is sampled uniformly from [min, max].
    # Mean 0.056 (= 0.07 × 0.8); range scaled −20% from prior [0.04, 0.10].
    CUBOID_MOVE_SPEED_MIN_DEFAULT = 0.032
    CUBOID_MOVE_SPEED_MAX_DEFAULT = 0.08
    RANDOMIZE_CUBOID_COLOR_DEFAULT = False
    CUBOID_COLOR_DEFAULT = [0.40, 0.40, 0.42]
    CUBOID_COLOR_POOL = (
        [0.85, 0.20, 0.18],  # red
        [0.95, 0.78, 0.15],  # yellow
        [0.55, 0.25, 0.80],  # purple
        [0.20, 0.70, 0.30],  # green
        [0.20, 0.45, 0.90],  # blue
        [0.95, 0.55, 0.15],  # orange
        [0.40, 0.40, 0.42],  # gray (legacy default)
    )

    def setup_demo(self, **kwags):
        # capture task-scoped params BEFORE init (kwags isn't stored on self otherwise)
        task_args = kwags.get("task_args", {}) or {}
        self._cfg = task_args.get("catch_cuboid", task_args.get("catch_rat", {}))
        self._parse_option()
        # init bookkeeping before base init (may call _update_kinematic_tasks)
        self._global_step = 0
        self.cuboids = []
        self._cuboid_rigids = []
        self._cuboid_holes = []
        self._cuboid_names = []
        self._cuboid_raised = []
        self._cuboid_auto_motion = []
        self._cuboid_pop_target_z = []
        self._cuboid_hidden_z = []
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
        raise ValueError(f"catch_cuboid expected a boolean, got {value!r}")

    def _parse_option(self):
        """Read option-1 / option-2 config flags (independent).

        Preferred keys:
          catch_two_cuboids   — option 1
          opaque_surface   — option 2

        Legacy ``option`` / record_demo ``--option`` convenience:
          1 / catch_two_cuboids / catch_two_mice / dual_catch  -> enable option 1
          2 / opaque_surface / opaque      -> enable option 2
        """
        cfg = self._cfg
        # Prefer catch_two_cuboids; accept legacy catch_two_mice.
        catch_two = cfg.get("catch_two_cuboids", cfg.get("catch_two_mice", None))
        opaque = cfg.get("opaque_surface", None)

        # Legacy single ``option`` value from older configs / --option CLI.
        legacy = cfg.get("option", None)
        if legacy is not None:
            aliases = {
                1: "catch_two_cuboids",
                2: "opaque_surface",
                "1": "catch_two_cuboids",
                "2": "opaque_surface",
                "catch_two_cuboids": "catch_two_cuboids",
                "catch_two_mice": "catch_two_cuboids",
                "dual_catch": "catch_two_cuboids",
                "opaque_surface": "opaque_surface",
                "opaque": "opaque_surface",
                # old names (mapped to the new flags they roughly meant)
                "transparent_grid": None,  # both flags false
            }
            if legacy not in aliases:
                raise ValueError(
                    "catch_cuboid option must be 1/catch_two_cuboids or "
                    "2/opaque_surface (or set catch_two_cuboids / opaque_surface booleans)")
            key = aliases[legacy]
            if key == "catch_two_cuboids" and catch_two is None:
                catch_two = True
            elif key == "opaque_surface" and opaque is None:
                opaque = True

        self.catch_two_cuboids = self._as_bool(catch_two, self.CATCH_TWO_CUBOIDS_DEFAULT)
        self.opaque_surface = self._as_bool(opaque, self.OPAQUE_SURFACE_DEFAULT)
        # aliases used elsewhere in the task
        self.dual_catch = self.catch_two_cuboids
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
            top_thickness=self.BOARD_PANEL_THICKNESS,
            bottom_thickness=0.0,
            bar_thickness=self.hole_bar_thickness,
            top_transparent=self.transparent_grid,
            panel_overhang=0.0,
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
            raise ValueError("catch_cuboid catch_two_cuboids needs holes on both table sides")
        pairs = [
            (a, b) for a in left for b in right
            if not self._holes_too_close(a, b)
        ]
        if not pairs:
            raise ValueError(
                "catch_cuboid catch_two_cuboids: no left/right hole pair with a "
                "one-cell gap (arms would collide)")
        preferred = [
            (a, b) for a, b in pairs
            if abs(self.hole_rc[a][0] - self.hole_rc[b][0]) <= 1
        ]
        pool = preferred if preferred else pairs
        a, b = pool[int(np.random.randint(0, len(pool)))]
        return int(a), int(b)

    def _spawn_cuboid(self, hole_idx, name):
        """Create one kinematic cuboid body under the given hole."""
        cuboid = create_box(
            self.scene,
            sapien.Pose(p=self._cuboid_pose_p(hole_idx, raised=False).tolist()),
            half_size=self.cuboid_half,
            color=list(self.cuboid_color),
            is_static=False,
            name=name,
        )
        cuboid.set_mass(0.02)
        rigid = None
        for c in cuboid.actor.get_components():
            if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                rigid = c
        if rigid is not None:
            rigid.set_kinematic(True)
            hidden_pose = self._cuboid_pose_p(hole_idx, raised=False)
            raised_z = self.board_top_z + self.cuboid_half[2] + 1e-4
            rigid.set_kinematic_target(sapien.Pose(p=hidden_pose.tolist()))
            self._cuboid_auto_motion.append("rising")
            self._cuboid_pop_target_z.append(float(raised_z))
            self._cuboid_hidden_z.append(float(hidden_pose[2]))
        else:
            self._cuboid_auto_motion.append(False)
            self._cuboid_pop_target_z.append(0.0)
            self._cuboid_hidden_z.append(0.0)
        self.cuboids.append(cuboid)
        self._cuboid_rigids.append(rigid)
        self._cuboid_holes.append(int(hole_idx))
        self._cuboid_names.append(name)
        self._cuboid_raised.append(False)
        print_c(f"{name} at {cuboid.get_pose().p} (hole {hole_idx})", color="blue")
        return cuboid

    # ---------------------------------------------------------------- actors
    def load_actors(self):
        # init per-step / bookkeeping state FIRST (base setup may call _update_kinematic_tasks)
        self._global_step = 0
        self.cuboids = []
        self._cuboid_rigids = []
        self._cuboid_holes = []
        self._cuboid_names = []
        self._cuboid_raised = []
        self._cuboid_auto_motion = []
        self._cuboid_pop_target_z = []
        self._cuboid_hidden_z = []
        self.catches = 0
        self.appearances_done = 0
        self._grab_offsets = []

        self.num_appearances = int(self._cfg.get("num_appearances", self.NUM_APPEARANCES_DEFAULT))
        self.pop_steps = int(self._cfg.get("pop_steps", self.POP_STEPS_DEFAULT))
        self.pre_pop_steps = int(self._cfg.get("pre_pop_steps", self.PRE_POP_STEPS_DEFAULT))
        self.grasp_tol = float(self._cfg.get("grasp_tol", self.GRASP_TOL_DEFAULT))
        self.cuboid_move_speed_min = float(
            self._cfg.get("cuboid_move_speed_min", self.CUBOID_MOVE_SPEED_MIN_DEFAULT))
        self.cuboid_move_speed_max = float(
            self._cfg.get("cuboid_move_speed_max", self.CUBOID_MOVE_SPEED_MAX_DEFAULT))
        if self.cuboid_move_speed_min <= 0 or self.cuboid_move_speed_max <= 0:
            raise ValueError("catch_cuboid cuboid_move_speed_min/max must be > 0")
        if self.cuboid_move_speed_min > self.cuboid_move_speed_max:
            raise ValueError("catch_cuboid cuboid_move_speed_min must be <= cuboid_move_speed_max")

        self.randomize_cuboid_color = self._as_bool(
            self._cfg.get("randomize_cuboid_color", self.RANDOMIZE_CUBOID_COLOR_DEFAULT),
            self.RANDOMIZE_CUBOID_COLOR_DEFAULT,
        )
        pool = self._cfg.get("cuboid_colors", self.CUBOID_COLOR_POOL)
        pool = [list(c)[:3] for c in pool]
        if self.randomize_cuboid_color and pool:
            self.cuboid_color = list(pool[int(np.random.randint(0, len(pool)))])
        else:
            fixed = self._cfg.get("cuboid_color", self.CUBOID_COLOR_DEFAULT)
            self.cuboid_color = list(fixed)[:3]

        self.table_top = 0.74 + self.table_z_bias
        self.create_board()

        # Get the size of the cuboid based on the hole size, ensuring it fits within the hole.
        cuboid_size = float(self.hole_size) - 0.004
        if cuboid_size <= 0:
            raise ValueError("hole_size must be larger than 4 mm to fit the cuboid")
        cuboid_half_xy = cuboid_size / 2.0
        self.cuboid_half = [cuboid_half_xy, cuboid_half_xy, self.CUBOID_HALF[2]]
        # One shared speed per episode so dual cuboids stay synchronized.
        self._cuboid_pop_speed = float(np.random.uniform(
            self.cuboid_move_speed_min, self.cuboid_move_speed_max))
        print_c(
            f"cuboid move speed: {self._cuboid_pop_speed:.4f} m/s "
            f"(range [{self.cuboid_move_speed_min:.4f}, {self.cuboid_move_speed_max:.4f}]) "
            f"color={self.cuboid_color}",
            color="blue",
        )

        if self.dual_catch:
            h_left, h_right = self._pick_dual_holes()
            self._spawn_cuboid(h_left, "cuboid_body_left")
            self._spawn_cuboid(h_right, "cuboid_body_right")
        else:
            hole = int(np.random.randint(0, self.num_holes))
            self._spawn_cuboid(hole, "cuboid_body")

        # convenience aliases used by the single-cuboid expert / obs
        self.cuboid = self.cuboids[0]
        self._cuboid_rigid = self._cuboid_rigids[0]
        self._active_hole = self._cuboid_holes[0]

        # keep clutter off the board
        self.prohibited_area.append([
            self.board_center[0] - self.BOARD_HALF[0] - 0.03,
            self.board_center[1] - self.BOARD_HALF[1] - 0.03,
            self.board_center[0] + self.BOARD_HALF[0] + 0.03,
            self.board_center[1] + self.BOARD_HALF[1] + 0.03,
        ])
        self.faling_once = False

    # ---------------------------------------------------------- cuboid kinematics
    def _cuboid_pose_p(self, hole_idx, raised):
        h = self.holes[hole_idx]
        if raised:
            # center the cuboid so its bottom is just above the board top (no overlap)
            z = self.board_top_z + self.cuboid_half[2] + 1e-4
        else:
            z = self.board_top_z - self.HIDE_DEPTH
        return np.array([h[0], h[1], z])

    def _set_cuboid(self, hole_idx, raised, cuboid_idx=0):
        self._cuboid_holes[cuboid_idx] = hole_idx
        self._cuboid_raised[cuboid_idx] = raised
        if cuboid_idx == 0:
            self._active_hole = hole_idx
        rigid = self._cuboid_rigids[cuboid_idx]
        if rigid is not None:
            rigid.set_kinematic_target(
                sapien.Pose(p=self._cuboid_pose_p(hole_idx, raised).tolist()))

    def _step_one_cuboid(self, i):
        rigid = self._cuboid_rigids[i]
        if rigid is None:
            return
        motion = self._cuboid_auto_motion[i]
        if motion:
            dt = float(self.scene.get_timestep())
            current_pose = rigid.entity.get_pose()
            cur_z = float(current_pose.p[2])
            if motion == "rising":
                next_z = cur_z + self._cuboid_pop_speed * dt
                reached = next_z >= self._cuboid_pop_target_z[i]
                if reached:
                    next_z = self._cuboid_pop_target_z[i]
                target_p = np.array(current_pose.p)
                target_p[2] = next_z
                rigid.set_kinematic_target(sapien.Pose(p=target_p, q=current_pose.q))
                if reached:
                    self._cuboid_auto_motion[i] = "falling"
                    self._cuboid_raised[i] = True
                return
            if motion == "falling":
                next_z = cur_z - self._cuboid_pop_speed * dt
                reached = next_z <= self._cuboid_hidden_z[i]
                if reached:
                    next_z = self._cuboid_hidden_z[i]
                target_p = np.array(current_pose.p)
                target_p[2] = next_z
                rigid.set_kinematic_target(sapien.Pose(p=target_p, q=current_pose.q))
                if reached:
                    self._cuboid_auto_motion[i] = "rising"
                    self._cuboid_raised[i] = False
                return
            print_c(f"cuboid auto motion[{i}]: {motion}", "red")
            return

        # pin kinematic cuboids that aren't auto-moving
        if rigid.get_kinematic():
            rigid.set_kinematic_target(
                sapien.Pose(p=self._cuboid_pose_p(self._cuboid_holes[i], self._cuboid_raised[i]).tolist()))

    def _update_kinematic_tasks(self):
        # base hook drives DOMINO dynamic-object motion; runs every physics step.
        super()._update_kinematic_tasks()
        self._global_step = getattr(self, "_global_step", 0) + 1
        if not getattr(self, "_cuboid_rigids", None) or not getattr(self, "holes", None):
            return
        for i in range(len(self._cuboid_rigids)):
            self._step_one_cuboid(i)

    # ------------------------------------------------------------- dwell
    def _dwell(self, steps):
        """Advance sim `steps`, driving kinematics and recording frames periodically."""
        
        for i in range(int(steps)):
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (self._global_step % self.save_freq == 0):
                self._take_picture()

    def _release_cuboid(self, cuboid_idx=0):
        # turn the cuboid into a free dynamic body so the gripper can carry it off
        rigid = self._cuboid_rigids[cuboid_idx]
        if rigid is not None:
            try:
                print_c(
                    f"{self._cuboid_names[cuboid_idx]} released at {rigid.get_pose().p}",
                    color="red",
                )
                rigid.set_kinematic(False)
                self._cuboid_rigids[cuboid_idx] = None
                self._cuboid_auto_motion[cuboid_idx] = False
                if cuboid_idx == 0:
                    self._cuboid_rigid = None
            except Exception:
                pass

    def _arm_for_hole(self, hole_idx):
        return ArmTag("right" if self.holes[hole_idx][0] > 0 else "left")

    def _approach_cuboid(self, cuboid_idx, arm_tag):
        """Move the arm above the cuboid's hole, preserving EE orientation."""
        current = np.array(self.get_arm_pose(arm_tag), dtype=float)
        cur_pos = current[:3]
        cuboid_pos = np.array(self.cuboids[cuboid_idx].get_pose().p)
        target = np.array(
            [float(cuboid_pos[0]), float(cuboid_pos[1]), float(self.board_top_z + 0.16)])
        dx = float(target[0] - cur_pos[0])
        dy = float(target[1] - cur_pos[1])
        dz = float(target[2] - cur_pos[2])
        return self.move_by_displacement(
            arm_tag=arm_tag, x=dx, y=dy, z=dz, move_axis="world")

    def _wait_for_rising(self, cuboid_indices, max_iters=200):
        """Hold still until listed cuboids are rising and clear of the board top."""
        obj_poses = {i: [] for i in cuboid_indices}
        for _ in range(max_iters):
            hold = [
                self.move_by_displacement(arm_tag=self._arm_for_hole(self._cuboid_holes[i]),
                                          z=0.0, move_axis="arm")
                for i in cuboid_indices
            ]
            # one arm hold is enough to tick the sim when dual; move both if dual
            if len(hold) == 1:
                self.move(hold[0])
            else:
                self.move(hold[0], hold[1])
            ready = True
            for i in cuboid_indices:
                cuboid_top = float(self.cuboids[i].get_pose().p[2]) + self.cuboid_half[2]
                hist = obj_poses[i]
                hist.append(cuboid_top)
                if len(hist) > 2:
                    hist.pop(0)
                if len(hist) < 2:
                    ready = False
                    continue
                if not (hist[0] < hist[1] and cuboid_top >= self.board_top_z + 0.01):
                    ready = False
            if ready:
                return True
        return False

    def _grasp_offset(self, cuboid_idx, arm_tag):
        raised_p = self.cuboids[cuboid_idx].get_pose().p
        ee = np.array(self.get_arm_pose(arm_tag)[:3])
        return float(np.linalg.norm(ee[:2] - raised_p[:2]))

    def _gripper_closed(self, arm_tag):
        return bool(
            self.is_left_gripper_close() if arm_tag == "left"
            else self.is_right_gripper_close()
        )

    def _try_catch(self, cuboid_idx, arm_tag):
        """True only for a real pinch: closed jaws, near the cuboid, and contacting it."""
        offset = self._grasp_offset(cuboid_idx, arm_tag)
        in_contact = bool(
            self.get_gripper_actor_contact_position(self._cuboid_names[cuboid_idx])
        )
        caught = bool(
            self._gripper_closed(arm_tag)
            and in_contact
            and offset < self.grasp_tol * 2.5
        )
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
        arm_tag = ArmTag("right" if self.cuboid.get_pose().p[0] > 0 else "left")
        self.move(self._approach_cuboid(0, arm_tag))

        for _ in range(10):
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.0, move_axis="arm"))
        self._wait_for_rising([0])

        self.move(self.close_gripper(arm_tag=arm_tag))
        caught, _ = self._try_catch(0, arm_tag)
        if caught:
            self._release_cuboid(0)
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.12, move_axis="arm"))

        self.info["info"] = {
            "{A}": "cuboid_body",
            "{B}": "hole_board",
            "{a}": str(arm_tag),
            "{o}": self._option_label(),
        }
        return self.info

    def _option_label(self):
        return (
            f"catch_two_cuboids={str(self.catch_two_cuboids).lower()},"
            f"opaque_surface={str(self.opaque_surface).lower()}"
        )

    def _play_dual(self):
        """Both arms approach opposite-side cuboids and close simultaneously."""
        left = ArmTag("left")
        right = ArmTag("right")
        # cuboids were spawned left then right
        idx_left, idx_right = 0, 1

        self.move(
            self._approach_cuboid(idx_left, left),
            self._approach_cuboid(idx_right, right),
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
        held_l = self._cuboid_in_gripper(self._cuboid_names[idx_left])
        held_r = self._cuboid_in_gripper(self._cuboid_names[idx_right])
        # Dual success requires BOTH cuboids; only release/lift if both are held.
        if caught_l and caught_r and held_l and held_r:
            self._release_cuboid(idx_left)
            self._release_cuboid(idx_right)
            self.move(
                self.move_by_displacement(arm_tag=left, z=0.12, move_axis="arm"),
                self.move_by_displacement(arm_tag=right, z=0.12, move_axis="arm"),
            )

        self.info["info"] = {
            "{A}": "cuboid_body_left",
            "{C}": "cuboid_body_right",
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

    def _cuboid_in_gripper(self, name, arm_tag=None):
        """True when the cuboid is pinched (closed gripper + contact + proximity).

        A bare finger touch with an open gripper is not enough.
        """
        try:
            cuboid_idx = self._cuboid_names.index(name)
        except (ValueError, AttributeError):
            return False
        if arm_tag is None:
            if cuboid_idx >= len(getattr(self, "_cuboid_holes", [])):
                return False
            arm_tag = self._arm_for_hole(self._cuboid_holes[cuboid_idx])
        try:
            if not self.get_gripper_actor_contact_position(name):
                return False
        except Exception:
            return False
        if not self._gripper_closed(arm_tag):
            return False
        return self._grasp_offset(cuboid_idx, arm_tag) < self.grasp_tol * 2.5

    def _cuboids_held(self):
        """Per-cuboid grasp flags (same order as self._cuboid_names)."""
        return [self._cuboid_in_gripper(n) for n in getattr(self, "_cuboid_names", [])]

    # ------------------------------------------------------------- success
    def check_success(self):
        held = self._cuboids_held()
        if self.catch_two_cuboids:
            # Option 1: both cuboids must be picked up; one catch is not enough.
            return bool(len(held) == 2 and all(held) and self.catches >= 2)
        return bool(held and held[0])

    # record per-frame whack-a-mole state into the trajectory
    def get_obs(self):
        obs = super().get_obs()
        held = self._cuboids_held()
        obs["catch_cuboid"] = {
            "catch_two_cuboids": bool(getattr(self, "catch_two_cuboids", False)),
            "opaque_surface": bool(getattr(self, "opaque_surface", False)),
            "cuboid_move_speed": float(getattr(self, "_cuboid_pop_speed", 0.0)),
            "active_hole": int(self._cuboid_holes[0]) if self._cuboid_holes else -1,
            "holes": [int(h) for h in getattr(self, "_cuboid_holes", [])],
            "cuboid_raised": [bool(r) for r in getattr(self, "_cuboid_raised", [])],
            "cuboids_held": [bool(h) for h in held],
            "catches": int(getattr(self, "catches", 0)),
            "appearances": int(getattr(self, "appearances_done", 0)),
        }
        return obs
