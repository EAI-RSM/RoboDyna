from ._base_task import Base_Task
from .utils import *
import sapien
import sapien.render
import numpy as np
from scene_utils import print_c, get_quat_euler

class catch_rat(Base_Task):
    """Whack-a-mole. A fixed board with a small grid of holes spans the mid zone of the
    table, reaching across BOTH arms (LEFT holes are owned by the LEFT arm, RIGHT holes by
    the RIGHT arm). A small graspable rat body pops up from one hole at a time for a short
    window, then retracts. The arm whose side owns the active hole must descend and close
    its gripper around the rat while it is RAISED (it is grippable only while up).

    The rat is modeled as a kinematic actor: while retracted it sits hidden below the board
    top, and during its pop window a step-driven schedule (in `_update_kinematic_tasks`)
    raises it to a grippable height above the hole, then drops it back down.

    Appearance sequence (hole order, pop durations, number of appearances) is randomized.

    Metric: per successful grab, the horizontal offset from the rat body center is measured;
    catch_score = mean over grabs of clamp(1 - offset / grasp_tol, 0, 1). Also reported:
    catches / appearances. Success = at least one rat caught on an appearance.
    """

    # ---- task params (class defaults; override via task_args.catch_rat in the config) ----
    NUM_APPEARANCES_DEFAULT = 3        # how many times a rat pops up over the episode
    POP_STEPS_DEFAULT = 90             # sim steps the rat stays raised per appearance (its window)
    PRE_POP_STEPS_DEFAULT = 12         # short settle before each pop while the arm pre-positions
    GRASP_TOL_DEFAULT = 0.01         # horizontal tolerance (m) used to normalize the catch offset

    RAT_HALF = [0.020, 0.026, 0.035]  # rat body half-extents (small graspable box)
    BOARD_HALF = [0.30, 0.13, 0.048]  # hole board half-extents (wide span, thin)
    POP_HEIGHT = 0.055                # how far above the board top the rat rises when popped
    HIDE_DEPTH = 0.075                # how far below the board top the rat hides when retracted
    RAT_MOVE_SPEED = 0.06     # meters per second for the rat's start-up motion


    def setup_demo(self, **kwags):
        # capture task-scoped params BEFORE init (kwags isn't stored on self otherwise)
        self._cfg = kwags.get("task_args", {}).get("catch_rat", {})
        super()._init_task_env_(**kwags)


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
        )
        self.board_top_z = self.board_center[2] + self.BOARD_HALF[2]

        # --- hole grid: calculate positions for N holes based on the chosen square size.
        hole_rows = int(np.floor(np.sqrt(self.hole_count)))
        hole_cols = int(np.ceil(self.hole_count / hole_rows))
        print_c(f"Hole grid: {hole_rows} x {hole_cols} = {hole_rows * hole_cols}", color="blue")
        
        x_half = self.BOARD_HALF[0]
        y_half = self.BOARD_HALF[1]
        gap_x = (2 * x_half - hole_cols * self.hole_size) / (hole_cols + 1)
        gap_y = (2 * y_half - hole_rows * self.hole_size) / (hole_rows + 1)
        if gap_x < self.hole_bar_thickness or gap_y < self.hole_bar_thickness:
            raise ValueError("Requested hole_size is too large for the board top")
        x_centers = np.linspace(
            -x_half + gap_x + self.hole_size / 2,
            x_half - gap_x - self.hole_size / 2,
            hole_cols,
        )
        y_centers = np.linspace(
            -y_half + gap_y + self.hole_size / 2,
            y_half - gap_y - self.hole_size / 2,
            hole_rows,
        )
        positions = [(self.board_center[0] + cx, self.board_center[1] + dy)
                     for cx in x_centers for dy in y_centers]
        self.holes = [np.array(pos) for pos in positions[:self.hole_count]]
        self.num_holes = len(self.holes)
    # ---------------------------------------------------------------- actors
    def load_actors(self):
        # init per-step / bookkeeping state FIRST (base setup may call _update_kinematic_tasks)
        self._global_step = 0
        self._active_hole = 0
        self._rat_raised = False
        self._rat_rigid = None
        self.catches = 0
        self.appearances_done = 0
        self._grab_offsets = []

        self.num_appearances = int(self._cfg.get("num_appearances", self.NUM_APPEARANCES_DEFAULT))
        self.pop_steps = int(self._cfg.get("pop_steps", self.POP_STEPS_DEFAULT))
        self.pre_pop_steps = int(self._cfg.get("pre_pop_steps", self.PRE_POP_STEPS_DEFAULT))
        self.grasp_tol = float(self._cfg.get("grasp_tol", self.GRASP_TOL_DEFAULT))


        self.table_top = 0.74 + self.table_z_bias

        self.create_board()

        # --- the rat: a small graspable box body, kinematic, starts hidden under a random hole ---
        self._active_hole = int(np.random.randint(0, self.num_holes))
        self._rat_raised = False

        # Get the size of the rat based on the hole size, ensuring it fits within the hole.
        rat_size = float(self.hole_size) - 0.004
        if rat_size <= 0:
            raise ValueError("hole_size must be larger than 4 mm to fit the rat")
        rat_half_xy = rat_size / 2.0
        rat_half = [rat_half_xy, rat_half_xy, self.RAT_HALF[2]]
        self.rat_half = rat_half


        self.rat = create_box(
            self.scene,
            sapien.Pose(p=self._rat_pose_p(self._active_hole, raised=False).tolist()),
            half_size=rat_half,
            color=[0.40, 0.40, 0.42],
            is_static=False,
            name="rat_body",
        )
        print_c(f"rat_body at {self.rat.get_pose().p}", color="blue")

        self.rat.set_mass(0.02)
        self._rat_rigid = None
        for c in self.rat.actor.get_components():
            if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                self._rat_rigid = c
        if self._rat_rigid is not None:
            self._rat_rigid.set_kinematic(True)
            self._rat_rigid.set_kinematic_target(
                sapien.Pose(p=self._rat_pose_p(self._active_hole, raised=False).tolist()))

            hidden_pose = self._rat_pose_p(self._active_hole, raised=False)
            raised_pose = hidden_pose.copy()
            # place the rat entirely above the board top so it does not overlap.
            # center z = board top + rat half-height + small gap
            raised_pose[2] = self.board_top_z + self.rat_half[2] + 1e-4
            rise_distance = raised_pose[2] - hidden_pose[2]
            # use per-step kinematic control at fixed speed
            self._rat_auto_motion = "rising"
            self._rat_pop_target_z = raised_pose[2]
            self._rat_hidden_z = hidden_pose[2]
            self._rat_pop_speed = float(self.RAT_MOVE_SPEED)

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

    def _set_rat(self, hole_idx, raised):
        self._active_hole = hole_idx
        self._rat_raised = raised
        if self._rat_rigid is not None:
            self._rat_rigid.set_kinematic_target(
                sapien.Pose(p=self._rat_pose_p(hole_idx, raised).tolist()))
    
    def _update_kinematic_tasks(self):
        # base hook drives DOMINO dynamic-object motion; runs every physics step.
        super()._update_kinematic_tasks()
        self._global_step = getattr(self, "_global_step", 0) + 1
        if getattr(self, "_rat_rigid", None) is None or not getattr(self, "holes", None):
            return
        if getattr(self, "_rat_auto_motion", False):
            # perform per-step fixed-speed motion rather than using active_kinematic_tasks
            dt = float(self.scene.get_timestep())
            current_pose = self._rat_rigid.entity.get_pose()
            cur_z = float(current_pose.p[2])
            if self._rat_auto_motion == "rising":
                next_z = cur_z + self._rat_pop_speed * dt
                reached = next_z >= self._rat_pop_target_z
                if reached:
                    next_z = self._rat_pop_target_z
                target_p = np.array(current_pose.p)
                target_p[2] = next_z
                self._rat_rigid.set_kinematic_target(sapien.Pose(p=target_p, q=current_pose.q))
                if reached:
                    self._rat_auto_motion = "falling"
                return
            elif self._rat_auto_motion == "falling":
                next_z = cur_z - self._rat_pop_speed * dt
                reached = next_z <= self._rat_hidden_z
                if reached:
                    next_z = self._rat_hidden_z
                target_p = np.array(current_pose.p)
                target_p[2] = next_z
                self._rat_rigid.set_kinematic_target(sapien.Pose(p=target_p, q=current_pose.q))
                if reached:
                    self._rat_auto_motion = "rising"
                return
            print_c(f"rat auto motion: {self._rat_auto_motion}", "red")
        # while the rat is being driven kinematically (not yet caught), keep pinning it to its
        # commanded slot so contact with the gripper doesn't shove it off. Do NOT override
        # the kinematic target while the automated rise/fall task is running (it would
        # teleport the rat to the final pose immediately).
        if not getattr(self, "_rat_auto_motion", False):
            if self._rat_rigid.get_kinematic():
                self._rat_rigid.set_kinematic_target(
                    sapien.Pose(p=self._rat_pose_p(self._active_hole, self._rat_raised).tolist()))

    # ------------------------------------------------------------- dwell
    def _dwell(self, steps):
        """Advance sim `steps`, driving kinematics and recording frames periodically."""
        
        for i in range(int(steps)):
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (self._global_step % self.save_freq == 0):
                self._take_picture()

    def _release_rat(self):
        # turn the rat into a free dynamic body so the gripper can carry it off
        if self._rat_rigid is not None:
            try:
                print_c(f"rat_body released at {self._rat_rigid.get_pose().p}", color="red")
                self._rat_rigid.set_kinematic(False)
                self._rat_rigid = None
                # self._rat_rigid.set_linear_velocity(np.zeros(3))
                # self._rat_rigid.set_angular_velocity(np.zeros(3))
            except Exception:
                pass

    # ------------------------------------------------------------- policy
    def play_once(self):

        # self._dwell(1)
        arm_tag = ArmTag("right" if self.rat.get_pose().p[0] > 0 else "left")

        # choose arm that owns the rat
        # current EE pose [x,y,z,qx,qy,qz,qw]
        current = np.array(self.get_arm_pose(arm_tag), dtype=float)
        cur_pos = current[:3]

        # rat center XY and desired Z (0.16 above board top)
        rat_pos = np.array(self.rat.get_pose().p)
        target = np.array([float(rat_pos[0]), float(rat_pos[1]), float(self.board_top_z + 0.16)])

        # world-frame displacement
        dx, dy, dz = float(target[0] - cur_pos[0]), float(target[1] - cur_pos[1]), \
                    float(target[2] - cur_pos[2])

        # move preserving EE orientation
        self.move(self.move_by_displacement(arm_tag=arm_tag, x=dx, y=dy, z=dz, move_axis="world"))

        for _ in range(10):
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.0, move_axis="arm"))
        obj_poses = []
        for _ in range(200):
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.0, move_axis="arm"))
            rat_top_pose = float(self.rat.get_pose().p[2]) + self.rat_half[2]
            obj_poses.append(rat_top_pose)
            if len(obj_poses) == 1:
                continue
            if len(obj_poses) > 2:
                obj_poses.pop(0)
            # object is at rising stage
            if obj_poses[0] < obj_poses[1] and rat_top_pose >= self.board_top_z + 0.01:
                break
        

        # self.move(self.grasp_actor(self.rat, arm_tag=arm_tag, pre_grasp_dis=0.08, gripper_pos=0.0))
        self.move(self.close_gripper(arm_tag=arm_tag))

        # measure the catch: gripper closed near the rat while it is still up?
        raised_p = self.rat.get_pose().p
        ee = np.array(self.get_arm_pose(arm_tag)[:3])
        offset = float(np.linalg.norm(ee[:2] - raised_p[:2]))
        gripper_closed = (self.is_left_gripper_close() if arm_tag == "left"
                          else self.is_right_gripper_close())
        caught = bool(gripper_closed and offset < self.grasp_tol * 2.5)

        if caught:
            self.catches += 1
            self._grab_offsets.append(min(offset, self.grasp_tol))
            caught_any = True
            # lift the caught rat clear of the board (it is now a free body in the gripper)
            self._release_rat()
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.12, move_axis="arm"))
            # only one rat body exists; once caught, stop spawning further appearances.
            # (no further kinematic pinning -- the rat is now free in the gripper.)
            # break
       

        self.info["info"] = {
            "{A}": "rat_body",
            "{B}": "hole_board",
            "{a}": arm_tag,
        }
        return self.info

    def play_once_bk(self):
        left = ArmTag("left")
        right = ArmTag("right")
        caught_any = False
        self._set_rat(0, raised=True)

        # for idx, app in enumerate(self.appearances):
        #     if not self.plan_success:
        #         break
        #     hole = app["hole"]
        #     duration = app["duration"]
        #     hole_p = self.holes[hole]
        #     arm = ArmTag("right" if hole_p[0] > 0 else "left")
        #     other = ArmTag("left") if arm == "right" else ArmTag("right")

        #     # keep rat retracted while the owning arm pre-positions above the hole
        #     self._set_rat(hole, raised=False)
        #     ready_pose = [
        #         float(hole_p[0]), float(hole_p[1]),
        #         self.board_top_z + 0.16,
        #         0, 1, 0, 0,
        #     ]
        #     # self.move(self.move_to_pose(arm, ready_pose))
        #     self._dwell(self.pre_pop_steps)
        #     if not self.plan_success:
        #         continue

        #     # --- rat pops up: grippable only while raised ---
        #     self._set_rat(hole, raised=True)
        #     self.appearances_done += 1
        #     # short dwell so the rendered pop is captured before the descent
        #     self._dwell(max(4, duration // 6))

        #     # owning arm descends and closes around the raised rat body
        #     # self.move(self.grasp_actor(self.rat, arm_tag=arm, pre_grasp_dis=0.08, gripper_pos=0.0))

        #     # # measure the catch: gripper closed near the rat while it is still up?
        #     # raised_p = self._rat_pose_p(hole, raised=True)
        #     # ee = np.array(self.get_arm_pose(arm)[:3])
        #     # offset = float(np.linalg.norm(ee[:2] - raised_p[:2]))
        #     # gripper_closed = (self.is_left_gripper_close() if arm == "left"
        #     #                   else self.is_right_gripper_close())
        #     # caught = bool(self.plan_success and gripper_closed and offset < self.grasp_tol * 2.5)
        #     caught = False
        #     if caught:
        #         self.catches += 1
        #         self._grab_offsets.append(min(offset, self.grasp_tol))
        #         caught_any = True
        #         # lift the caught rat clear of the board (it is now a free body in the gripper)
        #         self._release_rat()
        #         self.move(self.move_by_displacement(arm_tag=arm, z=0.12, move_axis="arm"))
        #         # only one rat body exists; once caught, stop spawning further appearances.
        #         # (no further kinematic pinning -- the rat is now free in the gripper.)
        #         break
        #     else:
        #         # missed: rat retracts, arm resets, wait for the next appearance
        #         self._dwell(max(4, duration // 3))
        #         self._set_rat(hole, raised=False)
        #         # self.move(self.back_to_origin(arm))

        self.info["info"] = {
            "{A}": "rat_body",
            "{B}": "hole_board",
            "{a}": str(ArmTag("right" if self.holes[self.appearances[0]["hole"]][0] > 0 else "left")),
        }
        return self.info

    # --------------------------------------------------------- metric
    def _catch_score(self):
        if not self._grab_offsets:
            return 0.0
        vals = [float(np.clip(1.0 - o / self.grasp_tol, 0.0, 1.0)) for o in self._grab_offsets]
        return float(np.mean(vals))

    # ------------------------------------------------------------- success
    def check_success(self):
        rat_in_gripper = False
        try:
            rat_in_gripper = bool(self.get_gripper_actor_contact_position("rat_body"))
        except Exception:
            rat_in_gripper = False
        return rat_in_gripper 

    # record per-frame whack-a-mole state into the trajectory
    def get_obs(self):
        obs = super().get_obs()
        obs["catch_rat"] = {
            "active_hole": int(getattr(self, "_active_hole", -1)),
            "rat_raised": bool(getattr(self, "_rat_raised", False)),
            "catches": int(getattr(self, "catches", 0)),
            "appearances": int(getattr(self, "appearances_done", 0)),
        }
        return obs
