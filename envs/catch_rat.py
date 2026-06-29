from ._base_task import Base_Task
from .utils import *
import sapien
import sapien.render
import numpy as np


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
    GRASP_TOL_DEFAULT = 0.05          # horizontal tolerance (m) used to normalize the catch offset

    RAT_HALF = [0.020, 0.026, 0.030]  # rat body half-extents (small graspable box)
    BOARD_HALF = [0.30, 0.13, 0.018]  # hole board half-extents (wide span, thin)
    POP_HEIGHT = 0.055                # how far above the board top the rat rises when popped
    HIDE_DEPTH = 0.075                # how far below the board top the rat hides when retracted

    def setup_demo(self, **kwags):
        # capture task-scoped params BEFORE init (kwags isn't stored on self otherwise)
        self._cfg = kwags.get("task_args", {}).get("catch_rat", {})
        super()._init_task_env_(**kwags)

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

        # --- hole board: fixed, static, spanning the mid zone across both arms' reach ---
        board_cy = float(np.random.uniform(-0.05, 0.0))
        self.board_center = np.array([0.0, board_cy, self.table_top + self.BOARD_HALF[2]])
        self.board = create_box(
            self.scene,
            sapien.Pose(p=self.board_center.tolist()),
            half_size=self.BOARD_HALF,
            color=[0.45, 0.32, 0.18],
            is_static=True,
            name="hole_board",
        )
        self.board_top_z = self.board_center[2] + self.BOARD_HALF[2]

        # --- hole grid: 2 rows (near/far) x 3 cols (left / center / right). Every hole is
        #     inside the combined two-arm reach; col x-signs decide the owning arm. The center
        #     column is biased to a clear side so it never lands ambiguously on the centerline.
        col_x = [-0.20, float(np.random.choice([-0.07, 0.07])), 0.20]
        row_dy = [-0.05, 0.05]
        self.holes = []   # list of (x, y) in world coords
        for cx in col_x:
            for dy in row_dy:
                self.holes.append((self.board_center[0] + cx, self.board_center[1] + dy))
        self.holes = [np.array(h) for h in self.holes]
        self.num_holes = len(self.holes)

        # --- the rat: a small graspable box body, kinematic, starts hidden in hole 0 ---
        self._active_hole = 0
        self._rat_raised = False
        self.rat = create_box(
            self.scene,
            sapien.Pose(p=self._rat_pose_p(0, raised=False).tolist()),
            half_size=self.RAT_HALF,
            color=[0.40, 0.40, 0.42],
            is_static=False,
            name="rat_body",
        )
        self.rat.set_mass(0.02)
        self._rat_rigid = None
        for c in self.rat.actor.get_components():
            if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                self._rat_rigid = c
        if self._rat_rigid is not None:
            self._rat_rigid.set_kinematic(True)
            self._rat_rigid.set_kinematic_target(
                sapien.Pose(p=self._rat_pose_p(0, raised=False).tolist()))

        # keep clutter off the board
        self.prohibited_area.append([
            self.board_center[0] - self.BOARD_HALF[0] - 0.03,
            self.board_center[1] - self.BOARD_HALF[1] - 0.03,
            self.board_center[0] + self.BOARD_HALF[0] + 0.03,
            self.board_center[1] + self.BOARD_HALF[1] + 0.03,
        ])

        # --- build the randomized appearance schedule (step-driven, identical in both passes) ---
        rng = np.random
        n = self.num_appearances
        order = [int(rng.randint(0, self.num_holes)) for _ in range(n)]
        durations = [int(round(self.pop_steps * float(rng.uniform(0.8, 1.25)))) for _ in range(n)]
        self.appearances = [{"hole": order[i], "duration": durations[i]} for i in range(n)]

        # per-grab bookkeeping for the metric
        self.catches = 0
        self.appearances_done = 0
        self._grab_offsets = []
        self._global_step = 0

    # ---------------------------------------------------------- rat kinematics
    def _rat_pose_p(self, hole_idx, raised):
        h = self.holes[hole_idx]
        if raised:
            z = self.board_top_z + self.POP_HEIGHT
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
        # while the rat is being driven kinematically (not yet caught), keep pinning it to its
        # commanded slot so contact with the gripper doesn't shove it off.
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
                self._rat_rigid.set_kinematic(False)
                self._rat_rigid.set_linear_velocity(np.zeros(3))
                self._rat_rigid.set_angular_velocity(np.zeros(3))
            except Exception:
                pass

    # ------------------------------------------------------------- policy
    def play_once(self):
        left = ArmTag("left")
        right = ArmTag("right")
        caught_any = False

        for idx, app in enumerate(self.appearances):
            if not self.plan_success:
                break
            hole = app["hole"]
            duration = app["duration"]
            hole_p = self.holes[hole]
            arm = ArmTag("right" if hole_p[0] > 0 else "left")
            other = ArmTag("left") if arm == "right" else ArmTag("right")

            # keep rat retracted while the owning arm pre-positions above the hole
            self._set_rat(hole, raised=False)
            ready_pose = [
                float(hole_p[0]), float(hole_p[1]),
                self.board_top_z + 0.16,
                0, 1, 0, 0,
            ]
            self.move(self.move_to_pose(arm, ready_pose))
            self._dwell(self.pre_pop_steps)
            if not self.plan_success:
                continue

            # --- rat pops up: grippable only while raised ---
            self._set_rat(hole, raised=True)
            self.appearances_done += 1
            # short dwell so the rendered pop is captured before the descent
            self._dwell(max(4, duration // 6))

            # owning arm descends and closes around the raised rat body
            self.move(self.grasp_actor(self.rat, arm_tag=arm, pre_grasp_dis=0.08, gripper_pos=0.0))

            # measure the catch: gripper closed near the rat while it is still up?
            raised_p = self._rat_pose_p(hole, raised=True)
            ee = np.array(self.get_arm_pose(arm)[:3])
            offset = float(np.linalg.norm(ee[:2] - raised_p[:2]))
            gripper_closed = (self.is_left_gripper_close() if arm == "left"
                              else self.is_right_gripper_close())
            caught = bool(self.plan_success and gripper_closed and offset < self.grasp_tol * 2.5)

            if caught:
                self.catches += 1
                self._grab_offsets.append(min(offset, self.grasp_tol))
                caught_any = True
                # lift the caught rat clear of the board (it is now a free body in the gripper)
                self._release_rat()
                self.move(self.move_by_displacement(arm_tag=arm, z=0.12, move_axis="arm"))
                # only one rat body exists; once caught, stop spawning further appearances.
                # (no further kinematic pinning -- the rat is now free in the gripper.)
                break
            else:
                # missed: rat retracts, arm resets, wait for the next appearance
                self._dwell(max(4, duration // 3))
                self._set_rat(hole, raised=False)
                self.move(self.back_to_origin(arm))

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
        success = self.catches >= 1
        self.info["catch_score"] = self._catch_score()
        self.info["catches"] = int(self.catches)
        self.info["appearances"] = int(self.appearances_done)
        self.info["catch_rate"] = (float(self.catches) / self.appearances_done
                                   if self.appearances_done > 0 else 0.0)
        return bool(success)

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
