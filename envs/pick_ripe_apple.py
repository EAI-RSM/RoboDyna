from ._base_task import Base_Task
from .utils import *
import os
import sapien
import sapien.render
import numpy as np


class pick_ripe_apple(Base_Task):
    """Dual-arm ripeness-timing task.

    One apple ripens on a board on EACH side of the table (left and right), each at its OWN random
    rate, so the two red (ripe) windows occur at DIFFERENT times. Each arm independently watches its
    side's apple and -- following the observe-then-act pattern -- grasps it when it turns red, then
    drops it into the bowl on that side.

    An apple ripens green -> red -> black ONLY while it rests on its board; the instant it LEAVES the
    board (is lifted above the surface) its colour/ripeness FREEZES (detected physically by the
    apple's height, so the rule holds for a learned policy too). Each apple is grasped top-down with a
    normal friction grasp and dropped into the same-side bowl.

    Metric: per-apple ripeness_score = clamp(1 - |r_grasp - 0.5| / 0.5, 0, 1), where r_grasp is the
    ripeness latched when the apple left its board; final score = mean over apples, and an episode
    succeeds only if BOTH apples come to rest in their bowls.
    """

    # ---- task params (class defaults; override via task_args.pick_ripe_apple in the config) ----
    RIPEN_STEPS_DEFAULT = 2500       # base sim steps for r: 0 -> 1 (per-apple rate scales this). SLOW
                                     # enough that the apple barely ripens during the ~300-step grasp+
                                     # lift, so it leaves the board still ~red (not over-ripened black).
    RED_WINDOW_MIN_DEFAULT = 0.48    # per-episode red target is randomized in this band (mirrors
    RED_WINDOW_MAX_DEFAULT = 0.52    # cook_meat's target_doneness 0.45-0.55); 0.48-0.52 are all vivid red
    RED_TOLERANCE_DEFAULT = 0.12     # how close to the target we try to fire the grasp
    GRASP_LEAD_STEPS = 500           # est. steps from the grasp trigger until the apple leaves the
                                     # board; trigger the grasp this much EARLY so it crosses red right
                                     # as it lifts off -> r_grasp lands near 0.5 (leads the ripening)
    LEAVE_BOARD_DZ = 0.035           # apple-centre height above its board surface beyond which
                                     # ripening FREEZES (the apple has left the board)
    # 3-stop ripeness gradient: green -> vivid red -> near-black
    COLOR_STOPS = [
        (0.0, [0.20, 0.62, 0.18]),   # unripe: green
        (0.5, [0.92, 0.10, 0.08]),   # ripe: vivid red
        (1.0, [0.05, 0.04, 0.03]),   # overripe: near-black
    ]

    def setup_demo(self, **kwags):
        # capture task-scoped params BEFORE init (kwags isn't stored on self otherwise)
        self._cfg = kwags.get("task_args", {}).get("pick_ripe_apple", {})
        super()._init_task_env_(**kwags)
        # start ripening only AFTER setup (the base settles the scene during _init_task_env_ by
        # stepping ~2000x, which would otherwise ripen the apples before the episode begins). Set
        # here -- not in play_once -- so ripening also runs during a policy-evaluation rollout.
        self._ripen_started = True

    # ---------------------------------------------------------------- actors
    def load_actors(self):
        base_steps = int(self._cfg.get("ripen_steps", self.RIPEN_STEPS_DEFAULT))
        # per-episode red target, randomized in a tight band (like cook_meat's target_doneness)
        self.red_window = float(np.random.uniform(
            self._cfg.get("red_window_min", self.RED_WINDOW_MIN_DEFAULT),
            self._cfg.get("red_window_max", self.RED_WINDOW_MAX_DEFAULT)))
        self.red_tol = float(self._cfg.get("red_tolerance", self.RED_TOLERANCE_DEFAULT))
        bz = 0.74 + self.table_z_bias
        board_th = 0.013                                  # ~board thickness at scale_mult 0.08
        apple_sm = float(os.environ.get("PRA_APPLE_SCALE", "0.82"))

        # per-apple ripen rates: one FAST, one SLOW, randomly assigned to left/right. The stagger keeps
        # the two red windows well apart so the (sequential) expert can grasp one at red, then wait for
        # the other -- and it makes the "independently identify left/right timing" skill explicit.
        rates = [base_steps * float(np.random.uniform(0.75, 1.0)),
                 base_steps * float(np.random.uniform(1.9, 2.4))]
        np.random.shuffle(rates)                          # which side ripens faster is random
        self.ripen_steps = [int(r) for r in rates]        # index 0 = left, 1 = right

        self.ripeness = [0.0, 0.0]
        self._ripen_started = False                       # gates ripening OFF during the setup settle
        self._ripen_active = [True, True]                 # per-apple; False once it leaves its board
        self.r_grasp = [None, None]                       # ripeness latched at leave-board
        self.sides = [-1.0, 1.0]                          # left, right
        # weld state: a friction grasp on the small round apple is physically unreliable (slips ~1 in 4
        # lifts, and marginally -> non-deterministic across the collector's two passes). Since the
        # task's subject is RIPENESS TIMING, not the grasp, kinematically weld the grasped apple to its
        # arm's EE so the carry is deterministic (same approach as collect_falling_bowl's bowl).
        self._welded = [False, False]
        self._weld_offset = [None, None]
        self._weld_arm = [None, None]

        self.apples, self.boards, self.bowls = [], [], []
        self._apple_rigids, self._apple_shapes_list, self._board_top = [], [], []
        self.bowl_ids, self.bowl_start_z = [], []

        for side in self.sides:
            board_xy, bowl_xy = self._sample_side_layout(side)

            # cutting board (104_board, static, scaled) UNDER the apple; qpos lays its thin axis vertical
            # so it sits flat, well below the grasp. (104_board's Actor.config is None -> don't prohibit.)
            board = create_actor(
                self, pose=sapien.Pose([float(board_xy[0]), float(board_xy[1]), bz + board_th / 2],
                                       [0.707, 0.707, 0, 0]),
                modelname="104_board", model_id=0, convex=True, is_static=True, scale_mult=0.08,
            )
            self.boards.append(board)
            board_top = bz + board_th
            self._board_top.append(board_top)

            # apple resting on the board surface, oriented so the grasp contact frame approaches top-down
            apple = create_actor(
                self, pose=sapien.Pose([float(board_xy[0]), float(board_xy[1]), board_top + 0.024],
                                       [0.707, 0.707, 0, 0]),
                modelname="035_apple", model_id=0, convex=True, is_static=False, scale_mult=apple_sm,
            )
            apple.set_mass(0.05)
            # apples REST on the board (dynamic) and are grasped with a normal friction grasp; high
            # damping + high friction keep the (spherical) apple from rolling and from squirting out of
            # the gripper, and zero restitution stops it bouncing back out of the bowl.
            rigid = next((c for c in apple.actor.get_components()
                          if isinstance(c, sapien.physx.PhysxRigidDynamicComponent)), None)
            if rigid is not None:
                try:
                    rigid.set_linear_damping(5.0)
                    rigid.set_angular_damping(20.0)
                    for s in rigid.get_collision_shapes():
                        m = s.get_physical_material()
                        m.set_static_friction(4.0)
                        m.set_dynamic_friction(4.0)
                        m.set_restitution(0.0)
                except Exception:
                    pass
            shapes = []
            for c in apple.actor.get_components():
                if isinstance(c, sapien.render.RenderBodyComponent):
                    shapes = list(c.render_shapes)
            self.apples.append(apple)
            self._apple_rigids.append(rigid)
            self._apple_shapes_list.append(shapes)

            # open BOWL on the SAME side, at the rejection-sampled bowl_xy (clear of the board); the
            # arm drops the ripe apple straight into it.
            bid = int(np.random.choice([1, 2, 3, 4, 5, 6, 7]))   # 002_bowl variants (no base0)
            bowl = create_actor(
                self, pose=sapien.Pose([float(bowl_xy[0]), float(bowl_xy[1]), bz],
                                       [0.5, 0.5, 0.5, 0.5]),
                modelname="002_bowl", model_id=bid, convex=True, is_static=False,
            )
            bowl.set_mass(0.5)
            self.bowl_ids.append(bid)
            self.bowls.append(bowl)
            self.bowl_start_z.append(float(bowl.get_pose().p[2]))

            self.add_prohibit_area(apple, padding=0.03)
            self.add_prohibit_area(bowl, padding=0.05)

        self._set_all_colors()                            # start both apples green

    def _sample_side_layout(self, side):
        # Place the board(+apple) and bowl on ONE side, footprints separated (rand_pose does no
        # collision check) and within a short, reliably-reachable pick->drop carry. The arm can GRASP
        # over a wide area but only LIFT where y is FRONT-ish; so the board gets a wide x but a shallow
        # front y, and the bowl goes FRONT or beside it (never behind) so the carry stays IK-feasible.
        BOARD_R, BOWL_R = 0.075, 0.065
        MIN_SEP = BOARD_R + BOWL_R + 0.03               # ~0.17 -> guaranteed no overlap
        MAX_CARRY = 0.19
        board_xy = np.array([side * np.random.uniform(0.10, 0.22),
                             np.random.uniform(-0.02, 0.05)])
        bowl_xy = None
        for _ in range(200):
            k = np.array([side * np.random.uniform(0.06, 0.24),
                          np.random.uniform(-0.16, float(board_xy[1]) + 0.02)])
            if MIN_SEP <= float(np.linalg.norm(board_xy - k)) <= MAX_CARRY:
                bowl_xy = k
                break
        if bowl_xy is None:                             # safe fallback: bowl straight in front
            bowl_xy = np.array([float(board_xy[0]), float(board_xy[1]) - 0.16])
        return board_xy, bowl_xy

    # ----------------------------------------------------------- ripeness
    def _color_for(self, r):
        d = float(np.clip(r, 0.0, 1.0))
        stops = self.COLOR_STOPS
        rgb = stops[-1][1]
        for i in range(len(stops) - 1):
            d0, c0 = stops[i]
            d1, c1 = stops[i + 1]
            if d <= d1 or i == len(stops) - 2:
                t = 0.0 if d1 == d0 else (d - d0) / (d1 - d0)
                t = float(np.clip(t, 0.0, 1.0))
                rgb = [c0[k] + (c1[k] - c0[k]) * t for k in range(3)]
                break
        return list(rgb) + [1.0]

    def _set_apple_color(self, i, r):
        col = self._color_for(r)
        for s in self._apple_shapes_list[i]:
            try:
                s.material.set_base_color(col)
            except Exception:
                pass

    def _set_all_colors(self):
        for i in range(len(self.apples)):
            self._set_apple_color(i, self.ripeness[i])

    # --------------------------------------------------------- grasp weld
    def _ee_pos(self, arm):
        p = self.robot.get_left_ee_pose() if arm == "left" else self.robot.get_right_ee_pose()
        return np.array(p[:3], dtype=float)

    def _weld_apple_to_ee(self, i, arm):
        # rigidly attach the grasped apple to its arm's EE (kinematic, fixed offset) -> deterministic
        # lift+carry with no slip. The apple still ripens until it leaves the board (z threshold).
        self._weld_arm[i] = ("left" if str(arm) == "left" else "right")
        self._weld_offset[i] = np.array(self.apples[i].get_pose().p, dtype=float) - self._ee_pos(self._weld_arm[i])
        rigid = self._apple_rigids[i]
        if rigid is not None:
            rigid.set_disable_gravity(True)
            rigid.set_kinematic(True)
        self._welded[i] = True

    def _update_welded_apples(self):
        if not hasattr(self, "_welded"):
            return
        for i in range(len(self.apples)):
            if not self._welded[i]:
                continue
            target = self._ee_pos(self._weld_arm[i]) + self._weld_offset[i]
            q = self.apples[i].get_pose().q
            rigid = self._apple_rigids[i]
            if rigid is not None:
                rigid.set_kinematic_target(sapien.Pose(target, q))
            else:
                self.apples[i].actor.set_pose(sapien.Pose(target, q))

    def _release_apple(self, i):
        # un-weld: turn the apple back into a free dynamic body so it drops into the bowl
        self._welded[i] = False
        rigid = self._apple_rigids[i]
        if rigid is not None:
            try:
                rigid.set_kinematic(False)
                rigid.set_disable_gravity(False)
                rigid.set_linear_velocity(np.zeros(3))
                rigid.set_angular_velocity(np.zeros(3))
            except Exception:
                pass

    def _update_kinematic_tasks(self):
        # base hook drives DOMINO dynamic-object motion; runs every physics step (collection AND eval)
        super()._update_kinematic_tasks()
        self._update_welded_apples()                      # held apples track their EE every step
        if not getattr(self, "_ripen_started", False):
            return                                        # no ripening until the episode actually starts
        for i in range(len(self.apples)):
            if not self._ripen_active[i]:
                continue
            apz = float(self.apples[i].get_pose().p[2])
            if apz > self._board_top[i] + self.LEAVE_BOARD_DZ:
                # the apple has LEFT its board -> freeze ripeness/colour and latch the grasp ripeness
                self._ripen_active[i] = False
                if self.r_grasp[i] is None:
                    self.r_grasp[i] = float(self.ripeness[i])
            else:
                self.ripeness[i] = min(1.0, self.ripeness[i] + 1.0 / max(1, self.ripen_steps[i]))
                self._set_apple_color(i, self.ripeness[i])

    def _ripen_until(self, i, target):
        # OBSERVE: dwell (both apples keep ripening on their boards) until apple i reaches `target`
        # ripeness, recording frames. Bails if the apple has already left its board.
        max_steps = int(self.ripen_steps[i]) + 600
        for j in range(max_steps):
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (j % self.save_freq == 0):
                self._take_picture()
            if (not self._ripen_active[i]) or self.ripeness[i] >= target:
                break

    # ------------------------------------------------------------- policy
    def play_once(self):
        # handle the apples in the order they turn red (the faster-ripening one first); each arm acts
        # on its own side, observing its apple ripen then grasping it at the red window.
        order = sorted(range(len(self.apples)), key=lambda i: self.ripen_steps[i])
        if os.environ.get("PRA_DEBUG"):
            print(f"[PRA] order={order} ripen_steps={self.ripen_steps} sides={self.sides}", flush=True)
        for i in order:
            arm = ArmTag("left" if self.sides[i] < 0 else "right")
            # OBSERVE: wait until this side's apple is ALMOST red -- lead the grasp by the time it
            # takes to lift the apple off the board, so it crosses red exactly as it leaves -> r_grasp
            # lands near 0.5. (Lead is larger for the faster-ripening apple.)
            target = max(0.18, self.red_window - self.GRASP_LEAD_STEPS / max(1, self.ripen_steps[i]))
            self._ripen_until(i, target)
            if os.environ.get("PRA_DEBUG"):
                print(f"[PRA] apple{i} side={self.sides[i]:+.0f} target={target:.2f} "
                      f"trigger_ripeness={self.ripeness[i]:.2f}", flush=True)
            # ACT: grasp it top-down (lifting it off the board freezes its ripeness), carry to the
            # same-side bowl, and open to drop it in. A RELATIVE carry (keeping the grasp
            # orientation) plans far more reliably than an absolute place onto the rim.
            self.move(self.grasp_actor(self.apples[i], arm_tag=arm, pre_grasp_dis=0.1, gripper_pos=0.0))
            self._weld_apple_to_ee(i, arm)                # attach to the EE so the lift can't slip
            self.move(self.move_by_displacement(arm_tag=arm, z=0.10, move_axis="world"))   # lift
            if os.environ.get("PRA_DEBUG"):
                print(f"[PRA] apple{i} after lift: ripeness={self.ripeness[i]:.2f} "
                      f"r_grasp={self.r_grasp[i]} z={self.apples[i].get_pose().p[2]:.3f} "
                      f"plan={self.plan_success}", flush=True)
            bp = np.array(self.bowls[i].get_pose().p)
            ap = np.array(self.apples[i].get_pose().p)
            self.move(self.move_by_displacement(arm_tag=arm, x=float(bp[0] - ap[0]), y=float(bp[1] - ap[1])))
            placed = self.plan_success
            self.move(self.open_gripper(arm))             # open fingers...
            self._release_apple(i)                        # ...and un-weld -> apple drops into the bowl
            self.move(self.move_by_displacement(arm_tag=arm, z=0.08, move_axis="arm"))   # retreat
            self.plan_success = placed                    # a failed cosmetic retreat must not flag failure

        self.info["info"] = {
            "{A}": "035_apple/base0",
            "{B}": f"002_bowl/base{self.bowl_ids[0]}",
        }
        return self.info

    # --------------------------------------------------------- ripeness score
    def _ripeness_score(self, i):
        if self.r_grasp[i] is None:
            return 0.0
        # closeness to THIS episode's red target (randomized in the band)
        return float(np.clip(1.0 - abs(self.r_grasp[i] - self.red_window) / 0.5, 0.0, 1.0))

    # ------------------------------------------------------------- success
    def check_success(self):
        n = len(self.apples)
        all_in = True
        for i in range(n):
            ap = np.array(self.apples[i].get_pose().p)
            bp = np.array(self.bowls[i].get_pose().p)
            xy_close = float(np.linalg.norm(ap[:2] - bp[:2])) < 0.12
            not_floor = ap[2] > (0.60 + self.table_z_bias)
            settled = ap[2] < (bp[2] + 0.10)             # dropped in, not held above the rim
            upright = (bp[2] - self.bowl_start_z[i]) > -0.05
            in_bowl = bool(xy_close and settled and not_floor and upright)
            all_in = all_in and in_bowl and (self.r_grasp[i] is not None)
        success = bool(all_in)

        mean_ripe = float(np.mean([self._ripeness_score(i) for i in range(n)]))
        self.info["ripeness_score"] = mean_ripe
        self.info["r_grasp_left"] = float(self.r_grasp[0]) if self.r_grasp[0] is not None else -1.0
        self.info["r_grasp_right"] = float(self.r_grasp[1]) if self.r_grasp[1] is not None else -1.0
        self.info["final_score"] = mean_ripe if success else 0.0
        self.info["in_bowl"] = success
        return success

    # record per-frame ripeness state into the trajectory
    def get_obs(self):
        obs = super().get_obs()
        rip = getattr(self, "ripeness", [0.0, 0.0])
        rg = getattr(self, "r_grasp", [None, None])
        obs["ripening"] = {
            "ripeness_left": float(rip[0]),
            "ripeness_right": float(rip[1]),
            "r_grasp_left": float(rg[0]) if rg[0] is not None else -1.0,
            "r_grasp_right": float(rg[1]) if rg[1] is not None else -1.0,
            "red_window": float(getattr(self, "red_window", 0.5)),
        }
        return obs
