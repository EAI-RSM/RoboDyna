from ._base_task import Base_Task
from .utils import *
import sapien
import sapien.physx
import numpy as np


class cup_curtain_slot(Base_Task):
    """Double-dynamic, single-arm (RIGHT) task.

    The right arm starts holding a cup in the near zone. In the mid zone a curtain of
    vertical strips sways laterally (collidable -- touching it fails the attempt). Behind
    the curtain a belt carries cup-holder slots with ONE empty slot; the whole belt
    translates laterally. The policy must wait for a window when (a) a curtain gap opens
    over the empty slot's x AND (b) the empty slot is within the arm's reach, then pass
    the cup through the gap and deposit it into the moving slot, then retract.

    Both the curtain phase and the slot position are pure functions of an internal step
    counter so the two collector passes (plan-only and render) stay deterministic.

    Metric: placement_score = clamp(1 - center_offset(cup, slot)/tol, 0, 1); forced to 0
    if the cup ever collided with the curtain.
    """

    # ---- tunable params (CLASS DEFAULTS; overridable via task_args.cup_curtain_slot) ----
    CURTAIN_FREQ_DEFAULT = 0.8          # curtain oscillation: cycles per (sim-second-equivalent)
    CURTAIN_AMP_DEFAULT = 0.05          # lateral sway amplitude of the curtain group (m)
    CURTAIN_PHASE_DEFAULT = 0.0         # initial phase offset (rad)
    BELT_SPEED_DEFAULT = 0.04           # belt lateral speed magnitude (m per "second")
    BELT_RANGE_DEFAULT = 0.06           # belt travels +/- this in x before reversing (m)
    SLOT_START_DEFAULT = 0.0            # initial slot x offset within belt range (m)
    PLACEMENT_TOL_DEFAULT = 0.07        # tolerance for placement_score (m)
    GAP_TOL_DEFAULT = 0.06              # how close (in x) the gap center must be to the corridor for "open"
    REACH_TOL_DEFAULT = 0.07            # how close the slot x must be to the corridor

    REACH_STEPS_EST = 60                # est. physics steps for the reach-through to complete
    N_STRIPS = 3                        # curtain vertical strips
    STRIP_GAP_IDX = 1                   # which inter-strip gap is "the gap" (center)
    CURTAIN_Y = 0.0                     # mid-zone y of the curtain plane
    BELT_Y = 0.06                       # far-mid y of the belt (behind the curtain)
    BELT_Z_OFF = 0.01                   # belt platform half-height sits this far above table
    DT = 1.0 / 250.0                    # matches default scene timestep

    def setup_demo(self, **kwags):
        # capture task-scoped params BEFORE init (kwags not stored on self otherwise)
        self._cfg = kwags.get("task_args", {}).get("cup_curtain_slot", {})
        # CRITICAL: the env instance is REUSED across episodes. _init_task_env_ resets the scene
        # and calls _update_kinematic_tasks (via load_camera) BEFORE our load_actors rebuilds the
        # bodies. Invalidate stale kinematic-body handles now so the per-step hook is a no-op until
        # load_actors recreates them -- otherwise set_kinematic_target on a freed body segfaults.
        self.curtain_strips = []
        self.belt_pegs = []
        self._attempt_active = False
        if hasattr(self, "_kin_step"):
            del self._kin_step
        super()._init_task_env_(**kwags)

    # --------------------------------------------------------------- actors
    def load_actors(self):
        c = self._cfg
        self.curtain_freq = float(c.get("curtain_freq", self.CURTAIN_FREQ_DEFAULT))
        self.curtain_amp = float(c.get("curtain_amp", self.CURTAIN_AMP_DEFAULT))
        self.belt_speed = float(c.get("belt_speed", self.BELT_SPEED_DEFAULT))
        self.belt_range = float(c.get("belt_range", self.BELT_RANGE_DEFAULT))
        self.placement_tol = float(c.get("placement_tol", self.PLACEMENT_TOL_DEFAULT))
        self.gap_tol = float(c.get("gap_tol", self.GAP_TOL_DEFAULT))
        self.reach_tol = float(c.get("reach_tol", self.REACH_TOL_DEFAULT))

        # seed-randomized curtain phase / belt start / slot start
        self.curtain_phase = float(c.get("curtain_phase",
                                         np.random.uniform(0, 2 * np.pi)))
        self.slot_start = float(c.get("slot_start",
                                      np.random.uniform(-0.5, 0.5) * self.belt_range))
        self.belt_dir = float(np.random.choice([-1.0, 1.0]))

        # internal step counter drives BOTH dynamics (two-pass determinism)
        self._kin_step = 0
        self._curtain_hit = False          # set True if the cup touches a strip during the attempt
        self._attempt_active = False       # curtain collisions only count once the cup is in motion
        self._deposit_step = None
        self._slot_x_at_deposit = None
        self._belt_frozen_offset = None

        z0 = 0.74 + self.table_z_bias

        # ----- the cup: RIGHT side, near zone, the arm starts by grasping it (= "held") -----
        # Spawn recipe mirrors the working sibling pick_cup_behind_fan (qpos lays the cup so the
        # side contact points are reachable; mass 0.15 so the grasp is stable).
        self.cup_id = 0
        self.cup_x = float(np.random.uniform(0.10, 0.14))
        self.cup_y = float(np.random.uniform(-0.055, -0.045))
        cup_pose = rand_pose(
            xlim=[self.cup_x, self.cup_x], ylim=[self.cup_y, self.cup_y], zlim=[z0],
            qpos=[0.707, 0.707, 0.0, 0.0], rotate_rand=False,
        )
        self.cup = create_actor(
            self, pose=cup_pose, modelname="021_cup",
            model_id=self.cup_id, convex=True, is_static=False,
        )
        self.cup.set_mass(0.15)

        # ----- the curtain: a row of thin vertical strips in the mid zone -----
        # Strips span the right half so a single-arm (right) task stays same-side. The curtain
        # corridor is centred on the cup so the whole grasp->reach-through->deposit stays in one
        # reachable column. Strips are short (the cup is lifted and threaded through the gap).
        self.curtain_center_x = float(np.clip(self.cup_x, 0.10, 0.16))
        self.strip_half = (0.015, 0.004, 0.022)   # (x,y,z) half-sizes: thin, short strips
        self.strip_spacing = 0.045
        self._strip_base_x = []            # rest x of each strip (group-relative removed)
        self.curtain_strips = []
        n = self.N_STRIPS
        for i in range(n):
            bx = self.curtain_center_x + (i - (n - 1) / 2.0) * self.strip_spacing
            self._strip_base_x.append(bx)
            strip = create_box(
                scene=self,
                pose=sapien.Pose([bx, self.CURTAIN_Y, z0 + self.strip_half[2]],
                                 [1, 0, 0, 0]),
                half_size=self.strip_half,
                color=(0.20, 0.45, 0.75),
                name=f"curtain_strip_{i}",
                is_static=False,
            )
            for comp in strip.actor.get_components():
                if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent):
                    comp.set_kinematic(True)
            self.curtain_strips.append(strip)

        # ----- the belt: a kinematic platform carrying cup-slots, ONE empty -----
        # The belt row sits just behind the curtain. The empty slot is a gap in a row of
        # short walls; we track its x. The whole row translates laterally (belt motion).
        self.slot_z = z0 + self.BELT_Z_OFF
        self.belt_center_x = self.curtain_center_x
        self.slot_spacing = 0.075
        self.n_slots = 3
        self.empty_slot_idx = 1            # the center slot is empty
        # belt base platform (kinematic, collidable floor of the belt) -- thin so the cup rests on it
        self.belt_plate = create_box(
            scene=self,
            pose=sapien.Pose([self.belt_center_x, self.BELT_Y, z0 + 0.004], [1, 0, 0, 0]),
            half_size=(0.14, 0.045, 0.004),
            color=(0.35, 0.35, 0.40),
            name="belt_plate",
            is_static=False,
        )
        for comp in self.belt_plate.actor.get_components():
            if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent):
                comp.set_kinematic(True)

        # filler "cups already on the belt" as short pegs occupying the non-empty slots. Kept low
        # so they don't obstruct the planner's descent into the (clear) empty slot between them.
        self.belt_pegs = []
        self._peg_base_x = []
        for i in range(self.n_slots):
            if i == self.empty_slot_idx:
                continue
            bx = self.belt_center_x + (i - (self.n_slots - 1) / 2.0) * self.slot_spacing
            peg = create_box(
                scene=self,
                pose=sapien.Pose([bx, self.BELT_Y, z0 + 0.022], [1, 0, 0, 0]),
                half_size=(0.02, 0.02, 0.018),
                color=(0.8, 0.7, 0.3),
                name=f"belt_peg_{i}",
                is_static=False,
            )
            for comp in peg.actor.get_components():
                if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent):
                    comp.set_kinematic(True)
            self.belt_pegs.append(peg)
            self._peg_base_x.append(bx)

        # rest x of the empty slot (belt offset added each step)
        self._slot_base_x = (self.belt_center_x +
                             (self.empty_slot_idx - (self.n_slots - 1) / 2.0) * self.slot_spacing)

        # reserve space so randomized clutter never lands on the scene
        self.add_prohibit_area(self.cup, padding=0.05)
        self.add_prohibit_area([self.curtain_center_x, self.CURTAIN_Y, z0, 1, 0, 0, 0],
                               padding=0.20)
        self.add_prohibit_area([self.belt_center_x, self.BELT_Y, z0, 1, 0, 0, 0],
                               padding=0.20)

        # apply the initial dynamic transforms (kin_step = 0)
        self._apply_kinematics(initial=True)

    # ------------------------------------------------------- dynamics model
    def _t(self):
        """Internal time (seconds-equivalent) from the step counter."""
        return self._kin_step * self.DT

    def _curtain_offset(self, t=None):
        if t is None:
            t = self._t()
        return self.curtain_amp * np.sin(2 * np.pi * self.curtain_freq * t + self.curtain_phase)

    def _belt_offset(self, t=None):
        """Triangle-wave belt travel within +/- belt_range."""
        if getattr(self, "_belt_frozen_offset", None) is not None:
            return self._belt_frozen_offset
        if t is None:
            t = self._t()
        if self.belt_range <= 1e-6:
            return self.slot_start
        # triangle wave in [-range, range], starting from slot_start moving belt_dir
        period = 4 * self.belt_range / max(self.belt_speed, 1e-6)
        phase = (self.belt_dir * self.belt_speed * t + self.slot_start)
        # fold into triangle wave
        m = (phase + self.belt_range) % (2 * self.belt_range)
        tri = m - self.belt_range
        # reflect to make it continuous triangle (so motion reverses smoothly)
        period2 = 2 * self.belt_range
        k = int(np.floor((phase + self.belt_range) / period2))
        if k % 2 == 1:
            tri = -tri
        return float(np.clip(tri, -self.belt_range, self.belt_range))

    def slot_x(self, t=None):
        return self._slot_base_x + self._belt_offset(t)

    def gap_center_x(self, t=None):
        """World x of the curtain gap (the STRIP_GAP_IDX-th inter-strip gap)."""
        i = self.STRIP_GAP_IDX
        rest = 0.5 * (self._strip_base_x[i] + self._strip_base_x[i + 1])
        return rest + self._curtain_offset(t)

    def _set_body(self, actor, target_pose, initial=False):
        """Drive a kinematic body to target_pose. On the first call also hard-set the
        entity pose; thereafter use set_kinematic_target so PhysX interpolates each step."""
        if initial:
            actor.actor.set_pose(target_pose)
        for comp in actor.actor.get_components():
            if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent):
                comp.set_kinematic_target(target_pose)

    def _apply_kinematics(self, initial=False):
        """Push current step-driven targets onto the kinematic bodies."""
        co = self._curtain_offset()
        for i, strip in enumerate(self.curtain_strips):
            p = strip.get_pose()
            self._set_body(strip,
                           sapien.Pose([self._strip_base_x[i] + co, p.p[1], p.p[2]], p.q),
                           initial=initial)
        bo = self._belt_offset()
        for j, peg in enumerate(self.belt_pegs):
            p = peg.get_pose()
            self._set_body(peg,
                           sapien.Pose([self._peg_base_x[j] + bo, p.p[1], p.p[2]], p.q),
                           initial=initial)

    def _check_curtain_contact(self):
        if self._curtain_hit or not self._attempt_active:
            return
        for i in range(self.N_STRIPS):
            try:
                if self.check_actors_contact("021_cup", f"curtain_strip_{i}"):
                    self._curtain_hit = True
                    import os
                    if os.environ.get("CCS_DEBUG"):
                        cz = float(self.cup.get_pose().p[2])
                        print(f"[CCS] CURTAIN HIT strip {i} at step {self._kin_step} cup_z={cz:.3f}",
                              flush=True)
                    return
            except Exception:
                pass

    def _update_kinematic_tasks(self):
        # base hook first (drives any built-in DOMINO motion), then our step-driven dynamics
        super()._update_kinematic_tasks()
        # this hook can fire (via base scene setup) before load_actors builds our scene
        if not hasattr(self, "_kin_step") or not getattr(self, "curtain_strips", None):
            return
        self._kin_step += 1
        self._apply_kinematics()
        self._check_curtain_contact()

    # ------------------------------------------------------------- helpers
    def _dwell(self, steps):
        """Advance both dynamics `steps` physics steps while recording frames."""
        for i in range(int(steps)):
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (i % self.save_freq == 0):
                self._take_picture()

    def _window_open(self, t=None):
        """True when a curtain gap is open AND the empty slot is at the reachable corridor.
        The corridor is the curtain center column (where the arm reaches straight ahead)."""
        slot = self.slot_x(t)
        gap = self.gap_center_x(t)
        gap_ok = abs(gap - self.curtain_center_x) < self.gap_tol     # gap is over the corridor
        reach_ok = abs(slot - self.curtain_center_x) < self.reach_tol  # slot is at the corridor
        return gap_ok and reach_ok

    def _steps_until_window(self, lead_steps=0, max_steps=1500):
        """Analytically scan future step counts for the first one whose window (lead_steps later)
        is open. Returns the number of dwell steps to take now (>=0), or 0 if none found."""
        for s in range(int(max_steps)):
            t = (self._kin_step + s + lead_steps) * self.DT
            if self._window_open(t=t):
                return s
        return 0

    def _wait_for_window(self, lead_steps=0, max_steps=1500):
        """Compute the dwell needed for the alignment window to be open `lead_steps` ahead, then
        dwell exactly that many physics steps (recording frames). Step-driven -> deterministic."""
        n = self._steps_until_window(lead_steps=lead_steps, max_steps=max_steps)
        self._dwell(n)
        return self._window_open(t=(self._kin_step + lead_steps) * self.DT)

    def _freeze_belt(self):
        """Stop the belt so the deposited cup settles into a fixed slot position."""
        self._belt_frozen_offset = self._belt_offset()

    # ------------------------------------------------------------- policy
    def _dbg(self, tag):
        import os
        if os.environ.get("CCS_DEBUG"):
            print(f"[CCS] {tag}: plan_success={self.plan_success}", flush=True)

    def play_once(self):
        arm_tag = ArmTag("right")     # single-arm, right (cup spawns on the right half)

        # 1) right arm picks up the cup (establishes "holding the cup") and lifts it clear of
        #    the (short) curtain. Grasp recipe mirrors the working pick_cup_behind_fan sibling.
        self.move(self.grasp_actor(self.cup, arm_tag=arm_tag, pre_grasp_dis=0.1))
        self._dbg("after_grasp")
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.10, move_axis="arm"))
        self._dbg("after_lift")
        self._attempt_active = True            # from here a curtain collision fails the attempt

        # 2) track BOTH moving things; dwell until the curtain gap is open over the corridor AND
        #    the empty slot will be at the corridor by the time the reach completes.
        self._dwell(8)                         # observe the dynamics a moment
        self._wait_for_window(lead_steps=self.REACH_STEPS_EST, max_steps=1500)
        self._reach_step = self._kin_step

        # 3) pass the cup forward through the gap to above the slot, freeze the belt so the slot
        #    holds still, nudge in x to centre over the (now fixed) slot, then lower it in.
        dy = self.BELT_Y - self.cup_y          # reach from the near cup row to the belt row
        self.move(self.move_by_displacement(arm_tag=arm_tag, y=dy))
        self._dbg("after_reach")
        self._freeze_belt()                    # stop belt so the cup settles in a fixed slot
        dx = float(np.clip(self.slot_x() - self.cup.get_pose().p[0], -0.05, 0.05))
        self.move(self.move_by_displacement(arm_tag=arm_tag, x=dx))
        self._dbg("after_xcorrect")
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=-0.09))
        self._dbg("after_lower")
        self._deposit_step = self._kin_step
        self._slot_x_at_deposit = self.slot_x()
        self.move(self.open_gripper(arm_tag))

        # let it settle (curtain keeps swaying; records frames)
        self._dwell(15)

        # 4) retract up and home
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.10, move_axis="arm"))
        self._dbg("after_retract")
        self.move(self.back_to_origin(arm_tag))

        self.info["info"] = {
            "{A}": f"021_cup/base{self.cup_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    # ------------------------------------------------------------- metric
    def _center_offset(self):
        """Horizontal distance from the cup bottom to the (current) empty slot center."""
        cup_p = np.array(self.cup.get_functional_point(0, "pose").p[:2])
        slot_p = np.array([self.slot_x(), self.BELT_Y])
        return float(np.linalg.norm(cup_p - slot_p))

    def placement_score(self):
        if self._curtain_hit:
            return 0.0
        off = self._center_offset()
        return float(np.clip(1.0 - off / max(self.placement_tol, 1e-6), 0.0, 1.0))

    # ------------------------------------------------------------- success
    def check_success(self):
        if self._curtain_hit:
            return False
        if self._deposit_step is None:
            return False
        # cup must be seated near the slot (low above the belt) and not still aloft / dropped
        cup_p = self.cup.get_functional_point(0, "pose").p
        z_ok = (self.slot_z - 0.04) < float(cup_p[2]) < (self.slot_z + 0.12)
        off = self._center_offset()
        seated = off < self.placement_tol
        import os
        if os.environ.get("CCS_DEBUG"):
            print(f"[CCS] check: cup_xyz={np.round(np.array(cup_p),3).tolist()} "
                  f"slot_x={self.slot_x():.3f} off={off:.3f} z_ok={z_ok} seated={seated} "
                  f"hit={self._curtain_hit} score={self.placement_score():.2f}", flush=True)
        return bool(z_ok and seated and self.placement_score() > 0.0)

    # ------------------------------------------------------------- obs
    def get_obs(self):
        obs = super().get_obs()
        obs["curtain_slot"] = {
            "curtain_phase": float(2 * np.pi * self.curtain_freq * self._t() + self.curtain_phase),
            "curtain_offset": float(self._curtain_offset()),
            "gap_center_x": float(self.gap_center_x()),
            "slot_x": float(self.slot_x()),
            "window_open": bool(self._window_open()),
            "curtain_hit": bool(self._curtain_hit),
            "placement_score": float(self.placement_score()),
        }
        return obs
