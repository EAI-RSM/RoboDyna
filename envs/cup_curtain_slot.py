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
    BELT_SPEED_DEFAULT = 0.2           # belt lateral speed magnitude (m per "second")
    BELT_RANGE_DEFAULT = 0.12           # belt travels +/- this in x before reversing (m)
    SLOT_START_DEFAULT = 0.0            # initial slot x offset within belt range (m)
    PLACEMENT_TOL_DEFAULT = 0.07        # tolerance for placement_score (m)
    GAP_TOL_DEFAULT = 0.06              # how close (in x) the gap center must be to the corridor for "open"
    REACH_TOL_DEFAULT = 0.07            # how close the slot x must be to the corridor
    MOVING_COMPONENT_SPACING_DEFAULT = 0.075
    BLUE_CURTAINS_ENABLED_DEFAULT = True
    BLUE_CURTAIN_DYNAMIC_ENABLED_DEFAULT = False
    PLACE_CUP_IN_MIDDLE_OF_YELLOW_TOOLS_DEFAULT = True
    LIFT_OFF_ONLY_AFTER_PLACE_DEFAULT = False
    POST_PLACE_LIFT_Z_DEFAULT = 0.10

    REACH_STEPS_EST = 60                # est. physics steps for the reach-through to complete
    N_STRIPS = 3                        # curtain vertical strips
    STRIP_GAP_IDX = 1                   # which inter-strip gap is "the gap" (center)
    CURTAIN_Y = 0.03                     # mid-zone y of the curtain plane
    BELT_CENTER_Y_OFFSET = 0.12         # place the gray surface +0.2m from table center in y
    BELT_Z_OFF = 0.01                   # belt platform half-height sits this far above table
    DT = 1.0 / 250.0                    # matches default scene timestep
    BELT_PLATE_HALF_X = 0.14 * 2.0      # double the original x half-size
    BELT_PLATE_HALF_Y = 0.045 * 1.5     # 1.5x the original y half-size
    MOVING_COMPONENT_HALF_X = 0.02 * 0.5
    MOVING_COMPONENT_HALF_Z = 0.018
    BLUE_CUP_RGBA = [0.10, 0.35, 0.95, 1.0]

    def setup_demo(self, **kwags):
        # capture task-scoped params BEFORE init (kwags not stored on self otherwise)
        self._cfg = kwags.get("task_args", {}).get("cup_curtain_slot", {})
        # CRITICAL: the env instance is REUSED across episodes. _init_task_env_ resets the scene
        # and calls _update_kinematic_tasks (via load_camera) BEFORE our load_actors rebuilds the
        # bodies. Invalidate stale kinematic-body handles now so the per-step hook is a no-op until
        # load_actors recreates them -- otherwise set_kinematic_target on a freed body segfaults.
        self.curtain_strips = []
        self.belt_pegs = []
        self.belt_plate = None
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
        self.slot_spacing = float(c.get("moving_component_spacing",
                                        c.get("slot_spacing", self.MOVING_COMPONENT_SPACING_DEFAULT)))
        self.blue_curtains_enabled = bool(c.get("blue_curtains_enabled",
                                                c.get("curtain_enabled", self.BLUE_CURTAINS_ENABLED_DEFAULT)))
        self.blue_curtain_dynamic_enabled = bool(
            c.get("blue_curtain_dynamic_enabled",
                  c.get("blue_curtain_dynamic", self.BLUE_CURTAIN_DYNAMIC_ENABLED_DEFAULT))
        )
        self.place_cup_in_middle_of_yellow_tools = bool(
            c.get(
                "place_cup_in_middle_of_yellow_tools",
                self.PLACE_CUP_IN_MIDDLE_OF_YELLOW_TOOLS_DEFAULT,
            )
        )
        self.lift_off_only_after_place = bool(
            c.get("lift_off_only_after_place", self.LIFT_OFF_ONLY_AFTER_PLACE_DEFAULT)
        )
        self.post_place_lift_z = float(c.get("post_place_lift_z", self.POST_PLACE_LIFT_Z_DEFAULT))

        # seed-randomized curtain phase / belt start / slot start
        self.curtain_phase = float(c.get("curtain_phase",
                                         np.random.uniform(0, 2 * np.pi)))
        slot_start_cfg = c.get("slot_start", None)
        self.belt_dir = float(np.random.choice([-1.0, 1.0]))

        # internal step counter drives BOTH dynamics (two-pass determinism)
        self._kin_step = 0
        self._curtain_hit = False          # set True if the cup touches a strip during the attempt
        self._attempt_active = False       # curtain collisions only count once the cup is in motion
        self._deposit_step = None
        self._slot_x_at_deposit = None
        self._belt_frozen_offset = None

        z0 = 0.74 + self.table_z_bias
        self.belt_center_x = float(c.get("belt_center_x", self.table_xy_bias[0]))
        self.belt_y = float(c.get("belt_y", self.table_xy_bias[1] + self.BELT_CENTER_Y_OFFSET))
        self.curtain_y = float(c.get("curtain_y", self.CURTAIN_Y))
        self.belt_plate_half_size = (
            self.BELT_PLATE_HALF_X,
            self.BELT_PLATE_HALF_Y,
            0.004,
        )
        self.moving_component_half_size = (
            self.MOVING_COMPONENT_HALF_X,
            self.belt_plate_half_size[1],
            self.MOVING_COMPONENT_HALF_Z,
        )
        self.belt_motion_limit = max(
            0.0,
            self.belt_plate_half_size[0] - self.moving_component_half_size[0] - self.slot_spacing,
        )
        if slot_start_cfg is None:
            self.slot_start = float(np.random.uniform(-self.belt_motion_limit, self.belt_motion_limit))
        else:
            self.slot_start = float(np.clip(float(slot_start_cfg), -self.belt_motion_limit, self.belt_motion_limit))

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
        self._set_cup_color(self.BLUE_CUP_RGBA)
        for comp in self.cup.actor.get_components():
            if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent):
                try:
                    comp.set_linear_damping(3.0)
                    comp.set_angular_damping(8.0)
                except Exception:
                    pass
        self.cup_height = float(self._actor_world_size(self.cup)[2])

        # ----- the curtain: a row of thin vertical strips in the mid zone -----
        # Strips span the right half so a single-arm (right) task stays same-side. The curtain
        # corridor is centred on the cup so the whole grasp->reach-through->deposit stays in one
        # reachable column. Strips are short (the cup is lifted and threaded through the gap).
        curtain_gap_width = max(
            2.0 * max(self.slot_spacing - self.moving_component_half_size[0], 0.005),
            0.02,
        )
        strip_full_x = curtain_gap_width / max(self.N_STRIPS, 1)
        strip_gap_x = 0.5 * strip_full_x
        strip_half_x = 0.5 * strip_full_x
        curtain_half_z = 0.75 * self.cup_height
        self.strip_half = (strip_half_x, 0.004, curtain_half_z)
        self.strip_spacing = strip_full_x + strip_gap_x
        self._strip_base_x = []            # rest x of each strip (group-relative removed)
        self.curtain_strips = []
        n = self.N_STRIPS
        strip_offsets = [
            (i - (n - 1) / 2.0) * self.strip_spacing
            for i in range(n)
        ]
        curtain_half_span_x = (max(abs(off) for off in strip_offsets) + self.strip_half[0]) if strip_offsets else 0.0
        self.curtain_motion_limit = max(0.0, self.belt_plate_half_size[0] - curtain_half_span_x)
        if self.blue_curtains_enabled and not self.blue_curtain_dynamic_enabled:
            x_low = self.belt_center_x - self.curtain_motion_limit
            x_high = self.belt_center_x + self.curtain_motion_limit
            self.curtain_center_x = float(np.random.uniform(x_low, x_high)) if x_high > x_low else float(self.belt_center_x)
            belt_front_y = self.belt_y - self.belt_plate_half_size[1] - self.strip_half[1] - 0.01
            y_low = float(c.get("static_curtain_y_min", min(self.CURTAIN_Y, belt_front_y)))
            y_high = float(c.get("static_curtain_y_max", belt_front_y))
            if y_high < y_low:
                y_low = y_high
            self.curtain_y = float(np.random.uniform(y_low, y_high)) if y_high > y_low else float(y_low)
        else:
            self.curtain_center_x = float(self.belt_center_x)
        if self.blue_curtains_enabled:
            for i, off in enumerate(strip_offsets):
                bx = self.curtain_center_x + off
                self._strip_base_x.append(bx)
                strip = create_box(
                    scene=self,
                    pose=sapien.Pose([bx, self.curtain_y, z0 + self.strip_half[2]],
                                     [1, 0, 0, 0]),
                    half_size=self.strip_half,
                    color=(0.20, 0.45, 0.75),
                    name=f"curtain_strip_{i}",
                    is_static=not self.blue_curtain_dynamic_enabled,
                )
                if self.blue_curtain_dynamic_enabled:
                    for comp in strip.actor.get_components():
                        if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent):
                            comp.set_kinematic(True)
                self.curtain_strips.append(strip)

        # ----- the belt: a kinematic platform carrying cup-slots, ONE empty -----
        # The belt row sits just behind the curtain. The empty slot is a gap in a row of
        # short walls; we track its x. The whole row translates laterally (belt motion).
        self.slot_z = z0 + self.BELT_Z_OFF
        self.n_slots = 3
        self.empty_slot_idx = 1            # the center slot is empty
        # belt base platform (kinematic, collidable floor of the belt) -- thin so the cup rests on it
        self.belt_plate = create_box(
            scene=self,
            pose=sapien.Pose([self.belt_center_x, self.belt_y, z0 + self.belt_plate_half_size[2]], [1, 0, 0, 0]),
            half_size=self.belt_plate_half_size,
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
                pose=sapien.Pose([bx, self.belt_y, z0 + self.BELT_Z_OFF + self.moving_component_half_size[2]],
                                 [1, 0, 0, 0]),
                half_size=self.moving_component_half_size,
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
        if self.blue_curtains_enabled:
            for strip in self.curtain_strips:
                self.add_prohibit_area(strip, padding=0.03)
        self.add_prohibit_area(self.belt_plate, padding=0.04)

        # apply the initial dynamic transforms (kin_step = 0)
        self._apply_kinematics(initial=True)

    # ------------------------------------------------------- dynamics model
    def _t(self):
        """Internal time (seconds-equivalent) from the step counter."""
        return self._kin_step * self.DT

    def _curtain_offset(self, t=None):
        if not getattr(self, "blue_curtain_dynamic_enabled", False):
            return 0.0
        if t is None:
            t = self._t()
        limit = float(getattr(self, "curtain_motion_limit", 0.0))
        if limit <= 1e-6:
            return 0.0
        return limit * np.sin(2 * np.pi * self.curtain_freq * t + self.curtain_phase)

    def _belt_offset(self, t=None):
        """Triangle-wave belt travel until one yellow mover reaches the belt edge."""
        if getattr(self, "_belt_frozen_offset", None) is not None:
            return self._belt_frozen_offset
        if t is None:
            t = self._t()
        travel_limit = float(getattr(self, "belt_motion_limit", 0.0))
        if travel_limit <= 1e-6:
            return self.slot_start
        # Triangle wave in [-limit, limit], where the limit is set by the outer yellow mover
        # reaching the corresponding edge of the gray belt plate.
        period = 4 * travel_limit / max(self.belt_speed, 1e-6)
        phase = (self.belt_dir * self.belt_speed * t + self.slot_start)
        # fold into triangle wave
        m = (phase + travel_limit) % (2 * travel_limit)
        tri = m - travel_limit
        # reflect to make it continuous triangle (so motion reverses smoothly)
        period2 = 2 * travel_limit
        k = int(np.floor((phase + travel_limit) / period2))
        if k % 2 == 1:
            tri = -tri
        return float(np.clip(tri, -travel_limit, travel_limit))

    def slot_x(self, t=None):
        return self._slot_base_x + self._belt_offset(t)

    def gap_center_x(self, t=None):
        """World x of the curtain gap (the STRIP_GAP_IDX-th inter-strip gap)."""
        if not self.blue_curtains_enabled or len(self._strip_base_x) <= self.STRIP_GAP_IDX + 1:
            return float(self.curtain_center_x)
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

    def _set_cup_color(self, rgba):
        if getattr(self, "cup", None) is None:
            return
        for comp in self.cup.actor.get_components():
            if isinstance(comp, sapien.render.RenderBodyComponent):
                for shape in comp.render_shapes:
                    try:
                        shape.material.set_base_color(rgba)
                    except Exception:
                        pass

    def _actor_world_size(self, actor):
        if not hasattr(actor, "config") or actor.config is None:
            return np.array([0.0, 0.0, 0.0], dtype=np.float64)
        scale = actor.config.get("scale", [1.0, 1.0, 1.0])
        if isinstance(scale, (int, float)):
            scale = [float(scale)] * 3
        extents = np.array(actor.config.get("extents", [0.0, 0.0, 0.0]), dtype=np.float64)
        local_half = 0.5 * extents * np.array(scale, dtype=np.float64)
        rot = actor.get_pose().to_transformation_matrix()[:3, :3]
        world_half = np.abs(rot) @ local_half
        return 2.0 * world_half

    def _apply_kinematics(self, initial=False):
        """Push current step-driven targets onto the kinematic bodies."""
        if self.blue_curtains_enabled and self.blue_curtain_dynamic_enabled:
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
        if self._curtain_hit or not self._attempt_active or not self.blue_curtains_enabled:
            return
        for i in range(len(self.curtain_strips)):
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
        if not hasattr(self, "_kin_step") or getattr(self, "belt_plate", None) is None:
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
        gap_ok = ((not self.blue_curtains_enabled)
                  or abs(gap - self.curtain_center_x) < self.gap_tol)  # gap is over the corridor
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

    def _find_cup_grasp(self, arm_tag: ArmTag):
        # Try the cup's side contact points first; fall back to the generic grasp search.
        candidates = [
            (0, 0.10),
            (1, 0.10),
            (0, 0.08),
            (1, 0.08),
            (None, 0.10),
            (2, 0.10),
            (3, 0.10),
        ]
        for contact_point_id, pre_grasp_dis in candidates:
            pre_g, g = self.choose_grasp_pose(
                self.cup,
                arm_tag=arm_tag,
                pre_dis=pre_grasp_dis,
                target_dis=0.0,
                contact_point_id=contact_point_id,
            )
            if pre_g is not None and g is not None:
                return contact_point_id, pre_grasp_dis
        return None, None

    def _placement_target_x(self, t=None):
        if self.place_cup_in_middle_of_yellow_tools:
            return float(self.slot_x(t))
        return float(self.curtain_center_x)

    # ------------------------------------------------------------- policy
    def _dbg(self, tag):
        import os
        if os.environ.get("CCS_DEBUG"):
            print(f"[CCS] {tag}: plan_success={self.plan_success}", flush=True)

    def play_once(self):
        arm_tag = ArmTag("right")     # single-arm, right (cup spawns on the right half)

        # 1) Pick up the blue cup, lift it, and align with the corridor that bisects the moving
        #    gap between the two yellow shapes.
        grasp_contact_id, pre_grasp_dis = self._find_cup_grasp(arm_tag)
        if pre_grasp_dis is None:
            self.plan_success = False
            self.info["info"] = {
                "{A}": f"021_cup/base{self.cup_id}",
                "{a}": str(arm_tag),
            }
            return self.info
        self.move(self.close_gripper(arm_tag, pos=0.6))
        self.move(
            self.grasp_actor(
                self.cup,
                arm_tag=arm_tag,
                pre_grasp_dis=pre_grasp_dis,
                gripper_pos=0.0,
                contact_point_id=grasp_contact_id,
            )
        )
        self._dbg("after_grasp")
        if not self.plan_success:
            self.info["info"] = {
                "{A}": f"021_cup/base{self.cup_id}",
                "{a}": str(arm_tag),
            }
            return self.info
        self._dwell(6)
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.10, move_axis="arm"))
        self._dbg("after_lift")
        align_dx = float(self.curtain_center_x - self.cup.get_pose().p[0])
        if abs(align_dx) > 1e-4:
            self.move(self.move_by_displacement(arm_tag=arm_tag, x=align_dx))
            self._dbg("after_xalign")
        self._attempt_active = True            # from here a curtain collision fails the attempt

        # 2) Wait until the moving gap between the yellow shapes will arrive under that corridor
        #    by the time the forward reach completes. If the blue curtain exists, its gap must be
        #    open at the same time.
        self._dwell(8)                         # observe the dynamics a moment
        self._wait_for_window(lead_steps=self.REACH_STEPS_EST, max_steps=1500)
        self._reach_step = self._kin_step

        # 3) Reach to the gray surface, track the live yellow-shape gap, then lock that gap in
        #    place and fine-center before lowering the cup between the shapes.
        dy = self.belt_y - self.cup_y          # reach from the near cup row to the belt row
        self.move(self.move_by_displacement(arm_tag=arm_tag, y=dy))
        self._dbg("after_reach")
        target_x_live = self._placement_target_x()
        dx_live = float(np.clip(target_x_live - self.cup.get_pose().p[0], -0.05, 0.05))
        if abs(dx_live) > 1e-4:
            self.move(self.move_by_displacement(arm_tag=arm_tag, x=dx_live))
            self._dbg("after_gap_track")
        # self._freeze_belt()                    # lock the current yellow-shape gap for release
        target_x_lock = self._placement_target_x()
        dx_lock = float(np.clip(target_x_lock - self.cup.get_pose().p[0], -0.03, 0.03))
        if abs(dx_lock) > 1e-4:
            self.move(self.move_by_displacement(arm_tag=arm_tag, x=dx_lock))
        self._dbg("after_xcorrect")
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=-0.09))
        self._dbg("after_lower")
        self._deposit_step = self._kin_step
        self._slot_x_at_deposit = self.slot_x()
        self.move(self.open_gripper(arm_tag))

        # let it settle (curtain keeps swaying; records frames)
        self._dwell(15)

        # 4) lift off after placing, then optionally return home.
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=self.post_place_lift_z, move_axis="arm"))
        self._dbg("after_retract")
        if not self.lift_off_only_after_place:
            self.move(self.back_to_origin(arm_tag))

        self.info["info"] = {
            "{A}": f"021_cup/base{self.cup_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    def play_once_bk(self):
        return self.play_once()

    # ------------------------------------------------------------- metric
    def _center_offset(self):
        """Horizontal distance from the cup bottom to the gap center between yellow shapes."""
        cup_p = np.array(self.cup.get_functional_point(0, "pose").p[:2])
        slot_p = np.array([self.slot_x(), self.belt_y])
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
            "belt_motion_limit": float(getattr(self, "belt_motion_limit", 0.0)),
            "window_open": bool(self._window_open()),
            "curtain_hit": bool(self._curtain_hit),
            "placement_score": float(self.placement_score()),
            "blue_curtains_enabled": bool(self.blue_curtains_enabled),
            "blue_curtain_dynamic_enabled": bool(getattr(self, "blue_curtain_dynamic_enabled", False)),
            "moving_component_spacing": float(self.slot_spacing),
            "place_cup_in_middle_of_yellow_tools": bool(self.place_cup_in_middle_of_yellow_tools),
            "lift_off_only_after_place": bool(self.lift_off_only_after_place),
        }
        return obs
