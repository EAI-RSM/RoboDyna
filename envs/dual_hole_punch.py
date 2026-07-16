from ._base_task import Base_Task
from .utils import *
from .utils.action import Action, ArmTag
from ._GLOBAL_CONFIGS import *
import os
import sapien
import sapien.render
import numpy as np


class dual_hole_punch(Base_Task):
    """Dual hole-punch on two independent belts, serviced simultaneously by both arms.

    Two parallel belts (LEFT belt entirely in the left zone, RIGHT belt entirely in the
    right zone) each carry a row of square punch cards toward a fixed gantry-mounted
    punch head. The card placement along each belt can be configured as equal-spacing
    or variable-spacing. The two belts always share the same square pattern, so matching
    indexed squares reach the stamp line together. Optionally, one tile can be removed
    from a random side; at that stop only the side with a square should press.
    Each belt has its own button placed in its arm's near reach. The left arm presses the
    left button to fire the left punch head; the right arm presses the right button to fire
    the right punch head.

    The punch heads are NOT on the arms -- they are fixed gantry actuators that descend on
    their own button press. Each arm only presses its side's button. If both squares align
    together both arms press simultaneously; otherwise only the ready side presses.

    A press counts toward whichever page is currently under that side's punch head; the
    punch offset is the |page_center_x - punch_x| at the instant of the press. Per belt we
    score mean clamp(1 - offset/tol, 0, 1) over the punched pages.
    """

    # ---- belt geometry (per side; mirrored across x=0) ---------------------
    N_PAGES_DEFAULT = 4                 # moving squares per belt
    PUNCH_TOL_DEFAULT = 0.035           # offset tolerance (m) for full score
    BELT_HALF = (0.13, 0.05, 0.008)     # legacy reference half-extents (x,y,z)
    PAGE_HALF = (0.022, 0.022, 0.004)   # square punch-card half-extents
    BUTTON_HALF = (0.018, 0.018, 0.016) # button box half-extents (graspable / pressable)
    PUNCH_HALF = (0.02, 0.02, 0.05)     # gantry punch-head half-extents
    PUNCH_MODEL = "100_seal"
    PUNCH_MODEL_IDS = (2, 3, 4)
    PUNCH_Q = [0.5, 0.5, 0.5, 0.5]
    BELT_INWARD_REACH_DEFAULT = 0.13
    BELT_OUTWARD_SCALE_DEFAULT = 2.0
    SQUARE_START_MARGIN_DEFAULT = 0.035
    SQUARE_GAP_DEFAULT = 0.015
    SQUARE_GAP_MIN_DEFAULT = 0.010
    SQUARE_GAP_MAX_DEFAULT = 0.030
    BELT_EDGE_MARGIN_DEFAULT = 0.025
    SQUARE_PLACEMENT_MODE_DEFAULT = "equal"
    MISSING_TILE_MODE_DEFAULT = "none"
    BELT_CONTINOUS_MOTION_DEFAULT = False
    BELT_SPEED_MIN_DEFAULT = 0.0016
    BELT_SPEED_MAX_DEFAULT = 0.0028

    # belt center y (toward the robot's working area) and surface z above table
    BELT_Y = -0.05
    SURF_DZ = 0.016                     # page sits this high above belt-slab center top
    PUNCH_ANIM_STEPS = 24
    TILE_PAUSE_STEPS_DEFAULT = PUNCH_ANIM_STEPS + 4
    PAGE_EXIT_MARGIN = 0.002
    HIDE_Z = -10.0
    PUNCH_REST_Z_EXTRA = 0.03

    def setup_demo(self, **kwags):
        # capture task-scoped params from the (general) config's task_args block
        self._cfg = kwags.get("task_args", {}).get("dual_hole_punch", {})
        super()._init_task_env_(**kwags)

    def _normalize_tile_spacing_mode(self):
        raw_mode = self._cfg.get("equal_tile_spacing", None)
        if raw_mode is None:
            raw_mode = self._cfg.get(
                "placement_mode",
                self._cfg.get("square_placement_mode", self.SQUARE_PLACEMENT_MODE_DEFAULT),
            )
        if isinstance(raw_mode, bool):
            return "equal" if raw_mode else "variable"
        if isinstance(raw_mode, (int, float)):
            return "equal" if int(raw_mode) != 0 else "variable"
        mode = str(raw_mode).strip().lower()
        if mode in ("equal", "fixed", "constant", "opt1"):
            return "equal"
        return "variable"

    def _normalize_missing_tile_mode(self):
        raw_mode = self._cfg.get("missing_tile_mode", self.MISSING_TILE_MODE_DEFAULT)
        if isinstance(raw_mode, bool):
            return "single_random" if raw_mode else "none"
        if isinstance(raw_mode, (int, float)):
            return "single_random" if int(raw_mode) != 0 else "none"
        mode = str(raw_mode).strip().lower()
        if mode in ("single_random", "single", "one_missing", "random_one", "enabled", "on", "true"):
            return "single_random"
        return "none"

    def _normalize_belt_continous_motion(self):
        raw_mode = self._cfg.get(
            "belt_continous_motion",
            self._cfg.get("belt_motion_mode", self.BELT_CONTINOUS_MOTION_DEFAULT),
        )
        if isinstance(raw_mode, bool):
            return raw_mode
        if isinstance(raw_mode, (int, float)):
            return int(raw_mode) != 0
        mode = str(raw_mode).strip().lower()
        if mode in ("continuous", "moving", "run", "stream", "always_on", "true", "on", "yes"):
            return True
        if mode in ("stop_per_tile", "stepwise", "paused", "false", "off", "no"):
            return False
        return bool(self.BELT_CONTINOUS_MOTION_DEFAULT)

    def _get_tile_pause_steps(self):
        return max(1, int(self._cfg.get("tile_pause_steps", self.TILE_PAUSE_STEPS_DEFAULT)))

    def _sample_start_margin(self):
        base_margin = max(
            float(self._cfg.get("square_start_margin", self.SQUARE_START_MARGIN_DEFAULT)),
            self.PAGE_HALF[0] + 0.004,
        )
        return base_margin

    def _sample_square_layout(self, placement_mode, start_margin):
        start_margin = max(float(start_margin), self.PAGE_HALF[0] + 0.004)
        fixed_gap = max(0.0, float(self._cfg.get("square_gap", self.SQUARE_GAP_DEFAULT)))
        gap_min = max(0.0, float(self._cfg.get("square_gap_min", self.SQUARE_GAP_MIN_DEFAULT)))
        gap_max = max(gap_min, float(self._cfg.get("square_gap_max", self.SQUARE_GAP_MAX_DEFAULT)))

        offsets = []
        gaps = []
        offset = start_margin
        center_step_base = 2.0 * self.PAGE_HALF[0]
        for k in range(self.n_pages):
            offsets.append(float(offset))
            if k == self.n_pages - 1:
                break
            gap = fixed_gap if placement_mode == "equal" else float(np.random.uniform(gap_min, gap_max))
            gap = max(0.0, float(gap))
            gaps.append(gap)
            offset += center_step_base + gap
        return offsets, gaps

    def _build_square_layouts(self):
        self.tile_spacing_mode = self._normalize_tile_spacing_mode()
        self.square_offsets = {}
        self.square_gaps = {}
        start_margin = self._sample_start_margin()
        offsets, gaps = self._sample_square_layout(
            self.tile_spacing_mode,
            start_margin=start_margin,
        )
        for side in ("left", "right"):
            self.square_offsets[side] = list(offsets)
            self.square_gaps[side] = list(gaps)

    def _sample_missing_tile(self):
        self.missing_tile_mode = self._normalize_missing_tile_mode()
        self.missing_tile_side = None
        self.missing_tile_index = None
        self.page_missing = {side: [False] * self.n_pages for side in ("left", "right")}
        if self.missing_tile_mode != "single_random" or self.n_pages <= 0:
            return
        self.missing_tile_side = str(np.random.choice(["left", "right"]))
        self.missing_tile_index = int(np.random.randint(0, self.n_pages))
        self.page_missing[self.missing_tile_side][self.missing_tile_index] = True

    def _build_arrival_schedule(self, side, speed):
        speed = max(float(speed), 1e-8)
        arrival_eff_steps = []
        aligned_offsets = []
        prev_eff = None
        prev_offset = None
        for offset in self.square_offsets[side]:
            if prev_eff is None:
                eff = max(0, int(round(offset / speed)))
            else:
                gap_eff = max(1, int(round((offset - prev_offset) / speed)))
                eff = prev_eff + gap_eff
            arrival_eff_steps.append(eff)
            aligned_offsets.append(float(eff * speed))
            prev_eff = eff
            prev_offset = offset
        return arrival_eff_steps, aligned_offsets

    def _belt_reaches_for_side(self, side):
        inward_reach = float(
            self._cfg.get("belt_inward_reach", self.BELT_INWARD_REACH_DEFAULT)
        )
        outward_reach_cfg = self._cfg.get("belt_outward_reach", None)
        if outward_reach_cfg is None:
            outward_reach = inward_reach * float(
                self._cfg.get("belt_outward_scale", self.BELT_OUTWARD_SCALE_DEFAULT)
            )
        else:
            outward_reach = float(outward_reach_cfg)
        edge_margin = float(self._cfg.get("belt_edge_margin", self.BELT_EDGE_MARGIN_DEFAULT))
        required_offsets = getattr(self, "aligned_square_offsets", {}).get(
            side,
            self.square_offsets[side],
        )
        required_outward = required_offsets[-1] + self.PAGE_HALF[0] + edge_margin
        outward_reach = max(outward_reach, required_outward)
        return float(inward_reach), float(outward_reach)

    def _sample_belt_timing(self):
        belt_speed_cfg = self._cfg.get("belt_speed", None)
        if belt_speed_cfg is not None:
            speed = max(1e-8, float(belt_speed_cfg))
        else:
            speed_min = float(self._cfg.get("belt_speed_min", self.BELT_SPEED_MIN_DEFAULT))
            speed_max = float(self._cfg.get("belt_speed_max", self.BELT_SPEED_MAX_DEFAULT))
            speed_lo = min(speed_min, speed_max)
            speed_hi = max(speed_min, speed_max)
            speed = float(np.random.uniform(speed_lo, speed_hi))
        phase = int(np.random.randint(0, 60))             # start delay (steps)
        return speed, phase

    # ------------------------------------------------------------ actors
    def load_actors(self):
        self.n_pages = max(1, int(self._cfg.get("n_pages", self.N_PAGES_DEFAULT)))
        self.punch_tol = float(self._cfg.get("punch_tol", self.PUNCH_TOL_DEFAULT))
        self._build_square_layouts()
        self._sample_missing_tile()

        z0 = 0.74 + self.table_z_bias       # table top surface z

        # Side sign: left=-1, right=+1. Belt timing and square layouts are shared, so
        # matching indexed squares stay synchronized between the two sides.
        self._sides = {}
        self.belt = {}
        self.punch_head = {}
        self.button = {}
        self.pages = {}           # side -> list[Actor]
        self.page_target_x = {}   # side -> list[float]  (target punch x for each page)
        self.page_start_x = {}    # side -> list[float]
        self.page_punched = {}    # side -> list[bool]
        self.page_offset = {}     # side -> list[float]  (recorded offset at punch)
        self.belt_speed = {}      # side -> px/step
        self.belt_phase = {}      # side -> int start-delay steps
        self._punch_x = {}        # side -> punch gantry x (world)
        self._punch_rest_z = {}   # side -> punch head rest z
        self._punch_y = {}        # side -> punch/belt y
        self._punch_press = {}    # side -> remaining descend frames (visual)
        self.belt_inner_reach = {}
        self.belt_outward_reach = {}
        self.belt_bounds_x = {}
        self.belt_inner_edge_x = {}
        self.page_arrival_eff = {}
        self.aligned_square_offsets = {}
        self.page_hidden = {}
        self.page_missed = {}
        self.belt_continous_motion = self._normalize_belt_continous_motion()
        self.tile_pause_steps = self._get_tile_pause_steps()
        self.invalid_empty_press = False
        self.invalid_empty_press_count = 0
        self.invalid_empty_press_sides = []
        self.punch_model_id = int(np.random.choice(self.PUNCH_MODEL_IDS))
        shared_belt_timing = self._sample_belt_timing()

        for sign, side in ((-1.0, "left"), (1.0, "right")):
            punch_x = sign * 0.20
            speed, phase = shared_belt_timing
            self.belt_speed[side] = speed
            self.belt_phase[side] = phase
            arrival_eff_steps, aligned_offsets = self._build_arrival_schedule(side, speed)
            self.page_arrival_eff[side] = arrival_eff_steps
            self.aligned_square_offsets[side] = aligned_offsets
            inward_reach, outward_reach = self._belt_reaches_for_side(side)
            inner_edge_x = punch_x - sign * inward_reach
            outer_edge_x = punch_x + sign * outward_reach
            belt_cx = 0.5 * (inner_edge_x + outer_edge_x)
            belt_half_x = 0.5 * abs(outer_edge_x - inner_edge_x)
            self.belt_inner_reach[side] = inward_reach
            self.belt_outward_reach[side] = outward_reach
            self.belt_bounds_x[side] = (float(min(inner_edge_x, outer_edge_x)), float(max(inner_edge_x, outer_edge_x)))
            self.belt_inner_edge_x[side] = float(inner_edge_x)
            self._punch_y[side] = self.BELT_Y

            # belt slab (static scenery)
            belt = create_box(
                self,
                pose=sapien.Pose([belt_cx, self.BELT_Y, z0 + self.BELT_HALF[2]], [1, 0, 0, 0]),
                half_size=(belt_half_x, self.BELT_HALF[1], self.BELT_HALF[2]),
                color=(0.15, 0.15, 0.18),
                name=f"belt_{side}",
                is_static=True,
            )
            self.belt[side] = belt
            belt_top_z = z0 + 2 * self.BELT_HALF[2]

            # punch head: fixed gantry actuator near the inner working area, while the longer
            # outward belt segment holds the queued cards before they travel inward to the head.
            self._punch_x[side] = punch_x
            punch_rest_z = belt_top_z + self.PAGE_HALF[2] * 2 + self.PUNCH_HALF[2] + self.PUNCH_REST_Z_EXTRA
            self._punch_rest_z[side] = punch_rest_z
            head = create_actor(
                self,
                pose=sapien.Pose([punch_x, self.BELT_Y, punch_rest_z], self.PUNCH_Q),
                modelname=self.PUNCH_MODEL,
                model_id=self.punch_model_id,
                convex=True,
                is_static=False,
            )
            self._make_kinematic(head)
            head.set_name(f"punch_{side}")
            self.punch_head[side] = head
            self._punch_press[side] = 0

            # button: small pressable box in the arm's NEAR reach (outer, in front).
            button_x = sign * 0.26
            button = create_box(
                self,
                pose=sapien.Pose([button_x, self.BELT_Y + 0.13, z0 + self.BUTTON_HALF[2]], [1, 0, 0, 0]),
                half_size=self.BUTTON_HALF,
                color=(0.1, 0.6, 0.1),
                name=f"button_{side}",
                is_static=True,
            )
            self.button[side] = button

            # cards queue on the OUTER side of the punch head and march inward toward it. Page 0
            # is nearest the head (arrives first); each higher-index page uses the configured
            # placement interval pattern, so they reach the head in index order k = 0, 1, 2, ...
            pages = []
            tx_list = []
            sx_list = []
            for k, offset in enumerate(self.aligned_square_offsets[side]):
                sx = punch_x + sign * offset                          # further out = arrives later
                # per-page randomized target offset relative to punch center
                toff = float(np.random.uniform(-0.012, 0.012))
                tx = punch_x + toff
                missing = bool(self.page_missing[side][k])
                page_z = self.HIDE_Z if missing else belt_top_z + self.PAGE_HALF[2]
                page = create_box(
                    self,
                    pose=sapien.Pose(
                        [sx, self.BELT_Y, page_z],
                        [1, 0, 0, 0],
                    ),
                    half_size=self.PAGE_HALF,
                    color=(0.93, 0.93, 0.88),
                    name=f"page_{side}_{k}",
                    is_static=False,   # kinematic: scripted via set_pose, never falls
                )
                self._make_kinematic(page)
                pages.append(page)
                tx_list.append(tx)
                sx_list.append(sx)

            self.pages[side] = pages
            self.page_target_x[side] = tx_list
            self.page_start_x[side] = sx_list
            self.page_punched[side] = [bool(v) for v in self.page_missing[side]]
            self.page_missed[side] = [False] * self.n_pages
            self.page_offset[side] = [None] * self.n_pages
            self.page_hidden[side] = [bool(v) for v in self.page_missing[side]]
            self._sides[side] = sign

            # reserve space so clutter / randomizers stay clear
            self.add_prohibit_area(belt, padding=0.02)
            self.add_prohibit_area(button, padding=0.03)

        # belt simulation clock (shared step counter; each belt reads its own phase/speed)
        self._belt_step = 0
        self._belt_active = False
        self._belt_running = False   # only True inside the explicit dwell loops
        # which page index (if any) is currently under each punch head
        self._under_head = {"left": None, "right": None}

    # --------------------------------------------------- belt kinematics
    @staticmethod
    def _make_kinematic(actor):
        """Turn a dynamic rigid box into a kinematic body: it holds its pose under gravity
        and can be repositioned every step with set_pose (used for the moving pages and the
        descending gantry punch heads)."""
        for c in actor.actor.get_components():
            if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                try:
                    c.set_kinematic(True)
                except Exception:
                    pass

    def _page_x_at(self, side, k, step):
        """World x of page k of `side` at the given belt step. Pages enter from the OUTER end,
        march toward the punch head (belt center), pass under it, and continue off the inner
        end -- so once punched a page leaves the head zone and the next page becomes current."""
        sign = self._sides[side]
        eff = max(0, step - self.belt_phase[side])
        return self.page_start_x[side][k] - sign * self.belt_speed[side] * eff

    def _set_page_pose(self, side, k, x):
        p = self.pages[side][k]
        cur = p.get_pose()
        p.actor.set_pose(sapien.Pose([x, cur.p[1], cur.p[2]], cur.q))

    def _hide_page(self, side, k):
        if self.page_hidden[side][k]:
            return
        p = self.pages[side][k].get_pose()
        self.pages[side][k].actor.set_pose(sapien.Pose([p.p[0], p.p[1], self.HIDE_Z], p.q))
        self.page_hidden[side][k] = True

    def _move_with_belt_motion(self, action1, action2=None, advance_belts=False):
        prev = self._belt_running
        self._belt_running = bool(advance_belts)
        try:
            if action2 is not None:
                self.move(action1, action2)
            else:
                self.move(action1)
        finally:
            self._belt_running = prev

    def _page_has_exited_belt(self, side, x):
        sign = self._sides[side]
        inner_edge_x = self.belt_inner_edge_x[side]
        return sign * (x - inner_edge_x) <= -(self.PAGE_HALF[0] + self.PAGE_EXIT_MARGIN)

    def _refresh_pages_at_current_step(self):
        for side in ("left", "right"):
            best_k, best_d = None, 1e9
            for k in range(self.n_pages):
                if self.page_hidden[side][k]:
                    continue
                x = self._page_x_at(side, k, self._belt_step)
                if self.page_punched[side][k] and self._page_has_exited_belt(side, x):
                    self._hide_page(side, k)
                    continue
                self._set_page_pose(side, k, x)
                if self.page_punched[side][k]:
                    continue
                d = abs(x - self._punch_x[side])
                if d < best_d:
                    best_d, best_k = d, k
            self._under_head[side] = best_k if best_d < 0.05 else None

    def _update_punch_heads(self):
        for side in ("left", "right"):
            if self._punch_press[side] <= 0:
                continue
            self._punch_press[side] -= 1
            frac = 1.0 - self._punch_press[side] / max(1, self.PUNCH_ANIM_STEPS)
            tri = 1.0 - abs(2 * frac - 1.0)
            drop = tri * 0.045
            h = self.punch_head[side]
            hp = h.get_pose()
            h.actor.set_pose(sapien.Pose([hp.p[0], hp.p[1], self._punch_rest_z[side] - drop], hp.q))

    def _align_page_under_punch(self, side, k):
        """Snap page `k` so its center is exactly under the punch at the current held step."""
        x = self._page_x_at(side, k, self._belt_step)
        dx = self._punch_x[side] - x
        if abs(dx) > 0.0:
            self.page_start_x[side][k] += dx
            x = self._page_x_at(side, k, self._belt_step)
        self._set_page_pose(side, k, x)
        self._under_head[side] = k

    def _update_kinematic_tasks(self):
        # base hook (drives any DOMINO dynamic objects); runs EVERY physics step
        super()._update_kinematic_tasks()
        if not getattr(self, "_belt_active", False):
            return
        # The belt advances only when _belt_running is enabled. Stepwise mode enables it only
        # during explicit dwell loops, while continuous mode also enables it during press/release
        # arm motions so the tiles keep streaming past the punch heads.
        if not getattr(self, "_belt_running", False):
            # still refresh page poses so they render at their held position, but don't advance
            self._refresh_pages_at_current_step()
            self._update_punch_heads()
            return
        self._belt_step += 1
        self._refresh_pages_at_current_step()
        self._update_punch_heads()

    def _trigger_punch_head(self, side):
        self._punch_press[side] = self.PUNCH_ANIM_STEPS

    def _mark_invalid_empty_press(self, side):
        if self.missing_tile_mode == "none":
            return
        self.invalid_empty_press = True
        self.invalid_empty_press_count += 1
        self.invalid_empty_press_sides.append(str(side))
        if os.environ.get("DHP_DEBUG"):
            print(f"[dhp] INVALID_EMPTY_PRESS side={side} step={self._belt_step}", flush=True)

    def _mark_missed_page(self, side, k):
        if self.page_punched[side][k]:
            return
        self.page_missed[side][k] = True
        self.page_punched[side][k] = True
        if os.environ.get("DHP_DEBUG"):
            print(f"[dhp] MISS {side} page{k} step={self._belt_step}", flush=True)

    def _fire_punch(self, side, k=Ellipsis):
        """Register a punch on `side` at the current belt step: punch whichever page is
        under the head, recording its x-offset from the punch center, and trigger the
        head's visual descent."""
        self._trigger_punch_head(side)
        if k is Ellipsis:
            k = self._under_head[side]
        if k is None:
            self._mark_invalid_empty_press(side)
            return
        if not self._page_satisfies_stamp_criterion(side, k):
            self._mark_missed_page(side, k)
            return
        if self.page_punched[side][k]:
            return
        x = self._page_x_at(side, k, self._belt_step)
        off = abs(x - self.page_target_x[side][k])
        self.page_punched[side][k] = True
        self.page_offset[side][k] = float(off)
        if os.environ.get("DHP_DEBUG"):
            print(f"[dhp] FIRE {side} page{k} off={off:.4f} step={self._belt_step}", flush=True)
        # mark the punched page with a dark hole-ish recolor for visibility
        for c in self.pages[side][k].actor.get_components():
            if isinstance(c, sapien.render.RenderBodyComponent):
                for s in c.render_shapes:
                    try:
                        s.material.set_base_color([0.45, 0.42, 0.40, 1.0])
                    except Exception:
                        pass

    def _belt_idle(self, steps, advance_belts=True):
        """Dwell for `steps` physics steps while optionally advancing the belts."""
        prev = self._belt_running
        self._belt_running = bool(advance_belts)
        try:
            for i in range(steps):
                self._update_kinematic_tasks()
                self.scene.step()
                if self.save_freq and (i % self.save_freq == 0):
                    self._take_picture()
        finally:
            self._belt_running = prev

    # -------------------------------------------------------- press poses
    def _button_press_actions(self, side, descend):
        """Build a (arm_tag, [Action]) that moves the side's gripper straight down by
        `descend` onto its button (relative move along the world -z)."""
        arm = ArmTag(side)
        return self.move_by_displacement(arm_tag=arm, z=-descend)

    def _button_release_actions(self, side, ascend):
        arm = ArmTag(side)
        return self.move_by_displacement(arm_tag=arm, z=ascend)

    def _hover_button(self, side):
        """Move the side's gripper to hover just above its button (top-down)."""
        arm = ArmTag(side)
        # grasp_actor on the button's top-down contact point positions the (closed) gripper
        # right above the button without committing the press.
        return self.grasp_actor(self.button[side], arm_tag=arm, pre_grasp_dis=0.09,
                                grasp_dis=0.09, contact_point_id=0)

    # ------------------------------------------------------------- policy
    def _dbg(self, tag):
        if os.environ.get("DHP_DEBUG"):
            print(f"[dhp] {tag}: plan={self.plan_success} step={self._belt_step} "
                  f"L_pun={self.page_punched['left']} R_pun={self.page_punched['right']} "
                  f"under={self._under_head}", flush=True)

    def _page_arrival_step(self, side, k):
        """The belt step at which page k of `side` is centered on its punch head."""
        return int(self.page_arrival_eff[side][k]) + self.belt_phase[side]

    def _next_unpunched_page(self, side):
        for k in range(self.n_pages):
            if not self.page_punched[side][k]:
                return k
        return None

    def _page_is_under_stamp(self, side, k):
        x = self._page_x_at(side, k, self._belt_step)
        tol = max(1e-6, 0.25 * self.belt_speed[side])
        return abs(x - self._punch_x[side]) <= tol

    @staticmethod
    def _interval_overlap_length(center_a, half_a, center_b, half_b):
        lo = max(center_a - half_a, center_b - half_b)
        hi = min(center_a + half_a, center_b + half_b)
        return max(0.0, hi - lo)

    def _page_stamp_overlap_ratio(self, side, k):
        page_x = self._page_x_at(side, k, self._belt_step)
        punch_x = self._punch_x[side]
        x_overlap = self._interval_overlap_length(
            page_x,
            self.PAGE_HALF[0],
            punch_x,
            self.PUNCH_HALF[0],
        )
        y_overlap = self._interval_overlap_length(
            self.BELT_Y,
            self.PAGE_HALF[1],
            self._punch_y[side],
            self.PUNCH_HALF[1],
        )
        page_area = (2.0 * self.PAGE_HALF[0]) * (2.0 * self.PAGE_HALF[1])
        if page_area <= 1e-8:
            return 0.0
        return float((x_overlap * y_overlap) / page_area)

    def _page_satisfies_stamp_criterion(self, side, k):
        if self.belt_continous_motion:
            return self._page_stamp_overlap_ratio(side, k) > 0.5
        return self._page_is_under_stamp(side, k)

    def _ready_pages_at_current_step(self):
        ready_by_side = {}
        for side in ("left", "right"):
            k = self._next_unpunched_page(side)
            if k is None:
                continue
            if self._page_is_under_stamp(side, k):
                ready_by_side[side] = k
                self._under_head[side] = k
        return ready_by_side

    def _final_belt_runout_steps(self):
        travel_steps = []
        for side in ("left", "right"):
            speed = max(self.belt_speed[side], 1e-8)
            travel = self.belt_inner_reach[side] + self.PAGE_HALF[0] + self.PAGE_EXIT_MARGIN
            travel_steps.append(int(np.ceil(travel / speed)))
        return max(40, max(travel_steps) + 8)

    def _pages_arriving_at_step(self, step_target):
        pages_by_side = {}
        for side in ("left", "right"):
            k = self._next_unpunched_page(side)
            if k is None:
                continue
            if self._page_arrival_step(side, k) == step_target:
                pages_by_side[side] = k
        return pages_by_side

    def _mark_missed_ready_pages(self, ready_by_side):
        for side, k in ready_by_side.items():
            self._mark_missed_page(side, k)

    def _mark_overdue_pages(self):
        marked_any = False
        for side in ("left", "right"):
            k = self._next_unpunched_page(side)
            if k is None:
                continue
            if self._belt_step > self._page_arrival_step(side, k) and not self._page_satisfies_stamp_criterion(side, k):
                self._mark_missed_page(side, k)
                marked_any = True
        return marked_any

    def _build_press_plan(self, ready_by_side, descend=0.05):
        pressed_sides = [side for side in ("left", "right") if side in ready_by_side]
        if not pressed_sides:
            return [], None, None
        if len(pressed_sides) == 2:
            return (
                pressed_sides,
                self._button_press_actions("left", descend),
                self._button_press_actions("right", descend),
            )
        return pressed_sides, self._button_press_actions(pressed_sides[0], descend), None

    def _estimate_press_duration(self, ready_by_side, descend=0.05):
        pressed_sides, action1, action2 = self._build_press_plan(ready_by_side, descend=descend)
        if not pressed_sides:
            return 0, pressed_sides, action1, action2
        duration = self.calculate_move_duration(action1, action2)
        return int(duration), pressed_sides, action1, action2

    def _press_ready_sides(self, ready_by_side, descend=0.05, advance_belts=False):
        pressed_sides, action1, action2 = self._build_press_plan(ready_by_side, descend=descend)
        if not pressed_sides:
            return []
        self._move_with_belt_motion(action1, action2, advance_belts=advance_belts)
        for side in pressed_sides:
            self._fire_punch(side, ready_by_side[side])
        return pressed_sides

    def _release_pressed_sides(self, pressed_sides, ascend=0.05, advance_belts=False):
        if not pressed_sides:
            return
        if len(pressed_sides) == 2:
            self._move_with_belt_motion(
                self._button_release_actions("left", ascend),
                self._button_release_actions("right", ascend),
                advance_belts=advance_belts,
            )
        else:
            self._move_with_belt_motion(
                self._button_release_actions(pressed_sides[0], ascend),
                advance_belts=advance_belts,
            )

    def play_once(self):
        # 1) Bring both grippers to hover over their buttons simultaneously (truly dual-arm).
        continuous_motion = self.belt_continous_motion
        self._move_with_belt_motion(
            self._hover_button("left"),
            self._hover_button("right"),
            advance_belts=False,
        )
        self._dbg("after hover")

        # start the belts running
        self._belt_active = True

        # 2) Stepwise mode stops on each tile until the required key press happens.
        #    Continuous mode keeps the belt moving and times the press so the punch lands
        #    while the tile still sufficiently overlaps the stamp head.
        while not all(all(self.page_punched[s]) for s in ("left", "right")):
            if continuous_motion and self._mark_overdue_pages():
                continue
            next_steps = []
            for side in ("left", "right"):
                k = self._next_unpunched_page(side)
                if k is not None:
                    next_steps.append(self._page_arrival_step(side, k))
            if not next_steps:
                break
            step_target = min(next_steps)
            if continuous_motion:
                target_by_side = self._pages_arriving_at_step(step_target)
                if not target_by_side:
                    if self._mark_overdue_pages():
                        continue
                    break
                press_duration, _, _, _ = self._estimate_press_duration(target_by_side, descend=0.05)
                press_start_step = max(self._belt_step, step_target - press_duration)
                self._run_belts_to(press_start_step)
                pressed_sides = self._press_ready_sides(
                    target_by_side,
                    descend=0.05,
                    advance_belts=True,
                )
                self._dbg(f"after press @step={step_target} sides={pressed_sides}")
                self._release_pressed_sides(pressed_sides, ascend=0.05, advance_belts=True)
                continue

            self._run_belts_to(step_target)
            ready_by_side = self._ready_pages_at_current_step()
            if not ready_by_side:
                continue
            pressed_sides = self._press_ready_sides(ready_by_side, descend=0.05, advance_belts=False)
            self._mark_missed_ready_pages(ready_by_side)
            self._dbg(f"after press @step={step_target} sides={pressed_sides}")
            self._release_pressed_sides(pressed_sides, ascend=0.05, advance_belts=False)

        # let stamped squares clear the inner belt edge and disappear before ending the episode
        self._belt_idle(self._final_belt_runout_steps(), advance_belts=True)
        self._belt_active = False

        self.info["info"] = {"{a}": "left", "{b}": "right"}
        return self.info

    def _run_belts_to(self, step_target, max_extra=400):
        """Advance the belts until self._belt_step reaches step_target, recording frames."""
        prev = self._belt_running
        self._belt_running = True
        try:
            guard = 0
            while self._belt_step < step_target and guard < (step_target + max_extra):
                self._update_kinematic_tasks()
                self.scene.step()
                if self.save_freq and (guard % self.save_freq == 0):
                    self._take_picture()
                guard += 1
        finally:
            self._belt_running = prev

    # ------------------------------------------------------------- scoring
    def _side_score(self, side):
        offs = [o for o in self.page_offset[side] if o is not None]
        if not offs:
            return 0.0
        return float(np.mean([np.clip(1.0 - o / self.punch_tol, 0.0, 1.0) for o in offs]))

    def check_success(self):
        all_punched = all(all(self.page_punched[s]) for s in ("left", "right"))
        any_missed = any(any(self.page_missed[s]) for s in ("left", "right"))
        self.punch_score_L = self._side_score("left")
        self.punch_score_R = self._side_score("right")
        self.punch_score_mean = 0.5 * (self.punch_score_L + self.punch_score_R)
        return bool(all_punched and not any_missed and not self.invalid_empty_press)

    # record per-belt punch state into the trajectory (per frame)
    def get_obs(self):
        obs = super().get_obs()
        obs["hole_punch"] = {
            "left_punched": [bool(b) for b in self.page_punched["left"]],
            "right_punched": [bool(b) for b in self.page_punched["right"]],
            "left_offsets": [None if o is None else float(o) for o in self.page_offset["left"]],
            "right_offsets": [None if o is None else float(o) for o in self.page_offset["right"]],
            "equal_tile_spacing": self.tile_spacing_mode == "equal",
            "tile_spacing_mode": self.tile_spacing_mode,
            "placement_mode": self.tile_spacing_mode,
            "square_placement_mode": self.tile_spacing_mode,
            "missing_tile_mode": self.missing_tile_mode,
            "missing_tile_side": self.missing_tile_side,
            "missing_tile_index": self.missing_tile_index,
            "belt_continous_motion": bool(self.belt_continous_motion),
            "left_missing_tiles": [bool(v) for v in self.page_missing["left"]],
            "right_missing_tiles": [bool(v) for v in self.page_missing["right"]],
            "tile_pause_steps": int(self.tile_pause_steps),
            "left_missed_tiles": [bool(v) for v in self.page_missed["left"]],
            "right_missed_tiles": [bool(v) for v in self.page_missed["right"]],
            "invalid_empty_press": bool(self.invalid_empty_press),
            "invalid_empty_press_count": int(self.invalid_empty_press_count),
            "invalid_empty_press_sides": list(self.invalid_empty_press_sides),
            "square_gaps": [float(g) for g in self.square_gaps["left"]],
            "left_square_offsets": [float(o) for o in self.square_offsets["left"]],
            "right_square_offsets": [float(o) for o in self.square_offsets["right"]],
            "left_square_gaps": [float(g) for g in self.square_gaps["left"]],
            "right_square_gaps": [float(g) for g in self.square_gaps["right"]],
            "left_belt_outward_reach": float(self.belt_outward_reach["left"]),
            "right_belt_outward_reach": float(self.belt_outward_reach["right"]),
            "belt_speed_left": float(self.belt_speed["left"]),
            "belt_speed_right": float(self.belt_speed["right"]),
            "punch_score_L": float(self._side_score("left")),
            "punch_score_R": float(self._side_score("right")),
            "punch_tol": float(self.punch_tol),
        }
        return obs
