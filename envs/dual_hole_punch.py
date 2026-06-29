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
    right zone) each carry a row of paper pages toward a fixed gantry-mounted punch head.
    Each belt has its own button placed in its arm's near reach. The left arm presses the
    left button to fire the left punch head; the right arm presses the right button to fire
    the right punch head. The two belts have independent (randomized) speeds, page spacings
    and per-page target offsets, and run AT THE SAME TIME.

    The punch heads are NOT on the arms -- they are fixed gantry actuators that descend on
    their own button press. Each arm only presses its side's button. Both presses are issued
    together via self.move(left_action, right_action) so the two belts are serviced truly
    simultaneously.

    A press counts toward whichever page is currently under that side's punch head; the
    punch offset is the |page_center_x - punch_x| at the instant of the press. Per belt we
    score mean clamp(1 - offset/tol, 0, 1) over the punched pages.
    """

    # ---- belt geometry (per side; mirrored across x=0) ---------------------
    N_PAGES_DEFAULT = 3                 # pages per belt
    PUNCH_TOL_DEFAULT = 0.035           # offset tolerance (m) for full score
    BELT_HALF = (0.13, 0.05, 0.008)     # belt slab half-extents (x,y,z)
    PAGE_HALF = (0.022, 0.028, 0.004)   # page half-extents
    BUTTON_HALF = (0.018, 0.018, 0.016) # button box half-extents (graspable / pressable)
    PUNCH_HALF = (0.02, 0.02, 0.05)     # gantry punch-head half-extents

    # belt center y (toward the robot's working area) and surface z above table
    BELT_Y = -0.05
    SURF_DZ = 0.016                     # page sits this high above belt-slab center top

    def setup_demo(self, **kwags):
        # capture task-scoped params from the (general) config's task_args block
        self._cfg = kwags.get("task_args", {}).get("dual_hole_punch", {})
        super()._init_task_env_(**kwags)

    # ------------------------------------------------------------ actors
    def load_actors(self):
        self.n_pages = int(self._cfg.get("n_pages", self.N_PAGES_DEFAULT))
        self.punch_tol = float(self._cfg.get("punch_tol", self.PUNCH_TOL_DEFAULT))

        z0 = 0.74 + self.table_z_bias       # table top surface z

        # Per-side independent randomization: belt speed (m/step via px/step),
        # page spacing, and per-page target offset. Side sign: left=-1, right=+1.
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

        for sign, side in ((-1.0, "left"), (1.0, "right")):
            # belt slab is centered in the side's zone; gantry/punch at the inner end so the
            # button (outer) is in the arm's comfortable near reach.
            belt_cx = sign * 0.20
            self._punch_y[side] = self.BELT_Y

            # belt slab (static scenery)
            belt = create_box(
                self,
                pose=sapien.Pose([belt_cx, self.BELT_Y, z0 + self.BELT_HALF[2]], [1, 0, 0, 0]),
                half_size=self.BELT_HALF,
                color=(0.15, 0.15, 0.18),
                name=f"belt_{side}",
                is_static=True,
            )
            self.belt[side] = belt
            belt_top_z = z0 + 2 * self.BELT_HALF[2]

            # punch head: fixed gantry actuator at the INNER end of the belt (toward center),
            # so the page travels outward-to-inner OR the punch is mid-belt. We place the punch
            # near the belt center and feed pages from the inner end toward the outer end is
            # awkward; instead pages start at the inner end and move outward to the punch which
            # sits at the OUTER-middle, keeping the button reachable. Simpler & reliable: punch
            # at the belt center x.
            punch_x = belt_cx
            self._punch_x[side] = punch_x
            punch_rest_z = belt_top_z + self.PAGE_HALF[2] * 2 + self.PUNCH_HALF[2] + 0.04
            self._punch_rest_z[side] = punch_rest_z
            head = create_box(
                self,
                pose=sapien.Pose([punch_x, self.BELT_Y, punch_rest_z], [1, 0, 0, 0]),
                half_size=self.PUNCH_HALF,
                color=(0.55, 0.05, 0.05),
                name=f"punch_{side}",
                is_static=False,
            )
            self._make_kinematic(head)
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

            # independent belt dynamics
            speed = float(np.random.uniform(0.0016, 0.0028))    # m per physics step
            phase = int(np.random.randint(0, 60))               # start delay (steps)
            spacing = float(np.random.uniform(0.060, 0.080))    # m between pages
            start_margin = 0.035                                # page0 sits just outboard of punch
            self.belt_speed[side] = speed
            self.belt_phase[side] = phase

            # pages queue on the OUTER side of the punch head and march inward toward it. Page 0
            # is nearest the head (arrives first); each higher-index page starts one spacing
            # further out, so they reach the head in index order k = 0, 1, 2, ...
            pages = []
            tx_list = []
            sx_list = []
            for k in range(self.n_pages):
                sx = punch_x + sign * (start_margin + k * spacing)   # further out = arrives later
                # per-page randomized target offset relative to punch center
                toff = float(np.random.uniform(-0.012, 0.012))
                tx = punch_x + toff
                page = create_box(
                    self,
                    pose=sapien.Pose(
                        [sx, self.BELT_Y, belt_top_z + self.PAGE_HALF[2]],
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
            self.page_punched[side] = [False] * self.n_pages
            self.page_offset[side] = [None] * self.n_pages
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

    def _update_kinematic_tasks(self):
        # base hook (drives any DOMINO dynamic objects); runs EVERY physics step
        super()._update_kinematic_tasks()
        if not getattr(self, "_belt_active", False):
            return
        # The belt advances only during explicit dwell loops (_run_belts_to / _belt_idle), NOT
        # during the arm-motion planning steps -- otherwise an arm move (hundreds of sim steps)
        # would whisk the page far past the head before the press registers. Freezing the belt
        # while the grippers travel keeps the punch timing controllable and deterministic across
        # the collector's plan + render passes (both replay the same step counts).
        if not getattr(self, "_belt_running", False):
            # still refresh page poses so they render at their held position, but don't advance
            for side in ("left", "right"):
                for k in range(self.n_pages):
                    self._set_page_pose(side, k, self._page_x_at(side, k, self._belt_step))
            return
        self._belt_step += 1
        for side in ("left", "right"):
            best_k, best_d = None, 1e9
            for k in range(self.n_pages):
                x = self._page_x_at(side, k, self._belt_step)
                self._set_page_pose(side, k, x)
                # only an UNPUNCHED page can be the current target under the head
                if self.page_punched[side][k]:
                    continue
                d = abs(x - self._punch_x[side])
                if d < best_d:
                    best_d, best_k = d, k
            # the closest still-unpunched page, if it is genuinely near the head
            self._under_head[side] = best_k if best_d < 0.05 else None
            # animate punch head descending while a press is active
            if self._punch_press[side] > 0:
                self._punch_press[side] -= 1
                frac = 1.0 - self._punch_press[side] / max(1, self._punch_total)
                # simple down-then-up triangle
                tri = 1.0 - abs(2 * frac - 1.0)
                drop = tri * 0.045
                h = self.punch_head[side]
                hp = h.get_pose()
                h.actor.set_pose(sapien.Pose([hp.p[0], hp.p[1], self._punch_rest_z[side] - drop], hp.q))

    def _fire_punch(self, side):
        """Register a punch on `side` at the current belt step: punch whichever page is
        under the head, recording its x-offset from the punch center, and trigger the
        head's visual descent."""
        self._punch_total = 24
        self._punch_press[side] = self._punch_total
        k = self._under_head[side]
        if k is None or self.page_punched[side][k]:
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

    def _belt_idle(self, steps):
        """Advance the belts `steps` physics steps while recording frames (no arm motion)."""
        self._belt_running = True
        try:
            for i in range(steps):
                self._update_kinematic_tasks()
                self.scene.step()
                if self.save_freq and (i % self.save_freq == 0):
                    self._take_picture()
        finally:
            self._belt_running = False

    # -------------------------------------------------------- press poses
    def _button_press_actions(self, side, descend):
        """Build a (arm_tag, [Action]) that moves the side's gripper straight down by
        `descend` onto its button (relative move along the world -z)."""
        arm = ArmTag(side)
        return self.move_by_displacement(arm_tag=arm, z=-descend)

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
        sign = self._sides[side]
        # page_start_x - sign*speed*eff = punch_x  ->  eff = (start - punch)/(sign*speed)
        eff = (self.page_start_x[side][k] - self._punch_x[side]) / (sign * self.belt_speed[side])
        return int(round(eff)) + self.belt_phase[side]

    def play_once(self):
        # 1) Bring both grippers to hover over their buttons simultaneously (truly dual-arm).
        self.move(
            self._hover_button("left"),
            self._hover_button("right"),
        )
        self._dbg("after hover")

        # start the belts running
        self._belt_active = True

        # 2) Build a single timeline of "page reaches its punch head" events across BOTH belts
        #    (each belt has its own speed/phase, so the events interleave). We service them in
        #    time order: at each event we run the belts up to that step, then press BOTH buttons
        #    together (a truly simultaneous dual-arm action). Each side's punch independently
        #    fires on whatever page is currently under its head, so a press cleanly catches the
        #    arriving page; the other arm's press either catches its own arriving page or fires
        #    harmlessly. This clears every page on both belts.
        events = []
        for side in ("left", "right"):
            for k in range(self.n_pages):
                events.append((self._page_arrival_step(side, k), side, k))
        events.sort(key=lambda e: e[0])

        for step_target, side, k in events:
            if all(all(self.page_punched[s]) for s in ("left", "right")):
                break
            self._run_belts_to(step_target)
            # press BOTH buttons at once: descend both grippers together
            self.move(
                self._button_press_actions("left", 0.05),
                self._button_press_actions("right", 0.05),
            )
            # fire each side's punch on whatever page is under its head right now
            self._fire_punch("left")
            self._fire_punch("right")
            self._dbg(f"after press @ev({side},{k})")
            # let the punch-head descent animation play out on camera
            self._belt_idle(28)
            # lift both grippers back up together, ready for the next event
            self.move(
                self.move_by_displacement(ArmTag("left"), z=0.05),
                self.move_by_displacement(ArmTag("right"), z=0.05),
            )

        # let any trailing pages roll to the end so the final frames are clean
        self._belt_idle(40)
        self._belt_active = False

        self.info["info"] = {"{a}": "left", "{b}": "right"}
        return self.info

    def _run_belts_to(self, step_target, max_extra=400):
        """Advance the belts until self._belt_step reaches step_target, recording frames."""
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
            self._belt_running = False

    # ------------------------------------------------------------- scoring
    def _side_score(self, side):
        offs = [o for o in self.page_offset[side] if o is not None]
        if not offs:
            return 0.0
        return float(np.mean([np.clip(1.0 - o / self.punch_tol, 0.0, 1.0) for o in offs]))

    def check_success(self):
        all_punched = all(all(self.page_punched[s]) for s in ("left", "right"))
        self.punch_score_L = self._side_score("left")
        self.punch_score_R = self._side_score("right")
        self.punch_score_mean = 0.5 * (self.punch_score_L + self.punch_score_R)
        return bool(all_punched)

    # record per-belt punch state into the trajectory (per frame)
    def get_obs(self):
        obs = super().get_obs()
        obs["hole_punch"] = {
            "left_punched": [bool(b) for b in self.page_punched["left"]],
            "right_punched": [bool(b) for b in self.page_punched["right"]],
            "left_offsets": [None if o is None else float(o) for o in self.page_offset["left"]],
            "right_offsets": [None if o is None else float(o) for o in self.page_offset["right"]],
            "punch_score_L": float(self._side_score("left")),
            "punch_score_R": float(self._side_score("right")),
            "punch_tol": float(self.punch_tol),
        }
        return obs
