from ._base_task import Base_Task
from .utils import *
import sapien
import sapien.render
import numpy as np


class sort_apples_belt(Base_Task):
    """Sort a random number (4-10) of red/green apples streaming down a conveyor into the
    color-matched basket, by pressing the matching side button to aim a pivoting diverter blade.

    Apples spawn at the far end with spacing and ride the belt as a continuous STREAM (2-3 on the
    belt at once; a new one spawns once the previous has travelled ~1/3 of the belt) at a steady,
    per-episode-random speed. Each apple's color is an independent coin flip. Two color-coded
    baskets sit at the fork (which side is green vs red is randomized). At the belt end a single
    diverter BLADE pivots smoothly between two diagonals: the LEFT side button aims it to route
    apples LEFT, the RIGHT button routes RIGHT.

    The diverter flips ONLY when a gripper PHYSICALLY presses a side button (EE-to-button proximity,
    detected every step in `_update_kinematic_tasks`) -- routing is driven by the actual press, so
    the task is policy-evaluable: the scripted expert and a learned policy use the same mechanism.
    The stream + diverter run autonomously (started in `setup_demo`), so the scene is live during a
    policy rollout, not only during the expert's `play_once`. Each apple rides the belt kinematically
    (prescribed roll-without-slip), is released at the edge, and FALLS into the basket under real
    physics (inelastic, damped).

    Metric: sorting_accuracy = correct / n_apples (success = ALL apples in the right basket), scored
    from where each apple PHYSICALLY settles, plus macro-F1 over {red, green}.
    """

    # ---- class-default task params (override via task_args.sort_apples_belt) ----
    # apple count and belt speed are RANDOMIZED per episode within these ranges
    N_APPLES_MIN_DEFAULT = 4
    N_APPLES_MAX_DEFAULT = 10
    BELT_SPEED_MIN_DEFAULT = 0.0005    # m per "advance" tick; steady per episode, random in this range
    BELT_SPEED_MAX_DEFAULT = 0.0010    # ~0.03-0.06 m/s surface -> constant flow slow enough that the
                                       # arm can press per apple WITHOUT ever stopping the belt
    ADVANCE_EVERY_DEFAULT = 4          # physics steps between belt advances
    GATE_DEFAULT_LEFT_DEFAULT = True   # gate direction before the first press
    OBSERVE_DIST = 0.10                # apple travels this far into view before the robot presses
    GATE_TILT = 0.6                    # blade yaw magnitude (rad) for each diagonal ('/' = +, '\' = -)
    SPAWN_GAP = 0.16                   # belt-distance between consecutive apple spawns -> ~2 apples ride
                                       # the belt; > DECISION_DY so the previous apple has deposited
                                       # (left the steer zone) before the gate flips for the next one
    DECISION_DY = 0.15                 # the robot routes the front apple when it reaches FORK+DECISION_DY
                                       # (well above the 0.07 steer zone -> gate is set before it steers)
    PRESS_XY = 0.06                    # a gripper within this xy of a side button (and pressed down,
    PRESS_DZ = 0.18                    # EE z < button_top+PRESS_DZ) actuates that side's diverter --
                                       # the PHYSICAL button->splitter hook (works for expert & policy)
    BUTTON_DX = 0.25                   # the two side buttons sit at x = -+BUTTON_DX (out by the baskets)
    BUTTON_Y_DEFAULT = -0.16           # near-front, lateral -> off the camera's central belt corridor

    RED = [0.85, 0.10, 0.10]
    GREEN = [0.15, 0.70, 0.18]

    # belt geometry (table-frame); belt runs along y, fork near the robot (low y)
    BELT_X_DEFAULT = 0.0               # belt centered on the table centerline
    BELT_HALF_W = 0.10                 # belt half-width (x) -- WIDER belt
    BELT_Y_FAR = 0.24                  # far end (apples spawn here) -- LONGER belt
    BELT_Y_FORK = -0.06                # fork: where apples divert into a basket
    BELT_TOP_DZ = 0.025                # belt slab thickness
    BELT_LIFT = 0.135                  # belt surface height above table (elevated on legs) so apples
                                       # roll off the fork edge and FALL into the baskets below, with
                                       # clear vertical separation from the basket rims (no collision)
    APPLE_SCALE = 0.5                  # apples small (easily fit in the basket)
    BASKET_SCALE = 0.85                # baskets BIGGER -> forgiving catch target (known-good value)
    APPLE_R = 0.017                    # apple radius at APPLE_SCALE (rolling spin + belt offset)
    BASKET_CATCH_R = 0.085             # apple counts as "in basket" if within this of basket center
    N_SLATS = 4                        # fewer, wider-spaced slats -> per-frame motion < half spacing
                                       # (avoids the wagon-wheel reversal)

    def setup_demo(self, **kwags):
        self._cfg = kwags.get("task_args", {}).get("sort_apples_belt", {})
        super()._init_task_env_(**kwags)
        # activate the stream AFTER setup (the base settles the scene during _init_task_env_ with the
        # stream OFF, so apples don't deposit before the episode). Set here -- NOT in play_once -- so
        # the belt/apples also run during a policy-evaluation rollout (where play_once is not called).
        self._stream_active = True

    # ----------------------------------------------------------------- actors
    def load_actors(self):
        cfg = self._cfg
        # RANDOM per episode: apple count (4-10) and belt speed -- deterministic per seed, so the
        # plan and render passes draw the same values and the routing replays identically.
        n_min = int(cfg.get("n_apples_min", self.N_APPLES_MIN_DEFAULT))
        n_max = int(cfg.get("n_apples_max", self.N_APPLES_MAX_DEFAULT))
        self.n_apples = int(np.random.randint(n_min, n_max + 1))
        s_min = float(cfg.get("belt_speed_min", self.BELT_SPEED_MIN_DEFAULT))
        s_max = float(cfg.get("belt_speed_max", self.BELT_SPEED_MAX_DEFAULT))
        self.belt_speed = float(np.random.uniform(s_min, s_max)) * float(cfg.get("belt_speed_scale", 1.0))
        self.advance_every = int(cfg.get("advance_every", self.ADVANCE_EVERY_DEFAULT))
        self.gate_default_left = bool(cfg.get("gate_default_left", self.GATE_DEFAULT_LEFT_DEFAULT))

        self._z0 = 0.74 + self.table_z_bias
        belt_top = self._z0 + self.BELT_LIFT          # elevated belt surface
        self._belt_surf = belt_top

        # ---- randomization: per-apple color, which side is red vs green ----
        # each apple's color is an INDEPENDENT 50/50 coin flip (0=red, 1=green): counts vary freely
        # per episode (an episode could even be all one color).
        self.apple_colors = [int(np.random.rand() < 0.5) for _ in range(self.n_apples)]
        # which side holds which color basket: green on left or right
        self.green_on_left = bool(np.random.rand() < 0.5)

        # basket x positions at the fork (left = -x, right = +x). Delivery is kinematic, so baskets
        # may live on either side of the centerline (only the button must be right-arm reachable).
        self._basket_x = {"left": -0.15, "right": 0.15}     # big baskets just past the belt edges
        self._basket_y = self.BELT_Y_FORK - 0.01
        # map color -> side and side -> color
        if self.green_on_left:
            self._color_side = {1: "left", 0: "right"}   # green left, red right
        else:
            self._color_side = {1: "right", 0: "left"}
        self._side_color = {v: k for k, v in self._color_side.items()}

        # ---- belt: a long static slab along y, surface at belt_top (elevated on legs) ----
        belt_len = self.BELT_Y_FAR - self.BELT_Y_FORK + 0.10
        belt_cy = (self.BELT_Y_FAR + self.BELT_Y_FORK) / 2.0
        belt_slab_cz = belt_top - self.BELT_TOP_DZ / 2.0           # slab center (top at belt_top)
        self.belt = create_box(
            self.scene,
            sapien.Pose([self.BELT_X_DEFAULT, belt_cy, belt_slab_cz]),
            half_size=[self.BELT_HALF_W, belt_len / 2.0, self.BELT_TOP_DZ / 2.0],
            color=[0.20, 0.20, 0.22], is_static=True, name="belt",
        )
        # support legs from the table up to the belt underside (4 corners)
        leg_top = belt_top - self.BELT_TOP_DZ
        leg_h = leg_top - self._z0
        for lx in (-self.BELT_HALF_W * 0.45, self.BELT_HALF_W * 0.45):   # inward -> clear of baskets
            for ly in (belt_cy - belt_len / 2.0 + 0.04, belt_cy + belt_len / 2.0 - 0.04):
                create_box(
                    self.scene, sapien.Pose([lx, ly, self._z0 + leg_h / 2.0]),
                    half_size=[0.012, 0.012, leg_h / 2.0], color=[0.3, 0.3, 0.33],
                    is_static=True, name="belt_leg",
                )

        # ---- moving belt: cross-slats that travel with the surface (re-posed each tick) so the belt
        # visibly RUNS; recycled to the far end when they pass the fork ----
        self._slat_spacing = belt_len / self.N_SLATS
        self._slat_near = belt_cy - belt_len / 2.0
        self.slats = []
        self._slat_ys = []
        for k in range(self.N_SLATS):
            sy = self._slat_near + k * self._slat_spacing
            sl = create_box(
                self.scene, sapien.Pose([self.BELT_X_DEFAULT, sy, belt_top + 0.001]),
                half_size=[self.BELT_HALF_W * 0.92, 0.006, 0.003],
                color=[0.10, 0.10, 0.12], is_static=True, name=f"slat_{k}",
            )
            self.slats.append(sl)
            self._slat_ys.append(sy)

        # ---- splitter: ONE full-width diagonal blade at the belt end, pivoting about its CENTER.
        # '/' (yaw>0) walls off the right and funnels apples LEFT; '\' (yaw<0) funnels them RIGHT.
        # The button flips it between the two diagonals (railroad-switch style diverter). ----
        self.gate_left = self.gate_default_left  # True = route LEFT ('/')
        self._blade_half_len = 0.13              # long enough to span the belt when diagonal
        self.gate = create_box(
            self.scene, sapien.Pose([self.BELT_X_DEFAULT, self.BELT_Y_FORK, belt_top + 0.014]),
            half_size=[self._blade_half_len, 0.007, 0.024],   # long blade, thin, tall enough to deflect
            color=[0.60, 0.62, 0.66], is_static=True, name="diverter_gate",
        )
        # the blade ROTATES smoothly toward a target yaw (set by a button press) rather than snapping,
        # so the diverter visibly turns when the button is pushed. yaw +GATE_TILT='/' (route LEFT),
        # -GATE_TILT='\' (route RIGHT). _animate_gate (called every physics step) drives _gate_yaw
        # toward _gate_yaw_target at _gate_rate rad/step -- step-driven, so it replays across passes.
        self._gate_yaw = self.GATE_TILT if self.gate_left else -self.GATE_TILT
        self._gate_yaw_target = self._gate_yaw
        self._gate_rate = 0.012                  # rad/step -> a full +-GATE_TILT flip sweeps in ~0.4 s
                                                 # (clearly visible turn at the 16.7 Hz capture rate)
        self._apply_gate_pose()

        # ---- TWO side buttons (left + right), out by the baskets and off the camera's central belt
        # corridor so the pressing arm doesn't occlude the view. Each button SETS the gate to its side
        # (left button -> route LEFT, right button -> route RIGHT) and is color-coded to that side's
        # target basket color, so the robot observes the apple then presses the matching-color button. ----
        btn_y = float(cfg.get("button_y", self.BUTTON_Y_DEFAULT))
        self._button_xy = {"left": (-self.BUTTON_DX, btn_y), "right": (self.BUTTON_DX, btn_y)}
        self._button_top_z = self._z0 + 0.06
        self.buttons = {}
        for side, (bx, by) in self._button_xy.items():
            create_box(
                self.scene, sapien.Pose([bx, by, self._z0 + 0.015]),
                half_size=[0.032, 0.032, 0.015], color=[0.10, 0.10, 0.12],
                is_static=True, name=f"button_base_{side}",
            )
            btn_color = self.RED if self._side_color[side] == 0 else self.GREEN
            self.buttons[side] = create_box(
                self.scene, sapien.Pose([bx, by, self._z0 + 0.045]),
                half_size=[0.022, 0.022, 0.015], color=btn_color,
                is_static=True, name=f"button_{side}",
            )

        # ---- baskets at the fork, pre-seeded with 2 reference apples each ----
        # basket model 3 is banned: its oversized rim (scale 0.15 vs 0.12) sits in the left-routed
        # apple's launch path and clips it, so it was the SOLE source of landing misses in a 20-seed
        # sweep (models 0/1/2 = 93/93 = 100%, model 3 = 36/41 = 88%). Use 0/1/2 only.
        self.basket_id = int(np.random.choice([0, 1, 2]))
        self.baskets = {}
        self._ref_apples = []
        for side in ("left", "right"):
            bx = self._basket_x[side]
            basket = create_actor(
                self, pose=sapien.Pose([bx, self._basket_y, self._z0],
                                       [0.707, 0.707, 0, 0]),
                modelname="110_basket", model_id=self.basket_id,
                convex=True, is_static=True, scale_mult=self.BASKET_SCALE,
            )
            self.baskets[side] = basket
            color = self.RED if self._side_color[side] == 0 else self.GREEN
            # 2 reference apples in this basket
            for k in range(2):
                ra = create_actor(
                    self, pose=sapien.Pose([bx + (k - 0.5) * 0.03, self._basket_y,
                                            self._z0 + 0.03]),
                    modelname="220_apple_plain", model_id=0, convex=True, is_static=True,
                    scale_mult=self.APPLE_SCALE,
                )
                self._recolor(ra, color)
                self._ref_apples.append(ra)

        # (no back-stop walls: the pivoting-blade splitter + elevated belt land apples reliably in the
        # baskets, so the outboard catch-walls are unnecessary and looked like boards in the baskets.)

        # ---- the 10 incoming apples: stacked off-belt at the far end, fed one at a time ----
        self.apples = []
        self.apple_shapes = []
        self._stage_pose = sapien.Pose([self.BELT_X_DEFAULT, self.BELT_Y_FAR + 0.30,
                                        self._z0 + 0.30])  # parking spot above/behind
        # apples are DYNAMIC but kept kinematic on the belt; released to real physics at the fork
        # so they fall off the edge into the basket below (deterministic outcome; gate decides side).
        self.apples = []
        self._apple_comps = []
        inelastic = sapien.physx.PhysxMaterial(static_friction=0.9, dynamic_friction=0.9, restitution=0.0)
        for i in range(self.n_apples):
            ap = create_actor(
                self, pose=sapien.Pose([self._stage_pose.p[0],
                                        self._stage_pose.p[1] + i * 0.02,
                                        self._stage_pose.p[2]]),
                modelname="220_apple_plain", model_id=0, convex=True, is_static=False,
                scale_mult=self.APPLE_SCALE,
            )
            self._recolor(ap, self.RED if self.apple_colors[i] == 0 else self.GREEN)
            comp = None
            for cc in ap.actor.get_components():
                if isinstance(cc, sapien.physx.PhysxRigidDynamicComponent):
                    comp = cc
            if comp is not None:
                for sh in comp.get_collision_shapes():
                    sh.set_physical_material(inelastic)
                comp.set_disable_gravity(True)
                comp.set_kinematic(True)              # frozen until released at the fork
            self.apples.append(ap)
            self._apple_comps.append(comp)

        # ---- streaming / routing / scoring state ----
        self.cur_idx = 0                  # front in-flight apple (for obs)
        self._step_ctr = 0
        self._apple_y = [None] * self.n_apples    # per-apple y on the belt (None = staged or deposited)
        self._apple_roll = [0.0] * self.n_apples  # per-apple rolling angle
        self._spawned = 0                          # how many apples have entered the belt so far
        self._deposited = [False] * self.n_apples  # released off the belt into a basket
        self._decided = [False] * self.n_apples    # robot has routed (pressed if needed) for this apple
        self._stream_active = False                # stream runs ONLY during play_once (not during the
                                                   # base env's setup stepping, which would otherwise
                                                   # deposit every apple before the policy starts)
        self._routed = [None] * self.n_apples    # gate side the apple was released toward (routing)
        self.delivered = [None] * self.n_apples  # basket the apple PHYSICALLY ended in (eval)
        self.results = [None] * self.n_apples    # True/False physically correct
        self.press_count = 0
        self._pic_ctr = 0                         # running frame counter across the feed loops

        # prohibit clutter from spawning on belt / baskets / buttons
        self.add_prohibit_area(self.belt, padding=0.04)
        for b in self.buttons.values():
            self.add_prohibit_area(b, padding=0.05)
        for b in self.baskets.values():
            self.add_prohibit_area(b, padding=0.04)

    # --------------------------------------------------------------- rendering
    def _shapes_of(self, actor):
        out = []
        for c in actor.actor.get_components():
            if isinstance(c, sapien.render.RenderBodyComponent):
                out = list(c.render_shapes)
        return out

    def _recolor(self, actor, rgb):
        for s in self._shapes_of(actor):
            try:
                s.material.set_base_color(list(rgb) + [1.0])
            except Exception:
                pass

    def _apply_gate_pose(self):
        # render the blade at its CURRENT yaw (may be mid-sweep): pivot about its CENTER (z-axis).
        q = [np.cos(self._gate_yaw / 2), 0.0, 0.0, np.sin(self._gate_yaw / 2)]
        self.gate.actor.set_pose(
            sapien.Pose([self.BELT_X_DEFAULT, self.BELT_Y_FORK, self._belt_surf + 0.014], q))

    def _set_gate_target(self):
        # a button press points the blade at the matching diagonal: '/' (+GATE_TILT) routes LEFT,
        # '\' (-GATE_TILT) routes RIGHT. _animate_gate then sweeps the blade there over several steps.
        self._gate_yaw_target = self.GATE_TILT if self.gate_left else -self.GATE_TILT

    def _animate_gate(self):
        # smoothly rotate the blade toward its target so the diverter visibly TURNS on a press
        # instead of snapping. Step-driven (fixed rate/step) -> identical in the plan + render passes.
        if not hasattr(self, "_gate_yaw"):
            return
        d = self._gate_yaw_target - self._gate_yaw
        if abs(d) < 1e-6:
            return
        step = self._gate_rate if d > 0 else -self._gate_rate
        self._gate_yaw = self._gate_yaw_target if abs(step) >= abs(d) else self._gate_yaw + step
        self._apply_gate_pose()

    # --------------------------------------------------------------- belt sim
    def _hide(self, actor, idx_offset):
        actor.actor.set_pose(sapien.Pose([self._stage_pose.p[0] + idx_offset,
                                          self._stage_pose.p[1] + 1.0,
                                          self._stage_pose.p[2] - 1.0]))

    def _spawn(self, idx):
        # put apple idx onto the far end of the belt; it then advances with the stream
        self._apple_y[idx] = self.BELT_Y_FAR
        self._apple_roll[idx] = 0.0
        comp = self._apple_comps[idx]
        if comp is not None:
            comp.set_kinematic_target(
                sapien.Pose([self.BELT_X_DEFAULT, self.BELT_Y_FAR, self._belt_surf + self.APPLE_R]))

    def _deposit(self, idx):
        # release the apple off the belt edge toward the gate-selected side; it then falls under real
        # physics into the basket. Scoring is done later from where it PHYSICALLY lands.
        side = "left" if self.gate_left else "right"
        self._routed[idx] = side
        comp = self._apple_comps[idx]
        if comp is not None:
            sgn = -1.0 if side == "left" else 1.0
            # lift the apple just OFF the belt edge before releasing it (so it can't stick to the
            # high-friction belt at the fork or get wedged by the rotating blade) -> clean arc in.
            self.apples[idx].actor.set_pose(sapien.Pose(
                [sgn * (self.BELT_HALF_W + 0.005), self.BELT_Y_FORK, self._belt_surf + 0.035]))
            comp.set_kinematic(False)
            comp.set_disable_gravity(False)
            vx = sgn * 0.45                                   # outward toward the basket center (+-0.15)
            vy = -0.03 - (idx % 3) * 0.012                    # gentle spread so apples don't tower up
            comp.set_linear_velocity([vx, vy, -0.25])        # outward + down -> arcs into the basket
            comp.set_angular_velocity([0.0, 0.0, 0.0])
            try:
                comp.set_linear_damping(0.2)
                comp.set_angular_damping(1.0)
            except Exception:
                pass
        self._deposited[idx] = True

    def _advance_slats(self):
        # belt always runs: translate each slat -y, recycle to the far end when it passes the near end
        if not getattr(self, "slats", None):
            return
        span = self.N_SLATS * self._slat_spacing
        for k in range(self.N_SLATS):
            y = self._slat_ys[k] - self.belt_speed
            if y < self._slat_near:
                y += span
            self._slat_ys[k] = y
            self.slats[k].actor.set_pose(sapien.Pose([self.BELT_X_DEFAULT, y, self._belt_surf + 0.001]))

    def _advance_stream(self):
        # spawn apples with spacing and advance EVERY in-flight apple together (a stream of 2-3 on the
        # belt). Steering/deposit use the shared gate; SPAWN_GAP > steer_dist guarantees only the
        # front apple is ever inside the steer zone, so the gate steers exactly the apple it was set
        # for. (Called only when the belt is not frozen -- see _update_kinematic_tasks.)
        # spawn the next apple once the previous has travelled SPAWN_GAP down the belt
        if self._spawned < self.n_apples:
            prev = self._spawned - 1
            if self._spawned == 0 or (self._apple_y[prev] is not None
                                      and self._apple_y[prev] <= self.BELT_Y_FAR - self.SPAWN_GAP):
                self._spawn(self._spawned)
                self._spawned += 1
        steer_dist = 0.07
        edge_x = (-self.BELT_HALF_W * 0.9) if self.gate_left else (self.BELT_HALF_W * 0.9)
        front = None
        for i in range(self._spawned):
            if self._apple_y[i] is None or self._deposited[i]:
                continue
            if front is None:
                front = i
            self._apple_y[i] -= self.belt_speed
            if self._apple_y[i] <= self.BELT_Y_FORK:
                self._deposit(i)
                continue
            if self._apple_y[i] < self.BELT_Y_FORK + steer_dist:
                frac = 1.0 - max(0.0, (self._apple_y[i] - self.BELT_Y_FORK)) / steer_dist
                ax = edge_x * max(0.0, min(1.0, frac))
            else:
                ax = self.BELT_X_DEFAULT
            self._apple_roll[i] += self.belt_speed / self.APPLE_R
            a = self._apple_roll[i]
            q = [np.cos(a / 2), np.sin(a / 2), 0, 0]            # spin about x -> rolls in -y
            comp = self._apple_comps[i]
            if comp is not None:
                comp.set_kinematic_target(
                    sapien.Pose([ax, self._apple_y[i], self._belt_surf + self.APPLE_R], q))
        if front is not None:
            self.cur_idx = front

    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
        # this hook fires during _init_task_env_ (load_camera) BEFORE load_actors() runs,
        # so guard against not-yet-initialized belt state.
        if not hasattr(self, "_step_ctr"):
            return
        self._step_ctr += 1
        self._animate_gate()               # smooth blade sweep, every step (independent of belt rate)
        if self._stream_active:
            self._detect_button_press()    # physical button -> splitter coupling (expert AND policy)
            if self._step_ctr % max(1, self.advance_every) == 0:
                self._advance_slats()      # belt runs CONSTANTLY (no freeze) once the stream is active
                self._advance_stream()

    def _detect_button_press(self):
        # PHYSICAL hook: when a side's gripper is pressing its side button (close in xy and pushed
        # down), actuate the diverter to that side. This is what couples the button to the splitter
        # for the scripted expert AND for a learned policy at eval time -- routing depends on the
        # actual press, not on any scripted gate-set.
        if not hasattr(self, "robot"):
            return
        for side, get_ee in (("left", self.robot.get_left_ee_pose),
                             ("right", self.robot.get_right_ee_pose)):
            try:
                ee = np.array(get_ee()[:3], dtype=float)
            except Exception:
                continue
            bx, by = self._button_xy[side]
            if (abs(ee[0] - bx) < self.PRESS_XY and abs(ee[1] - by) < self.PRESS_XY
                    and ee[2] < self._button_top_z + self.PRESS_DZ):
                if self.gate_left != (side == "left"):
                    self.gate_left = (side == "left")
                    self.press_count += 1
                    self._set_gate_target()

    # --------------------------------------------------------------- helpers
    def _step_record(self):
        # one sim step with frame capture on the running counter
        self._update_kinematic_tasks()
        self.scene.step()
        if self.save_freq and (self._pic_ctr % self.save_freq == 0):
            self._take_picture()
        self._pic_ctr += 1

    def _press_side_button(self, side):
        # the same-side arm PHYSICALLY presses the matching side button; the diverter then flips via
        # _detect_button_press (the gripper-at-button hook), NOT via any scripted gate-set here -- so
        # the same physical mechanism drives routing for the expert and for a policy. The belt keeps
        # running throughout (no freeze); it is slow enough that the press lands before the apple steers.
        arm_tag = ArmTag("left" if side == "left" else "right")
        self.move(self.grasp_actor(self.buttons[side], arm_tag=arm_tag, pre_grasp_dis=0.08,
                                   grasp_dis=0.08, contact_point_id=0, gripper_pos=0.0))
        self.move(self.move_by_displacement(arm_tag, z=-0.06))   # push DOWN onto the button
        self.move(self.move_by_displacement(arm_tag, z=0.06))    # lift; belt + apples keep moving

    def _front_undecided(self):
        for i in range(self._spawned):
            if not self._decided[i] and not self._deposited[i]:
                return i
        return None

    # --------------------------------------------------------------- policy
    def play_once(self):
        # continuous stream: apples spawn with spacing and ride the belt together; as each nears the
        # fork the robot routes it -- pressing the matching side button ONLY when the diverter must
        # change side (a run of same-color apples needs no press). The belt freezes during a press.
        self._stream_active = True          # apples now start spawning/advancing
        travel = (self.BELT_Y_FAR - self.BELT_Y_FORK + self.n_apples * self.SPAWN_GAP) \
            / max(1e-4, self.belt_speed) * self.advance_every
        max_steps = int(travel) + self.n_apples * 400 + 800
        guard = 0
        while guard < max_steps:
            guard += 1
            if self._spawned >= self.n_apples and all(self._deposited):
                break
            fi = self._front_undecided()
            if (fi is not None and self._apple_y[fi] is not None and not self._deposited[fi]
                    and self._apple_y[fi] <= self.BELT_Y_FORK + self.DECISION_DY):
                side = self._color_side[self.apple_colors[fi]]
                if self.gate_left != (side == "left"):
                    self._press_side_button(side)       # route it (belt frozen, blade sweeps)
                self._decided[fi] = True
            self._step_record()
        # settle: let the last released apples come to rest in the baskets
        for _ in range(160):
            self._step_record()

        self.info["info"] = {
            "{A}": "220_apple_plain/base0",
            "{B}": f"110_basket/base{self.basket_id}",
            "{a}": "both",
        }
        return self.info

    # --------------------------------------------------------------- success
    def _settled_side(self, idx):
        # which basket the apple PHYSICALLY ended up in (None if it missed / is still on the belt)
        p = np.array(self.apples[idx].get_pose().p)
        if p[2] > self._z0 + 0.12:      # still up on the belt / mid-air -> not settled in a basket
            return None
        for side in ("left", "right"):
            if np.hypot(p[0] - self._basket_x[side], p[1] - self._basket_y) < self.BASKET_CATCH_R:
                return side
        return None

    def _eval_landings(self):
        # honest scoring from the apples' ACTUAL positions: an apple is correct only if it physically
        # landed in the color-matched basket (not merely routed there by the gate).
        for idx in range(self.n_apples):
            if self._routed[idx] is None:
                continue                # not fed yet
            side = self._settled_side(idx)
            self.delivered[idx] = side
            self.results[idx] = (side is not None and side == self._color_side[self.apple_colors[idx]])

    def _macro_f1(self):
        labels = [0, 1]  # red, green
        f1s = []
        for c in labels:
            tp = sum(1 for i in range(self.n_apples)
                     if self.apple_colors[i] == c and self.results[i])
            fp = sum(1 for i in range(self.n_apples)
                     if self.apple_colors[i] != c and self.delivered[i] is not None
                     and self._side_color[self.delivered[i]] == c)
            fn = sum(1 for i in range(self.n_apples)
                     if self.apple_colors[i] == c and not self.results[i])
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            f1s.append(f1)
        return float(np.mean(f1s))

    def check_success(self):
        self._eval_landings()           # score from where the apples PHYSICALLY landed
        fed = all(r is not None for r in self._routed)
        correct = sum(1 for r in self.results if r)
        self.sorting_accuracy = correct / float(self.n_apples)
        self.macro_f1 = self._macro_f1()
        return bool(fed and correct == self.n_apples)

    # record routing/scoring state into the trajectory (per-frame)
    def get_obs(self):
        obs = super().get_obs()
        if hasattr(self, "_routed"):
            self._eval_landings()      # physical landings of whatever has settled so far
        correct = sum(1 for r in self.results if r)
        obs["sorting"] = {
            "gate_left": bool(getattr(self, "gate_left", True)),
            "cur_idx": int(getattr(self, "cur_idx", 0)),
            "press_count": int(getattr(self, "press_count", 0)),
            "delivered_count": int(sum(1 for d in self.delivered if d is not None)),
            "correct": int(correct),
            "sorting_accuracy": float(correct / float(self.n_apples)),
            "macro_f1": float(self._macro_f1()),
            "green_on_left": bool(getattr(self, "green_on_left", True)),
            "n_apples": int(getattr(self, "n_apples", 0)),
            "belt_speed": float(getattr(self, "belt_speed", 0.0)),
        }
        return obs
