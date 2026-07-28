from ._base_task import Base_Task
from .utils import *
from ._GLOBAL_CONFIGS import *
import sapien
import sapien.render
import numpy as np
import os


class packing(Base_Task):
    """Pack apples and oranges from two moving belts into breadbaskets.

    Two conveyor slabs sit with a gap centered on the table and run toward the
    robot (-y). Spawn behaviour is controlled by ``spawn_mode``:

    - ``random`` (Opt 1 / default): each wave independently rolls a coin and
      spawns either a single fruit (apple xor orange; either belt when
      ``single_wave_any_belt``) or an apple+orange pair (Y-gap ~ U(0,
      fruit_diameter) when ``pair_stagger_enabled``).
    - ``parallel``: always apple+orange pair waves (same Y-gap sampling).
    - ``single``: only one fruit at a time; either color on either belt.

    Opt 2 (independent; can combine with Opt 1): ``distractor_enabled`` adds
    black distractor fruit on the belts (never packed / counted).

    Success requires every fruit to rest in its color-matched basket.

    Belt speed is sampled each episode as nominal × U(1 ± belt_speed_jitter)
    (default ±20%), independently per belt.
    """

    N_PER_COLOR_DEFAULT = 3
    BELT_GAP_DEFAULT = 0.10
    BELT_SPEED_DEFAULT = 0.0008       # m advanced per belt tick (slow enough to pick)
    BELT_SPEED_JITTER_DEFAULT = 0.20  # fraction; speed ~ U((1-j)*nom, (1+j)*nom)
    ADVANCE_EVERY_DEFAULT = 3         # physics steps between belt ticks
    SPAWN_GAP_DEFAULT = 0.16          # y-gap between consecutive spawns on a belt
    SPAWN_MODE_DEFAULT = "random"     # "single" | "parallel" | "random"
    SPAWN_DELAY_S_DEFAULT = 2.0       # unused (kept for config compat); spawn waits on drop/despawn
    # pair wave: per-wave Y gap ~ U(0, max); max defaults to fruit diameter
    PAIR_STAGGER_ENABLED_DEFAULT = False
    PAIR_STAGGER_Y_DEFAULT = None     # None → fruit diameter (2 * fruit_r)
    # "random" mode single-fruit wave: let it appear on either belt instead
    # of always its color-dedicated one (arm/basket stay color-matched)
    SINGLE_WAVE_ANY_BELT_DEFAULT = False

    # same belt slab dimensions as quality_control
    BELT_HALF_LEN = 0.30
    BELT_HALF_WID = 0.07
    BELT_THICK = 0.012
    BELT_Y = 0.0
    BELT_Y_FAR = 0.26                 # spawn y (far end)
    BELT_Y_NEAR = -0.26               # leave / wrap y
    PICK_Y = 0.24                     # begin moving the arm into place (fruit keeps rolling)
    PICK_Y_END = -0.16                # give up past this (still moving; never park)
    PICK_STATION_Y = 0.02             # hover / grab y; fruit rolls through here
    # attach only once the gripper has approached within ~2 cm
    ATTACH_XY = 0.02
    ATTACH_Z_MAX = 0.055              # TCP may sit slightly above the fruit
    HIDE_Z = -10.0

    FRUIT_MODEL = "035_apple"
    FRUIT_SCALE = 0.80                # larger -> easier grasp
    # 035_apple model_data0.json (authored scale already applied by create_actor)
    _APPLE_AUTHOR_SCALE = 0.7
    _APPLE_CENTER_Y = 0.03814048367178239
    _APPLE_EXTENT_Y = 0.0919138697184135
    FRUIT_R = 0.026                   # approx half-extent at FRUIT_SCALE
    FRUIT_MASS = 0.025
    # orientation that exposes a top-down contact frame (see pick_ripe_apple);
    # rotates local +y -> world +z, so mesh center offset affects ride height
    FRUIT_Q = [0.707, 0.707, 0.0, 0.0]
    BASKET_SCALE = 1.15                # bigger opening — easier drop-in target
    BASKET_CATCH_R = 0.12              # scaled up to match the larger basket opening
    BASKET_Y = 0.0                    # table midline (toward the belts / "higher")
    BASKET_X = 0.34                   # nudged out so the bigger basket clears the belts
    # after grasp: raise the gripper this far along world +Z (in place),
    # then slide horizontally over the basket at that height, then release.
    PICK_LIFT = 0.10

    N_SLATS = 5
    APPLE_COLOR = [0.85, 0.12, 0.10]
    ORANGE_COLOR = [0.95, 0.55, 0.08]
    BELT_COLOR = [0.18, 0.18, 0.20]
    SLAT_COLOR = [0.10, 0.10, 0.12]

    # ---- distractor fruit (Opt 2; spawn-side only; never packed/counted) ----
    DISTRACTOR_ENABLED_DEFAULT = False
    DISTRACTOR_PROB_DEFAULT = 0.35        # per real spawn-wave chance of also spawning a distractor
    DISTRACTOR_COLOR_DEFAULT = [0.05, 0.05, 0.05]  # black; distinct from apple/orange
    # min center-to-center Y gap from any active same-belt real fruit, as a
    # multiple of fruit diameter (2*FRUIT_R) — "at least twice the fruit's size"
    DISTRACTOR_MIN_GAP_MULT_DEFAULT = 2.0

    # fruit type -> owning side / arm / basket
    TYPE_SIDE = {"apple": "left", "orange": "right"}

    def setup_demo(self, **kwags):
        self._cfg = kwags.get("task_args", {}).get("packing", {})
        # guards: _update_kinematic_tasks runs before load_actors finishes
        self._belt_ready = False
        self._belt_running = False
        super()._init_task_env_(**kwags)

    # --------------------------------------------------------------- actors
    def load_actors(self):
        cfg = self._cfg
        n_per = cfg.get("n_per_color", None)
        if n_per is not None:
            self.n_apple = self.n_orange = int(n_per)
        else:
            self.n_apple = int(cfg.get("n_apple", self.N_PER_COLOR_DEFAULT))
            self.n_orange = int(cfg.get("n_orange", self.N_PER_COLOR_DEFAULT))
        # legacy total override (split evenly when per-color not set)
        if "n_items" in cfg and n_per is None and "n_apple" not in cfg and "n_orange" not in cfg:
            total = int(cfg["n_items"])
            self.n_apple = total // 2
            self.n_orange = total - self.n_apple
        self.n_items = int(self.n_apple + self.n_orange)

        mode = str(cfg.get("spawn_mode", self.SPAWN_MODE_DEFAULT)).lower().strip()
        if mode in ("opt1", "mixed", "randomized", "rand"):
            mode = "random"
        elif mode in ("simultaneous", "dual", "both"):
            mode = "parallel"
        elif mode in ("one", "sequential"):
            mode = "single"
        if mode not in ("single", "parallel", "random"):
            mode = self.SPAWN_MODE_DEFAULT
        self.spawn_mode = mode
        self.pick_lift = float(cfg.get("pick_lift", self.PICK_LIFT))
        self.spawn_delay_s = float(cfg.get("spawn_delay_s", self.SPAWN_DELAY_S_DEFAULT))
        self.pair_stagger_enabled = bool(cfg.get("pair_stagger_enabled", self.PAIR_STAGGER_ENABLED_DEFAULT))
        # optional max Y gap (m); None / omitted → fruit diameter at spawn time
        _stagger_raw = cfg.get("pair_stagger_y", self.PAIR_STAGGER_Y_DEFAULT)
        self.pair_stagger_y_max = None if _stagger_raw is None else float(_stagger_raw)
        self._pair_stagger_y = 0.0  # last sampled per-wave gap (see _spawn_wave_pair)
        self.single_wave_any_belt = bool(cfg.get("single_wave_any_belt", self.SINGLE_WAVE_ANY_BELT_DEFAULT))
        # Opt 2: black distractor fruit (never packed)
        _dist = cfg.get("distractor_enabled", cfg.get("opt2", self.DISTRACTOR_ENABLED_DEFAULT))
        self.distractor_enabled = bool(_dist)
        self.distractor_prob = float(cfg.get("distractor_prob", self.DISTRACTOR_PROB_DEFAULT))
        self.distractor_color = list(cfg.get("distractor_color", self.DISTRACTOR_COLOR_DEFAULT))[:3]
        self.distractor_min_gap_mult = float(cfg.get("distractor_min_gap_mult", self.DISTRACTOR_MIN_GAP_MULT_DEFAULT))

        self.belt_gap = float(cfg.get("belt_gap", self.BELT_GAP_DEFAULT))
        # shared default speed; optional per-side overrides (belt_speed_left / belt_speed_right)
        default_speed = float(cfg.get("belt_speed", self.BELT_SPEED_DEFAULT))
        speed_jitter = float(cfg.get("belt_speed_jitter", self.BELT_SPEED_JITTER_DEFAULT))
        speed_jitter = float(np.clip(speed_jitter, 0.0, 0.95))
        lo, hi = 1.0 - speed_jitter, 1.0 + speed_jitter
        # Per-episode sample around each belt's nominal (±jitter, default ±20%).
        self.belt_speed = {
            "left": float(cfg.get("belt_speed_left", default_speed)) * float(np.random.uniform(lo, hi)),
            "right": float(cfg.get("belt_speed_right", default_speed)) * float(np.random.uniform(lo, hi)),
        }
        self.advance_every = int(cfg.get("advance_every", self.ADVANCE_EVERY_DEFAULT))
        self.spawn_gap = float(cfg.get("spawn_gap", self.SPAWN_GAP_DEFAULT))
        self.pick_y = float(cfg.get("pick_y", self.PICK_Y))
        self.pick_y_end = float(cfg.get("pick_y_end", self.PICK_Y_END))
        self.pick_station_y = float(cfg.get("pick_station_y", self.PICK_STATION_Y))
        self.basket_y = float(cfg.get("basket_y", cfg.get("box_y", self.BASKET_Y)))
        self.basket_scale = float(cfg.get("basket_scale", self.BASKET_SCALE))
        self.fruit_scale = float(cfg.get("fruit_scale", self.FRUIT_SCALE))
        # half-size along the axis that becomes world-z under FRUIT_Q
        s = self._APPLE_AUTHOR_SCALE * self.fruit_scale
        self.fruit_r = 0.5 * self._APPLE_EXTENT_Y * s

        z0 = 0.74 + self.table_z_bias
        self.table_top = z0
        self.belt_top_z = z0 + 2.0 * self.BELT_THICK
        # seat mesh bottom on the belt (not actor origin + sphere radius)
        z_bottom_from_pose = self._APPLE_CENTER_Y * s - 0.5 * self._APPLE_EXTENT_Y * s
        self._fruit_ride_z = self.belt_top_z - z_bottom_from_pose + 0.002

        half_gap = 0.5 * self.belt_gap
        self.belt_cx = {
            "left":  -(half_gap + self.BELT_HALF_WID),
            "right": +(half_gap + self.BELT_HALF_WID),
        }

        # ---- two belt slabs + moving slats ----
        self.belts = {}
        self.slats = {"left": [], "right": []}
        self._slat_ys = {"left": [], "right": []}
        self._slat_spacing = (2.0 * self.BELT_HALF_LEN) / self.N_SLATS
        self._slat_near = self.BELT_Y - self.BELT_HALF_LEN
        for side, cx in self.belt_cx.items():
            belt = create_box(
                scene=self,
                pose=sapien.Pose([cx, self.BELT_Y, z0 + self.BELT_THICK], [1, 0, 0, 0]),
                half_size=[self.BELT_HALF_WID, self.BELT_HALF_LEN, self.BELT_THICK],
                color=self.BELT_COLOR,
                name=f"belt_{side}",
                is_static=True,
            )
            self.belts[side] = belt
            self.add_prohibit_area(belt, padding=0.02)
            for k in range(self.N_SLATS):
                sy = self._slat_near + k * self._slat_spacing
                sl = create_box(
                    scene=self,
                    pose=sapien.Pose([cx, sy, self.belt_top_z + 0.001], [1, 0, 0, 0]),
                    half_size=[self.BELT_HALF_WID * 0.92, 0.006, 0.003],
                    color=self.SLAT_COLOR,
                    name=f"slat_{side}_{k}",
                    is_static=True,
                )
                self.slats[side].append(sl)
                self._slat_ys[side].append(sy)

        # ---- breadbaskets: apple left / orange right ----
        self.basket_id = int(np.random.choice([0, 1, 2, 3, 4]))
        self.basket_x = float(cfg.get("basket_x", self.BASKET_X))
        self.basket_centers = {
            "apple":  np.array([-self.basket_x, self.basket_y], dtype=np.float64),
            "orange": np.array([+self.basket_x, self.basket_y], dtype=np.float64),
        }
        self.baskets = {}
        self.basket_base_z = {}
        self.basket_top_z = {}
        for ftype, center in self.basket_centers.items():
            basket = create_actor(
                self,
                pose=sapien.Pose(
                    [float(center[0]), float(center[1]), z0],
                    [0.5, 0.5, 0.5, 0.5],
                ),
                modelname="076_breadbasket",
                model_id=self.basket_id,
                convex=True,
                is_static=True,
                scale_mult=self.basket_scale,
            )
            # tint basket rim toward fruit color for readability
            tint = self.APPLE_COLOR if ftype == "apple" else self.ORANGE_COLOR
            self._recolor(basket, [0.55 * tint[0] + 0.35,
                                   0.55 * tint[1] + 0.30,
                                   0.55 * tint[2] + 0.20])
            self.baskets[ftype] = basket
            self.basket_base_z[ftype] = float(basket.get_pose().p[2])
            # the [0.5,0.5,0.5,0.5] pose orientation maps local +y (the
            # mesh's authored "up"/height axis, origin at the base) onto
            # world +z, so world rim height above the origin is simply
            # extents[1] * scale[1] from the (already scale_mult-applied)
            # model config. Used so the carry/hover target clears the
            # actual basket wall instead of the old fixed guess, which sat
            # below the rim and caused the gripper to clip the basket.
            cfg = getattr(basket, "config", None) or {}
            extents = cfg.get("extents", [0.0, 0.7, 0.0])
            scale = cfg.get("scale", [self.basket_scale] * 3)
            basket_height = float(extents[1]) * float(scale[1])
            if basket_height <= 0.0:
                basket_height = 0.07 * self.basket_scale
            self.basket_top_z[ftype] = self.basket_base_z[ftype] + basket_height
            self.add_prohibit_area(basket, padding=0.04)

        # ---- fruit sequence: fixed per-color counts, shuffled order ----
        types = (["apple"] * self.n_apple) + (["orange"] * self.n_orange)
        np.random.shuffle(types)
        self.item_types = [str(t) for t in types]
        if self.spawn_mode == "single":
            # either color on either belt; the color-matched arm reaches to that belt
            self.item_sides = [
                str(np.random.choice(["left", "right"])) for _ in self.item_types
            ]
        else:
            # apples left / oranges right
            self.item_sides = [self.TYPE_SIDE[t] for t in self.item_types]
        # arm that packs this fruit (color-matched); may differ from belt side in single mode
        self.item_arms = [self.TYPE_SIDE[t] for t in self.item_types]

        # stage all fruits off-table; they appear gradually on the belts
        self.items = []
        self._item_comps = []
        self._item_y = [None] * self.n_items       # None = not yet on belt / packed
        self._item_roll = [0.0] * self.n_items
        self._spawned_mask = [False] * self.n_items
        self._spawned = 0
        self._packed = [False] * self.n_items
        self._missed = [False] * self.n_items  # rode off belt end without packing
        # gripper has reached the drop pose above this fruit's basket
        self._over_basket = [False] * self.n_items
        self._place_counts = {"apple": 0, "orange": 0}
        self._welded = [False] * self.n_items
        self._weld_offset = [None] * self.n_items
        self._weld_arm = [None] * self.n_items
        # wave-partner tracking: set by _spawn_wave_pair so a staggered pair
        # (one fruit given a small head start) still gets routed through
        # _pack_pair instead of being solo-packed the instant only the lead
        # fruit enters the ready window (see _active_pair_partner)
        self._pair_partner = [None] * self.n_items
        self._grasping_idxs = set()  # fruits mid-intercept; stay on the moving stream
        # nestable hold for the pick→above-basket→drop→return cycle; blocks new spawns
        self._spawn_hold_depth = 0
        self._stage_pose = sapien.Pose([0.0, 1.2, z0 + 0.4], [1, 0, 0, 0])

        for i, ftype in enumerate(self.item_types):
            rgb = self.APPLE_COLOR if ftype == "apple" else self.ORANGE_COLOR
            fruit = create_actor(
                self,
                pose=sapien.Pose(
                    [self._stage_pose.p[0] + 0.03 * i,
                     self._stage_pose.p[1],
                     self._stage_pose.p[2]],
                    self.FRUIT_Q,
                ),
                modelname=self.FRUIT_MODEL,
                model_id=0,
                convex=True,
                is_static=False,
                scale_mult=self.fruit_scale,
            )
            self._recolor(fruit, rgb)
            fruit.set_mass(self.FRUIT_MASS)
            comp = None
            for c in fruit.actor.get_components():
                if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                    comp = c
                    try:
                        c.set_linear_damping(5.0)
                        c.set_angular_damping(20.0)
                        for s in c.get_collision_shapes():
                            m = s.get_physical_material()
                            m.set_static_friction(4.0)
                            m.set_dynamic_friction(4.0)
                            m.set_restitution(0.0)
                    except Exception:
                        pass
                    c.set_kinematic(True)
                    c.set_disable_gravity(True)
            self.items.append(fruit)
            self._item_comps.append(comp)

        # ---- distractor fruits: same mesh/scale as real fruit but recolored
        # brown, pre-staged off-table like self.items above. Tracked in their
        # OWN lists (never self.items/_item_*) so no packing/grasp/success
        # code path (which only ever iterates self.items) can see them.
        self.n_distractor_slots = int(self.n_items) if self.distractor_enabled else 0
        self.distractors = []
        self._distractor_comps = []
        self._distractor_y = [None] * self.n_distractor_slots
        self._distractor_roll = [0.0] * self.n_distractor_slots
        self._distractor_side = [None] * self.n_distractor_slots
        for s in range(self.n_distractor_slots):
            distractor = create_actor(
                self,
                pose=sapien.Pose(
                    [self._stage_pose.p[0] + 0.03 * s + 0.5,
                     self._stage_pose.p[1],
                     self._stage_pose.p[2]],
                    self.FRUIT_Q,
                ),
                modelname=self.FRUIT_MODEL,
                model_id=0,
                convex=True,
                is_static=False,
                scale_mult=self.fruit_scale,
            )
            self._recolor(distractor, self.distractor_color)
            distractor.set_mass(self.FRUIT_MASS)
            d_comp = None
            for c in distractor.actor.get_components():
                if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                    d_comp = c
                    try:
                        c.set_linear_damping(5.0)
                        c.set_angular_damping(20.0)
                        for sh in c.get_collision_shapes():
                            m = sh.get_physical_material()
                            m.set_static_friction(4.0)
                            m.set_dynamic_friction(4.0)
                            m.set_restitution(0.0)
                    except Exception:
                        pass
                    c.set_kinematic(True)
                    c.set_disable_gravity(True)
            self.distractors.append(distractor)
            self._distractor_comps.append(d_comp)

        self._step_ctr = 0
        self._pic_ctr = 0
        self._belt_ready = True
        self._belt_running = False

    # ------------------------------------------------------------- rendering
    def _recolor(self, actor, rgb):
        for c in actor.actor.get_components():
            if isinstance(c, sapien.render.RenderBodyComponent):
                for s in c.render_shapes:
                    try:
                        s.material.set_base_color([*list(rgb)[:3], 1.0])
                    except Exception:
                        pass

    # ------------------------------------------------------------- belt sim
    def _hide(self, idx):
        self.items[idx].actor.set_pose(sapien.Pose(
            [self._stage_pose.p[0] + 0.05 * idx,
             self._stage_pose.p[1] + 0.5,
             self.HIDE_Z],
            [1, 0, 0, 0],
        ))
        self._item_y[idx] = None

    def _spawn(self, idx, y_offset=0.0):
        """Place fruit idx onto the far end of its matching belt.

        ``y_offset`` (<=0) gives a pair-wave partner a small head start —
        it spawns that much closer to the pick station (instead of both
        fruits starting at exactly ``BELT_Y_FAR``) so the pair doesn't ride
        in perfect lockstep (see ``_spawn_wave_pair`` / ``_pair_stagger_y``).
        Only ever negative so spawns stay within the belt's physical
        length (never past ``BELT_Y_FAR``).
        """
        side = self.item_sides[idx]
        y0 = self.BELT_Y_FAR + float(y_offset)
        self._item_y[idx] = y0
        self._item_roll[idx] = 0.0
        self._over_basket[idx] = False
        comp = self._item_comps[idx]
        if comp is not None:
            comp.set_kinematic(True)
            comp.set_disable_gravity(True)
        self._set_fruit_pose(
            idx, self.belt_cx[side], y0, self._fruit_ride_z
        )

    def _spawn_wave_pair(self):
        """Spawn one fruit per belt side together as a pair wave.

        When ``pair_stagger_enabled`` is set, one of the two fruits (picked
        at random) gets a head start toward the pick station. The gap is
        sampled per wave from U(0, max), where max defaults to one fruit
        diameter (``2 * fruit_r``) so the pair may start aligned or offset
        by up to one fruit. ``_wait_pair_at_station`` widens its arrival
        window by the same sampled amount.
        """
        picked = []
        for i in range(self.n_items):
            if self._spawned_mask[i] or self._packed[i] or self._missed[i]:
                continue
            side = self.item_sides[i]
            if any(self.item_sides[j] == side for j in picked):
                continue
            picked.append(i)
            if len(picked) >= 2:
                break
        if self.pair_stagger_enabled:
            stagger_max = (
                self.pair_stagger_y_max
                if self.pair_stagger_y_max is not None
                else (2.0 * float(self.fruit_r))
            )
            stagger = float(np.random.uniform(0.0, max(0.0, stagger_max)))
        else:
            stagger = 0.0
        self._pair_stagger_y = stagger
        leading_k = 1 if bool(np.random.rand() < 0.5) else 0
        for k, i in enumerate(picked):
            y_offset = -stagger if k == leading_k else 0.0
            self._spawn(i, y_offset=y_offset)
            self._spawned_mask[i] = True
        if len(picked) == 2:
            self._pair_partner[picked[0]] = picked[1]
            self._pair_partner[picked[1]] = picked[0]
        self._spawned = int(sum(self._spawned_mask))

    def _active_pair_partner(self, idx):
        """Return idx's still-outstanding wave-partner, or None.

        Lets the caller keep routing a staggered pair through ``_pack_pair``
        (which tolerates a bounded arrival gap via ``_wait_pair_at_station``)
        even when the head-started fruit enters the ready window well
        before its partner does.
        """
        j = self._pair_partner[idx]
        if j is None:
            return None
        if self._packed[j] or self._missed[j] or j in self._grasping_idxs:
            return None
        if not self._spawned_mask[j] or self._item_y[j] is None:
            return None
        return j

    def _set_fruit_pose(self, idx, x, y, z, roll=0.0):
        # keep the authored grasp orientation (FRUIT_Q); optional spin about world -y
        # is composed on top so the contact frame stays top-down-approachable
        base_q = np.array(self.FRUIT_Q, dtype=np.float64)
        if abs(roll) > 1e-6:
            # spin about world x (belt travel axis) while preserving top-down graspability poorly;
            # use identity spin — visual motion comes from translation + slats
            q = base_q
        else:
            q = base_q
        pose = sapien.Pose([float(x), float(y), float(z)], q.tolist())
        # always write the entity pose so grasp/contact queries see the true location
        self.items[idx].actor.set_pose(pose)
        comp = self._item_comps[idx]
        if comp is not None:
            try:
                comp.set_kinematic_target(pose)
            except Exception:
                pass

    def _advance_slats(self):
        span = self.N_SLATS * self._slat_spacing
        for side, cx in self.belt_cx.items():
            speed = self.belt_speed[side]
            for k in range(self.N_SLATS):
                y = self._slat_ys[side][k] - speed
                if y < self._slat_near:
                    y += span
                self._slat_ys[side][k] = y
                self.slats[side][k].actor.set_pose(
                    sapien.Pose([cx, y, self.belt_top_z + 0.001], [1, 0, 0, 0])
                )

    def _fruit_blocks_spawn(self, idx):
        """True while this fruit must delay the next spawn wave.

        Blocks from appearance until it is packed into a basket or has left the
        belt. Carried / welded fruit always blocks (not packed yet).
        """
        if not self._spawned_mask[idx]:
            return False
        if self._packed[idx] or self._missed[idx]:
            return False
        return True

    def _can_spawn_next(self):
        """Next wave only when no fruit is outstanding and no pack cycle is running."""
        if int(getattr(self, "_spawn_hold_depth", 0)) > 0:
            return False
        if self._has_active_fruit():
            return False
        return True

    def _has_active_fruit(self):
        return any(self._fruit_blocks_spawn(i) for i in range(self.n_items))

    def _begin_spawn_hold(self):
        self._spawn_hold_depth = int(getattr(self, "_spawn_hold_depth", 0)) + 1

    def _end_spawn_hold(self):
        self._spawn_hold_depth = max(0, int(getattr(self, "_spawn_hold_depth", 0)) - 1)

    # -------------------------------------------------------- distractors
    def _hide_distractor(self, slot):
        """Off-table park (mirrors ``_hide``, but for a distractor slot)."""
        self.distractors[slot].actor.set_pose(sapien.Pose(
            [self._stage_pose.p[0] + 0.05 * slot + 0.5,
             self._stage_pose.p[1] + 0.5,
             self.HIDE_Z],
            [1, 0, 0, 0],
        ))
        self._distractor_y[slot] = None
        self._distractor_side[slot] = None

    def _set_distractor_pose(self, slot, x, y, z):
        q = np.array(self.FRUIT_Q, dtype=np.float64)
        pose = sapien.Pose([float(x), float(y), float(z)], q.tolist())
        self.distractors[slot].actor.set_pose(pose)
        comp = self._distractor_comps[slot]
        if comp is not None:
            try:
                comp.set_kinematic_target(pose)
            except Exception:
                pass

    def _spawn_distractor(self):
        """Put one distractor on a randomly chosen belt (independent of
        which belt(s) the current real wave used), honoring the minimum
        center-to-center Y gap (``distractor_min_gap_mult`` * fruit
        diameter) from any active real fruit currently on that SAME belt.
        Since both then ride at that belt's shared speed, a sufficient
        initial gap is preserved for the whole ride. Fully independent of
        ``self.items`` / ``_spawned_mask`` / ``_item_y`` — this never
        touches (and is never touched by) spawn-gating, grasp, or success
        bookkeeping.
        """
        slot = None
        for s in range(self.n_distractor_slots):
            if self._distractor_y[s] is None:
                slot = s
                break
        if slot is None:
            return  # every slot busy this wave; skip

        side = str(np.random.choice(["left", "right"]))
        min_gap = float(self.distractor_min_gap_mult) * (2.0 * self.FRUIT_R)
        active_ys = [
            self._item_y[i] for i in range(self.n_items)
            if self.item_sides[i] == side and self._item_y[i] is not None
        ]
        if active_ys:
            closest_to_far = max(active_ys)
            candidate = closest_to_far + min_gap
            y0 = candidate if candidate <= self.BELT_Y_FAR + 1e-9 else closest_to_far - min_gap
        else:
            y0 = self.BELT_Y_FAR - float(np.random.uniform(0.0, 0.03))

        self._distractor_side[slot] = side
        self._distractor_y[slot] = y0
        self._distractor_roll[slot] = 0.0
        comp = self._distractor_comps[slot]
        if comp is not None:
            comp.set_kinematic(True)
            comp.set_disable_gravity(True)
        self._set_distractor_pose(slot, self.belt_cx[side], y0, self._fruit_ride_z)
        if bool(os.environ.get("PACKING_DEBUG")):
            gap_note = (f"gap_to_nearest={min(abs(y0 - y) for y in active_ys):.4f}"
                        if active_ys else "no active real fruit on belt")
            print(f"[packing]  distractor_{slot} spawn side={side} y0={y0:.4f} "
                  f"min_gap_req={min_gap:.4f} {gap_note}", flush=True)

    def _maybe_spawn_distractor(self):
        """Rolled once per real spawn-wave (see ``_maybe_spawn``)."""
        if not getattr(self, "distractor_enabled", False):
            return
        if not bool(np.random.rand() < self.distractor_prob):
            return
        self._spawn_distractor()

    def _advance_distractors(self):
        """Mirror of the real-fruit belt-ride step below, but fully
        decoupled: no spawn-gating, no grasp/pack interaction, no "missed"
        bookkeeping on despawn — a distractor was never a real item, so it
        just quietly disappears once it rides off the near end.
        """
        for s in range(self.n_distractor_slots):
            if self._distractor_y[s] is None:
                continue
            side = self._distractor_side[s]
            speed = self.belt_speed[side]
            self._distractor_y[s] -= speed
            if self._distractor_y[s] < self.BELT_Y_NEAR:
                if bool(os.environ.get("PACKING_DEBUG")):
                    print(f"[packing]  distractor_{s} left belt — despawn", flush=True)
                self._hide_distractor(s)
                continue
            self._distractor_roll[s] += speed / max(self.fruit_r, 1e-4)
            self._set_distractor_pose(s, self.belt_cx[side], self._distractor_y[s], self._fruit_ride_z)

    def _despawn_off_belt(self, idx):
        """Fruit reached the near end without a pick — hide it and free the wave."""
        import os
        if self._packed[idx] or self._missed[idx]:
            return
        self._missed[idx] = True
        self._over_basket[idx] = False
        self._item_y[idx] = None
        self._welded[idx] = False
        self._grasping_idxs.discard(idx)
        self._hide(idx)
        if bool(os.environ.get("PACKING_DEBUG")):
            print(f"[packing]  {self.item_types[idx]}_{idx} left belt — despawn",
                  flush=True)

    def _maybe_spawn(self):
        """Spawn the next fruit / pair only after the current wave is fully clear."""
        import os
        if self._spawned >= self.n_items:
            return
        # wait until drop/despawn of the current wave AND the pack cycle finished
        # (gripper has been above the basket and returned — not mid-pick)
        if not self._can_spawn_next():
            return

        spawned_wave = False

        if self.spawn_mode == "single":
            for i in range(self.n_items):
                if self._spawned_mask[i] or self._packed[i] or self._missed[i]:
                    continue
                self._spawn(i)
                self._spawned_mask[i] = True
                self._spawned = int(sum(self._spawned_mask))
                spawned_wave = True
                break

        elif self.spawn_mode == "random":
            # each wave independently rolls single-vs-pair (falls back to
            # single automatically if only one color remains outstanding)
            want_pair = bool(np.random.rand() < 0.5)
            if bool(os.environ.get("PACKING_DEBUG")):
                print(f"[packing]  random spawn wave: "
                      f"{'pair' if want_pair else 'single'}", flush=True)
            if want_pair:
                self._spawn_wave_pair()
                spawned_wave = True
            else:
                for i in range(self.n_items):
                    if self._spawned_mask[i] or self._packed[i] or self._missed[i]:
                        continue
                    if self.single_wave_any_belt:
                        old_side = self.item_sides[i]
                        self.item_sides[i] = str(np.random.choice(["left", "right"]))
                        if bool(os.environ.get("PACKING_DEBUG")) and self.item_sides[i] != old_side:
                            print(f"[packing]  single wave belt override: "
                                  f"{self.item_types[i]}_{i} {old_side} -> {self.item_sides[i]}",
                                  flush=True)
                    self._spawn(i)
                    self._spawned_mask[i] = True
                    self._spawned = int(sum(self._spawned_mask))
                    spawned_wave = True
                    break

        else:
            # parallel: one apple (left) + one orange (right) per wave, together
            self._spawn_wave_pair()
            spawned_wave = True

        if spawned_wave:
            self._maybe_spawn_distractor()

    def _advance_stream(self):
        self._maybe_spawn()

        for i in range(self.n_items):
            if (not self._spawned_mask[i] or self._packed[i] or self._missed[i]
                    or self._welded[i] or self._item_y[i] is None):
                continue
            side = self.item_sides[i]
            speed = self.belt_speed[side]
            # continuous motion — never park / pause on the belt
            self._item_y[i] -= speed
            if self._item_y[i] < self.BELT_Y_NEAR and i not in self._grasping_idxs:
                # leave the belt (do not wrap / reappear)
                self._despawn_off_belt(i)
                continue
            self._item_roll[i] += speed / max(self.fruit_r, 1e-4)
            self._set_fruit_pose(
                i, self.belt_cx[side], self._item_y[i], self._fruit_ride_z,
                roll=self._item_roll[i],
            )

        self._advance_distractors()

    def _ee_pos(self, arm):
        p = (self.robot.get_left_ee_pose() if arm == "left"
             else self.robot.get_right_ee_pose())
        return np.array(p[:3], dtype=float)

    def _ee_pose_full(self, arm):
        """Full 6-DOF planning-EE pose (position + orientation), as a sapien.Pose."""
        p = (self.robot.get_left_ee_pose() if arm == "left"
             else self.robot.get_right_ee_pose())
        return sapien.Pose(list(p[:3]), list(p[3:7]))

    def _tcp_pos(self, arm):
        """Gripper-center pose (fingertips), not the retracted planning EE frame."""
        p = (self.robot.get_left_tcp_pose() if arm == "left"
             else self.robot.get_right_tcp_pose())
        return np.array(p[:3], dtype=float)

    def _weld_fruit_to_ee(self, idx, arm):
        """Rigidly attach fruit to the EE so lifts/carries cannot slip or rotate.

        The offset is captured once, as a full 6-DOF pose (position +
        orientation) expressed in the planning-EE's own local frame:
        ``local_offset = ee_pose.inv() * fruit_pose``. Every subsequent step,
        ``_update_welded_fruits`` recomputes ``ee_pose_now * local_offset`` —
        a proper rigid-body transform composition, not a world-frame
        translation — so the fruit stays glued to the gripper (zero slip,
        zero relative rotation) even while the wrist reorients during the
        lift/slide.
        """
        arm_name = "left" if str(arm) == "left" else "right"
        self._weld_arm[idx] = arm_name
        ee_pose = self._ee_pose_full(arm_name)
        fruit_pose = self.items[idx].get_pose()
        self._weld_offset[idx] = ee_pose.inv() * fruit_pose
        rigid = self._item_comps[idx]
        if rigid is not None:
            rigid.set_disable_gravity(True)
            rigid.set_kinematic(True)
            try:
                rigid.set_linear_velocity(np.zeros(3))
                rigid.set_angular_velocity(np.zeros(3))
            except Exception:
                pass
        # leave the belt stream; fruit now tracks the gripper
        # (still "active" until dropped in the basket — blocks next spawn)
        self._item_y[idx] = None
        self._welded[idx] = True
        # reset the JERK baseline so a stale pre-release position (from a
        # previous weld cycle on this same fruit index, e.g. after a
        # miss->resend->re-pick loop) isn't diffed against the fresh
        # post-attach position and misreported as a physical jolt
        if not hasattr(self, "_dbg_last_fruit_p"):
            self._dbg_last_fruit_p = {}
        self._dbg_last_fruit_p[idx] = np.array(self.items[idx].get_pose().p, dtype=float)

    def _update_welded_fruits(self):
        """Re-glue every welded fruit to its gripper's current pose.

        Called every physics step (see ``_update_kinematic_tasks``), including
        during arm motion, waits, and dwells — not just once at attach time —
        so the fruit rigidly tracks the full gripper pose (no drift/wobble)
        for the entire carry until ``_release_fruit``.

        The weld reads ``ee_pose`` from ``_ee_pose_full``, which is the EE
        link's *actual simulated* global pose (``left_ee.global_pose`` /
        ``right_ee.global_pose``), not the planned trajectory waypoint. If a
        contact force (e.g. wrist vs. basket rim) perturbs the real link
        pose even slightly, the welded fruit inherits that perturbation
        one-for-one, every step. PACKING_DEBUG=1 flags any single-step fruit
        position jump above ``_JERK_THRESH`` so a physical "contact knocked
        it loose"-looking event can be told apart from a normal smooth move.
        """
        if not getattr(self, "_welded", None):
            return
        dbg = bool(os.environ.get("PACKING_DEBUG"))
        for i in range(self.n_items):
            if not self._welded[i]:
                continue
            ee_pose = self._ee_pose_full(self._weld_arm[i])
            pose = ee_pose * self._weld_offset[i]
            if dbg:
                prev = getattr(self, "_dbg_last_fruit_p", {}).get(i)
                newp = np.array(pose.p, dtype=float)
                if prev is not None:
                    jump = float(np.linalg.norm(newp - prev))
                    if jump > 0.008:  # > 8mm in one physics step is not a smooth glide
                        print(f"[packing]  JERK fruit_{i} step={self._step_ctr} "
                              f"jump={jump:.4f} prev={prev.round(4)} new={newp.round(4)}",
                              flush=True)
                if not hasattr(self, "_dbg_last_fruit_p"):
                    self._dbg_last_fruit_p = {}
                self._dbg_last_fruit_p[i] = newp
            self.items[i].actor.set_pose(pose)
            rigid = self._item_comps[i]
            if rigid is not None:
                try:
                    rigid.set_kinematic_target(pose)
                except Exception:
                    pass

    def _release_fruit(self, idx):
        """Un-weld so the fruit can drop into the basket under gravity."""
        if bool(os.environ.get("PACKING_DEBUG")):
            p = np.array(self.items[idx].get_pose().p, dtype=float)
            print(f"[packing]  RELEASE fruit_{idx} step={self._step_ctr} p={p.round(4)}", flush=True)
        self._welded[idx] = False
        rigid = self._item_comps[idx]
        if rigid is not None:
            try:
                rigid.set_kinematic(False)
                rigid.set_disable_gravity(False)
                rigid.set_linear_velocity(np.zeros(3))
                rigid.set_angular_velocity(np.zeros(3))
            except Exception:
                pass

    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
        if not getattr(self, "_belt_ready", False):
            return
        # welded fruit tracks the EE every physics step (including during arm moves)
        self._update_welded_fruits()
        if bool(os.environ.get("PACKING_DEBUG")):
            self._accumulate_basket_contacts()
        if not getattr(self, "_belt_running", False):
            return
        self._step_ctr += 1
        if self._step_ctr % max(1, self.advance_every) == 0:
            self._advance_slats()
            self._advance_stream()

    def _belt_dwell(self, steps):
        """Advance belts + stream for `steps` while recording frames. Belt stays on."""
        self._belt_running = True
        for _ in range(int(steps)):
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (self._pic_ctr % self.save_freq == 0):
                self._take_picture()
            self._pic_ctr += 1

    # ------------------------------------------------------------- packing
    def _fruit_in_basket(self, idx):
        ftype = self.item_types[idx]
        p = np.array(self.items[idx].get_pose().p, dtype=np.float64)
        c = self.basket_centers[ftype]
        in_xy = float(np.linalg.norm(p[:2] - c)) <= self.BASKET_CATCH_R
        above = p[2] >= (self.basket_base_z[ftype] - 0.02)
        below = p[2] <= (self.basket_base_z[ftype] + 0.18)
        return bool(in_xy and above and below)

    def _mark_packed(self, idx):
        ftype = self.item_types[idx]
        if not self._packed[idx]:
            self._place_counts[ftype] = self._place_counts[ftype] + 1
        self._packed[idx] = True
        self._over_basket[idx] = True
        self._item_y[idx] = None
        self._welded[idx] = False
        rigid = self._item_comps[idx]
        if rigid is not None:
            try:
                rigid.set_kinematic(True)
                rigid.set_disable_gravity(True)
            except Exception:
                pass

    def _restore_fruit_stream_pose(self, idx):
        """Put actor back on the live belt pose after a temporary planning teleport."""
        if self._item_y[idx] is None:
            return
        side = self.item_sides[idx]
        self._set_fruit_pose(
            idx, self.belt_cx[side], self._item_y[idx], self._fruit_ride_z,
            roll=self._item_roll[idx],
        )

    def _basket_target_xy(self, idx, slot_offset=0, basket=None):
        """Drop pose in a basket; ``basket`` overrides the fruit's own color."""
        ftype = basket or self.item_types[idx]
        slot = self._place_counts[ftype] + int(slot_offset)
        c = self.basket_centers[ftype]
        offsets = [
            (0.0, 0.0), (0.028, 0.017), (-0.028, 0.017),
            (0.028, -0.017), (-0.028, -0.017),
        ]
        ox, oy = offsets[slot % len(offsets)]
        return c + np.array([ox, oy], dtype=float)

    def _plan_station_pre(self, idx, arm):
        side = self.item_sides[idx]
        cx = self.belt_cx[side]
        self._set_fruit_pose(idx, cx, self.pick_station_y, self._fruit_ride_z)
        try:
            # hover close enough that a short descend can reach the attach distance
            pre_pose, _ = self.choose_grasp_pose(
                self.items[idx], arm_tag=arm, pre_dis=0.05, target_dis=0.0,
            )
        finally:
            self._restore_fruit_stream_pose(idx)
        return pre_pose

    def _wait_fruit_at_station(self, idx, side):
        """Dwell until the fruit nears the pick station (belt keeps moving)."""
        import os
        dbg = bool(os.environ.get("PACKING_DEBUG"))
        speed = max(self.belt_speed[side], 1e-6)
        arrive_lead = 60.0 * speed  # small lead so the reach can still catch it
        max_wait = int((self.BELT_Y_FAR - self.BELT_Y_NEAR) / speed) + 80
        for _ in range(max_wait):
            y = self._item_y[idx]
            if y is None:
                return False
            if y < self.pick_y_end:
                if dbg:
                    print(f"[packing]  fruit passed station y={y:.3f}", flush=True)
                return False
            if y <= self.pick_station_y + arrive_lead:
                return True
            self._belt_dwell(max(1, self.advance_every))
        return False

    def _wait_pair_at_station(self, idx_l, idx_r):
        """Dwell until both fruits near the pick station together.

        Widens its arrival window by the last sampled ``_pair_stagger_y`` when
        the pair-gap option is on, so a staggered pair (one fruit deliberately
        trailing the other on the belt — see ``_spawn_wave_pair``) is still
        treated as "arrived together" instead of forcing extra dwell time that
        could carry the lead fruit past ``pick_y_end``.
        """
        speed = max(min(self.belt_speed.values()), 1e-6)
        stagger = self._pair_stagger_y if self.pair_stagger_enabled else 0.0
        arrive_lead = 60.0 * speed + stagger
        max_wait = int((self.BELT_Y_FAR - self.BELT_Y_NEAR) / speed) + 100
        for _ in range(max_wait):
            yl, yr = self._item_y[idx_l], self._item_y[idx_r]
            if yl is None or yr is None:
                return False
            if yl < self.pick_y_end or yr < self.pick_y_end:
                return False
            if yl <= self.pick_station_y + arrive_lead and yr <= self.pick_station_y + arrive_lead:
                return True
            self._belt_dwell(max(1, self.advance_every))
        return False

    def _tcp_near_fruit(self, idx, arm):
        """True if the gripper has approached close enough to attach (~2 cm)."""
        arm_name = "left" if str(arm) == "left" else "right"
        tcp = self._tcp_pos(arm_name)
        fp = np.array(self.items[idx].get_pose().p, dtype=float)
        xy = float(np.linalg.norm(fp[:2] - tcp[:2]))
        dz = float(tcp[2] - fp[2])
        return xy <= self.ATTACH_XY and 0.0 <= dz <= self.ATTACH_Z_MAX

    def _fruit_gripper_dist(self, idx, arm):
        arm_name = "left" if str(arm) == "left" else "right"
        tcp = self._tcp_pos(arm_name)
        fp = np.array(self.items[idx].get_pose().p, dtype=float)
        xy = float(np.linalg.norm(fp[:2] - tcp[:2]))
        dz = float(tcp[2] - fp[2])
        return xy, dz

    def _plan_final_grasp(self, idx, arm):
        """Grasp pose right at the fruit's current position (no teleport)."""
        try:
            _, grasp_pose = self.choose_grasp_pose(
                self.items[idx], arm_tag=arm, pre_dis=0.05, target_dis=0.0,
            )
        except Exception:
            return None
        return grasp_pose

    def _attach_fruit_to_gripper(self, idx, arm):
        """Place fruit between the fingers and weld (no gripper close — fast)."""
        import os
        dbg = bool(os.environ.get("PACKING_DEBUG"))
        arm_name = "left" if str(arm) == "left" else "right"
        tcp = self._tcp_pos(arm_name)
        pos = tcp.copy()
        pos[2] -= 0.015
        self._item_y[idx] = None
        self.items[idx].actor.set_pose(sapien.Pose(pos.tolist(), self.FRUIT_Q))
        rigid = self._item_comps[idx]
        if rigid is not None:
            try:
                rigid.set_kinematic(True)
                rigid.set_disable_gravity(True)
                rigid.set_linear_velocity(np.zeros(3))
                rigid.set_angular_velocity(np.zeros(3))
                rigid.set_kinematic_target(
                    sapien.Pose(pos.tolist(), self.FRUIT_Q)
                )
            except Exception:
                pass
        self._weld_fruit_to_ee(idx, arm)
        if dbg:
            print(f"[packing]  attach {self.item_types[idx]}_{idx} at "
                  f"{arm_name} tcp={np.round(tcp, 3)}", flush=True)

    def _close_on_attached(self, *arm_idx_pairs):
        """Close gripper(s) on already-attached fruit and re-seat the weld."""
        acts = []
        for arm, idx in arm_idx_pairs:
            if idx is None:
                continue
            acts.append(self.close_gripper(arm, pos=0.0))
        if not acts:
            return
        self.plan_success = True
        if len(acts) == 1:
            self.move(acts[0])
        else:
            self.move(*acts)
        self.plan_success = True
        for arm, idx in arm_idx_pairs:
            if idx is None:
                continue
            arm_name = "left" if str(arm) == "left" else "right"
            tcp = self._tcp_pos(arm_name)
            pos = tcp.copy()
            pos[2] -= 0.015
            self.items[idx].actor.set_pose(sapien.Pose(pos.tolist(), self.FRUIT_Q))
            self._weld_fruit_to_ee(idx, arm)

    def _snap_fruit_into_gripper(self, idx, arm):
        """Attach fruit then close the gripper (single-arm path)."""
        self._attach_fruit_to_gripper(idx, arm)
        self._close_on_attached((arm, idx))

    def _reach_and_attach(self, idx, arm):
        """Reach straight for the fruit; attach once the gripper ends up close.

        A plain move-to-grasp-pose motion (like reaching for any static
        object) — at most one extra correction reach if the fruit rolled a
        little further while the arm was moving.
        """
        import os
        dbg = bool(os.environ.get("PACKING_DEBUG"))
        for attempt in range(3):
            if self._item_y[idx] is None or self._item_y[idx] < self.pick_y_end:
                return False
            grasp_pose = self._plan_final_grasp(idx, arm)
            if grasp_pose is None:
                if dbg:
                    print("[packing]  no final grasp pose", flush=True)
                return False
            self.plan_success = True
            self.move(self.move_to_pose(arm, grasp_pose))
            self.plan_success = True
            if self._item_y[idx] is not None and self._tcp_near_fruit(idx, arm):
                self._snap_fruit_into_gripper(idx, arm)
                return True
        if dbg:
            print("[packing]  reach finished but not close enough to attach",
                  flush=True)
        return False

    def _reach_and_attach_pair(self, idx_l, idx_r, arm_l, arm_r):
        """Reach for both fruits together; attach whichever ends up close.

        Both arms attach (without closing) as soon as they are near enough,
        then both grippers close together so one fruit never has to wait on
        the other's gripper-close motion.
        """
        import os
        dbg = bool(os.environ.get("PACKING_DEBUG"))
        got_l = got_r = False
        for attempt in range(3):
            need_l = (not got_l and self._item_y[idx_l] is not None
                      and self._item_y[idx_l] >= self.pick_y_end)
            need_r = (not got_r and self._item_y[idx_r] is not None
                      and self._item_y[idx_r] >= self.pick_y_end)
            if not need_l and not need_r:
                break
            pose_l = self._plan_final_grasp(idx_l, arm_l) if need_l else None
            pose_r = self._plan_final_grasp(idx_r, arm_r) if need_r else None
            acts = []
            if pose_l is not None:
                acts.append(self.move_to_pose(arm_l, pose_l))
            if pose_r is not None:
                acts.append(self.move_to_pose(arm_r, pose_r))
            if not acts:
                break
            self.plan_success = True
            self.move(*acts)
            self.plan_success = True
            if need_l and self._item_y[idx_l] is not None and self._tcp_near_fruit(idx_l, arm_l):
                self._attach_fruit_to_gripper(idx_l, arm_l)
                got_l = True
            if need_r and self._item_y[idx_r] is not None and self._tcp_near_fruit(idx_r, arm_r):
                self._attach_fruit_to_gripper(idx_r, arm_r)
                got_r = True

        pairs = []
        if got_l:
            pairs.append((arm_l, idx_l))
        if got_r:
            pairs.append((arm_r, idx_r))
        if pairs:
            self._close_on_attached(*pairs)
        if dbg:
            print(f"[packing]  pair reach done gotL={got_l} gotR={got_r}", flush=True)
        return got_l, got_r

    def _settle_after_drop(self, idx, target_xy, resend_on_miss=True):
        """Mark packed if the fruit fell into the basket; otherwise resend.

        No teleport / "near-miss nudge" — the fruit must land under gravity.
        With ``resend_on_miss`` cleared the fruit stays where it landed, so a
        deliberate mis-sort is final instead of getting another lap.
        """
        import os
        dbg = bool(os.environ.get("PACKING_DEBUG"))
        fruit = self.items[idx]
        side = self.item_sides[idx]
        if self._fruit_in_basket(idx):
            self._mark_packed(idx)
            if dbg:
                print(f"[packing]  dropped in basket "
                      f"p={np.round(fruit.get_pose().p, 3)}", flush=True)
            return
        if not resend_on_miss:
            self._mark_packed(idx)
            if dbg:
                print(f"[packing]  mis-packed {self.item_types[idx]}_{idx} "
                      f"p={np.round(fruit.get_pose().p, 3)}", flush=True)
            return
        p = np.array(fruit.get_pose().p, dtype=float)
        if dbg:
            print(f"[packing]  miss p={np.round(p, 3)} — resend", flush=True)
        rigid = self._item_comps[idx]
        if rigid is not None:
            rigid.set_kinematic(True)
            rigid.set_disable_gravity(True)
        self._over_basket[idx] = False
        self._item_y[idx] = float(self.BELT_Y_FAR)
        self._set_fruit_pose(
            idx, self.belt_cx[side], self.BELT_Y_FAR, self._fruit_ride_z
        )

    def _intercept_and_grasp(self, idx, arm, side):
        """Hover above the belt, wait for the fruit, then reach and attach.

        The fruit never pauses on the belt. The arm hovers over the pick
        station, waits for the fruit to arrive, then does a single reaching
        move to it; once the gripper ends up within ~2 cm, the fruit is
        placed between the fingers and welded.
        """
        import os
        dbg = bool(os.environ.get("PACKING_DEBUG"))

        self.plan_success = True
        self.move(self.open_gripper(arm))
        self.plan_success = True

        if self._item_y[idx] is None or self._item_y[idx] < self.pick_y_end:
            return False

        pre_pose = self._plan_station_pre(idx, arm)
        if pre_pose is None:
            if dbg:
                print("[packing]  no pre-grasp pose at station", flush=True)
            return False

        self.plan_success = True
        ok = self.move(self.move_to_pose(arm, pre_pose))
        self.plan_success = True
        if ok is False:
            if dbg:
                print("[packing]  failed to reach station hover", flush=True)
            return False

        if self._item_y[idx] is None or self._item_y[idx] < self.pick_y_end:
            if dbg:
                print(f"[packing]  missed while approaching station "
                      f"y={self._item_y[idx]}", flush=True)
            return False

        if not self._wait_fruit_at_station(idx, side):
            return False

        return self._reach_and_attach(idx, arm)

    def _accumulate_basket_contacts(self):
        """PACKING_DEBUG-only: called every physics step so transient
        mid-trajectory contacts (not just the contact state at the instant
        a move finishes) are caught between ``_debug_report_basket_contact``
        checkpoints. Uses the full articulation link set (not just
        ``robot.gripper_name``, which is only the finger links) so
        wrist/forearm clipping against the rim shows up too, not just
        fingertip contact.
        """
        if not hasattr(self, "_robot_link_names"):
            self._robot_link_names = set(
                l.get_name() for l in
                (self.robot.left_entity.get_links() + self.robot.right_entity.get_links())
            )
        if not hasattr(self, "_contact_hits_pending"):
            self._contact_hits_pending = set()
        if not hasattr(self, "_contact_min_sep_pending"):
            self._contact_min_sep_pending = None
        if not hasattr(self, "_contact_step_range_pending"):
            self._contact_step_range_pending = None
        basket_names = {b.actor.get_name() for b in self.baskets.values()}
        hit_this_step = False
        for c in self.scene.get_contacts():
            n0 = c.bodies[0].entity.name
            n1 = c.bodies[1].entity.name
            hit_link = None
            if n0 in basket_names and n1 in self._robot_link_names:
                hit_link = n1
            elif n1 in basket_names and n0 in self._robot_link_names:
                hit_link = n0
            if hit_link is None:
                continue
            hit_this_step = True
            self._contact_hits_pending.add(hit_link)
            for pt in c.points:
                sep = float(pt.separation)
                if self._contact_min_sep_pending is None or sep < self._contact_min_sep_pending:
                    self._contact_min_sep_pending = sep
        if hit_this_step:
            s = int(self._step_ctr)
            if self._contact_step_range_pending is None:
                self._contact_step_range_pending = [s, s]
            else:
                self._contact_step_range_pending[0] = min(self._contact_step_range_pending[0], s)
                self._contact_step_range_pending[1] = max(self._contact_step_range_pending[1], s)

    def _debug_reset_basket_contact(self):
        self._contact_hits_pending = set()
        self._contact_min_sep_pending = None
        self._contact_step_range_pending = None
        self._contact_phase_start_step = int(getattr(self, "_step_ctr", 0))

    def _debug_report_basket_contact(self, tag):
        """Report + clear whatever robot-link/basket contacts have
        accumulated (via ``_accumulate_basket_contacts``) since the last
        reset/report. ``min_sep`` is the deepest (most negative = real
        penetration; small positive = within the collision margin but not
        actually overlapping) separation seen across all contact points.
        ``first_step``/``last_step`` (relative to the phase's start step)
        pin down WHEN within the phase's trajectory the contact occurred —
        near 0 means "already touching at the start pose", a value close to
        the phase's total step count means "only at the very end", and
        anything comfortably in between (with start/end themselves clear)
        means the collision happened mid-trajectory (e.g. the interpolated
        joint-space arc dipping the wrist through/near the rim), not at
        either planned endpoint.
        """
        if not bool(os.environ.get("PACKING_DEBUG")):
            return
        hits = getattr(self, "_contact_hits_pending", set())
        min_sep = getattr(self, "_contact_min_sep_pending", None)
        step_range = getattr(self, "_contact_step_range_pending", None)
        phase_start = getattr(self, "_contact_phase_start_step", None)
        phase_len = (int(self._step_ctr) - phase_start) if phase_start is not None else None
        if hits:
            sep_str = f"{min_sep:.4f}" if min_sep is not None else "?"
            if step_range is not None and phase_start is not None:
                rel = f"steps[{step_range[0]-phase_start}:{step_range[1]-phase_start}]/{phase_len}"
            else:
                rel = "steps=?"
            print(f"[packing]  BASKET-CONTACT [{tag}] links={sorted(hits)} min_sep={sep_str} {rel}", flush=True)
        self._contact_hits_pending = set()
        self._contact_min_sep_pending = None
        self._contact_step_range_pending = None

    def _weld_target_ee_pose(self, idx, xyz):
        """EE pose that places the welded fruit at world ``xyz`` (same orientation)."""
        fruit_pose = self.items[idx].get_pose()
        target_fruit = sapien.Pose(np.asarray(xyz, dtype=float).tolist(), list(fruit_pose.q))
        return target_fruit * self._weld_offset[idx].inv()

    def _ik_arm_joints_for_ee(self, arm, ee_pose7):
        """IK joint solution for a planning-EE pose ``[x,y,z,qw,qx,qy,qz]``.

        Uses Curobo ``solve_ik`` (not trajopt). Returns a length-6 numpy
        array, or None if IK fails. Needed because ``move_to_pose``/
        ``move_by_displacement`` silently no-op pure +Z lifts from the
        post-grasp configuration.
        """
        import torch
        from curobo.types.math import Pose as CuroboPose

        arm_name = "left" if str(arm) == "left" else "right"
        trans_target = self.robot._trans_from_gripper_to_endlink(
            list(ee_pose7), arm_tag=arm_name,
        )
        planner = self.robot.left_planner if arm_name == "left" else self.robot.right_planner
        world_target = np.concatenate([np.array(trans_target.p), np.array(trans_target.q)])
        world_base = np.concatenate([
            np.array(planner.robot_origion_pose.p),
            np.array(planner.robot_origion_pose.q),
        ])
        tp_p, tp_q = planner._trans_from_world_to_base(world_base, world_target)
        tp_p = np.array(tp_p, dtype=float)
        tp_q = np.array(tp_q, dtype=float)
        if "aloha-agilex" not in str(getattr(planner, "yml_path", "")):
            tp_p = tp_p + np.array(planner.frame_bias, dtype=float)
        goal = CuroboPose.from_list(list(tp_p) + list(tp_q))
        ik = planner.motion_gen.solve_ik(goal, return_seeds=1)
        if not bool(ik.success.reshape(-1)[0].item()):
            return None
        return ik.solution.detach().cpu().numpy().reshape(-1).astype(float)

    def _drive_arm_joints(self, arm, q_goal, n_steps=50):
        """Smoothly drive arm joints from the current drive targets to ``q_goal``.

        Uses per-step ``set_qpos`` + drive targets so the motion actually
        reaches the IK solution (drive-only interpolation was observed to
        stall near the grasp configuration).
        """
        arm_name = "left" if str(arm) == "left" else "right"
        joints = (self.robot.left_arm_joints if arm_name == "left"
                  else self.robot.right_arm_joints)
        entity = (self.robot.left_entity if arm_name == "left"
                  else self.robot.right_entity)
        planner = (self.robot.left_planner if arm_name == "left"
                   else self.robot.right_planner)
        active = entity.get_active_joints()
        name_to_qpos_i = {j.get_name(): i for i, j in enumerate(active)}
        q_start = np.array([float(j.get_drive_target()[0]) for j in joints], dtype=float)
        q_goal = np.asarray(q_goal, dtype=float).reshape(-1)
        n = max(1, int(n_steps))
        for i in range(1, n + 1):
            a = float(i) / float(n)
            q = (1.0 - a) * q_start + a * q_goal
            v = (q_goal - q_start) / float(n)
            qpos = np.array(entity.get_qpos(), dtype=float)
            for j, jn in enumerate(planner.active_joints_name):
                if j >= len(q):
                    break
                if jn in name_to_qpos_i:
                    qpos[name_to_qpos_i[jn]] = q[j]
            entity.set_qpos(qpos)
            self.robot.set_arm_joints(q, v, arm_name)
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (self._pic_ctr % max(1, self.save_freq) == 0):
                self._take_picture()
            self._pic_ctr += 1

    def _raise_along_z(self, idx, arm, lift_z=None):
        """Raise the gripper (and welded fruit) straight up by ``lift_z`` meters."""
        import os
        dbg = bool(os.environ.get("PACKING_DEBUG"))
        if lift_z is None:
            lift_z = self.pick_lift
        arm_name = "left" if str(arm) == "left" else "right"
        ee0 = np.array(
            self.robot.get_left_ee_pose() if arm_name == "left"
            else self.robot.get_right_ee_pose(),
            dtype=float,
        )
        fp0 = np.array(self.items[idx].get_pose().p, dtype=float)
        raised = ee0.copy()
        raised[2] += float(lift_z)
        q_goal = self._ik_arm_joints_for_ee(arm, raised)
        if q_goal is None:
            if dbg:
                print("[packing]  raise IK failed — falling back to displacement", flush=True)
            self.plan_success = True
            self.move(self.move_by_displacement(arm, z=float(lift_z), move_axis="world"))
            self.plan_success = True
        else:
            self._drive_arm_joints(arm, q_goal, n_steps=50)
        if dbg:
            ee1 = np.array(
                self.robot.get_left_ee_pose() if arm_name == "left"
                else self.robot.get_right_ee_pose(),
                dtype=float,
            )
            fp1 = np.array(self.items[idx].get_pose().p, dtype=float)
            print(f"[packing]  raise +Z asked={lift_z:.3f} "
                  f"ee_dz={ee1[2]-ee0[2]:.3f} fruit_dz={fp1[2]-fp0[2]:.3f} "
                  f"fruit_z={fp1[2]:.3f}", flush=True)

    def _slide_over_basket(self, idx, arm, target_xy, lift_z=None, tries=3, tol=0.03):
        """Raise ≥10 cm along Z in place, then slide horizontally over the basket."""
        import os
        dbg = bool(os.environ.get("PACKING_DEBUG"))
        if lift_z is None:
            lift_z = self.pick_lift

        # 1) raise straight up (same XY)
        self._raise_along_z(idx, arm, lift_z=lift_z)

        # 2) horizontal slide at the post-lift height (Z held fixed)
        hover_z = float(self.items[idx].get_pose().p[2])
        for _try in range(tries):
            fp = np.array(self.items[idx].get_pose().p, dtype=float)
            target = np.array([float(target_xy[0]), float(target_xy[1]), hover_z], dtype=float)
            gap_xy = float(np.hypot(target[0] - fp[0], target[1] - fp[1]))
            if dbg:
                print(f"[packing]  slide try={_try} fp={fp.round(4)} "
                      f"gap_xy={gap_xy:.4f} hover_z={hover_z:.3f}", flush=True)
            if gap_xy < tol:
                break
            self.plan_success = True
            self.move(self.move_to_pose(arm, self._weld_target_ee_pose(idx, target)))
            self.plan_success = True
        if dbg:
            fp = np.array(self.items[idx].get_pose().p, dtype=float)
            print(f"[packing]  slide residual_xy="
                  f"{np.hypot(target_xy[0] - fp[0], target_xy[1] - fp[1]):.4f} "
                  f"fruit_z={fp[2]:.3f} basket_top="
                  f"{self.basket_top_z[self.item_types[idx]]:.3f}", flush=True)

    def _carry_and_drop(self, idx, arm, target_xy, resend_on_miss=True):
        """Raise along Z, slide over the basket, open gripper, let fruit fall."""
        self._slide_over_basket(idx, arm, target_xy, lift_z=self.pick_lift)
        self._over_basket[idx] = True

        self.plan_success = True
        self.move(self.open_gripper(arm))
        self._release_fruit(idx)
        self._belt_dwell(80)  # give gravity time to settle into the basket
        self._settle_after_drop(idx, target_xy, resend_on_miss=resend_on_miss)
        self.plan_success = True
        self.move(self.back_to_origin(arm))
        self.plan_success = True

    def _pack_item(self, idx):
        """Intercept one still-moving fruit → weld → carry → release."""
        import os
        dbg = bool(os.environ.get("PACKING_DEBUG"))
        fruit = self.items[idx]
        ftype = self.item_types[idx]
        belt_side = self.item_sides[idx]
        arm_side = self.item_arms[idx]
        arm = ArmTag(arm_side)
        target_xy = self._basket_target_xy(idx)

        self._begin_spawn_hold()
        self._grasping_idxs.add(idx)
        try:
            if self._item_y[idx] is None:
                return

            if dbg:
                print(f"[packing] pack {ftype}_{idx} belt={belt_side} arm={arm_side} "
                      f"stream_y={self._item_y[idx]:.3f} "
                      f"pose={np.round(fruit.get_pose().p, 3)}", flush=True)

            # grasp uses the fruit's belt for geometry; arm is color-matched
            if not self._intercept_and_grasp(idx, arm, belt_side):
                if dbg:
                    print("[packing]  grasp failed — fruit stays on belt", flush=True)
                self.plan_success = True
                try:
                    self.move(self.back_to_origin(arm))
                except Exception:
                    pass
                self.plan_success = True
                return

            if dbg:
                print(f"[packing]  welded; ee={np.round(self._ee_pos(arm_side), 3)} "
                      f"fruit={np.round(fruit.get_pose().p, 3)}", flush=True)

            self._carry_and_drop(idx, arm, target_xy)
        finally:
            self._grasping_idxs.discard(idx)
            self._end_spawn_hold()

    def _pack_pair(self, idx_l, idx_r):
        """Pick left+right fruits simultaneously, then carry to baskets together."""
        import os
        dbg = bool(os.environ.get("PACKING_DEBUG"))
        left, right = ArmTag("left"), ArmTag("right")
        target_l = self._basket_target_xy(idx_l)
        target_r = self._basket_target_xy(idx_r)

        self._begin_spawn_hold()
        self._grasping_idxs.update({idx_l, idx_r})
        try:
            if dbg:
                print(f"[packing] pack_pair L={self.item_types[idx_l]}_{idx_l} "
                      f"R={self.item_types[idx_r]}_{idx_r}", flush=True)

            self.plan_success = True
            self.move(self.open_gripper(left), self.open_gripper(right))
            self.plan_success = True

            pre_l = self._plan_station_pre(idx_l, left)
            pre_r = self._plan_station_pre(idx_r, right)
            if pre_l is None or pre_r is None:
                if dbg:
                    print("[packing]  pair: missing pre-grasp — fallback single",
                          flush=True)
                self._grasping_idxs.discard(idx_l)
                self._grasping_idxs.discard(idx_r)
                self._pack_item(idx_l)
                self._pack_item(idx_r)
                return

            self.plan_success = True
            ok = self.move(
                self.move_to_pose(left, pre_l),
                self.move_to_pose(right, pre_r),
            )
            self.plan_success = True
            if ok is False:
                if dbg:
                    print("[packing]  pair: hover failed — fallback single", flush=True)
                self._grasping_idxs.discard(idx_l)
                self._grasping_idxs.discard(idx_r)
                self._pack_item(idx_l)
                self._pack_item(idx_r)
                return

            if not self._wait_pair_at_station(idx_l, idx_r):
                if dbg:
                    print("[packing]  pair: wait at station failed — fallback",
                          flush=True)
                self._grasping_idxs.discard(idx_l)
                self._grasping_idxs.discard(idx_r)
                if self._item_y[idx_l] is not None:
                    self._pack_item(idx_l)
                if self._item_y[idx_r] is not None:
                    self._pack_item(idx_r)
                return

            ok_l, ok_r = self._reach_and_attach_pair(idx_l, idx_r, left, right)

            if not ok_l and not ok_r:
                if dbg:
                    print("[packing]  pair: neither close enough — fallback",
                          flush=True)
                self._grasping_idxs.discard(idx_l)
                self._grasping_idxs.discard(idx_r)
                if self._item_y[idx_l] is not None:
                    self._pack_item(idx_l)
                if self._item_y[idx_r] is not None:
                    self._pack_item(idx_r)
                return

            if ok_l and not ok_r:
                if dbg:
                    print("[packing]  pair: only left attached", flush=True)
                # the other fruit is still time-critical on the belt — try it
                # first; the attached one is welded and safe to carry after
                self._grasping_idxs.discard(idx_r)
                if self._item_y[idx_r] is not None:
                    self._pack_item(idx_r)
                else:
                    self.move(self.back_to_origin(right))
                    self.plan_success = True
                self._carry_and_drop(idx_l, left, target_l)
                return
            if ok_r and not ok_l:
                if dbg:
                    print("[packing]  pair: only right attached", flush=True)
                self._grasping_idxs.discard(idx_l)
                if self._item_y[idx_l] is not None:
                    self._pack_item(idx_l)
                else:
                    self.move(self.back_to_origin(left))
                    self.plan_success = True
                self._carry_and_drop(idx_r, right, target_r)
                return

            if dbg:
                print("[packing]  pair: both attached", flush=True)

            # 1) raise both grippers +Z in place (IK-driven, not trajopt)
            self._raise_along_z(idx_l, left, lift_z=self.pick_lift)
            self._raise_along_z(idx_r, right, lift_z=self.pick_lift)
            hover_z_l = float(self.items[idx_l].get_pose().p[2])
            hover_z_r = float(self.items[idx_r].get_pose().p[2])

            # 2) slide horizontally over each basket at the raised height
            for _try in range(3):
                fp_l = np.array(self.items[idx_l].get_pose().p, dtype=float)
                fp_r = np.array(self.items[idx_r].get_pose().p, dtype=float)
                tgt_l = np.array([float(target_l[0]), float(target_l[1]), hover_z_l], dtype=float)
                tgt_r = np.array([float(target_r[0]), float(target_r[1]), hover_z_r], dtype=float)
                gl = float(np.hypot(tgt_l[0] - fp_l[0], tgt_l[1] - fp_l[1]))
                gr = float(np.hypot(tgt_r[0] - fp_r[0], tgt_r[1] - fp_r[1]))
                if dbg:
                    print(f"[packing]  pair slide try={_try} gap_l={gl:.4f} gap_r={gr:.4f}", flush=True)
                if gl < 0.03 and gr < 0.03:
                    break
                self.plan_success = True
                self.move(
                    self.move_to_pose(left, self._weld_target_ee_pose(idx_l, tgt_l)),
                    self.move_to_pose(right, self._weld_target_ee_pose(idx_r, tgt_r)),
                )
                self.plan_success = True

            if dbg:
                fp_l = np.array(self.items[idx_l].get_pose().p, dtype=float)
                fp_r = np.array(self.items[idx_r].get_pose().p, dtype=float)
                print(f"[packing]  pair slide residual_xy "
                      f"l={np.hypot(target_l[0] - fp_l[0], target_l[1] - fp_l[1]):.4f} "
                      f"r={np.hypot(target_r[0] - fp_r[0], target_r[1] - fp_r[1]):.4f} "
                      f"fruit_z l={fp_l[2]:.3f} r={fp_r[2]:.3f}", flush=True)
            self._over_basket[idx_l] = True
            self._over_basket[idx_r] = True

            self.plan_success = True
            self.move(self.open_gripper(left), self.open_gripper(right))
            self._release_fruit(idx_l)
            self._release_fruit(idx_r)
            self._belt_dwell(80)
            self._settle_after_drop(idx_l, target_l)
            self._settle_after_drop(idx_r, target_r)

            self.move(self.back_to_origin(left), self.back_to_origin(right))
            self.plan_success = True
        finally:
            self._grasping_idxs.discard(idx_l)
            self._grasping_idxs.discard(idx_r)
            self._end_spawn_hold()

    def _ready_by_side(self):
        """Return oldest ready fruit index per belt (or None)."""
        if self._grasping_idxs:
            return {"left": None, "right": None}
        latest_start = self.pick_station_y + 0.08
        ready = {"left": None, "right": None}
        for i in range(self.n_items):
            if (not self._spawned_mask[i] or self._packed[i]
                    or self._welded[i] or self._item_y[i] is None):
                continue
            y = self._item_y[i]
            if latest_start <= y <= self.pick_y:
                side = self.item_sides[i]
                if ready[side] is None:
                    ready[side] = i
        return ready

    def _dispatch_pack(self, ready):
        """Pick the right pack call for the current ready fruit(s).

        Prefers pairing a ready fruit with its still-outstanding wave
        partner (see ``_active_pair_partner``) even if the partner hasn't
        entered the ready window yet — this is what makes a staggered pair
        (``pair_stagger_enabled``) still get carried together instead of
        the head-started fruit being solo-packed the instant it alone
        becomes ready. Falls back to the plain "both already ready"
        pairing, then to solo packing.
        """
        left_i, right_i = ready["left"], ready["right"]
        if self.spawn_mode in ("parallel", "random"):
            for i in (left_i, right_i):
                if i is None:
                    continue
                partner = self._active_pair_partner(i)
                if partner is None:
                    continue
                idx_l, idx_r = (i, partner) if self.item_sides[i] == "left" else (partner, i)
                self._pack_pair(idx_l, idx_r)
                return True
            if left_i is not None and right_i is not None:
                self._pack_pair(left_i, right_i)
                return True
        if left_i is not None:
            self._pack_item(left_i)
            return True
        if right_i is not None:
            self._pack_item(right_i)
            return True
        return False

    # ------------------------------------------------------------- policy
    def play_once(self):
        import os
        dbg = bool(os.environ.get("PACKING_DEBUG"))

        # belts run continuously for the whole episode (including during arm motion,
        # because take_dense_action calls _update_kinematic_tasks every step)
        self._belt_running = True
        min_speed = max(min(self.belt_speed.values()), 1e-6)
        max_steps = int(
            (self.BELT_Y_FAR - self.BELT_Y_NEAR + self.n_items * self.spawn_gap)
            / min_speed
        ) * self.advance_every + self.n_items * 1200 + 4000

        guard = 0
        def _wave_done():
            return all(
                self._packed[i] or self._missed[i] for i in range(self.n_items)
            ) and self._spawned >= self.n_items

        while guard < max_steps and not _wave_done():
            guard += 1
            ready = self._ready_by_side()
            if not self._dispatch_pack(ready):
                self._belt_dwell(max(1, self.advance_every))

        for _ in range(self.n_items):
            ready = self._ready_by_side()
            if ready["left"] is None and ready["right"] is None:
                break
            self._dispatch_pack(ready)

        self._belt_dwell(40)
        self._belt_running = False

        if self.check_success():
            self.plan_success = True

        if dbg:
            print(f"[packing] done plan={self.plan_success} mode={self.spawn_mode} "
                  f"types={self.item_types} sides={self.item_sides}", flush=True)
            for i in range(self.n_items):
                p = self.items[i].get_pose().p
                print(f"[packing]  {self.item_types[i]}_{i} p={np.round(p, 3)} "
                      f"in={self._fruit_in_basket(i)} packed={self._packed[i]} "
                      f"missed={self._missed[i]}",
                      flush=True)

        self.info["info"] = {
            "{A}": f"{self.n_apple} apples + {self.n_orange} oranges",
            "{B}": f"076_breadbasket/base{self.basket_id}",
            "{a}": "both arms",
        }
        return self.info

    # ------------------------------------------------------------- success
    def check_success(self):
        return all(self._fruit_in_basket(i) for i in range(self.n_items))

    def get_obs(self):
        obs = super().get_obs()
        in_ok = [bool(self._fruit_in_basket(i)) for i in range(self.n_items)]
        obs["packing"] = {
            "n_items": int(self.n_items),
            "n_apple": int(self.n_apple),
            "n_orange": int(self.n_orange),
            "n_spawned": int(self._spawned),
            "n_packed": int(sum(1 for p in self._packed if p)),
            "n_missed": int(sum(1 for m in self._missed if m)),
            "n_correct": int(sum(in_ok)),
            "spawn_mode": str(self.spawn_mode),
            "spawn_delay_s": float(self.spawn_delay_s),
            "item_types": list(self.item_types),
            "item_sides": list(self.item_sides),
            "item_y": [None if y is None else float(y) for y in self._item_y],
            "item_in_correct_basket": in_ok,
            "belt_speed_left": float(self.belt_speed["left"]),
            "belt_speed_right": float(self.belt_speed["right"]),
            "belt_gap": float(self.belt_gap),
            "belt_running": bool(self._belt_running),
            "fruit_scale": float(self.fruit_scale),
        }
        return obs
