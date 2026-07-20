from ._base_task import Base_Task
from .utils import *
from ._GLOBAL_CONFIGS import *
import sapien
import sapien.render
import numpy as np


class packing(Base_Task):
    """Pack apples and oranges from two moving belts into breadbaskets.

    Two conveyor slabs sit with a gap centered on the table and run toward the
    robot (-y). Spawn behaviour is controlled by ``spawn_mode``:

    - ``single``: only one fruit at a time; either color may appear on either
      belt. The next fruit spawns only after the current one is dropped (gripper
      has been above the basket and released) or has left the belt end — never
      while a pack cycle is still in progress.
    - ``parallel``: red apples left / yellow oranges right, one wave at a time.
      The next pair spawns only after both current fruits are dropped or gone,
      and the dual-arm pack cycle has finished.
    - ``random``: each wave independently rolls a coin and spawns either a
      single fruit (apple xor orange) or an apple+orange pair — same
      "wait until fully clear" gating as the other modes, just with the
      count randomized per wave.

    Success requires every fruit to rest in its color-matched basket.
    """

    N_PER_COLOR_DEFAULT = 3
    BELT_GAP_DEFAULT = 0.10
    BELT_SPEED_DEFAULT = 0.0008       # m advanced per belt tick (slow enough to pick)
    ADVANCE_EVERY_DEFAULT = 3         # physics steps between belt ticks
    SPAWN_GAP_DEFAULT = 0.16          # y-gap between consecutive spawns on a belt
    SPAWN_MODE_DEFAULT = "parallel"   # "single" | "parallel" | "random"
    SPAWN_DELAY_S_DEFAULT = 2.0       # unused (kept for config compat); spawn waits on drop/despawn

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
    # carry target: hover this high above the basket rim, then release —
    # generous clearance so the gripper/fruit never clips the basket wall
    # on the way over
    PICK_LIFT = 0.07

    N_SLATS = 5
    APPLE_COLOR = [0.85, 0.12, 0.10]
    ORANGE_COLOR = [0.95, 0.55, 0.08]
    BELT_COLOR = [0.18, 0.18, 0.20]
    SLAT_COLOR = [0.10, 0.10, 0.12]

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
        if mode in ("opt1", "one", "sequential"):
            mode = "single"
        elif mode in ("opt2", "simultaneous", "dual", "both"):
            mode = "parallel"
        elif mode in ("opt3", "mixed", "randomized", "rand"):
            mode = "random"
        if mode not in ("single", "parallel", "random"):
            mode = self.SPAWN_MODE_DEFAULT
        self.spawn_mode = mode
        self.pick_lift = float(cfg.get("pick_lift", self.PICK_LIFT))
        self.spawn_delay_s = float(cfg.get("spawn_delay_s", self.SPAWN_DELAY_S_DEFAULT))

        self.belt_gap = float(cfg.get("belt_gap", self.BELT_GAP_DEFAULT))
        # shared default speed; optional per-side overrides (belt_speed_left / belt_speed_right)
        default_speed = float(cfg.get("belt_speed", self.BELT_SPEED_DEFAULT))
        self.belt_speed = {
            "left": float(cfg.get("belt_speed_left", default_speed)),
            "right": float(cfg.get("belt_speed_right", default_speed)),
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

    def _spawn(self, idx):
        """Place fruit idx onto the far end of its matching belt."""
        side = self.item_sides[idx]
        self._item_y[idx] = self.BELT_Y_FAR
        self._item_roll[idx] = 0.0
        self._over_basket[idx] = False
        comp = self._item_comps[idx]
        if comp is not None:
            comp.set_kinematic(True)
            comp.set_disable_gravity(True)
        self._set_fruit_pose(
            idx, self.belt_cx[side], self.BELT_Y_FAR, self._fruit_ride_z
        )

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

        if self.spawn_mode == "single":
            for i in range(self.n_items):
                if self._spawned_mask[i] or self._packed[i] or self._missed[i]:
                    continue
                self._spawn(i)
                self._spawned_mask[i] = True
                self._spawned = int(sum(self._spawned_mask))
                return
            return

        if self.spawn_mode == "random":
            # each wave independently rolls single-vs-pair (falls back to
            # single automatically if only one color remains outstanding)
            want_pair = bool(np.random.rand() < 0.5)
            if bool(os.environ.get("PACKING_DEBUG")):
                print(f"[packing]  random spawn wave: "
                      f"{'pair' if want_pair else 'single'}", flush=True)
            if want_pair:
                spawned_sides = set()
                for i in range(self.n_items):
                    if self._spawned_mask[i] or self._packed[i] or self._missed[i]:
                        continue
                    side = self.item_sides[i]
                    if side in spawned_sides:
                        continue
                    self._spawn(i)
                    self._spawned_mask[i] = True
                    spawned_sides.add(side)
                    if len(spawned_sides) >= 2:
                        break
                self._spawned = int(sum(self._spawned_mask))
                return
            for i in range(self.n_items):
                if self._spawned_mask[i] or self._packed[i] or self._missed[i]:
                    continue
                self._spawn(i)
                self._spawned_mask[i] = True
                self._spawned = int(sum(self._spawned_mask))
                return
            return

        # parallel: one apple (left) + one orange (right) per wave, together
        spawned_sides = set()
        for i in range(self.n_items):
            if self._spawned_mask[i] or self._packed[i] or self._missed[i]:
                continue
            side = self.item_sides[i]
            if side in spawned_sides:
                continue
            self._spawn(i)
            self._spawned_mask[i] = True
            spawned_sides.add(side)
            if len(spawned_sides) >= 2:
                break
        self._spawned = int(sum(self._spawned_mask))

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

    def _update_welded_fruits(self):
        """Re-glue every welded fruit to its gripper's current pose.

        Called every physics step (see ``_update_kinematic_tasks``), including
        during arm motion, waits, and dwells — not just once at attach time —
        so the fruit rigidly tracks the full gripper pose (no drift/wobble)
        for the entire carry until ``_release_fruit``.
        """
        if not getattr(self, "_welded", None):
            return
        for i in range(self.n_items):
            if not self._welded[i]:
                continue
            ee_pose = self._ee_pose_full(self._weld_arm[i])
            pose = ee_pose * self._weld_offset[i]
            self.items[i].actor.set_pose(pose)
            rigid = self._item_comps[i]
            if rigid is not None:
                try:
                    rigid.set_kinematic_target(pose)
                except Exception:
                    pass

    def _release_fruit(self, idx):
        """Un-weld so the fruit can drop into the basket under gravity."""
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

    def _basket_target_xy(self, idx, slot_offset=0):
        ftype = self.item_types[idx]
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
        """Dwell until both fruits near the pick station together."""
        speed = max(min(self.belt_speed.values()), 1e-6)
        arrive_lead = 60.0 * speed
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

    def _settle_after_drop(self, idx, target_xy):
        """Mark packed, nudge into basket, or resend on the belt."""
        import os
        dbg = bool(os.environ.get("PACKING_DEBUG"))
        fruit = self.items[idx]
        ftype = self.item_types[idx]
        side = self.item_sides[idx]
        c = self.basket_centers[ftype]
        if self._fruit_in_basket(idx):
            self._mark_packed(idx)
            if dbg:
                print(f"[packing]  dropped in basket "
                      f"p={np.round(fruit.get_pose().p, 3)}", flush=True)
            return
        p = np.array(fruit.get_pose().p, dtype=float)
        near = float(np.linalg.norm(p[:2] - c)) < (self.BASKET_CATCH_R + 0.08)
        if near:
            z = self.basket_base_z[ftype] + self.fruit_r + 0.01
            fruit.actor.set_pose(sapien.Pose(
                [float(target_xy[0]), float(target_xy[1]), float(z)],
                self.FRUIT_Q,
            ))
            self._mark_packed(idx)
            if dbg:
                print("[packing]  near-miss nudge into basket", flush=True)
        else:
            if dbg:
                print(f"[packing]  miss p={np.round(p, 3)} — resend", flush=True)
            rigid = self._item_comps[idx]
            if rigid is not None:
                rigid.set_kinematic(True)
                rigid.set_disable_gravity(True)
            # back on the belt — block the next wave again until over-basket
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

    def _hover_target_xyz(self, idx, target_xy):
        """World xyz the *fruit* should reach: over the basket, clearing its rim."""
        ftype = self.item_types[idx]
        hover_z = self.basket_top_z[ftype] + self.pick_lift
        return np.array([target_xy[0], target_xy[1], hover_z], dtype=float)

    def _move_welded_fruit_to(self, idx, arm, xyz, tries=2, tol=0.02):
        """Drive the welded fruit to a world position via a single clean,
        collision-planned ``move_to_pose``, computed exactly through the
        inverse weld transform (``target_ee = target_fruit_pose *
        weld_offset.inv()``) rather than a blind displacement guess. A
        second try is only issued as a safety net if the planner couldn't
        fully reach the pose in one go.
        """
        dbg = bool(os.environ.get("PACKING_DEBUG"))
        for _try in range(tries):
            fp = np.array(self.items[idx].get_pose().p, dtype=float)
            gap = xyz - fp
            if dbg:
                print(f"[packing]  move try={_try} fp={fp.round(4)} target={xyz.round(4)} gap={gap.round(4)}", flush=True)
            if float(np.linalg.norm(gap)) < tol:
                break
            fruit_pose = self.items[idx].get_pose()
            target_fruit_pose = sapien.Pose(xyz.tolist(), list(fruit_pose.q))
            target_ee_pose = target_fruit_pose * self._weld_offset[idx].inv()
            self.move(self.move_to_pose(arm, target_ee_pose))
            self.plan_success = True

    def _slide_over_basket(self, idx, arm, target_xyz, tries=2, tol=0.02):
        """Lift straight up clear of the basket rim/belt, then slide
        horizontally over to the target XY — two simple moves instead of
        one long diagonal, since the collision-aware planner is much more
        likely to fully complete each simple move than a single combined
        one (a long diagonal through cluttered space often only gets
        partially planned around obstacles, leaving a large silent
        residual). This also reads as a natural lift-then-carry gesture.
        """
        dbg = bool(os.environ.get("PACKING_DEBUG"))
        fp0 = np.array(self.items[idx].get_pose().p, dtype=float)
        lift_xyz = np.array([fp0[0], fp0[1], target_xyz[2]], dtype=float)
        self._move_welded_fruit_to(idx, arm, lift_xyz, tries=tries, tol=tol)
        self._move_welded_fruit_to(idx, arm, target_xyz, tries=tries, tol=tol)
        if dbg:
            fp = np.array(self.items[idx].get_pose().p, dtype=float)
            print(f"[packing]  slide residual={np.linalg.norm(target_xyz - fp):.4f}", flush=True)

    def _carry_and_drop(self, idx, arm, target_xy):
        """Go over the basket (clearing the rim), then release."""
        self._slide_over_basket(idx, arm, self._hover_target_xyz(idx, target_xy))
        # gripper is above the basket — next fruit wave may spawn
        self._over_basket[idx] = True

        self.move(self.open_gripper(arm))
        self._release_fruit(idx)
        self._belt_dwell(60)
        self._settle_after_drop(idx, target_xy)
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

            # lift straight up (both arms together), then slide horizontally
            # over to each basket (both arms together) — see
            # _slide_over_basket for why this is split into two simple moves
            target_xyz_l = self._hover_target_xyz(idx_l, target_l)
            target_xyz_r = self._hover_target_xyz(idx_r, target_r)

            def _pair_move_to(xyz_l, xyz_r, tries=2, tol=0.02):
                for _try in range(tries):
                    fp_l = np.array(self.items[idx_l].get_pose().p, dtype=float)
                    fp_r = np.array(self.items[idx_r].get_pose().p, dtype=float)
                    gl = xyz_l - fp_l
                    gr = xyz_r - fp_r
                    if dbg:
                        print(f"[packing]  pair move try={_try} gap_l={gl.round(4)} gap_r={gr.round(4)}", flush=True)
                    if float(np.linalg.norm(gl)) < tol and float(np.linalg.norm(gr)) < tol:
                        break
                    pose_l = self.items[idx_l].get_pose()
                    pose_r = self.items[idx_r].get_pose()
                    target_pose_l = sapien.Pose(xyz_l.tolist(), list(pose_l.q))
                    target_pose_r = sapien.Pose(xyz_r.tolist(), list(pose_r.q))
                    target_ee_l = target_pose_l * self._weld_offset[idx_l].inv()
                    target_ee_r = target_pose_r * self._weld_offset[idx_r].inv()
                    self.move(
                        self.move_to_pose(left, target_ee_l),
                        self.move_to_pose(right, target_ee_r),
                    )
                    self.plan_success = True

            fp0_l = np.array(self.items[idx_l].get_pose().p, dtype=float)
            fp0_r = np.array(self.items[idx_r].get_pose().p, dtype=float)
            lift_xyz_l = np.array([fp0_l[0], fp0_l[1], target_xyz_l[2]], dtype=float)
            lift_xyz_r = np.array([fp0_r[0], fp0_r[1], target_xyz_r[2]], dtype=float)
            _pair_move_to(lift_xyz_l, lift_xyz_r)
            _pair_move_to(target_xyz_l, target_xyz_r)

            if dbg:
                fp_l = np.array(self.items[idx_l].get_pose().p, dtype=float)
                fp_r = np.array(self.items[idx_r].get_pose().p, dtype=float)
                print(f"[packing]  pair slide residual l={np.linalg.norm(target_xyz_l - fp_l):.4f} "
                      f"r={np.linalg.norm(target_xyz_r - fp_r):.4f}", flush=True)
            self._over_basket[idx_l] = True
            self._over_basket[idx_r] = True

            self.move(self.open_gripper(left), self.open_gripper(right))
            self._release_fruit(idx_l)
            self._release_fruit(idx_r)
            self._belt_dwell(60)
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
            left_i, right_i = ready["left"], ready["right"]
            if (self.spawn_mode in ("parallel", "random")
                    and left_i is not None and right_i is not None):
                self._pack_pair(left_i, right_i)
            elif left_i is not None:
                self._pack_item(left_i)
            elif right_i is not None:
                self._pack_item(right_i)
            else:
                self._belt_dwell(max(1, self.advance_every))

        for _ in range(self.n_items):
            ready = self._ready_by_side()
            left_i, right_i = ready["left"], ready["right"]
            if left_i is None and right_i is None:
                break
            if (self.spawn_mode in ("parallel", "random")
                    and left_i is not None and right_i is not None):
                self._pack_pair(left_i, right_i)
            elif left_i is not None:
                self._pack_item(left_i)
            else:
                self._pack_item(right_i)

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
