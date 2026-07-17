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

    Success requires every fruit to rest in its color-matched basket.
    """

    N_PER_COLOR_DEFAULT = 3
    BELT_GAP_DEFAULT = 0.10
    BELT_SPEED_DEFAULT = 0.0008       # m advanced per belt tick (slow enough to pick)
    ADVANCE_EVERY_DEFAULT = 3         # physics steps between belt ticks
    SPAWN_GAP_DEFAULT = 0.16          # y-gap between consecutive spawns on a belt
    SPAWN_MODE_DEFAULT = "parallel"   # "single" | "parallel"
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
    # when TCP is this close to a still-moving fruit, snap it into the fingers
    SNAP_XY_MAX = 0.07
    SNAP_Z_MAX = 0.14
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
    BASKET_SCALE = 0.75               # smaller breadbaskets
    BASKET_CATCH_R = 0.08
    BASKET_Y = -0.16
    BASKET_X = 0.30                   # farther outboard from the belts
    # release height: fruit center above basket floor (modest, not a high dump)
    DROP_ABOVE_BASKET = 0.06

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
        if mode not in ("single", "parallel"):
            mode = self.SPAWN_MODE_DEFAULT
        self.spawn_mode = mode
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

    def _drop_height(self, idx):
        """World-z for fruit center when releasing above the basket."""
        ftype = self.item_types[idx]
        return float(self.basket_base_z[ftype] + self.fruit_r + self.DROP_ABOVE_BASKET)

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

    def _tcp_pos(self, arm):
        """Gripper-center pose (fingertips), not the retracted planning EE frame."""
        p = (self.robot.get_left_tcp_pose() if arm == "left"
             else self.robot.get_right_tcp_pose())
        return np.array(p[:3], dtype=float)

    def _weld_fruit_to_ee(self, idx, arm):
        """Rigidly attach fruit to the EE so lifts/carries cannot slip."""
        arm_name = "left" if str(arm) == "left" else "right"
        # offset in planning-EE frame (same frame used by _update_welded_fruits)
        self._weld_arm[idx] = arm_name
        self._weld_offset[idx] = (
            np.array(self.items[idx].get_pose().p, dtype=float) - self._ee_pos(arm_name)
        )
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
        if not getattr(self, "_welded", None):
            return
        for i in range(self.n_items):
            if not self._welded[i]:
                continue
            target = self._ee_pos(self._weld_arm[i]) + self._weld_offset[i]
            q = self.items[i].get_pose().q
            pose = sapien.Pose(target.tolist(), q)
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
            (0.0, 0.0), (0.02, 0.012), (-0.02, 0.012),
            (0.02, -0.012), (-0.02, -0.012),
        ]
        ox, oy = offsets[slot % len(offsets)]
        return c + np.array([ox, oy], dtype=float)

    def _plan_station_pre(self, idx, arm):
        side = self.item_sides[idx]
        cx = self.belt_cx[side]
        self._set_fruit_pose(idx, cx, self.pick_station_y, self._fruit_ride_z)
        try:
            pre_pose, _ = self.choose_grasp_pose(
                self.items[idx], arm_tag=arm, pre_dis=0.12, target_dis=0.0,
            )
        finally:
            self._restore_fruit_stream_pose(idx)
        return pre_pose

    def _wait_fruit_at_station(self, idx, side):
        """Dwell until fruit is near the station. Fruit keeps moving (no pause)."""
        import os
        dbg = bool(os.environ.get("PACKING_DEBUG"))
        speed = max(self.belt_speed[side], 1e-6)
        arrive_lead = 100.0 * speed
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

    def _tcp_near_fruit(self, idx, arm):
        """True if the gripper TCP is close enough to snap-capture the fruit."""
        arm_name = "left" if str(arm) == "left" else "right"
        tcp = self._tcp_pos(arm_name)
        fp = np.array(self.items[idx].get_pose().p, dtype=float)
        xy = float(np.linalg.norm(fp[:2] - tcp[:2]))
        dz = float(tcp[2] - fp[2])
        return xy <= self.SNAP_XY_MAX and 0.0 <= dz <= self.SNAP_Z_MAX

    def _snap_fruit_into_gripper(self, idx, arm):
        """Place the still-moving fruit between the fingers and weld it on."""
        import os
        dbg = bool(os.environ.get("PACKING_DEBUG"))
        arm_name = "left" if str(arm) == "left" else "right"
        tcp = self._tcp_pos(arm_name)
        # nest the fruit in the gripper center, just below the TCP
        pos = tcp.copy()
        pos[2] -= 0.015
        # leave the belt stream immediately so close_gripper steps cannot roll it away
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
        if dbg:
            print(f"[packing]  snap {self.item_types[idx]}_{idx} into "
                  f"{arm_name} tcp={np.round(tcp, 3)}", flush=True)
        self.plan_success = True
        self.move(self.close_gripper(arm, pos=0.0))
        self.plan_success = True
        # re-seat after close (fingers may shift TCP slightly) then weld
        tcp = self._tcp_pos(arm_name)
        pos = tcp.copy()
        pos[2] -= 0.015
        self.items[idx].actor.set_pose(sapien.Pose(pos.tolist(), self.FRUIT_Q))
        self._weld_fruit_to_ee(idx, arm)

    def _track_toward_fruit(self, idx, arm):
        """One hop that follows the moving fruit down-belt and down in z."""
        belt = self.item_sides[idx]
        speed = max(self.belt_speed[belt], 1e-6)
        y = self._item_y[idx]
        if y is None:
            return None
        arm_name = "left" if str(arm) == "left" else "right"
        ee = self._ee_pos(arm_name)
        tcp = self._tcp_pos(arm_name)
        fp = np.array(self.items[idx].get_pose().p, dtype=float)
        # aim slightly ahead along the belt; target TCP just above the fruit
        aim_y = float(y) - 20.0 * speed
        aim_x = self.belt_cx[belt]
        aim_tcp_z = self._fruit_ride_z + 0.03
        # convert TCP aim -> planning-EE displacement (TCP is ~0.12 further along approach)
        dz_tcp = aim_tcp_z - float(tcp[2])
        return (
            float(np.clip(aim_x - ee[0], -0.12, 0.12)),
            float(np.clip(aim_y - ee[1], -0.12, 0.12)),
            float(np.clip(dz_tcp, -0.10, 0.06)),
            float(np.linalg.norm(fp[:2] - tcp[:2])),
        )

    def _chase_and_snap(self, idx, arm):
        """Follow a still-moving fruit; when close, snap it into the gripper."""
        import os
        dbg = bool(os.environ.get("PACKING_DEBUG"))
        lose_y = self.BELT_Y_NEAR + 0.02

        for hop in range(14):
            y = self._item_y[idx]
            if y is None or y < lose_y:
                if dbg:
                    print(f"[packing]  chase lost y={y}", flush=True)
                return False
            if self._tcp_near_fruit(idx, arm):
                self._snap_fruit_into_gripper(idx, arm)
                return True
            hop_info = self._track_toward_fruit(idx, arm)
            if hop_info is None:
                return False
            dx, dy, dz, xy = hop_info
            if dbg and hop % 3 == 0:
                print(f"[packing]  chase hop={hop} xy={xy:.3f} y={y:.3f}",
                      flush=True)
            self.move(self.move_by_displacement(
                arm, x=dx, y=dy, z=dz, move_axis="world",
            ))
            self.plan_success = True

        if self._tcp_near_fruit(idx, arm):
            self._snap_fruit_into_gripper(idx, arm)
            return True
        # last chance: still roughly under the gripper → force snap
        if self._item_y[idx] is not None and self._item_y[idx] >= lose_y:
            arm_name = "left" if str(arm) == "left" else "right"
            tcp = self._tcp_pos(arm_name)
            fp = np.array(self.items[idx].get_pose().p, dtype=float)
            if (float(np.linalg.norm(fp[:2] - tcp[:2])) <= 0.10
                    and float(tcp[2] - fp[2]) <= 0.18):
                self._snap_fruit_into_gripper(idx, arm)
                return True
        if dbg:
            print("[packing]  chase timed out without snap", flush=True)
        return False

    def _chase_and_snap_pair(self, idx_l, idx_r, arm_l, arm_r):
        """Chase both moving fruits; snap each when its gripper is close enough."""
        import os
        dbg = bool(os.environ.get("PACKING_DEBUG"))
        lose_y = self.BELT_Y_NEAR + 0.02
        got_l = got_r = False

        for hop in range(14):
            yl, yr = self._item_y[idx_l], self._item_y[idx_r]
            # already snapped fruits have item_y=None
            if not got_l and (yl is None or yl < lose_y):
                pass
            if not got_r and (yr is None or yr < lose_y):
                pass
            if not got_l and self._item_y[idx_l] is not None and self._tcp_near_fruit(idx_l, arm_l):
                self._snap_fruit_into_gripper(idx_l, arm_l)
                got_l = True
            if not got_r and self._item_y[idx_r] is not None and self._tcp_near_fruit(idx_r, arm_r):
                self._snap_fruit_into_gripper(idx_r, arm_r)
                got_r = True
            if got_l and got_r:
                return True, True

            acts_l = acts_r = None
            if not got_l and self._item_y[idx_l] is not None and self._item_y[idx_l] >= lose_y:
                hl = self._track_toward_fruit(idx_l, arm_l)
                if hl is not None:
                    acts_l = self.move_by_displacement(
                        arm_l, x=hl[0], y=hl[1], z=hl[2], move_axis="world",
                    )
            if not got_r and self._item_y[idx_r] is not None and self._item_y[idx_r] >= lose_y:
                hr = self._track_toward_fruit(idx_r, arm_r)
                if hr is not None:
                    acts_r = self.move_by_displacement(
                        arm_r, x=hr[0], y=hr[1], z=hr[2], move_axis="world",
                    )
            if acts_l is not None and acts_r is not None:
                self.move(acts_l, acts_r)
            elif acts_l is not None:
                self.move(acts_l)
            elif acts_r is not None:
                self.move(acts_r)
            else:
                break
            self.plan_success = True

        if not got_l and self._item_y[idx_l] is not None and self._tcp_near_fruit(idx_l, arm_l):
            self._snap_fruit_into_gripper(idx_l, arm_l)
            got_l = True
        if not got_r and self._item_y[idx_r] is not None and self._tcp_near_fruit(idx_r, arm_r):
            self._snap_fruit_into_gripper(idx_r, arm_r)
            got_r = True
        if dbg:
            print(f"[packing]  pair chase done gotL={got_l} gotR={got_r}", flush=True)
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

    def _move_fruit_above_basket(self, idx, arm, target_xy):
        """Carry welded fruit to a modest height above its basket (XY then Z)."""
        fruit = self.items[idx]
        # clear the belt
        self.move(self.move_by_displacement(arm, z=0.10, move_axis="world"))
        self.plan_success = True
        for _ in range(4):
            ap = np.array(fruit.get_pose().p, dtype=float)
            dxy = target_xy - ap[:2]
            if float(np.linalg.norm(dxy)) < 0.02:
                break
            self.move(self.move_by_displacement(
                arm,
                x=float(np.clip(dxy[0], -0.16, 0.16)),
                y=float(np.clip(dxy[1], -0.16, 0.16)),
                move_axis="world",
            ))
            self.plan_success = True
        drop_z = self._drop_height(idx)
        for _ in range(4):
            fz = float(fruit.get_pose().p[2])
            dz = drop_z - fz
            if abs(dz) < 0.012:
                break
            self.move(self.move_by_displacement(
                arm, z=float(np.clip(dz, -0.14, 0.08)), move_axis="world",
            ))
            self.plan_success = True
        # gripper is above the basket — next fruit wave may spawn
        self._over_basket[idx] = True

    def _move_pair_above_baskets(self, idx_l, idx_r, arm_l, arm_r, target_l, target_r):
        """Carry both welded fruits to modest heights above their baskets."""
        # clear the belts
        self.move(
            self.move_by_displacement(arm_l, z=0.10, move_axis="world"),
            self.move_by_displacement(arm_r, z=0.10, move_axis="world"),
        )
        self.plan_success = True
        for _ in range(4):
            pl = np.array(self.items[idx_l].get_pose().p, dtype=float)
            pr = np.array(self.items[idx_r].get_pose().p, dtype=float)
            dl = target_l - pl[:2]
            dr = target_r - pr[:2]
            if (float(np.linalg.norm(dl)) < 0.02
                    and float(np.linalg.norm(dr)) < 0.02):
                break
            self.move(
                self.move_by_displacement(
                    arm_l,
                    x=float(np.clip(dl[0], -0.16, 0.16)),
                    y=float(np.clip(dl[1], -0.16, 0.16)),
                    move_axis="world",
                ),
                self.move_by_displacement(
                    arm_r,
                    x=float(np.clip(dr[0], -0.16, 0.16)),
                    y=float(np.clip(dr[1], -0.16, 0.16)),
                    move_axis="world",
                ),
            )
            self.plan_success = True
        drop_l = self._drop_height(idx_l)
        drop_r = self._drop_height(idx_r)
        for _ in range(4):
            zl = float(self.items[idx_l].get_pose().p[2])
            zr = float(self.items[idx_r].get_pose().p[2])
            dzl = drop_l - zl
            dzr = drop_r - zr
            if abs(dzl) < 0.012 and abs(dzr) < 0.012:
                break
            self.move(
                self.move_by_displacement(
                    arm_l, z=float(np.clip(dzl, -0.14, 0.08)), move_axis="world",
                ),
                self.move_by_displacement(
                    arm_r, z=float(np.clip(dzr, -0.14, 0.08)), move_axis="world",
                ),
            )
            self.plan_success = True
        self._over_basket[idx_l] = True
        self._over_basket[idx_r] = True

    def _intercept_and_grasp(self, idx, arm, side):
        """Hover above the belt, chase the still-moving fruit, snap when close.

        Fruit never pauses on the belt. When the gripper TCP is near enough,
        the fruit is placed between the fingers and welded.
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

        # Arm is already hovering at the station; fruit has rolled underneath.
        # Snap it between the fingers — no pause, no multi-hop chase.
        if self._item_y[idx] is None or self._item_y[idx] < self.pick_y_end:
            return False
        self._snap_fruit_into_gripper(idx, arm)
        return True

    def _carry_and_drop(self, idx, arm, target_xy):
        """Carry welded fruit above the basket, release from a modest height, settle."""
        self._move_fruit_above_basket(idx, arm, target_xy)
        self.move(self.open_gripper(arm))
        self._release_fruit(idx)
        self._belt_dwell(70)
        self._settle_after_drop(idx, target_xy)
        self.move(self.move_by_displacement(arm, z=0.06, move_axis="arm"))
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

            # wait until both fruits are near the station (they keep rolling)
            speed = max(min(self.belt_speed.values()), 1e-6)
            max_wait = int((self.BELT_Y_FAR - self.BELT_Y_NEAR) / speed) + 100
            arrive_lead = 100.0 * speed
            for _ in range(max_wait):
                yl, yr = self._item_y[idx_l], self._item_y[idx_r]
                if yl is None or yr is None:
                    break
                if yl < self.pick_y_end or yr < self.pick_y_end:
                    break
                if (yl <= self.pick_station_y + arrive_lead
                        and yr <= self.pick_station_y + arrive_lead):
                    break
                self._belt_dwell(max(1, self.advance_every))

            if (self._item_y[idx_l] is None or self._item_y[idx_r] is None
                    or self._item_y[idx_l] < self.pick_y_end
                    or self._item_y[idx_r] < self.pick_y_end):
                if dbg:
                    print("[packing]  pair: fruit missed — fallback single", flush=True)
                self._grasping_idxs.discard(idx_l)
                self._grasping_idxs.discard(idx_r)
                if self._item_y[idx_l] is not None:
                    self._pack_item(idx_l)
                if self._item_y[idx_r] is not None:
                    self._pack_item(idx_r)
                return

            # Both fruits under the hovering grippers — snap in place (no chase).
            if (self._item_y[idx_l] is None or self._item_y[idx_r] is None
                    or self._item_y[idx_l] < self.pick_y_end
                    or self._item_y[idx_r] < self.pick_y_end):
                if dbg:
                    print("[packing]  pair: missed at snap — fallback", flush=True)
                self._grasping_idxs.discard(idx_l)
                self._grasping_idxs.discard(idx_r)
                if self._item_y[idx_l] is not None:
                    self._pack_item(idx_l)
                if self._item_y[idx_r] is not None:
                    self._pack_item(idx_r)
                return

            self._snap_fruit_into_gripper(idx_l, left)
            self._snap_fruit_into_gripper(idx_r, right)
            if dbg:
                print("[packing]  pair: both snapped into grippers", flush=True)

            self._move_pair_above_baskets(
                idx_l, idx_r, left, right, target_l, target_r,
            )
            self.move(self.open_gripper(left), self.open_gripper(right))
            self._release_fruit(idx_l)
            self._release_fruit(idx_r)
            self._belt_dwell(70)
            self._settle_after_drop(idx_l, target_l)
            self._settle_after_drop(idx_r, target_r)

            self.move(
                self.move_by_displacement(left, z=0.06, move_axis="arm"),
                self.move_by_displacement(right, z=0.06, move_axis="arm"),
            )
            self.plan_success = True
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
            if (self.spawn_mode == "parallel"
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
            if (self.spawn_mode == "parallel"
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
