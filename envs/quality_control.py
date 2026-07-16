from ._base_task import Base_Task
from .utils import *
from ._GLOBAL_CONFIGS import *
import sapien
import sapien.render
import numpy as np


class quality_control(Base_Task):
    """A centered conveyor carries colored tiles past a fixed gantry stamp. Two keys sit
    symmetrically beside the belt — red on the left, green on the right. The arms press the
    key that matches each tile's color so the descending stamp marks it.

    Tiles start in a light shade of red or green. A correct stamp darkens the tile; an
    incorrect stamp (wrong key) turns it black. Pressing any key while no tile is under the
    stamp is an empty-press failure. Success requires every tile stamped correctly and no
    empty presses.

    Tile gaps are either equal (`spacing_mode=equal`) or randomly sampled
    (`spacing_mode=random`); random / intentionally enlarged gaps create empty windows
    under the stamp.

    Belt motion and stamp-head descent are step-driven via `_update_kinematic_tasks` so the
    plan / render collection passes stay identical."""

    # ----------------------------------------------------------- class defaults
    N_TILES_DEFAULT = 6
    BELT_SPEED_DEFAULT = 0.0030       # m advanced per physics step
    TILE_SPACING_DEFAULT = 0.08       # center-to-center spacing for equal mode
    SPACING_MODE_DEFAULT = "equal"    # "equal" | "random"
    SPACING_MIN_DEFAULT = 0.07        # random-mode min center-to-center gap
    SPACING_MAX_DEFAULT = 0.22        # random-mode max (large enough for empty stamp)
    EMPTY_GAP_PROB_DEFAULT = 0.35     # chance to insert a large empty gap (both modes)
    EMPTY_GAP_SIZE_DEFAULT = 0.20     # center-to-center size of an intentional empty gap
    COLOR_MODE_DEFAULT = "alternating"  # "alternating" | "random"

    # geometry of the fixed installation (table-local; z added to table top)
    BELT_X = 0.0                      # belt centered on the table (x and y)
    BELT_Y = 0.0
    BELT_HALF_LEN = 0.30              # belt extent along y (far <-> near)
    BELT_HALF_WID = 0.07              # belt extent along x
    BELT_THICK = 0.012                # belt slab half-thickness
    STAMP_Y = -0.20                   # stamp near the near/table edge (tiles exit -y)
    STAMP_UP_DZ = 0.16                # stamp resting height above the belt surface
    STAMP_DOWN_DZ = 0.012             # stamp height above belt surface at full descent
    STAMP_TRAVEL_STEPS = 10           # physics steps for a full stamp down-stroke
    STAMP_HOLD_STEPS = 4              # steps held at the bottom (the mark dwell)

    TILE_HALF = [0.028, 0.032, 0.012]  # smaller tiles so more fit on the belt
    BELT_Y_START = 0.20               # y where the lead tile enters (far end)
    BELT_Y_END = -0.27                # y where tiles leave the belt (near end)
    HIDE_Z = -10.0                    # park exited tiles under the table
    TILE_EXIT_MARGIN = 0.01           # hide once the tile clears the near belt edge

    # symmetric keys: red left / green right, beside the near-edge stamp
    KEY_X = 0.20
    KEY_Y = -0.18                     # next to the stamp near the table edge
    KEY_HALF = [0.025, 0.025, 0.016]
    KEY_HOVER_DIS = 0.06              # fingertip hover height above key top
    KEY_PRESS_DEPTH = 0.065           # tip descend onto / slightly into the key
    # EE frame is 0.12 m behind the TCP along gripper +x (see robot._trans_endpose)
    EE_TO_TCP = 0.12

    LIGHT_COLORS = {
        "red":   [0.95, 0.45, 0.45],
        "green": [0.45, 0.90, 0.50],
    }
    DARK_COLORS = {
        "red":   [0.55, 0.05, 0.05],
        "green": [0.05, 0.45, 0.12],
    }
    BLACK_COLOR = [0.05, 0.05, 0.05]
    KEY_COLORS = {
        "red":   [0.85, 0.10, 0.10],
        "green": [0.10, 0.65, 0.18],
    }

    def setup_demo(self, **kwags):
        self._cfg = kwags.get("task_args", {}).get("quality_control", {})
        # guards: _update_kinematic_tasks runs from the first gripper-open inside
        # _init_task_env_ (BEFORE load_actors) and throughout check_stable.
        self._stamp_ready = False
        self._belt_running = False
        super()._init_task_env_(**kwags)

    # --------------------------------------------------------------- actors
    def load_actors(self):
        cfg = self._cfg
        self.n_tiles = int(cfg.get("n_tiles", cfg.get("n_files", self.N_TILES_DEFAULT)))
        self.color_mode = str(cfg.get("color_mode", self.COLOR_MODE_DEFAULT)).lower()
        if self.color_mode not in ("alternating", "random"):
            self.color_mode = self.COLOR_MODE_DEFAULT

        self.spacing_mode = str(cfg.get("spacing_mode", self.SPACING_MODE_DEFAULT)).lower()
        if self.spacing_mode not in ("equal", "random"):
            self.spacing_mode = self.SPACING_MODE_DEFAULT
        self.tile_spacing = float(cfg.get("tile_spacing",
                                           cfg.get("file_spacing", self.TILE_SPACING_DEFAULT)))
        self.spacing_min = float(cfg.get("spacing_min", self.SPACING_MIN_DEFAULT))
        self.spacing_max = float(cfg.get("spacing_max", self.SPACING_MAX_DEFAULT))
        if self.spacing_max < self.spacing_min:
            self.spacing_min, self.spacing_max = self.spacing_max, self.spacing_min
        self.empty_gap_prob = float(cfg.get("empty_gap_prob", self.EMPTY_GAP_PROB_DEFAULT))
        self.empty_gap_size = float(cfg.get("empty_gap_size", self.EMPTY_GAP_SIZE_DEFAULT))

        self.belt_speed = float(cfg.get("belt_speed",
                                        self.BELT_SPEED_DEFAULT * np.random.uniform(0.8, 1.25)))

        z0 = 0.74 + self.table_z_bias
        self.belt_top_z = z0 + 2 * self.BELT_THICK

        # ---- the belt slab, centered on the table ----
        self.belt = create_box(
            scene=self,
            pose=sapien.Pose([self.BELT_X, self.BELT_Y, z0 + self.BELT_THICK], [1, 0, 0, 0]),
            half_size=[self.BELT_HALF_WID, self.BELT_HALF_LEN, self.BELT_THICK],
            color=[0.18, 0.18, 0.20],
            name="belt",
            is_static=True,
        )

        # ---- fixed gantry + stamp head over a fixed belt point ----
        self.stamp_x = self.BELT_X
        self.stamp_y = self.STAMP_Y
        self.stamp_up_z = self.belt_top_z + self.STAMP_UP_DZ
        self.stamp_down_z = self.belt_top_z + self.STAMP_DOWN_DZ

        post_h = self.STAMP_UP_DZ + 0.10
        for sx in (-1, 1):
            create_box(
                scene=self,
                pose=sapien.Pose([self.stamp_x + sx * (self.BELT_HALF_WID + 0.02),
                                  self.stamp_y, z0 + post_h / 2], [1, 0, 0, 0]),
                half_size=[0.012, 0.012, post_h / 2],
                color=[0.45, 0.45, 0.48],
                name=f"gantry_post_{sx}",
                is_static=True,
            )
        create_box(
            scene=self,
            pose=sapien.Pose([self.stamp_x, self.stamp_y, z0 + post_h], [1, 0, 0, 0]),
            half_size=[self.BELT_HALF_WID + 0.035, 0.014, 0.014],
            color=[0.45, 0.45, 0.48],
            name="gantry_bar",
            is_static=True,
        )

        self.stamp_half = [0.022, 0.022, 0.022]
        self.stamp = create_box(
            scene=self,
            pose=sapien.Pose([self.stamp_x, self.stamp_y, self.stamp_up_z], [1, 0, 0, 0]),
            half_size=self.stamp_half,
            color=[0.55, 0.55, 0.58],
            name="stamp_head",
            is_static=False,
        )
        self._make_kinematic(self.stamp)
        self.stamp_shapes = []
        for c in self.stamp.actor.get_components():
            if isinstance(c, sapien.render.RenderBodyComponent):
                self.stamp_shapes = list(c.render_shapes)

        # ---- red key (left) and green key (right), symmetric about the belt ----
        self.keys = {}
        for color, sign in (("red", -1.0), ("green", 1.0)):
            key = create_box(
                scene=self,
                pose=sapien.Pose(
                    [sign * self.KEY_X, self.KEY_Y, z0 + self.KEY_HALF[2]],
                    [1, 0, 0, 0],
                ),
                half_size=self.KEY_HALF,
                color=self.KEY_COLORS[color],
                name=f"key_{color}",
                is_static=True,
            )
            self.keys[color] = key
            self.add_prohibit_area(key, padding=0.05)

        # ---- tiles riding the belt (light red / light green) ----
        self.tile_colors = self._sample_tile_colors(self.n_tiles)
        self.tile_gaps, self.tile_ys = self._sample_tile_layout(self.n_tiles)
        self.tiles = []
        self.tile_marked = []
        self.tile_correct = []
        self.tile_hidden = []
        self.tile_shapes = []
        self._tile_ride_z = self.belt_top_z + self.TILE_HALF[2]
        for i in range(self.n_tiles):
            y_i = self.tile_ys[i]
            color_name = self.tile_colors[i]
            t = create_box(
                scene=self,
                pose=sapien.Pose(
                    [self.BELT_X, y_i, self._tile_ride_z],
                    [1, 0, 0, 0],
                ),
                half_size=self.TILE_HALF,
                color=self.LIGHT_COLORS[color_name],
                name=f"tile_{i}",
                is_static=False,
            )
            self._make_kinematic(t)
            self.tiles.append(t)
            self.tile_marked.append(False)
            self.tile_correct.append(False)
            self.tile_hidden.append(False)
            shapes = []
            for c in t.actor.get_components():
                if isinstance(c, sapien.render.RenderBodyComponent):
                    shapes = list(c.render_shapes)
            self.tile_shapes.append(shapes)

        # ---- stamp / belt runtime state (step-driven) ----
        self.stamp_phase = "up"       # "up" | "down" | "hold" | "rising"
        self.stamp_phase_step = 0
        self.stamp_requested = False
        self.stamp_key_color = None   # color of the key that triggered the current stroke
        self.empty_press = False      # True if a key was pressed with no tile under stamp
        self.empty_press_count = 0
        self._belt_accum = 0.0
        self.n_correct = 0

        self._tile_init_poses = [t.get_pose() for t in self.tiles]
        self._stamp_init_z = self.stamp_up_z

        self.add_prohibit_area(self.belt, padding=0.02)
        self._stamp_ready = True

    def _sample_tile_colors(self, n):
        if self.color_mode == "random":
            return [str(np.random.choice(["red", "green"])) for _ in range(n)]
        # alternating: randomize which color leads, then G/R/G/R... or R/G/R/G...
        first = str(np.random.choice(["red", "green"]))
        other = "green" if first == "red" else "red"
        return [first if (i % 2 == 0) else other for i in range(n)]

    def _sample_tile_layout(self, n):
        """Return (gaps, ys). gaps[i] is the center-to-center distance from tile i to i+1.

        equal: constant `tile_spacing`, with optional random large empty gaps.
        random: each gap ~ Uniform(spacing_min, spacing_max), plus optional empty gaps.
        Large gaps leave the stamp empty between tiles — pressing then is a failure.
        """
        gaps = []
        for _ in range(max(0, n - 1)):
            if self.spacing_mode == "random":
                gap = float(np.random.uniform(self.spacing_min, self.spacing_max))
            else:
                gap = float(self.tile_spacing)
            # randomly enlarge some gaps so the stamp sees an empty window
            if self.empty_gap_prob > 0.0 and np.random.rand() < self.empty_gap_prob:
                gap = max(gap, float(self.empty_gap_size))
            gaps.append(gap)

        ys = [float(self.BELT_Y_START)]
        for g in gaps:
            ys.append(ys[-1] + g)
        return gaps, ys

    def _reset_belt(self):
        """Restore tiles + stamp to their authored start state at the policy start."""
        for t, pose in zip(self.tiles, self._tile_init_poses):
            t.actor.set_pose(pose)
        for i, shapes in enumerate(self.tile_shapes):
            c = self.LIGHT_COLORS[self.tile_colors[i]]
            for s in shapes:
                try:
                    s.material.set_base_color([*c, 1.0])
                except Exception:
                    pass
        self.tile_marked = [False] * self.n_tiles
        self.tile_correct = [False] * self.n_tiles
        self.tile_hidden = [False] * self.n_tiles
        self.n_correct = 0
        self.empty_press = False
        self.empty_press_count = 0
        self._set_pose_z(self.stamp, self.stamp_up_z)
        self._recolor_stamp(None)
        self.stamp_phase = "up"
        self.stamp_phase_step = 0
        self.stamp_requested = False
        self.stamp_key_color = None

    def _hide_tile(self, i):
        if self.tile_hidden[i]:
            return
        p = self.tiles[i].get_pose()
        self.tiles[i].actor.set_pose(
            sapien.Pose([p.p[0], p.p[1], self.HIDE_Z], p.q)
        )
        self.tile_hidden[i] = True

    def _tile_has_exited_belt(self, y, marked=False):
        # tiles travel in -y. Hide once off the near belt edge; stamped tiles may
        # disappear a bit earlier (past the stamp) so they don't pile up in view.
        exit_y = -self.BELT_HALF_LEN - self.TILE_EXIT_MARGIN
        if marked:
            exit_y = max(exit_y, self.stamp_y - 0.08)
        return y < exit_y

    def _make_kinematic(self, actor):
        for c in actor.actor.get_components():
            if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                c.set_kinematic(True)

    def _set_pose_z(self, actor, z):
        p = actor.get_pose()
        actor.actor.set_pose(sapien.Pose([p.p[0], p.p[1], z], p.q))

    def _recolor_stamp(self, color_name):
        if color_name is None:
            rgba = [0.55, 0.55, 0.58, 1.0]
        else:
            rgba = [*self.KEY_COLORS[color_name], 1.0]
        for s in self.stamp_shapes:
            try:
                s.material.set_base_color(rgba)
            except Exception:
                pass

    # ----------------------------------------------------- step-driven dynamics
    def _advance_belt(self):
        for i, t in enumerate(self.tiles):
            if self.tile_hidden[i]:
                continue
            p = t.actor.get_pose()
            ny = p.p[1] - self.belt_speed
            if self._tile_has_exited_belt(ny, marked=self.tile_marked[i]):
                self._hide_tile(i)
                continue
            t.actor.set_pose(sapien.Pose([p.p[0], ny, self._tile_ride_z], p.q))

    def _tile_under_stamp(self, require_unmarked=True):
        """Index of the tile currently under the stamp footprint, or None if empty."""
        sx, sy = self.stamp_x, self.stamp_y
        best_i, best_d = None, 1e9
        for i, t in enumerate(self.tiles):
            if self.tile_hidden[i]:
                continue
            if require_unmarked and self.tile_marked[i]:
                continue
            tp = t.get_pose().p
            if abs(tp[0] - sx) < (self.TILE_HALF[0] + self.stamp_half[0]) and \
               abs(tp[1] - sy) < (self.TILE_HALF[1] + self.stamp_half[1]):
                d = abs(tp[1] - sy)
                if d < best_d:
                    best_d, best_i = d, i
        return best_i

    def _record_mark(self):
        """At stamp contact, mark the tile under the stamp — or flag an empty press."""
        pressed = self.stamp_key_color
        if pressed is None:
            return
        best_i = self._tile_under_stamp(require_unmarked=True)
        if best_i is None:
            # key pressed with nothing under the stamp → failure
            self.empty_press = True
            self.empty_press_count += 1
            return

        tile_color = self.tile_colors[best_i]
        correct = (pressed == tile_color)
        self.tile_marked[best_i] = True
        self.tile_correct[best_i] = correct
        if correct:
            rgba = [*self.DARK_COLORS[tile_color], 1.0]
        else:
            rgba = [*self.BLACK_COLOR, 1.0]
        for s in self.tile_shapes[best_i]:
            try:
                s.material.set_base_color(rgba)
            except Exception:
                pass
        self.n_correct = int(sum(1 for c in self.tile_correct if c))

    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()

        if not getattr(self, "_stamp_ready", False) or not getattr(self, "_belt_running", False):
            return

        self._advance_belt()

        if self.stamp_phase == "up":
            self._set_pose_z(self.stamp, self.stamp_up_z)
            if self.stamp_requested:
                self.stamp_requested = False
                self.stamp_phase = "down"
                self.stamp_phase_step = 0
                self._recolor_stamp(self.stamp_key_color)
        elif self.stamp_phase == "down":
            self.stamp_phase_step += 1
            t = min(1.0, self.stamp_phase_step / self.STAMP_TRAVEL_STEPS)
            z = self.stamp_up_z + (self.stamp_down_z - self.stamp_up_z) * t
            self._set_pose_z(self.stamp, z)
            if self.stamp_phase_step >= self.STAMP_TRAVEL_STEPS:
                self.stamp_phase = "hold"
                self.stamp_phase_step = 0
                self._record_mark()
        elif self.stamp_phase == "hold":
            self.stamp_phase_step += 1
            if self.stamp_phase_step >= self.STAMP_HOLD_STEPS:
                self.stamp_phase = "rising"
                self.stamp_phase_step = 0
        elif self.stamp_phase == "rising":
            self.stamp_phase_step += 1
            t = min(1.0, self.stamp_phase_step / self.STAMP_TRAVEL_STEPS)
            z = self.stamp_down_z + (self.stamp_up_z - self.stamp_down_z) * t
            self._set_pose_z(self.stamp, z)
            if self.stamp_phase_step >= self.STAMP_TRAVEL_STEPS:
                self.stamp_phase = "up"
                self.stamp_phase_step = 0
                self._recolor_stamp(None)
                self.stamp_key_color = None

    def _press_key(self, color):
        """Fire the stamp with the given key color (red or green)."""
        self.stamp_key_color = color
        self.stamp_requested = True

    def _belt_dwell(self, steps):
        """Advance belt + stamp for `steps` physics steps, recording frames periodically."""
        prev = self._belt_running
        self._belt_running = True
        for i in range(int(steps)):
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (i % self.save_freq == 0):
                self._take_picture()
        self._belt_running = prev

    # ------------------------------------------------------------- policy
    def _key_tip_pose(self, color, tip_z_above_top):
        """EE pose with fingertips pointing straight down onto the key.

        Uses GRASP top_down quats where gripper local +x = world -z, so the
        closed fingertip (TCP) is the lowest point — not the wrist. The planned
        EE frame sits EE_TO_TCP behind the TCP along that axis.
        """
        sign = -1.0 if color == "red" else 1.0
        z0 = 0.74 + self.table_z_bias
        key_top_z = z0 + 2.0 * self.KEY_HALF[2]
        tcp_z = key_top_z + tip_z_above_top
        # local +x -> -z, so EE_z = TCP_z + EE_TO_TCP
        ee_z = tcp_z + self.EE_TO_TCP
        # pure top_down: gripper local +x = world -z so closed fingertips point straight down
        quat = GRASP_DIRECTION_DIC["top_down"]
        return [sign * self.KEY_X, self.KEY_Y, ee_z, *quat]

    def _hover_key(self, color):
        arm = ArmTag("left" if color == "red" else "right")
        return self.move_to_pose(arm, self._key_tip_pose(color, self.KEY_HOVER_DIS))

    def play_once(self):
        import os
        dbg = bool(os.environ.get("QC_DEBUG") or os.environ.get("STAMP_DEBUG"))
        left = ArmTag("left")
        right = ArmTag("right")

        # Belt motion advances ONLY inside _belt_dwell, never during arm self.move(...).
        self._reset_belt()
        self._belt_running = False

        # close grippers into a tip, then hover tip-down over both keys
        self.move(self.close_gripper(left), self.close_gripper(right))
        self.move(self._hover_key("red"), self._hover_key("green"))

        lead = self.belt_speed * self.STAMP_TRAVEL_STEPS
        cycle_steps = (2 * self.STAMP_TRAVEL_STEPS + self.STAMP_HOLD_STEPS) + 4
        press = self.KEY_PRESS_DEPTH

        max_wait = 8000
        waited = 0
        idx = 0
        n = self.n_tiles
        while idx < n and waited < max_wait:
            i = idx
            if self.tile_marked[i] or self.tile_hidden[i]:
                idx += 1
                continue
            tile_y = float(self.tiles[i].get_pose().p[1])
            d = tile_y - self.stamp_y
            if d <= lead + 1e-6 and d > -0.04:
                color = self.tile_colors[i]
                arm = left if color == "red" else right
                # re-seat tip above the key, then press straight down with the fingertip
                self.move(self._hover_key(color))
                self.move(self.move_by_displacement(arm, z=-press))
                self._press_key(color)
                self.move(self.move_by_displacement(arm, z=press))
                self._belt_dwell(cycle_steps)
                if dbg:
                    print(f"[qc] fired tile {i} color={color}: marked={self.tile_marked[i]} "
                          f"correct={self.tile_correct[i]} plan={self.plan_success}",
                          flush=True)
                idx += 1
            else:
                step = max(1, int(min(8, (d - lead) / max(self.belt_speed, 1e-6))))
                self._belt_dwell(step)
                waited += step

        # run the belt out so stamped tiles clear the near edge and disappear
        exit_y = -self.BELT_HALF_LEN - self.TILE_EXIT_MARGIN
        furthest = min(
            (float(t.get_pose().p[1]) for t, h in zip(self.tiles, self.tile_hidden) if not h),
            default=exit_y,
        )
        exit_steps = int(np.ceil(
            (furthest - exit_y + 0.08) / max(self.belt_speed, 1e-6)
        )) + 30
        self._belt_dwell(max(80, exit_steps))
        # force-hide anything past mid-belt and grab a few empty-belt frames
        for i, t in enumerate(self.tiles):
            if not self.tile_hidden[i] and float(t.get_pose().p[1]) < 0.05:
                self._hide_tile(i)
        self._belt_dwell(max(self.save_freq or 15, 15) * 3)

        if dbg:
            print(f"[qc] done colors={self.tile_colors} gaps={self.tile_gaps} "
                  f"marked={self.tile_marked} correct={self.tile_correct} "
                  f"hidden={self.tile_hidden} empty_press={self.empty_press} "
                  f"plan={self.plan_success}", flush=True)

        self.info["info"] = {
            "{A}": "colored tiles",
            "{a}": "both arms",
        }
        return self.info

    # ------------------------------------------------------------- success
    def check_success(self):
        # empty key-press (no tile under stamp) is an immediate failure
        if self.empty_press:
            return False
        # every tile must be stamped with the matching color
        if not all(self.tile_marked):
            return False
        return all(self.tile_correct)

    def get_obs(self):
        obs = super().get_obs()
        obs["quality_control"] = {
            "n_tiles": int(self.n_tiles),
            "n_marked": int(sum(1 for m in self.tile_marked if m)),
            "n_correct": int(sum(1 for c in self.tile_correct if c)),
            "belt_speed": float(self.belt_speed),
            "color_mode": str(self.color_mode),
            "spacing_mode": str(self.spacing_mode),
            "tile_gaps": [float(g) for g in self.tile_gaps],
            "tile_colors": list(self.tile_colors),
            "tile_marked": [bool(m) for m in self.tile_marked],
            "tile_correct": [bool(c) for c in self.tile_correct],
            "tile_hidden": [bool(h) for h in self.tile_hidden],
            "tile_positions": [float(t.get_pose().p[1]) for t in self.tiles],
            "empty_press": bool(self.empty_press),
            "empty_press_count": int(self.empty_press_count),
            "stamp_z": float(self.stamp.get_pose().p[2]),
            "stamp_key_color": None if self.stamp_key_color is None else str(self.stamp_key_color),
            "tile_under_stamp": (
                self._tile_under_stamp(require_unmarked=False)
                if getattr(self, "_stamp_ready", False) else None
            ),
        }
        return obs
