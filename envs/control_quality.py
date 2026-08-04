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

    Task options (set in ``task_args.quality_control``; independent toggles):
      - Default — alternating red/green, no black distractors
          ``color_mode: alternating``
          ``black_frac_max: 0.0``
      - Option 1 — randomized red/green pattern: ``color_mode: random``
          Non-black tiles are independently red or green (not alternating).
          CLI: ``--task-arg color_mode=random`` or legacy ``--option 1``.
      - Option 2 — black distractor tiles: ``black_frac_max`` > 0
          Randomly blacken up to this fraction of tiles (default 0.50 when enabled via
          ``--option 2``). Do not press any key on black; pressing black fails.
          CLI: ``--task-arg black_frac_max=0.5`` or legacy ``--option 2``.

    Each tile stops under the stamp for up to ``tile_pause_s`` (default 2.0 s). Failing to
    correctly stamp a red/green tile within that window marks it missed → episode failure.
    Wrong-key stamps and key presses on black also fail.

    Inter-tile gaps are either equal (``spacing_mode=equal``) or randomly sampled
    (``spacing_mode=random``).

    Belt motion and stamp-head descent are step-driven via ``_update_kinematic_tasks`` so the
    plan / render collection passes stay identical."""

    # ----------------------------------------------------------- class defaults
    N_TILES_DEFAULT = 6
    BELT_SPEED_DEFAULT = 0.0030       # m advanced per physics step
    TILE_SPACING_DEFAULT = 0.08       # center-to-center spacing for equal mode
    SPACING_MODE_DEFAULT = "equal"    # "equal" | "random"
    SPACING_MIN_DEFAULT = 0.07        # random-mode min center-to-center gap
    SPACING_MAX_DEFAULT = 0.14        # random-mode max center-to-center gap
    # Default = no distractors; Opt 2 enables black tiles via black_frac_max > 0
    BLACK_FRAC_MAX_DEFAULT = 0.0
    BLACK_FRAC_WHEN_OPT2 = 0.50       # used when legacy --option 2 enables distractors
    COLOR_MODE_DEFAULT = "alternating"  # Default; Opt 1 = "random"
    TILE_PAUSE_S_DEFAULT = 2.0        # max hold under stamp (seconds)

    # geometry of the fixed installation (table-local; z added to table top)
    BELT_X = 0.0
    BELT_Y = 0.0
    BELT_HALF_LEN = 0.30
    BELT_HALF_WID = 0.07
    BELT_THICK = 0.012
    STAMP_Y = -0.20
    STAMP_UP_DZ = 0.16
    STAMP_DOWN_DZ = 0.012
    STAMP_TRAVEL_STEPS = 10
    STAMP_HOLD_STEPS = 4

    TILE_HALF = [0.028, 0.032, 0.012]
    BELT_Y_START = 0.20
    BELT_Y_END = -0.27
    HIDE_Z = -10.0
    TILE_EXIT_MARGIN = 0.01

    KEY_X = 0.20
    KEY_Y = -0.18
    KEY_HALF = [0.025, 0.025, 0.016]          # colored keycap
    # Larger, thinner black base under each key (matches catch_marbles_trapdoors look).
    KEY_BASE_HALF = [0.032, 0.032, 0.005]
    KEY_BASE_COLOR = [0.08, 0.08, 0.08]
    KEY_HOVER_DIS = 0.06
    KEY_PRESS_DEPTH = 0.065
    EE_TO_TCP = 0.12

    LIGHT_COLORS = {
        "red":   [0.95, 0.45, 0.45],
        "green": [0.45, 0.90, 0.50],
        # deep charcoal — distinct from the mid-gray belt so reject tiles read as black
        "black": [0.02, 0.02, 0.02],
    }
    DARK_COLORS = {
        "red":   [0.55, 0.05, 0.05],
        "green": [0.05, 0.45, 0.12],
        "black": [0.02, 0.02, 0.02],
    }
    BLACK_COLOR = [0.02, 0.02, 0.02]   # wrong-key stamp outcome
    KEY_COLORS = {
        "red":   [0.85, 0.10, 0.10],
        "green": [0.10, 0.65, 0.18],
    }
    BELT_COLOR = [0.42, 0.42, 0.45]    # lighter belt so black tiles contrast

    def setup_demo(self, **kwags):
        self._cfg = kwags.get("task_args", {}).get("quality_control", {})
        self._apply_legacy_option()
        self._stamp_ready = False
        self._belt_running = False
        self._stamp_active = False
        super()._init_task_env_(**kwags)

    def _apply_legacy_option(self):
        """Map record_demo ``--option`` / config ``option`` onto named toggles.

        1 / random / color_mode → Opt 1 color_mode=random
        2 / black / distractor / black_frac_max → Opt 2 black_frac_max=0.50
        """
        legacy = self._cfg.get("option", None)
        if legacy is None:
            return
        key = {
            1: "color_mode_random",
            2: "black_distractor",
            "1": "color_mode_random",
            "2": "black_distractor",
            "random": "color_mode_random",
            "color_mode": "color_mode_random",
            "color_mode_random": "color_mode_random",
            "black": "black_distractor",
            "distractor": "black_distractor",
            "black_frac_max": "black_distractor",
            "black_distractor": "black_distractor",
        }.get(legacy if not isinstance(legacy, str) else legacy.strip().lower())
        if key == "color_mode_random":
            if "color_mode" not in self._cfg:
                self._cfg["color_mode"] = "random"
        elif key == "black_distractor":
            if "black_frac_max" not in self._cfg:
                self._cfg["black_frac_max"] = self.BLACK_FRAC_WHEN_OPT2
        else:
            raise ValueError(
                "quality_control option must be 1/color_mode=random or "
                "2/black_frac_max (or set color_mode / black_frac_max directly)"
            )

    def _option_label(self) -> str:
        parts = []
        if getattr(self, "color_mode", self.COLOR_MODE_DEFAULT) == "random":
            parts.append("option 1")
        if float(getattr(self, "black_frac_max", 0.0)) > 0.0:
            parts.append("option 2")
        return ", ".join(parts) if parts else "baseline"

    def _get_tile_pause_steps(self):
        """Max hold under the stamp, in physics steps.

        Prefer tile_pause_s (seconds, default 2.0). Explicit tile_pause_steps still
        wins when provided without tile_pause_s, for older configs.
        """
        has_steps = "tile_pause_steps" in self._cfg
        has_secs = "tile_pause_s" in self._cfg or "tile_pause_sec" in self._cfg
        if has_steps and not has_secs:
            return max(1, int(self._cfg.get("tile_pause_steps")))
        pause_s = float(
            self._cfg.get(
                "tile_pause_s",
                self._cfg.get("tile_pause_sec", self.TILE_PAUSE_S_DEFAULT),
            )
        )
        pause_s = max(0.0, pause_s)
        dt = float(self.scene.get_timestep()) if hasattr(self, "scene") else (1.0 / 250.0)
        return max(1, int(round(pause_s / max(dt, 1e-8))))

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
        self.black_frac_max = float(cfg.get("black_frac_max", self.BLACK_FRAC_MAX_DEFAULT))
        self.black_frac_max = float(np.clip(self.black_frac_max, 0.0, 1.0))

        self.tile_pause_steps = self._get_tile_pause_steps()
        self.tile_pause_s = float(self.tile_pause_steps) * float(self.scene.get_timestep())

        self.belt_speed = float(cfg.get("belt_speed",
                                        self.BELT_SPEED_DEFAULT * np.random.uniform(0.8, 1.25)))

        z0 = 0.74 + self.table_z_bias
        self.belt_top_z = z0 + 2 * self.BELT_THICK

        self.belt = create_box(
            scene=self,
            pose=sapien.Pose([self.BELT_X, self.BELT_Y, z0 + self.BELT_THICK], [1, 0, 0, 0]),
            half_size=[self.BELT_HALF_WID, self.BELT_HALF_LEN, self.BELT_THICK],
            color=self.BELT_COLOR,
            name="belt",
            is_static=True,
        )

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

        self.keys = {}
        self.key_bases = {}
        base_hz = float(self.KEY_BASE_HALF[2])
        cap_hz = float(self.KEY_HALF[2])
        for color, sign in (("red", -1.0), ("green", 1.0)):
            kx = sign * self.KEY_X
            base_z = z0 + base_hz
            cap_z = z0 + 2.0 * base_hz + cap_hz
            base = create_box(
                scene=self,
                pose=sapien.Pose([kx, self.KEY_Y, base_z], [1, 0, 0, 0]),
                half_size=list(self.KEY_BASE_HALF),
                color=list(self.KEY_BASE_COLOR),
                name=f"key_base_{color}",
                is_static=True,
            )
            key = create_box(
                scene=self,
                pose=sapien.Pose([kx, self.KEY_Y, cap_z], [1, 0, 0, 0]),
                half_size=list(self.KEY_HALF),
                color=self.KEY_COLORS[color],
                name=f"key_{color}",
                is_static=True,
            )
            self.key_bases[color] = base
            self.keys[color] = key
            self.add_prohibit_area(base, padding=0.05)
            self.add_prohibit_area(key, padding=0.05)

        # ---- tiles (red / green / black) ----
        self.tile_gaps, self.tile_ys = self._sample_tile_layout(self.n_tiles)
        self.tile_colors = self._sample_tile_colors(self.n_tiles)
        self.tiles = []
        self.tile_marked = []       # stamped (red/green) or incorrectly pressed
        self.tile_correct = []      # correct stamp for red/green
        self.tile_skipped = []      # black tile correctly passed without a press
        self.tile_missed = []       # red/green not stamped in time (or wrong)
        self.tile_hidden = []
        self.tile_shapes = []
        self._tile_ride_z = self.belt_top_z + self.TILE_HALF[2]
        for i in range(self.n_tiles):
            color_name = self.tile_colors[i]
            t = create_box(
                scene=self,
                pose=sapien.Pose(
                    [self.BELT_X, self.tile_ys[i], self._tile_ride_z],
                    [1, 0, 0, 0],
                ),
                half_size=self.TILE_HALF,
                color=self.LIGHT_COLORS[color_name],
                name=f"tile_{i}_{color_name}",
                is_static=False,
            )
            self._make_kinematic(t)
            self.tiles.append(t)
            self.tile_marked.append(False)
            self.tile_correct.append(False)
            self.tile_skipped.append(False)
            self.tile_missed.append(False)
            self.tile_hidden.append(False)
            shapes = []
            for c in t.actor.get_components():
                if isinstance(c, sapien.render.RenderBodyComponent):
                    shapes = list(c.render_shapes)
            self.tile_shapes.append(shapes)
            # force material color (sapien sometimes needs an explicit set after create)
            self._paint_tile(i, self.LIGHT_COLORS[color_name])

        self.stamp_phase = "up"
        self.stamp_phase_step = 0
        self.stamp_requested = False
        self.stamp_key_color = None
        self.black_press = False      # pressed a key on a black tile
        self.black_press_count = 0
        self._stamp_active = False
        # The autonomous policy explicitly owns its stop windows. Interactive
        # launchers opt into this step-driven equivalent.
        self._interactive_tile_pause = False
        self._interactive_pause_tile = None
        self._interactive_pause_steps = 0
        self._interactive_pause_released = None
        self.n_correct = 0

        self._tile_init_poses = [t.get_pose() for t in self.tiles]
        self._stamp_init_z = self.stamp_up_z

        self.add_prohibit_area(self.belt, padding=0.02)
        self._stamp_ready = True

    def _sample_tile_colors(self, n):
        """Sample red/green (per color_mode), then randomly blacken up to black_frac_max.

        Guarantees at least one black tile when black_frac_max > 0 and n >= 2, so reject
        tiles reliably appear (uniform 0..max previously often rolled zero).
        """
        if self.color_mode == "random":
            colors = [str(np.random.choice(["red", "green"])) for _ in range(n)]
        else:
            first = str(np.random.choice(["red", "green"]))
            other = "green" if first == "red" else "red"
            colors = [first if (i % 2 == 0) else other for i in range(n)]

        max_black = int(np.floor(n * self.black_frac_max + 1e-9))
        if max_black <= 0 or n < 1:
            return colors
        # at least 1 black when enabled; at most floor(n * black_frac_max)
        lo = 1 if n >= 2 else 0
        n_black = int(np.random.randint(lo, max_black + 1))
        n_black = int(np.clip(n_black, 0, n))
        if n_black > 0:
            idxs = np.random.choice(n, size=n_black, replace=False)
            for i in np.atleast_1d(idxs):
                colors[int(i)] = "black"
        return colors

    def _sample_tile_layout(self, n):
        min_gap = max(float(2.0 * self.TILE_HALF[1]), float(self.spacing_min))
        gaps = []
        for _ in range(max(0, n - 1)):
            if self.spacing_mode == "random":
                gap = float(np.random.uniform(min_gap, max(min_gap, self.spacing_max)))
            else:
                gap = max(float(self.tile_spacing), min_gap)
            gaps.append(gap)
        ys = [float(self.BELT_Y_START)]
        for g in gaps:
            ys.append(ys[-1] + g)
        return gaps, ys

    def _tile_base_color(self, i):
        return self.LIGHT_COLORS[self.tile_colors[i]]

    def _paint_tile(self, i, rgb):
        rgba = [float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0]
        for s in self.tile_shapes[i]:
            try:
                mat = s.material
                mat.set_base_color(rgba)
                if hasattr(mat, "set_metallic"):
                    mat.set_metallic(0.0)
                if hasattr(mat, "set_roughness"):
                    mat.set_roughness(0.85)
            except Exception:
                try:
                    s.material.base_color = rgba
                except Exception:
                    pass

    def _reset_belt(self):
        for t, pose in zip(self.tiles, self._tile_init_poses):
            t.actor.set_pose(pose)
        for i in range(self.n_tiles):
            self._paint_tile(i, self._tile_base_color(i))
        self.tile_marked = [False] * self.n_tiles
        self.tile_correct = [False] * self.n_tiles
        self.tile_skipped = [False] * self.n_tiles
        self.tile_missed = [False] * self.n_tiles
        self.tile_hidden = [False] * self.n_tiles
        self.n_correct = 0
        self.black_press = False
        self.black_press_count = 0
        self._stamp_active = False
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

    def _tile_has_exited_belt(self, y, done=False):
        exit_y = -self.BELT_HALF_LEN - self.TILE_EXIT_MARGIN
        if done:
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
    def enable_interactive_tile_pause(self):
        """Use the task's physics-step pause window for interactive control."""
        self._interactive_tile_pause = True
        self._interactive_pause_tile = None
        self._interactive_pause_steps = 0
        self._interactive_pause_released = None

    def _advance_belt(self):
        if self._interactive_tile_pause:
            paused_tile = self._interactive_pause_tile
            if paused_tile is not None:
                self._interactive_pause_steps += 1
                if self._interactive_pause_steps < self.tile_pause_steps:
                    return
                self._interactive_pause_tile = None
                self._interactive_pause_steps = 0
                self._interactive_pause_released = paused_tile

        for i, t in enumerate(self.tiles):
            if self.tile_hidden[i]:
                continue
            p = t.actor.get_pose()
            ny = p.p[1] - self.belt_speed
            done = self.tile_marked[i] or self.tile_skipped[i] or self.tile_missed[i]
            if self._tile_has_exited_belt(ny, done=done):
                self._hide_tile(i)
                continue
            t.actor.set_pose(sapien.Pose([p.p[0], ny, self._tile_ride_z], p.q))

        if self._interactive_tile_pause:
            under = self._tile_under_stamp(require_unhandled=True)
            if under is None:
                self._interactive_pause_released = None
            elif under != self._interactive_pause_released:
                # A tile enters the capture band on a discrete belt step. Snap
                # its center onto the stamp before beginning the dwell so every
                # interactive stop has the same, gap-free alignment.
                pose = self.tiles[under].get_pose()
                self.tiles[under].actor.set_pose(sapien.Pose(
                    [self.stamp_x, self.stamp_y, self._tile_ride_z], pose.q
                ))
                self._interactive_pause_tile = under
                self._interactive_pause_steps = 0

    def _tile_under_stamp(self, require_unhandled=True):
        sx, sy = self.stamp_x, self.stamp_y
        best_i, best_d = None, 1e9
        for i, t in enumerate(self.tiles):
            if self.tile_hidden[i]:
                continue
            if require_unhandled and (
                self.tile_marked[i] or self.tile_skipped[i] or self.tile_missed[i]
            ):
                continue
            tp = t.get_pose().p
            if abs(tp[0] - sx) < (self.TILE_HALF[0] + self.stamp_half[0]) and \
               abs(tp[1] - sy) < (self.TILE_HALF[1] + self.stamp_half[1]):
                d = abs(tp[1] - sy)
                if d < best_d:
                    best_d, best_i = d, i
        return best_i

    def _mark_missed_tile(self, i):
        if self.tile_missed[i] or self.tile_marked[i] or self.tile_skipped[i]:
            return
        self.tile_missed[i] = True
        # keep light color so a miss is visually distinct from a wrong-key black stamp

    def _run_belt_until_under_stamp(self, i):
        """Advance the belt continuously until tile i is centered under the stamp.

        No pose snapping — tiles ride the belt at ``belt_speed`` until their y
        reaches ``stamp_y``, then the belt freezes for the stop window.
        """
        tol = 0.5 * max(float(self.belt_speed), 1e-6)
        while True:
            if self.tile_hidden[i]:
                return False
            tile_y = float(self.tiles[i].get_pose().p[1])
            d = tile_y - self.stamp_y
            if d <= tol:
                return True
            # Chunk when far; single steps near the stamp so we don't overshoot hard.
            if d > 8.0 * self.belt_speed:
                self._belt_dwell(8)
            else:
                self._belt_dwell(1)

    def _record_mark(self):
        """At stamp contact: mark red/green, or fail if the tile is black / missing."""
        pressed = self.stamp_key_color
        if pressed is None:
            return
        best_i = self._tile_under_stamp(require_unhandled=True)
        if best_i is None:
            return

        tile_color = self.tile_colors[best_i]
        if tile_color == "black":
            # any key press on a black tile is a failure
            self.black_press = True
            self.black_press_count += 1
            self.tile_marked[best_i] = True
            self.tile_correct[best_i] = False
            self._paint_tile(best_i, self.BLACK_COLOR)
            return

        correct = (pressed == tile_color)
        self.tile_marked[best_i] = True
        self.tile_correct[best_i] = correct
        if correct:
            self._paint_tile(best_i, self.DARK_COLORS[tile_color])
        else:
            self._paint_tile(best_i, self.BLACK_COLOR)
            self.tile_missed[best_i] = True
        self.n_correct = int(sum(
            1 for i, c in enumerate(self.tile_correct)
            if c and self.tile_colors[i] != "black"
        ))

    def _step_stamp(self):
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

    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
        if not getattr(self, "_stamp_ready", False):
            return
        if getattr(self, "_belt_running", False):
            self._advance_belt()
        if getattr(self, "_belt_running", False) or getattr(self, "_stamp_active", False):
            self._step_stamp()

    def _press_key(self, color):
        self.stamp_key_color = color
        self.stamp_requested = True

    def _belt_dwell(self, steps):
        prev = self._belt_running
        self._belt_running = True
        for i in range(int(steps)):
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (i % self.save_freq == 0):
                self._take_picture()
        self._belt_running = prev

    def _stationary_dwell(self, steps):
        """Freeze the belt while still allowing a stamp press / cycle."""
        prev_belt = self._belt_running
        prev_stamp = getattr(self, "_stamp_active", False)
        self._belt_running = False
        self._stamp_active = True
        for i in range(int(steps)):
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (i % self.save_freq == 0):
                self._take_picture()
        self._belt_running = prev_belt
        self._stamp_active = prev_stamp

    def _stamp_cycle_steps(self):
        return int(2 * self.STAMP_TRAVEL_STEPS + self.STAMP_HOLD_STEPS) + 4

    def _wait_stamp_idle(self, max_steps=None):
        """Advance physics (belt frozen) until the stamp returns to rest."""
        budget = int(max_steps if max_steps is not None else (self._stamp_cycle_steps() + 8))
        prev_belt = self._belt_running
        prev_stamp = getattr(self, "_stamp_active", False)
        self._belt_running = False
        self._stamp_active = True
        for i in range(max(1, budget)):
            if (self.stamp_phase == "up"
                    and not self.stamp_requested
                    and self.stamp_key_color is None):
                break
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (i % self.save_freq == 0):
                self._take_picture()
        self._belt_running = prev_belt
        self._stamp_active = prev_stamp

    def _handle_tile_stop(self, i, dbg=False):
        """Freeze the belt with tile i under the stamp for up to tile_pause_s.

        Caller must already have driven the belt continuously so the tile sits under
        the punch (no teleport). The stamp descends when the matching key is pressed
        (gantry is driven concurrently with the press motion). ``tile_pause_s`` is a
        maximum deadline: failing to correctly stamp a red/green tile fails the
        episode; black tiles must be skipped with no key press.
        """
        pause_budget = max(1, int(self.tile_pause_steps))
        color = self.tile_colors[i]
        cycle_steps = self._stamp_cycle_steps()

        if color == "black":
            # Brief pause under the stamp — do NOT press either key.
            self._stationary_dwell(min(pause_budget, cycle_steps))
            if self.tile_marked[i]:
                self.tile_missed[i] = True
            else:
                self.tile_skipped[i] = True
            if dbg:
                print(f"[qc] skipped black tile {i} "
                      f"black_press={self.black_press} "
                      f"y={float(self.tiles[i].get_pose().p[1]):.4f}", flush=True)
            return

        arm = ArmTag("left" if color == "red" else "right")
        press = self.KEY_PRESS_DEPTH
        # Keep belt frozen; keep stamp active so the head descends while the key is pressed.
        prev_belt = self._belt_running
        prev_stamp = getattr(self, "_stamp_active", False)
        self._belt_running = False
        self._stamp_active = True
        self.move(self._hover_key(color))
        # Trigger the gantry as the press begins — stamp fires with the button press.
        self._press_key(color)
        self.move(self.move_by_displacement(arm, z=-press))
        self.move(self.move_by_displacement(arm, z=press))
        self._belt_running = prev_belt
        self._stamp_active = prev_stamp

        # Finish any remaining stamp travel while still frozen under the deadline.
        self._wait_stamp_idle(min(pause_budget, cycle_steps + 8))

        if not (self.tile_marked[i] and self.tile_correct[i]):
            # Failed to stamp a valid red/green tile within the stop window.
            self._mark_missed_tile(i)

        if dbg:
            print(f"[qc] stop tile {i} color={color}: "
                  f"marked={self.tile_marked[i]} correct={self.tile_correct[i]} "
                  f"missed={self.tile_missed[i]} plan={self.plan_success} "
                  f"y={float(self.tiles[i].get_pose().p[1]):.4f}", flush=True)
    # ------------------------------------------------------------- policy
    def _key_tip_pose(self, color, tip_z_above_top):
        sign = -1.0 if color == "red" else 1.0
        z0 = 0.74 + self.table_z_bias
        # Cap sits on the thin black base.
        key_top_z = z0 + 2.0 * float(self.KEY_BASE_HALF[2]) + 2.0 * float(self.KEY_HALF[2])
        tcp_z = key_top_z + tip_z_above_top
        ee_z = tcp_z + self.EE_TO_TCP
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

        self._reset_belt()
        self._belt_running = False

        self.move(self.close_gripper(left), self.close_gripper(right))
        self.move(self._hover_key("red"), self._hover_key("green"))

        if dbg:
            print(f"[qc] tile_colors={self.tile_colors} "
                  f"n_black={sum(1 for c in self.tile_colors if c == 'black')} "
                  f"color_mode={self.color_mode} black_frac_max={self.black_frac_max} "
                  f"tile_pause_s={self.tile_pause_s:.3f}",
                  flush=True)

        # Drive the belt continuously: each tile rides in, stops under the punch,
        # gets stamped (or skipped if black), then the belt resumes.
        for i in range(self.n_tiles):
            if (self.tile_marked[i] or self.tile_skipped[i]
                    or self.tile_missed[i] or self.tile_hidden[i]):
                continue
            color = self.tile_colors[i]
            # Pre-hover the matching key so the press coincides with the stop.
            if color != "black":
                self.move(self._hover_key(color))
            arrived = self._run_belt_until_under_stamp(i)
            if not arrived:
                if color != "black":
                    self._mark_missed_tile(i)
                continue
            self._handle_tile_stop(i, dbg=dbg)

        # Any red/green that never got a correct stamp counts as missed.
        for i, color in enumerate(self.tile_colors):
            if color == "black":
                continue
            if not (self.tile_marked[i] and self.tile_correct[i]):
                if not self.tile_missed[i]:
                    self._mark_missed_tile(i)

        exit_y = -self.BELT_HALF_LEN - self.TILE_EXIT_MARGIN
        furthest = min(
            (float(t.get_pose().p[1]) for t, h in zip(self.tiles, self.tile_hidden) if not h),
            default=exit_y,
        )
        exit_steps = int(np.ceil(
            (furthest - exit_y + 0.08) / max(self.belt_speed, 1e-6)
        )) + 30
        self._belt_dwell(max(80, exit_steps))
        for i, t in enumerate(self.tiles):
            if not self.tile_hidden[i] and float(t.get_pose().p[1]) < 0.05:
                self._hide_tile(i)
        self._belt_dwell(max(self.save_freq or 15, 15) * 3)

        if dbg:
            print(f"[qc] done colors={self.tile_colors} "
                  f"marked={self.tile_marked} correct={self.tile_correct} "
                  f"skipped={self.tile_skipped} missed={self.tile_missed} "
                  f"black_press={self.black_press} plan={self.plan_success}", flush=True)

        n_black = sum(1 for c in self.tile_colors if c == "black")
        self.info["info"] = {
            "{A}": f"colored tiles ({n_black} black)",
            "{a}": "both arms",
            "{o}": self._option_label(),
        }
        return self.info

    # ------------------------------------------------------------- success
    def check_success(self):
        if self.black_press:
            return False
        if any(self.tile_missed):
            return False
        for i, color in enumerate(self.tile_colors):
            if color == "black":
                # must be skipped (no press), not marked by a bad stamp
                if self.tile_marked[i] or not self.tile_skipped[i]:
                    return False
            else:
                if not self.tile_marked[i] or not self.tile_correct[i]:
                    return False
        return True

    def get_obs(self):
        obs = super().get_obs()
        obs["quality_control"] = {
            "n_tiles": int(self.n_tiles),
            "n_black": int(sum(1 for c in self.tile_colors if c == "black")),
            "n_marked": int(sum(1 for m in self.tile_marked if m)),
            "n_correct": int(sum(
                1 for i, c in enumerate(self.tile_correct)
                if c and self.tile_colors[i] != "black"
            )),
            "n_skipped": int(sum(1 for s in self.tile_skipped if s)),
            "n_missed": int(sum(1 for m in self.tile_missed if m)),
            "belt_speed": float(self.belt_speed),
            "color_mode": str(self.color_mode),
            "spacing_mode": str(self.spacing_mode),
            "black_frac_max": float(self.black_frac_max),
            "tile_pause_steps": int(self.tile_pause_steps),
            "tile_pause_s": float(getattr(
                self, "tile_pause_s",
                self.tile_pause_steps * self.scene.get_timestep(),
            )),
            "option_label": self._option_label(),
            "tile_gaps": [float(g) for g in self.tile_gaps],
            "tile_colors": list(self.tile_colors),
            "tile_marked": [bool(m) for m in self.tile_marked],
            "tile_correct": [bool(c) for c in self.tile_correct],
            "tile_skipped": [bool(s) for s in self.tile_skipped],
            "tile_missed": [bool(m) for m in self.tile_missed],
            "tile_hidden": [bool(h) for h in self.tile_hidden],
            "tile_positions": [float(t.get_pose().p[1]) for t in self.tiles],
            "black_press": bool(self.black_press),
            "black_press_count": int(self.black_press_count),
            "stamp_z": float(self.stamp.get_pose().p[2]),
            "stamp_key_color": None if self.stamp_key_color is None else str(self.stamp_key_color),
            "tile_under_stamp": (
                self._tile_under_stamp(require_unhandled=False)
                if getattr(self, "_stamp_ready", False) else None
            ),
        }
        return obs
