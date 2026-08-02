"""Measure olive oil into a marked jar, then weigh it on a kitchen scale.

KitchenS prep-counter scene (no sink / tap / stove): silver oil dispenser, marked
glass jar, electronic scale, and baking props (bread on a cutting board, flour
sack, chocolate chips, bowl of eggs). Click the red nozzle switch to start the
pour; oil flows while the switch stays on. Click again to stop at the target
ring, then place the jar on ``072_electronicscale``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import sapien
import sapien.render
import transforms3d as t3d

from ._kitchens_base_task import KitchenS_base_task
from ._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
from .utils import *


class measure_ingredient(KitchenS_base_task):
    """Fill the jar to a target ring, then place it on the electronic scale."""

    EGG_ORANGE = [0.95, 0.48, 0.10, 1.0]
    YUP_Q = [0.5, 0.5, 0.5, 0.5]
    BOARD_QPOS = [0.707, 0.707, 0.0, 0.0]
    # +30% vs the original 0.07 baking-board scale.
    BOARD_SCALE_DEFAULT = 0.091
    MICROWAVE_SCALE_DEFAULT = 1.3

    JAR_MODEL = "253_glass_jar"

    # Geometry (matches fill_coffee_jar jar; dispenser is a silver oil can).
    JAR_INNER_R = 0.035
    JAR_HEIGHT = 0.125
    JAR_BOTTOM_T = 0.005

    BODY_R = 0.040
    BODY_H = 0.155
    DOME_R = 0.040
    PEDESTAL_HALF = (0.052, 0.052, 0.045)
    PLATFORM_HALF = (0.055, 0.055, 0.007)

    NOZZLE_R = 0.006
    # Tiny red push-switch on the nozzle arm (click = toggle on/off).
    SWITCH_BASE_HALF = (0.007, 0.007, 0.0035)   # dark housing
    SWITCH_BTN_HALF = (0.0045, 0.0045, 0.0025)  # red button cap
    SWITCH_BTN_UP = 0.0045     # button stick-out when OFF
    SWITCH_BTN_DOWN = 0.0010   # depressed when ON
    SWITCH_RED = [0.90, 0.08, 0.08]

    EE_TO_TCP = 0.12
    SWITCH_HOVER_Z = 0.07
    SWITCH_PRESS_Z = 0.008     # EE above button top when pressing

    # Ring marks at 25/50/75%; 100% = jar rim (no ring).
    FILL_LEVELS = (0.25, 0.50, 0.75, 1.0)
    FILL_TOL = 0.05             # ±5% absolute fill tolerance
    # Slow enough that opening the switch does not already overshoot the mark;
    # oil still rises continuously while the switch stays on.
    POUR_RATE = 0.00022         # fill fraction per physics step while switch on
    # Relative half-width for sampling pour speed: rate ∼ Uniform(1±jitter)*pour_rate.
    POUR_RATE_JITTER = 0.15
    # Jar is full at 1.0; further pour (switch still ON) spills onto the table.
    OVERFLOW_LEVEL = 1.0
    # Layout randomization defaults (see task_args.measure_ingredient).
    MICROWAVE_Y_DEFAULT = 0.18  # stay at the back; only X diversifies
    # Keep MW out of the left-arm pour lane (see _sample_microwave_override).
    MICROWAVE_X_RANGE = (-0.42, -0.22)
    STATION_X_RANGE = (-0.20, -0.02)   # left-arm pour station
    SCALE_NEAR_DX_RANGE = (-0.28, -0.16)  # scale toward −x of jar (same arm)
    SCALE_NEAR_DY_RANGE = (-0.12, -0.02)  # slightly toward robot
    DECOR_ON_MICROWAVE_PROB = 0.35
    # Axis-aligned footprint half-sizes (m) + gap used for non-overlap layout.
    LAYOUT_MARGIN = 0.035
    DISP_HALF_XY = (0.058, 0.058)       # oil-can pedestal / body
    JAR_HALF_XY = (0.042, 0.042)
    SCALE_HALF_XY = (0.095, 0.075)
    DECOR_HALF_XY = {
        "bread": (0.12, 0.085),          # cutting board + bread
        "flour": (0.075, 0.075),
        "chips": (0.075, 0.055),
        "bowl": (0.085, 0.085),
    }
    # Spill puddle grows while switch stays on after the jar is full.
    SPILL_RATE = 0.00045        # spill_amount per physics step while overflowing
    SPILL_RADIUS_MIN = 0.035    # m; first visible puddle under the jar
    SPILL_RADIUS_MAX = 0.095    # m; fully spilled puddle
    SPILL_HALF_H = 0.0018       # m; flat disk thickness on the table

    # Oil look (``oil_style`` task_arg):
    #   solid       — opaque sunflower-yellow oil (default)
    #   transparent — see-through amber (glass-jar recipe)
    OIL_STYLE_DEFAULT = "solid"
    OIL_COLOR_TRANSPARENT = [0.90, 0.92, 0.62, 0.16]
    OIL_STREAM_TRANSPARENT = [0.88, 0.90, 0.55, 0.14]
    OIL_MENISCUS_TRANSPARENT = [0.88, 0.90, 0.50, 0.20]
    # Opaque sunflower-oil yellow.
    OIL_COLOR_SOLID = [0.96, 0.78, 0.12, 0.95]
    OIL_STREAM_SOLID = [0.94, 0.74, 0.08, 0.92]
    # Same yellow family for the table spill (slightly more opaque).
    OIL_SPILL_SOLID = [0.96, 0.78, 0.12, 0.98]
    OIL_SPILL_TRANSPARENT = [0.90, 0.92, 0.62, 0.55]
    UPRIGHT_CYL_Q = [0.70710678, 0.0, -0.70710678, 0.0]
    SILVER = [0.78, 0.80, 0.84, 1.0]
    SILVER_DARK = [0.55, 0.57, 0.60, 1.0]
    RING_RED = [0.78, 0.05, 0.05]
    GLASS = [0.88, 0.95, 0.98, 0.14]
    # Interactive viewer look (matches trap_bug plain trap): no transmission/IOR.
    PLAIN_GLASS = [0.18, 0.32, 0.48, 0.55]
    VERTICAL_CYL_Q = [0.70710678, 0.0, 0.70710678, 0.0]

    def setup_demo(self, **kwags):
        self._cfg = dict(kwags.get("task_args", {}).get("measure_ingredient", {}))
        if kwags.get("scene_id") is None:
            kwags["scene_id"] = int(self._cfg.get("scene_id", 0))
        # Bare baking counter: remove sink, faucet tap, and stove.
        self.clear_sink_and_range = True
        self.replace_sink_with_range = False
        self.microwave_scale_mult = float(
            self._cfg.get("microwave_scale_mult", self.MICROWAVE_SCALE_DEFAULT)
        )
        # Microwave pose must be chosen before super() builds the kitchen.
        self._layout_seed = int(kwags.get("seed", 0) or 0)
        self._sample_microwave_override(self._cfg, self._layout_seed)

        # Per-step state before early _update_kinematic_tasks (camera init).
        self._loaded = False
        self.tab_open = False
        self.liquid_level = 0.0
        self.overflowed = False
        self.spill_amount = 0.0
        self.opened_once = False
        self.closed_after_pour = False
        self.jar_on_scale = False
        self.target_fill = 0.25
        self._decor_layout = {}
        self._apply_oil_style(self._parse_oil_style(self._cfg))
        self._plain_glass = bool(self._cfg.get("plain_glass", False))
        self._liquid_entity = None
        self._stream_entity = None
        self._spill_entity = None
        self._spill_radius_cached = -1.0
        self._spill_xy = None
        self._liquid_half_h_cached = -1.0
        self._switch_parts = []
        self._switch_btn = None
        self._ring_entities = []
        self._touch_latched = False
        self._ignore_tab = False
        self._jar_locked = True
        self._jar_carry = False
        self._jar_carry_offset = None
        self._jar_seated_pose = None
        self.jar = None
        self.jar_visual = None
        self.scale = None
        self.table_top = 0.74

        super().setup_demo(**kwags)
        self._configure_observer_camera()

    def _layout_rng(self, salt: int = 0) -> np.random.RandomState:
        return np.random.RandomState(int(getattr(self, "_layout_seed", 0)) + int(salt))

    def _parse_range(self, cfg, key, default):
        raw = cfg.get(key, default)
        if raw is None:
            raw = default
        return (float(raw[0]), float(raw[1]))

    def _estimate_microwave_half_xy(self, scale_mult=None):
        """World-XY half extents of the yawed microwave collision box."""
        if scale_mult is None:
            scale_mult = float(
                getattr(self, "microwave_scale_mult", self.MICROWAVE_SCALE_DEFAULT)
            )
        scale = 0.15 * float(scale_mult)
        # Mesh bounds before yaw; after +90° Z, (x,y) → (−y, x).
        bounds_min = np.array([-0.761475, -0.495360, -0.567261], dtype=float)
        bounds_max = np.array([0.795292, 0.435315, 0.630773], dtype=float)
        half_x = 0.5 * (bounds_max[1] - bounds_min[1]) * scale
        half_y = 0.5 * (bounds_max[0] - bounds_min[0]) * scale
        return np.array([float(half_x), float(half_y)], dtype=float)

    @staticmethod
    def _aabb_overlap(c1, h1, c2, h2, margin=0.0):
        """True if two axis-aligned footprints overlap (with optional gap)."""
        c1 = np.asarray(c1, dtype=float)
        c2 = np.asarray(c2, dtype=float)
        h1 = np.asarray(h1, dtype=float)
        h2 = np.asarray(h2, dtype=float)
        m = float(margin)
        return bool(
            abs(c1[0] - c2[0]) < (h1[0] + h2[0] + m)
            and abs(c1[1] - c2[1]) < (h1[1] + h2[1] + m)
        )

    def _footprint_clear(self, center, half, blockers, margin=None):
        """Return True if ``center/half`` does not overlap any blocker AABB."""
        if margin is None:
            margin = self.LAYOUT_MARGIN
        for b_c, b_h in blockers:
            if self._aabb_overlap(center, half, b_c, b_h, margin=margin):
                return False
        return True

    def _microwave_pose_for_layout(self, cfg):
        """Return (mw_xy, mw_half) used for collision-aware placement."""
        mw_xy = getattr(self, "microwave_xy", None)
        if mw_xy is None:
            mw_xy = getattr(self, "microwave_xy_override", None)
        if mw_xy is None:
            mw_xy = [-0.32, float(cfg.get("microwave_y", self.MICROWAVE_Y_DEFAULT))]
        mw_xy = np.asarray(mw_xy, dtype=float)
        mw_half = getattr(self, "microwave_half_xy", None)
        if mw_half is None:
            mw_half = self._estimate_microwave_half_xy()
        mw_half = np.asarray(mw_half, dtype=float)
        return mw_xy, mw_half

    def _station_clears_microwave(self, side_x, disp_y, jar_y, mw_xy, mw_half, margin=None):
        if margin is None:
            margin = self.LAYOUT_MARGIN
        disp = np.array([side_x, disp_y], dtype=float)
        jar = np.array([side_x, jar_y], dtype=float)
        if self._aabb_overlap(disp, self.DISP_HALF_XY, mw_xy, mw_half, margin):
            return False
        if self._aabb_overlap(jar, self.JAR_HALF_XY, mw_xy, mw_half, margin):
            return False
        return True

    def _nudge_x_clear_of_microwave(self, side_x, disp_y, jar_y, mw_xy, mw_half, x_lo, x_hi):
        """Push station X away from the microwave until footprints clear."""
        need = (
            float(mw_half[0])
            + max(float(self.DISP_HALF_XY[0]), float(self.JAR_HALF_XY[0]))
            + float(self.LAYOUT_MARGIN)
            + 1e-3
        )
        # Prefer the side that still lies inside [x_lo, x_hi].
        left_x = float(mw_xy[0]) - need
        right_x = float(mw_xy[0]) + need
        candidates = []
        if x_lo <= left_x <= x_hi:
            candidates.append(left_x)
        if x_lo <= right_x <= x_hi:
            candidates.append(right_x)
        # Also try clamping the preferred direction from the current sample.
        if side_x <= mw_xy[0]:
            candidates.append(float(np.clip(left_x, x_lo, x_hi)))
        else:
            candidates.append(float(np.clip(right_x, x_lo, x_hi)))
        for x in candidates:
            if self._station_clears_microwave(x, disp_y, jar_y, mw_xy, mw_half):
                return float(x)
        # Last resort: pick the in-range X maximizing |x - mw_x|.
        grid = np.linspace(x_lo, x_hi, 25)
        best = float(side_x)
        best_d = -1.0
        for x in grid:
            if not self._station_clears_microwave(x, disp_y, jar_y, mw_xy, mw_half):
                continue
            d = abs(float(x) - float(mw_xy[0]))
            if d > best_d:
                best, best_d = float(x), d
        return best

    def _sample_microwave_override(self, cfg, seed: int):
        """Microwave stays at the back (fixed Y); only X may diversify.

        Reject X samples that leave no collision-free pour-station placement.
        """
        mw_y = float(cfg.get("microwave_y", self.MICROWAVE_Y_DEFAULT))
        randomize = bool(cfg.get("randomize_layout", False))
        if "microwave_x" in cfg and cfg.get("microwave_x") is not None:
            self.microwave_xy_override = [float(cfg["microwave_x"]), mw_y]
            return
        if not randomize:
            self.microwave_xy_override = None
            return

        lo, hi = self._parse_range(cfg, "microwave_x_range", self.MICROWAVE_X_RANGE)
        sx_lo, sx_hi = self._parse_range(cfg, "station_x_range", self.STATION_X_RANGE)
        disp_y = float(cfg.get("disp_y", 0.08))
        jar_y = float(cfg.get("jar_y", -0.06))
        mw_half = self._estimate_microwave_half_xy(
            cfg.get("microwave_scale_mult", self.MICROWAVE_SCALE_DEFAULT)
        )
        rng = np.random.RandomState(int(seed) + 17)
        mw_x = None
        for _ in range(80):
            cand = float(rng.uniform(lo, hi))
            mw_xy = np.array([cand, mw_y], dtype=float)
            # Must leave at least one valid station X in range.
            ok = False
            for x in np.linspace(sx_lo, sx_hi, 21):
                if self._station_clears_microwave(
                    float(x), disp_y, jar_y, mw_xy, mw_half
                ):
                    ok = True
                    break
            if ok:
                mw_x = cand
                break
        if mw_x is None:
            # Park MW on the far-left so the left-arm station stays clear.
            need = (
                float(mw_half[0])
                + max(float(self.DISP_HALF_XY[0]), float(self.JAR_HALF_XY[0]))
                + float(self.LAYOUT_MARGIN)
            )
            mw_x = float(np.clip(sx_lo - need - 0.02, lo, hi))
        self.microwave_xy_override = [mw_x, mw_y]

    def _resolve_target_fill(self, cfg, rng: np.random.RandomState) -> float:
        """Parse target_fill: 0.25/0.5/0.75/1.0 or ``random``."""
        tf = cfg.get("target_fill", 0.25)
        if isinstance(tf, str) and tf.strip().lower() == "random":
            levels = cfg.get("fill_levels", self.FILL_LEVELS)
            choices = [float(x) for x in levels]
            for c in choices:
                if c not in self.FILL_LEVELS:
                    raise ValueError(
                        f"fill_levels entry {c} not in {self.FILL_LEVELS}"
                    )
            return float(rng.choice(choices))
        level = 0.25 if tf is None else float(tf)
        if level not in self.FILL_LEVELS:
            raise ValueError(f"target_fill must be one of {self.FILL_LEVELS} or 'random'")
        return level

    def _sample_free_xy_aabb(
        self, rng, half, blockers, x_range, y_range, tries=60, margin=None
    ):
        """Sample a table XY whose footprint clears all blocker AABBs."""
        if margin is None:
            margin = self.LAYOUT_MARGIN
        x_lo, x_hi = float(x_range[0]), float(x_range[1])
        y_lo, y_hi = float(y_range[0]), float(y_range[1])
        half = np.asarray(half, dtype=float)
        for _ in range(int(tries)):
            p = np.array(
                [rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)], dtype=float
            )
            if self._footprint_clear(p, half, blockers, margin=margin):
                return p
        # Prefer the open right / front counter if the full box is crowded.
        safe_boxes = [
            (max(x_lo, 0.12), x_hi, y_lo, min(y_hi, 0.06)),
            (x_lo, x_hi, y_lo, y_hi),
        ]
        best, best_score = None, -1e9
        for bx0, bx1, by0, by1 in safe_boxes:
            if bx1 <= bx0 or by1 <= by0:
                continue
            for _ in range(80):
                p = np.array(
                    [rng.uniform(bx0, bx1), rng.uniform(by0, by1)], dtype=float
                )
                if self._footprint_clear(p, half, blockers, margin=margin):
                    return p
                score = min(
                    min(
                        abs(p[0] - b_c[0]) - (half[0] + b_h[0] + margin),
                        abs(p[1] - b_c[1]) - (half[1] + b_h[1] + margin),
                    )
                    for b_c, b_h in blockers
                ) if blockers else 1.0
                if score > best_score:
                    best, best_score = p, score
        if best is not None and best_score >= 0.0:
            return best
        # Last resort: park on the far-right front corner.
        return np.array([min(x_hi, 0.42), max(y_lo, -0.22)], dtype=float)

    def _resolve_layout(self, cfg):
        """Station (jar under dispenser), nearby scale, sparse/on-microwave decor.

        With ``randomize_layout: true``, fixed ``*_xy`` / ``station_x`` / ``scale_*``
        in yaml are ignored in favor of the ``*_range`` parameters (unless a pin
        key is set, e.g. ``pin_station_x``). All table props use AABB clearance
        so the dispenser / jar / scale never sit inside the microwave.
        """
        rng = self._layout_rng(101)
        randomize = bool(cfg.get("randomize_layout", False))
        mw_xy, mw_half = self._microwave_pose_for_layout(cfg)
        mw_top = getattr(self, "microwave_top_z", None)
        if mw_top is None:
            mw_top = self.table_top + 0.22
        mw_top = float(mw_top)
        table_z = float(self.table_top + 0.001)

        disp_y = float(cfg.get("disp_y", 0.08))
        jar_y = float(cfg.get("jar_y", -0.06))
        if randomize and bool(cfg.get("randomize_station_y", False)):
            dlo, dhi = self._parse_range(cfg, "disp_y_range", (0.04, 0.12))
            jlo, jhi = self._parse_range(cfg, "jar_y_range", (-0.10, -0.02))
            gap = disp_y - jar_y
            disp_y = float(rng.uniform(dlo, dhi))
            jar_y = float(np.clip(disp_y - gap, jlo, jhi))

        sx_lo, sx_hi = self._parse_range(cfg, "station_x_range", self.STATION_X_RANGE)
        if randomize:
            if cfg.get("pin_station_x") is not None:
                side_x = float(cfg["pin_station_x"])
            else:
                side_x = float(rng.uniform(sx_lo, sx_hi))
                if not self._station_clears_microwave(
                    side_x, disp_y, jar_y, mw_xy, mw_half
                ):
                    # Resample, then nudge if still blocked.
                    found = None
                    for _ in range(40):
                        cand = float(rng.uniform(sx_lo, sx_hi))
                        if self._station_clears_microwave(
                            cand, disp_y, jar_y, mw_xy, mw_half
                        ):
                            found = cand
                            break
                    side_x = (
                        found
                        if found is not None
                        else self._nudge_x_clear_of_microwave(
                            side_x, disp_y, jar_y, mw_xy, mw_half, sx_lo, sx_hi
                        )
                    )
            if not self._station_clears_microwave(
                side_x, disp_y, jar_y, mw_xy, mw_half
            ):
                side_x = self._nudge_x_clear_of_microwave(
                    side_x, disp_y, jar_y, mw_xy, mw_half, sx_lo, sx_hi
                )
        else:
            side_x = float(cfg.get("station_x", -0.08))
            if not self._station_clears_microwave(
                side_x, disp_y, jar_y, mw_xy, mw_half
            ):
                # Fixed configs that still collide (e.g. large MW) get nudged.
                side_x = self._nudge_x_clear_of_microwave(
                    side_x, disp_y, jar_y, mw_xy, mw_half, -0.45, 0.20
                )

        self.dispenser_xy = np.array([side_x, disp_y], dtype=float)
        self.jar_xy = np.array([side_x, jar_y], dtype=float)
        self.arm = ArmTag("left" if side_x <= 0 else "right")

        blockers = [
            (mw_xy.copy(), mw_half.copy()),
            (self.dispenser_xy.copy(), np.asarray(self.DISP_HALF_XY, dtype=float)),
            (self.jar_xy.copy(), np.asarray(self.JAR_HALF_XY, dtype=float)),
        ]

        # Scale stays near the jar so the pour arm can place it — never in MW.
        if randomize:
            if cfg.get("pin_scale_x") is not None and cfg.get("pin_scale_y") is not None:
                scale_xy = np.array(
                    [float(cfg["pin_scale_x"]), float(cfg["pin_scale_y"])],
                    dtype=float,
                )
            else:
                dx_lo, dx_hi = self._parse_range(
                    cfg, "scale_near_dx_range", self.SCALE_NEAR_DX_RANGE
                )
                dy_lo, dy_hi = self._parse_range(
                    cfg, "scale_near_dy_range", self.SCALE_NEAR_DY_RANGE
                )
                scale_xy = None
                for _ in range(50):
                    dx = float(rng.uniform(dx_lo, dx_hi))
                    dy = float(rng.uniform(dy_lo, dy_hi))
                    if side_x <= 0:
                        cand = np.array([side_x + dx, jar_y + dy], dtype=float)
                    else:
                        cand = np.array([side_x - dx, jar_y + dy], dtype=float)
                    if self._footprint_clear(cand, self.SCALE_HALF_XY, blockers):
                        scale_xy = cand
                        break
                if scale_xy is None:
                    # Fallback: beside the jar, away from the microwave in X.
                    away = -1.0 if side_x <= float(mw_xy[0]) else 1.0
                    scale_xy = np.array(
                        [side_x + away * 0.22, jar_y - 0.06], dtype=float
                    )
                    if not self._footprint_clear(scale_xy, self.SCALE_HALF_XY, blockers):
                        scale_xy = np.array(
                            [side_x + away * 0.28, jar_y - 0.08], dtype=float
                        )
        else:
            scale_xy = np.array(
                [
                    float(
                        cfg.get(
                            "scale_x",
                            side_x - 0.26 if side_x <= 0 else side_x + 0.26,
                        )
                    ),
                    float(cfg.get("scale_y", -0.14)),
                ],
                dtype=float,
            )
            if not self._footprint_clear(scale_xy, self.SCALE_HALF_XY, blockers):
                away = -1.0 if side_x <= float(mw_xy[0]) else 1.0
                scale_xy = np.array([side_x + away * 0.24, jar_y - 0.06], dtype=float)
        self.scale_xy = scale_xy
        blockers.append(
            (self.scale_xy.copy(), np.asarray(self.SCALE_HALF_XY, dtype=float))
        )

        # Baking decor: free counter (AABB-clear), or on top of the microwave.
        defaults = {
            "bread": cfg.get("bread_xy", [0.28, -0.20]),
            "flour": cfg.get("flour_xy", [0.38, 0.14]),
            "chips": cfg.get("chips_xy", [0.12, -0.22]),
            "bowl": cfg.get("bowl_xy", [0.40, -0.02]),
        }
        on_mw_prob = float(
            cfg.get("decor_on_microwave_prob", self.DECOR_ON_MICROWAVE_PROB)
        )
        decor_x_range = self._parse_range(cfg, "decor_x_range", (-0.45, 0.48))
        decor_y_range = self._parse_range(cfg, "decor_y_range", (-0.26, 0.20))
        # Keep table decor off the microwave footprint and out of the pour lane.
        mw_blockers = list(blockers)
        on_mw_blockers = []  # footprints already placed on the microwave top

        self._decor_layout = {}
        for name, default_xy in defaults.items():
            half = self.DECOR_HALF_XY.get(name, (0.08, 0.08))
            on_mw = False
            if randomize:
                want_mw = bool(rng.rand() < on_mw_prob)
                # Skip MW top if the prop footprint cannot fit with margin.
                if want_mw and (
                    float(half[0]) > 0.70 * float(mw_half[0])
                    or float(half[1]) > 0.70 * float(mw_half[1])
                ):
                    want_mw = False
                xy = None
                if want_mw:
                    # Place on MW top with mutual clearance between decor items.
                    inset = 0.40
                    for _ in range(50):
                        cand = np.array(
                            [
                                mw_xy[0]
                                + rng.uniform(-inset * mw_half[0], inset * mw_half[0]),
                                mw_xy[1]
                                + rng.uniform(-inset * mw_half[1], inset * mw_half[1]),
                            ],
                            dtype=float,
                        )
                        if self._footprint_clear(
                            cand, half, on_mw_blockers, margin=0.02
                        ):
                            xy = cand
                            on_mw = True
                            break
                if xy is None:
                    on_mw = False
                    xy = self._sample_free_xy_aabb(
                        rng, half, mw_blockers, decor_x_range, decor_y_range
                    )
                z = (mw_top + 0.002) if on_mw else table_z
            else:
                xy = np.array([float(default_xy[0]), float(default_xy[1])], dtype=float)
                z = table_z
                # Fixed decor that would land in the MW / station gets nudged away.
                if not self._footprint_clear(xy, half, mw_blockers):
                    xy = self._sample_free_xy_aabb(
                        rng, half, mw_blockers, decor_x_range, decor_y_range
                    )
            self._decor_layout[name] = {
                "xy": xy,
                "z": float(z),
                "on_microwave": bool(on_mw),
            }
            if on_mw:
                on_mw_blockers.append((xy.copy(), np.asarray(half, dtype=float)))
            else:
                mw_blockers.append((xy.copy(), np.asarray(half, dtype=float)))

        # Hard assert for the critical failure mode reported by the user.
        if not self._station_clears_microwave(
            float(self.dispenser_xy[0]),
            float(self.dispenser_xy[1]),
            float(self.jar_xy[1]),
            mw_xy,
            mw_half,
            margin=0.0,
        ):
            print(
                f"[measure_ingredient] WARNING: station still overlaps microwave "
                f"disp={self.dispenser_xy.tolist()} mw={mw_xy.tolist()} "
                f"mw_half={mw_half.tolist()} — forcing X nudge"
            )
            side_x = self._nudge_x_clear_of_microwave(
                float(self.dispenser_xy[0]),
                float(self.dispenser_xy[1]),
                float(self.jar_xy[1]),
                mw_xy,
                mw_half,
                -0.45,
                0.25,
            )
            self.dispenser_xy[0] = side_x
            self.jar_xy[0] = side_x
            self.arm = ArmTag("left" if side_x <= 0 else "right")

        self.target_fill = self._resolve_target_fill(cfg, rng)

    def _configure_observer_camera(self):
        cams = getattr(self, "cameras", None)
        if cams is None or getattr(cams, "observer_camera", None) is None:
            return
        camera = cams.observer_camera
        camera_pos = np.array([0.05, -0.55, 1.40], dtype=np.float64)
        look_at = np.array([-0.05, -0.05, 0.92], dtype=np.float64)
        forward = look_at - camera_pos
        forward /= np.linalg.norm(forward)
        left = np.cross(np.array([0.0, 0.0, 1.0]), forward)
        left /= np.linalg.norm(left)
        up = np.cross(forward, left)
        m = np.eye(4)
        m[:3, :3] = np.stack([forward, left, up], axis=1)
        m[:3, 3] = camera_pos
        camera.entity.set_pose(sapien.Pose(m))

    # ------------------------------------------------------------------ materials
    def _metallic_material(self, rgb, roughness=0.22, metallic=0.95):
        rgba = list(rgb[:3]) + [1.0]
        mat = sapien.render.RenderMaterial(base_color=rgba)
        try:
            mat.set_roughness(float(roughness))
            mat.set_metallic(float(metallic))
        except Exception:
            mat.roughness = float(roughness)
            mat.metallic = float(metallic)
        return mat

    def _plain_glass_material(self):
        """Simple alpha-transparent plastic — no glass transmission/IOR (viewer-friendly)."""
        mat = sapien.render.RenderMaterial(base_color=list(self.PLAIN_GLASS))
        try:
            mat.set_transmission(0.0)
            mat.set_transmission_roughness(1.0)
            mat.set_roughness(0.55)
            mat.set_metallic(0.0)
        except Exception:
            mat.roughness = 0.55
            mat.metallic = 0.0
        try:
            mat.set_ior(1.0)
        except Exception:
            pass
        return mat

    def _glass_material(self, rgba=None, transmission=0.90):
        if bool(getattr(self, "_plain_glass", False)):
            return self._plain_glass_material()
        c = list(rgba if rgba is not None else self.GLASS)
        if len(c) == 3:
            c = c + [0.18]
        mat = sapien.render.RenderMaterial(base_color=c)
        try:
            mat.set_transmission(float(transmission))
            mat.set_transmission_roughness(0.02)
            mat.set_roughness(0.05)
            mat.set_metallic(0.0)
        except Exception:
            mat.transmission = float(transmission)
            mat.roughness = 0.05
            mat.metallic = 0.0
        try:
            mat.set_ior(1.45)
        except Exception:
            pass
        return mat

    def _opaque_material(self, rgb, alpha=1.0):
        rgba = list(rgb[:3]) + [float(alpha)]
        mat = sapien.render.RenderMaterial(base_color=rgba)
        try:
            mat.set_roughness(0.45)
            mat.set_metallic(0.0)
        except Exception:
            mat.roughness = 0.45
            mat.metallic = 0.0
        return mat

    def _parse_oil_style(self, cfg) -> str:
        """Return ``transparent`` or ``solid`` from task_args."""
        raw = cfg.get("oil_style", None)
        if raw is None and "oil_transparent" in cfg:
            raw = "transparent" if bool(cfg.get("oil_transparent")) else "solid"
        style = str(raw if raw is not None else self.OIL_STYLE_DEFAULT).strip().lower()
        aliases = {
            "transparent": "transparent",
            "clear": "transparent",
            "see_through": "transparent",
            "see-through": "transparent",
            "glass": "transparent",
            "solid": "solid",
            "opaque": "solid",
            "previous": "solid",
            "dark": "solid",
            "green": "solid",
        }
        if style not in aliases:
            raise ValueError(
                f"oil_style must be 'transparent' or 'solid' (got {raw!r})"
            )
        return aliases[style]

    def _apply_oil_style(self, style: str):
        """Set active oil colors / material mode from ``oil_style``."""
        self.oil_style = style
        self.oil_transparent = style == "transparent"
        if self.oil_transparent:
            self.oil_color = list(self.OIL_COLOR_TRANSPARENT)
            self.oil_stream_color = list(self.OIL_STREAM_TRANSPARENT)
            self.oil_meniscus = list(self.OIL_MENISCUS_TRANSPARENT)
            self.oil_spill_color = list(self.OIL_SPILL_TRANSPARENT)
        else:
            self.oil_color = list(self.OIL_COLOR_SOLID)
            self.oil_stream_color = list(self.OIL_STREAM_SOLID)
            self.oil_meniscus = None
            self.oil_spill_color = list(self.OIL_SPILL_SOLID)

    def _fluid_material(self, rgba, transmission=None):
        """Oil material: glass-clear when transparent, beer-style when solid."""
        mat = sapien.render.RenderMaterial(base_color=list(rgba))
        if getattr(self, "oil_transparent", True):
            t = 1.0 if transmission is None else float(transmission)
            try:
                mat.set_transmission(t)
                mat.set_transmission_roughness(0.0)
                mat.set_roughness(0.04)
                mat.set_metallic(0.0)
            except Exception:
                mat.roughness = 0.04
                mat.metallic = 0.0
            try:
                mat.set_ior(1.0)
            except Exception:
                pass
        else:
            # Previous dark-green look (pour_beer fluid recipe).
            try:
                mat.set_roughness(0.18)
                mat.set_metallic(0.0)
            except Exception:
                mat.roughness = 0.18
                mat.metallic = 0.0
        return mat

    def _make_oil_column(self, radius, half_h, world_xyz, rgba, name, local_z=0.0):
        """Visual-only oil cylinder via RenderShapeCylinder."""
        ent = sapien.Entity()
        ent.set_name(name)
        ent.set_pose(
            sapien.Pose([float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2])])
        )
        body = sapien.render.RenderBodyComponent()
        mat = self._fluid_material(rgba)
        col = sapien.render.RenderShapeCylinder(
            radius=float(radius),
            half_length=max(0.002, float(half_h)),
            material=mat,
        )
        col.set_local_pose(sapien.Pose([0.0, 0.0, float(local_z)], self.UPRIGHT_CYL_Q))
        body.attach(col)
        ent.add_component(body)
        self.scene.add_entity(ent)
        return ent

    def _add_static_box(self, pose, half_size, material=None, color=None, name="", collision=True):
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        if collision:
            builder.add_box_collision(
                pose=sapien.Pose(),
                half_size=list(half_size),
                material=self.scene.default_physical_material,
            )
        if material is None:
            rgba = list(color if color is not None else [0.8, 0.8, 0.8, 1.0])
            if len(rgba) == 3:
                rgba = rgba + [1.0]
            material = self._opaque_material(rgba[:3], rgba[3])
        builder.add_box_visual(pose=sapien.Pose(), half_size=list(half_size), material=material)
        builder.set_initial_pose(pose)
        return builder.build(name=name)

    def _add_static_mesh_visual(self, filename, pose, material, name):
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        builder.add_visual_from_file(
            filename=str(Path(filename).resolve()),
            material=material,
        )
        builder.set_initial_pose(pose)
        return builder.build(name=name)

    def _remove_entity(self, ent):
        if ent is None:
            return None
        try:
            self.scene.remove_entity(ent)
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------ actors
    def load_actors(self):
        cfg = self._cfg
        self.table_top = float(self.kitchens_info["table_height"]) + float(self.table_z_bias)

        base_pour = float(cfg.get("pour_rate", self.POUR_RATE))
        self.pour_rate_base = base_pour
        # pour_rate_jitter: 0 → fixed; 0.15 → sample Uniform([0.85, 1.15]) * pour_rate.
        jitter = float(cfg.get("pour_rate_jitter", self.POUR_RATE_JITTER))
        self.pour_rate_jitter = max(0.0, jitter)
        if self.pour_rate_jitter > 0.0:
            scale = float(
                self._layout_rng(211).uniform(
                    1.0 - self.pour_rate_jitter, 1.0 + self.pour_rate_jitter
                )
            )
            self.pour_rate = max(1e-8, base_pour * scale)
        else:
            self.pour_rate = base_pour
        self.fill_tol = float(cfg.get("fill_tol", self.FILL_TOL))
        self.overflow_level = float(cfg.get("overflow_level", self.OVERFLOW_LEVEL))
        self.spill_rate = float(cfg.get("spill_rate", self.SPILL_RATE))
        self.force_overflow = bool(cfg.get("force_overflow", False))
        self._apply_oil_style(self._parse_oil_style(cfg))

        self.tab_open = False
        self.liquid_level = 0.0
        self.overflowed = False
        self.spill_amount = 0.0
        self.opened_once = False
        self.closed_after_pour = False
        self.jar_on_scale = False
        self._liquid_entity = None
        self._stream_entity = None
        self._spill_entity = self._remove_entity(getattr(self, "_spill_entity", None))
        self._spill_radius_cached = -1.0
        self._spill_xy = None
        self._liquid_half_h_cached = -1.0
        self._switch_parts = []
        self._switch_btn = None
        self._ring_entities = []
        self._touch_latched = False
        self._ignore_tab = False
        self._jar_locked = True
        self._jar_carry = False
        self._jar_carry_offset = None
        self._jar_seated_pose = None

        # Jar always under dispenser; scale nearby; decor free / on microwave.
        self._resolve_layout(cfg)

        self._build_dispenser()
        self._build_jar()
        self._build_fill_rings()
        self._build_scale()
        self._build_baking_props()
        self._rebuild_liquid(force=True)
        self._sync_stream()
        self._sync_jar_followers()

        self.add_prohibit_area(
            sapien.Pose([*self.dispenser_xy, self.table_top + 0.1]), padding=0.08
        )
        self.add_prohibit_area(
            sapien.Pose([*self.jar_xy, self.table_top + 0.05]), padding=0.05
        )
        if self.scale is not None:
            self.add_prohibit_area(self.scale, padding=0.04)

        self._loaded = True
        mw = getattr(self, "microwave_xy_override", None) or getattr(
            self, "microwave_xy", None
        )
        decor_on_mw = [
            n for n, d in (self._decor_layout or {}).items() if d.get("on_microwave")
        ]
        print(
            f"[measure_ingredient] KitchenS scene={self.scene_id} "
            f"target={self.target_fill:.0%}±{self.fill_tol:.0%} arm={self.arm} "
            f"pour_rate={self.pour_rate:.6g} "
            f"(base={self.pour_rate_base:.6g}±{self.pour_rate_jitter:.0%}) "
            f"station={self.jar_xy.tolist()} scale={self.scale_xy.tolist()} "
            f"mw={None if mw is None else [float(mw[0]), float(mw[1])]} "
            f"decor_on_mw={decor_on_mw} oil_style={self.oil_style}"
        )

    def _build_dispenser(self):
        """Silver metallic cylindrical oil can with dome + nozzle tab over the jar."""
        x, y = self.dispenser_xy
        z0 = self.table_top
        _, _, pedestal_hz = self.PEDESTAL_HALF
        _, _, plat_hz = self.PLATFORM_HALF
        silver = self._metallic_material(self.SILVER)
        silver_dark = self._metallic_material(self.SILVER_DARK, roughness=0.30, metallic=0.90)

        # Pedestal + platform raise the can so the nozzle clears the jar rim.
        self._add_static_box(
            pose=sapien.Pose([x, y, z0 + pedestal_hz]),
            half_size=self.PEDESTAL_HALF,
            material=silver_dark,
            name="oil_dispenser_pedestal",
        )
        self._add_static_box(
            pose=sapien.Pose([x, y, z0 + 2.0 * pedestal_hz + plat_hz]),
            half_size=self.PLATFORM_HALF,
            material=silver_dark,
            name="oil_dispenser_platform",
        )

        body_bottom = z0 + 2.0 * pedestal_hz + 2.0 * plat_hz
        body_half = self.BODY_H * 0.5
        body_z = body_bottom + body_half

        # Main cylindrical body (Sapien cylinder axis = local +X → rotate to +Z).
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        builder.add_cylinder_collision(
            pose=sapien.Pose([0, 0, 0], self.VERTICAL_CYL_Q),
            radius=self.BODY_R,
            half_length=body_half,
            material=self.scene.default_physical_material,
        )
        builder.add_cylinder_visual(
            pose=sapien.Pose([0, 0, 0], self.VERTICAL_CYL_Q),
            radius=self.BODY_R,
            half_length=body_half,
            material=silver,
        )
        builder.set_initial_pose(sapien.Pose([x, y, body_z]))
        builder.build(name="oil_dispenser_body")

        # Dome cap.
        dome_z = body_bottom + self.BODY_H + self.DOME_R * 0.35
        dome_b = self.scene.create_actor_builder()
        dome_b.set_physx_body_type("static")
        dome_b.add_sphere_collision(
            pose=sapien.Pose(),
            radius=self.DOME_R,
            material=self.scene.default_physical_material,
        )
        dome_b.add_sphere_visual(pose=sapien.Pose(), radius=self.DOME_R, material=silver)
        dome_b.set_initial_pose(sapien.Pose([x, y, dome_z]))
        dome_b.build(name="oil_dispenser_dome")

        # Front glass window + olive-oil reservoir fill (shows contents).
        win_y = y - self.BODY_R + 0.004
        reservoir_rgba = (
            list(self.oil_color)
            if getattr(self, "oil_transparent", True)
            else list(self.OIL_COLOR_SOLID)
        )
        oil_fill = self._fluid_material(reservoir_rgba)
        self._add_static_box(
            pose=sapien.Pose([x, win_y - 0.001, body_z - 0.01]),
            half_size=[self.BODY_R * 0.55, 0.002, body_half * 0.70],
            material=oil_fill,
            name="oil_reservoir_fill",
            collision=False,
        )
        self._add_static_box(
            pose=sapien.Pose([x, win_y - 0.004, body_z - 0.01]),
            half_size=[self.BODY_R * 0.58, 0.0015, body_half * 0.72],
            material=self._glass_material([0.90, 0.96, 0.99, 0.20]),
            name="oil_window",
            collision=False,
        )

        # Nozzle arm from hopper front toward the jar (ends short of jar center).
        jar_x, jar_y = self.jar_xy
        hopper_front_y = y - self.BODY_R
        tip_y = jar_y + 0.018
        nozzle_joint_z = self.table_top + self.JAR_HEIGHT + 0.070
        nozzle_outlet_z = self.table_top + self.JAR_HEIGHT + 0.035
        nozzle_y = 0.5 * (hopper_front_y + tip_y)

        # Visual-only nozzle (no collision) so the jar stays graspable from above.
        self._add_static_box(
            pose=sapien.Pose([x, nozzle_y, nozzle_joint_z]),
            half_size=[0.007, abs(tip_y - hopper_front_y) * 0.5, 0.006],
            material=silver_dark,
            name="oil_nozzle_arm",
            collision=False,
        )
        tip_half_z = 0.5 * (nozzle_joint_z - nozzle_outlet_z)
        self._add_static_box(
            pose=sapien.Pose(
                [jar_x, tip_y, 0.5 * (nozzle_joint_z + nozzle_outlet_z)]
            ),
            half_size=[0.007, 0.007, tip_half_z],
            material=silver_dark,
            name="oil_nozzle_tip",
            collision=False,
        )
        # Nozzle opening ring.
        self._add_static_box(
            pose=sapien.Pose([jar_x, tip_y, nozzle_outlet_z]),
            half_size=[self.NOZZLE_R, self.NOZZLE_R, 0.002],
            material=self._metallic_material([0.25, 0.25, 0.27], roughness=0.35),
            name="oil_nozzle_opening",
            collision=False,
        )

        self.nozzle_outlet_xyz = np.array([jar_x, tip_y, nozzle_outlet_z], dtype=float)
        # Tiny red switch on TOP of the nozzle arm near the tip.
        self.switch_base_xyz = np.array(
            [jar_x, tip_y + 0.012, nozzle_joint_z + 0.006], dtype=float
        )
        self._build_switch()

    def _switch_button_center(self, open_: bool | None = None):
        """World XYZ of the red button center (depressed when ON)."""
        if open_ is None:
            open_ = bool(getattr(self, "tab_open", False))
        bx, by, bz = self.switch_base_xyz
        base_top = bz + float(self.SWITCH_BASE_HALF[2])
        stick = self.SWITCH_BTN_DOWN if open_ else self.SWITCH_BTN_UP
        half_t = float(self.SWITCH_BTN_HALF[2])
        return np.array([bx, by, base_top + stick + half_t], dtype=float)

    def _switch_top_z(self, open_: bool | None = None):
        c = self._switch_button_center(open_)
        return float(c[2] + float(self.SWITCH_BTN_HALF[2]))

    def _sync_switch_touch_points(self):
        top = self._switch_button_center()
        top[2] = self._switch_top_z()
        self.tab_touch_xyz = top.copy()
        self.touch_xy = top[:2].copy()
        self.touch_top_z = float(top[2])

    def _clear_switch_parts(self):
        for part in list(getattr(self, "_switch_parts", []) or []):
            self._remove_entity(part)
        self._switch_parts = []
        self._switch_btn = None

    def _build_switch(self):
        """Dark housing + red push button; click toggles pour on/off."""
        self._clear_switch_parts()
        bx, by, bz = self.switch_base_xyz
        housing = self._metallic_material([0.18, 0.18, 0.20], roughness=0.45, metallic=0.55)
        base = self._add_static_box(
            pose=sapien.Pose([bx, by, bz + float(self.SWITCH_BASE_HALF[2])]),
            half_size=list(self.SWITCH_BASE_HALF),
            material=housing,
            name="oil_switch_base",
            collision=True,
        )
        self._switch_parts.append(base)
        self._rebuild_switch_button()
        self._sync_switch_touch_points()

    def _rebuild_switch_button(self):
        """Recreate the red button at the raised (OFF) or depressed (ON) height."""
        if getattr(self, "_switch_btn", None) is not None:
            self._remove_entity(self._switch_btn)
            if self._switch_btn in (self._switch_parts or []):
                self._switch_parts.remove(self._switch_btn)
            self._switch_btn = None

        center = self._switch_button_center()
        red = self._opaque_material(self.SWITCH_RED)
        btn = self._add_static_box(
            pose=sapien.Pose(center.tolist()),
            half_size=list(self.SWITCH_BTN_HALF),
            material=red,
            name="oil_switch_button",
            collision=True,
        )
        self._switch_btn = btn
        self._switch_parts.append(btn)
        self._sync_switch_touch_points()

    def _build_jar(self):
        """See-through coffee-task jar: dynamic convex collision + cylinder visual.

        Kept locked under the nozzle while pouring; unlocked for the final grasp.
        """
        x, y = self.jar_xy
        z0 = self.table_top + 0.001
        outer_r = self.JAR_INNER_R + 0.0035
        h = self.JAR_HEIGHT
        bottom_t = self.JAR_BOTTOM_T
        upright_q = [0.70710678, 0.0, -0.70710678, 0.0]

        md_path = Path("assets/objects/253_glass_jar/model_data0.json")
        with open(md_path, "r") as f:
            md = json.load(f)

        # Solid cylinder collision (hollow mesh is a poor grasp target).
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("dynamic")
        builder.add_cylinder_collision(
            pose=sapien.Pose([0.0, 0.0, h * 0.5], self.VERTICAL_CYL_Q),
            radius=float(outer_r),
            half_length=float(h * 0.5),
            material=self.scene.default_physical_material,
        )
        builder.set_initial_pose(sapien.Pose([x, y, z0]))
        entity = builder.build(name="glass_jar")
        try:
            entity.set_name("glass_jar")
        except Exception:
            pass

        self.jar = Actor(entity, md, mass=0.10)
        self._jar_home_pose = sapien.Pose([x, y, z0])
        self._jar_locked = True
        self._set_jar_damping(50.0)
        # Help the gripper keep the jar once closed.
        for component in self.jar.actor.get_components():
            if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
                try:
                    for shape in component.get_collision_shapes():
                        shape.set_physical_material(
                            self.scene.create_physical_material(1.2, 1.2, 0.01)
                        )
                except Exception:
                    pass

        if bool(getattr(self, "_plain_glass", False)):
            glass = self._plain_glass_material()
        else:
            glass = sapien.render.RenderMaterial(base_color=[0.93, 0.97, 1.0, 0.10])
            glass.set_transmission(1.0)
            glass.set_transmission_roughness(0.0)
            glass.set_roughness(0.04)
            glass.set_metallic(0.0)
            try:
                glass.set_ior(1.0)
            except Exception:
                pass

        wall_h = h - bottom_t
        wall_half = wall_h * 0.5
        wall_z = bottom_t + wall_half

        vis = sapien.Entity()
        vis.set_name("glass_jar_visual")
        vis.set_pose(sapien.Pose([x, y, z0]))
        render_body = sapien.render.RenderBodyComponent()

        wall = sapien.render.RenderShapeCylinder(
            radius=outer_r,
            half_length=wall_half,
            material=glass,
        )
        wall.set_local_pose(sapien.Pose([0.0, 0.0, wall_z], upright_q))
        render_body.attach(wall)

        floor = sapien.render.RenderShapeCylinder(
            radius=outer_r * 0.98,
            half_length=max(0.0015, bottom_t * 0.5),
            material=glass,
        )
        floor.set_local_pose(sapien.Pose([0.0, 0.0, bottom_t * 0.5], upright_q))
        render_body.attach(floor)

        vis.add_component(render_body)
        self.scene.add_entity(vis)
        self.jar_visual = vis

        self.jar_bottom_z = self.table_top + self.JAR_BOTTOM_T
        self.jar_fillable_h = self.JAR_HEIGHT - self.JAR_BOTTOM_T

    def _set_jar_damping(self, damping: float):
        if self.jar is None:
            return
        for component in self.jar.actor.get_components():
            if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
                try:
                    component.set_linear_damping(float(damping))
                    component.set_angular_damping(float(damping))
                except Exception:
                    pass

    def _build_fill_rings(self):
        """Three subtle red rings that follow the jar pose."""
        self._ring_entities = []
        ring_material = self._opaque_material(self.RING_RED, 0.70)
        ring_mesh = str(Path("assets/objects/253_glass_jar/rings/thin_ring.glb").resolve())
        x, y = self.jar_xy
        for frac in (0.25, 0.50, 0.75):
            z_local = self.JAR_BOTTOM_T + frac * self.jar_fillable_h
            builder = self.scene.create_actor_builder()
            builder.set_physx_body_type("kinematic")
            builder.add_visual_from_file(filename=ring_mesh, material=ring_material)
            builder.set_initial_pose(sapien.Pose([x, y, self.table_top + 0.001 + z_local]))
            ent = builder.build(name=f"fill_ring_{int(frac * 100)}")
            self._ring_entities.append((float(frac), ent))

    def _build_scale(self):
        """Electronic kitchen scale on the same arm side as the jar."""
        cfg = self._cfg
        scale_xy = getattr(self, "scale_xy", None)
        if scale_xy is None:
            side = float(self.jar_xy[0])
            scale_xy = np.array(
                [
                    float(cfg.get("scale_x", side - 0.26 if side <= 0 else side + 0.26)),
                    float(cfg.get("scale_y", -0.14)),
                ],
                dtype=float,
            )
            self.scale_xy = scale_xy
        scale_id = int(cfg.get("scale_id", 0))
        self.scale = create_actor(
            scene=self,
            pose=sapien.Pose(
                [float(scale_xy[0]), float(scale_xy[1]), self.table_top + 0.001],
                self.YUP_Q,
            ),
            modelname="072_electronicscale",
            model_id=scale_id,
            convex=True,
            is_static=True,
        )
        self.scale_id = scale_id

    def _recolor_actor(self, actor, rgba):
        if actor is None:
            return
        ent = getattr(actor, "actor", actor)
        for comp in ent.get_components():
            if isinstance(comp, sapien.render.RenderBodyComponent):
                for s in comp.render_shapes:
                    try:
                        s.material.set_base_color(list(rgba))
                    except Exception:
                        pass

    def _get_rigid(self, entity):
        obj = entity.actor if hasattr(entity, "actor") else entity
        for c in obj.get_components():
            if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                return c
        return None

    def _make_dynamic(self, entity, mass=None, lin_damp=0.8, ang_damp=1.2):
        rigid = self._get_rigid(entity)
        if rigid is None:
            return None
        try:
            rigid.set_kinematic(False)
            rigid.set_disable_gravity(False)
            rigid.set_linear_damping(float(lin_damp))
            rigid.set_angular_damping(float(ang_damp))
            if mass is not None:
                try:
                    entity.set_mass(float(mass))
                except Exception:
                    rigid.mass = float(mass)
            rigid.set_linear_velocity(np.zeros(3))
            rigid.set_angular_velocity(np.zeros(3))
        except Exception:
            pass
        return rigid

    def _tune_actor_friction(self, entity, static_f=2.4, dynamic_f=2.0, restitution=0.0):
        rigid = self._get_rigid(entity)
        if rigid is None:
            return
        try:
            for s in rigid.get_collision_shapes():
                m = s.get_physical_material()
                m.set_static_friction(float(static_f))
                m.set_dynamic_friction(float(dynamic_f))
                m.set_restitution(float(restitution))
        except Exception:
            pass

    def _settle_physics(self, steps=60):
        """Advance PhysX without recording (egg settle into the bowl)."""
        for _ in range(int(steps)):
            self._update_kinematic_tasks()
            self.scene.step()

    def _decor_pose(self, name, default_xy, default_z=None):
        layout = (getattr(self, "_decor_layout", None) or {}).get(name)
        if layout is None:
            z = self.table_top + 0.001 if default_z is None else float(default_z)
            return (
                np.array([float(default_xy[0]), float(default_xy[1])], dtype=float),
                z,
            )
        return np.asarray(layout["xy"], dtype=float), float(layout["z"])

    def _build_baking_props(self):
        """Static baking clutter: board+bread, flour, chocolate chips, bowl+eggs."""
        cfg = self._cfg
        q = self.YUP_Q

        # Cutting board with bread on top (table or microwave top).
        bread_xy, bread_z = self._decor_pose("bread", cfg.get("bread_xy", [0.28, -0.20]))
        board_scale = float(cfg.get("board_scale_mult", self.BOARD_SCALE_DEFAULT))
        with open("assets/objects/104_board/model_data0.json", encoding="utf-8") as f:
            board_data = json.load(f)
        board_th = float(board_data["extents"][1]) * board_scale
        board_pose = sapien.Pose(
            [float(bread_xy[0]), float(bread_xy[1]), bread_z + 0.5 * board_th],
            list(self.BOARD_QPOS),
        )
        self.board = create_actor(
            scene=self,
            pose=board_pose,
            modelname="104_board",
            model_id=0,
            convex=True,
            is_static=True,
            scale_mult=board_scale,
        )
        self.board.set_name("104_board")
        self.board.config = {
            "scale": [board_scale, board_scale, board_scale],
            "extents": board_data["extents"],
            "center": board_data["center"],
        }
        board_top_z = bread_z + board_th

        self.bread = create_actor(
            scene=self,
            pose=sapien.Pose(
                [float(bread_xy[0]), float(bread_xy[1]), board_top_z + 0.001],
                q,
            ),
            modelname="075_bread",
            model_id=int(cfg.get("bread_id", 0)),
            convex=True,
            is_static=True,
        )

        flour_xy, flour_z = self._decor_pose("flour", cfg.get("flour_xy", [0.38, 0.14]))
        self.flour = create_actor(
            scene=self,
            pose=sapien.Pose([float(flour_xy[0]), float(flour_xy[1]), flour_z], q),
            modelname="261_flour_sack",
            model_id=0,
            convex=True,
            is_static=True,
        )

        chips_xy, chips_z = self._decor_pose("chips", cfg.get("chips_xy", [0.12, -0.22]))
        self.chocolate_chips = create_actor(
            scene=self,
            pose=sapien.Pose([float(chips_xy[0]), float(chips_xy[1]), chips_z], q),
            modelname="263_chocolate_chips_bag",
            model_id=0,
            convex=True,
            is_static=True,
        )

        bowl_xy, bowl_z = self._decor_pose("bowl", cfg.get("bowl_xy", [0.40, -0.02]))
        bowl_id = int(cfg.get("bowl_id", 1))
        bowl_scale_mult = float(cfg.get("bowl_scale_mult", 1.0))
        # Static + nonconvex keeps a hollow interior so eggs can fall inside.
        self.bowl = create_actor(
            scene=self,
            pose=sapien.Pose([float(bowl_xy[0]), float(bowl_xy[1]), bowl_z], q),
            modelname="002_bowl",
            model_id=bowl_id,
            convex=False,
            is_static=True,
            scale_mult=bowl_scale_mult,
        )
        with open(
            f"assets/objects/002_bowl/model_data{bowl_id}.json", encoding="utf-8"
        ) as f:
            bowl_data = json.load(f)
        bowl_scale = float(bowl_data["scale"][1]) * bowl_scale_mult
        bowl_h = float(bowl_data["extents"][1]) * bowl_scale
        bowl_r = 0.5 * min(
            float(bowl_data["extents"][0]), float(bowl_data["extents"][2])
        ) * bowl_scale

        # Drop full-size eggs from just above the rim; gravity seats them inside.
        # 262_egg pose origin = mesh bottom (local Y≈0).
        drop_offsets = [
            (-0.010, 0.0),
            (0.010, 0.0),
            (0.0, 0.008),
        ]
        self.eggs = []
        for i, (dx, dy) in enumerate(drop_offsets):
            drop_z = bowl_z + bowl_h + 0.012 + 0.010 * i
            egg = create_actor(
                scene=self,
                pose=sapien.Pose(
                    [
                        float(bowl_xy[0]) + dx,
                        float(bowl_xy[1]) + dy,
                        float(drop_z),
                    ],
                    q,
                ),
                modelname="262_egg",
                model_id=0,
                convex=True,
                is_static=False,
            )
            self._recolor_actor(egg, self.EGG_ORANGE)
            self._make_dynamic(egg, mass=0.05, lin_damp=1.2, ang_damp=1.8)
            self._tune_actor_friction(egg, static_f=3.0, dynamic_f=2.6, restitution=0.0)
            self.eggs.append(egg)
            self._settle_physics(60)
        self._settle_physics(100)

        # Re-drop any escapee from a short height (still PhysX, no pose snap).
        bowl_center = np.asarray([float(bowl_xy[0]), float(bowl_xy[1])], dtype=float)
        for _ in range(2):
            any_bad = False
            for egg, (dx, dy) in zip(self.eggs, drop_offsets):
                ep = np.asarray(egg.get_pose().p, dtype=float)
                xy = float(np.linalg.norm(ep[:2] - bowl_center))
                inside = (
                    xy < 0.80 * bowl_r
                    and ep[2] > bowl_z - 0.005
                    and ep[2] < bowl_z + bowl_h + 0.07
                )
                if inside:
                    continue
                any_bad = True
                retry = sapien.Pose(
                    [
                        float(bowl_xy[0]) + 0.5 * dx,
                        float(bowl_xy[1]) + 0.5 * dy,
                        float(bowl_z + bowl_h + 0.018),
                    ],
                    q,
                )
                try:
                    egg.actor.set_pose(retry)
                except Exception:
                    egg.set_pose(retry)
                rigid = self._get_rigid(egg)
                if rigid is not None:
                    try:
                        rigid.set_linear_velocity(np.zeros(3))
                        rigid.set_angular_velocity(np.zeros(3))
                    except Exception:
                        pass
            if not any_bad:
                break
            self._settle_physics(80)

        for egg in self.eggs:
            rigid = self._get_rigid(egg)
            if rigid is None:
                continue
            try:
                rigid.set_linear_velocity(np.zeros(3))
                rigid.set_angular_velocity(np.zeros(3))
                rigid.set_linear_damping(2.5)
                rigid.set_angular_damping(3.5)
            except Exception:
                pass

    def _sync_jar_followers(self):
        """Keep glass visual, fill rings, and oil column attached to the jar."""
        if self.jar is None:
            return
        pose = self.jar.get_pose()
        if self.jar_visual is not None:
            self.jar_visual.set_pose(pose)
        for frac, ent in getattr(self, "_ring_entities", []) or []:
            z_local = self.JAR_BOTTOM_T + frac * self.jar_fillable_h
            ent.set_pose(pose * sapien.Pose([0.0, 0.0, z_local]))
        if self._liquid_entity is not None:
            self._liquid_entity.set_pose(pose * sapien.Pose([0.0, 0.0, self.JAR_BOTTOM_T]))
        # Cache jar XY for stream / debug; liquid uses jar pose directly.
        self.jar_xy = np.asarray(pose.p[:2], dtype=float)
        self.jar_bottom_z = float(pose.p[2]) + self.JAR_BOTTOM_T

    # ------------------------------------------------------------------ oil visuals / dynamics
    def _set_tab_open(self, open_: bool):
        """Toggle pour state; red button depresses when ON."""
        open_ = bool(open_)
        if open_ == bool(self.tab_open):
            self._rebuild_switch_button()
            self._sync_stream()
            return
        was_open = bool(self.tab_open)
        self.tab_open = open_
        if self.tab_open and not was_open:
            self.opened_once = True
        if was_open and not self.tab_open and self.liquid_level > 0.05:
            self.closed_after_pour = True
        self._rebuild_switch_button()
        self._sync_stream()

    def _sync_stream(self):
        """Narrow oil cylinder from nozzle outlet down to the table (tab-gated)."""
        self._stream_entity = self._remove_entity(self._stream_entity)
        if not self.tab_open:
            return
        ox, oy, oz = self.nozzle_outlet_xyz
        z_lo = self.table_top + 0.001
        half_h = max(0.01, 0.5 * (oz - z_lo))
        z_c = 0.5 * (oz + z_lo)
        self._stream_entity = self._make_oil_column(
            radius=float(self.NOZZLE_R),
            half_h=half_h,
            world_xyz=[ox, oy, z_c],
            rgba=self.oil_stream_color,
            name="oil_stream",
            local_z=0.0,
        )

    def _rebuild_liquid(self, force: bool = False):
        """Rising oil column in the jar (opaque sunflower yellow or transparent)."""
        if not getattr(self, "jar_fillable_h", None):
            return
        liq_h = max(0.0, float(self.liquid_level)) * self.jar_fillable_h
        liq_half = max(0.002, 0.5 * liq_h) if self.liquid_level > 1e-4 else 0.0
        # While pouring, refresh a bit more often so the level keeps visibly
        # rising through the off-click approach (not an early halt).
        min_dh = 0.0008 if bool(getattr(self, "tab_open", False)) else 0.002
        if (
            not force
            and abs(liq_half - self._liquid_half_h_cached) < min_dh
        ):
            return
        self._liquid_half_h_cached = liq_half
        self._liquid_entity = self._remove_entity(self._liquid_entity)
        if self.liquid_level <= 1e-4:
            return

        if self.jar is not None:
            jar_pose = self.jar.get_pose()
            liquid_pose = jar_pose * sapien.Pose([0.0, 0.0, self.JAR_BOTTOM_T])
        else:
            liquid_pose = sapien.Pose(
                [float(self.jar_xy[0]), float(self.jar_xy[1]), float(self.jar_bottom_z)]
            )
        ent = sapien.Entity()
        ent.set_name("olive_oil_liquid")
        ent.set_pose(liquid_pose)
        body = sapien.render.RenderBodyComponent()

        bulk_mat = self._fluid_material(self.oil_color)
        bulk = sapien.render.RenderShapeCylinder(
            radius=self.JAR_INNER_R * 0.90,
            half_length=liq_half,
            material=bulk_mat,
        )
        bulk.set_local_pose(sapien.Pose([0.0, 0.0, liq_half], self.UPRIGHT_CYL_Q))
        body.attach(bulk)

        # Meniscus only for the see-through style (marks fill without murk).
        if self.oil_transparent and self.oil_meniscus is not None:
            men_half = 0.0012
            men_mat = self._fluid_material(self.oil_meniscus)
            men = sapien.render.RenderShapeCylinder(
                radius=self.JAR_INNER_R * 0.92,
                half_length=men_half,
                material=men_mat,
            )
            men.set_local_pose(
                sapien.Pose(
                    [0.0, 0.0, max(men_half, 2.0 * liq_half - men_half)],
                    self.UPRIGHT_CYL_Q,
                )
            )
            body.attach(men)

        ent.add_component(body)
        self.scene.add_entity(ent)
        self._liquid_entity = ent

    def _spill_radius(self):
        """Table puddle radius from spill_amount in [0, 1]."""
        a = float(np.clip(self.spill_amount, 0.0, 1.0))
        if a <= 1e-4:
            return 0.0
        # Ease-out growth so the puddle appears quickly, then spreads.
        t = float(np.sqrt(a))
        return float(self.SPILL_RADIUS_MIN + t * (self.SPILL_RADIUS_MAX - self.SPILL_RADIUS_MIN))

    def _rebuild_spill(self, force: bool = False):
        """Yellow oil circle on the table under the jar (overflow puddle)."""
        radius = self._spill_radius()
        if not force and abs(radius - self._spill_radius_cached) < 0.0015:
            return
        self._spill_radius_cached = radius
        self._spill_entity = self._remove_entity(self._spill_entity)
        if radius <= 1e-4:
            return

        if self._spill_xy is None:
            if self.jar is not None:
                jp = np.asarray(self.jar.get_pose().p, dtype=float)
                self._spill_xy = np.array([jp[0], jp[1]], dtype=float)
            else:
                self._spill_xy = np.asarray(self.jar_xy, dtype=float).copy()

        sx, sy = float(self._spill_xy[0]), float(self._spill_xy[1])
        z = float(self.table_top) + 0.0022
        rgba = list(getattr(self, "oil_spill_color", self.OIL_SPILL_SOLID))
        # Flat cylinder lying on the table (same upright quat as oil columns).
        self._spill_entity = self._make_oil_column(
            radius=radius,
            half_h=self.SPILL_HALF_H,
            world_xyz=[sx, sy, z + self.SPILL_HALF_H],
            rgba=rgba,
            name="oil_spill",
            local_z=0.0,
        )

    def _step_oil(self):
        # Natural pour: oil rises whenever the switch is on — including the
        # whole approach to the off-click. Never halt on fill target alone.
        # Once the jar is full, further pour spills onto the table as a yellow disk.
        if self.tab_open:
            next_lvl = float(self.liquid_level) + float(self.pour_rate)
            if next_lvl <= float(self.overflow_level):
                self.liquid_level = next_lvl
            else:
                self.liquid_level = float(self.overflow_level)
                # Excess oil puddles under the jar.
                self.spill_amount = min(
                    1.0, float(self.spill_amount) + float(self.spill_rate)
                )
                if self.spill_amount > 1e-4:
                    self.overflowed = True
            # Keep the stream alive every step while ON (only the off-click
            # flips tab_open and removes it via _set_tab_open).
            if getattr(self, "_stream_entity", None) is None:
                self._sync_stream()
        self._rebuild_liquid(force=False)
        self._rebuild_spill(force=False)

    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
        if not getattr(self, "_loaded", False):
            return
        if getattr(self, "_jar_carry", False) and self.jar is not None:
            self._apply_jar_carry()
        elif getattr(self, "_jar_seated_pose", None) is not None and self.jar is not None:
            # Firm seat on the scale — no slow PhysX drop after release.
            try:
                self.jar.actor.set_pose(self._jar_seated_pose)
                for component in self.jar.actor.get_components():
                    if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
                        component.set_linear_velocity(np.zeros(3))
                        component.set_angular_velocity(np.zeros(3))
            except Exception:
                pass
        elif getattr(self, "_jar_locked", False) and self.jar is not None:
            try:
                self.jar.actor.set_pose(self._jar_home_pose)
                for component in self.jar.actor.get_components():
                    if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
                        component.set_linear_velocity(np.zeros(3))
                        component.set_angular_velocity(np.zeros(3))
            except Exception:
                pass
        self._detect_tab_touch()
        self._step_oil()
        self._sync_jar_followers()

    def _clear_tab_collision(self):
        """Drop switch collision after the pour so the jar is reachable."""
        self._ignore_tab = True
        open_ = bool(self.tab_open)
        self._clear_switch_parts()
        # Keep a visual-only housing + button (no collision).
        bx, by, bz = self.switch_base_xyz
        housing = self._metallic_material([0.18, 0.18, 0.20], roughness=0.45, metallic=0.55)
        self._switch_parts.append(
            self._add_static_box(
                pose=sapien.Pose([bx, by, bz + float(self.SWITCH_BASE_HALF[2])]),
                half_size=list(self.SWITCH_BASE_HALF),
                material=housing,
                name="oil_switch_base_visual",
                collision=False,
            )
        )
        center = self._switch_button_center(open_)
        self._switch_btn = self._add_static_box(
            pose=sapien.Pose(center.tolist()),
            half_size=list(self.SWITCH_BTN_HALF),
            material=self._opaque_material(self.SWITCH_RED),
            name="oil_switch_button_visual",
            collision=False,
        )
        self._switch_parts.append(self._switch_btn)
        self._sync_switch_touch_points()

    def _start_jar_carry(self, arm_tag: ArmTag):
        """Kinematically attach the jar under the EE (physics grasp is flaky here)."""
        if str(arm_tag) == "left":
            ee = np.asarray(self.robot.get_left_ee_pose(), dtype=float)
        else:
            ee = np.asarray(self.robot.get_right_ee_pose(), dtype=float)
        jar_p = np.asarray(self.jar.get_pose().p, dtype=float)
        # Keep jar upright under the EE; offset is EE→jar in world frame.
        self._jar_carry_offset = jar_p - ee[:3]
        self._jar_carry = True
        self._jar_locked = False
        self._apply_jar_carry()

    def _stop_jar_carry(self, place_xyz=None):
        """Release the kinematic attach. Optionally settle at ``place_xyz``."""
        self._jar_carry = False
        self._jar_carry_offset = None
        if place_xyz is not None and self.jar is not None:
            self.jar.actor.set_pose(
                sapien.Pose(
                    [float(place_xyz[0]), float(place_xyz[1]), float(place_xyz[2])]
                )
            )
            for component in self.jar.actor.get_components():
                if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
                    try:
                        component.set_linear_velocity(np.zeros(3))
                        component.set_angular_velocity(np.zeros(3))
                    except Exception:
                        pass
        self._sync_jar_followers()

    def _apply_jar_carry(self):
        if self.jar is None or self._jar_carry_offset is None:
            return
        arm = self.arm
        if str(arm) == "left":
            ee = np.asarray(self.robot.get_left_ee_pose(), dtype=float)
        else:
            ee = np.asarray(self.robot.get_right_ee_pose(), dtype=float)
        p = ee[:3] + self._jar_carry_offset
        try:
            self.jar.actor.set_pose(sapien.Pose([float(p[0]), float(p[1]), float(p[2])]))
            for component in self.jar.actor.get_components():
                if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
                    component.set_linear_velocity(np.zeros(3))
                    component.set_angular_velocity(np.zeros(3))
        except Exception:
            pass

    def _ee_pose_for_carried_jar(self, jar_xyz):
        """Top-down EE pose that puts the carried jar at ``jar_xyz``."""
        offset = np.asarray(self._jar_carry_offset, dtype=float)
        ee = np.asarray(jar_xyz, dtype=float) - offset
        return [
            float(ee[0]),
            float(ee[1]),
            float(ee[2]),
            *GRASP_DIRECTION_DIC["top_down"],
        ]

    def _carry_jar_to(self, arm_tag: ArmTag, dest_xyz, tol: float = 0.018):
        """Drive the arm (absolute pose, then axis tweaks) until jar is at dest."""
        dest = np.asarray(dest_xyz, dtype=float)
        if self.jar is None or self._jar_carry_offset is None:
            return False

        self.move(self.move_to_pose(arm_tag, self._ee_pose_for_carried_jar(dest)))
        if not self.plan_success:
            self.plan_success = True

        # Axis-separated nudges until the jar is actually there (no end snap).
        for _ in range(8):
            self._apply_jar_carry()
            jar_p = np.asarray(self.jar.get_pose().p, dtype=float)
            err = dest - jar_p
            if float(np.linalg.norm(err)) <= tol:
                return True
            moved = False
            for axis, val in (("x", err[0]), ("y", err[1]), ("z", err[2])):
                if abs(float(val)) < 0.008:
                    continue
                kw = {axis: float(val)}
                self.move(self.move_by_displacement(arm_tag, **kw))
                moved = True
                if not self.plan_success:
                    self.plan_success = True
            if not moved:
                break
        self._apply_jar_carry()
        jar_p = np.asarray(self.jar.get_pose().p, dtype=float)
        return float(np.linalg.norm(dest - jar_p)) <= tol * 1.5

    def _seat_jar_on_scale(self, place_xyz):
        """Release carry and lock the jar where it already is (plate height)."""
        self._jar_carry = False
        self._jar_carry_offset = None
        self._jar_locked = False
        # Use current XY (arm already delivered it); only set Z to the plate.
        if self.jar is not None:
            cur = np.asarray(self.jar.get_pose().p, dtype=float)
            target = np.asarray(place_xyz, dtype=float)
            # Tiny XY blend only if already near center — never a long snap.
            dist_xy = float(np.linalg.norm(cur[:2] - target[:2]))
            if dist_xy < 0.025:
                xy = target[:2]
            else:
                xy = cur[:2]
                print(
                    f"[measure_ingredient] seat without snap "
                    f"(jar still {dist_xy:.3f}m from center)"
                )
            pose = sapien.Pose([float(xy[0]), float(xy[1]), float(target[2])])
        else:
            pose = sapien.Pose(
                [float(place_xyz[0]), float(place_xyz[1]), float(place_xyz[2])]
            )
        self._jar_seated_pose = pose
        if self.jar is None:
            return
        self.jar.actor.set_pose(pose)
        for component in self.jar.actor.get_components():
            if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
                try:
                    component.set_linear_velocity(np.zeros(3))
                    component.set_angular_velocity(np.zeros(3))
                    component.set_linear_damping(50.0)
                    component.set_angular_damping(50.0)
                except Exception:
                    pass
        self._sync_jar_followers()

    def _detect_tab_touch(self):
        """Edge-trigger: fingertip on the red switch toggles on/off."""
        if getattr(self, "_ignore_tab", False):
            self._touch_latched = False
            return
        if not hasattr(self, "robot"):
            return
        arm = getattr(self, "arm", None)
        if arm is None:
            return
        try:
            if str(arm) == "left":
                ee = np.asarray(self.robot.get_left_ee_pose()[:3], dtype=float)
            else:
                ee = np.asarray(self.robot.get_right_ee_pose()[:3], dtype=float)
        except Exception:
            return
        # Fixed switch XY so rebuilding the button mid-press cannot re-trigger.
        base = getattr(self, "switch_base_xyz", None)
        if base is None:
            return
        top_z = self._switch_top_z()
        xy_ok = float(np.linalg.norm(ee[:2] - base[:2])) <= 0.035
        z_ok = ee[2] <= float(top_z) + self.EE_TO_TCP + 0.015
        touching = bool(xy_ok and z_ok)
        if touching and not self._touch_latched:
            self._set_tab_open(not self.tab_open)
        self._touch_latched = touching

    def _idle_steps(self, n_steps: int, until=None):
        save_freq = self.save_freq if self.save_freq is not None else 15
        for i in range(int(n_steps)):
            if until is not None and until():
                break
            self._update_kinematic_tasks()
            self.scene.step()
            if self.render_freq and i % max(1, int(self.render_freq)) == 0:
                self._update_render()
                if hasattr(self, "viewer") and self.viewer is not None:
                    self.viewer.render()
            if self.save_freq is not None and i % save_freq == 0:
                self._take_picture()

    # ------------------------------------------------------------------ expert
    def _switch_ee_pose(self, tip_z_above: float):
        """Top-down EE pose above the red switch button."""
        c = self._switch_button_center()
        top_z = self._switch_top_z()
        return [
            float(c[0]),
            float(c[1]),
            float(top_z + tip_z_above + self.EE_TO_TCP),
            *GRASP_DIRECTION_DIC["top_down"],
        ]

    def _press_switch(self, arm_tag: ArmTag, want_open: bool):
        """Click the red switch (hover → press → release) to the desired state.

        Oil keeps flowing for the entire approach when turning OFF; ``tab_open``
        flips only at the depress/click instant below.
        """
        was = self.tab_open
        # Ignore ambient touch; expert sets the state only at the click.
        self._ignore_tab = True

        self.move(self.close_gripper(arm_tag))
        if not self.plan_success:
            return False

        self.move(
            self.move_to_pose(arm_tag, self._switch_ee_pose(self.SWITCH_HOVER_Z))
        )
        if not self.plan_success:
            print(f"[measure_ingredient] switch hover failed want_open={want_open}")
            return False

        self.move(
            self.move_to_pose(arm_tag, self._switch_ee_pose(self.SWITCH_PRESS_Z))
        )
        self._idle_steps(2)
        # Depress to "click" — this is the moment the switch state changes.
        self.move(self.move_by_displacement(arm_tag, z=-0.012))
        if not self.plan_success:
            self.plan_success = True
        self._idle_steps(2)
        self._set_tab_open(want_open)
        self._idle_steps(2)
        self.move(self.move_by_displacement(arm_tag, z=0.06))
        self._touch_latched = False
        self._ignore_tab = True
        print(
            f"[measure_ingredient] click switch {was}→{self.tab_open} "
            f"(want={want_open}) liq={self.liquid_level:.2f}"
        )
        return bool(self.plan_success)

    def play_once(self):
        arm = self.arm
        self.move(self.close_gripper(arm))
        if not self.plan_success:
            print("[measure_ingredient] close_gripper failed")
            return self.info

        # 1) Click the red switch ON → stream appears, oil starts filling.
        if not self._press_switch(arm, want_open=True):
            return self.info
        if not self.tab_open:
            print("[measure_ingredient] switch failed to open")
            self.plan_success = False
            return self.info

        self._ignore_tab = True

        # 2) Wait near the target ring. Oil (and the stream) keep going for the
        # whole off-click approach; only the button press stops the pour.
        # Start the approach a little early so fill lands near the mark.
        # force_overflow: deliberately leave the switch ON past full so a yellow
        # spill disk grows under the jar (failure demo).
        if getattr(self, "force_overflow", False):
            # Faster pour/spill for a short showcase (visuals unchanged).
            self.pour_rate = max(float(self.pour_rate), 0.0015)
            self.spill_rate = max(float(self.spill_rate), 0.0025)
            spill_target = float(self._cfg.get("spill_demo_amount", 0.55))
            max_wait = int(1.05 / max(1e-6, self.pour_rate)) + int(
                spill_target / max(1e-6, self.spill_rate)
            ) + 200
            self._idle_steps(
                max_wait,
                until=lambda: self.spill_amount >= spill_target,
            )
            print(
                f"[measure_ingredient] force_overflow liq={self.liquid_level:.2f} "
                f"spill={self.spill_amount:.2f} r={self._spill_radius():.3f}m "
                f"tab_open={self.tab_open}"
            )
            # Click OFF, then hold so the puddle is clearly visible on camera.
            self._ignore_tab = True
            if self.plan_success and self.tab_open:
                self._press_switch(arm, want_open=False)
            self._ignore_tab = True
            if self.plan_success:
                self.move(self.move_by_displacement(arm, z=0.08))
            self._idle_steps(90)
            self.plan_success = False  # overflowed episode is a failure
            level_pct = int(round(self.target_fill * 100))
            self.info["info"] = {
                "{A}": "olive oil dispenser",
                "{B}": f"glass jar ({level_pct}% line)",
                "{C}": "red nozzle switch",
                "{D}": f"072_electronicscale/base{getattr(self, 'scale_id', 0)}",
                "{a}": str(arm),
                "{L}": f"{level_pct}%",
            }
            return self.info

        # Start the off-click early so fill lands inside ±fill_tol after approach.
        # Approach still pours; lead ≈ expected overshoot (+ small margin).
        lead = float(
            self._cfg.get(
                "pour_close_lead",
                max(0.05, min(0.14, 220.0 * float(self.pour_rate) + 0.03)),
            )
        )
        # For 100% (rim), stop earlier so we do not spill over the edge.
        if float(self.target_fill) >= 0.999:
            lead = max(lead, 0.12)
        close_start = max(0.05, float(self.target_fill) - lead)
        max_wait = int(close_start / max(1e-6, self.pour_rate)) + 160
        self._idle_steps(
            max_wait,
            until=lambda: (
                self.liquid_level >= close_start or self.overflowed
            ),
        )
        print(
            f"[measure_ingredient] after pour liq={self.liquid_level:.2f} "
            f"target={self.target_fill:.2f} overflow={self.overflowed} "
            f"spill={self.spill_amount:.2f} tab_open={self.tab_open}"
        )

        # 3) Click the switch OFF → stream stops only when the button clicks.
        self._ignore_tab = True
        if self.plan_success:
            if not self.tab_open:
                print("[measure_ingredient] WARNING: switch already off before off-click")
            self._press_switch(arm, want_open=False)
        self._ignore_tab = True

        if self.plan_success:
            self.move(self.move_by_displacement(arm, z=0.08))

        # 4) Grasp the filled jar and place it on the electronic scale.
        if self.plan_success and self.scale is not None:
            self._clear_tab_collision()
            self._jar_locked = False
            self._set_jar_damping(3.0)
            self.move(self.open_gripper(arm))
            if self.plan_success:
                jar_p = np.asarray(self.jar.get_pose().p, dtype=float)
                # Approach from the robot side, then descend onto the jar.
                hover = [
                    float(jar_p[0]),
                    float(jar_p[1] - 0.10),
                    float(jar_p[2] + 0.22),
                    *GRASP_DIRECTION_DIC["top_down"],
                ]
                self.move(self.move_to_pose(arm, hover))
            if self.plan_success:
                jar_p = np.asarray(self.jar.get_pose().p, dtype=float)
                above = [
                    float(jar_p[0]),
                    float(jar_p[1]),
                    float(jar_p[2] + 0.22),
                    *GRASP_DIRECTION_DIC["top_down"],
                ]
                self.move(self.move_to_pose(arm, above))
            if self.plan_success:
                # Descend in small steps — full IK to grasp depth is unreliable here.
                for _ in range(4):
                    self.move(self.move_by_displacement(arm, z=-0.04))
                    if not self.plan_success:
                        self.plan_success = True
                        break
                self.move(self.close_gripper(arm))
                self._start_jar_carry(arm)
                self.move(self.move_by_displacement(arm, z=0.12))
            if not self.plan_success:
                print("[measure_ingredient] jar grasp failed")
            if self.plan_success:
                self._place_jar_on_scale(arm)
            if not self.plan_success:
                print("[measure_ingredient] place jar on scale failed")
            else:
                print(
                    f"[measure_ingredient] jar on scale? "
                    f"{self.check_success()} pose={self.jar.get_pose().p}"
                )

        if self.check_success():
            self.plan_success = True

        level_pct = int(round(self.target_fill * 100))
        self.info["info"] = {
            "{A}": "olive oil dispenser",
            "{B}": f"glass jar ({level_pct}% line)",
            "{C}": "red nozzle switch",
            "{D}": f"072_electronicscale/base{getattr(self, 'scale_id', 0)}",
            "{a}": str(arm),
            "{L}": f"{level_pct}%",
        }
        return self.info

    def _scale_place_xyz(self):
        """Jar-bottom target on the scale plate center (functional point)."""
        scale_fp = self.scale.get_functional_point(0)
        if isinstance(scale_fp, sapien.Pose):
            target = np.asarray(scale_fp.p, dtype=float)
        else:
            target = np.asarray(scale_fp[:3], dtype=float)
        return np.array(
            [float(target[0]), float(target[1]), float(target[2]) + 0.002],
            dtype=float,
        )

    def _place_jar_on_scale(self, arm_tag: ArmTag):
        """Carry the jar over the scale center with the arm, then release."""
        place_xyz = self._scale_place_xyz()
        jar_now = np.asarray(self.jar.get_pose().p, dtype=float)

        # Waypoints: lift → mid → hover over plate center → lower. Absolute EE
        # poses keep the jar in the gripper and over the center before release.
        lift_xyz = np.array(
            [jar_now[0], jar_now[1], float(place_xyz[2] + 0.18)], dtype=float
        )
        mid_xyz = np.array(
            [
                0.5 * (jar_now[0] + place_xyz[0]),
                0.5 * (jar_now[1] + place_xyz[1]),
                float(place_xyz[2] + 0.18),
            ],
            dtype=float,
        )
        hover_xyz = place_xyz + np.array([0.0, 0.0, 0.14], dtype=float)
        lower_xyz = place_xyz + np.array([0.0, 0.0, 0.02], dtype=float)

        self._carry_jar_to(arm_tag, lift_xyz, tol=0.03)
        self._carry_jar_to(arm_tag, mid_xyz, tol=0.03)
        ok_hover = self._carry_jar_to(arm_tag, hover_xyz, tol=0.02)
        ok_lower = self._carry_jar_to(arm_tag, lower_xyz, tol=0.015)
        jar_p = np.asarray(self.jar.get_pose().p, dtype=float)
        dist_xy = float(np.linalg.norm(jar_p[:2] - place_xyz[:2]))
        print(
            f"[measure_ingredient] pre-release jar xy err={dist_xy:.3f}m "
            f"hover_ok={ok_hover} lower_ok={ok_lower}"
        )

        # Open only once the jar is over the plate; seat without a long snap.
        self.move(self.open_gripper(arm_tag))
        self._seat_jar_on_scale(place_xyz)
        self._idle_steps(6)
        if self.plan_success:
            self.move(self.move_by_displacement(arm_tag, z=0.10))
            if not self.plan_success:
                self.plan_success = True
            self._idle_steps(6)

    def check_success(self):
        """Oil in [target±tol], no spill/overflow, switch off, jar on scale."""
        if self.overflowed or float(getattr(self, "spill_amount", 0.0)) > 1e-4:
            return False
        if not self.opened_once:
            return False
        if self.tab_open:
            return False
        if not self.closed_after_pour:
            return False
        lo = float(self.target_fill) - float(self.fill_tol)
        hi = float(self.target_fill) + float(self.fill_tol)
        # 100% rim: cannot exceed 1.0 without overflowing; allow up to overflow_level.
        if float(self.target_fill) >= 0.999:
            hi = min(hi, float(self.overflow_level) + 1e-6)
        lvl = float(self.liquid_level)
        if lvl + 1e-3 < lo:
            return False
        if lvl - 1e-3 > hi:
            return False
        if self.jar is None or self.scale is None:
            return False

        jar_p = np.asarray(self.jar.get_pose().p, dtype=float)
        scale_fp = self.scale.get_functional_point(0)
        if isinstance(scale_fp, sapien.Pose):
            scale_p = np.asarray(scale_fp.p, dtype=float)
        else:
            scale_p = np.asarray(scale_fp[:3], dtype=float)
        dist_xy = float(np.linalg.norm(jar_p[:2] - scale_p[:2]))
        on_scale = dist_xy < 0.05 and jar_p[2] > (scale_p[2] - 0.02)
        self.jar_on_scale = bool(on_scale)
        return bool(on_scale)

    def get_obs(self):
        obs = super().get_obs()
        mw = getattr(self, "microwave_xy_override", None) or getattr(
            self, "microwave_xy", None
        )
        obs["measure_ingredient"] = {
            "target_fill": float(self.target_fill),
            "fill_tol": float(getattr(self, "fill_tol", self.FILL_TOL)),
            "pour_rate": float(getattr(self, "pour_rate", self.POUR_RATE)),
            "pour_rate_base": float(
                getattr(self, "pour_rate_base", getattr(self, "pour_rate", self.POUR_RATE))
            ),
            "pour_rate_jitter": float(
                getattr(self, "pour_rate_jitter", self.POUR_RATE_JITTER)
            ),
            "liquid_level": float(self.liquid_level),
            "tab_open": bool(self.tab_open),
            "overflowed": bool(self.overflowed),
            "spill_amount": float(getattr(self, "spill_amount", 0.0)),
            "opened_once": bool(self.opened_once),
            "closed_after_pour": bool(self.closed_after_pour),
            "jar_on_scale": bool(getattr(self, "jar_on_scale", False)),
            "oil_style": str(getattr(self, "oil_style", self.OIL_STYLE_DEFAULT)),
            "scene_id": int(getattr(self, "scene_id", 0)),
            "station_xy": (
                None
                if getattr(self, "jar_xy", None) is None
                else [float(self.jar_xy[0]), float(self.jar_xy[1])]
            ),
            "scale_xy": (
                None
                if getattr(self, "scale_xy", None) is None
                else [float(self.scale_xy[0]), float(self.scale_xy[1])]
            ),
            "microwave_xy": (
                None if mw is None else [float(mw[0]), float(mw[1])]
            ),
        }
        return obs
