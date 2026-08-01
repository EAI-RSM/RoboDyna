"""Pour beer from a bar tap into a glass (KitchenS).

Chrome draft tower with a hinged wooden lever. The robot opens/closes the lever
by hand along its joint arc — not a binary click. Beer stream + foam behave like
the bottle pour: while the lever is open, liquid and foam keep rising; flow rate
scales with how far the lever is pulled. Overflow fails with a yellow stain.

Episode randomization (task_args.pour_beer):
  - ``randomize_layout``: cup/tap station + bar props with AABB non-overlap
  - ``randomize_rates`` / ``pour_rate_range`` / ``foam_gain_range``: fill & foam speed
"""
from __future__ import annotations

import numpy as np
import sapien
import sapien.render
import transforms3d as t3d

from ._kitchens_base_task import KitchenS_base_task
from ._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
from .utils import *
from .utils.create_actor import create_actor


class pour_beer(KitchenS_base_task):
    """Drive the tap lever by hand to fill the glass without foam overflow."""

    GLASS_MODEL = "257_beer_glass"
    GLASS_UPRIGHT_Q = [0.70710678, 0.70710678, 0.0, 0.0]
    VERTICAL_CYL_Q = [0.70710678, 0.0, 0.70710678, 0.0]

    # Tap tower geometry (meters).
    TOWER_R = 0.018
    TOWER_H = 0.22
    BASE_R = 0.055
    BASE_H = 0.012
    ARM_LEN = 0.055
    SPOUT_DROP = 0.035
    NOZZLE_R = 0.007
    LEVER_LEN = 0.085
    LEVER_R = 0.012
    LEVER_OPEN_RAD = 0.55 * np.pi  # ~99° forward (−Y) when fully open
    LEVER_DEADZONE = 0.06  # rad — below this, no flow
    LEVER_OPEN_THRESH = 0.18  # rad — counts as "opened" for success

    EE_TO_TCP = 0.12
    LEVER_HOVER = 0.05
    LEVER_GRASP_Z = 0.0

    TARGET_LIQUID = 0.90
    FULL_LIQUID_TOL = 0.05
    # Max rate at full open (per physics step); scales with lever angle.
    # Keep modest — long IK arcs while open still advance fill every tick.
    POUR_RATE = 0.00055
    FOAM_GAIN = 0.80
    FOAM_DECAY = 0.0045
    # Per-episode sample ranges when randomize_rates / [lo,hi] yaml values are used.
    POUR_RATE_RANGE = (0.00040, 0.00075)
    FOAM_GAIN_RANGE = (0.55, 1.05)
    FOAM_DECAY_RANGE = (0.0035, 0.0060)
    OVERFLOW_LEVEL = 1.0
    EXPERT_FOAM_PAUSE = 0.16
    EXPERT_FOAM_RESUME = 0.09
    SAFE_TOTAL = 0.90

    # Non-overlap layout (axis-aligned footprint half-sizes, meters).
    LAYOUT_MARGIN = 0.030
    # Reserved clear zone covering glass + tap base + lever arc.
    STATION_HALF_XY = (0.10, 0.16)
    STATION_X_RANGE = (0.06, 0.16)
    CUP_Y_RANGE = (-0.14, -0.05)
    PROP_X_RANGE = (-0.55, 0.55)
    PROP_Y_RANGE = (-0.26, 0.28)

    BEER_COLOR = [0.78, 0.52, 0.10, 0.78]
    FOAM_COLOR = [0.97, 0.95, 0.90, 0.88]
    STREAM_COLOR = [0.82, 0.55, 0.12, 0.85]
    STAIN_COLOR = [0.95, 0.78, 0.05, 1.0]
    STAIN_COLOR_LOBE = [0.93, 0.70, 0.04, 1.0]
    STAIN_COLOR_DRIP = [0.90, 0.68, 0.08, 0.95]
    GLASS_RGBA = [0.82, 0.93, 0.98, 0.22]
    CHROME = [0.78, 0.80, 0.84, 1.0]
    CHROME_DARK = [0.50, 0.52, 0.56, 1.0]
    WOOD = [0.55, 0.32, 0.14, 1.0]
    BASE_WOOD = [0.42, 0.26, 0.12, 1.0]

    def setup_demo(self, **kwags):
        self._cfg = dict(kwags.get("task_args", {}).get("pour_beer", {}))
        if kwags.get("scene_id") is None:
            kwags["scene_id"] = int(self._cfg.get("scene_id", 0))
        self.clear_sink_and_range = True
        self.replace_sink_with_range = False
        self._layout_seed = int(kwags.get("seed", 0) or 0)

        self._loaded = False
        self.lever_angle = 0.0
        self._lever_angle_max = 0.0
        self.overflowed = False
        self.opened_once = False
        self.closed_after_pour = False
        self.liquid_level = 0.0
        self.foam_level = 0.0
        self._liquid_entity = None
        self._foam_entity = None
        self._stream_entity = None
        self._stain_entity = None
        self._drip_entity = None
        self._lever_entity = None
        self._lever_comp = None
        self._tap_parts = []
        self._bar_props = []
        self._prop_footprints = []
        self._liquid_half_h_cached = -1.0
        self._foam_half_h_cached = -1.0
        self._lever_held = False
        self._stream_frac_cached = -1.0
        self.spill_amount = 0.0
        self.cup = None
        self.table_top = 0.74
        self.coaster_top_z = 0.75

        super().setup_demo(**kwags)
        self._style_bar_wall()
        self._configure_observer_camera()

    def _load_microwave(self, table_height, table_xy_bias):
        return

    # ------------------------------------------------------------------ scene polish
    def _style_bar_wall(self):
        wall = getattr(self, "wall", None)
        if wall is None:
            return
        try:
            ent = wall.actor if hasattr(wall, "actor") else wall
            for c in ent.get_components():
                if not isinstance(c, sapien.render.RenderBodyComponent):
                    continue
                for s in c.render_shapes:
                    try:
                        s.material.set_base_color([0.22, 0.12, 0.10, 1.0])
                        s.material.set_roughness(0.85)
                    except Exception:
                        try:
                            s.material.base_color = [0.22, 0.12, 0.10, 1.0]
                        except Exception:
                            pass
        except Exception:
            pass

    def _configure_observer_camera(self):
        cams = getattr(self, "cameras", None)
        if cams is None or getattr(cams, "observer_camera", None) is None:
            return
        camera = cams.observer_camera
        camera_pos = np.array([0.10, -0.55, 1.30], dtype=np.float64)
        look_at = np.array([0.10, 0.02, 0.92], dtype=np.float64)
        forward = look_at - camera_pos
        forward /= np.linalg.norm(forward)
        left = np.cross(np.array([0.0, 0.0, 1.0]), forward)
        left /= np.linalg.norm(left)
        up = np.cross(forward, left)
        m = np.eye(4)
        m[:3, :3] = np.stack([forward, left, up], axis=1)
        m[:3, 3] = camera_pos
        camera.entity.set_pose(sapien.Pose(m))

    # ------------------------------------------------------------------ materials / builders
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

    def _opaque_material(self, rgb, alpha=1.0):
        mat = sapien.render.RenderMaterial(base_color=list(rgb[:3]) + [float(alpha)])
        try:
            mat.set_roughness(0.55)
            mat.set_metallic(0.0)
        except Exception:
            mat.roughness = 0.55
            mat.metallic = 0.0
        return mat

    def _fluid_material(self, rgba):
        mat = sapien.render.RenderMaterial(base_color=list(rgba))
        try:
            mat.set_roughness(0.18)
            mat.set_metallic(0.0)
        except Exception:
            mat.roughness = 0.18
            mat.metallic = 0.0
        return mat

    def _remove_entity(self, ent):
        if ent is None:
            return None
        try:
            self.scene.remove_entity(ent)
        except Exception:
            pass
        return None

    def _add_static_box(self, pose, half_size, material=None, name="", collision=True):
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        if collision:
            builder.add_box_collision(
                pose=sapien.Pose(),
                half_size=list(half_size),
                material=self.scene.default_physical_material,
            )
        if material is None:
            material = self._opaque_material([0.8, 0.8, 0.8])
        builder.add_box_visual(pose=sapien.Pose(), half_size=list(half_size), material=material)
        builder.set_initial_pose(pose)
        return builder.build(name=name)

    def _add_static_cylinder(self, pose, radius, half_h, material, name="", collision=True):
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        cyl_pose = sapien.Pose([0, 0, 0], self.VERTICAL_CYL_Q)
        if collision:
            builder.add_cylinder_collision(
                pose=cyl_pose,
                radius=float(radius),
                half_length=float(half_h),
                material=self.scene.default_physical_material,
            )
        builder.add_cylinder_visual(
            pose=cyl_pose,
            radius=float(radius),
            half_length=float(half_h),
            material=material,
        )
        builder.set_initial_pose(pose)
        return builder.build(name=name)

    def _make_column(self, half_h, radius, z_center, rgba, name, xy=None):
        xy = self._cup_center_xy() if xy is None else xy
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        builder.add_cylinder_visual(
            pose=sapien.Pose([0, 0, 0], self.VERTICAL_CYL_Q),
            radius=float(radius),
            half_length=max(0.002, float(half_h)),
            material=self._fluid_material(rgba),
        )
        builder.set_initial_pose(
            sapien.Pose(p=[float(xy[0]), float(xy[1]), float(z_center)])
        )
        return builder.build(name=name)

    def _make_glass_transparent(self, actor):
        try:
            for c in actor.actor.get_components():
                if not isinstance(c, sapien.render.RenderBodyComponent):
                    continue
                for s in c.render_shapes:
                    try:
                        s.material.set_base_color(list(self.GLASS_RGBA))
                        s.material.set_transmission(0.88)
                        s.material.set_transmission_roughness(0.03)
                        s.material.set_roughness(0.05)
                        s.material.set_metallic(0.0)
                        s.material.set_ior(1.45)
                    except Exception:
                        try:
                            s.material.base_color = list(self.GLASS_RGBA)
                            s.material.transmission = 0.88
                        except Exception:
                            pass
        except Exception:
            pass

    # ------------------------------------------------------------------ layout / rates
    def _layout_rng(self, salt: int = 0) -> np.random.RandomState:
        return np.random.RandomState(int(getattr(self, "_layout_seed", 0)) + int(salt))

    def _parse_range(self, cfg, key, default):
        raw = cfg.get(key, default)
        if raw is None:
            raw = default
        return (float(raw[0]), float(raw[1]))

    def _sample_scalar_or_range(self, cfg, key, default, rng, range_key=None, range_default=None):
        """Return a float from a scalar yaml value or a [lo, hi] range.

        - ``key: [lo, hi]`` always samples once per episode.
        - With ``randomize_rates: true``, sample from ``range_key`` (or class
          ``range_default``) when the scalar ``key`` is not itself a range.
        - Otherwise use the scalar ``key`` / ``default``.
        """
        raw = cfg.get(key, default)
        if isinstance(raw, (list, tuple)) and len(raw) == 2:
            return float(rng.uniform(float(raw[0]), float(raw[1])))
        if bool(cfg.get("randomize_rates", False)):
            if range_key is not None and cfg.get(range_key) is not None:
                lo, hi = self._parse_range(cfg, range_key, range_default or (default, default))
                return float(rng.uniform(lo, hi))
            if range_default is not None:
                return float(rng.uniform(float(range_default[0]), float(range_default[1])))
        return float(raw if raw is not None else default)

    @staticmethod
    def _aabb_overlap(c1, h1, c2, h2, margin=0.0):
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
        if margin is None:
            margin = self.LAYOUT_MARGIN
        for b_c, b_h in blockers:
            if self._aabb_overlap(center, half, b_c, b_h, margin=margin):
                return False
        return True

    def _sample_free_xy(self, rng, half, blockers, x_range, y_range, tries=80):
        x_lo, x_hi = float(x_range[0]), float(x_range[1])
        y_lo, y_hi = float(y_range[0]), float(y_range[1])
        half = np.asarray(half, dtype=float)
        for _ in range(int(tries)):
            p = np.array([rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)], dtype=float)
            if self._footprint_clear(p, half, blockers):
                return p
        # Fallback: densest search for max clearance.
        best, best_score = None, -1e9
        for _ in range(120):
            p = np.array([rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)], dtype=float)
            if not blockers:
                return p
            score = min(
                min(
                    abs(p[0] - b_c[0]) - (half[0] + b_h[0] + self.LAYOUT_MARGIN),
                    abs(p[1] - b_c[1]) - (half[1] + b_h[1] + self.LAYOUT_MARGIN),
                )
                for b_c, b_h in blockers
            )
            if score > best_score:
                best, best_score = p, score
            if score >= 0.0:
                return p
        return best if best is not None else np.array([0.45, -0.20], dtype=float)

    def _station_center_xy(self):
        """Midpoint between glass and tap (reserved clear zone center)."""
        return 0.5 * (np.asarray(self.cup_xy, dtype=float) + np.asarray(self.tap_xy, dtype=float))

    def _resolve_station_layout(self, cfg, rng):
        """Sample cup/tap XY; keep spout aligned over the glass (shared X, fixed dy)."""
        tap_dy = float(cfg.get("tap_dy", 0.12))
        randomize = bool(cfg.get("randomize_layout", False))
        if not randomize:
            side = float(cfg.get("station_x", 0.10))
            cup_y = float(cfg.get("cup_y", -0.08))
            return side, cup_y, tap_dy

        x_lo, x_hi = self._parse_range(cfg, "station_x_range", self.STATION_X_RANGE)
        y_lo, y_hi = self._parse_range(cfg, "cup_y_range", self.CUP_Y_RANGE)
        for _ in range(60):
            side = float(rng.uniform(x_lo, x_hi))
            cup_y = float(rng.uniform(y_lo, y_hi))
            # Keep the pour station on the reachable right-arm side of the counter.
            if side < 0.02:
                continue
            return side, cup_y, tap_dy
        return 0.10, -0.08, tap_dy

    def _resolve_rates(self, cfg, rng):
        self.pour_rate = self._sample_scalar_or_range(
            cfg, "pour_rate", self.POUR_RATE, rng,
            range_key="pour_rate_range", range_default=self.POUR_RATE_RANGE,
        )
        self.foam_gain = self._sample_scalar_or_range(
            cfg, "foam_gain", self.FOAM_GAIN, rng,
            range_key="foam_gain_range", range_default=self.FOAM_GAIN_RANGE,
        )
        self.foam_decay = self._sample_scalar_or_range(
            cfg, "foam_decay", self.FOAM_DECAY, rng,
            range_key="foam_decay_range", range_default=self.FOAM_DECAY_RANGE,
        )
        # Clamp to safe positive bounds.
        self.pour_rate = float(np.clip(self.pour_rate, 1e-5, 0.005))
        self.foam_gain = float(np.clip(self.foam_gain, 0.05, 2.5))
        self.foam_decay = float(np.clip(self.foam_decay, 1e-4, 0.05))

    # ------------------------------------------------------------------ actors
    def load_actors(self):
        cfg = self._cfg
        self.table_top = float(self.kitchens_info["table_height"]) + float(self.table_z_bias)
        rng = self._layout_rng(101)

        self._resolve_rates(cfg, rng)
        self.overflow_level = float(cfg.get("overflow_level", self.OVERFLOW_LEVEL))
        self.target_liquid = float(cfg.get("target_liquid", self.TARGET_LIQUID))
        self.full_liquid_tol = float(cfg.get("full_liquid_tol", self.FULL_LIQUID_TOL))
        self.expert_foam_pause = float(cfg.get("expert_foam_pause", self.EXPERT_FOAM_PAUSE))
        self.expert_foam_resume = float(cfg.get("expert_foam_resume", self.EXPERT_FOAM_RESUME))
        self.safe_total = float(cfg.get("safe_total", self.SAFE_TOTAL))
        self.lever_open_rad = float(cfg.get("lever_open_rad", self.LEVER_OPEN_RAD))

        side, cup_y, tap_dy = self._resolve_station_layout(cfg, rng)
        self.arm = ArmTag("right" if side >= 0 else "left")
        self.cup_xy = np.array([side, cup_y], dtype=float)
        self.tap_xy = np.array([side, cup_y + tap_dy], dtype=float)

        self.liquid_level = 0.0
        self.foam_level = 0.0
        self.lever_angle = 0.0
        self._lever_angle_max = 0.0
        self.overflowed = False
        self.opened_once = False
        self.closed_after_pour = False
        self._liquid_entity = None
        self._foam_entity = None
        self._stream_entity = None
        self._stain_entity = self._remove_entity(getattr(self, "_stain_entity", None))
        self._drip_entity = self._remove_entity(getattr(self, "_drip_entity", None))
        self._lever_entity = None
        self._lever_comp = None
        self._tap_parts = []
        self._liquid_half_h_cached = -1.0
        self._foam_half_h_cached = -1.0
        self._stream_frac_cached = -1.0
        self._lever_held = False
        self.spill_amount = 0.0
        self._bar_props = []
        self._prop_footprints = []

        # Background décor — randomized non-overlapping when enabled.
        self._build_bar_props(rng)
        self._spawn_coaster()
        self._spawn_glass()
        self._build_tap()
        self._spawn_lever()
        self._apply_lever_pose(0.0)
        self._rebuild_fluids(force=True)
        self._sync_stream(force=True)

        self._loaded = True
        print(
            f"[pour_beer] tap scene={self.scene_id} arm={self.arm} seed={self._layout_seed} "
            f"cup={self.cup_xy} tap={self.tap_xy} spout={self.nozzle_outlet_xyz} "
            f"pivot={self.lever_pivot_xyz} target={self.target_liquid:.2f} "
            f"pour_rate={self.pour_rate:.5f} foam_gain={self.foam_gain:.2f} "
            f"foam_decay={self.foam_decay:.4f} bar_props={len(self._bar_props)}"
        )

    def _yaw_upright(self, yaw_deg: float = 0.0) -> list:
        """Y-up mesh upright quat with an optional world yaw (degrees)."""
        base = np.array(self.GLASS_UPRIGHT_Q, dtype=float)
        if abs(yaw_deg) < 1e-6:
            return base.tolist()
        yaw_q = t3d.euler.euler2quat(0.0, 0.0, np.deg2rad(yaw_deg), axes="sxyz")
        return t3d.quaternions.qmult(yaw_q, base).tolist()

    def _spawn_static_prop(
        self,
        modelname: str,
        xy,
        model_id: int = 0,
        yaw_deg: float = 0.0,
        z_off: float = 0.001,
        scale_mult: float = 1.0,
        half_xy=None,
    ):
        pose = sapien.Pose(
            [float(xy[0]), float(xy[1]), self.table_top + float(z_off)],
            self._yaw_upright(yaw_deg),
        )
        try:
            actor = create_actor(
                self,
                pose=pose,
                modelname=modelname,
                model_id=int(model_id),
                convex=True,
                is_static=True,
                scale_mult=scale_mult,
            )
        except Exception as e:
            print(f"[pour_beer] skip prop {modelname}/base{model_id}: {e}")
            return None
        if actor is None:
            print(f"[pour_beer] skip prop {modelname}/base{model_id}")
            return None
        actor.set_name(f"bar_{modelname}_{model_id}")
        self._bar_props.append(actor)
        if half_xy is not None:
            self._prop_footprints.append(
                (np.asarray(xy, dtype=float), np.asarray(half_xy, dtype=float))
            )
        return actor

    def _build_bar_props(self, rng=None):
        """Sparse bar décor — keep the tap station clear; optional non-overlap randomize."""
        if rng is None:
            rng = self._layout_rng(202)
        cfg = self._cfg
        randomize = bool(cfg.get("randomize_layout", False))

        # (model, id, half_xy, scale, default_xy, default_yaw, region_xy)
        catalog = [
            ("255_beer_bottle", 0, (0.040, 0.040), 1.00, [-0.50, 0.24], -10,
             ((-0.55, -0.20), (0.16, 0.28))),
            ("001_bottle", 2, (0.035, 0.035), 1.00, [-0.32, 0.24], 8,
             ((-0.40, -0.12), (0.16, 0.28))),
            ("001_bottle", 5, (0.035, 0.035), 1.00, [0.34, 0.24], -8,
             ((0.22, 0.55), (0.16, 0.28))),
            ("255_beer_bottle", 0, (0.040, 0.040), 1.00, [0.52, 0.24], 12,
             ((0.30, 0.55), (0.16, 0.28))),
            ("088_wineglass", 0, (0.040, 0.040), 0.38, [-0.08, 0.22], 15,
             ((-0.25, 0.00), (0.14, 0.26))),
            ("039_mug", 0, (0.045, 0.045), 0.65, [0.20, 0.22], 30,
             ((0.18, 0.40), (0.14, 0.26))),
            ("025_chips-tub", 0, (0.055, 0.045), 1.00, [-0.46, -0.10], -20,
             ((-0.55, -0.28), (-0.22, 0.10))),
            ("025_chips-tub", 2, (0.055, 0.045), 1.00, [0.50, -0.12], 30,
             ((0.30, 0.55), (-0.22, 0.10))),
            ("071_can", 0, (0.035, 0.035), 1.00, [-0.50, -0.22], -35,
             ((-0.55, -0.30), (-0.28, -0.08))),
            ("054_baguette", 2, (0.080, 0.035), 1.00, [-0.52, 0.14], 65,
             ((-0.55, -0.30), (0.05, 0.22))),
        ]

        station_c = self._station_center_xy()
        station_h = np.asarray(self.STATION_HALF_XY, dtype=float)
        blockers = [(station_c, station_h)]

        for model, mid, half, scale, default_xy, default_yaw, region in catalog:
            half = np.asarray(half, dtype=float)
            if randomize:
                xy = self._sample_free_xy(rng, half, blockers, region[0], region[1])
                # Reject if it still clips the station (extra guard).
                if not self._footprint_clear(xy, half, [(station_c, station_h)]):
                    # Park far from station on the region's far edge.
                    x_lo, x_hi = region[0]
                    y_lo, y_hi = region[1]
                    candidates = [
                        np.array([x_lo, y_lo]),
                        np.array([x_lo, y_hi]),
                        np.array([x_hi, y_lo]),
                        np.array([x_hi, y_hi]),
                        np.array([0.5 * (x_lo + x_hi), y_lo]),
                    ]
                    xy = max(
                        candidates,
                        key=lambda p: min(
                            abs(p[0] - station_c[0]) - (half[0] + station_h[0]),
                            abs(p[1] - station_c[1]) - (half[1] + station_h[1]),
                        ),
                    )
                yaw = float(default_yaw) + float(rng.uniform(-25.0, 25.0))
            else:
                xy = np.asarray(default_xy, dtype=float)
                yaw = float(default_yaw)
                if not self._footprint_clear(xy, half, blockers):
                    # Nudge away from station along +/−x.
                    for dx in (0.08, -0.08, 0.14, -0.14, 0.20, -0.20):
                        cand = xy + np.array([dx, 0.0])
                        if self._footprint_clear(cand, half, blockers):
                            xy = cand
                            break

            actor = self._spawn_static_prop(
                model, xy, model_id=mid, yaw_deg=yaw, scale_mult=scale, half_xy=half
            )
            if actor is not None:
                blockers.append((np.asarray(xy, dtype=float), half))

    def _spawn_coaster(self):
        try:
            pose = sapien.Pose(
                [float(self.cup_xy[0]), float(self.cup_xy[1]), self.table_top + 0.001],
                self.GLASS_UPRIGHT_Q,
            )
            coaster = create_actor(
                self, pose=pose, modelname="019_coaster", model_id=0, convex=True, is_static=True
            )
            coaster.set_name("glass_coaster")
            cfg = getattr(coaster, "config", {}) or {}
            ext = np.array(cfg.get("extents", [0.15, 0.008, 0.15]), dtype=float)
            sc = cfg.get("scale", [1, 1, 1])
            sc = float(sc[0] if isinstance(sc, (list, tuple)) else sc)
            self.coaster_top_z = self.table_top + float(ext[1] * sc) + 0.001
        except Exception:
            self.coaster_top_z = self.table_top + 0.008

    def _spawn_glass(self):
        pose = sapien.Pose(
            [float(self.cup_xy[0]), float(self.cup_xy[1]), self.coaster_top_z + 0.001],
            self.GLASS_UPRIGHT_Q,
        )
        self.cup = create_actor(
            self, pose=pose, modelname=self.GLASS_MODEL, model_id=0, convex=True, is_static=True
        )
        self.cup.set_name("beer_glass")
        self._make_glass_transparent(self.cup)

        cfg = getattr(self.cup, "config", {}) or {}
        ext = np.array(cfg.get("extents", [0.08, 0.13, 0.08]), dtype=float)
        sc = cfg.get("scale", [1, 1, 1])
        sc = float(sc[0] if isinstance(sc, (list, tuple)) else sc)
        center = np.array(cfg.get("center", [0.0, 0.0, 0.0]), dtype=float)
        pose_z = float(self.cup.get_pose().p[2])
        half_y = 0.5 * float(ext[1]) * sc
        cy = float(center[1]) * sc
        mesh_bottom_z = pose_z + cy - half_y
        mesh_top_z = pose_z + cy + half_y
        try:
            for c in self.cup.actor.get_components():
                if hasattr(c, "get_global_aabb_fast"):
                    aabb = np.asarray(c.get_global_aabb_fast(), dtype=float)
                    mesh_bottom_z = float(aabb[0, 2])
                    mesh_top_z = float(aabb[1, 2])
                    break
        except Exception:
            pass
        self.cup_inner_r = 0.30 * float(max(ext[0], ext[2]) * sc)
        self.cup_bottom_z = mesh_bottom_z + 0.012
        self.cup_rim_z = mesh_top_z - 0.008
        self.cup_fillable_h = max(0.05, float(self.cup_rim_z - self.cup_bottom_z))

    def _build_tap(self):
        """Chrome draft tower + swan-neck spout over the glass."""
        chrome = self._metallic_material(self.CHROME)
        chrome_d = self._metallic_material(self.CHROME_DARK, roughness=0.35, metallic=0.90)
        wood_base = self._opaque_material(self.BASE_WOOD)
        x, y = float(self.tap_xy[0]), float(self.tap_xy[1])
        z0 = float(self.table_top)

        base = self._add_static_cylinder(
            pose=sapien.Pose([x, y, z0 + 0.5 * self.BASE_H]),
            radius=self.BASE_R,
            half_h=0.5 * self.BASE_H,
            material=wood_base,
            name="beer_tap_base",
            collision=True,
        )
        self._tap_parts.append(base)
        collar = self._add_static_cylinder(
            pose=sapien.Pose([x, y, z0 + self.BASE_H + 0.006]),
            radius=self.BASE_R * 0.62,
            half_h=0.006,
            material=chrome_d,
            name="beer_tap_collar",
            collision=False,
        )
        self._tap_parts.append(collar)

        tower_bottom = z0 + self.BASE_H
        tower = self._add_static_cylinder(
            pose=sapien.Pose([x, y, tower_bottom + 0.5 * self.TOWER_H]),
            radius=self.TOWER_R,
            half_h=0.5 * self.TOWER_H,
            material=chrome,
            name="beer_tap_tower",
            collision=True,
        )
        self._tap_parts.append(tower)

        head_z = tower_bottom + self.TOWER_H + 0.012
        head = self._add_static_box(
            pose=sapien.Pose([x, y - 0.008, head_z]),
            half_size=[0.018, 0.028, 0.014],
            material=chrome_d,
            name="beer_tap_head",
            collision=True,
        )
        self._tap_parts.append(head)

        arm_y = y - 0.5 * self.ARM_LEN - 0.01
        arm_z = head_z
        arm = self._add_static_box(
            pose=sapien.Pose([x, arm_y, arm_z]),
            half_size=[0.011, 0.5 * self.ARM_LEN, 0.011],
            material=chrome,
            name="beer_tap_arm",
            collision=True,
        )
        self._tap_parts.append(arm)

        spout_xy = np.array(
            [float(self.cup_xy[0]), float(self.cup_xy[1]) + 0.012], dtype=float
        )
        spout_root = np.array([x, y - self.ARM_LEN - 0.01, arm_z], dtype=float)
        spout_out = np.array(
            [spout_xy[0], spout_xy[1], arm_z - self.SPOUT_DROP], dtype=float
        )
        spout_mid = 0.5 * (spout_root + spout_out)
        spout_dir = spout_out - spout_root
        tip = self._add_static_box(
            pose=sapien.Pose(spout_mid.tolist()),
            half_size=[
                0.009,
                max(0.012, 0.5 * abs(spout_dir[1])),
                max(0.012, 0.5 * abs(spout_dir[2])),
            ],
            material=chrome_d,
            name="beer_tap_spout",
            collision=False,
        )
        self._tap_parts.append(tip)
        opening = self._add_static_box(
            pose=sapien.Pose(spout_out.tolist()),
            half_size=[self.NOZZLE_R, self.NOZZLE_R, 0.002],
            material=self._metallic_material([0.25, 0.25, 0.28], roughness=0.4),
            name="beer_tap_opening",
            collision=False,
        )
        self._tap_parts.append(opening)

        self.nozzle_outlet_xyz = spout_out.astype(float)
        self.lever_pivot_xyz = np.array([x, y - 0.012, head_z + 0.016], dtype=float)

    # ------------------------------------------------------------------ hinged lever
    def _lever_tip_xyz(self, angle: float | None = None) -> np.ndarray:
        if angle is None:
            angle = float(self.lever_angle)
        ang = float(np.clip(angle, 0.0, self.lever_open_rad))
        pivot = np.asarray(self.lever_pivot_xyz, dtype=float)
        # Closed: +Z. Open: rotate about +X toward −Y.
        dy = -np.sin(ang) * self.LEVER_LEN
        dz = np.cos(ang) * self.LEVER_LEN
        return pivot + np.array([0.0, dy, dz], dtype=float)

    def _lever_open_frac(self, angle: float | None = None) -> float:
        if angle is None:
            angle = float(self.lever_angle)
        span = max(1e-6, float(self.lever_open_rad) - float(self.LEVER_DEADZONE))
        return float(np.clip((angle - self.LEVER_DEADZONE) / span, 0.0, 1.0))

    def _spawn_lever(self):
        """Kinematic wooden handle hinged at the faucet head (pose updated each step)."""
        self._lever_entity = self._remove_entity(self._lever_entity)
        self._lever_comp = None
        wood = self._opaque_material(self.WOOD)
        length = float(self.LEVER_LEN)
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("dynamic")
        builder.add_cylinder_collision(
            pose=sapien.Pose(),
            radius=self.LEVER_R,
            half_length=0.5 * length,
            material=self.scene.default_physical_material,
        )
        builder.add_cylinder_visual(
            pose=sapien.Pose(),
            radius=self.LEVER_R,
            half_length=0.5 * length,
            material=wood,
        )
        builder.add_sphere_visual(
            pose=sapien.Pose([0.5 * length, 0.0, 0.0]),
            radius=self.LEVER_R * 1.35,
            material=wood,
        )
        tip0 = self._lever_tip_xyz(0.0)
        mid0 = 0.5 * (np.asarray(self.lever_pivot_xyz, dtype=float) + tip0)
        # Local +X along lever; upright → +Z world.
        quat0 = t3d.quaternions.axangle2quat([0.0, 1.0, 0.0], -0.5 * np.pi)
        builder.set_initial_pose(sapien.Pose(mid0.tolist(), quat0.tolist()))
        self._lever_entity = builder.build(name="beer_tap_lever")
        for c in self._lever_entity.get_components():
            if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                self._lever_comp = c
                try:
                    c.set_mass(0.05)
                except Exception:
                    pass
                try:
                    c.set_disable_gravity(True)
                except Exception:
                    pass
                c.set_kinematic(True)
                break

    def _apply_lever_pose(self, angle: float):
        """Set lever pose for a hinge angle about +X at the pivot (0 = upright)."""
        ang = float(np.clip(angle, 0.0, self.lever_open_rad))
        self.lever_angle = ang
        self._lever_angle_max = max(self._lever_angle_max, ang)
        if ang >= self.LEVER_OPEN_THRESH:
            self.opened_once = True
        if (
            self.opened_once
            and ang <= self.LEVER_DEADZONE + 0.02
            and self.liquid_level > 0.05
        ):
            self.closed_after_pour = True

        pivot = np.asarray(self.lever_pivot_xyz, dtype=float)
        tip = self._lever_tip_xyz(ang)
        mid = 0.5 * (pivot + tip)
        direction = tip - pivot
        length = float(np.linalg.norm(direction))
        if length < 1e-6:
            direction = np.array([0.0, 0.0, 1.0])
            length = self.LEVER_LEN
        direction = direction / length

        # Cylinder local +X → aim along ``direction``.
        x_axis = np.array([1.0, 0.0, 0.0])
        axis = np.cross(x_axis, direction)
        n = float(np.linalg.norm(axis))
        if n < 1e-8:
            quat = np.array([1.0, 0.0, 0.0, 0.0]) if direction[0] >= 0 else np.array(
                [0.0, 0.0, 1.0, 0.0]
            )
        else:
            axis = axis / n
            rot = float(np.arccos(np.clip(np.dot(x_axis, direction), -1.0, 1.0)))
            quat = t3d.quaternions.axangle2quat(axis, rot)

        pose = sapien.Pose(mid.tolist(), quat.tolist())
        if self._lever_entity is None:
            return
        if self._lever_comp is not None:
            try:
                self._lever_comp.set_kinematic_target(pose)
            except Exception:
                self._lever_entity.set_pose(pose)
        else:
            self._lever_entity.set_pose(pose)
        self.lever_tip_xyz = tip.copy()
        self.touch_xy = tip[:2].copy()
        self.touch_top_z = float(tip[2])

    def _angle_from_tip_point(self, tip_xyz: np.ndarray) -> float:
        """Hinge angle implied by a world tip point near the lever arc."""
        pivot = np.asarray(self.lever_pivot_xyz, dtype=float)
        v = np.asarray(tip_xyz, dtype=float) - pivot
        # Angle from +Z toward −Y in the YZ plane.
        ang = float(np.arctan2(-v[1], max(v[2], 1e-6)))
        return float(np.clip(ang, 0.0, self.lever_open_rad))

    def _drive_lever_from_ee(self):
        """While held, map the hand's TCP onto the lever arc (continuous joint)."""
        if not self._lever_held or self.overflowed:
            return
        ee = self._ee_pos(self.arm)
        # Approximate TCP under the EE for a top-down grasp on the knob.
        tip_est = ee - np.array([0.0, 0.0, self.EE_TO_TCP], dtype=float)
        ang = self._angle_from_tip_point(tip_est)
        # Soft slew so the visual joint tracks the hand smoothly.
        cur = float(self.lever_angle)
        step = 0.12  # rad per physics tick while held
        if abs(ang - cur) <= step:
            self._apply_lever_pose(ang)
        else:
            self._apply_lever_pose(cur + step * np.sign(ang - cur))

    # ------------------------------------------------------------------ fluids
    def _cup_center_xy(self) -> np.ndarray:
        return np.asarray(self.cup_xy, dtype=float)

    def _total_fill(self) -> float:
        return float(self.liquid_level) + float(self.foam_level)

    def _clamped_fill_fracs(self):
        liq = max(0.0, min(1.0, float(self.liquid_level)))
        foam = min(max(0.0, float(self.foam_level)), max(0.0, 1.0 - liq))
        return liq, foam

    def _rebuild_fluids(self, force: bool = False):
        if not getattr(self, "cup_fillable_h", None):
            return
        liq_frac, foam_frac = self._clamped_fill_fracs()
        liq_h = liq_frac * self.cup_fillable_h
        foam_h = foam_frac * self.cup_fillable_h
        top = liq_h + foam_h
        if top > self.cup_fillable_h:
            scale = self.cup_fillable_h / max(top, 1e-6)
            liq_h *= scale
            foam_h *= scale
        liq_half = max(0.002, 0.5 * liq_h) if liq_frac > 1e-4 else 0.0
        foam_half = max(0.002, 0.5 * foam_h) if foam_frac > 1e-4 else 0.0
        # Tiny threshold so beer/foam rise looks continuous (bottle-like), not stepped.
        if (
            not force
            and abs(liq_half - self._liquid_half_h_cached) < 0.00035
            and abs(foam_half - self._foam_half_h_cached) < 0.00035
        ):
            return
        self._liquid_half_h_cached = liq_half
        self._foam_half_h_cached = foam_half
        self._liquid_entity = self._remove_entity(self._liquid_entity)
        self._foam_entity = self._remove_entity(self._foam_entity)
        r = self.cup_inner_r
        if liq_frac > 1e-4:
            self._liquid_entity = self._make_column(
                liq_half, r, self.cup_bottom_z + liq_half, self.BEER_COLOR, "beer_liquid"
            )
        if foam_frac > 1e-4:
            self._foam_entity = self._make_column(
                foam_half,
                r * 0.98,
                self.cup_bottom_z + liq_h + foam_half,
                self.FOAM_COLOR,
                "beer_foam",
            )

    def _sync_stream(self, force: bool = False):
        """Beer cylinder from spout → glass; thickness scales with lever open frac."""
        frac = self._lever_open_frac()
        if (
            not force
            and abs(frac - self._stream_frac_cached) < 0.04
            and ((frac > 0.02) == (self._stream_entity is not None))
        ):
            # Still refresh length as the surface rises.
            if self._stream_entity is None:
                return
        self._stream_frac_cached = frac
        self._stream_entity = self._remove_entity(self._stream_entity)
        if frac < 0.02:
            return
        ox, oy, oz = self.nozzle_outlet_xyz
        liq_frac, foam_frac = self._clamped_fill_fracs()
        surface_z = self.cup_bottom_z + (liq_frac + foam_frac) * self.cup_fillable_h
        z_lo = max(self.cup_bottom_z + 0.01, min(surface_z + 0.01, oz - 0.02))
        half_h = max(0.01, 0.5 * (oz - z_lo))
        z_c = 0.5 * (oz + z_lo)
        radius = self.NOZZLE_R * (0.45 + 0.55 * frac)
        self._stream_entity = self._make_column(
            half_h, radius, z_c, self.STREAM_COLOR, "beer_stream", xy=[ox, oy]
        )

    def _spawn_overflow_stain(self, force: bool = False):
        """Yellow beer puddle on the counter under/around the glass (+ rim drip)."""
        if (not force) and getattr(self, "_stain_entity", None) is not None:
            # Grow the puddle if more beer keeps spilling after the first crest.
            self._rebuild_spill_puddle()
            return
        self._rebuild_spill_puddle()
        print("[pour_beer] OVERFLOW — yellow beer spilled under the glass")

    def _rebuild_spill_puddle(self):
        """Yellow beer puddle around the glass; grows with spill_amount.

        Uses flat box plates (not cylinders) so the spill always reads from the
        top-down cameras, extending well past the coaster.
        """
        self._stain_entity = self._remove_entity(getattr(self, "_stain_entity", None))
        self._drip_entity = self._remove_entity(getattr(self, "_drip_entity", None))
        xy = self._cup_center_xy()
        spill = float(getattr(self, "spill_amount", 0.0))
        # Coaster is ~8 cm radius; puddle must clearly ring it.
        outer = 0.14 + min(0.08, 0.12 * spill)
        half_z = 0.0018 + min(0.0025, 0.004 * spill)

        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        # Flat yellow discs (axis → +Z) so the spill reads as a pooled liquid.
        builder.add_cylinder_visual(
            pose=sapien.Pose([0, 0, 0], self.VERTICAL_CYL_Q),
            radius=float(outer),
            half_length=float(half_z),
            material=self._fluid_material(self.STAIN_COLOR),
        )
        builder.add_cylinder_visual(
            pose=sapien.Pose(
                [outer * 0.50, -outer * 0.28, 0.0003], self.VERTICAL_CYL_Q
            ),
            radius=float(outer * 0.55),
            half_length=float(half_z * 0.9),
            material=self._fluid_material(self.STAIN_COLOR_LOBE),
        )
        builder.add_cylinder_visual(
            pose=sapien.Pose(
                [-outer * 0.42, outer * 0.38, 0.0003], self.VERTICAL_CYL_Q
            ),
            radius=float(outer * 0.48),
            half_length=float(half_z * 0.85),
            material=self._fluid_material(self.STAIN_COLOR_LOBE),
        )
        builder.add_cylinder_visual(
            pose=sapien.Pose(
                [outer * 0.12, outer * 0.52, 0.0003], self.VERTICAL_CYL_Q
            ),
            radius=float(outer * 0.40),
            half_length=float(half_z * 0.8),
            material=self._fluid_material(self.STAIN_COLOR),
        )
        # Above the coaster top so nothing occludes the yellow pool.
        z = float(getattr(self, "coaster_top_z", self.table_top)) + half_z + 0.003
        builder.set_initial_pose(sapien.Pose(p=[float(xy[0]), float(xy[1]), z]))
        self._stain_entity = builder.build(name="beer_overflow_spill")

        # Drip from the rim into the puddle.
        rim_z = float(getattr(self, "cup_rim_z", self.table_top + 0.12))
        drip_xy = [
            float(xy[0]) + float(getattr(self, "cup_inner_r", 0.025)) * 1.15,
            float(xy[1]) - float(getattr(self, "cup_inner_r", 0.025)) * 0.40,
        ]
        z_lo = z + half_z
        z_hi = rim_z - 0.003
        if z_hi > z_lo + 0.015:
            drip_half = 0.5 * (z_hi - z_lo)
            drip_zc = 0.5 * (z_hi + z_lo)
            self._drip_entity = self._make_column(
                drip_half,
                max(0.006, float(self.NOZZLE_R)),
                drip_zc,
                self.STAIN_COLOR_DRIP,
                "beer_overflow_drip",
                xy=drip_xy,
            )
        print(
            f"[pour_beer] spill puddle r≈{outer:.3f} z={z:.3f} "
            f"amount={spill:.3f} drip={self._drip_entity is not None}",
            flush=True,
        )

    def _mark_overflow(self, spilled: float = 0.0):
        first = not self.overflowed
        self.overflowed = True
        self.spill_amount = float(getattr(self, "spill_amount", 0.0)) + max(
            0.0, float(spilled)
        )
        # Keep glass contents at the rim; the puddle shows what went over.
        if self._total_fill() > 1.0:
            self.foam_level = max(0.0, 1.0 - float(self.liquid_level))
            if self.liquid_level > 1.0:
                self.liquid_level = 1.0
                self.foam_level = 0.0
        self._spawn_overflow_stain(force=first)
        self._rebuild_fluids(force=True)
        if first:
            # Cut the tap once it crests — spill puddle already marks the fail.
            self._lever_held = False
            self._apply_lever_pose(0.0)

    def _step_fluids(self):
        """Bottle-like fill: while lever open, beer+foam rise continuously with rate∝angle."""
        frac = self._lever_open_frac()
        if frac > 1e-4 and not self.overflowed:
            d = float(self.pour_rate) * frac
            add_liq = d
            add_foam = d * self.foam_gain
            new_liq = float(self.liquid_level) + add_liq
            new_foam = float(self.foam_level) + add_foam
            if new_liq + new_foam >= float(self.overflow_level) - 1e-6:
                room = max(0.0, float(self.overflow_level) - self._total_fill())
                spilled = max(0.0, (add_liq + add_foam) - room)
                if room > 1e-6:
                    liq_share = add_liq / max(add_liq + add_foam, 1e-9)
                    self.liquid_level = min(1.0, self.liquid_level + room * liq_share)
                    self.foam_level = max(0.0, float(self.overflow_level) - self.liquid_level)
                self._mark_overflow(spilled=max(spilled, 0.08))
            else:
                self.liquid_level = min(1.0, new_liq)
                self.foam_level = max(0.0, new_foam)
        elif frac > 1e-4 and self.overflowed:
            # Extra beer after crest keeps growing the counter puddle.
            d = float(self.pour_rate) * frac
            self.spill_amount = float(self.spill_amount) + d * (1.0 + self.foam_gain)
            self._rebuild_spill_puddle()
        elif (not self.overflowed) and self.foam_level > 1e-6:
            self.foam_level = max(0.0, self.foam_level - self.foam_decay)
        if (not self.overflowed) and self._total_fill() >= self.overflow_level + 1e-4:
            self._mark_overflow(spilled=0.08)
        self._rebuild_fluids(force=False)
        self._sync_stream(force=False)

    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
        if not getattr(self, "_loaded", False):
            return
        self._drive_lever_from_ee()
        self._step_fluids()

    def _idle_steps(self, n_steps: int, until=None):
        save_freq = self.save_freq if self.save_freq is not None else 15
        for i in range(int(n_steps)):
            self._update_kinematic_tasks()
            self.scene.step()
            if until is not None and until():
                break
            if self.render_freq and i % max(1, int(self.render_freq)) == 0:
                self._update_render()
                if hasattr(self, "viewer") and self.viewer is not None:
                    self.viewer.render()
            if self.save_freq is not None and i % save_freq == 0:
                self._take_picture()

    # ------------------------------------------------------------------ arm / lever control
    def _ee_pos(self, arm: ArmTag) -> np.ndarray:
        p = self.get_arm_pose(str(arm))
        return np.asarray(p[:3], dtype=float)

    def _lever_ee_pose(self, angle: float, z_above: float):
        tip = self._lever_tip_xyz(angle)
        return [
            float(tip[0]),
            float(tip[1]),
            float(tip[2] + z_above + self.EE_TO_TCP),
            *GRASP_DIRECTION_DIC["top_down"],
        ]

    def _move_ok(self, arm: ArmTag, dx=0.0, dy=0.0, dz=0.0) -> bool:
        self.plan_success = True
        self.move(
            self.move_by_displacement(
                arm, x=float(dx), y=float(dy), z=float(dz), move_axis="world"
            )
        )
        ok = bool(self.plan_success)
        if not ok:
            self.plan_success = True
        return ok

    def _grasp_lever(self, arm: ArmTag) -> bool:
        """Approach the upright tip and latch a hand-drive on the lever joint."""
        self._lever_held = False
        self.plan_success = True
        self.move(self.close_gripper(arm))
        ang0 = float(self.lever_angle)
        self.move(self.move_to_pose(arm, self._lever_ee_pose(ang0, self.LEVER_HOVER)))
        if not self.plan_success:
            print("[pour_beer] lever hover failed — continuing with forced hold")
            self.plan_success = True
        self.move(self.move_to_pose(arm, self._lever_ee_pose(ang0, self.LEVER_GRASP_Z)))
        if not self.plan_success:
            self.plan_success = True
        self._idle_steps(2)
        self._lever_held = True
        return True

    def _sweep_lever_to(
        self,
        arm: ArmTag,
        target_frac: float,
        n_steps: int = 10,
        stop_on_foam: bool = False,
    ):
        """Walk the EE along the lever tip arc; joint angle tracks the hand.

        When closing, the joint angle is stepped down *before* each EE move so
        flow drops with the hinge (long IK while still open was overflowing).
        """
        self._lever_held = True
        target_ang = float(np.clip(target_frac, 0.0, 1.0)) * float(self.lever_open_rad)
        start = float(self.lever_angle)
        closing = target_ang < start - 1e-3
        n = max(2, int(n_steps))
        if closing:
            n = min(n, 5)
        for i in range(1, n + 1):
            if self.overflowed:
                break
            if stop_on_foam and (
                self.foam_level >= self.expert_foam_pause
                or self._total_fill() >= self.safe_total
            ):
                break
            ang = start + (target_ang - start) * (i / n)
            if closing:
                # Cut flow with the joint first, then the hand follows the tip.
                self._apply_lever_pose(ang)
            tip = self._lever_tip_xyz(ang)
            goal_ee = tip + np.array(
                [0.0, 0.0, self.EE_TO_TCP + self.LEVER_GRASP_Z], dtype=float
            )
            # A few short nudges beat one long IK that pours the whole way.
            for _ in range(3 if closing else 2):
                if self.overflowed:
                    break
                ee = self._ee_pos(arm)
                delta = goal_ee - ee
                if float(np.linalg.norm(delta)) < 0.028:
                    break
                step = 0.055 if closing else 0.035
                self._move_ok(
                    arm,
                    dx=float(np.clip(delta[0], -step, step)),
                    dy=float(np.clip(delta[1], -step, step)),
                    dz=float(np.clip(delta[2], -step, step)),
                )
            if self.overflowed:
                break
            if not closing:
                self._apply_lever_pose(ang)
            # Short dwell so cameras catch intermediate hinge poses.
            self._idle_steps(2)
            if self.overflowed:
                break
        if closing and not self.overflowed:
            self._apply_lever_pose(target_ang)
        print(
            f"[pour_beer] lever→{self._lever_open_frac():.2f} "
            f"ang={np.degrees(self.lever_angle):.1f}° "
            f"liq={self.liquid_level:.2f} foam={self.foam_level:.2f}"
        )

    def _release_lever(self, arm: ArmTag):
        self._lever_held = False
        self._move_ok(arm, dz=0.06)
        self._idle_steps(2)

    # ------------------------------------------------------------------ expert
    def play_once(self):
        arm = self.arm
        self.plan_success = True
        self._lever_held = False
        self.move(self.close_gripper(arm))

        # 1) Grasp the upright lever and open it gradually by hand (staged).
        self._grasp_lever(arm)
        for open_frac in (0.40, 0.70):
            if self.overflowed:
                break
            self._sweep_lever_to(
                arm, target_frac=open_frac, n_steps=8, stop_on_foam=True
            )
            if self.foam_level >= self.expert_foam_pause or self._total_fill() >= self.safe_total:
                break

        # 2) Pour with foam management — foam keeps rising while the lever is open
        #    (bottle-like). Close by sweeping the joint back with the hand.
        # Scale wait/cycles when pour_rate is below the nominal (randomized rates).
        rate_scale = float(self.POUR_RATE) / max(float(self.pour_rate), 1e-6)
        rate_scale = float(np.clip(rate_scale, 0.75, 2.2))
        pour_idle = int(round(150 * rate_scale))
        max_cycles = int(round(14 * max(1.0, rate_scale)))
        close_at = max(0.10, float(self.target_liquid) - 0.02)
        for cycle in range(max_cycles):
            if self.overflowed or self.liquid_level >= close_at:
                break
            if self._lever_open_frac() < 0.20:
                if self.liquid_level < close_at - 0.20:
                    reopen = 0.70
                elif self.liquid_level < close_at - 0.08:
                    reopen = 0.45
                else:
                    reopen = 0.30
                # Reach the pour angle first; foam is managed in the idle below
                # (stopping mid-open on residual foam caused open/close thrashing).
                self._sweep_lever_to(
                    arm, target_frac=reopen, n_steps=7, stop_on_foam=False
                )

            self._idle_steps(
                pour_idle,
                until=lambda: (
                    self.overflowed
                    or self.liquid_level >= close_at
                    or self.foam_level >= self.expert_foam_pause
                    or self._total_fill() >= self.safe_total
                ),
            )
            print(
                f"[pour_beer] cycle={cycle} liq={self.liquid_level:.2f} "
                f"foam={self.foam_level:.2f} total={self._total_fill():.2f} "
                f"lever={self._lever_open_frac():.2f} overflow={self.overflowed}"
            )
            if self.overflowed or self.liquid_level >= close_at:
                break

            need_close = (
                self.foam_level >= self.expert_foam_pause
                or self._total_fill() >= self.safe_total
            )
            if need_close:
                self._sweep_lever_to(arm, target_frac=0.0, n_steps=4)
                if self.overflowed:
                    break
                self._idle_steps(
                    140,
                    until=lambda: (
                        self.foam_level <= self.expert_foam_resume or self.overflowed
                    ),
                )

        # 3) Ensure the lever is closed by hand.
        if (not self.overflowed) and self._lever_open_frac() > 0.05:
            self._sweep_lever_to(arm, target_frac=0.0, n_steps=5)
        self._release_lever(arm)
        self._idle_steps(50, until=lambda: self.foam_level < 0.05 or self.overflowed)

        if self.overflowed:
            self.plan_success = False
        elif self.check_success():
            self.plan_success = True

        self.info["info"] = {
            "{A}": "beer tap",
            "{B}": f"{self.GLASS_MODEL}/base0",
            "{C}": "tap lever",
            "{a}": str(arm),
        }
        return self.info

    def check_success(self):
        if self.overflowed:
            return False
        liquid_ok = self.liquid_level >= self.target_liquid - self.full_liquid_tol
        foam_ok = self._total_fill() < self.overflow_level - 0.02
        closed = (
            self._lever_open_frac() < 0.05
            and self.closed_after_pour
            and self.opened_once
        )
        return bool(liquid_ok and foam_ok and closed)

    @property
    def tab_open(self) -> bool:
        """Compatibility: treat lever past deadzone as open."""
        return self._lever_open_frac() > 0.02

    def get_obs(self):
        obs = super().get_obs()
        obs["beer_pour"] = {
            "liquid_level": float(self.liquid_level),
            "foam_level": float(self.foam_level),
            "total_fill": float(self._total_fill()),
            "lever_angle": float(self.lever_angle),
            "lever_open_frac": float(self._lever_open_frac()),
            "tab_open": bool(self.tab_open),
            "overflowed": bool(self.overflowed),
            "spill_amount": float(getattr(self, "spill_amount", 0.0)),
            "opened_once": bool(self.opened_once),
            "closed_after_pour": bool(self.closed_after_pour),
            "pour_rate": float(getattr(self, "pour_rate", self.POUR_RATE)),
            "foam_gain": float(getattr(self, "foam_gain", self.FOAM_GAIN)),
            "foam_decay": float(getattr(self, "foam_decay", self.FOAM_DECAY)),
            "cup_xy": np.asarray(self.cup_xy, dtype=float).tolist(),
            "tap_xy": np.asarray(self.tap_xy, dtype=float).tolist(),
            "scene_id": int(getattr(self, "scene_id", 0)),
        }
        return obs
