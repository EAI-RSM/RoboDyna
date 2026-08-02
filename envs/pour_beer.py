"""Pour beer from a bar tap into a beer mug (KitchenS).

Chrome draft tower with a springy wooden lever. The handle only moves when the
robot gripper presses the knob along the hinge arc; releasing contact lets a
spring return it upright. Beer stream thickness and fill/foam rates scale with
how far the handle is turned. Overflow fails with a yellow stain.

The drinking vessel is a procedural glass beer mug (body + D-handle). Glass
materials mirror ``measure_ingredient``'s jar: transmission cylinder for demo
cameras, hollow alpha shell for the interactive SAPIEN viewer.

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
    """Drive the tap lever by hand to fill the mug without foam overflow."""

    GLASS_MODEL = "beer_mug"
    GLASS_UPRIGHT_Q = [0.70710678, 0.70710678, 0.0, 0.0]
    # Match measure_ingredient jar cylinder orientations.
    UPRIGHT_CYL_Q = [0.70710678, 0.0, -0.70710678, 0.0]
    VERTICAL_CYL_Q = [0.70710678, 0.0, 0.70710678, 0.0]

    # Procedural beer-mug geometry (meters). Reference: classic seidel mug —
    # tall body, thick foot slightly wider than the rim, D-handle on +X.
    MUG_INNER_R = 0.034
    MUG_WALL_T = 0.0035
    MUG_HEIGHT = 0.135
    MUG_BOTTOM_T = 0.014
    MUG_BASE_R = 0.042
    MUG_RIM_LIP = 0.0025
    MUG_HANDLE_REACH = 0.034
    MUG_HANDLE_THICK = 0.010
    MUG_FACET_FRAC = 0.62  # vertical panels cover lower ~2/3 of the wall

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
    LEVER_DEADZONE = 0.03  # rad — small; past this, flow starts promptly
    LEVER_OPEN_THRESH = 0.18  # rad — counts as "opened" for success
    # Spring return (rad/tick) when the gripper is not pressing the knob.
    LEVER_RETURN_STEP = 0.055
    # Gripper↔knob engagement (pressure proxy + contact).
    LEVER_ARC_RADIAL_TOL = 0.045  # m; |r_yz − LEVER_LEN|
    LEVER_ARC_X_TOL = 0.040  # m; lateral off the hinge plane
    LEVER_CONTACT_R = 0.055  # m; TCP near current tip also counts
    LEVER_TRACK_STEP = 0.10  # rad/tick while pressed (follows hand)
    LEVER_CONTACT_FORCE_GAIN = 0.002  # rad boost per Newton of contact
    # Flow vs open fraction: near-linear (exp≈1). Mild floor so the first
    # meaningful crack of the tap already pours instead of a long dead start.
    FLOW_START_FRAC = 0.22
    FLOW_CURVE_EXP = 1.0
    FLOW_RATE_SCALE = 1.55
    # Stream cylinder radius at trickle → full open (meters).
    STREAM_R_MIN = 0.0025
    STREAM_R_MAX = 0.014

    EE_TO_TCP = 0.12
    LEVER_HOVER = 0.05
    LEVER_GRASP_Z = 0.0

    TARGET_LIQUID = 0.90
    FULL_LIQUID_TOL = 0.05
    # Max rate at full open (per physics step); scales with lever angle.
    # Keep modest — long IK arcs while open still advance fill every tick.
    POUR_RATE = 0.00055
    # Keep foam below beer so amber level is visible early (not foam-only).
    # Slightly higher gain → hits pause sooner → more stop-and-pour cycles.
    FOAM_GAIN = 0.60
    FOAM_DECAY = 0.0045
    # Fraction of collapsing foam that becomes beer (modest — no end surge).
    FOAM_TO_LIQUID = 0.28
    # Per-episode sample ranges when randomize_rates / [lo,hi] yaml values are used.
    POUR_RATE_RANGE = (0.00040, 0.00075)
    FOAM_GAIN_RANGE = (0.45, 0.80)
    FOAM_DECAY_RANGE = (0.0035, 0.0060)
    OVERFLOW_LEVEL = 1.0
    EXPERT_FOAM_PAUSE = 0.16
    EXPERT_FOAM_RESUME = 0.09
    SAFE_TOTAL = 0.90
    # Tap must sit idle this many consecutive sim steps before success.
    # Long enough that mid-pour foam pauses (spring-shut) don't latch success
    # the instant liquid briefly sits in-band while foam is still collapsing.
    TAP_SETTLE_STEPS = 36
    TAP_IDLE_ANGLE = 0.02  # rad — upright enough to count as closed
    TAP_IDLE_VEL = 0.012  # rad/step — not still springing
    # Liquid must not rise more than this while idle (blocks foam→beer "filling").
    LIQUID_STABLE_EPS = 1e-5

    # Non-overlap layout (axis-aligned footprint half-sizes, meters).
    LAYOUT_MARGIN = 0.030
    # Reserved clear zone covering glass + tap base + lever arc.
    STATION_HALF_XY = (0.10, 0.16)
    STATION_X_RANGE = (0.06, 0.16)
    CUP_Y_RANGE = (-0.14, -0.05)
    PROP_X_RANGE = (-0.55, 0.55)
    PROP_Y_RANGE = (-0.26, 0.28)

    # Beer/foam must stay near-opaque: the glass shell occludes translucent
    # interiors via depth/alpha sorting (stream outside the cup stays visible).
    BEER_COLOR = [0.80, 0.52, 0.10, 1.0]
    FOAM_COLOR = [0.98, 0.96, 0.90, 1.0]
    BEER_COLOR_PLAIN = [0.86, 0.58, 0.12, 1.0]
    FOAM_COLOR_PLAIN = [0.99, 0.97, 0.92, 1.0]
    BEER_MENISCUS = [0.92, 0.72, 0.28, 1.0]
    STREAM_COLOR = [0.86, 0.58, 0.12, 1.0]
    STAIN_COLOR = [0.95, 0.78, 0.05, 1.0]
    STAIN_COLOR_LOBE = [0.93, 0.70, 0.04, 1.0]
    STAIN_COLOR_DRIP = [0.90, 0.68, 0.08, 0.95]
    # Glass RGBA lives in ``_mug_glass_material`` (jar structure, darker blue).
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
        self._lever_pressed = False
        self._lever_inhibit_press = False
        self._lever_ang_vel = 0.0
        self._tap_idle_steps = 0
        self._liquid_stable_steps = 0
        self._liquid_level_prev = 0.0
        self._stream_frac_cached = -1.0
        self.spill_amount = 0.0
        self.cup = None
        self.mug_visual = None
        self._mug_visual_hollow = False
        self.table_top = 0.74
        self.coaster_top_z = 0.75
        # Kept for optional prop/stream overrides; mug glass ignores this flag
        # (same as measure_ingredient jar — transmission vs hollow viewer shell).
        self._plain_glass = bool(self._cfg.get("plain_glass", False))

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

    def _set_camera_pose(self, camera, camera_pos, forward, left):
        camera_pos = np.asarray(camera_pos, dtype=np.float64)
        forward = np.asarray(forward, dtype=np.float64)
        forward = forward / np.linalg.norm(forward)
        left = np.asarray(left, dtype=np.float64)
        left = left / np.linalg.norm(left)
        up = np.cross(forward, left)
        m = np.eye(4)
        m[:3, :3] = np.stack([forward, left, up], axis=1)
        m[:3, 3] = camera_pos
        camera.entity.set_pose(sapien.Pose(m))

    def _configure_observer_camera(self):
        """Restore previous observer framing (pos + look-at geometry)."""
        cams = getattr(self, "cameras", None)
        if cams is None or getattr(cams, "observer_camera", None) is None:
            return
        camera_pos = np.array([0.10, -0.55, 1.30], dtype=np.float64)
        look_at = np.array([0.10, 0.02, 0.92], dtype=np.float64)
        forward = look_at - camera_pos
        left = np.cross(np.array([0.0, 0.0, 1.0], dtype=np.float64), forward)
        if float(np.linalg.norm(left)) < 1e-6:
            left = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
        self._set_camera_pose(cams.observer_camera, camera_pos, forward, left)

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
        """Opaque fluid (no transmission) so columns read inside alpha glass."""
        c = list(rgba)
        if len(c) == 3:
            c = c + [1.0]
        # Force a high alpha — translucent interiors vanish behind the cup mesh.
        c[3] = max(0.92, float(c[3]))
        mat = sapien.render.RenderMaterial(base_color=c)
        try:
            mat.set_transmission(0.0)
            mat.set_transmission_roughness(1.0)
            mat.set_roughness(0.35)
            mat.set_metallic(0.0)
        except Exception:
            mat.roughness = 0.35
            mat.metallic = 0.0
        try:
            mat.set_ior(1.0)
        except Exception:
            pass
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
        """Visual cylinder via actor builder (stream / spill drip outside the cup)."""
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

    def _make_cup_fluid(self, half_h, radius, z_bottom, rgba, name, meniscus_rgba=None):
        """Beer/foam column inside the mug.

        Expert / demo cameras: opaque column at the inner radius under the
        transmission wall (same as measure_ingredient oil). Interactive hollow
        shell: fully opaque viewer material so the level reads through alpha glass.
        """
        xy = self._cup_center_xy()
        hh = max(0.0025, float(half_h))
        r = float(radius)
        viewer_shell = bool(getattr(self, "_mug_visual_hollow", False))
        if viewer_shell:
            mat = self._viewer_beer_material(rgba)
        else:
            mat = self._fluid_material(rgba)
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        builder.add_cylinder_visual(
            pose=sapien.Pose([0, 0, 0], self.VERTICAL_CYL_Q),
            radius=r,
            half_length=hh,
            material=mat,
        )
        if meniscus_rgba is not None and hh > 0.003 and not viewer_shell:
            men_half = 0.0018
            builder.add_cylinder_visual(
                pose=sapien.Pose(
                    [0.0, 0.0, hh - men_half], self.VERTICAL_CYL_Q
                ),
                radius=r * 1.06,
                half_length=men_half,
                material=self._fluid_material(meniscus_rgba),
            )
        builder.set_initial_pose(
            sapien.Pose(p=[float(xy[0]), float(xy[1]), float(z_bottom) + hh])
        )
        return builder.build(name=name)

    def _beer_rgba(self):
        # Hollow viewer shell: brighter opaque amber (same role as plain colors).
        if bool(getattr(self, "_mug_visual_hollow", False)):
            return list(self.BEER_COLOR_PLAIN)
        return list(self.BEER_COLOR)

    def _foam_rgba(self):
        if bool(getattr(self, "_mug_visual_hollow", False)):
            return list(self.FOAM_COLOR_PLAIN)
        return list(self.FOAM_COLOR)

    def _mug_glass_material(self, viewer_shell: bool = False):
        """Glass for the beer mug — measure_ingredient jar structure, darker blue tint.

        Demo cameras use transmission glass. The interactive SAPIEN viewer does
        not composite opaque beer behind transmission materials, so the viewer
        shell uses plain alpha glass — same trick as the measure jar / trap_bug.
        """
        if viewer_shell:
            # Hollow viewer shell: darker / bluer than the jar default, slightly
            # higher alpha so the mug reads clearly against the bar.
            glass = sapien.render.RenderMaterial(
                base_color=[0.685, 0.804, 0.958, 0.28]
            )
            try:
                glass.set_transmission(0.0)
                glass.set_transmission_roughness(1.0)
                glass.set_roughness(0.10)
                glass.set_metallic(0.0)
            except Exception:
                glass.roughness = 0.10
                glass.metallic = 0.0
            try:
                glass.set_ior(1.0)
            except Exception:
                pass
            return glass

        # Expert / demo transmission glass — darker blue tint, a bit more presence.
        glass = sapien.render.RenderMaterial(base_color=[0.734, 0.846, 0.972, 0.14])
        try:
            glass.set_transmission(1.0)
            glass.set_transmission_roughness(0.0)
            glass.set_roughness(0.04)
            glass.set_metallic(0.0)
        except Exception:
            pass
        try:
            glass.set_ior(1.0)
        except Exception:
            pass
        return glass

    def _viewer_beer_material(self, rgba):
        """Fully opaque beer/foam for viewer compositing through alpha glass walls."""
        c = list(rgba)
        if len(c) == 3:
            c = c + [1.0]
        c[3] = 1.0
        mat = sapien.render.RenderMaterial(base_color=c)
        try:
            mat.set_transmission(0.0)
            mat.set_transmission_roughness(1.0)
            mat.set_roughness(0.22)
            mat.set_metallic(0.0)
        except Exception:
            mat.roughness = 0.22
            mat.metallic = 0.0
        try:
            mat.set_ior(1.0)
        except Exception:
            pass
        return mat

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
        self.flow_rate_scale = float(cfg.get("flow_rate_scale", self.FLOW_RATE_SCALE))

        side, cup_y, tap_dy = self._resolve_station_layout(cfg, rng)
        self.arm = ArmTag("right" if side >= 0 else "left")
        self.arm_side = str(self.arm)
        # Match other household interactives: pre-select the working arm for teleop.
        self._interactive_selected_arms = (self.arm_side,)
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
        self._lever_pressed = False
        self._lever_inhibit_press = False
        self._lever_ang_vel = 0.0
        self._tap_idle_steps = 0
        self._liquid_stable_steps = 0
        self._liquid_level_prev = 0.0
        self.spill_amount = 0.0
        self._bar_props = []
        self._prop_footprints = []
        self.mug_visual = self._remove_entity(getattr(self, "mug_visual", None))
        self._mug_visual_hollow = False
        if getattr(self, "cup", None) is not None:
            try:
                self.scene.remove_entity(self.cup.actor if hasattr(self.cup, "actor") else self.cup)
            except Exception:
                pass
            self.cup = None

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

    def _mug_outer_r(self) -> float:
        return float(self.MUG_INNER_R + self.MUG_WALL_T)

    def _attach_mug_handle(self, render_body, glass, outer_r, h, bottom_t):
        """D-shaped glass handle on +X (visual only — no collision)."""
        thick = float(self.MUG_HANDLE_THICK)
        reach = float(self.MUG_HANDLE_REACH)
        # Attach just below rim and just above the thick foot.
        z_top = float(h) - 0.016
        z_bot = float(bottom_t) + 0.018
        z_mid = 0.5 * (z_top + z_bot)
        half_v = 0.5 * max(0.02, z_top - z_bot)
        x0 = float(outer_r) - 0.001
        x1 = float(outer_r) + reach

        # Top / bottom stubs into the wall + outer vertical post.
        for z, name_half in ((z_top, 0.5 * thick), (z_bot, 0.5 * thick)):
            stub = sapien.render.RenderShapeBox(
                [0.5 * reach, 0.5 * thick, name_half],
                glass,
            )
            stub.set_local_pose(
                sapien.Pose([x0 + 0.5 * reach, 0.0, z])
            )
            render_body.attach(stub)

        post = sapien.render.RenderShapeBox(
            [0.5 * thick, 0.5 * thick, half_v],
            glass,
        )
        post.set_local_pose(sapien.Pose([x1, 0.0, z_mid]))
        render_body.attach(post)

        # Mild outer rounding at the corners of the D.
        for z in (z_top, z_bot):
            knob = sapien.render.RenderShapeSphere(0.55 * thick, glass)
            knob.set_local_pose(sapien.Pose([x1, 0.0, z]))
            render_body.attach(knob)

    def _attach_mug_facets(self, render_body, glass, outer_r, h, bottom_t):
        """Subtle vertical seidel panels on the lower body (visual only)."""
        facet_h = float(self.MUG_FACET_FRAC) * (float(h) - float(bottom_t))
        if facet_h < 0.02:
            return
        wall_half = 0.5 * facet_h
        wall_z = float(bottom_t) + wall_half
        n_seg = 10
        wall_t = 0.0012
        wall_radius = float(outer_r) + 0.0006
        tangent_half = wall_radius * np.tan(np.pi / n_seg) * 0.72
        for ang in np.linspace(0.0, 2.0 * np.pi, n_seg, endpoint=False):
            # Leave a gap around the +X handle attachment.
            if abs(((ang + np.pi) % (2.0 * np.pi)) - np.pi) < 0.55:
                continue
            px = float(wall_radius * np.cos(ang))
            py = float(wall_radius * np.sin(ang))
            yaw = float(ang + 0.5 * np.pi)
            q = [
                float(np.cos(0.5 * yaw)),
                0.0,
                0.0,
                float(np.sin(0.5 * yaw)),
            ]
            panel = sapien.render.RenderShapeBox(
                [float(tangent_half), float(0.5 * wall_t), float(wall_half)],
                glass,
            )
            panel.set_local_pose(sapien.Pose([px, py, wall_z], q))
            render_body.attach(panel)

    def _build_mug_visual(self, hollow: bool = False):
        """Beer-mug visual. ``hollow=True`` for SAPIEN viewer (open interior).

        Camera / expert demos keep the smooth solid transmission cylinder (looks
        correct in offline render). The interactive viewer treats that cylinder
        as an opaque volume, so viewer mode uses a thin alpha-glass shell instead
        — same pattern as ``measure_ingredient._build_jar_visual``.
        """
        self.mug_visual = self._remove_entity(getattr(self, "mug_visual", None))
        if self.cup is None:
            return

        outer_r = self._mug_outer_r()
        inner_r = float(self.MUG_INNER_R)
        h = float(self.MUG_HEIGHT)
        bottom_t = float(self.MUG_BOTTOM_T)
        base_r = float(self.MUG_BASE_R)
        upright_q = list(self.UPRIGHT_CYL_Q)
        wall_h = h - bottom_t
        wall_half = wall_h * 0.5
        wall_z = bottom_t + wall_half
        glass = self._mug_glass_material(viewer_shell=bool(hollow))
        pose = self.cup.get_pose()

        vis = sapien.Entity()
        vis.set_name("beer_mug_visual")
        vis.set_pose(pose)
        render_body = sapien.render.RenderBodyComponent()

        # Thick foot — slightly wider than the body (classic mug base).
        foot = sapien.render.RenderShapeCylinder(
            radius=base_r,
            half_length=max(0.002, bottom_t * 0.45),
            material=glass,
        )
        foot.set_local_pose(
            sapien.Pose([0.0, 0.0, bottom_t * 0.45], upright_q)
        )
        render_body.attach(foot)

        floor = sapien.render.RenderShapeCylinder(
            radius=outer_r * 0.98,
            half_length=max(0.0015, bottom_t * 0.35),
            material=glass,
        )
        floor.set_local_pose(
            sapien.Pose([0.0, 0.0, bottom_t * 0.55], upright_q)
        )
        render_body.attach(floor)

        if hollow:
            # Thin faceted glass shell — empty inside so beer level is visible.
            wall_t = 0.0024
            n_seg = 36
            wall_radius = outer_r - 0.5 * wall_t
            tangent_half = wall_radius * np.tan(np.pi / n_seg) * 1.03
            for ang in np.linspace(0.0, 2.0 * np.pi, n_seg, endpoint=False):
                px = float(wall_radius * np.cos(ang))
                py = float(wall_radius * np.sin(ang))
                yaw = float(ang + 0.5 * np.pi)
                q = [
                    float(np.cos(0.5 * yaw)),
                    0.0,
                    0.0,
                    float(np.sin(0.5 * yaw)),
                ]
                panel = sapien.render.RenderShapeBox(
                    [float(tangent_half), float(0.5 * wall_t), float(wall_half)],
                    glass,
                )
                panel.set_local_pose(sapien.Pose([px, py, wall_z], q))
                render_body.attach(panel)
        else:
            wall = sapien.render.RenderShapeCylinder(
                radius=outer_r,
                half_length=wall_half,
                material=glass,
            )
            wall.set_local_pose(sapien.Pose([0.0, 0.0, wall_z], upright_q))
            render_body.attach(wall)

        # Slight rim lip.
        rim = sapien.render.RenderShapeCylinder(
            radius=outer_r + float(self.MUG_RIM_LIP),
            half_length=0.0022,
            material=glass,
        )
        rim.set_local_pose(sapien.Pose([0.0, 0.0, h - 0.0022], upright_q))
        render_body.attach(rim)

        self._attach_mug_facets(render_body, glass, outer_r, h, bottom_t)
        self._attach_mug_handle(render_body, glass, outer_r, h, bottom_t)

        vis.add_component(render_body)
        self.scene.add_entity(vis)
        self.mug_visual = vis
        self._mug_visual_hollow = bool(hollow)
        # Keep cup_inner_r in sync for fluid / spill helpers.
        self.cup_outer_r = outer_r
        self.cup_inner_r = inner_r

    def use_viewer_hollow_mug(self):
        """Swap to hollow alpha-glass shell for interactive SAPIEN viewer only."""
        self._build_mug_visual(hollow=True)
        self._rebuild_fluids(force=True)
        print(
            "[pour_beer] viewer mug: hollow alpha-glass shell "
            "(beer visible from the side)"
        )

    def _spawn_glass(self):
        """Procedural glass beer mug on the coaster (collision + glass visual).

        Default visual is the smooth transmission cylinder (demo cameras).
        Interactive viewer calls ``use_viewer_hollow_mug()`` after setup —
        same path as ``measure_ingredient`` / ``interactive_measure_ingredient``.
        """
        x, y = float(self.cup_xy[0]), float(self.cup_xy[1])
        z0 = float(getattr(self, "coaster_top_z", self.table_top)) + 0.001
        outer_r = self._mug_outer_r()
        h = float(self.MUG_HEIGHT)
        bottom_t = float(self.MUG_BOTTOM_T)

        # Solid cylinder collision (handle is visual-only so pour clearance
        # and stream aiming stay clean).
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        builder.add_cylinder_collision(
            pose=sapien.Pose([0.0, 0.0, h * 0.5], self.VERTICAL_CYL_Q),
            radius=float(outer_r),
            half_length=float(h * 0.5),
            material=self.scene.default_physical_material,
        )
        # Wider foot collision so the thick base sits stably on the coaster.
        builder.add_cylinder_collision(
            pose=sapien.Pose(
                [0.0, 0.0, bottom_t * 0.45], self.VERTICAL_CYL_Q
            ),
            radius=float(self.MUG_BASE_R),
            half_length=float(max(0.002, bottom_t * 0.45)),
            material=self.scene.default_physical_material,
        )
        builder.set_initial_pose(sapien.Pose([x, y, z0]))
        entity = builder.build(name="beer_mug")
        try:
            entity.set_name("beer_mug")
        except Exception:
            pass

        # Lightweight Actor-like holder so existing get_pose / set_name call sites work.
        class _MugActor:
            def __init__(self, ent):
                self.actor = ent
                self.config = {
                    "scale": [1.0, 1.0, 1.0],
                    "extents": [2.0 * outer_r, h, 2.0 * outer_r],
                    "center": [0.0, 0.5 * h, 0.0],
                }

            def get_pose(self):
                return self.actor.get_pose()

            def set_name(self, name):
                try:
                    self.actor.set_name(name)
                except Exception:
                    pass

        self.cup = _MugActor(entity)
        self.cup.set_name("beer_mug")
        self._build_mug_visual(hollow=False)

        seat_z = z0
        self.cup_outer_r = outer_r
        self.cup_inner_r = float(self.MUG_INNER_R)
        self.cup_bottom_z = seat_z + bottom_t
        self.cup_rim_z = seat_z + h
        self.cup_fillable_h = max(0.05, float(self.cup_rim_z - self.cup_bottom_z))
        print(
            f"[pour_beer] mug fill seat={seat_z:.3f} bottom={self.cup_bottom_z:.3f} "
            f"rim={self.cup_rim_z:.3f} h={self.cup_fillable_h:.3f} "
            f"inner_r={self.cup_inner_r:.3f} outer_r={self.cup_outer_r:.3f}",
            flush=True,
        )

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

    def _flow_frac(self, angle: float | None = None) -> float:
        """Pour strength vs lever open fraction (linear + small start floor)."""
        frac = self._lever_open_frac(angle)
        if frac <= 0.0:
            return 0.0
        exp = float(getattr(self, "FLOW_CURVE_EXP", 1.0))
        shaped = float(frac ** exp) if abs(exp - 1.0) > 1e-6 else float(frac)
        start = float(getattr(self, "FLOW_START_FRAC", 0.0))
        return float(np.clip(start + (1.0 - start) * shaped, 0.0, 1.0))

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
        prev = float(getattr(self, "lever_angle", 0.0))
        ang = float(np.clip(angle, 0.0, self.lever_open_rad))
        self._lever_ang_vel = ang - prev
        self.lever_angle = ang
        self._lever_angle_max = max(self._lever_angle_max, ang)
        if ang >= self.LEVER_OPEN_THRESH:
            self.opened_once = True

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
        ang = float(np.arctan2(-v[1], v[2]))
        return float(np.clip(ang, 0.0, self.lever_open_rad))

    def _sim_dt(self) -> float:
        try:
            return float(self.scene.get_timestep())
        except Exception:
            return 1.0 / 250.0

    def _lever_contact_force(self) -> float:
        """PhysX contact force (N) between gripper links and the wooden lever."""
        if not hasattr(self, "robot") or self._lever_entity is None:
            return 0.0
        dt = self._sim_dt()
        lever_name = "beer_tap_lever"
        grip = set(getattr(self.robot, "gripper_name", []) or [])
        imp = 0.0
        try:
            for contact in self.scene.get_contacts():
                n0 = contact.bodies[0].entity.name
                n1 = contact.bodies[1].entity.name
                if lever_name not in (n0, n1):
                    continue
                other = n1 if n0 == lever_name else n0
                o = str(other).lower()
                if grip and other not in grip:
                    if not (
                        o.startswith("left")
                        or o.startswith("right")
                        or "finger" in o
                        or "pad" in o
                        or "hand" in o
                        or "gripper" in o
                    ):
                        continue
                for pt in getattr(contact, "points", None) or []:
                    v = np.asarray(getattr(pt, "impulse", [0.0, 0.0, 0.0]), dtype=float)
                    imp += float(np.linalg.norm(v))
        except Exception:
            return 0.0
        return float(imp / max(dt, 1e-6))

    def _tcp_for_arm(self, arm: ArmTag) -> np.ndarray | None:
        side = str(arm)
        # Prefer interactive teleop command (same UniversalRobotControls as other
        # household tasks) so the spring lever tracks the hand without lag.
        cmd = getattr(self, "_interactive_cmd_pose", None)
        if isinstance(cmd, dict) and side in cmd:
            ee = np.asarray(cmd[side][:3], dtype=float)
            return ee - np.array([0.0, 0.0, self.EE_TO_TCP], dtype=float)
        try:
            ee = self._ee_pos(arm)
        except Exception:
            return None
        return ee - np.array([0.0, 0.0, self.EE_TO_TCP], dtype=float)

    def _lever_press_signal(self):
        """Best gripper press on the knob/arc — drives the spring lever.

        The handle is kinematic; pressure is a proximity proxy along the hinge
        arc (same pattern as fill_coffee_jar's spring key) plus optional PhysX
        contact force on ``beer_tap_lever``.
        """
        if self.overflowed or not hasattr(self, "robot"):
            return None
        if bool(getattr(self, "_lever_inhibit_press", False)):
            return None
        pivot = np.asarray(self.lever_pivot_xyz, dtype=float)
        tip = self._lever_tip_xyz()
        contact_n = self._lever_contact_force()
        sides = []
        try:
            sides.append(self.arm)
        except Exception:
            pass
        for side in ("left", "right"):
            tag = ArmTag(side)
            if tag not in sides:
                sides.append(tag)

        best = None
        for arm in sides:
            tcp = self._tcp_for_arm(arm)
            if tcp is None:
                continue
            v = tcp - pivot
            r_yz = float(np.hypot(v[1], v[2]))
            x_err = abs(float(tcp[0] - pivot[0]))
            tip_dist = float(np.linalg.norm(tcp - tip))
            on_arc = (
                abs(r_yz - float(self.LEVER_LEN)) <= float(self.LEVER_ARC_RADIAL_TOL)
                and x_err <= float(self.LEVER_ARC_X_TOL)
                and float(v[2]) > -0.02
            )
            near_tip = tip_dist <= float(self.LEVER_CONTACT_R)
            if not (on_arc or near_tip or contact_n > 1.5):
                continue
            ang = self._angle_from_tip_point(tcp)
            # Stronger engagement when closer to the arc / tip, or with contact.
            score = (
                -abs(r_yz - float(self.LEVER_LEN))
                - 0.5 * x_err
                - 0.35 * tip_dist
                + 0.001 * contact_n
            )
            cand = {
                "arm": arm,
                "tcp": tcp,
                "angle": ang,
                "score": score,
                "contact_n": contact_n,
            }
            if best is None or cand["score"] > best["score"]:
                best = cand
        return best

    def _update_lever_from_pressure(self):
        """Spring lever: track gripper pressure on the knob; return when free."""
        if self.overflowed:
            # Snap shut after a spill so the stream cuts immediately.
            if self.lever_angle > 1e-4:
                self._apply_lever_pose(0.0)
            self._lever_pressed = False
            return

        sig = self._lever_press_signal()
        if sig is not None:
            self._lever_pressed = True
            target = float(sig["angle"])
            # Contact force nudges the handle a bit further open.
            target = float(
                np.clip(
                    target
                    + float(self.LEVER_CONTACT_FORCE_GAIN)
                    * min(float(sig.get("contact_n", 0.0)), 40.0),
                    0.0,
                    float(self.lever_open_rad),
                )
            )
            cur = float(self.lever_angle)
            step = float(self.LEVER_TRACK_STEP)
            if abs(target - cur) <= step:
                self._apply_lever_pose(target)
            else:
                self._apply_lever_pose(cur + step * np.sign(target - cur))
            return

        # No gripper pressure → spring back to upright.
        self._lever_pressed = False
        if self.lever_angle <= 1e-4:
            # Stay at rest; clear residual spring delta so idle settle can latch.
            self._lever_ang_vel = 0.0
            return
        cur = float(self.lever_angle)
        span = max(self.lever_open_rad, 1e-6)
        rate = max(
            0.014,
            float(self.LEVER_RETURN_STEP) * (0.40 + 0.60 * cur / span),
        )
        self._apply_lever_pose(max(0.0, cur - rate))

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
        # Refresh often while pouring so the level keeps rising in interactive.
        pouring = float(self._flow_frac()) > 1e-4
        interactive = bool(getattr(self, "_interactive_robot_mode", False)) or bool(
            getattr(self, "_interactive_universal_controls", False)
        )
        if pouring:
            min_dh = 0.00025 if interactive else 0.00045
        else:
            min_dh = 0.0008
        if (
            not force
            and abs(liq_half - self._liquid_half_h_cached) < min_dh
            and abs(foam_half - self._foam_half_h_cached) < min_dh
        ):
            return
        self._liquid_half_h_cached = liq_half
        self._foam_half_h_cached = foam_half
        self._liquid_entity = self._remove_entity(self._liquid_entity)
        self._foam_entity = self._remove_entity(self._foam_entity)
        # Match measure_ingredient liquid radii: fuller under hollow alpha shell.
        viewer_shell = bool(getattr(self, "_mug_visual_hollow", False))
        r_scale = 0.97 if viewer_shell else 0.90
        r = float(self.cup_inner_r) * r_scale
        if liq_frac > 1e-4:
            self._liquid_entity = self._make_cup_fluid(
                liq_half,
                r,
                self.cup_bottom_z,
                self._beer_rgba(),
                "beer_liquid",
                meniscus_rgba=self.BEER_MENISCUS,
            )
        if foam_frac > 1e-4:
            foam_rgba = self._foam_rgba()
            self._foam_entity = self._make_cup_fluid(
                foam_half,
                r,
                self.cup_bottom_z + liq_h,
                foam_rgba,
                "beer_foam",
                meniscus_rgba=foam_rgba,
            )

    def _sync_stream(self, force: bool = False):
        """Beer cylinder from spout → glass; diameter scales with lever open frac."""
        frac = self._flow_frac()
        if (
            not force
            and abs(frac - self._stream_frac_cached) < 0.02
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
        # Thin trickle when barely open → thick column when fully pulled.
        radius = float(self.STREAM_R_MIN + (self.STREAM_R_MAX - self.STREAM_R_MIN) * frac)
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
            # Cut the tap once it crests — spring / force-cut shuts the handle.
            self._lever_held = False
            self._lever_pressed = False
            self._apply_lever_pose(0.0)

    def _step_fluids(self):
        """Bottle-like fill: while lever open, beer+foam rise with rate∝handle angle."""
        frac = self._flow_frac()
        if frac > 1e-4 and not self.overflowed:
            scale = float(getattr(self, "flow_rate_scale", self.FLOW_RATE_SCALE))
            d = float(self.pour_rate) * frac * scale
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
            scale = float(getattr(self, "flow_rate_scale", self.FLOW_RATE_SCALE))
            d = float(self.pour_rate) * frac * scale
            self.spill_amount = float(self.spill_amount) + d * (1.0 + self.foam_gain)
            self._rebuild_spill_puddle()
        elif (not self.overflowed) and self.foam_level > 1e-6:
            # Collapse foam into a little beer (not 1:1 — avoids an end surge).
            decay = min(float(self.foam_decay), float(self.foam_level))
            self.foam_level = max(0.0, float(self.foam_level) - decay)
            to_beer = decay * float(getattr(self, "FOAM_TO_LIQUID", 0.28))
            if to_beer > 0.0 and float(self.liquid_level) < 1.0:
                room = max(0.0, 1.0 - float(self.liquid_level))
                self.liquid_level = min(1.0, float(self.liquid_level) + min(to_beer, room))
        if (not self.overflowed) and self._total_fill() >= self.overflow_level + 1e-4:
            self._mark_overflow(spilled=0.08)
        self._rebuild_fluids(force=False)
        self._sync_stream(force=False)

    def _tap_is_idle_instant(self) -> bool:
        """Tap not open/pressed, upright, no flow, not still moving."""
        ang = float(self.lever_angle)
        vel = abs(float(getattr(self, "_lever_ang_vel", 0.0)))
        return (
            not bool(self.tab_open)
            and not bool(getattr(self, "_lever_pressed", False))
            and ang <= float(self.TAP_IDLE_ANGLE)
            and float(self._flow_frac()) <= 1e-4
            and vel <= float(self.TAP_IDLE_VEL)
        )

    def _tap_fully_stopped(self) -> bool:
        """Require idle for several consecutive sim steps (settle)."""
        return int(getattr(self, "_tap_idle_steps", 0)) >= int(
            getattr(self, "TAP_SETTLE_STEPS", 36)
        )

    def _liquid_fully_stable(self) -> bool:
        """Liquid has not risen for TAP_SETTLE_STEPS while the tap is idle."""
        return int(getattr(self, "_liquid_stable_steps", 0)) >= int(
            getattr(self, "TAP_SETTLE_STEPS", 36)
        )

    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
        if not getattr(self, "_loaded", False):
            return
        self._update_lever_from_pressure()
        if self._tap_is_idle_instant():
            self._tap_idle_steps = int(getattr(self, "_tap_idle_steps", 0)) + 1
        else:
            self._tap_idle_steps = 0
        if (
            self.opened_once
            and self._tap_fully_stopped()
            and float(self.liquid_level) > 0.05
        ):
            self.closed_after_pour = True
        # Advance fluids first, then score liquid stability (blocks foam→beer rise).
        self._step_fluids()
        liq = float(self.liquid_level)
        prev = float(getattr(self, "_liquid_level_prev", liq))
        eps = float(getattr(self, "LIQUID_STABLE_EPS", 1e-5))
        if self._tap_is_idle_instant() and (liq - prev) <= eps:
            self._liquid_stable_steps = int(
                getattr(self, "_liquid_stable_steps", 0)
            ) + 1
        else:
            self._liquid_stable_steps = 0
        self._liquid_level_prev = liq

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
        """Approach the upright knob so gripper pressure can bend the spring lever."""
        self._lever_held = False
        self._lever_inhibit_press = False
        self.plan_success = True
        self.move(self.close_gripper(arm))
        ang0 = float(self.lever_angle)
        self.move(self.move_to_pose(arm, self._lever_ee_pose(ang0, self.LEVER_HOVER)))
        if not self.plan_success:
            print("[pour_beer] lever hover failed — continuing toward knob")
            self.plan_success = True
        self.move(self.move_to_pose(arm, self._lever_ee_pose(ang0, self.LEVER_GRASP_Z)))
        if not self.plan_success:
            self.plan_success = True
        # Brief dwell so pressure coupling engages (no pose teleport).
        self._idle_steps(4)
        self._lever_held = True
        return True

    def _sweep_lever_to(
        self,
        arm: ArmTag,
        target_frac: float,
        n_steps: int = 10,
        stop_on_foam: bool = False,
    ):
        """Walk the EE along the knob arc — lever angle follows gripper pressure only.

        The handle is never teleported here; ``_update_lever_from_pressure`` bends
        it when the TCP presses the knob and springs it back when contact is lost.
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
            tip = self._lever_tip_xyz(ang)
            goal_ee = tip + np.array(
                [0.0, 0.0, self.EE_TO_TCP + self.LEVER_GRASP_Z], dtype=float
            )
            # Short nudges keep TCP on the arc so pressure continuously drives the hinge.
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
            self._idle_steps(3)
            if self.overflowed:
                break
        print(
            f"[pour_beer] lever→{self._lever_open_frac():.2f} "
            f"ang={np.degrees(self.lever_angle):.1f}° "
            f"pressed={self._lever_pressed} "
            f"liq={self.liquid_level:.2f} foam={self.foam_level:.2f}"
        )

    def _release_lever(self, arm: ArmTag):
        """Clear the knob so the spring returns the handle upright.

        Press stays inhibited until the next ``_grasp_lever`` so a nearby hand
        cannot immediately re-bend the spring after retreat.
        """
        self._lever_held = False
        self._lever_inhibit_press = True
        # Up and slightly toward +Y (away from the open-tip −Y swing).
        self._move_ok(arm, dy=0.06, dz=0.12)
        self._idle_steps(
            60,
            until=lambda: self.lever_angle < 0.02 or self.overflowed,
        )

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
        # Leave headroom so spring-return ticks after release cannot crest the rim.
        pour_cap = min(float(self.safe_total) - 0.04, close_at)
        for cycle in range(max_cycles):
            if self.overflowed or self.liquid_level >= close_at:
                break
            if self._lever_open_frac() < 0.20:
                if self.liquid_level < close_at - 0.20:
                    reopen = 0.65
                elif self.liquid_level < close_at - 0.08:
                    reopen = 0.40
                else:
                    reopen = 0.28
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
                    or self._total_fill() >= pour_cap
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
                or self._total_fill() >= pour_cap
                or self.liquid_level >= close_at
            )
            if need_close:
                # Lift off the knob — spring snaps the handle shut (no teleport).
                self._release_lever(arm)
                if self.overflowed:
                    break
                self._idle_steps(
                    140,
                    until=lambda: (
                        self.foam_level <= self.expert_foam_resume or self.overflowed
                    ),
                )
                if self.overflowed or self.liquid_level >= close_at:
                    break
                # Re-engage the knob before the next pour push.
                self._grasp_lever(arm)

        # 3) Lift off — spring returns the handle to upright; wait until settled.
        self._release_lever(arm)
        settle_need = int(getattr(self, "TAP_SETTLE_STEPS", 36))
        self._idle_steps(
            max(220, settle_need * 6),
            until=lambda: (
                self.overflowed
                or (
                    self.foam_level < 0.05
                    and not bool(self.tab_open)
                    and float(self._flow_frac()) <= 1e-4
                    and self._tap_fully_stopped()
                    and self._liquid_fully_stable()
                )
            ),
        )

        if self.overflowed:
            self.plan_success = False
        elif self.check_success():
            self.plan_success = True

        self.info["info"] = {
            "{A}": "beer tap",
            "{B}": self.GLASS_MODEL,
            "{C}": "tap lever",
            "{a}": str(arm),
        }
        return self.info

    def check_success(self):
        if self.overflowed:
            return False
        if not self.opened_once:
            return False
        # Hard gate: never succeed while tap is open, flowing, or pressed.
        # Mid-pour spring-shut foam pauses must not count until fully settled.
        if (
            bool(self.tab_open)
            or float(self._flow_frac()) > 1e-4
            or bool(getattr(self, "_lever_pressed", False))
        ):
            return False
        if not self._tap_fully_stopped():
            return False
        # Foam collapse still raises liquid — wait until level stops rising.
        if not self._liquid_fully_stable():
            return False
        # Foam should have settled before judging beer level.
        foam_cap = max(0.08, float(getattr(self, "expert_foam_resume", 0.09)))
        if float(self.foam_level) > foam_cap:
            return False
        if not bool(getattr(self, "closed_after_pour", False)):
            return False
        lo = float(self.target_liquid) - float(self.full_liquid_tol)
        hi = float(self.target_liquid) + float(self.full_liquid_tol)
        liquid_ok = lo <= float(self.liquid_level) <= hi
        not_overfull = self._total_fill() < float(self.overflow_level) - 0.02
        return bool(liquid_ok and not_overfull)

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
            "flow_frac": float(self._flow_frac()),
            "lever_pressed": bool(getattr(self, "_lever_pressed", False)),
            "tab_open": bool(self.tab_open),
            "overflowed": bool(self.overflowed),
            "spill_amount": float(getattr(self, "spill_amount", 0.0)),
            "opened_once": bool(self.opened_once),
            "closed_after_pour": bool(self.closed_after_pour),
            "tap_fully_stopped": bool(self._tap_fully_stopped()),
            "tap_idle_steps": int(getattr(self, "_tap_idle_steps", 0)),
            "liquid_stable_steps": int(getattr(self, "_liquid_stable_steps", 0)),
            "liquid_fully_stable": bool(self._liquid_fully_stable()),
            "pour_rate": float(getattr(self, "pour_rate", self.POUR_RATE)),
            "foam_gain": float(getattr(self, "foam_gain", self.FOAM_GAIN)),
            "foam_decay": float(getattr(self, "foam_decay", self.FOAM_DECAY)),
            "cup_xy": np.asarray(self.cup_xy, dtype=float).tolist(),
            "tap_xy": np.asarray(self.tap_xy, dtype=float).tolist(),
            "scene_id": int(getattr(self, "scene_id", 0)),
        }
        return obs
