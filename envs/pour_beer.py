"""Pour beer from a bar tap into a beer mug (KitchenS).

Chrome draft tower with a low-resistance wooden lever. The handle has collision
so the gripper can press the knob; while contacted it free-follows the hand,
and a snappy spring returns it upright (cutting flow) as soon as contact is
lost. Beer stream thickness and fill rate scale with how far the handle is
turned. Foaminess of the stream
ramps up the longer the tap stays open in one continuous pour, and resets when
the tap is closed then reopened. Overflow fails with a yellow stain.

The drinking vessel is a simple procedural glass beer mug (body + D-handle).
Demo cameras use transmission glass; the interactive viewer uses a hollow
alpha shell so beer composites through the walls.

Episode randomization (task_args.pour_beer):
  - ``randomize_layout``: cup/tap station + bar props with AABB non-overlap
  - ``randomize_rates`` / ``pour_rate_range`` / ``foam_gain_range``: fill & peak foam %
  - ``foam_gain_start`` / ``foam_ramp_steps``: stream foam % ramp over open time
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
from .utils.create_actor import create_actor


class pour_beer(KitchenS_base_task):
    """Drive the tap lever by hand to fill the mug without foam overflow."""

    GLASS_MODEL = "beer_mug"
    GLASS_UPRIGHT_Q = [0.70710678, 0.70710678, 0.0, 0.0]
    # Match measure_ingredient jar cylinder orientations.
    UPRIGHT_CYL_Q = [0.70710678, 0.0, -0.70710678, 0.0]
    VERTICAL_CYL_Q = [0.70710678, 0.0, 0.70710678, 0.0]

    # Procedural beer-mug geometry (meters) — simplified seidel mug:
    # smooth body, thin floor, D-handle. No facet panels / stacked foot.
    MUG_INNER_R = 0.034
    MUG_WALL_T = 0.0035
    MUG_HEIGHT = 0.135
    MUG_BOTTOM_T = 0.008
    MUG_BASE_R = 0.040
    MUG_RIM_LIP = 0.0020
    MUG_HANDLE_REACH = 0.032
    MUG_HANDLE_THICK = 0.009
    MUG_FACET_FRAC = 0.0  # no seidel panels (reduced detail)

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
    # Snappy spring return (rad/tick) — only when the gripper is NOT contacting.
    # ~0.06 rad/tick closes a full open (~1.73 rad) in ~0.12s and cuts flow fast.
    LEVER_RETURN_STEP = 0.06
    # Gripper↔knob engagement (pressure proxy + contact).
    # Keep engage tight; release radius is larger so the spring doesn't flick
    # back into the hand from brief proximity gaps (that shoved the arm).
    LEVER_ARC_RADIAL_TOL = 0.016  # m; |r_yz − LEVER_LEN| (~lever radius + slack)
    LEVER_ARC_X_TOL = 0.016  # m; lateral off the hinge plane
    LEVER_CONTACT_R = 0.022  # m; must be near the current tip to engage
    LEVER_RELEASE_R = 0.055  # m; stay engaged until this far from the tip
    LEVER_CONTACT_FORCE_N = 1.2  # N; light finger contact on the knob engages
    LEVER_HOLD_FORCE_N = 0.4  # N; residual contact keeps hold (no spring)
    LEVER_PRESS_LOST_GRACE = 12  # ~50ms at 250Hz before spring snaps shut
    # Slow track + hold deadband: stable TCP must freeze the lever (no bounce).
    LEVER_TRACK_STEP = 0.028  # rad/tick while pressed (~1.6°/tick)
    LEVER_HOLD_DEADBAND = 0.035  # rad; |target−cur| below this → hold pose
    LEVER_TARGET_EMA = 0.22  # blend toward new TCP angle while pressed
    LEVER_CONTACT_FORCE_GAIN = 0.0  # disabled — impulse spikes caused bounce
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

    # Success requires liquid_level strictly above this (e.g. >90%).
    TARGET_LIQUID = 0.90
    # Kept for config/UI compatibility; success no longer uses a ± band.
    FULL_LIQUID_TOL = 0.05
    # Max rate at full open (per physics step); scales with lever angle.
    # Keep modest — long IK arcs while open still advance fill every tick.
    POUR_RATE = 0.000715
    # Foam % of the stream ramps with continuous open time (resets on close).
    # Start low so a fresh crack is mostly beer; peak encourages pause-and-pour.
    FOAM_GAIN_START = 0.18
    FOAM_GAIN = 0.85  # peak foam fraction after FOAM_RAMP_STEPS of continuous flow
    FOAM_RAMP_STEPS = 70  # sim steps of continuous pour to reach peak foam %
    FOAM_DECAY = 0.0045
    # Fraction of collapsing foam that becomes beer (modest — no end surge).
    FOAM_TO_LIQUID = 0.28
    # Per-episode sample ranges when randomize_rates / [lo,hi] yaml values are used.
    POUR_RATE_RANGE = (0.00052, 0.000975)
    FOAM_GAIN_RANGE = (0.65, 1.05)  # peak foam % range when randomize_rates
    FOAM_GAIN_START_RANGE = (0.10, 0.28)
    FOAM_RAMP_STEPS_RANGE = (50, 100)
    FOAM_DECAY_RANGE = (0.0035, 0.0060)
    OVERFLOW_LEVEL = 1.0
    EXPERT_FOAM_PAUSE = 0.16
    EXPERT_FOAM_RESUME = 0.09
    # Expert pour cap (beer+foam) — above TARGET so liquid can clear 90%.
    SAFE_TOTAL = 0.96
    # Tap must stay fully idle this long before success can pass.
    # 1s gap avoids mid-pour / spring-return flicker latching success while the
    # lever still looks open or foam is still collapsing into beer.
    TAP_SETTLE_SEC = 1.0
    # If the tap stays closed this long after an opening (no reopen), end the
    # episode and score via check_success (success if criteria met, else fail).
    OPEN_GAP_TIMEOUT_SEC = 5.0
    TAP_IDLE_ANGLE = 0.02  # rad — upright enough to count as closed
    TAP_IDLE_VEL = 0.05  # rad/step — below return step once nearly shut
    # Liquid must not rise more than this while idle (blocks foam→beer "filling").
    LIQUID_STABLE_EPS = 1e-5

    # Non-overlap layout (axis-aligned footprint half-sizes, meters).
    LAYOUT_MARGIN = 0.030
    # Keep prop AABBs this far inside the counter rim (static props must look seated;
    # larger than a hairline so long items like the baguette don't read as overhanging).
    TABLE_EDGE_MARGIN = 0.04
    # Reserved clear zone covering glass + tap base + lever arc.
    STATION_HALF_XY = (0.10, 0.16)
    # Reachable pour stations on either arm half (mirrored).
    STATION_X_RANGE_LEFT = (-0.16, -0.06)
    STATION_X_RANGE_RIGHT = (0.06, 0.16)
    # Legacy single-sided fallback (right); prefer left/right ranges when randomizing.
    STATION_X_RANGE = (0.06, 0.16)
    CUP_Y_RANGE = (-0.14, -0.05)
    # Tap Y jitter: upper bound = baseline tap_y, lower = baseline − TAP_Y_DOWN.
    TAP_Y_DOWN = 0.10  # 10 cm toward the robot
    PROP_X_RANGE = (-0.55, 0.55)
    PROP_Y_RANGE = (-0.26, 0.28)
    # Keep the apron in front of the mug clear for approach / viewing.
    MUG_CORRIDOR_HALF_XY = (0.10, 0.14)

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
        self._foam_open_steps = 0
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
        self._lever_press_lost_steps = 0
        self._lever_ang_vel = 0.0
        self._lever_target_filt = None
        self._tap_idle_steps = 0
        self._liquid_stable_steps = 0
        self._liquid_level_prev = 0.0
        self._stream_frac_cached = -1.0
        self._closed_since_open_steps = 0
        self._pour_gap_timed_out = False
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
        """Glass for the procedural beer mug (measure_ingredient jar structure).

        Demo cameras use transmission glass. The interactive SAPIEN viewer uses
        plain alpha glass so opaque beer composites through the walls.
        """
        if viewer_shell:
            glass = sapien.render.RenderMaterial(
                base_color=[0.78, 0.88, 0.96, 0.22]
            )
            try:
                glass.set_transmission(0.0)
                glass.set_transmission_roughness(1.0)
                glass.set_roughness(0.08)
                glass.set_metallic(0.0)
            except Exception:
                glass.roughness = 0.08
                glass.metallic = 0.0
            try:
                glass.set_ior(1.0)
            except Exception:
                pass
            return glass

        glass = sapien.render.RenderMaterial(base_color=[0.82, 0.90, 0.98, 0.12])
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

    def _table_half_xy(self):
        """Half-extents of the KitchenS counter top in world XY (meters)."""
        area = (getattr(self, "kitchens_info", None) or {}).get("table_area", [1.2, 0.7])
        return 0.5 * float(area[0]), 0.5 * float(area[1])

    def _footprint_on_table(self, center, half, margin=None):
        """True iff the axis-aligned footprint lies fully on the counter."""
        if margin is None:
            margin = self.TABLE_EDGE_MARGIN
        thx, thy = self._table_half_xy()
        bias = getattr(self, "table_xy_bias", [0.0, 0.0])
        c = np.asarray(center, dtype=float)
        h = np.asarray(half, dtype=float)
        cx = float(c[0] - float(bias[0]))
        cy = float(c[1] - float(bias[1]))
        return (
            cx - h[0] >= -thx + margin
            and cx + h[0] <= thx - margin
            and cy - h[1] >= -thy + margin
            and cy + h[1] <= thy - margin
        )

    def _clamp_footprint_to_table(self, center, half, margin=None):
        """Shift a footprint center so its AABB stays on the counter."""
        if margin is None:
            margin = self.TABLE_EDGE_MARGIN
        thx, thy = self._table_half_xy()
        bias = getattr(self, "table_xy_bias", [0.0, 0.0])
        h = np.asarray(half, dtype=float)
        xy = np.asarray(center, dtype=float).copy()
        # Work in table-local coords, then restore bias.
        lx = float(xy[0] - float(bias[0]))
        ly = float(xy[1] - float(bias[1]))
        lx = float(np.clip(lx, -thx + h[0] + margin, thx - h[0] - margin))
        ly = float(np.clip(ly, -thy + h[1] + margin, thy - h[1] - margin))
        return np.array([lx + float(bias[0]), ly + float(bias[1])], dtype=float)

    @staticmethod
    def _model_data(modelname: str, model_id: int) -> dict:
        path = Path("assets/objects") / modelname / f"model_data{int(model_id)}.json"
        with open(path) as f:
            return json.load(f)

    def _yup_local_half_xy(self, modelname: str, model_id: int, scale_mult: float = 1.0):
        """Local XY half-extents for a Y-up mesh after ``GLASS_UPRIGHT_Q`` (pre-yaw)."""
        data = self._model_data(modelname, model_id)
        sc = data.get("scale") or [1.0, 1.0, 1.0]
        if isinstance(sc, (int, float)):
            sc = [float(sc)] * 3
        try:
            mult = [float(m) for m in scale_mult]
        except TypeError:
            mult = [float(scale_mult)] * 3
        ext = data["extents"]
        # Upright quat maps local Y→world Z; footprint uses local X/Z.
        return (
            0.5 * float(sc[0]) * float(mult[0]) * float(ext[0]),
            0.5 * float(sc[2]) * float(mult[2]) * float(ext[2]),
        )

    @staticmethod
    def _rotated_half_xy(half_xy, yaw_deg: float):
        """World AABB half-extents of a local footprint rotated by yaw about Z."""
        hx, hy = float(half_xy[0]), float(half_xy[1])
        yaw = np.deg2rad(float(yaw_deg))
        c, s = abs(np.cos(yaw)), abs(np.sin(yaw))
        return np.array([hx * c + hy * s, hx * s + hy * c], dtype=float)

    def _prop_pose_z(self, modelname: str, model_id: int, scale_mult: float, surface_z: float):
        """Pose Z so a Y-up mesh under ``GLASS_UPRIGHT_Q`` rests on ``surface_z``."""
        data = self._model_data(modelname, model_id)
        sc = data.get("scale") or [1.0, 1.0, 1.0]
        if isinstance(sc, (int, float)):
            sc = [float(sc)] * 3
        try:
            mult = [float(m) for m in scale_mult]
        except TypeError:
            mult = [float(scale_mult)] * 3
        final_sy = float(sc[1]) * float(mult[1])
        cy = float(data.get("center", [0.0, 0.0, 0.0])[1])
        ey = float(data["extents"][1])
        bottom_local_y = cy - 0.5 * ey
        return float(surface_z - bottom_local_y * final_sy)

    def _inset_xy_ranges(self, x_range, y_range, half):
        """Shrink a sampling rectangle so the footprint AABB stays on-table."""
        thx, thy = self._table_half_xy()
        m = self.TABLE_EDGE_MARGIN
        h = np.asarray(half, dtype=float)
        x_lo = max(float(x_range[0]), -thx + h[0] + m)
        x_hi = min(float(x_range[1]), thx - h[0] - m)
        y_lo = max(float(y_range[0]), -thy + h[1] + m)
        y_hi = min(float(y_range[1]), thy - h[1] - m)
        return (x_lo, x_hi), (y_lo, y_hi)

    def _sample_free_xy(self, rng, half, blockers, x_range, y_range, tries=80):
        half = np.asarray(half, dtype=float)
        (x_lo, x_hi), (y_lo, y_hi) = self._inset_xy_ranges(x_range, y_range, half)
        if x_lo > x_hi or y_lo > y_hi:
            # Region too small for this footprint — park at the clamped region center.
            mid = np.array(
                [
                    0.5 * (float(x_range[0]) + float(x_range[1])),
                    0.5 * (float(y_range[0]) + float(y_range[1])),
                ],
                dtype=float,
            )
            return self._clamp_footprint_to_table(mid, half)
        for _ in range(int(tries)):
            p = np.array([rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)], dtype=float)
            if self._footprint_clear(p, half, blockers) and self._footprint_on_table(p, half):
                return p
        # Fallback: densest search for max clearance (still on-table).
        best, best_score = None, -1e9
        for _ in range(120):
            p = np.array([rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)], dtype=float)
            if not self._footprint_on_table(p, half):
                continue
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
        if best is not None:
            return best
        mid = np.array([0.5 * (x_lo + x_hi), 0.5 * (y_lo + y_hi)], dtype=float)
        return self._clamp_footprint_to_table(mid, half)

    def _station_center_xy(self):
        """Midpoint between glass and tap (reserved clear zone center)."""
        return 0.5 * (np.asarray(self.cup_xy, dtype=float) + np.asarray(self.tap_xy, dtype=float))

    def _resolve_station_layout(self, cfg, rng):
        """Sample tap XY; mug stays under the nozzle (shared X, fixed tap→cup dy).

        Tap Y uses the baseline (``cup_y + tap_dy``) as the *upper* bound and
        baseline − ``tap_y_down`` (default 10 cm) as the lower bound — matching
        the fill_coffee_jar dispenser jitter pattern.

        Station X is sampled on the left or right apron half (``station_side``:
        ``left`` / ``right`` / ``random``) so the matching arm can pour.
        """
        tap_dy = float(cfg.get("tap_dy", 0.12))
        base_side = float(cfg.get("station_x", 0.10))
        base_cup_y = float(cfg.get("cup_y", -0.08))
        base_tap_y = base_cup_y + tap_dy
        tap_y_down = float(cfg.get("tap_y_down", self.TAP_Y_DOWN))
        randomize = bool(cfg.get("randomize_layout", False))
        if not randomize:
            return base_side, base_cup_y, tap_dy

        left_range = self._parse_range(
            cfg, "station_x_range_left", self.STATION_X_RANGE_LEFT
        )
        right_range = self._parse_range(
            cfg, "station_x_range_right", self.STATION_X_RANGE_RIGHT
        )
        side_pref = str(cfg.get("station_side", "random")).lower().strip()
        if side_pref in ("left", "l"):
            side_order = ["left"]
        elif side_pref in ("right", "r"):
            side_order = ["right"]
        else:
            # Coin-flip which arm half hosts the tap+mug station.
            side_order = ["left", "right"]
            rng.shuffle(side_order)

        y_lo = base_tap_y - tap_y_down
        y_hi = base_tap_y
        for side_name in side_order:
            x_lo, x_hi = left_range if side_name == "left" else right_range
            for _ in range(40):
                side = float(rng.uniform(x_lo, x_hi))
                # Keep a small dead-band at the centerline so arm choice is unambiguous.
                if abs(side) < 0.02:
                    continue
                tap_y = float(rng.uniform(y_lo, y_hi))
                cup_y = tap_y - tap_dy
                return side, cup_y, tap_dy

        # Fallback: keep the configured baseline side.
        return base_side, base_cup_y, tap_dy

    def _resolve_rates(self, cfg, rng):
        self.pour_rate = self._sample_scalar_or_range(
            cfg, "pour_rate", self.POUR_RATE, rng,
            range_key="pour_rate_range", range_default=self.POUR_RATE_RANGE,
        )
        # foam_gain = peak foam % of stream after a continuous open pour.
        self.foam_gain = self._sample_scalar_or_range(
            cfg, "foam_gain", self.FOAM_GAIN, rng,
            range_key="foam_gain_range", range_default=self.FOAM_GAIN_RANGE,
        )
        self.foam_gain_start = self._sample_scalar_or_range(
            cfg, "foam_gain_start", self.FOAM_GAIN_START, rng,
            range_key="foam_gain_start_range",
            range_default=self.FOAM_GAIN_START_RANGE,
        )
        self.foam_ramp_steps = int(round(self._sample_scalar_or_range(
            cfg, "foam_ramp_steps", self.FOAM_RAMP_STEPS, rng,
            range_key="foam_ramp_steps_range",
            range_default=self.FOAM_RAMP_STEPS_RANGE,
        )))
        self.foam_decay = self._sample_scalar_or_range(
            cfg, "foam_decay", self.FOAM_DECAY, rng,
            range_key="foam_decay_range", range_default=self.FOAM_DECAY_RANGE,
        )
        # Clamp to safe positive bounds.
        self.pour_rate = float(np.clip(self.pour_rate, 1e-5, 0.005))
        self.foam_gain = float(np.clip(self.foam_gain, 0.05, 2.5))
        self.foam_gain_start = float(np.clip(self.foam_gain_start, 0.0, self.foam_gain))
        self.foam_ramp_steps = int(np.clip(self.foam_ramp_steps, 1, 2000))
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
        self.tap_settle_sec = float(cfg.get("tap_settle_sec", self.TAP_SETTLE_SEC))
        self.tap_settle_sec = float(np.clip(self.tap_settle_sec, 0.05, 10.0))
        self.open_gap_timeout_sec = float(
            cfg.get("open_gap_timeout_sec", self.OPEN_GAP_TIMEOUT_SEC)
        )
        self.open_gap_timeout_sec = float(np.clip(self.open_gap_timeout_sec, 0.5, 60.0))

        side, cup_y, tap_dy = self._resolve_station_layout(cfg, rng)
        self.arm = ArmTag("right" if side >= 0 else "left")
        self.arm_side = str(self.arm)
        # Match other household interactives: pre-select the working arm for teleop.
        self._interactive_selected_arms = (self.arm_side,)
        self.cup_xy = np.array([side, cup_y], dtype=float)
        self.tap_xy = np.array([side, cup_y + tap_dy], dtype=float)

        self.liquid_level = 0.0
        self.foam_level = 0.0
        self._foam_open_steps = 0
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
        self._lever_press_lost_steps = 0
        self._lever_ang_vel = 0.0
        self._lever_target_filt = None
        self._tap_idle_steps = 0
        self._liquid_stable_steps = 0
        self._liquid_level_prev = 0.0
        self._closed_since_open_steps = 0
        self._pour_gap_timed_out = False
        self.spill_amount = 0.0
        self._bar_props = []
        self._prop_footprints = []
        # mug_visual may alias the cup Actor — only remove a separate visual entity.
        mv = getattr(self, "mug_visual", None)
        cup = getattr(self, "cup", None)
        cup_ent = cup.actor if cup is not None and hasattr(cup, "actor") else cup
        if mv is not None and mv is not cup and mv is not cup_ent:
            self._remove_entity(mv)
        self.mug_visual = None
        self._mug_visual_hollow = False
        if cup is not None:
            try:
                self.scene.remove_entity(cup_ent)
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
            f"pour_rate={self.pour_rate:.5f} foam_gain={self.foam_gain_start:.2f}→"
            f"{self.foam_gain:.2f}/{self.foam_ramp_steps}steps "
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
        try:
            z = self._prop_pose_z(modelname, model_id, scale_mult, self.table_top) + float(z_off)
        except Exception:
            z = self.table_top + float(z_off)
        pose = sapien.Pose(
            [float(xy[0]), float(xy[1]), float(z)],
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
        """Sparse bar décor — keep the tap station clear; optional non-overlap randomize.

        Footprints come from each asset's ``model_data`` (scaled + yaw-rotated AABB) and
        are clamped so the full base stays on the counter — long props like the baguette
        must not hang off the rim.
        """
        if rng is None:
            rng = self._layout_rng(202)
        cfg = self._cfg
        randomize = bool(cfg.get("randomize_layout", False))

        # (model, id, scale, default_xy, default_yaw, region_xy)
        # default_xy is a preferred center; it is clamped onto the table using the
        # real yaw-aware footprint before spawn.
        catalog = [
            ("255_beer_bottle", 0, 1.00, [-0.42, 0.22], -10,
             ((-0.50, -0.18), (0.14, 0.26))),
            ("001_bottle", 2, 1.00, [-0.28, 0.22], 8,
             ((-0.38, -0.12), (0.14, 0.26))),
            ("001_bottle", 5, 1.00, [0.30, 0.22], -8,
             ((0.18, 0.48), (0.14, 0.26))),
            ("255_beer_bottle", 0, 1.00, [0.42, 0.22], 12,
             ((0.28, 0.50), (0.14, 0.26))),
            ("088_wineglass", 0, 0.38, [-0.06, 0.20], 15,
             ((-0.22, 0.02), (0.12, 0.24))),
            ("039_mug", 0, 0.65, [0.18, 0.20], 30,
             ((0.12, 0.36), (0.12, 0.24))),
            ("025_chips-tub", 0, 1.00, [-0.38, -0.14], -20,
             ((-0.50, -0.24), (-0.20, 0.02))),
            ("025_chips-tub", 2, 1.00, [0.36, -0.12], 30,
             ((0.22, 0.48), (-0.22, 0.04))),
            ("071_can", 0, 1.00, [-0.42, -0.22], -35,
             ((-0.52, -0.30), (-0.26, -0.08))),
            # Baguette is ~28 cm long — park well inboard, long axis along +X.
            ("054_baguette", 2, 1.00, [-0.26, 0.12], 90,
             ((-0.40, -0.14), (0.02, 0.20))),
        ]

        station_c = self._station_center_xy()
        station_h = np.asarray(self.STATION_HALF_XY, dtype=float)
        # Corridor on the robot side of the mug — props must not block the apron.
        mug_c = np.asarray(self.cup_xy, dtype=float)
        corridor_c = np.array([mug_c[0], mug_c[1] - 0.14], dtype=float)
        corridor_h = np.asarray(self.MUG_CORRIDOR_HALF_XY, dtype=float)
        tap_c = np.asarray(self.tap_xy, dtype=float)
        tap_h = np.array([self.BASE_R + 0.02, self.BASE_R + 0.04], dtype=float)
        blockers = [
            (station_c, station_h),
            (mug_c, np.array([0.06, 0.06], dtype=float)),
            (tap_c, tap_h),
            (corridor_c, corridor_h),
        ]

        for model, mid, scale, default_xy, default_yaw, region in catalog:
            try:
                local_half = self._yup_local_half_xy(model, mid, scale)
            except Exception as e:
                print(f"[pour_beer] skip prop {model}/base{mid}: {e}")
                continue

            if randomize:
                yaw = float(default_yaw) + float(rng.uniform(-25.0, 25.0))
                half = self._rotated_half_xy(local_half, yaw)
                xy = self._sample_free_xy(rng, half, blockers, region[0], region[1])
                # Reject if it still clips the station (extra guard).
                if not self._footprint_clear(xy, half, [(station_c, station_h)]):
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
                    xy = self._clamp_footprint_to_table(xy, half)
            else:
                yaw = float(default_yaw)
                half = self._rotated_half_xy(local_half, yaw)
                xy = self._clamp_footprint_to_table(np.asarray(default_xy, dtype=float), half)
                if not self._footprint_clear(xy, half, blockers):
                    # Prefer inward nudges (toward x=0) so we don't walk props off the rim.
                    sign = -1.0 if float(xy[0]) >= 0.0 else 1.0
                    deltas = []
                    for mag in (0.06, 0.10, 0.14, 0.18, 0.22):
                        deltas.append(np.array([sign * mag, 0.0]))
                        deltas.append(np.array([0.0, mag]))
                        deltas.append(np.array([0.0, -mag]))
                        deltas.append(np.array([-sign * mag, 0.0]))  # outward last
                    for dxy in deltas:
                        cand = self._clamp_footprint_to_table(xy + dxy, half)
                        if self._footprint_clear(cand, half, blockers) and self._footprint_on_table(
                            cand, half
                        ):
                            xy = cand
                            break

            # Final safety: never spawn a prop whose base leaves the counter.
            xy = self._clamp_footprint_to_table(xy, half)
            if not self._footprint_on_table(xy, half):
                print(
                    f"[pour_beer] skip prop {model}/base{mid}: cannot fit on table "
                    f"(half={half.tolist()})"
                )
                continue

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
                self,
                pose=pose,
                modelname="019_coaster",
                model_id=0,
                convex=True,
                is_static=True,
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
        """Simple D-handle on +X (visual only — no corner knobs)."""
        thick = float(self.MUG_HANDLE_THICK)
        reach = float(self.MUG_HANDLE_REACH)
        z_top = float(h) - 0.016
        z_bot = float(bottom_t) + 0.016
        z_mid = 0.5 * (z_top + z_bot)
        half_v = 0.5 * max(0.02, z_top - z_bot)
        x0 = float(outer_r) - 0.001
        x1 = float(outer_r) + reach

        for z in (z_top, z_bot):
            stub = sapien.render.RenderShapeBox(
                [0.5 * reach, 0.5 * thick, 0.5 * thick],
                glass,
            )
            stub.set_local_pose(sapien.Pose([x0 + 0.5 * reach, 0.0, z]))
            render_body.attach(stub)

        post = sapien.render.RenderShapeBox(
            [0.5 * thick, 0.5 * thick, half_v],
            glass,
        )
        post.set_local_pose(sapien.Pose([x1, 0.0, z_mid]))
        render_body.attach(post)

    def _build_mug_visual(self, hollow: bool = False):
        """Simplified procedural mug: floor + body + rim + handle (no facets)."""
        self.mug_visual = self._remove_entity(getattr(self, "mug_visual", None))
        if self.cup is None:
            return

        outer_r = self._mug_outer_r()
        inner_r = float(self.MUG_INNER_R)
        h = float(self.MUG_HEIGHT)
        bottom_t = float(self.MUG_BOTTOM_T)
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

        # Single thin floor (no stacked thick foot — that darkened the base).
        floor_half = max(0.0015, 0.5 * bottom_t)
        floor = sapien.render.RenderShapeCylinder(
            radius=outer_r * 0.98,
            half_length=floor_half,
            material=glass,
        )
        floor.set_local_pose(sapien.Pose([0.0, 0.0, floor_half], upright_q))
        render_body.attach(floor)

        if hollow:
            wall_t = 0.0024
            n_seg = 28
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

        rim = sapien.render.RenderShapeCylinder(
            radius=outer_r + float(self.MUG_RIM_LIP),
            half_length=0.0018,
            material=glass,
        )
        rim.set_local_pose(sapien.Pose([0.0, 0.0, h - 0.0018], upright_q))
        render_body.attach(rim)

        self._attach_mug_handle(render_body, glass, outer_r, h, bottom_t)

        vis.add_component(render_body)
        self.scene.add_entity(vis)
        self.mug_visual = vis
        self._mug_visual_hollow = bool(hollow)
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
        """Procedural glass beer mug on the coaster (collision + glass visual)."""
        x, y = float(self.cup_xy[0]), float(self.cup_xy[1])
        z0 = float(getattr(self, "coaster_top_z", self.table_top)) + 0.001
        outer_r = self._mug_outer_r()
        h = float(self.MUG_HEIGHT)
        bottom_t = float(self.MUG_BOTTOM_T)

        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        builder.add_cylinder_collision(
            pose=sapien.Pose([0.0, 0.0, h * 0.5], self.VERTICAL_CYL_Q),
            radius=float(outer_r),
            half_length=float(h * 0.5),
            material=self.scene.default_physical_material,
        )
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
        """Kinematic wooden handle hinged at the faucet head (pose updated each step).

        Collision is enabled so gripper fingers contact the knob (shaft + tip
        ball). Motion is still soft free-follow while pressed and a gentle
        spring only when free, so the kinematic body does not shove the arm.
        """
        self._lever_entity = self._remove_entity(self._lever_entity)
        self._lever_comp = None
        wood = self._opaque_material(self.WOOD)
        length = float(self.LEVER_LEN)
        tip_r = float(self.LEVER_R) * 1.35
        builder = self.scene.create_actor_builder()
        # Dynamic + set_kinematic: PhysX contacts with the gripper work reliably.
        builder.set_physx_body_type("dynamic")
        builder.add_cylinder_collision(
            pose=sapien.Pose(),
            radius=self.LEVER_R,
            half_length=0.5 * length,
            material=self.scene.default_physical_material,
        )
        builder.add_sphere_collision(
            pose=sapien.Pose([0.5 * length, 0.0, 0.0]),
            radius=tip_r,
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
            radius=tip_r,
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

        Engagement is tip proximity and/or PhysX contact on the lever collider.
        Release radius is larger than engage radius so brief gaps don't drop
        contact into spring-return. While already pressed, residual contact
        alone keeps engagement even if the tip briefly leaves the engage ball.
        """
        if self.overflowed or not hasattr(self, "robot"):
            return None
        if bool(getattr(self, "_lever_inhibit_press", False)):
            return None
        pivot = np.asarray(self.lever_pivot_xyz, dtype=float)
        tip = self._lever_tip_xyz()
        contact_n = self._lever_contact_force()
        engage_r = float(self.LEVER_CONTACT_R)
        release_r = float(getattr(self, "LEVER_RELEASE_R", engage_r * 2.0))
        tip_r = release_r if bool(getattr(self, "_lever_pressed", False)) else engage_r
        hold_n = float(getattr(self, "LEVER_HOLD_FORCE_N", 0.4))
        force_n = float(getattr(self, "LEVER_CONTACT_FORCE_N", 2.5))
        pressed = bool(getattr(self, "_lever_pressed", False))
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
            near_tip = tip_dist <= tip_r
            # Engage: near tip or firm contact. Stay: near tip OR any hold force.
            if pressed:
                engaged = near_tip or contact_n > hold_n
            else:
                engaged = near_tip or contact_n > force_n
            if not engaged:
                continue
            # If only contact is keeping us engaged (TCP far from tip), freeze at
            # the current hinge angle so kinematic tip motion can't chase itself.
            if near_tip or on_arc:
                ang = self._angle_from_tip_point(tcp)
                follow = True
            else:
                ang = float(self.lever_angle)
                follow = False
            score = (
                -abs(r_yz - float(self.LEVER_LEN))
                - 0.5 * x_err
                - 0.35 * tip_dist
                + 0.001 * contact_n
                + (0.01 if on_arc else 0.0)
                + (0.02 if near_tip else 0.0)
            )
            cand = {
                "arm": arm,
                "tcp": tcp,
                "angle": ang,
                "score": score,
                "contact_n": contact_n,
                "follow": follow,
            }
            if best is None or cand["score"] > best["score"]:
                best = cand
        return best

    def _update_lever_from_pressure(self):
        """Hold while gripper presses; spring only when free.

        Stable TCP ⇒ lever freezes at the commanded angle (hold deadband + EMA).
        No spring while contacted / in grace, so flow does not flicker.
        """
        if self.overflowed:
            # Snap shut after a spill so the stream cuts immediately.
            if self.lever_angle > 1e-4:
                self._apply_lever_pose(0.0)
            self._lever_pressed = False
            self._lever_press_lost_steps = 0
            self._lever_target_filt = None
            return

        contact_n = self._lever_contact_force()
        hold_n = float(getattr(self, "LEVER_HOLD_FORCE_N", 0.4))
        still_touching = contact_n > hold_n
        sig = self._lever_press_signal()

        if sig is not None or (
            bool(getattr(self, "_lever_pressed", False)) and still_touching
        ):
            self._lever_pressed = True
            self._lever_press_lost_steps = 0
            cur = float(self.lever_angle)
            if sig is not None and bool(sig.get("follow", True)):
                raw = float(
                    np.clip(
                        float(sig["angle"])
                        + float(self.LEVER_CONTACT_FORCE_GAIN)
                        * min(float(sig.get("contact_n", 0.0)), 40.0),
                        0.0,
                        float(self.lever_open_rad),
                    )
                )
                alpha = float(getattr(self, "LEVER_TARGET_EMA", 0.22))
                prev = getattr(self, "_lever_target_filt", None)
                if prev is None:
                    self._lever_target_filt = raw
                else:
                    self._lever_target_filt = (1.0 - alpha) * float(prev) + alpha * raw
                target = float(self._lever_target_filt)
            else:
                # Contact hold without a reliable TCP arc reading — freeze.
                target = cur
                if getattr(self, "_lever_target_filt", None) is None:
                    self._lever_target_filt = cur

            dead = float(getattr(self, "LEVER_HOLD_DEADBAND", 0.035))
            err = target - cur
            if abs(err) <= dead:
                # Arm stable → lever stable (zero hinge velocity, keep pose).
                self._lever_ang_vel = 0.0
                return
            step = float(self.LEVER_TRACK_STEP)
            self._apply_lever_pose(cur + float(np.clip(err, -step, step)))
            return

        # Lost signal: hold briefly so tip motion / teleop jitter doesn't spring
        # the handle back into the hand.
        if bool(getattr(self, "_lever_pressed", False)):
            grace = int(getattr(self, "LEVER_PRESS_LOST_GRACE", 50))
            self._lever_press_lost_steps = int(
                getattr(self, "_lever_press_lost_steps", 0)
            ) + 1
            if self._lever_press_lost_steps < grace or still_touching:
                self._lever_ang_vel = 0.0
                return

        # Truly free → snappy spring shut (constant rate so flow cuts promptly).
        self._lever_pressed = False
        self._lever_press_lost_steps = 0
        self._lever_target_filt = None
        if self.lever_angle <= 1e-4:
            self._lever_ang_vel = 0.0
            return
        cur = float(self.lever_angle)
        rate = float(self.LEVER_RETURN_STEP)
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
            self._lever_target_filt = None
            self._apply_lever_pose(0.0)

    def _effective_foam_gain(self) -> float:
        """Foam % of the current stream; ramps with continuous open time.

        Starts at ``foam_gain_start`` when the tap just begins flowing and
        linearly reaches peak ``foam_gain`` after ``foam_ramp_steps`` of
        continuous pour. Closing the tap (no flow) zeros the open-step
        counter so the next open starts low again.
        """
        start = float(getattr(self, "foam_gain_start", self.FOAM_GAIN_START))
        peak = float(getattr(self, "foam_gain", self.FOAM_GAIN))
        ramp = max(1, int(getattr(self, "foam_ramp_steps", self.FOAM_RAMP_STEPS)))
        t = min(1.0, float(getattr(self, "_foam_open_steps", 0)) / float(ramp))
        return float(start + (peak - start) * t)

    def _step_fluids(self):
        """Bottle-like fill: while lever open, beer+foam rise with rate∝handle angle.

        Foam fraction of the stream grows with how long the tap has been open
        continuously; it resets whenever flow stops.
        """
        frac = self._flow_frac()
        if frac > 1e-4:
            self._foam_open_steps = int(getattr(self, "_foam_open_steps", 0)) + 1
        else:
            self._foam_open_steps = 0
        foam_gain = self._effective_foam_gain()
        if frac > 1e-4 and not self.overflowed:
            scale = float(getattr(self, "flow_rate_scale", self.FLOW_RATE_SCALE))
            d = float(self.pour_rate) * frac * scale
            add_liq = d
            add_foam = d * foam_gain
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
            self.spill_amount = float(self.spill_amount) + d * (1.0 + foam_gain)
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

    def _tap_settle_steps(self) -> int:
        """Consecutive idle sim steps required before success (~TAP_SETTLE_SEC)."""
        sec = float(getattr(self, "tap_settle_sec", self.TAP_SETTLE_SEC))
        dt = max(1e-6, float(self._sim_dt()))
        return max(1, int(round(sec / dt)))

    def _open_gap_timeout_steps(self) -> int:
        """Closed steps after an opening before forced episode scoring."""
        sec = float(
            getattr(self, "open_gap_timeout_sec", self.OPEN_GAP_TIMEOUT_SEC)
        )
        dt = max(1e-6, float(self._sim_dt()))
        return max(1, int(round(sec / dt)))

    def _tap_is_flowing(self) -> bool:
        """True while the lever is open enough to count as a tap opening."""
        return bool(self.tab_open) or float(self._flow_frac()) > 1e-4

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
        """Require idle for TAP_SETTLE_SEC of consecutive sim time."""
        return int(getattr(self, "_tap_idle_steps", 0)) >= int(self._tap_settle_steps())

    def _liquid_fully_stable(self) -> bool:
        """Liquid has not risen for TAP_SETTLE_SEC while the tap is idle."""
        return int(getattr(self, "_liquid_stable_steps", 0)) >= int(
            self._tap_settle_steps()
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
        self._update_open_gap_timeout()

    def _update_open_gap_timeout(self):
        """End episode if tap stays closed >OPEN_GAP_TIMEOUT_SEC after an opening."""
        if bool(getattr(self, "_pour_gap_timed_out", False)):
            return
        if self._tap_is_flowing():
            self._closed_since_open_steps = 0
            return
        if not bool(getattr(self, "opened_once", False)):
            return
        self._closed_since_open_steps = int(
            getattr(self, "_closed_since_open_steps", 0)
        ) + 1
        if self._closed_since_open_steps < int(self._open_gap_timeout_steps()):
            return
        self._pour_gap_timed_out = True
        gap_s = float(
            getattr(self, "open_gap_timeout_sec", self.OPEN_GAP_TIMEOUT_SEC)
        )
        print(
            f"[pour_beer] tap closed >{gap_s:.0f}s since last opening — "
            "scoring episode",
            flush=True,
        )
        # Latch eval success immediately when criteria already hold.
        try:
            if self.check_success():
                self.eval_success = True
        except Exception:
            pass

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
        # Pour until beer is clearly past the >target success gate.
        close_at = max(0.10, float(self.target_liquid) + 0.02)
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
        settle_need = int(self._tap_settle_steps())
        self._idle_steps(
            max(settle_need + 50, settle_need * 2),
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
        # Also require ~1s continuous idle (TAP_SETTLE_SEC) so brief spring-shut
        # / contact flicker cannot latch success mid-pour.
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
        liquid_ok = float(self.liquid_level) > float(self.target_liquid)
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
            "tap_settle_steps": int(self._tap_settle_steps()),
            "tap_settle_sec": float(
                getattr(self, "tap_settle_sec", self.TAP_SETTLE_SEC)
            ),
            "open_gap_timeout_sec": float(
                getattr(self, "open_gap_timeout_sec", self.OPEN_GAP_TIMEOUT_SEC)
            ),
            "closed_since_open_steps": int(
                getattr(self, "_closed_since_open_steps", 0)
            ),
            "pour_gap_timed_out": bool(
                getattr(self, "_pour_gap_timed_out", False)
            ),
            "liquid_stable_steps": int(getattr(self, "_liquid_stable_steps", 0)),
            "liquid_fully_stable": bool(self._liquid_fully_stable()),
            "pour_rate": float(getattr(self, "pour_rate", self.POUR_RATE)),
            "foam_gain": float(getattr(self, "foam_gain", self.FOAM_GAIN)),
            "foam_gain_start": float(
                getattr(self, "foam_gain_start", self.FOAM_GAIN_START)
            ),
            "foam_ramp_steps": int(
                getattr(self, "foam_ramp_steps", self.FOAM_RAMP_STEPS)
            ),
            "foam_open_steps": int(getattr(self, "_foam_open_steps", 0)),
            "effective_foam_gain": float(self._effective_foam_gain()),
            "foam_decay": float(getattr(self, "foam_decay", self.FOAM_DECAY)),
            "cup_xy": np.asarray(self.cup_xy, dtype=float).tolist(),
            "tap_xy": np.asarray(self.tap_xy, dtype=float).tolist(),
            "scene_id": int(getattr(self, "scene_id", 0)),
        }
        return obs
