"""Fill a marked glass jar with beans from a glass-box dispenser (KitchenS).

Inherits ``KitchenS_base_task`` (microwave + dishrack + cooking range on a kitchen
counter). The dispenser is a raised clear glass hopper packed with real bean
meshes. Pressing the red button on top opens a nozzle above the jar and releases
beans into a glass jar marked with red ring lines at 25% / 50% / 75% (rim = full).

Dispense amount is gated by **press force** on the button (four thresholds), not
hold duration.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import sapien
import sapien.physx
import sapien.render
import transforms3d as t3d

from ._kitchens_base_task import KitchenS_base_task
from ._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
from .utils import *
from .utils.create_actor import create_actor, create_box


class fill_coffee_jar(KitchenS_base_task):
    """Press the dispenser button to fill a marked glass jar to a target level.

    Task options (``task_args.fill_coffee_jar``):
      - ``target_fill``: 0.25 | 0.50 | 0.75 | ``random``
      - ``fill_tol``: success band half-width (default 0.05 = ±5%)
      - ``randomize_layout``: seed-randomized non-overlapping station pose
      - ``force_thresholds``: 4 increasing force cutoffs (N)
      - ``beans_per_force_level``: beans dispensed at each force level
      - ``beans_full``: bean count that packs to the rim / 100%
      - ``scene_id``: 0 | 1 | 2 (KitchenS fixture layout)

    Peak button press force maps to four dispense levels; fill rises with beans.
    Success when fill ∈ [target_fill − fill_tol, target_fill + fill_tol].
    """

    BEAN_MODEL = "252_coffee_bean"
    JAR_MODEL = "253_glass_jar"
    # Max beans that may be dispensed (enough to pass the 25% ring densely).
    BEANS_FULL = 160
    # Four force thresholds (N) → beans per press level (light → hard).
    # Bean counts are spaced so each level raises fill by ~6.25% of the jar
    # (beans_full=160 → 10/20/30/40 beans ≈ 6%/13%/19%/25% of the rim).
    FORCE_THRESHOLDS = (3.0, 6.0, 10.0, 14.0)
    BEANS_PER_FORCE_LEVEL = (10, 20, 30, 40)
    # Spring proxy (N/m): F = k * tip engagement into the button zone.
    # Zone starts ~5 cm above the button so force builds as the tip descends
    # toward contact (the arm cannot penetrate the static button collider).
    FORCE_STIFFNESS = 300.0
    FORCE_ENGAGE_SLACK = 0.05
    # Expert press depths from hover for force levels 1..4 (stay above button).
    PRESS_DEPTHS = (0.020, 0.030, 0.044, 0.057)
    PRESS_SAMPLE_S = 0.40  # hold time to sample peak force
    FILL_LEVELS = (0.25, 0.50, 0.75)
    FILL_TOL = 0.05  # success band: target_fill ± fill_tol
    # Station footprints for non-overlap layout (half-extents, meters).
    DISP_HALF_XY = (0.065, 0.065)
    JAR_HALF_XY = (0.045, 0.045)
    LAYOUT_MARGIN = 0.025
    # Dense mound packing (used for freeze + bean-need estimates).
    _BEAN_R = 0.0055
    _BEAN_H = 0.0065
    _PILE_R_SCALE = 0.72

    # Glass-box dispenser (inspired by reference photo — tall clear column on a base).
    BOX_HALF = (0.035, 0.035, 0.090)       # tall slender glass box
    PEDESTAL_HALF = (0.052, 0.052, 0.050)  # raises the hopper above the jar
    PLATFORM_HALF = (0.058, 0.058, 0.008)  # platform between pedestal and hopper
    # Red push button on top of the glass lid (press target).
    BTN_BASE_HALF = (0.016, 0.016, 0.003)
    BTN_HALF = (0.011, 0.011, 0.010)
    BTN_BASE_COLOR = (0.12, 0.12, 0.14)
    BTN_COLOR = (0.90, 0.12, 0.10)
    BTN_TOUCH_XY_TOL = 0.028  # m; fingertip XY tolerance around button center
    BEAN_FILL_FRAC = 0.65                  # visual fill inside the glass box
    EE_TO_TCP = 0.12
    KEY_HOVER_DIS = 0.06
    KEY_PRESS_DEPTH = 0.057  # default = hardest force level
    SETTLE_STEPS = 80

    JAR_INNER_R = 0.035
    JAR_HEIGHT = 0.125
    JAR_BOTTOM_T = 0.005

    GLASS = [0.88, 0.95, 0.98, 0.14]
    # Interactive viewer look (matches trap_bug plain trap): no transmission/IOR.
    PLAIN_GLASS = [0.18, 0.32, 0.48, 0.55]
    BEAN_BROWN = [0.30, 0.14, 0.05]
    RING_RED = [0.95, 0.05, 0.05]

    def setup_demo(self, **kwags):
        self._cfg = dict(kwags.get("task_args", {}).get("fill_coffee_jar", {}))
        if kwags.get("scene_id") is None:
            kwags["scene_id"] = int(self._cfg.get("scene_id", 0))
        self._layout_seed = int(kwags.get("seed", 0) or 0)
        self.replace_sink_with_range = True
        self.omit_sink = True  # solid counter; no sink basin or faucet tap

        self._loaded = False
        self.beans = []
        self.beans_in_jar = 0
        self.press_count = 0
        self.target_fill = 0.25
        self.dispenser_touch_surface = None
        self.jar = None
        self.jar_visual = None
        self.fill_visual = None
        self._touch_latched = False
        self._dispensing = False
        self._press_active = False
        self._press_steps = 0
        self._press_spawned = 0
        self._press_hold_s = 0.0
        self._press_peak_force = 0.0
        self._press_force_level = 0
        self.table_top = 0.74
        self._plain_glass = bool(self._cfg.get("plain_glass", False))

        super().setup_demo(**kwags)
        self._configure_observer_camera()

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

    # ------------------------------------------------------------------ helpers
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
            try:
                mat.ior = 1.0
            except Exception:
                pass
        return mat

    def _glass_material(self, rgba=None, transmission=0.90):
        """Nearly-clear glass matching trap_bug / reference acrylic look."""
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
            try:
                mat.ior = 1.45
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

    def _add_static_box(self, pose, half_size, material=None, color=None, name="", collision=True):
        """Build a static box; supports translucent glass via RenderMaterial."""
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

    def _build_hollow_box(self, pose, half_size, wall, material, name, open_top=False, collision=True):
        """Hollow rectangular shell (4 walls + optional lid + bottom) — true glass box."""
        hx, hy, hz = [float(v) for v in half_size]
        wt = float(wall)
        top_hz = wt * 0.5
        side_hz = hz - (0.0 if open_top else top_hz)
        # Keep a floor so beans / fill sit inside.
        floor_hz = wt * 0.5
        side_z = -hz + floor_hz + side_hz
        parts = [
            (sapien.Pose([0, 0, -hz + floor_hz]), [hx, hy, floor_hz]),  # bottom
            (sapien.Pose([hx - wt * 0.5, 0, side_z]), [wt * 0.5, hy, side_hz]),
            (sapien.Pose([-hx + wt * 0.5, 0, side_z]), [wt * 0.5, hy, side_hz]),
            (sapien.Pose([0, hy - wt * 0.5, side_z]), [hx - wt, wt * 0.5, side_hz]),
            (sapien.Pose([0, -hy + wt * 0.5, side_z]), [hx - wt, wt * 0.5, side_hz]),
        ]
        if not open_top:
            parts.append((sapien.Pose([0, 0, hz - top_hz]), [hx, hy, top_hz]))

        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        if collision:
            for local_pose, half in parts:
                builder.add_box_collision(
                    pose=local_pose,
                    half_size=list(half),
                    material=self.scene.default_physical_material,
                )
        for local_pose, half in parts:
            builder.add_box_visual(pose=local_pose, half_size=list(half), material=material)
        builder.set_initial_pose(pose)
        return builder.build(name=name)

    def _add_static_mesh_visual(self, filename, pose, material, name):
        """Add a smooth mesh visual while forcing the intended render material."""
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        builder.add_visual_from_file(
            filename=str(Path(filename).resolve()),
            material=material,
        )
        builder.set_initial_pose(pose)
        return builder.build(name=name)

    # ------------------------------------------------------------------ layout
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

    def _layout_blockers(self):
        """Kitchen fixture footprints the station must not overlap."""
        blockers = []
        mw_xy = getattr(self, "microwave_xy", None)
        mw_half = getattr(self, "microwave_half_xy", None)
        if mw_xy is not None and mw_half is not None:
            blockers.append((np.asarray(mw_xy, dtype=float), np.asarray(mw_half, dtype=float)))
        range_xy = getattr(self, "range_xy", None)
        range_half = getattr(self, "range_half_size", None)
        if range_xy is not None and range_half is not None:
            blockers.append(
                (np.asarray(range_xy, dtype=float), np.asarray(range_half, dtype=float))
            )
        return blockers

    def _station_clear(self, side_x, disp_y, jar_y, blockers, margin=None):
        if margin is None:
            margin = self.LAYOUT_MARGIN
        disp = np.array([side_x, disp_y], dtype=float)
        jar = np.array([side_x, jar_y], dtype=float)
        # Dispenser ↔ jar must not overlap each other.
        if self._aabb_overlap(disp, self.DISP_HALF_XY, jar, self.JAR_HALF_XY, margin):
            return False
        for b_c, b_h in blockers:
            if self._aabb_overlap(disp, self.DISP_HALF_XY, b_c, b_h, margin):
                return False
            if self._aabb_overlap(jar, self.JAR_HALF_XY, b_c, b_h, margin):
                return False
        return True

    def _resolve_target_fill(self, cfg, rng: np.random.RandomState) -> float:
        tf = cfg.get("target_fill", 0.25)
        if tf is None:
            return 0.25
        if isinstance(tf, str) and tf.lower() == "random":
            return float(rng.choice(self.FILL_LEVELS))
        val = float(tf)
        if val not in self.FILL_LEVELS:
            raise ValueError(f"target_fill must be one of {self.FILL_LEVELS} or 'random'")
        return val

    def _sample_station_layout(self, cfg, rng: np.random.RandomState):
        """Place dispenser+jar on the left arm side without fixture overlap.

        Keeps a shared X (nozzle alignment) with the jar in front (−y) of the
        dispenser. When ``randomize_layout`` is false, uses fixed config defaults.
        """
        randomize = bool(cfg.get("randomize_layout", False))
        blockers = self._layout_blockers()
        # Jitter around the proven left-arm station (reachability is tight).
        base_x = float(cfg.get("station_x", -0.08))
        base_disp_y = float(cfg.get("disp_y", -0.02))
        base_jar_y = float(cfg.get("jar_y", -0.16))

        if not randomize:
            side_x, disp_y, jar_y = base_x, base_disp_y, base_jar_y
            if not self._station_clear(side_x, disp_y, jar_y, blockers):
                for x in np.linspace(side_x, -0.05, 20):
                    if self._station_clear(float(x), disp_y, jar_y, blockers):
                        side_x = float(x)
                        break
            return side_x, disp_y, jar_y

        for _ in range(80):
            # Small jitter only — larger offsets make the tall lid approach unplannable.
            side_x = float(np.clip(base_x + rng.uniform(-0.015, 0.015), -0.11, -0.05))
            disp_y = float(np.clip(base_disp_y + rng.uniform(-0.015, 0.015), -0.04, 0.01))
            gap = float((base_disp_y - base_jar_y) + rng.uniform(-0.015, 0.015))
            gap = float(np.clip(gap, 0.125, 0.155))
            jar_y = disp_y - gap
            if jar_y < -0.20:
                continue
            if self._station_clear(side_x, disp_y, jar_y, blockers):
                return side_x, disp_y, jar_y

        # Deterministic fallback that clears scene_0 fixtures.
        return base_x, base_disp_y, base_jar_y

    # ------------------------------------------------------------------ actors
    def load_actors(self):
        cfg = self._cfg
        self.table_top = float(self.kitchens_info["table_height"]) + float(self.table_z_bias)
        seed = int(getattr(self, "_layout_seed", 0) or 0)
        # Separate streams so target choice doesn't collide with layout jitter.
        rng_target = np.random.RandomState(seed + 202)
        rng_layout = np.random.RandomState(seed + 101)

        self.target_fill = self._resolve_target_fill(cfg, rng_target)
        self.beans_full = int(cfg.get("beans_full", self.BEANS_FULL))
        thr = cfg.get("force_thresholds", self.FORCE_THRESHOLDS)
        beans_lv = cfg.get("beans_per_force_level", self.BEANS_PER_FORCE_LEVEL)
        self.force_thresholds = tuple(float(x) for x in thr)
        self.beans_per_force_level = tuple(int(x) for x in beans_lv)
        if len(self.force_thresholds) != 4 or len(self.beans_per_force_level) != 4:
            raise ValueError("force_thresholds and beans_per_force_level need length 4")
        if any(
            self.force_thresholds[i] >= self.force_thresholds[i + 1]
            for i in range(3)
        ):
            raise ValueError("force_thresholds must be strictly increasing")
        self.force_stiffness = float(cfg.get("force_stiffness", self.FORCE_STIFFNESS))
        self.force_engage_slack = float(
            cfg.get("force_engage_slack", self.FORCE_ENGAGE_SLACK)
        )
        depths = cfg.get("press_depths", self.PRESS_DEPTHS)
        self.press_depths = tuple(float(x) for x in depths)
        if len(self.press_depths) != 4:
            raise ValueError("press_depths needs length 4")
        self.press_sample_s = float(cfg.get("press_sample_s", self.PRESS_SAMPLE_S))
        # Convenience aliases used by fill estimates / expert loop.
        self.beans_per_press_min = int(self.beans_per_force_level[0])
        self.beans_per_press_max = int(self.beans_per_force_level[-1])
        self.fill_tol = float(cfg.get("fill_tol", self.FILL_TOL))
        self.beans = []
        self.beans_in_jar = 0
        self.press_count = 0
        self._touch_latched = False
        self._dispensing = False
        self._press_active = False
        self._press_steps = 0
        self._press_spawned = 0
        self._press_hold_s = 0.0
        self._press_peak_force = 0.0
        self._press_force_level = 0

        side_x, disp_y, jar_y = self._sample_station_layout(cfg, rng_layout)
        self.dispenser_xy = np.array([side_x, disp_y], dtype=float)
        self.jar_xy = np.array([side_x, jar_y], dtype=float)
        self.layout_ok = bool(
            self._station_clear(side_x, disp_y, jar_y, self._layout_blockers())
        )

        self._build_dispenser()
        self._build_jar()
        self._build_fill_rings()

        self.add_prohibit_area(sapien.Pose([*self.dispenser_xy, self.table_top + 0.1]), padding=0.08)
        self.add_prohibit_area(sapien.Pose([*self.jar_xy, self.table_top + 0.05]), padding=0.05)

        self._loaded = True
        levels = ", ".join(
            f"≥{t:.0f}N→{n}" for t, n in zip(self.force_thresholds, self.beans_per_force_level)
        )
        print(
            f"[fill_coffee_jar] KitchenS scene={self.scene_id} seed={seed} "
            f"target={self.target_fill:.0%}±{self.fill_tol:.0%} "
            f"disp={self.dispenser_xy.tolist()} jar={self.jar_xy.tolist()} "
            f"layout_ok={self.layout_ok} "
            f"(~{self._beans_needed()}/{self.beans_full} beans; force {levels})"
        )

    def _build_dispenser(self):
        """Raised glass bean hopper; red top button opens a nozzle over the jar."""
        x, y = self.dispenser_xy
        z0 = self.table_top
        bx, by, bz = self.BOX_HALF
        px, py, pz = self.PLATFORM_HALF
        _, _, pedestal_hz = self.PEDESTAL_HALF
        wall = 0.0035
        glass = self._glass_material()

        # A simple solid pedestal raises the hopper high enough for the jar and nozzle.
        self._add_static_box(
            pose=sapien.Pose([x, y, z0 + pedestal_hz]),
            half_size=self.PEDESTAL_HALF,
            color=[0.13, 0.13, 0.15, 1.0],
            name="dispenser_pedestal",
        )
        self._add_static_box(
            pose=sapien.Pose([x, y, z0 + 2.0 * pedestal_hz + pz]),
            half_size=self.PLATFORM_HALF,
            color=[0.10, 0.10, 0.12, 1.0],
            name="dispenser_platform",
        )

        # Tall hollow clear hopper above the pedestal.
        hopper_bottom_z = z0 + 2.0 * pedestal_hz + 2.0 * pz
        box_z = hopper_bottom_z + bz
        self._build_hollow_box(
            pose=sapien.Pose([x, y, box_z]),
            half_size=self.BOX_HALF,
            wall=wall,
            material=glass,
            name="dispenser_glass_box",
            open_top=True,
            collision=True,
        )
        # Clear lid (visual/structural only) — press target is the button above.
        lid_hz = 0.003
        lid_z = box_z + bz + lid_hz
        self._add_static_box(
            pose=sapien.Pose([x, y, lid_z]),
            half_size=[bx * 1.01, by * 1.01, lid_hz],
            material=self._glass_material([0.90, 0.96, 0.99, 0.22]),
            name="dispenser_lid",
            collision=True,
        )
        lid_top = lid_z + lid_hz
        # Dark collar + red push button centered on the lid.
        bbx, bby, bbz = self.BTN_BASE_HALF
        bhx, bhy, bhz = self.BTN_HALF
        base_z = lid_top + bbz
        btn_z = lid_top + 2.0 * bbz + bhz
        self._add_static_box(
            pose=sapien.Pose([x, y, base_z]),
            half_size=[bbx, bby, bbz],
            color=[*self.BTN_BASE_COLOR, 1.0],
            name="dispenser_button_base",
            collision=True,
        )
        self.dispenser_touch_surface = self._add_static_box(
            pose=sapien.Pose([x, y, btn_z]),
            half_size=[bhx, bhy, bhz],
            color=[*self.BTN_COLOR, 1.0],
            name="dispenser_push_button",
            collision=True,
        )

        # One packed mesh containing many individual coffee beans (not a solid block).
        self._add_static_mesh_visual(
            filename=Path("assets/objects/252_coffee_bean/reservoir_fill.glb"),
            pose=sapien.Pose([x, y, box_z]),
            material=self._opaque_material(self.BEAN_BROWN),
            name="dispenser_reservoir_beans",
        )

        # Nozzle ends short of jar center so the fill column stays visible.
        nozzle_joint_z = self.table_top + self.JAR_HEIGHT + 0.070
        nozzle_outlet_z = self.table_top + self.JAR_HEIGHT + 0.035
        hopper_front_y = y - by
        jar_x, jar_y = self.jar_xy
        tip_y = jar_y + 0.018
        nozzle_y = 0.5 * (hopper_front_y + tip_y)
        self._add_static_box(
            pose=sapien.Pose([x, nozzle_y, nozzle_joint_z]),
            half_size=[0.006, abs(tip_y - hopper_front_y) * 0.5, 0.005],
            color=[0.42, 0.44, 0.47, 1.0],
            name="dispenser_nozzle_arm",
        )
        self._add_static_box(
            pose=sapien.Pose(
                [jar_x, tip_y, 0.5 * (nozzle_joint_z + nozzle_outlet_z)]
            ),
            half_size=[
                0.006,
                0.006,
                0.5 * (nozzle_joint_z - nozzle_outlet_z),
            ],
            color=[0.36, 0.38, 0.41, 1.0],
            name="dispenser_nozzle_tip",
        )
        self._add_static_box(
            pose=sapien.Pose([jar_x, tip_y, nozzle_outlet_z]),
            half_size=[0.007, 0.007, 0.002],
            color=[0.10, 0.08, 0.06, 1.0],
            name="dispenser_nozzle_opening",
            collision=False,
        )
        self.nozzle_outlet_xyz = np.array(
            [jar_x, jar_y, nozzle_outlet_z], dtype=float
        )

        self.touch_xy = np.array([x, y], dtype=float)
        self.touch_top_z = float(btn_z + bhz)

    def _build_jar(self):
        """Clear glass cylinder (original jar design) — no handle/spout.

        Smooth see-through cylinder via ``RenderShapeCylinder`` (IOR=1) + thin
        floor disk. Collision from the hollow jar mesh (no GLB visual).
        """
        x, y = self.jar_xy
        z0 = self.table_top + 0.001
        outer_r = self.JAR_INNER_R + 0.0035
        h = self.JAR_HEIGHT
        bottom_t = self.JAR_BOTTOM_T
        upright_q = [0.70710678, 0.0, -0.70710678, 0.0]

        col_path = Path(
            f"assets/objects/{self.JAR_MODEL}/collision/base0.glb"
        ).resolve()
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        builder.add_nonconvex_collision_from_file(filename=str(col_path), scale=[1, 1, 1])
        builder.set_initial_pose(sapien.Pose([x, y, z0]))
        self.jar = builder.build(name="glass_jar")
        try:
            self.jar.set_name("glass_jar")
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

    def _build_fill_rings(self):
        """Add three subtle, thin red rings around the glass jar body."""
        x, y = self.jar_xy
        ring_material = self._opaque_material([0.78, 0.05, 0.05], 0.70)
        ring_mesh = Path(f"assets/objects/{self.JAR_MODEL}/rings/thin_ring.glb")
        for frac in (0.25, 0.50, 0.75):
            z = self.jar_bottom_z + frac * self.jar_fillable_h
            self._add_static_mesh_visual(
                filename=ring_mesh,
                pose=sapien.Pose([x, y, z]),
                material=ring_material,
                name=f"fill_ring_{int(frac * 100)}",
            )

    # ------------------------------------------------------------------ dispense / fill
    def _beans_per_layer(self) -> int:
        pile_r = self.JAR_INNER_R * self._PILE_R_SCALE
        max_ring = max(1, int((pile_r - self._BEAN_R) / (2.0 * self._BEAN_R * 0.95)))
        n = 1
        for ring in range(1, max_ring + 1):
            n += max(6, int(round(2.0 * math.pi * ring)))
        return n

    def _beans_needed(self) -> int:
        """Beans required to reach the target fill fraction of the jar."""
        return int(math.ceil(float(self.target_fill) * float(self.beans_full)))

    def _beans_in_jar_list(self):
        x, y = self.jar_xy
        r = self.JAR_INNER_R + 0.008
        z_lo = self.jar_bottom_z - 0.005
        z_hi = self.table_top + self.JAR_HEIGHT + 0.02
        out = []
        for b in self.beans:
            p = np.asarray(b.get_pose().p, dtype=float)
            if (p[0] - x) ** 2 + (p[1] - y) ** 2 <= r * r and z_lo <= p[2] <= z_hi:
                out.append(b)
        return out

    def _count_beans_in_jar(self) -> int:
        return len(self._beans_in_jar_list())

    def _effective_bean_count(self) -> int:
        """Beans that count toward fill (in-jar + still streaming this press)."""
        n = int(getattr(self, "beans_in_jar", 0) or 0)
        if getattr(self, "_press_active", False):
            n += int(getattr(self, "_press_spawned", 0) or 0)
        return int(min(self.beans_full, max(0, n)))

    def _current_fill(self) -> float:
        """Fill fraction vs red-ring scale — proportional to beans dispensed.

        More coffee out → higher level. Uses bean count (not packed max-Z) so
        small force-level differences still move the visible column.
        """
        return float(self._effective_bean_count()) / float(max(1, self.beans_full))

    def _pile_height(self) -> float:
        """Coffee column height above the jar floor (meters), from fill fraction."""
        return float(self._current_fill() * self.jar_fillable_h)

    def _spawn_bean(self, pose: sapien.Pose):
        bean = create_actor(
            self,
            pose=pose,
            modelname=self.BEAN_MODEL,
            model_id=0,
            convex=True,
            is_static=False,
        )
        bean.set_mass(0.003)
        bean.set_name(f"coffee_bean_{len(self.beans)}")
        # Force dark-brown tint and damp so beans settle into a pile.
        try:
            for c in bean.actor.get_components():
                if isinstance(c, sapien.render.RenderBodyComponent):
                    for s in c.render_shapes:
                        try:
                            s.material.set_base_color([*self.BEAN_BROWN, 1.0])
                        except Exception:
                            pass
                if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                    c.set_linear_damping(1.8)
                    c.set_angular_damping(2.0)
                    try:
                        c.set_linear_velocity([0.0, 0.0, -0.35])
                        c.set_angular_velocity(np.zeros(3))
                    except Exception:
                        pass
        except Exception:
            pass
        self.beans.append(bean)
        return bean

    def _freeze_beans(self, beans):
        """Freeze beans into a mound whose top matches the bean-count fill height."""
        x, y = self.jar_xy
        pile_r = self.JAR_INNER_R * self._PILE_R_SCALE
        inside = []
        for bean in beans:
            p = np.asarray(bean.get_pose().p, dtype=float)
            if (p[0] - x) ** 2 + (p[1] - y) ** 2 <= (self.JAR_INNER_R - 0.002) ** 2:
                inside.append(bean)
        if not inside:
            return

        bean_r = self._BEAN_R
        bean_h = self._BEAN_H
        # Target top of the pile from dispensed count (same scale as fill column).
        fill = min(1.0, float(len(inside)) / float(max(1, self.beans_full)))
        target_top = self.jar_bottom_z + fill * self.jar_fillable_h
        max_ring = max(1, int((pile_r - bean_r) / (2.0 * bean_r * 0.95)))
        positions = []
        layer = 0
        while len(positions) < len(inside):
            z = self.jar_bottom_z + bean_h * 0.5 + layer * bean_h * 0.92
            # Cap so we never pack above the rim.
            if z > self.jar_bottom_z + self.jar_fillable_h - bean_h * 0.2:
                z = self.jar_bottom_z + self.jar_fillable_h - bean_h * 0.2
            slots = [(0.0, 0.0)]
            for ring in range(1, max_ring + 1):
                n_ring = max(6, int(round(2.0 * math.pi * ring)))
                rad = ring * (2.0 * bean_r * 0.95)
                if rad > pile_r - bean_r:
                    break
                for k in range(n_ring):
                    ang = 2.0 * math.pi * k / n_ring + (0.15 * layer)
                    slots.append((rad * math.cos(ang), rad * math.sin(ang)))
            for sx, sy in slots:
                if len(positions) >= len(inside):
                    break
                jx = float(np.random.uniform(-0.0008, 0.0008))
                jy = float(np.random.uniform(-0.0008, 0.0008))
                positions.append((x + sx + jx, y + sy + jy, z))
            layer += 1
            if layer > 60:
                break

        # Stretch / compress layers so the mound top tracks fill fraction.
        if positions:
            z0 = self.jar_bottom_z + bean_h * 0.5
            z_max = max(pz for _, _, pz in positions)
            span = max(1e-6, z_max - z0)
            scale = max(0.15, (target_top - z0) / span)
            positions = [
                (px, py, z0 + (pz - z0) * scale) for px, py, pz in positions
            ]

        for bean, (px, py, pz) in zip(inside, positions):
            yaw = float(np.random.uniform(0, 2 * np.pi))
            qx, qy, qz, qw = t3d.euler.euler2quat(0.25, 0.15, yaw)
            rest = sapien.Pose([px, py, pz], [qw, qx, qy, qz])
            # Actor wrapper has no set_pose — must move the underlying entity
            # or the render mesh stays on the jar floor while physx reports high.
            try:
                bean.actor.set_pose(rest)
            except Exception:
                pass
            try:
                for component in bean.actor.get_components():
                    if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
                        component.set_linear_velocity(np.zeros(3))
                        component.set_angular_velocity(np.zeros(3))
                        component.set_kinematic(True)
                        component.set_kinematic_target(rest)
            except Exception:
                pass

    def _sync_fill_visual(self):
        """Opaque coffee column proportional to beans dispensed (vs red rings).

        Top-down cameras flatten a real bean stack into a small floor sprinkle, so
        the fill body is what makes force-level differences readable.
        """
        if getattr(self, "fill_visual", None) is not None:
            try:
                self.scene.remove_entity(self.fill_visual)
            except Exception:
                pass
            self.fill_visual = None

        h = float(self._pile_height())
        if h < 0.003:
            return

        x, y = self.jar_xy
        upright_q = [0.70710678, 0.0, -0.70710678, 0.0]
        half = h * 0.5
        mat = sapien.render.RenderMaterial(base_color=[0.22, 0.10, 0.04, 0.92])
        try:
            mat.set_transmission(0.0)
            mat.set_roughness(0.85)
            mat.set_metallic(0.0)
        except Exception:
            pass

        ent = sapien.Entity()
        ent.set_name("coffee_fill_visual")
        ent.set_pose(sapien.Pose([x, y, self.jar_bottom_z]))
        body = sapien.render.RenderBodyComponent()
        col = sapien.render.RenderShapeCylinder(
            radius=self.JAR_INNER_R * 0.90,
            half_length=half,
            material=mat,
        )
        col.set_local_pose(sapien.Pose([0.0, 0.0, half], upright_q))
        body.attach(col)
        ent.add_component(body)
        self.scene.add_entity(ent)
        self.fill_visual = ent

    def _sim_dt(self) -> float:
        try:
            return float(self.scene.get_timestep())
        except Exception:
            return 1.0 / 250.0

    def _lid_contact_force(self) -> float:
        """PhysX contact force (N) between gripper links and the push button."""
        if not hasattr(self, "robot"):
            return 0.0
        dt = self._sim_dt()
        btn = "dispenser_push_button"
        grip = set(getattr(self.robot, "gripper_name", []) or [])
        imp = 0.0
        try:
            for contact in self.scene.get_contacts():
                n0 = contact.bodies[0].entity.name
                n1 = contact.bodies[1].entity.name
                if btn not in (n0, n1):
                    continue
                other = n1 if n0 == btn else n0
                if grip and other not in grip:
                    # Still accept left-arm / finger-like links.
                    o = other.lower()
                    if not (
                        o.startswith("left")
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

    def _lid_spring_force(self) -> float:
        """Spring proxy (N) from fingertip engagement into the button zone.

        Position-controlled presses often yield sparse contact impulses against a
        static button; the spring fills that gap so press depth ↔ force stays
        correlated for both the expert and learned policies.
        """
        if not hasattr(self, "robot"):
            return 0.0
        try:
            ee = np.asarray(self.robot.get_left_ee_pose()[:3], dtype=float)
        except Exception:
            return 0.0
        tip_z = float(ee[2] - self.EE_TO_TCP)
        engage_z = float(self.touch_top_z + self.force_engage_slack)
        pen = max(0.0, engage_z - tip_z)
        return float(self.force_stiffness * pen)

    def _lid_press_force(self) -> float:
        """Effective button press force (N) used for the 4 dispense thresholds.

        Position-controlled presses into a static button produce huge, noisy PhysX
        impulse spikes (often >100 N) that would collapse every press to the
        hardest level. The spring engagement force is a stable, depth-correlated
        signal that matches the expert press depths to the four thresholds.
        Contact impulse is still available via ``_lid_contact_force`` for debug.
        """
        return float(self._lid_spring_force())

    def _force_level(self, force_n: float) -> int:
        """Map force (N) → level in {0,1,2,3,4} (0 = below first threshold)."""
        level = 0
        for i, thr in enumerate(self.force_thresholds):
            if float(force_n) >= float(thr):
                level = i + 1
        return int(level)

    def _beans_for_force(self, force_n: float) -> int:
        """Map force (N) → bean count via the four thresholds."""
        level = self._force_level(force_n)
        if level <= 0:
            return 0
        return int(self.beans_per_force_level[level - 1])

    def _spawn_one_dispensed_bean(self):
        """Drop a single bean from the nozzle into the jar."""
        ang = float(np.random.uniform(0, 2 * np.pi))
        rad = float(np.random.uniform(0.0, 0.014))
        ox = rad * math.cos(ang)
        oy = rad * math.sin(ang)
        yaw = float(np.random.uniform(0, 2 * np.pi))
        qx, qy, qz, qw = t3d.euler.euler2quat(0.4, 0.2, yaw)
        pose = sapien.Pose(
            [
                self.nozzle_outlet_xyz[0] + ox,
                self.nozzle_outlet_xyz[1] + oy,
                self.nozzle_outlet_xyz[2],
            ],
            [qw, qx, qy, qz],
        )
        self._spawn_bean(pose)

    def _start_press(self):
        if self._press_active or self.beans_in_jar >= self.beans_full:
            return
        self._press_active = True
        self._dispensing = True
        self._press_steps = 0
        self._press_spawned = 0
        self._press_hold_s = 0.0
        self._press_peak_force = 0.0
        self._press_force_level = 0

    def _tick_press(self):
        """While the button is held, stream beans according to peak press force."""
        if not self._press_active:
            return
        if self.beans_in_jar + self._press_spawned >= self.beans_full:
            return
        self._press_steps += 1
        self._press_hold_s = self._press_steps * self._sim_dt()
        force_n = self._lid_press_force()
        if force_n > self._press_peak_force:
            self._press_peak_force = force_n
        self._press_force_level = self._force_level(self._press_peak_force)
        target = self._beans_for_force(self._press_peak_force)
        target = min(target, self.beans_full - self.beans_in_jar)
        while self._press_spawned < target:
            self._spawn_one_dispensed_bean()
            self._press_spawned += 1
            # Raise the fill column as beans stream out (force → amount → level).
            if self._press_spawned == target or self._press_spawned % 3 == 0:
                self._sync_fill_visual()
            # Small settle between beans so the cascade stays visible.
            for _ in range(8):
                super()._update_kinematic_tasks()
                self.scene.step()

    def _end_press(self):
        """Finalize a press: top up to the force-level mapping, settle, freeze."""
        if not self._press_active:
            return
        # One last force sample in case the peak arrived on the release frame.
        force_n = self._lid_press_force()
        if force_n > self._press_peak_force:
            self._press_peak_force = force_n
        self._press_force_level = self._force_level(self._press_peak_force)
        # Ignore ghost touches during approach (below first threshold → no beans).
        if self._press_force_level <= 0 and int(self._press_spawned) <= 0:
            self._press_active = False
            self._dispensing = False
            self._press_steps = 0
            self._press_spawned = 0
            self._press_hold_s = 0.0
            self._press_peak_force = 0.0
            self._press_force_level = 0
            return
        want = self._beans_for_force(self._press_peak_force)
        want = min(want, self.beans_full - self.beans_in_jar)
        while self._press_spawned < want:
            self._spawn_one_dispensed_bean()
            self._press_spawned += 1
            if self._press_spawned == want or self._press_spawned % 3 == 0:
                self._sync_fill_visual()
            for _ in range(8):
                super()._update_kinematic_tasks()
                self.scene.step()

        self.press_count += 1
        spawned = int(self._press_spawned)
        peak = float(self._press_peak_force)
        level = int(self._press_force_level)
        # Close the press session before settle so touch-detect cannot re-enter.
        self._press_active = False
        self._dwell(self.SETTLE_STEPS)
        self._freeze_beans(self.beans)
        self._dwell(6)
        self.beans_in_jar = self._count_beans_in_jar()
        self._sync_fill_visual()
        fill = self._current_fill()
        print(
            f"[fill_coffee_jar] force={peak:.1f}N level={level}/4 → {spawned} beans "
            f"fill={fill:.0%} "
            f"(thresholds {list(self.force_thresholds)} → {list(self.beans_per_force_level)})"
        )
        self._dispensing = False
        self._press_steps = 0
        self._press_spawned = 0
        self._press_hold_s = 0.0
        self._press_peak_force = 0.0
        self._press_force_level = 0

    def _dwell(self, steps: int):
        for i in range(max(0, int(steps))):
            if not getattr(self, "_dispensing", False):
                self._update_kinematic_tasks()
            else:
                super()._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (i % self.save_freq == 0):
                self._take_picture()

    def _touch_tip_pose(self, tip_z_above_top: float):
        tcp_z = self.touch_top_z + tip_z_above_top
        ee_z = tcp_z + self.EE_TO_TCP
        return [float(self.touch_xy[0]), float(self.touch_xy[1]), ee_z, *GRASP_DIRECTION_DIC["top_down"]]

    def _press_dispenser(self, arm_tag: ArmTag, force_level: int = 4):
        """Press the top button to the requested force level (1–4) and dispense."""
        level = int(np.clip(force_level, 1, 4))
        depth = float(self.press_depths[level - 1])
        # Floor for this press: commanded level's threshold (expert intent).
        commanded_force = float(self.force_thresholds[level - 1])
        # Two-stage approach: high waypoint, then drop to hover (more reliable IK).
        high_dis = self.KEY_HOVER_DIS + 0.08
        self.move(self.move_to_pose(arm_tag, self._touch_tip_pose(high_dis)))
        if not self.plan_success:
            # Fallback: direct hover.
            self.plan_success = True
            self.move(self.move_to_pose(arm_tag, self._touch_tip_pose(self.KEY_HOVER_DIS)))
        else:
            self.move(self.move_by_displacement(arm_tag, z=-(high_dis - self.KEY_HOVER_DIS)))
        if not self.plan_success:
            print(f"[fill_coffee_jar] hover failed to {self._touch_tip_pose(self.KEY_HOVER_DIS)}")
            return False
        self.move(self.move_by_displacement(arm_tag, z=-depth))
        if not self.plan_success:
            return False

        # Sample peak force while pressed (expert path; does not rely on the
        # per-step touch detector, which can miss during move()).
        self._start_press()
        self._press_peak_force = commanded_force
        self._press_force_level = level
        hold_steps = max(1, int(round(self.press_sample_s / self._sim_dt())))
        for _ in range(hold_steps):
            measured = self._lid_press_force()
            if measured > self._press_peak_force:
                self._press_peak_force = measured
            # Keep at least the commanded level even if the tip sits a bit high.
            self._press_peak_force = max(self._press_peak_force, commanded_force)
            self._tick_press()
            if not self._press_active:
                break
            super()._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (self._press_steps % max(1, int(self.save_freq)) == 0):
                self._take_picture()
        self._end_press()

        self.move(self.move_by_displacement(arm_tag, z=depth))
        self._dwell(8)
        return True

    def _detect_lid_touch(self):
        if self.dispenser_touch_surface is None or not hasattr(self, "robot"):
            return
        try:
            ee = np.asarray(self.robot.get_left_ee_pose()[:3], dtype=float)
        except Exception:
            return
        xy_tol = float(getattr(self, "BTN_TOUCH_XY_TOL", 0.028))
        xy_ok = float(np.linalg.norm(ee[:2] - self.touch_xy)) <= xy_tol
        # Engage when the fingertip enters the force-sensing zone above the button.
        z_ok = ee[2] <= self.touch_top_z + self.EE_TO_TCP + self.force_engage_slack
        touching = bool(xy_ok and z_ok and self._lid_press_force() > 0.5)
        if touching and not self._touch_latched:
            self._start_press()
        if touching and self._press_active:
            self._tick_press()
            # Auto-finish once the hardest level has been held briefly.
            if (
                self._press_force_level >= 4
                and self._press_hold_s >= self.press_sample_s
            ):
                self._end_press()
                touching = False
        if (not touching) and self._touch_latched and self._press_active:
            self._end_press()
        self._touch_latched = touching

    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
        if not getattr(self, "_loaded", False):
            return
        # While actively holding we still need touch edge detection for release
        # / max-duration stop; `_tick_press` is driven from `_detect_lid_touch`.
        if getattr(self, "_dispensing", False) and not self._press_active:
            return
        self._detect_lid_touch()

    # ------------------------------------------------------------------ expert / success
    def _fill_band(self):
        lo = float(self.target_fill) - float(self.fill_tol)
        hi = float(self.target_fill) + float(self.fill_tol)
        return max(0.0, lo), min(1.0, hi)

    def _force_level_for_remaining(self, fill: float) -> int:
        """Pick the smallest force level that enters the success band without overshoot."""
        _, hi = self._fill_band()
        lo, _ = self._fill_band()
        if fill + 1e-6 >= lo:
            return 1
        need_frac = max(0.0, float(self.target_fill) - float(fill))
        need_beans = int(math.ceil(need_frac * float(self.beans_full)))
        # Prefer a level that lands inside [lo, hi].
        best = 4
        for i, n in enumerate(self.beans_per_force_level):
            pred = fill + float(n) / float(self.beans_full)
            if pred + 1e-6 >= lo and pred - 1e-6 <= hi:
                return i + 1
            if n >= need_beans:
                best = i + 1
                break
        return int(best)

    def play_once(self):
        # Station is constrained to the left half; use left arm.
        arm = ArmTag("left" if float(self.dispenser_xy[0]) <= 0.0 else "right")
        self.move(self.close_gripper(arm))
        if not self.plan_success:
            print("[fill_coffee_jar] close_gripper failed")
            return self.info

        needed = self._beans_needed()
        lo, hi = self._fill_band()
        max_presses = int(math.ceil(needed / max(1, self.beans_per_press_min))) + 4
        for i in range(max_presses):
            self.beans_in_jar = self._count_beans_in_jar()
            fill = self._current_fill()
            if fill + 1e-6 >= lo:
                break
            if not self.plan_success:
                print(f"[fill_coffee_jar] plan failed before press {i}")
                break
            level = self._force_level_for_remaining(fill)
            # Fresh plan flag each press (a prior miss shouldn't abort the episode).
            self.plan_success = True
            ok = self._press_dispenser(arm, force_level=level)
            if not ok:
                self.plan_success = True
                ok = self._press_dispenser(arm, force_level=level)
            self.beans_in_jar = self._count_beans_in_jar()
            fill = self._current_fill()
            print(
                f"[fill_coffee_jar] press={i} force_lvl={level} ok={ok} "
                f"beans={self.beans_in_jar}/{needed} fill={fill:.0%} "
                f"(band {lo:.0%}–{hi:.0%}) plan={self.plan_success}"
            )

        # Target band reached → withdraw.
        if self.plan_success:
            self.move(self.move_by_displacement(arm, z=0.08))

        if self.check_success():
            self.plan_success = True

        level_pct = int(round(self.target_fill * 100))
        self.info["info"] = {
            "{A}": "coffee dispenser",
            "{B}": f"glass jar ({level_pct}% line)",
            "{C}": "252_coffee_bean/base0",
            "{a}": str(arm),
            "{L}": f"{level_pct}%",
        }
        return self.info

    def check_success(self):
        """Success when fill is inside target_fill ± fill_tol (default ±5%)."""
        if not getattr(self, "layout_ok", True):
            return False
        self.beans_in_jar = self._count_beans_in_jar()
        fill = self._current_fill()
        lo, hi = self._fill_band()
        return bool(self.beans_in_jar > 0 and lo - 1e-3 <= fill <= hi + 1e-3)

    def get_obs(self):
        obs = super().get_obs()
        lo, hi = self._fill_band()
        obs["coffee_jar"] = {
            "target_fill": float(self.target_fill),
            "fill_tol": float(self.fill_tol),
            "fill_lo": float(lo),
            "fill_hi": float(hi),
            "fill": float(self._current_fill()),
            "pile_height": float(self._pile_height()),
            "beans_in_jar": int(self.beans_in_jar),
            "beans_full": int(self.beans_full),
            "force_thresholds": [float(x) for x in self.force_thresholds],
            "beans_per_force_level": [int(x) for x in self.beans_per_force_level],
            "lid_force": float(self._lid_press_force()) if self._loaded else 0.0,
            "press_count": int(self.press_count),
            "dispenser_xy": [float(x) for x in self.dispenser_xy],
            "jar_xy": [float(x) for x in self.jar_xy],
            "layout_ok": bool(getattr(self, "layout_ok", True)),
            "scene_id": int(self.scene_id),
        }
        return obs
