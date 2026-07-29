"""Fill a marked glass jar with coffee beans from a glass-box dispenser (KitchenS).

Inherits ``KitchenS_base_task`` (microwave + dishrack + cooking range on a kitchen
counter). The dispenser is a raised clear glass hopper packed with real bean
meshes. Touching its lid opens a nozzle above the jar and releases beans into a
glass jar marked with red ring lines at 25% / 50% / 75% (rim = full).
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
    """Touch the dispenser lid to fill a marked glass jar to a target level.

    Task options (``task_args.fill_coffee_jar``):
      - ``target_fill``: minimum *visual* fill vs red rings (default 0.25)
      - ``press_duration_min`` / ``press_duration_max``: hold time range (sec)
      - ``beans_per_press_min`` / ``beans_per_press``: beans at min / max hold
      - ``beans_full``: bean count that packs to the rim / 100%
      - ``scene_id``: 0 | 1 | 2 (KitchenS fixture layout)

    Hold duration maps linearly to dispense amount between the min/max
    endpoints (clamped). Success is pile height vs the red-ring scale.
    """

    BEAN_MODEL = "252_coffee_bean"
    JAR_MODEL = "253_glass_jar"
    # Max beans that may be dispensed (enough to pass the 25% ring densely).
    BEANS_FULL = 160
    # Hold-time → amount mapping (overridable via task_args).
    PRESS_DURATION_MIN = 0.35
    PRESS_DURATION_MAX = 1.40
    BEANS_PER_PRESS_MIN = 2
    BEANS_PER_PRESS = 12  # beans at press_duration_max
    FILL_LEVELS = (0.25,)
    FILL_TOL = 0.02
    # Dense mound packing (used for freeze + bean-need estimates).
    _BEAN_R = 0.0055
    _BEAN_H = 0.0065
    _PILE_R_SCALE = 0.72

    # Glass-box dispenser (inspired by reference photo — tall clear column on a base).
    BOX_HALF = (0.035, 0.035, 0.090)       # tall slender glass box
    PEDESTAL_HALF = (0.052, 0.052, 0.050)  # raises the hopper above the jar
    PLATFORM_HALF = (0.058, 0.058, 0.008)  # platform between pedestal and hopper
    BEAN_FILL_FRAC = 0.65                  # visual fill inside the glass box
    EE_TO_TCP = 0.12
    KEY_HOVER_DIS = 0.06
    KEY_PRESS_DEPTH = 0.055
    SETTLE_STEPS = 80

    JAR_INNER_R = 0.035
    JAR_HEIGHT = 0.125
    JAR_BOTTOM_T = 0.005

    GLASS = [0.88, 0.95, 0.98, 0.14]
    BEAN_BROWN = [0.30, 0.14, 0.05]
    RING_RED = [0.95, 0.05, 0.05]

    def setup_demo(self, **kwags):
        self._cfg = dict(kwags.get("task_args", {}).get("fill_coffee_jar", {}))
        if kwags.get("scene_id") is None:
            kwags["scene_id"] = int(self._cfg.get("scene_id", 0))
        self.replace_sink_with_range = True

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
        self.table_top = 0.74

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
    def _glass_material(self, rgba=None, transmission=0.90):
        """Nearly-clear glass matching trap_bug / reference acrylic look."""
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

    # ------------------------------------------------------------------ actors
    def load_actors(self):
        cfg = self._cfg
        self.table_top = float(self.kitchens_info["table_height"]) + float(self.table_z_bias)
        tf = cfg.get("target_fill", 0.25)
        if tf is None:
            # Task objective: fill to the first (25%) red ring.
            self.target_fill = 0.25
        else:
            self.target_fill = float(tf)
            if self.target_fill not in self.FILL_LEVELS:
                raise ValueError(f"target_fill must be one of {self.FILL_LEVELS}")

        self.beans_full = int(cfg.get("beans_full", self.BEANS_FULL))
        self.beans_per_press_min = int(
            cfg.get("beans_per_press_min", self.BEANS_PER_PRESS_MIN)
        )
        # ``beans_per_press`` = amount at a full-duration hold (max).
        self.beans_per_press = int(cfg.get("beans_per_press", self.BEANS_PER_PRESS))
        self.beans_per_press_max = int(
            cfg.get("beans_per_press_max", self.beans_per_press)
        )
        self.press_duration_min = float(
            cfg.get("press_duration_min", self.PRESS_DURATION_MIN)
        )
        self.press_duration_max = float(
            cfg.get("press_duration_max", self.PRESS_DURATION_MAX)
        )
        if self.press_duration_max < self.press_duration_min:
            raise ValueError("press_duration_max must be >= press_duration_min")
        if self.beans_per_press_max < self.beans_per_press_min:
            raise ValueError("beans_per_press_max must be >= beans_per_press_min")
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
        # Free front workspace (toward robot, −y), clear of MW / dishrack / range.
        # scene_0: MW left-back, dishrack center-back, range right → use mid-left front.
        side_x = float(cfg.get("station_x", -0.08))
        disp_y = float(cfg.get("disp_y", -0.02))
        jar_y = float(cfg.get("jar_y", -0.16))

        self.dispenser_xy = np.array([side_x, disp_y], dtype=float)
        self.jar_xy = np.array([side_x, jar_y], dtype=float)

        self._build_dispenser()
        self._build_jar()
        self._build_fill_rings()

        self.add_prohibit_area(sapien.Pose([*self.dispenser_xy, self.table_top + 0.1]), padding=0.08)
        self.add_prohibit_area(sapien.Pose([*self.jar_xy, self.table_top + 0.05]), padding=0.05)

        self._loaded = True
        print(
            f"[fill_coffee_jar] KitchenS scene={self.scene_id} "
            f"success≥{self.target_fill:.0%} ring "
            f"(~{self._beans_needed()}/{self.beans_full} beans; "
            f"hold {self.press_duration_min:.2f}-{self.press_duration_max:.2f}s → "
            f"{self.beans_per_press_min}-{self.beans_per_press_max} beans)"
        )

    def _build_dispenser(self):
        """Raised glass bean hopper whose lid touch opens a nozzle over the jar."""
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
        # The clear lid itself is the touch target; there is no separate button.
        lid_z = box_z + bz + 0.003
        self.dispenser_touch_surface = self._add_static_box(
            pose=sapien.Pose([x, y, box_z + bz + 0.003]),
            half_size=[bx * 1.01, by * 1.01, 0.003],
            material=self._glass_material([0.90, 0.96, 0.99, 0.22]),
            name="dispenser_touch_lid",
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
        self.touch_top_z = lid_z + 0.003

    def _build_jar(self):
        """FROZEN jar design — matches demo ``v23``; do not modify this method.

        Smooth see-through cylinder via ``RenderShapeCylinder`` (IOR=1) + thin
        floor disk. Collision from the hollow jar mesh (no GLB visual).
        """
        x, y = self.jar_xy
        z0 = self.table_top + 0.001
        outer_r = self.JAR_INNER_R + 0.0035
        h = self.JAR_HEIGHT
        bottom_t = self.JAR_BOTTOM_T
        upright_q = [0.70710678, 0.0, -0.70710678, 0.0]

        col_path = Path("assets/objects/253_glass_jar/collision/base0.glb").resolve()
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        builder.add_nonconvex_collision_from_file(filename=str(col_path), scale=[1, 1, 1])
        builder.set_initial_pose(sapien.Pose([x, y, z0]))
        self.jar = builder.build(name="glass_jar")
        try:
            self.jar.set_name("glass_jar")
        except Exception:
            pass

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
        """Add three subtle, thin red rings around the smooth glass cylinder."""
        x, y = self.jar_xy
        ring_material = self._opaque_material([0.78, 0.05, 0.05], 0.70)
        ring_mesh = Path("assets/objects/253_glass_jar/rings/thin_ring.glb")
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
        """Approx. beans for a dense mound up to the target red ring."""
        target_h = self.target_fill * self.jar_fillable_h
        n_layers = max(1, int(math.ceil(target_h / self._BEAN_H)))
        return min(self.beans_full, n_layers * self._beans_per_layer())

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

    def _pile_height(self) -> float:
        """Height of the bean pile above the jar floor (meters)."""
        bean_half = self._BEAN_H * 0.5
        zs = []
        for b in self._beans_in_jar_list():
            zs.append(float(b.get_pose().p[2]))
        if not zs:
            return 0.0
        return max(0.0, max(zs) + bean_half - self.jar_bottom_z)

    def _current_fill(self) -> float:
        """Fill fraction from pile height vs the red-ring scale (0 = empty, 1 = rim)."""
        return float(self._pile_height() / max(1e-6, self.jar_fillable_h))

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
        """Freeze beans into a dense mound, stacking upward from the jar floor."""
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
        """Opaque coffee column up to the current pile height (readable vs red rings).

        Top-down cameras flatten a real bean stack into a small floor sprinkle, so
        success-by-height is invisible without an explicit fill body.
        """
        if getattr(self, "fill_visual", None) is not None:
            try:
                self.scene.remove_entity(self.fill_visual)
            except Exception:
                pass
            self.fill_visual = None

        h = float(self._pile_height())
        if h < 0.004:
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

    def _beans_for_duration(self, hold_s: float) -> int:
        """Map a hold duration (sec) → bean count between min/max endpoints."""
        t = float(np.clip(hold_s, self.press_duration_min, self.press_duration_max))
        span = self.press_duration_max - self.press_duration_min
        if span <= 1e-9:
            n = self.beans_per_press_max
        else:
            alpha = (t - self.press_duration_min) / span
            n = int(
                round(
                    self.beans_per_press_min
                    + alpha * (self.beans_per_press_max - self.beans_per_press_min)
                )
            )
        return int(np.clip(n, self.beans_per_press_min, self.beans_per_press_max))

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

    def _tick_press(self):
        """While the lid is held, stream beans according to elapsed hold time."""
        if not self._press_active:
            return
        if self.beans_in_jar + self._press_spawned >= self.beans_full:
            return
        self._press_steps += 1
        self._press_hold_s = self._press_steps * self._sim_dt()
        # Progressive target: ramp 0→max over duration_max, then clamp via
        # ``_beans_for_duration`` so short holds still grant the min amount
        # once finished (see ``_end_press``).
        if self._press_hold_s < self.press_duration_min:
            # Before min duration: linear ramp from 0 to beans_min.
            alpha = self._press_hold_s / max(1e-6, self.press_duration_min)
            target = int(math.floor(alpha * self.beans_per_press_min))
        else:
            target = self._beans_for_duration(self._press_hold_s)
        target = min(target, self.beans_full - self.beans_in_jar)
        while self._press_spawned < target:
            self._spawn_one_dispensed_bean()
            self._press_spawned += 1
            # Small settle between beans so the cascade stays visible.
            for _ in range(8):
                super()._update_kinematic_tasks()
                self.scene.step()

    def _end_press(self):
        """Finalize a hold: top up to the duration mapping, settle, freeze pile."""
        if not self._press_active:
            return
        # Guarantee the min/max mapping even if the stream lagged.
        hold_s = max(self._press_hold_s, self._sim_dt())
        want = self._beans_for_duration(hold_s)
        want = min(want, self.beans_full - self.beans_in_jar)
        while self._press_spawned < want:
            self._spawn_one_dispensed_bean()
            self._press_spawned += 1
            for _ in range(8):
                super()._update_kinematic_tasks()
                self.scene.step()

        self.press_count += 1
        spawned = int(self._press_spawned)
        # Close the press session before settle so touch-detect cannot re-enter.
        self._press_active = False
        self._dwell(self.SETTLE_STEPS)
        self._freeze_beans(self.beans)
        self._dwell(6)
        self.beans_in_jar = self._count_beans_in_jar()
        self._sync_fill_visual()
        print(
            f"[fill_coffee_jar] hold={hold_s:.2f}s → {spawned} beans "
            f"(map {self.beans_per_press_min}-{self.beans_per_press_max} "
            f"over {self.press_duration_min:.2f}-{self.press_duration_max:.2f}s)"
        )
        self._dispensing = False
        self._press_steps = 0
        self._press_spawned = 0
        self._press_hold_s = 0.0

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

    def _press_dispenser(self, arm_tag: ArmTag, duration: float | None = None):
        """Press and hold the lid for ``duration`` seconds (clamped to min/max)."""
        if duration is None:
            duration = self.press_duration_max
        duration = float(
            np.clip(duration, self.press_duration_min, self.press_duration_max)
        )
        self.move(self.move_to_pose(arm_tag, self._touch_tip_pose(self.KEY_HOVER_DIS)))
        if not self.plan_success:
            print(f"[fill_coffee_jar] hover failed to {self._touch_tip_pose(self.KEY_HOVER_DIS)}")
            return False
        self.move(self.move_by_displacement(arm_tag, z=-self.KEY_PRESS_DEPTH))
        if not self.plan_success:
            return False

        # Hold: stream beans for the requested duration (expert path; does not
        # rely on the per-step touch detector, which can miss during move()).
        self._start_press()
        hold_steps = max(1, int(round(duration / self._sim_dt())))
        for _ in range(hold_steps):
            self._tick_press()
            if not self._press_active:
                break
            super()._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (self._press_steps % max(1, int(self.save_freq)) == 0):
                self._take_picture()
        self._end_press()

        self.move(self.move_by_displacement(arm_tag, z=self.KEY_PRESS_DEPTH))
        self._dwell(8)
        return True

    def _detect_lid_touch(self):
        if self.dispenser_touch_surface is None or not hasattr(self, "robot"):
            return
        try:
            ee = np.asarray(self.robot.get_left_ee_pose()[:3], dtype=float)
        except Exception:
            return
        xy_ok = float(np.linalg.norm(ee[:2] - self.touch_xy)) <= 0.05
        # EE origin is about EE_TO_TCP above the fingertip. Trigger only when
        # the fingertip reaches the clear lid, not during the hover approach.
        z_ok = ee[2] <= self.touch_top_z + self.EE_TO_TCP + 0.02
        touching = bool(xy_ok and z_ok)
        if touching and not self._touch_latched:
            self._start_press()
        if touching and self._press_active:
            self._tick_press()
            # Auto-finish once the max hold has been reached.
            if self._press_hold_s >= self.press_duration_max:
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
    def play_once(self):
        arm = ArmTag("left")
        self.move(self.close_gripper(arm))
        if not self.plan_success:
            print("[fill_coffee_jar] close_gripper failed")
            return self.info

        needed = self._beans_needed()
        # Full-duration holds; shorter holds dispense less (see press_duration_*).
        max_presses = int(math.ceil(needed / max(1, self.beans_per_press_max))) + 3
        for i in range(max_presses):
            self.beans_in_jar = self._count_beans_in_jar()
            fill = self._current_fill()
            if fill >= self.target_fill:
                break
            if not self.plan_success:
                print(f"[fill_coffee_jar] plan failed before press {i}")
                break
            ok = self._press_dispenser(arm, duration=self.press_duration_max)
            self.beans_in_jar = self._count_beans_in_jar()
            fill = self._current_fill()
            print(
                f"[fill_coffee_jar] press={i} ok={ok} "
                f"beans={self.beans_in_jar}/{needed} fill={fill:.0%} "
                f"(need≥{self.target_fill:.0%}) plan={self.plan_success}"
            )

        # Target reached → stop interacting and withdraw.
        if self.plan_success:
            self.move(self.move_by_displacement(arm, z=0.08))

        fill = self._current_fill()
        if fill >= self.target_fill:
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
        """Success when the bean pile reaches at least the target red ring."""
        self.beans_in_jar = self._count_beans_in_jar()
        fill = self._current_fill()
        return bool(fill + 1e-3 >= self.target_fill and self.beans_in_jar > 0)

    def get_obs(self):
        obs = super().get_obs()
        obs["coffee_jar"] = {
            "target_fill": float(self.target_fill),
            "fill": float(self._current_fill()),
            "pile_height": float(self._pile_height()),
            "beans_in_jar": int(self.beans_in_jar),
            "beans_full": int(self.beans_full),
            "beans_per_press_min": int(self.beans_per_press_min),
            "beans_per_press_max": int(self.beans_per_press_max),
            "press_duration_min": float(self.press_duration_min),
            "press_duration_max": float(self.press_duration_max),
            "press_count": int(self.press_count),
            "scene_id": int(self.scene_id),
        }
        return obs
