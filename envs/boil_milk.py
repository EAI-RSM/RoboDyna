"""Boil milk on a KitchenS cooking range without letting it overflow.

KitchenS scene with microwave + cooking range (no sink/tap). A stainless
saucepan of milk sits on a burner; a milk carton and mug rest on the open
counter (ready for a pour afterward). The robot turns the stove on; the milk
level rises while boiling. Shut the stove off as the milk reaches the rim.
If left on past the rim, milk spills onto the cooktop (white puddle), the
flame goes out, the pot level drops back to baseline, and the episode fails.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import sapien
import sapien.render
import transforms3d as t3d
from transforms3d.euler import euler2quat
from transforms3d.quaternions import qmult

from ._kitchens_base_task import KitchenS_base_task
from ._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
from .utils import *
from .utils.create_actor import create_actor, create_box, UnStableError


class boil_milk(KitchenS_base_task):
    """Turn the range on, let the milk rise to the rim, then shut it off in time."""

    BOIL_STEPS_DEFAULT = 6000         # steps for liquid_level 0→1 while stove is on
    SETTLE_STEPS_DEFAULT = 1200       # steps for liquid_level → baseline while stove is off
    BASELINE_LEVEL_DEFAULT = 0.35     # resting fill fraction of pot height
    # The milk keeps rising for the whole knob reach (including the shutoff
    # approach), so commit early enough that the off-twist still lands under the rim.
    EXPERT_SHUTOFF_LEVEL_DEFAULT = 0.55
    OVERFLOW_LEVEL_DEFAULT = 0.98     # rim / spill threshold (failure if reached while on)
    KNOB_CONTACT_RADIUS_DEFAULT = 0.06   # pinch-to-knob distance that counts as held
    EE_TO_TCP = 0.12                  # EE frame → TCP offset (cook_meat / dispense_gummy)
    # Y-up meshes stood upright: target world heights for realistic counter props.
    MUG_TARGET_HEIGHT = 0.105         # ~10.5 cm coffee mug
    MILK_TARGET_HEIGHT = 0.200        # ~20 cm 1 L carton
    # Same corridor / grasp as make_soup — that grasp seats the jaws cleanly.
    KNOB_APPROACH_PATH = (
        (-0.13, -0.33, 0.06),
        (-0.08, -0.33, 0.00),
        (0.00, -0.25, 0.00),
    )
    KNOB_GRASP_STANDOFF = 0.015

    def setup_demo(self, **kwags):
        self._cfg = kwags.get("task_args", {}).get("boil_milk", {})
        # Prefer scene_0 so the range is on the right (reachable by the right arm).
        if kwags.get("scene_id") is None:
            kwags["scene_id"] = int(self._cfg.get("scene_id", 0))
        self.replace_sink_with_range = True
        # Bare counter beside the range — no sink basin / chrome tap.
        self.omit_sink = True
        self.clear_sink_and_range = False
        # Stove was 1.5× default, then cut 30% → 1.05×. Microwave +30%.
        self.range_scale_mult = float(self._cfg.get("range_scale_mult", 1.05))
        self.microwave_scale_mult = float(self._cfg.get("microwave_scale_mult", 1.3))
        # Range pulled forward off the backsplash so the right arm can reach the
        # front knob panel without over-extending.
        rel = self._cfg.get("range_xy", [0.42, 0.13])
        self.range_position_override = [float(rel[0]), float(rel[1])]

        # Per-step state must exist before any early _update_kinematic_tasks call.
        self.stove_on = False
        self.liquid_level = float(self._cfg.get("baseline_level", self.BASELINE_LEVEL_DEFAULT))
        self.baseline_level = float(self.liquid_level)
        self.max_liquid_level = float(self.liquid_level)
        self.overflowed = False
        self.turned_on_once = False
        self.turned_off_after_boil = False
        self._liquid_entity = None
        self._spill_entity = None
        self._spill_amount = 0.0
        self._burner_shapes = []
        self._knob_press_latched = False
        self._prev_knob_pressed = False
        self._ignore_knob = False
        self._expert_holding_knob = False
        self.force_overflow = bool(self._cfg.get("force_overflow", False))

        super().setup_demo(**kwags)

    # ---------------------------------------------------------------- actors
    def load_actors(self):
        cfg = self._cfg
        self.boil_steps = int(cfg.get("boil_steps", self.BOIL_STEPS_DEFAULT))
        self.settle_steps = int(cfg.get("settle_steps", self.SETTLE_STEPS_DEFAULT))
        self.baseline_level = float(cfg.get("baseline_level", self.BASELINE_LEVEL_DEFAULT))
        self.expert_shutoff_level = float(
            cfg.get("expert_shutoff_level", self.EXPERT_SHUTOFF_LEVEL_DEFAULT)
        )
        self.overflow_level = float(cfg.get("overflow_level", self.OVERFLOW_LEVEL_DEFAULT))
        self.knob_contact_radius = float(
            cfg.get("knob_contact_radius", self.KNOB_CONTACT_RADIUS_DEFAULT)
        )

        self.liquid_level = self.baseline_level
        self.max_liquid_level = self.baseline_level
        self.stove_on = False
        self.overflowed = False
        self.turned_on_once = False
        self.turned_off_after_boil = False
        self._knob_press_latched = False
        self._prev_knob_pressed = False
        self._expert_holding_knob = False
        self._ignore_knob = False
        self.force_overflow = bool(cfg.get("force_overflow", False))
        self._spill_amount = 0.0
        if getattr(self, "_spill_entity", None) is not None:
            try:
                self.scene.remove_entity(self._spill_entity)
            except Exception:
                pass
            self._spill_entity = None

        if not hasattr(self, "burner_xy"):
            raise UnStableError("cooking range missing — KitchenS base did not load a range")

        # Stainless saucepan (tea-pan style): thin hollow cylinder + dual-rod
        # handle with black grips — matches Desktop/images/tea_pan.jpeg.
        # Procedural so mplib/CuRobo never see a non-unit-scale mesh.
        self.pot_id = int(cfg.get("pot_id", 0))
        # Defaults: prior saucepan size then −30%.
        pot_r = float(cfg.get("pot_radius", 0.054))
        pot_h = float(cfg.get("pot_height", 0.0735))
        # Thin wall so the lip does not read as a second inner layer from above.
        wall = float(cfg.get("pot_wall", 0.0022))
        pot_z0 = float(self.range_top_z)
        cx, cy = float(self.burner_xy[0]), float(self.burner_xy[1])
        # Keep the stock burner disc under the pot (make_soup fire effect): the
        # disc glows blue with the ring, so the flame reads as a real burner.
        self._burner_shapes = []
        if getattr(self, "active_burner", None) is not None:
            self._burner_home_pose = sapien.Pose(
                p=[cx, cy, float(self.range_top_z) + 0.0015]
            )
            try:
                self.active_burner.set_pose(self._burner_home_pose)
            except Exception:
                pass
            for c in self.active_burner.get_components():
                if isinstance(c, sapien.render.RenderBodyComponent):
                    self._burner_shapes = list(c.render_shapes)
        self._disc_parts = []
        self._disc_shapes = []
        self._disc_home_poses = []
        vertical_q = [0.70710678, 0.0, 0.70710678, 0.0]

        metal = sapien.render.RenderMaterial(base_color=[0.74, 0.75, 0.77, 1.0])
        metal.metallic = 0.92
        metal.roughness = 0.16

        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        # Floor (inset so its rim is not visible inside the wall lip).
        floor_r = pot_r - wall
        floor_h = 0.002
        builder.add_cylinder_collision(
            pose=sapien.Pose([0, 0, floor_h / 2], vertical_q),
            radius=floor_r, half_length=floor_h / 2,
            material=self.scene.default_physical_material,
        )
        builder.add_cylinder_visual(
            pose=sapien.Pose([0, 0, floor_h / 2], vertical_q),
            radius=floor_r, half_length=floor_h / 2, material=metal,
        )

        # Faceted hollow wall (open top so the milk column stays visible).
        wall_segments = 32
        wall_radius = pot_r - wall / 2
        tangent_half = wall_radius * np.tan(np.pi / wall_segments) * 1.02
        for ang in np.linspace(0, 2 * np.pi, wall_segments, endpoint=False):
            px = wall_radius * np.cos(ang)
            py = wall_radius * np.sin(ang)
            tangent_angle = ang + np.pi / 2
            q = [
                float(np.cos(tangent_angle / 2)),
                0.0,
                0.0,
                float(np.sin(tangent_angle / 2)),
            ]
            wall_pose = sapien.Pose([px, py, pot_h / 2], q)
            builder.add_box_collision(
                pose=wall_pose,
                half_size=[tangent_half, wall / 2, pot_h / 2],
                material=self.scene.default_physical_material,
            )
            builder.add_box_visual(
                pose=wall_pose,
                half_size=[tangent_half, wall / 2, pot_h / 2],
                material=metal,
            )

        builder.set_initial_pose(sapien.Pose(p=[cx, cy, pot_z0]))
        self.pot = builder.build(name="tea_pot")

        # Dual-rod handle toward −X. Black sleeve per rod; rods stop short of the
        # tip so nothing protrudes / no hanging loop at the far end.
        handle_len = float(cfg.get("handle_length", 0.091))
        handle_z = pot_z0 + pot_h * 0.62
        rod_r = 0.0028
        rod_sep = 0.014
        tip_overhang = 0.004  # black grip extends past metal rods
        rod_x0 = cx - pot_r + 0.003
        tip_x = cx - pot_r - handle_len
        rod_x1 = tip_x + tip_overhang
        rod_mid_x = 0.5 * (rod_x0 + rod_x1)
        rod_half = 0.5 * (rod_x0 - rod_x1)
        # Grip covers most of each rod and ends at tip_x (clean tip, no cap piece).
        grip_x0 = rod_x0 - 0.012  # leave a short metal stub at the pot
        grip_x1 = tip_x
        grip_mid = 0.5 * (grip_x0 + grip_x1)
        grip_half = 0.5 * (grip_x0 - grip_x1)
        for side, tag in ((-1, "a"), (1, "b")):
            y = cy + side * rod_sep / 2
            create_box(
                self.scene,
                sapien.Pose(p=[rod_mid_x, y, handle_z]),
                half_size=[rod_half, rod_r, rod_r],
                color=(0.70, 0.71, 0.73),
                name=f"tea_pot_rod_{tag}",
                is_static=True,
            )
            create_box(
                self.scene,
                sapien.Pose(p=[grip_mid, y, handle_z]),
                half_size=[grip_half, rod_r + 0.0022, rod_r + 0.0022],
                color=(0.05, 0.05, 0.06),
                name=f"tea_pot_grip_{tag}",
                is_static=True,
            )

        # Milk fills to the inner wall — no silver floor ring inside.
        self.pot_inner_radius = pot_r - wall
        self.pot_inner_height = pot_h - floor_h
        self.pot_bottom_z = pot_z0 + floor_h
        self.pot_xy = (cx, cy)
        self.pot_radius = pot_r
        self.pot_height = pot_h

        # Blue fire halo around the pot base, concentric with the burner disc
        # (same geometry as make_soup).
        self._clear_stove_fire_ring()
        self._build_stove_fire_ring(
            cx,
            cy,
            float(self.range_top_z) + 0.0015,
            float(pot_r + 0.009),
            n=28,
            half_size=[0.007, 0.0035, 0.002],
        )
        self._rebuild_liquid(force=True)
        self._set_stove_fire(False)

        # Milk carton + mug on the open counter between microwave and stove.
        self._load_milk_box(cfg)
        self._load_mug(cfg)

        # Arm that can reach the knob (range is always on +x in scene_0 / scene_2).
        self.arm = ArmTag("right" if self.knob_xy[0] >= 0 else "left")

    def _counter_apron_bounds(self):
        """Open apron between microwave and range left edge (x, y limits)."""
        hx = getattr(self, "range_half_size", (0.14, 0.16))[0]
        rx = float(self.range_xy[0]) if hasattr(self, "range_xy") else 0.42
        x_lo, x_hi = -0.08, float(rx - hx - 0.08)
        y_lo, y_hi = -0.14, 0.10
        if x_hi <= x_lo + 0.04:
            x_lo, x_hi = -0.05, 0.12
        return x_lo, x_hi, y_lo, y_hi

    @staticmethod
    def _yup_authored_height(modelname: str, model_id: int) -> float:
        """World height (Y) of a Y-up mesh at ``scale_mult=1``."""
        path = Path("assets/objects") / modelname / f"model_data{model_id}.json"
        with open(path) as f:
            data = json.load(f)
        sc = data.get("scale") or [1.0, 1.0, 1.0]
        ext = data["extents"]
        return float(sc[1]) * float(ext[1])

    def _scale_for_target_height(
        self, modelname: str, model_id: int, target_h: float, override=None
    ) -> float:
        """``scale_mult`` so the upright mesh matches ``target_h`` (meters)."""
        if override is not None:
            return float(override)
        authored = self._yup_authored_height(modelname, model_id)
        if authored <= 1e-6:
            return 1.0
        return float(target_h) / authored

    def _load_milk_box(self, cfg):
        """Place a random ``038_milk-box`` variant on the bare counter (decorative)."""
        n_variants = 4
        mid = int(cfg.get("milk_box_id", -1))
        if mid < 0 or mid >= n_variants:
            mid = int(np.random.randint(0, n_variants))
        self.milk_box_id = mid
        target_h = float(cfg.get("milk_box_height", self.MILK_TARGET_HEIGHT))
        scale = self._scale_for_target_height(
            "038_milk-box", mid, target_h, cfg.get("milk_box_scale")
        )
        self.milk_box_scale = scale

        x_lo, x_hi, y_lo, y_hi = self._counter_apron_bounds()
        # Mid apron — leave the front clear for the right-arm knob approach.
        mx = float(np.random.uniform(x_lo, min(x_hi, 0.16)))
        my = float(np.random.uniform(max(y_lo + 0.04, -0.06), y_hi))
        yaw = float(np.random.uniform(-0.6, 0.6))
        # Mesh is Y-up; stand upright on the counter then apply yaw.
        upright = np.array([0.70710678, 0.70710678, 0.0, 0.0], dtype=np.float64)
        q = qmult(euler2quat(0.0, 0.0, yaw, axes="sxyz"), upright)
        z = 0.74 + float(self.table_z_bias) + 0.001
        pose = sapien.Pose([mx, my, z], q.tolist())
        try:
            self.milk_box = create_actor(
                self,
                pose=pose,
                modelname="038_milk-box",
                model_id=mid,
                convex=True,
                is_static=True,
                scale_mult=scale,
            )
            if self.milk_box is not None:
                self.milk_box.set_name(f"038_milk-box/base{mid}")
                # Some milk-box variants ship without model_data["scale"]; ensure
                # prohibit-area math has a config dict.
                if getattr(self.milk_box, "config", None) is None:
                    self.milk_box.config = {
                        "scale": [scale, scale, scale],
                        "center": [0.0, 0.0, 0.0],
                        "extents": [0.10, 0.22, 0.10],
                    }
                try:
                    self.add_prohibit_area(self.milk_box, padding=0.02)
                except Exception:
                    self.prohibited_area.append(
                        [mx - 0.06, my - 0.06, mx + 0.06, my + 0.06]
                    )
        except Exception as e:
            print(f"[boil_milk] failed to load milk box: {e}")
            self.milk_box = None
        self.milk_box_xy = (mx, my)

    def _load_mug(self, cfg):
        """Place a proper upright ``039_mug`` near the milk (not touching)."""
        # Prefer classic ceramic variants 0–9 (hanging_mug / pour_beer).
        n_variants = 10
        mid = int(cfg.get("mug_id", 0))
        if mid < 0 or mid >= n_variants:
            mid = int(np.random.randint(0, n_variants))
        self.mug_id = mid
        target_h = float(cfg.get("mug_height", self.MUG_TARGET_HEIGHT))
        scale = self._scale_for_target_height(
            "039_mug", mid, target_h, cfg.get("mug_scale")
        )
        self.mug_scale = scale

        mx, my = self.milk_box_xy
        x_lo, x_hi, y_lo, y_hi = self._counter_apron_bounds()
        # Nearby but clear of the right-arm knob approach corridor (−Y near +X).
        gap = float(cfg.get("mug_gap", 0.16))
        candidates = [
            (mx - gap, my),           # toward microwave
            (mx - gap, my + 0.03),
            (mx + gap, my + 0.03),    # toward stove but back on the apron
            (mx + gap, my),
            (mx - 0.5 * gap, my + 0.05),
            (mx, my + gap * 0.5),
        ]
        ux, uy = mx - gap, my
        for cx_, cy_ in candidates:
            # Keep mug off the front apron so the knob reach stays clear.
            if x_lo <= cx_ <= x_hi and max(y_lo + 0.04, -0.08) <= cy_ <= y_hi:
                ux, uy = float(cx_), float(cy_)
                break
        else:
            ux = float(np.clip(mx - gap, x_lo, x_hi))
            uy = float(np.clip(my, max(y_lo + 0.04, -0.08), y_hi))

        yaw = float(np.random.uniform(-0.5, 0.5))
        # Y-up mesh → stand upright (same as hanging_mug / pour_beer).
        upright = np.array([0.70710678, 0.70710678, 0.0, 0.0], dtype=np.float64)
        q = qmult(euler2quat(0.0, 0.0, yaw, axes="sxyz"), upright)
        z = 0.74 + float(self.table_z_bias) + 0.001
        pose = sapien.Pose([ux, uy, z], q.tolist())
        try:
            self.mug = create_actor(
                self,
                pose=pose,
                modelname="039_mug",
                model_id=mid,
                convex=True,
                is_static=True,
                scale_mult=scale,
            )
            if self.mug is not None:
                self.mug.set_name(f"039_mug/base{mid}")
                if getattr(self.mug, "config", None) is None:
                    self.mug.config = {
                        "scale": [scale, scale, scale],
                        "center": [0.0, 0.0, 0.0],
                        "extents": [0.14, 0.12, 0.14],
                    }
                try:
                    self.add_prohibit_area(self.mug, padding=0.02)
                except Exception:
                    self.prohibited_area.append(
                        [ux - 0.05, uy - 0.05, ux + 0.05, uy + 0.05]
                    )
        except Exception as e:
            print(f"[boil_milk] failed to load mug: {e}")
            self.mug = None
        self.mug_xy = (ux, uy)

    # ---------------------------------------------------------------- liquid / stove visuals
    def _liquid_half_height(self) -> float:
        return max(0.004, 0.5 * self.liquid_level * self.pot_inner_height)

    def _rebuild_liquid(self, force: bool = False):
        """Recreate the translucent liquid column to match ``liquid_level``."""
        half_h = self._liquid_half_height()
        # Skip tiny updates to keep planning pass cheap.
        if (
            not force
            and self._liquid_entity is not None
            and abs(half_h - getattr(self, "_liquid_half_h_cached", -1.0)) < 0.0015
        ):
            return
        self._liquid_half_h_cached = half_h

        if self._liquid_entity is not None:
            try:
                self.scene.remove_entity(self._liquid_entity)
            except Exception:
                pass
            self._liquid_entity = None

        # Visual-only static cylinder (no collision). Sapien cylinder axis = local +X;
        # VERTICAL_CYL_Q rotates it to world +Z (same as pick_ripe_apple trunk).
        radius = self.pot_inner_radius
        t = float(np.clip(
            (self.liquid_level - self.baseline_level)
            / max(1e-6, self.overflow_level - self.baseline_level),
            0.0, 1.0,
        ))
        color = [0.20 + 0.35 * t, 0.42 + 0.18 * t, 0.90 - 0.25 * t, 0.88]
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        mat = sapien.render.RenderMaterial(base_color=color)
        mat.metallic = 0.0
        mat.roughness = 0.15
        vertical_cyl_q = [0.70710678, 0.0, 0.70710678, 0.0]  # 90° about Y
        local_pose = sapien.Pose([0, 0, 0], vertical_cyl_q)
        builder.add_cylinder_visual(
            pose=local_pose, radius=radius, half_length=half_h, material=mat
        )
        z = self.pot_bottom_z + half_h
        builder.set_initial_pose(sapien.Pose(p=[self.pot_xy[0], self.pot_xy[1], z]))
        self._liquid_entity = builder.build(name="pot_liquid")

    def _set_burner_glow(self, on: bool):
        """Blue fire ring only; solid burner disc stays hidden after load_actors."""
        self._set_stove_fire(bool(on), intensity=1.0 if on else 0.0)

    def _set_stove(self, on: bool):
        on = bool(on)
        if on == self.stove_on:
            return
        self.stove_on = on
        if on:
            self.turned_on_once = True
        elif (
            self.turned_on_once
            and self.max_liquid_level > self.baseline_level + 0.05
            and not self.overflowed
        ):
            # Intentional shutoff before spill — not an overflow auto-kill.
            self.turned_off_after_boil = True
        self._set_burner_glow(on)

    def _spawn_spill_puddle(self, scale: float = 1.0):
        """Compact white milk puddle under the pot (visual only).

        Spill outer radius = 20% of pot diameter (small ring under the base).
        ``scale`` is kept for call-site compatibility; size is fixed.
        """
        del scale  # fixed size — no grow-animation
        if self._spill_entity is not None:
            try:
                self.scene.remove_entity(self._spill_entity)
            except Exception:
                pass
            self._spill_entity = None

        cx, cy = float(self.pot_xy[0]), float(self.pot_xy[1])
        z = float(self.range_top_z) + 0.003
        pot_r = float(self.pot_radius)
        pot_diameter = 2.0 * pot_r
        spill_r = 0.20 * pot_diameter
        milk = sapien.render.RenderMaterial(base_color=[0.96, 0.96, 0.93, 1.0])
        milk.metallic = 0.0
        milk.roughness = 0.55
        vertical_q = [0.70710678, 0.0, 0.70710678, 0.0]

        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        builder.add_cylinder_visual(
            pose=sapien.Pose([0, 0, 0], vertical_q),
            radius=float(spill_r),
            half_length=0.0012,
            material=milk,
        )
        builder.set_initial_pose(sapien.Pose(p=[cx, cy, z]))
        self._spill_entity = builder.build(name="milk_spill")

    def _trigger_overflow(self):
        """Rim reached while boiling: spill, kill flame, reset pot level, fail."""
        if self.overflowed:
            return
        self.overflowed = True
        self._spill_amount = 1.0
        # Kill the flame without counting as a successful shutoff.
        self.stove_on = False
        self._set_burner_glow(False)
        self._spawn_spill_puddle()
        # Milk drops back to the resting level after the spill.
        self.liquid_level = float(self.baseline_level)
        self._rebuild_liquid(force=True)

    # ---------------------------------------------------------------- per-step dynamics
    def _knob_is_pressed(self) -> bool:
        """True when the gripper pinch point is on the rotary knob."""
        if getattr(self, "_expert_holding_knob", False):
            return True
        if not hasattr(self, "knob_xyz") or self.knob_xyz is None:
            return False
        arm = getattr(self, "arm", None)
        if arm is None:
            return False
        try:
            ee_pose = np.array(self.get_arm_pose(str(arm)), dtype=float)
            ee_rot = t3d.quaternions.quat2mat(ee_pose[3:7])
            pinch = ee_pose[:3] + ee_rot @ np.array(
                [self.EE_TO_TCP, 0.0, 0.0], dtype=float
            )
        except Exception:
            return False
        return bool(
            np.linalg.norm(pinch - np.asarray(self.knob_xyz, dtype=float))
            < self.knob_contact_radius
        )

    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
        # Guard: _update_kinematic_tasks runs during camera init BEFORE load_actors.
        if not getattr(self, "pot_inner_height", None):
            return
        if not hasattr(self, "liquid_level"):
            return

        # Edge-trigger: each grasp-and-twist toggles the stove. Suppressed while
        # the arm is merely travelling so a fly-by cannot flip the burner.
        if not getattr(self, "_ignore_knob", False):
            pressed = self._knob_is_pressed()
            if pressed and not self._prev_knob_pressed:
                self._set_stove(not self.stove_on)
            self._prev_knob_pressed = pressed
        else:
            self._prev_knob_pressed = False

        # Boiling continues for every sim step while the burner is on — including
        # the entire shutoff reach. It must NOT pause for ``_ignore_knob`` or any
        # other arm-motion flag; the only stop is ``_set_stove(False)`` after the
        # off-twist (or an overflow committed once the hand leaves the knob).
        if self.overflowed:
            self.liquid_level = float(self.baseline_level)
        elif self.stove_on:
            self.liquid_level = min(
                1.0, self.liquid_level + 1.0 / max(1, self.boil_steps)
            )
            self.max_liquid_level = max(self.max_liquid_level, self.liquid_level)
            # Do not kill the flame mid-knob-turn: the expert is still reaching
            # to shut off. Overflow commits only once the hand is free again.
            if (
                self.liquid_level >= self.overflow_level
                and not getattr(self, "_ignore_knob", False)
            ):
                self._trigger_overflow()
        else:
            # Settle back toward the baseline once the stove is off.
            if self.liquid_level > self.baseline_level:
                self.liquid_level = max(
                    self.baseline_level,
                    self.liquid_level - 1.0 / max(1, self.settle_steps),
                )

        self._rebuild_liquid(force=False)

    def _idle_steps(self, n_steps: int, until=None):
        """Advance sim (and record frames) without arm motion — NOT delay().

        ``until`` is polled each step and stops the wait early. It must be passed
        here rather than wrapping ``_idle_steps(1)`` so the save_freq throttle
        keeps counting across the whole wait.
        """
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

    def _idle_until_level(self, level: float, max_steps: int = 4000):
        """Watch the pot until the liquid reaches ``level`` (or it stops rising)."""
        self._idle_steps(
            max_steps,
            until=lambda: self.liquid_level >= float(level) or self.overflowed,
        )

    # ---------------------------------------------------------------- expert motion
    def _knob_pose(self, offset, turn_angle: float) -> list[float]:
        """Front-facing EE pose at ``knob_center + offset``, wrist twisted by angle."""
        base_q = np.asarray(GRASP_DIRECTION_DIC["front"], dtype=float)
        ee_p = np.asarray(self.knob_xyz, dtype=float) + np.asarray(offset, dtype=float)
        twist_q = np.array(
            [np.cos(turn_angle / 2), np.sin(turn_angle / 2), 0.0, 0.0],
            dtype=float,
        )
        ee_q = t3d.quaternions.qmult(base_q, twist_q)
        return [*ee_p.tolist(), *ee_q.tolist()]

    def _knob_turn_pose(self, standoff: float, turn_angle: float) -> list[float]:
        """Grasp pose whose jaws close around the knob (same as make_soup)."""
        return self._knob_pose(
            [0.0, -(self.EE_TO_TCP + float(standoff)), 0.0], turn_angle
        )

    def _turn_knob(self, want_on: bool):
        """Reach / grasp / twist the knob — same action as make_soup._turn_knob_on."""
        arm = self.arm
        start_angle = -np.pi / 2 if self.stove_on else 0.0
        end_angle = -np.pi / 2 if bool(want_on) else 0.0
        path = self.KNOB_APPROACH_PATH

        self._ignore_knob = True
        self.move(self.open_gripper(arm))
        for offset in path:
            self.move(self.move_to_pose(arm, self._knob_pose(offset, start_angle)))
        self.move(
            self.move_to_pose(arm, self._knob_turn_pose(self.KNOB_GRASP_STANDOFF, start_angle))
        )
        self.move(self.close_gripper(arm))
        self._expert_holding_knob = True
        self.move(
            self.move_to_pose(arm, self._knob_turn_pose(self.KNOB_GRASP_STANDOFF, end_angle))
        )
        self._expert_holding_knob = False
        # Stove state commits only here — boiling keeps rising until this twist.
        self._set_stove(bool(want_on))
        self._idle_steps(8)
        self.move(self.open_gripper(arm))
        for offset in reversed(path):
            self.move(self.move_to_pose(arm, self._knob_pose(offset, end_angle)))
        self._ignore_knob = False
        self._prev_knob_pressed = False
        # If the milk crested the rim during the reach and the stove is STILL on
        # (missed shutoff), commit the overflow now that the hand is clear.
        if (
            self.stove_on
            and not self.overflowed
            and self.liquid_level >= self.overflow_level
        ):
            self._trigger_overflow()

    def play_once(self):
        arm = self.arm

        # 1) Turn the stove ON — the milk starts rising on this twist.
        self._turn_knob(want_on=True)

        if self.force_overflow:
            # Failure demo: leave the burner on until the milk hits the rim,
            # then hold so the white cooktop spill is visible.
            self._idle_until_level(self.overflow_level, max_steps=8000)
            if not self.overflowed:
                self._trigger_overflow()
            self._idle_steps(120)  # let the puddle finish spreading
        else:
            # 2) Watch until milk is high enough that the shutoff reach still
            #    finishes under the rim. Rising continues through the whole reach.
            self._idle_until_level(self.expert_shutoff_level)

            # 3) Grasp+twist OFF — boiling stops only on this twist, not before.
            self._turn_knob(want_on=False)

            # 4) Let the liquid settle back toward baseline (visible in the video).
            settle = max(0.0, self.liquid_level - self.baseline_level)
            steps_settle = int(round(settle * self.settle_steps))
            self._idle_steps(max(1, min(steps_settle, self.settle_steps)))

        self.info["info"] = {
            "{A}": "tea_pot",
            "{D}": (
                f"038_milk-box/base{self.milk_box_id}"
                if getattr(self, "milk_box", None) is not None
                else "038_milk-box"
            ),
            "{E}": (
                f"039_mug/base{self.mug_id}"
                if getattr(self, "mug", None) is not None
                else "039_mug"
            ),
            "{B}": "cooking_range",
            "{C}": "stove_knob",
            "{a}": str(arm),
        }
        return self.info

    def check_success(self):
        """Success: stove on (milk rose toward rim), then off before spill, stove off."""
        if self.overflowed:
            return False
        if not self.turned_on_once:
            return False
        if not self.turned_off_after_boil:
            return False
        if self.stove_on:
            return False
        # Liquid should be heading back / near baseline (permissive).
        if self.liquid_level > self.overflow_level - 0.05:
            return False
        return True

    def get_obs(self):
        obs = super().get_obs()
        obs["boiling"] = {
            "stove_on": bool(self.stove_on),
            "liquid_level": float(self.liquid_level),
            "max_liquid_level": float(self.max_liquid_level),
            "overflowed": bool(self.overflowed),
            "baseline_level": float(self.baseline_level),
            "spill_amount": float(getattr(self, "_spill_amount", 0.0)),
        }
        return obs
