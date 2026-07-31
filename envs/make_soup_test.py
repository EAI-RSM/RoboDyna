"""make_soup_test: full make_soup replica with a flush countertop gas cooktop.

Same pour-then-heat expert as ``make_soup``, but the freestanding oven range is
replaced by a CC0 4-burner gas cooktop (``268_countertop_gas_stove``) that sits
nearly flush with the counter. The interactive rotary knob is mounted on the
cooktop *top*, axis perpendicular to the table, so the arm must grasp and twist
it from above.
"""
from __future__ import annotations

import json
import os
from typing import Any, ClassVar

import numpy as np
import sapien
import sapien.render
import transforms3d as t3d

from ._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
from .make_soup import make_soup
from .utils import *
from .utils.actor_utils import Actor
from .utils.create_actor import create_visual_box


class make_soup_test(make_soup):
    """make_soup with an embedded countertop gas stove + top-facing knob."""

    # Flush cooktop: sit near the back of the apron; no tall front panel.
    RANGE_REL_XY: ClassVar[tuple[float, float]] = (0.18, 0.14)
    RANGE_SCALE_MULT: ClassVar[float] = 1.0
    ACTIVE_BURNER: ClassVar[str] = "left_front"
    # World-frame offsets from cooktop center (measured on 268 grate clusters).
    BURNER_OFFSETS: ClassVar[dict[str, tuple[float, float]]] = {
        "left_front": (-0.103, -0.103),
        "right_front": (0.103, -0.103),
        "left_rear": (-0.103, 0.103),
        "right_rear": (0.103, 0.103),
        "center": (0.0, 0.0),
    }

    # Approach straight down onto the vertical knob.
    KNOB_APPROACH_PATH: ClassVar[tuple] = (
        (0.00, 0.00, 0.22),
        (0.00, 0.00, 0.14),
        (0.00, 0.00, 0.08),
    )
    KNOB_GRASP_STANDOFF: ClassVar[float] = 0.012
    KNOB_RADIUS: ClassVar[float] = 0.020
    KNOB_HEIGHT: ClassVar[float] = 0.028  # full height above cooktop slab
    # Bottom-right corner of the cooktop (robot-facing front, +X).
    KNOB_LOCAL_XY: ClassVar[tuple[float, float]] = (0.165, -0.165)

    COOKTOP_ASSET: ClassVar[str] = "268_countertop_gas_stove"

    def setup_demo(self, **kwags: Any) -> None:
        # Parent reads task_args.make_soup — mirror our block into that key.
        ta = dict(kwags.get("task_args") or {})
        cfg = dict(ta.get("make_soup_test") or ta.get("make_soup") or {})
        ta["make_soup"] = cfg
        kwags["task_args"] = ta
        # Defaults that differ from the freestanding range.
        if "range_xy" not in cfg:
            cfg["range_xy"] = list(self.RANGE_REL_XY)
        if "range_scale_mult" not in cfg:
            cfg["range_scale_mult"] = float(self.RANGE_SCALE_MULT)
        if "burner" not in cfg:
            cfg["burner"] = self.ACTIVE_BURNER
        ta["make_soup"] = cfg
        super().setup_demo(**kwags)

    # ---------------------------------------------------------------- cooktop
    def _load_cooking_range(self, table_height, table_xy_bias):
        """Flat CC0 gas cooktop + vertical (top-facing) rotary knob."""
        override = getattr(self, "range_position_override", None)
        if override is not None and len(override) >= 2:
            rel_x, rel_y = float(override[0]), float(override[1])
        else:
            rel_x, rel_y = self._get_scene_obj_locations("range")
        x = rel_x + table_xy_bias[0]
        y = rel_y + table_xy_bias[1]

        asset_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "assets",
            "objects",
            self.COOKTOP_ASSET,
        )
        with open(os.path.join(asset_dir, "model_data0.json"), "r") as f:
            model_data = json.load(f)
        scale_mult = float(getattr(self, "range_scale_mult", 1.0))
        scale = np.asarray(model_data["scale"], dtype=float) * scale_mult
        center = np.asarray(model_data["center"], dtype=float)
        extents = np.asarray(model_data["extents"], dtype=float)

        # Asset is Y-up. Rotate +90° about X so +Y → world +Z and the model's
        # front (+Z) faces the robot (−world Y) — same convention as 254.
        range_q = [0.70710678, 0.70710678, 0.0, 0.0]
        center_scaled = center * scale
        origin_y = y + float(center_scaled[2])

        model_width = float(extents[0] * scale[0])
        model_depth = float(extents[2] * scale[2])
        # Mesh Y-up: after Rx+90°, local Y → world Z. Grate top = ymax_s;
        # slab sits ~2.5 cm below that. Leave grate top ~3 cm above the counter
        # so the slab reads as a flush built-in hob.
        ymax_s = float((center[1] + 0.5 * extents[1]) * scale[1])
        lip_above = 0.030
        cooktop_z0 = float(table_height + lip_above - ymax_s)
        range_top_z = float(table_height + lip_above)

        # Planner-safe box covering only the protruding lip (local Y-up frame).
        lip_half = 0.5 * lip_above
        body_half = np.array(
            [0.5 * model_width, lip_half, 0.5 * model_depth], dtype=float
        )
        body_center = np.array(
            [0.0, ymax_s - lip_half, 0.0], dtype=float
        )

        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        builder.add_box_collision(
            pose=sapien.Pose(p=body_center),
            half_size=body_half,
            material=self.scene.default_physical_material,
        )
        builder.add_visual_from_file(
            filename=os.path.join(asset_dir, "visual", "base0.glb"),
            scale=scale.tolist(),
        )
        builder.set_initial_pose(
            sapien.Pose(p=[x, origin_y, cooktop_z0], q=range_q)
        )
        range_entity = builder.build(name="cooking_range")
        self.range_body = Actor(
            range_entity,
            {
                "center": center.tolist(),
                "extents": extents.tolist(),
                "scale": scale.tolist(),
            },
        )

        self.range_xy = (float(x), float(y))
        self.range_half_size = (0.5 * model_width, 0.5 * model_depth)
        self.range_top_z = range_top_z

        cx, cy = float(x), float(y)
        self.burner_positions = {
            name: (float(cx + dx * scale_mult), float(cy + dy * scale_mult))
            for name, (dx, dy) in self.BURNER_OFFSETS.items()
        }
        self.burner_xy = self.burner_positions[self.ACTIVE_BURNER]

        burner_mat = sapien.render.RenderMaterial(
            base_color=[0.20, 0.20, 0.22, 1.0]
        )
        burner_r = 0.035 * scale_mult
        burner_builder = self.scene.create_actor_builder()
        burner_builder.add_cylinder_visual(
            pose=sapien.Pose(
                [0, 0, 0], [0.70710678, 0.0, 0.70710678, 0.0]
            ),
            radius=burner_r,
            half_length=0.0015 * scale_mult,
            material=burner_mat,
        )
        burner_builder.set_initial_pose(
            sapien.Pose(
                p=[
                    self.burner_xy[0],
                    self.burner_xy[1],
                    self.range_top_z + 0.002,
                ]
            )
        )
        self.active_burner = burner_builder.build(name="active_burner")
        self._burner_shapes = []
        for c in self.active_burner.get_components():
            if isinstance(c, sapien.render.RenderBodyComponent):
                self._burner_shapes = list(c.render_shapes)

        # Vertical knob on the cooktop slab (axis = world +Z), not on the
        # grate plane — keeps it clear of the burner rings/grates.
        knob_dx, knob_dy = self.KNOB_LOCAL_XY
        knob_x = float(x + knob_dx * scale_mult)
        knob_y = float(y + knob_dy * scale_mult)
        knob_r = float(self.KNOB_RADIUS) * scale_mult
        knob_half = float(self.KNOB_HEIGHT) * scale_mult / 2.0
        # Slab sits ~2.5 cm below the grate top on this asset.
        slab_top_z = float(self.range_top_z - 0.025)
        knob_z = float(slab_top_z + knob_half)
        knob_mat = sapien.render.RenderMaterial(
            base_color=[0.07, 0.07, 0.08, 1.0]
        )
        knob_mat.metallic = 0.15
        knob_mat.roughness = 0.35
        # Cylinder default axis is X; this quat aligns it with world +Z.
        knob_pose = sapien.Pose(
            [0, 0, 0], [0.70710678, 0.0, 0.70710678, 0.0]
        )
        knob_builder = self.scene.create_actor_builder()
        knob_builder.set_physx_body_type("static")
        knob_builder.add_cylinder_collision(
            pose=knob_pose,
            radius=knob_r,
            half_length=knob_half,
            material=self.scene.default_physical_material,
        )
        knob_builder.add_cylinder_visual(
            pose=knob_pose,
            radius=knob_r,
            half_length=knob_half,
            material=knob_mat,
        )
        knob_builder.set_initial_pose(sapien.Pose(p=[knob_x, knob_y, knob_z]))
        self.stove_knob = knob_builder.build(name="stove_knob")

        self._knob_radius = knob_r
        self._knob_half = knob_half
        # Indicator rides on the top face of the knob (XY plane).
        self._knob_indicator_z = float(knob_z + knob_half + 0.002)
        self.stove_knob_indicator = create_visual_box(
            self.scene,
            sapien.Pose(
                p=[knob_x, knob_y + knob_r * 0.55, self._knob_indicator_z]
            ),
            half_size=[0.0025, 0.007, 0.0015],
            color=(0.92, 0.92, 0.88),
            name="stove_knob_indicator",
        )
        # Unused by the top-knob indicator path, but keep for parent helpers.
        self._knob_front_y = float(knob_y)
        self.knob_xy = (knob_x, knob_y)
        self.knob_xyz = (knob_x, knob_y, knob_z)
        self.knob_top_z = float(knob_z + knob_half)
        self._top_knob = True

        pad = 0.03
        self.prohibited_area.append(
            [
                x - model_width / 2 - pad,
                y - model_depth / 2 - pad,
                x + model_width / 2 + pad,
                y + model_depth / 2 + pad,
            ]
        )
        self._ring_parts = getattr(self, "_ring_parts", []) or []
        self._ring_shapes = getattr(self, "_ring_shapes", []) or []
        self._ring_home_poses = getattr(self, "_ring_home_poses", []) or []
        self._disc_parts = getattr(self, "_disc_parts", []) or []
        self._disc_home_poses = getattr(self, "_disc_home_poses", []) or []
        self._burner_home_pose = sapien.Pose(
            p=[
                self.burner_xy[0],
                self.burner_xy[1],
                self.range_top_z + 0.002,
            ]
        )

    def _set_stove_fire(self, on: bool, intensity: float = 1.0) -> None:
        super()._set_stove_fire(on, intensity)
        # Parent indicator math assumes a front-facing knob; rewrite for top.
        if not getattr(self, "_top_knob", False):
            return
        if getattr(self, "stove_knob_indicator", None) is None:
            return
        lit = bool(on) and float(intensity) > 1e-3
        if hasattr(self, "knob_angle") and hasattr(self, "fire_intensity"):
            angle = float(self.knob_angle)
        else:
            angle = -np.pi / 2 if lit else 0.0
        radius = float(getattr(self, "_knob_radius", 0.018)) * 0.55
        kx, ky, _ = self.knob_xyz
        self.stove_knob_indicator.set_pose(
            sapien.Pose(
                p=[
                    float(kx + radius * np.sin(angle)),
                    float(ky + radius * np.cos(angle)),
                    float(self._knob_indicator_z),
                ],
                q=[
                    float(np.cos(angle / 2)),
                    0.0,
                    0.0,
                    float(np.sin(angle / 2)),
                ],
            )
        )

    # ---------------------------------------------------------------- top knob
    def _knob_pose(self, offset, turn_angle: float) -> list[float]:
        """Top-down EE pose above the knob, yawed by ``turn_angle`` about +Z."""
        base_q = np.asarray(GRASP_DIRECTION_DIC["top_down"], dtype=float)
        ee_p = np.asarray(self.knob_xyz, dtype=float) + np.asarray(offset, dtype=float)
        twist_q = np.array(
            [np.cos(turn_angle / 2), 0.0, 0.0, np.sin(turn_angle / 2)],
            dtype=float,
        )
        ee_q = t3d.quaternions.qmult(twist_q, base_q)
        return [*ee_p.tolist(), *ee_q.tolist()]

    def _knob_turn_pose(self, standoff: float, turn_angle: float) -> list[float]:
        """Grasp from above: EE sits ``EE_TO_TCP + standoff`` above the knob."""
        return self._knob_pose(
            [0.0, 0.0, self.EE_TO_TCP + float(standoff)], turn_angle
        )

    def _turn_knob_on(self) -> None:
        """Descend onto the top-facing knob and twist it on."""
        arm = self.arm
        start_angle = -np.pi / 2 if self.stove_on else 0.0
        end_angle = -np.pi / 2
        path = self.KNOB_APPROACH_PATH

        self._ignore_knob = True
        self.move(self.open_gripper(arm))
        for offset in path:
            self.move(self.move_to_pose(arm, self._knob_pose(offset, start_angle)))
        self.move(
            self.move_to_pose(
                arm, self._knob_turn_pose(self.KNOB_GRASP_STANDOFF, start_angle)
            )
        )
        self.move(self.close_gripper(arm))
        self._expert_holding_knob = True
        self.move(
            self.move_to_pose(
                arm, self._knob_turn_pose(self.KNOB_GRASP_STANDOFF, end_angle)
            )
        )
        self._expert_holding_knob = False
        self._set_stove(True)
        self._idle_steps(8)
        self.move(self.open_gripper(arm))
        for offset in reversed(path):
            self.move(self.move_to_pose(arm, self._knob_pose(offset, end_angle)))
        self._ignore_knob = False
        self._prev_knob_pressed = False

    def play_once(self) -> dict[str, Any]:
        info = super().play_once()
        # Relabel the appliance in language templates.
        if isinstance(self.info.get("info"), dict):
            self.info["info"]["{C}"] = "countertop_gas_stove"
        return info
