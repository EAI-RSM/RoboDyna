"""KitchenS scene base for RoboDynaExp.

Ports the robotwin_bench KitchenS counter layout (microwave + sink + backsplash
+ chrome gooseneck faucet) into RoboDynaExp's ``Base_Task``.

Default task layout (``replace_sink_with_range=True``): keep a real sink + tap
at the **center** of the counter, and place a cooking range where KitchenS's
sink slot used to be (right in scene_0). The tap always sits behind the sink,
never beside the stove.
"""
from __future__ import annotations

import json
import os

import numpy as np
import sapien
import transforms3d as t3d

from ._base_task import Base_Task
from .utils import *
from .utils.create_actor import create_box, create_visual_box


class KitchenS_base_task(Base_Task):
    """RoboDynaExp KitchenS environment with a cooking range in place of the sink."""

    FURNITURE_NAMES = {"table", "wall", "ground"}

    # Landmarks measured on assets/objects/254_kitchen_stove (Y-up, front = +Z).
    RANGE_PANEL_LOCAL_Z = 1.03      # front panel plane; knobs are moulded past it
    RANGE_KNOB_LOCAL_X = 0.10       # right-hand control of the three-knob row
    RANGE_KNOB_LOCAL_Y = 0.744      # knob row height above the cabinet base
    RANGE_KNOB_LENGTH = 0.055       # world m; deep enough for the gripper to wrap
    # Four cooktop burners as (dx, dy) from ``range_xy``.
    # Seeded from red-grate texture clusters on ``254_kitchen_stove``, then
    # nudged so a skillet bowl lands on the visible X-grate centers.
    RANGE_BURNER_OFFSETS = {
        "left_rear": (-0.050, 0.062),
        "right_rear": (0.050, 0.062),
        "left_front": (-0.050, -0.042),
        "right_front": (0.050, -0.042),
    }

    def setup_demo(self, **kwags):
        # scene_id before init so create_table_and_wall can place fixtures.
        self.scene_id = kwags.get("scene_id")
        if self.scene_id is None:
            self.scene_id = int(np.random.randint(0, 3))
        self.kitchens_info = {
            "table_height": 0.74,
            "table_area": [1.2, 0.7],
            "table_lims": [],
        }
        # Preserve subclass overrides set before super().setup_demo().
        if not hasattr(self, "replace_sink_with_range"):
            # Replace sink+tap with a cooking range (task default).
            self.replace_sink_with_range = True
        if not hasattr(self, "clear_sink_and_range"):
            # Bare counter: no sink, tap, or cooking range (solid top + microwave).
            self.clear_sink_and_range = False
        if not hasattr(self, "omit_sink"):
            # Solid counter + no tap/basin, but still load the cooking range.
            self.omit_sink = False
        if not hasattr(self, "range_position_override"):
            self.range_position_override = None
        super()._init_task_env_(**kwags)

    # ------------------------------------------------------------------
    # Scene layout helpers
    # ------------------------------------------------------------------
    def _get_scene_obj_locations(self, object_name="microwave"):
        """Return [x, y] for KitchenS fixtures.

        scene_0: MW left, Dishrack center, Sink/Range right
        scene_1: MW left, Sink/Range center, Dishrack right
        scene_2: MW center, Dishrack left, Sink/Range right
        """
        if self.scene_id == 0:
            locations = {
                "microwave": [-0.32, 0.18],
                "dishrack": [0.05, 0.17],
                "sink": [0.42, 0.08],
                "range": [0.42, 0.08],
            }
        elif self.scene_id == 1:
            locations = {
                "microwave": [-0.32, 0.18],
                "dishrack": [0.42, 0.17],
                "sink": [0.10, 0.08],
                "range": [0.10, 0.08],
            }
        elif self.scene_id == 2:
            locations = {
                "microwave": [0.10, 0.18],
                "dishrack": [-0.32, 0.17],
                "sink": [0.42, 0.08],
                "range": [0.42, 0.08],
            }
        else:
            raise ValueError(f"Invalid scene_id {self.scene_id}")
        if object_name not in locations:
            raise ValueError(f"Unknown object_name '{object_name}'")
        return locations[object_name]

    def create_table_and_wall(self, table_xy_bias=[0, 0], table_height=0.74):
        """Build the KitchenS counter + fixtures (range replaces sink by default)."""
        self.table_xy_bias = list(table_xy_bias)
        table_height = float(self.kitchens_info["table_height"]) + float(self.table_z_bias)

        if self.random_background:
            texture_type = "seen" if not self.eval_mode else "unseen"
            directory_path = f"./assets/background_texture/{texture_type}"
            file_count = len(
                [n for n in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, n))]
            )
            wall_texture = np.random.randint(0, file_count)
            table_texture = np.random.randint(0, file_count)
            floor_texture = np.random.randint(0, file_count)
            self.wall_texture = f"{texture_type}/{wall_texture}"
            self.table_texture = f"{texture_type}/{table_texture}"
            self.floor_texture = f"{texture_type}/{floor_texture}"
            if np.random.rand() <= self.clean_background_rate:
                self.wall_texture = None
            if np.random.rand() <= self.clean_background_rate:
                self.table_texture = None
            if np.random.rand() <= self.clean_background_rate:
                self.floor_texture = None
        else:
            self.wall_texture = self.table_texture = self.floor_texture = None

        # Floor tiles
        for i, pos in enumerate([[1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0]]):
            create_box(
                self.scene,
                sapien.Pose(p=pos),
                half_size=[1, 1, 0.005],
                color=(0.85, 0.85, 0.85),
                name=f"floor_{i}",
                texture_id=self.floor_texture,
                is_static=True,
            )

        self.wall = create_box(
            self.scene,
            sapien.Pose(p=[0, 1, 1.5]),
            half_size=[3, 0.6, 1.5],
            color=(1, 0.9, 0.9),
            name="wall",
            texture_id=self.wall_texture,
            is_static=True,
        )

        counter_length = self.kitchens_info["table_area"][0]
        counter_width = self.kitchens_info["table_area"][1]
        counter_thickness = 0.04
        self.kitchens_info["counter_thickness"] = counter_thickness

        # Sink stays at the center (dishrack slot). The cooking range occupies the
        # former KitchenS sink slot on the side — tap never goes next to the stove.
        clear_fixtures = bool(getattr(self, "clear_sink_and_range", False))
        omit_sink = bool(getattr(self, "omit_sink", False))
        if not clear_fixtures and not omit_sink:
            if getattr(self, "replace_sink_with_range", True):
                sink_rel_x, sink_rel_y = self._get_scene_obj_locations("dishrack")
                # Slightly forward of the shallow rack pose so the basin reads as a sink.
                sink_rel_y = 0.10
            else:
                sink_rel_x, sink_rel_y = self._get_scene_obj_locations("sink")
            self.kitchens_info["sink_geom"] = {
                "rel_p": [sink_rel_x, sink_rel_y],
                "hole_hx": 0.12,
                "hole_hy": 0.16,
                "depth": 0.09,
                "inner_hx": 0.11,
                "inner_hy": 0.15,
            }

        self._create_backsplash(counter_length, counter_width, table_height, table_xy_bias)

        if clear_fixtures or omit_sink:
            # Solid counter (no sink hole / tap). Range may still load when omit_sink.
            self._create_solid_counter(
                counter_length, counter_width, counter_thickness, table_height, table_xy_bias
            )
        else:
            # Cut a sink hole — the basin (and its tap) live at sink_geom.
            self._create_counter_with_sink_hole(
                counter_length, counter_width, counter_thickness, table_height, table_xy_bias
            )

        self._create_base_cabinets(counter_length, counter_width, table_height, counter_thickness, table_xy_bias)
        self._create_counter_edge_trim(counter_length, counter_width, table_height, counter_thickness, table_xy_bias)
        # Keep the upper wall clear: this task does not use the decorative shelves.

        self.kitchens_info["table_lims"] = [
            -counter_length / 2, -counter_width / 2,
            counter_length / 2, counter_width / 2,
        ]

        self._load_microwave(table_height, table_xy_bias)
        if clear_fixtures:
            return
        # Center is the sink; skip the dishrack stand-in in the range layout.
        if not getattr(self, "replace_sink_with_range", True):
            self._load_dishrack(table_height, table_xy_bias)
        if not omit_sink:
            self._load_sink(table_height, table_xy_bias)
        if getattr(self, "replace_sink_with_range", True):
            self._load_cooking_range(table_height, table_xy_bias)

    # ------------------------------------------------------------------
    # Counter & decorative elements
    # ------------------------------------------------------------------
    def _counter_material(self):
        if self.table_texture is not None:
            texture_path = f"./assets/background_texture/{self.table_texture}.png"
            texture2d = sapien.render.RenderTexture2D(texture_path)
            mat = sapien.render.RenderMaterial()
            mat.set_base_color_texture(texture2d)
            mat.base_color = [1, 1, 1, 1]
            mat.metallic = 0.1
            mat.roughness = 0.3
            return mat
        mat = sapien.render.RenderMaterial(base_color=[0.28, 0.27, 0.26, 1])
        mat.metallic = 0.12
        mat.roughness = 0.22
        return mat

    def _create_solid_counter(self, counter_length, counter_width, counter_thickness, table_height, table_xy_bias):
        th = counter_thickness / 2
        counter_top_z = table_height - th
        mat = self._counter_material()
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        builder.add_box_collision(
            pose=sapien.Pose([0, 0, 0]),
            half_size=[counter_length / 2, counter_width / 2, th],
            material=self.scene.default_physical_material,
        )
        builder.add_box_visual(
            pose=sapien.Pose([0, 0, 0]),
            half_size=[counter_length / 2, counter_width / 2, th],
            material=mat,
        )
        builder.set_initial_pose(sapien.Pose(p=[table_xy_bias[0], table_xy_bias[1], counter_top_z]))
        self.table = builder.build(name="table")

    def _create_counter_with_sink_hole(
        self, counter_length, counter_width, counter_thickness, table_height, table_xy_bias
    ):
        sink_geom = self.kitchens_info["sink_geom"]
        sink_rel_x, sink_rel_y = sink_geom["rel_p"]
        sink_hx, sink_hy = sink_geom["hole_hx"], sink_geom["hole_hy"]
        th = counter_thickness / 2
        counter_top_z = table_height - th
        cl, cr = -counter_length / 2, counter_length / 2
        cf, cb = -counter_width / 2, counter_width / 2
        hl, hr = sink_rel_x - sink_hx, sink_rel_x + sink_hx
        hf, hb = sink_rel_y - sink_hy, sink_rel_y + sink_hy
        mat = self._counter_material()
        counter_pieces = [
            ("right", hr, cr, cf, cb),
            ("left", cl, hl, cf, cb),
            ("front", hl, hr, cf, hf),
            ("back", hl, hr, hb, cb),
        ]
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        for _, x1, x2, y1, y2 in counter_pieces:
            hx = (x2 - x1) / 2
            hy = (y2 - y1) / 2
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            if hx <= 1e-6 or hy <= 1e-6:
                continue
            piece_pose = sapien.Pose([cx, cy, 0])
            builder.add_box_collision(
                pose=piece_pose, half_size=[hx, hy, th], material=self.scene.default_physical_material
            )
            builder.add_box_visual(pose=piece_pose, half_size=[hx, hy, th], material=mat)
        builder.set_initial_pose(sapien.Pose(p=[table_xy_bias[0], table_xy_bias[1], counter_top_z]))
        self.table = builder.build(name="table")

    def _create_backsplash(self, counter_length, counter_width, table_height, table_xy_bias):
        create_visual_box(
            self.scene,
            sapien.Pose(p=[
                table_xy_bias[0],
                table_xy_bias[1] + counter_width / 2 + 0.005,
                table_height + 0.22,
            ]),
            half_size=[counter_length / 2 + 0.02, 0.006, 0.22],
            color=(0.92, 0.94, 0.96),
            name="backsplash",
        )

    def _create_base_cabinets(self, counter_length, counter_width, table_height, counter_thickness, table_xy_bias):
        cabinet_h = (table_height - counter_thickness) / 2
        cx, cy = table_xy_bias[0], table_xy_bias[1]
        create_box(
            self.scene,
            sapien.Pose(p=[cx, cy, cabinet_h]),
            half_size=[counter_length / 2 - 0.01, counter_width / 2 - 0.01, cabinet_h],
            color=(0.42, 0.36, 0.30),
            name="base_cabinets",
            is_static=True,
        )

    def _create_counter_edge_trim(
        self, counter_length, counter_width, table_height, counter_thickness, table_xy_bias
    ):
        create_box(
            self.scene,
            sapien.Pose(p=[
                table_xy_bias[0],
                table_xy_bias[1] - counter_width / 2,
                table_height - counter_thickness / 2,
            ]),
            half_size=[counter_length / 2, 0.006, counter_thickness / 2 + 0.002],
            color=(0.12, 0.12, 0.12),
            name="counter_edge_trim",
            is_static=True,
        )

    def _create_upper_shelves(self, table_height, table_xy_bias):
        counter_width = self.kitchens_info["table_area"][1]
        cx, cy = table_xy_bias[0], table_xy_bias[1]
        shelf_base_z = table_height + 0.32
        for side, sx in (("left", -0.45), ("right", 0.45)):
            create_box(
                self.scene,
                sapien.Pose(p=[cx + sx, cy + counter_width / 2 - 0.08, shelf_base_z]),
                half_size=[0.12, 0.08, 0.01],
                color=(0.55, 0.48, 0.40),
                name=f"upper_shelf_{side}",
                is_static=True,
            )

    # ------------------------------------------------------------------
    # Fixtures
    # ------------------------------------------------------------------
    def _load_microwave(self, table_height, table_xy_bias):
        """Load the real 044_microwave mesh with planner-safe box collision.

        Loading the articulated URDF directly gives its collision meshes a
        non-unit scale, which mplib/CuRobo rejects. The visual mesh can still be
        scaled normally; only its collision is replaced by a unit-scale box.
        """
        x, y = self._get_scene_obj_locations("microwave")
        x += table_xy_bias[0]
        y += table_xy_bias[1]

        visual_path = "assets/objects/044_microwave/visual/base0.glb"
        scale = 0.15
        bounds_min = np.array([-0.761475, -0.495360, -0.567261], dtype=float)
        bounds_max = np.array([0.795292, 0.435315, 0.630773], dtype=float)
        center = 0.5 * (bounds_min + bounds_max) * scale
        half_size = 0.5 * (bounds_max - bounds_min) * scale

        # Authored GLB has the door on a side face; yaw +90° (KitchenS / robotwin)
        # faces the door toward the robot workspace.
        qx, qy, qz, qw = t3d.euler.euler2quat(0.0, 0.0, np.pi / 2.0, axes="sxyz")
        mw_q = [float(qw), float(qx), float(qy), float(qz)]
        R = t3d.quaternions.quat2mat(mw_q)
        corners = []
        for sx in (bounds_min[0], bounds_max[0]):
            for sy in (bounds_min[1], bounds_max[1]):
                for sz in (bounds_min[2], bounds_max[2]):
                    corners.append(R @ (np.array([sx, sy, sz], dtype=float) * scale))
        corners = np.asarray(corners, dtype=float)

        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        builder.add_box_collision(
            pose=sapien.Pose(p=center),
            half_size=half_size,
            material=self.scene.default_physical_material,
        )
        builder.add_visual_from_file(
            filename=visual_path,
            scale=[scale, scale, scale],
        )
        z = table_height - float(corners[:, 2].min())
        builder.set_initial_pose(sapien.Pose(p=[x, y, z], q=mw_q))
        microwave_entity = builder.build(name="microwave")
        self.microwave = Actor(
            microwave_entity,
            {
                "center": center.tolist(),
                "extents": (bounds_max - bounds_min).tolist(),
                "scale": scale,
            },
        )
        self.add_prohibit_area(self.microwave, padding=0.02)

    def _load_dishrack(self, table_height, table_xy_bias):
        """Procedural dish-rack stand-in (scaled GLB collisions trip mplib/CuRobo)."""
        x, y = self._get_scene_obj_locations("dishrack")
        x += table_xy_bias[0]
        y += table_xy_bias[1]
        # Shallow open basket on the counter.
        wall_t = 0.008
        hx, hy, hz = 0.10, 0.08, 0.045
        floor_z = table_height + 0.006
        floor = create_box(
            self.scene,
            sapien.Pose(p=[x, y, floor_z]),
            half_size=[hx, hy, 0.006],
            color=(0.75, 0.75, 0.78),
            name="dishrack_floor",
            is_static=True,
        )
        for name, pos, half in (
            ("front", [x, y - hy + wall_t / 2, table_height + hz / 2], [hx, wall_t / 2, hz / 2]),
            ("back", [x, y + hy - wall_t / 2, table_height + hz / 2], [hx, wall_t / 2, hz / 2]),
            ("left", [x - hx + wall_t / 2, y, table_height + hz / 2], [wall_t / 2, hy - wall_t, hz / 2]),
            ("right", [x + hx - wall_t / 2, y, table_height + hz / 2], [wall_t / 2, hy - wall_t, hz / 2]),
        ):
            create_box(
                self.scene,
                sapien.Pose(p=pos),
                half_size=half,
                color=(0.70, 0.70, 0.74),
                name=f"dishrack_{name}",
                is_static=True,
            )
        self.dishrack = floor
        self.dishrack.set_name("dishrack")
        self.basin_xy = (float(x), float(y))

        pad = 0.03
        self.prohibited_area.append([
            x - hx - pad, y - hy - pad, x + hx + pad, y + hy + pad
        ])

    def _load_cooking_range(self, table_height, table_xy_bias):
        """CC0 multi-burner range with a graspable front rotary knob.

        The textured source mesh supplies the oven, burners, grates, and control
        panel. Its collision is deliberately reduced to one unit-scale box so
        mplib/CuRobo never sees a scaled triangle mesh.
        """
        override = getattr(self, "range_position_override", None)
        if override is not None and len(override) >= 2:
            rel_x, rel_y = float(override[0]), float(override[1])
        else:
            rel_x, rel_y = self._get_scene_obj_locations("range")
        x = rel_x + table_xy_bias[0]
        y = rel_y + table_xy_bias[1]

        asset_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "assets", "objects", "254_kitchen_stove",
        )
        with open(os.path.join(asset_dir, "model_data0.json"), "r") as f:
            model_data = json.load(f)
        scale = np.asarray(model_data["scale"], dtype=float)
        center = np.asarray(model_data["center"], dtype=float)
        extents = np.asarray(model_data["extents"], dtype=float)

        # Asset convention is Y-up. Rotate +90° about X so +Y becomes world +Z
        # and the model's front (+Z) faces the robot (-world Y).
        range_q = [0.70710678, 0.70710678, 0.0, 0.0]
        center_scaled = center * scale
        # Offset the origin so the rotated visual footprint remains centered at (x, y).
        origin_y = y + float(center_scaled[2])

        # Collision covers the cabinet only. The control knobs protrude to
        # local z=1.258; including them would seal the gap the gripper needs.
        body_front_z = self.RANGE_PANEL_LOCAL_Z
        body_half = np.array(
            [extents[0] / 2, extents[1] / 2, body_front_z], dtype=float
        ) * scale
        body_center = np.array([0.0, extents[1] / 2, 0.0], dtype=float) * scale

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
            sapien.Pose(p=[x, origin_y, table_height], q=range_q)
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

        model_width = float(extents[0] * scale[0])
        model_depth = float(extents[2] * scale[2])
        model_height = float(extents[1] * scale[1])
        self.range_xy = (float(x), float(y))
        self.range_half_size = (0.5 * model_width, 0.5 * model_depth)
        self.range_top_z = float(table_height + model_height)

        # World XY of each of the four burners (grate centers).
        self.burner_positions = {
            name: (float(x + dx), float(y + dy))
            for name, (dx, dy) in self.RANGE_BURNER_OFFSETS.items()
        }
        # Default active burner: left-rear (boil_milk pot / cook_food pan).
        self.burner_xy = self.burner_positions["left_rear"]

        # Dark overlay on the active burner; the task changes it to orange while on.
        burner_mat = sapien.render.RenderMaterial(
            base_color=[0.20, 0.20, 0.22, 1.0]
        )
        burner_builder = self.scene.create_actor_builder()
        burner_builder.add_cylinder_visual(
            pose=sapien.Pose(
                [0, 0, 0], [0.70710678, 0.0, 0.70710678, 0.0]
            ),
            radius=0.030,
            half_length=0.0015,
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

        # Interactive rotary knob, mounted on the front panel concentric with one
        # of the mesh's moulded knobs so it reads as part of the appliance. It is
        # deeper than the moulded stub purely so a gripper can wrap around it.
        panel_y = float(origin_y - body_front_z * scale[2])
        knob_x = float(x + self.RANGE_KNOB_LOCAL_X * scale[0])
        knob_z = float(table_height + self.RANGE_KNOB_LOCAL_Y * scale[1])
        knob_r = 0.019
        knob_half_length = float(self.RANGE_KNOB_LENGTH / 2)
        knob_y = float(panel_y - knob_half_length)
        knob_mat = sapien.render.RenderMaterial(
            base_color=[0.07, 0.07, 0.08, 1.0]
        )
        knob_mat.metallic = 0.15
        knob_mat.roughness = 0.35
        knob_builder = self.scene.create_actor_builder()
        knob_builder.set_physx_body_type("static")
        knob_pose = sapien.Pose(
            [0, 0, 0], [0.70710678, 0.0, 0.0, 0.70710678]
        )
        knob_builder.add_cylinder_collision(
            pose=knob_pose,
            radius=knob_r,
            half_length=knob_half_length,
            material=self.scene.default_physical_material,
        )
        knob_builder.add_cylinder_visual(
            pose=knob_pose,
            radius=knob_r,
            half_length=knob_half_length,
            material=knob_mat,
        )
        knob_builder.set_initial_pose(sapien.Pose(p=[knob_x, knob_y, knob_z]))
        self.stove_knob = knob_builder.build(name="stove_knob")

        # White indicator visibly rotates from 12 o'clock (off) to 3 o'clock (on).
        self._knob_radius = knob_r
        self._knob_front_y = float(knob_y - knob_half_length - 0.002)
        self.stove_knob_indicator = create_visual_box(
            self.scene,
            sapien.Pose(
                p=[knob_x, self._knob_front_y, knob_z + knob_r * 0.55]
            ),
            half_size=[0.0025, 0.0015, 0.007],
            color=(0.92, 0.92, 0.88),
            name="stove_knob_indicator",
        )
        self.knob_xy = (knob_x, knob_y)
        self.knob_xyz = (knob_x, knob_y, knob_z)
        self.knob_top_z = float(knob_z + knob_r)

        pad = 0.03
        self.prohibited_area.append([
            x - model_width / 2 - pad,
            min(y - model_depth / 2, knob_y - knob_half_length) - pad,
            x + model_width / 2 + pad,
            y + model_depth / 2 + pad,
        ])

    def _load_faucet(self, x, faucet_y, table_height):
        """KitchenS chrome gooseneck tap: upright post + spout + side handle.

        Visual-only (no collision) so it never blocks the planner — same as
        robotwin_bench KitchenS.
        """
        def _chrome_box(name, pose, half_size):
            mat = sapien.render.RenderMaterial(base_color=[0.78, 0.80, 0.84, 1.0])
            mat.metallic = 0.9
            mat.roughness = 0.18
            entity = sapien.Entity()
            entity.set_name(name)
            render = sapien.render.RenderBodyComponent()
            render.attach(sapien.render.RenderShapeBox(list(half_size), mat))
            entity.add_component(render)
            entity.set_pose(pose)
            self.scene.add_entity(entity)
            return entity

        self.faucet_post = _chrome_box(
            "faucet_post",
            sapien.Pose(p=[x, faucet_y, table_height + 0.12]),
            [0.014, 0.014, 0.12],
        )
        self.faucet_spout = _chrome_box(
            "faucet_spout",
            sapien.Pose(p=[x, faucet_y - 0.07, table_height + 0.22]),
            [0.011, 0.08, 0.011],
        )
        self.faucet_handle = _chrome_box(
            "faucet_handle",
            sapien.Pose(p=[x + 0.045, faucet_y, table_height + 0.07]),
            [0.016, 0.008, 0.018],
        )
        self.faucet_xy = (float(x), float(faucet_y))

    def _load_sink(self, table_height, table_xy_bias):
        """KitchenS sink basin + chrome tap mounted behind the back rim."""
        sink_geom = self.kitchens_info["sink_geom"]
        rel_x, rel_y = sink_geom["rel_p"]
        x = rel_x + table_xy_bias[0]
        y = rel_y + table_xy_bias[1]
        hole_hx, hole_hy = sink_geom["hole_hx"], sink_geom["hole_hy"]
        depth = sink_geom["depth"]
        inner_hx, inner_hy = sink_geom["inner_hx"], sink_geom["inner_hy"]
        sink_z = table_height
        wall_thickness = hole_hx - inner_hx
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        material = sapien.render.RenderMaterial(base_color=[0.75, 0.75, 0.78, 1.0])
        material.metallic = 0.6
        material.roughness = 0.3
        bottom_half = [inner_hx, inner_hy, wall_thickness / 2]
        bottom_pose = sapien.Pose([0, 0, -depth + wall_thickness / 2])
        builder.add_box_collision(pose=bottom_pose, half_size=bottom_half)
        builder.add_box_visual(pose=bottom_pose, half_size=bottom_half, material=material)
        walls = [
            ([hole_hx - wall_thickness / 2, 0, -depth / 2], [wall_thickness / 2, hole_hy, depth / 2]),
            ([-(hole_hx - wall_thickness / 2), 0, -depth / 2], [wall_thickness / 2, hole_hy, depth / 2]),
            ([0, hole_hy - wall_thickness / 2, -depth / 2], [inner_hx, wall_thickness / 2, depth / 2]),
            ([0, -(hole_hy - wall_thickness / 2), -depth / 2], [inner_hx, wall_thickness / 2, depth / 2]),
        ]
        for w_pos, w_half in walls:
            wp = sapien.Pose(w_pos)
            builder.add_box_collision(pose=wp, half_size=w_half)
            builder.add_box_visual(pose=wp, half_size=w_half, material=material)
        builder.set_initial_pose(sapien.Pose(p=[x, y, sink_z]))
        self.sink = builder.build(name="sink")
        self.basin_xy = (float(x), float(y))
        # Tap behind the sink, spout over the basin — not beside the stove.
        faucet_y = float(y + inner_hy + 0.03)
        self._load_faucet(x, faucet_y, table_height)
        sink_pad = 0.03
        self.prohibited_area.append([
            x - hole_hx - sink_pad,
            y - hole_hy - sink_pad,
            x + hole_hx + sink_pad,
            faucet_y + sink_pad,
        ])
