"""Kitchen Large scene base for RoboDynaExp.

Ports the robotwin_bench Kitchen Large counter layout (fridge + microwave +
basket) into RoboDynaExp's ``Base_Task``, matching the fixture placements in
``robotwin_bench/.../kitchenl/_kitchen_base_large.py`` (wall cabinet omitted).
"""
from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import sapien
import transforms3d as t3d

from ._base_task import Base_Task
from .utils import *
from .utils.actor_utils import Actor, ArticulationActor


class KitchenL_base_task(Base_Task):
    """RoboDynaExp Kitchen Large environment (fridge / microwave / basket)."""

    FURNITURE_NAMES = {
        "table",
        "wall",
        "ground",
        "fridge_left",
        "microwave_center",
        "basket_right",
    }

    def setup_demo(self, **kwags):
        self.scene_id = kwags.get("scene_id")
        if self.scene_id is None:
            self.scene_id = int(np.random.randint(0, 3))
        self.jitter_basket = bool(kwags.get("jitter_basket", True))
        self.skip_microwave = bool(kwags.get("skip_microwave", False))
        self.skip_scene_basket = bool(kwags.get("skip_scene_basket", False))
        self.kitchenl_info = {
            "table_height": 0.74,
            "table_area": [1.2, 0.7],
            "table_lims": [],
        }
        # Appliance defaults. Yaw -90° faces the door toward the robot (−Y);
        # robotwin used +90° (door into the wall), which reads as a solid block.
        self.fridge_left_rot = [-90.0, 0.0, -90.0]
        self.fridge_left_scale = 0.5
        self.microwave_left_rot = [-90.0, 180.0, 0.0]
        self.microwave_left_scale = 1.4
        self.basket_right_rot = [0.0, 0.0, 90.0]
        self.basket_right_scale = 1.4
        self.basket_right_position_jitter_x = (-0.04, 0.04)
        self.basket_right_position_jitter_y = (-0.04, 0.04)
        self.basket_right_modelname = "063_tabletrashbin"
        self.basket_right_model_id = 6
        self.cabinet_scale = 0.5
        super()._init_task_env_(**kwags)

    # ------------------------------------------------------------------
    # Scene layout
    # ------------------------------------------------------------------
    def _get_scene_obj_locations(self, object_name="microwave"):
        """Return fixture XY (and optional Z) for Kitchen Large scene_id 0/1/2."""
        if self.scene_id == 0:
            microwave_location = [0.0, 0.30]
            basket_location = [-0.37, 0.12, 0.0]
        elif self.scene_id == 1:
            microwave_location = [-0.4, 0.30]
            basket_location = [0.0, 0.15, 0.0]
        elif self.scene_id == 2:
            microwave_location = [-0.4, 0.10]
            basket_location = [-0.4, 0.07, 0.927]
        else:
            raise ValueError(f"Invalid scene_id {self.scene_id}")
        if object_name == "microwave":
            return microwave_location
        if object_name == "basket":
            return basket_location
        raise ValueError(f"Object name {object_name} is not supported")

    def create_table_and_wall(self, table_xy_bias=[0, 0], table_height=0.74):
        """Build the Kitchen Large counter + fridge / microwave / basket."""
        self.table_xy_bias = list(table_xy_bias)
        table_height = float(self.kitchenl_info["table_height"]) + float(self.table_z_bias)

        if self.random_background:
            texture_type = "seen" if not self.eval_mode else "unseen"
            directory = Path("assets/background_texture") / texture_type
            count = len([p for p in directory.iterdir() if p.is_file()])
            # Prefer the Kitchen Large reference textures when available.
            wall_id = 43 if count > 43 else int(np.random.randint(0, count))
            table_id = 141 if count > 141 else int(np.random.randint(0, count))
            floor_id = 38 if count > 38 else int(np.random.randint(0, count))
            self.wall_texture = f"{texture_type}/{wall_id}"
            self.table_texture = f"{texture_type}/{table_id}"
            self.floor_texture = f"{texture_type}/{floor_id}"
            if np.random.rand() <= self.clean_background_rate:
                self.wall_texture = None
            if np.random.rand() <= self.clean_background_rate:
                self.table_texture = None
            if np.random.rand() <= self.clean_background_rate:
                self.floor_texture = None
        else:
            self.wall_texture = self.table_texture = self.floor_texture = None

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

        counter_length, counter_width = self.kitchenl_info["table_area"]
        counter_thickness = 0.05
        tabletop_pose = sapien.Pose([0.0, 0.0, -counter_thickness / 2])
        tabletop_half = [counter_length / 2, counter_width / 2, counter_thickness / 2]

        front_recess = 0.12
        base_height = max(0.0, table_height - counter_thickness)
        base_depth = max(0.0, counter_width - front_recess)
        base_half = [counter_length / 2, base_depth / 2, base_height / 2]
        base_pose = sapien.Pose(
            [0.0, front_recess / 2, -(counter_thickness / 2 + base_height / 2)]
        )

        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        builder.add_box_collision(
            pose=tabletop_pose,
            half_size=tabletop_half,
            material=self.scene.default_physical_material,
        )
        if base_height > 1e-6 and base_depth > 1e-6:
            builder.add_box_collision(
                pose=base_pose,
                half_size=base_half,
                material=self.scene.default_physical_material,
            )
        if self.table_texture is not None:
            texturepath = f"./assets/background_texture/{self.table_texture}.png"
            texture2d = sapien.render.RenderTexture2D(texturepath)
            material = sapien.render.RenderMaterial()
            material.set_base_color_texture(texture2d)
            material.base_color = [1, 1, 1, 1]
            material.metallic = 0.1
            material.roughness = 0.3
            builder.add_box_visual(pose=tabletop_pose, half_size=tabletop_half, material=material)
            base_material = sapien.render.RenderMaterial(base_color=[0.55, 0.47, 0.38, 1])
            base_material.metallic = 0.0
            base_material.roughness = 0.8
            if base_height > 1e-6 and base_depth > 1e-6:
                builder.add_box_visual(pose=base_pose, half_size=base_half, material=base_material)
        else:
            builder.add_box_visual(pose=tabletop_pose, half_size=tabletop_half, material=(1, 1, 1))
            if base_height > 1e-6 and base_depth > 1e-6:
                builder.add_box_visual(pose=base_pose, half_size=base_half, material=(0.55, 0.47, 0.38))

        builder.set_initial_pose(sapien.Pose(p=[table_xy_bias[0], table_xy_bias[1], table_height]))
        self.table = builder.build(name="table")

        self.kitchenl_info["table_lims"] = [
            -counter_length / 2,
            -counter_width / 2,
            counter_length / 2,
            counter_width / 2,
        ]

        self._load_fridge_on_table(table_height, table_xy_bias)
        if not getattr(self, "skip_microwave", False):
            self._load_microwave_on_table(table_height, table_xy_bias)
        if not getattr(self, "skip_scene_basket", False):
            self._load_basket_on_table(table_height, table_xy_bias)

    # ------------------------------------------------------------------
    # Fixtures
    # ------------------------------------------------------------------
    def _euler_deg_to_quat_wxyz(self, roll_pitch_yaw_deg):
        ax, ay, az = [math.radians(float(v)) for v in roll_pitch_yaw_deg]
        qx, qy, qz, qw = t3d.euler.euler2quat(ax, ay, az)
        return [float(qw), float(qx), float(qy), float(qz)]

    def _extract_intrinsic_scale(self, model_data: dict) -> float:
        base = model_data.get("scale", 1.0)
        if isinstance(base, (list, tuple)) and len(base) > 0:
            return float(base[0])
        return float(base)

    def _get_asset_model_scale_create_actor(self, modelname: str, model_id: int = 0) -> float:
        json_file = Path("assets/objects") / modelname / f"model_data{model_id}.json"
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                return self._extract_intrinsic_scale(json.load(f))
        except Exception:
            return 1.0

    def apply_srdf_collisions(self, articulation, srdf_path: Path) -> None:
        """Disable link-pair collisions listed in an SRDF (PhysX group bits)."""
        if srdf_path is None or not Path(srdf_path).exists():
            return
        try:
            root = ET.parse(str(srdf_path)).getroot()
        except Exception:
            return
        parsed_pairs = []
        for tag in root.findall(".//disable_collisions"):
            l1, l2 = tag.get("link1"), tag.get("link2")
            if l1 and l2:
                parsed_pairs.append((l1, l2))
        if not parsed_pairs:
            return

        link_map = {link.get_name(): link for link in articulation.get_links()}
        involved = {n for pair in parsed_pairs for n in pair}
        link_shapes = {}
        for name in involved:
            link = link_map.get(name)
            if link is None:
                continue
            try:
                shapes = link.get_collision_shapes()
            except Exception:
                shapes = None
            if shapes:
                link_shapes[name] = list(shapes)
        if not link_shapes:
            return

        srdf_collision_id = 0xBEEF

        def _apply_group_bit(shape, bit: int) -> None:
            groups = shape.get_collision_groups()
            if groups is None or len(groups) != 4:
                return
            g0, g1, g2, g3 = groups
            g3_new = (int(g3) & 0xFFFF0000) | srdf_collision_id
            g2_new = int(g2) | bit
            shape.set_collision_groups([int(g0), int(g1), int(g2_new), int(g3_new)])

        for shapes in link_shapes.values():
            for shape in shapes:
                try:
                    groups = shape.get_collision_groups()
                    if groups is None or len(groups) != 4:
                        continue
                    g0, g1, g2, g3 = groups
                    g3_new = (int(g3) & 0xFFFF0000) | srdf_collision_id
                    shape.set_collision_groups([int(g0), int(g1), int(g2), int(g3_new)])
                except Exception:
                    continue

        for bit_i, (l1, l2) in enumerate(parsed_pairs):
            bit = 1 << (bit_i % 31)
            for name in (l1, l2):
                for shape in link_shapes.get(name, []):
                    try:
                        _apply_group_bit(shape, bit)
                    except Exception:
                        continue

    def _create_objects_bench_cabinet(
        self,
        asset_dir_name: str,
        pose: sapien.Pose,
        fix_root_link: bool = True,
        extra_scale: float = 1.0,
    ) -> ArticulationActor | None:
        """Load an articulated fixture from ``assets/objects_bench/<name>``."""
        modeldir = Path("assets/objects_bench") / asset_dir_name
        urdf_path = modeldir / "mobility.urdf"
        if not urdf_path.exists():
            sapien_urdf_dir = modeldir / "sapien_urdf"
            if sapien_urdf_dir.is_dir():
                urdfs = list(sapien_urdf_dir.glob("*.urdf"))
                if urdfs:
                    urdf_path = urdfs[0]
        if not urdf_path.exists():
            print(f"[KitchenL] cabinet URDF not found under {modeldir}")
            return None

        srdf_path = None
        for cand in (urdf_path.with_suffix(".srdf"), urdf_path.parent / f"{urdf_path.stem}.srdf"):
            if cand.exists():
                srdf_path = cand
                break

        json_file = modeldir / "model_data.json"
        if json_file.exists():
            with open(json_file, "r", encoding="utf-8") as f:
                model_data = json.load(f)
            raw_scale = model_data.get("scale", 1.0)
            raw_scale_vec = np.array(raw_scale, dtype=float).reshape(-1)
            if raw_scale_vec.size == 1:
                raw_scale_vec = np.repeat(raw_scale_vec[0], 3)
            elif raw_scale_vec.size >= 3:
                raw_scale_vec = raw_scale_vec[:3]
            else:
                raw_scale_vec = np.ones(3, dtype=float)
            scaled = raw_scale_vec * float(extra_scale)
            model_data = dict(model_data)
            model_data["scale"] = scaled.tolist()
            scale_scalar = float(scaled[0])
            trans_mat = np.array(model_data.get("transform_matrix", np.eye(4)))
        else:
            model_data = {"scale": [1.0, 1.0, 1.0]}
            scale_scalar = float(extra_scale)
            trans_mat = np.eye(4)

        loader = self.scene.create_urdf_loader()
        loader.scale = float(scale_scalar)
        loader.fix_root_link = fix_root_link
        loader.load_multiple_collisions_from_file = True
        try:
            articulation = loader.load_multiple(str(urdf_path))[0][0]
            if srdf_path is not None:
                self.apply_srdf_collisions(articulation, srdf_path)
        except Exception as e:
            print(f"[KitchenL] failed to load {urdf_path}: {e}")
            return None

        pose_mat = pose.to_transformation_matrix()
        pose_with_offset = sapien.Pose(
            p=pose_mat[:3, 3] + trans_mat[:3, 3],
            q=t3d.quaternions.mat2quat(trans_mat[:3, :3] @ pose_mat[:3, :3]),
        )
        articulation.set_pose(pose_with_offset)
        init_qpos = model_data.get("init_qpos")
        if init_qpos is not None and len(init_qpos) > 0:
            articulation.set_qpos(np.array(init_qpos, dtype=float))
        for joint in articulation.get_joints():
            joint.set_drive_properties(damping=10.0, stiffness=0)
        articulation.set_name(asset_dir_name)
        return ArticulationActor(articulation, model_data)

    @staticmethod
    def _fridge_closed_door_T_base() -> np.ndarray:
        """Closed-door mesh frame relative to fridge ``base_link`` (from URDF).

        The door GLB is *not* in the base frame — URDF chains
        ``base → jointframe → link_0 → visual origin``.
        """
        def _xyz_rpy(xyz, rpy):
            T = np.eye(4, dtype=float)
            T[:3, :3] = t3d.euler.euler2mat(rpy[0], rpy[1], rpy[2], axes="sxyz")
            T[:3, 3] = np.asarray(xyz, dtype=float)
            return T

        # Matches assets/.../sapien_urdf/hivvdf.urdf at q=0.
        T_bj = _xyz_rpy(
            [0.39007673, 0.020861734, -0.31230012],
            [-1.5707963, 1.1920928e-07, 1.5707963],
        )
        T_jl = _xyz_rpy([0.0, 0.0, 0.0], [1.5707966, 1.5707962, 0.0])
        T_lv = _xyz_rpy([-0.39007673, -0.020861734, 0.31230012], [0.0, 0.0, 0.0])
        return T_bj @ T_jl @ T_lv

    def _load_fridge_on_table(self, table_height: float, table_xy_bias):
        """Fridge with articulated door (URDF), door cracked open for visibility."""
        y_front = table_xy_bias[1] + 0.30
        x_fridge = table_xy_bias[0] + 0.40
        # Match robotwin_bench Kitchen Large placement (origin ≈ mesh center).
        z_fridge = table_height + 0.24
        quat = self._euler_deg_to_quat_wxyz(self.fridge_left_rot)
        scale = float(self.fridge_left_scale)
        pose_fridge = sapien.Pose([x_fridge, y_front, z_fridge], quat)

        fridge = self._create_objects_bench_cabinet(
            asset_dir_name="124_fridge_hivvdf",
            pose=pose_fridge,
            fix_root_link=True,
            extra_scale=scale,
        )
        if fridge is not None:
            try:
                fridge.set_name("fridge_left")
            except Exception:
                pass
            # Keep door closed (facing the robot after yaw −90°).
            try:
                qlim = np.array(fridge.get_qlimits(), dtype=float)
                qpos = np.zeros(len(qlim), dtype=float)
                try:
                    fridge.set_properties(damping=100.0, stiffness=0.0)
                except Exception:
                    pass
                fridge.set_qpos(qpos)
                try:
                    art = fridge.actor if hasattr(fridge, "actor") else fridge
                    for j in art.get_active_joints():
                        j.set_drive_target(0.0)
                        j.set_drive_properties(damping=80.0, stiffness=400.0)
                except Exception:
                    pass
                fridge.set_qpos(qpos)
            except Exception as e:
                print(f"[KitchenL] fridge door close skip: {e}", flush=True)
            self.fridge_left = fridge
            self.add_prohibit_area(self.fridge_left, padding=0.05)
            try:
                from script.bench_script.scene_utils import change_object_texture

                tex = Path("assets/objects_bench/backgrounds/fridge/3.png")
                if not tex.exists():
                    tex = Path("assets/objects_bench/backgrounds/fridge/0.png")
                if tex.exists():
                    change_object_texture(
                        self, self.fridge_left, tex.resolve(), "fridge", refresh_render=True
                    )
            except Exception as e:
                print(f"[KitchenL] fridge texture skip: {e}", flush=True)
        else:
            self._load_fridge_static_body_and_door(
                x_fridge, y_front, z_fridge, quat, scale
            )

        bounds_min = np.array([-0.243, -0.479, -0.318], dtype=float)
        bounds_max = np.array([0.443, 0.433, 0.318], dtype=float)
        R = t3d.quaternions.quat2mat(quat)
        corners = []
        for sx in (bounds_min[0], bounds_max[0]):
            for sy in (bounds_min[1], bounds_max[1]):
                for sz in (bounds_min[2], bounds_max[2]):
                    corners.append(R @ (np.array([sx, sy, sz], dtype=float) * scale))
        corners = np.asarray(corners, dtype=float)
        self.fridge_footprint = {
            "x": float(x_fridge),
            "y": float(y_front),
            "z": float(z_fridge),
            "xy_min": np.array(
                [x_fridge + corners[:, 0].min(), y_front + corners[:, 1].min()],
                dtype=float,
            ),
            "xy_max": np.array(
                [x_fridge + corners[:, 0].max(), y_front + corners[:, 1].max()],
                dtype=float,
            ),
        }

    def _load_fridge_static_body_and_door(self, x, y, z, quat, scale):
        """Static fridge: body + door visuals with correct closed-door transform."""
        model_dir = Path("assets/objects_bench/124_fridge_hivvdf/blender_public/links")
        body_visual = model_dir / "base_link.glb"
        door_visual = model_dir / "link_0.glb"
        body_pose = sapien.Pose(p=[x, y, z], q=quat)
        scale_vec = [float(scale), float(scale), float(scale)]

        # Body
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        body_half = [0.18 * scale, 0.22 * scale, 0.18 * scale]
        builder.add_box_collision(
            pose=sapien.Pose(p=[0.0, 0.0, 0.0]),
            half_size=body_half,
            material=self.scene.default_physical_material,
        )
        if body_visual.exists():
            builder.add_visual_from_file(filename=str(body_visual), scale=scale_vec)
        builder.set_initial_pose(body_pose)
        body = builder.build(name="fridge_left")
        self.fridge_left = Actor(
            body,
            {"center": [0, 0, 0], "extents": [0.36 * scale] * 3, "scale": scale},
        )

        if door_visual.exists():
            # Door mesh frame ← base frame (closed), then into world.
            T_base = body_pose.to_transformation_matrix()
            T_door_base = self._fridge_closed_door_T_base().copy()
            T_door_base[:3, 3] *= float(scale)
            T_door_closed = T_base @ T_door_base

            # Crack the door ~55° about the URDF hinge axis so the panel reads
            # clearly (closed door is flush and looks like a solid block).
            T_bj = np.eye(4, dtype=float)
            T_bj[:3, :3] = t3d.euler.euler2mat(
                -1.5707963, 1.1920928e-07, 1.5707963, axes="sxyz"
            )
            T_bj[:3, 3] = np.array(
                [0.39007673, 0.020861734, -0.31230012], dtype=float
            ) * float(scale)
            T_jf = T_base @ T_bj
            hinge_world = T_jf[:3, 3]
            axis_world = T_jf[:3, :3] @ np.array([1.0, 0.0, 0.0], dtype=float)
            axis_world = axis_world / max(float(np.linalg.norm(axis_world)), 1e-8)
            # Negative angle swings the door out toward the robot (−Y).
            R_open = t3d.axangles.axangle2mat(axis_world, np.deg2rad(-55.0))
            T_hinge = np.eye(4, dtype=float)
            T_hinge[:3, 3] = hinge_world
            T_hinge_inv = np.eye(4, dtype=float)
            T_hinge_inv[:3, 3] = -hinge_world
            T_rot = np.eye(4, dtype=float)
            T_rot[:3, :3] = R_open
            T_door = T_hinge @ T_rot @ T_hinge_inv @ T_door_closed
            door_pose = sapien.Pose(
                T_door[:3, 3], t3d.quaternions.mat2quat(T_door[:3, :3])
            )

            d_builder = self.scene.create_actor_builder()
            d_builder.set_physx_body_type("static")
            d_builder.add_box_collision(
                pose=sapien.Pose(),
                half_size=[0.02 * scale, 0.22 * scale, 0.16 * scale],
                material=self.scene.default_physical_material,
            )
            d_builder.add_visual_from_file(filename=str(door_visual), scale=scale_vec)
            d_builder.set_initial_pose(door_pose)
            door = d_builder.build(name="fridge_left_door")
            self.fridge_left_door = Actor(
                door,
                {
                    "center": [0, 0, 0],
                    "extents": [0.04 * scale, 0.44 * scale, 0.32 * scale],
                    "scale": scale,
                },
            )
        self.add_prohibit_area(self.fridge_left, padding=0.05)
        # Match robotwin fridge look (textured body + door).
        try:
            from script.bench_script.scene_utils import change_object_texture

            tex = Path("assets/objects_bench/backgrounds/fridge/3.png")
            if not tex.exists():
                tex = Path("assets/objects_bench/backgrounds/fridge/0.png")
            if tex.exists():
                change_object_texture(
                    self, self.fridge_left, tex.resolve(), "fridge", refresh_render=False
                )
                if getattr(self, "fridge_left_door", None) is not None:
                    change_object_texture(
                        self,
                        self.fridge_left_door,
                        tex.resolve(),
                        "fridge",
                        refresh_render=True,
                    )
        except Exception as e:
            print(f"[KitchenL] fridge texture skip: {e}", flush=True)

    def _load_microwave_on_table(self, table_height: float, table_xy_bias):
        """Static microwave visual + unit-scale box collision (planner-safe)."""
        x_mw, y_front = self._get_scene_obj_locations("microwave")
        x_mw += table_xy_bias[0]
        y_front += table_xy_bias[1]

        visual_path = "assets/objects/044_microwave/visual/base0.glb"
        # Kitchen Large used URDF scale ≈ intrinsic * 1.4; KitchenS uses 0.15 on the GLB.
        # Match the Kitchen Large footprint roughly with a GLB scale near 0.18.
        scale = 0.18
        bounds_min = np.array([-0.761475, -0.495360, -0.567261], dtype=float)
        bounds_max = np.array([0.795292, 0.435315, 0.630773], dtype=float)
        center = 0.5 * (bounds_min + bounds_max) * scale
        half_size = 0.5 * (bounds_max - bounds_min) * scale

        mw_q = self._euler_deg_to_quat_wxyz(self.microwave_left_rot)
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
        builder.add_visual_from_file(filename=visual_path, scale=[scale, scale, scale])
        z = table_height - float(corners[:, 2].min())
        builder.set_initial_pose(sapien.Pose(p=[x_mw, y_front, z], q=mw_q))
        entity = builder.build(name="microwave_center")
        self.microwave_left = Actor(
            entity,
            {
                "center": center.tolist(),
                "extents": (bounds_max - bounds_min).tolist(),
                "scale": scale,
            },
        )
        self.add_prohibit_area(self.microwave_left, padding=0.05)

    def _load_basket_on_table(self, table_height: float, table_xy_bias):
        jx = float(np.random.uniform(*self.basket_right_position_jitter_x))
        jy = float(np.random.uniform(*self.basket_right_position_jitter_y))
        x_b, y_b, z_b = self._get_scene_obj_locations("basket")
        if self.jitter_basket:
            x_b += jx
            y_b += jy
        z_b = table_height + 0.02 if float(z_b) == 0.0 else float(z_b)
        pose = sapien.Pose(
            [x_b + table_xy_bias[0], y_b + table_xy_bias[1], z_b],
            self._euler_deg_to_quat_wxyz(self.basket_right_rot),
        )
        # Author scale is absolute via model_data; use scale_mult for Kitchen Large 1.4×.
        self.basket_right = create_actor(
            scene=self,
            pose=pose,
            modelname=str(self.basket_right_modelname),
            model_id=int(self.basket_right_model_id),
            is_static=True,
            convex=False,
            scale_mult=float(self.basket_right_scale),
        )
        if self.basket_right is not None:
            self.basket_right.set_name("basket_right")
            self.add_prohibit_area(self.basket_right, padding=0.05)

    def _load_cabinet_on_table(self, table_height: float, table_xy_bias):
        """Static wall-cabinet visual + box collision (planner-safe)."""
        x_center = table_xy_bias[0]
        y_center = table_xy_bias[1] + 0.12
        z_center = table_height + 0.55
        ax, ay, az = math.radians(90), math.radians(180), math.radians(-90)
        qx, qy, qz, qw = t3d.euler.euler2quat(ax, ay, az)
        quat = [qw, qx, qy, qz]
        model_dir = Path("assets/objects_bench/125_cabinet_tynnnw")
        visual = model_dir / "blender_public/links/base_link.glb"
        if not visual.exists():
            cands = sorted(model_dir.rglob("*.glb"))
            visual = cands[0] if cands else None
        scale = float(self.cabinet_scale)
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        builder.add_box_collision(
            pose=sapien.Pose(p=[0, 0, 0]),
            half_size=[0.28 * scale, 0.14 * scale, 0.22 * scale],
            material=self.scene.default_physical_material,
        )
        if visual is not None:
            builder.add_visual_from_file(filename=str(visual), scale=[scale, scale, scale])
        builder.set_initial_pose(sapien.Pose(p=[x_center, y_center, z_center], q=quat))
        entity = builder.build(name="cabinet")
        self.cabinet = Actor(
            entity,
            {"center": [0, 0, 0], "extents": [0.56 * scale, 0.28 * scale, 0.44 * scale], "scale": scale},
        )
        self.add_prohibit_area(self.cabinet, padding=0.02)

    def _add_cabinet_wall_filler(self):
        if getattr(self, "cabinet", None) is None:
            return
        cab_pose = np.array(self.cabinet.get_pose().p, dtype=float)
        scale_ratio = float(self.cabinet_scale) / 0.5
        self.cabinet_wall_filler = create_box(
            self.scene,
            sapien.Pose(
                p=[
                    float(cab_pose[0]),
                    float(cab_pose[1] + 0.16 * scale_ratio),
                    float(cab_pose[2]),
                ]
            ),
            half_size=[0.30 * scale_ratio, 0.11 * scale_ratio, 0.22 * scale_ratio],
            color=(0.32, 0.32, 0.32),
            name="cabinet_wall_filler",
            is_static=True,
        )
