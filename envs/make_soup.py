"""Make soup: pour chopping-board vegetables into a pot of water, then turn on the stove.

KitchenS scene with a cooking range. A chopping board holds colored vegetable cubes
(orange, green, purple) and a small red tomato. A pot of water sits on a burner.
The robot lifts the board, carries it roughly level over the pot, tips carefully so
the pieces fall in under physics (tilting too early / too far drops them onto the
table), then turns the stove on. Success requires every piece in the pot, none on
the table, and the stove on.
"""
from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import sapien
import sapien.physx
import sapien.render
import transforms3d as t3d
from transforms3d.euler import euler2quat
from transforms3d.quaternions import qmult

from ._kitchens_base_task import KitchenS_base_task
from ._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
from .utils import *
from .utils.actor_utils import Actor
from .utils.create_actor import create_box, create_sphere, create_visual_box, UnStableError


class make_soup(KitchenS_base_task):
    """Pour board vegetables into a pot of water, then turn the stove on."""

    EE_TO_TCP: ClassVar[float] = 0.12
    KNOB_CONTACT_RADIUS_DEFAULT: ClassVar[float] = 0.06
    KNOB_APPROACH_PATH: ClassVar[tuple] = (
        (-0.13, -0.33, 0.06),
        (-0.08, -0.33, 0.00),
        (0.00, -0.25, 0.00),
    )
    KNOB_GRASP_STANDOFF: ClassVar[float] = 0.015
    ACTIVE_BURNER: ClassVar[str] = "left_front"
    # Stove at the back of the counter: the front knob then sits inside the
    # right arm's comfortable reach, so the expert can grasp and twist it.
    RANGE_REL_XY: ClassVar[tuple[float, float]] = (0.18, 0.22)
    # boil_milk appliance scale.
    RANGE_SCALE_MULT: ClassVar[float] = 1.05
    MICROWAVE_SCALE_MULT: ClassVar[float] = 1.3
    # Original thick-walled soup pot, sized to the boil_milk saucepan
    # (same height, matching inner mouth once the 5 mm wall is accounted for).
    POT_RADIUS: ClassVar[float] = 0.058
    POT_HEIGHT: ClassVar[float] = 0.0735
    POT_WALL: ClassVar[float] = 0.005
    GLASS_SCALE: ClassVar[float] = 0.585  # prior 0.45 × 1.3

    BOARD_HALF: ClassVar[tuple[float, float, float]] = (0.095, 0.065, 0.010)
    # Grasping block on the robot-facing (−Y) end of the board.
    HANDLE_HALF: ClassVar[tuple[float, float, float]] = (0.022, 0.024, 0.022)
    BOARD_COLOR: ClassVar[tuple[float, float, float]] = (0.55, 0.38, 0.22)
    HANDLE_COLOR: ClassVar[tuple[float, float, float]] = (0.72, 0.55, 0.28)
    CUBE_HALF: ClassVar[float] = 0.012
    TOMATO_RADIUS: ClassVar[float] = 0.015
    GRASP_TCP_TOL: ClassVar[float] = 0.045
    VEG_COLORS: ClassVar[dict[str, tuple[float, float, float]]] = {
        "orange": (0.95, 0.45, 0.08),
        "green": (0.20, 0.65, 0.22),
        "purple": (0.55, 0.18, 0.70),
    }
    TOMATO_COLOR: ClassVar[tuple[float, float, float]] = (0.90, 0.10, 0.08)
    DECOR_QPOS: ClassVar[list[float]] = [0.70710678, 0.70710678, 0.0, 0.0]

    # Soft-friction model during an armed pour: pieces stay on the board until
    # the board's up-axis drops below this (≈35° tip). Past that they free-fall.
    TILT_HOLD_DOT: ClassVar[float] = 0.82
    # Accidental spill during carry (top-down grasp wobble is ~0.87–0.90): only
    # free-fall if tipped nearly sideways (~60°).
    TILT_SPILL_DOT: ClassVar[float] = 0.50
    # Wrist roll for the pour: 40° is well past the produce's friction angle.
    POUR_TIP_RAD: ClassVar[float] = float(np.deg2rad(40.0))
    WATER_LEVEL_DEFAULT: ClassVar[float] = 0.45
    BOARD_IGNORE_BIT: ClassVar[int] = 1 << 21
    BOARD_IGNORE_ID: ClassVar[int] = 0xB0A5

    def setup_demo(self, **kwags: Any) -> None:
        self._cfg = dict(kwags.get("task_args", {}).get("make_soup", {}))
        if kwags.get("scene_id") is None:
            kwags["scene_id"] = int(self._cfg.get("scene_id", 0))
        self.replace_sink_with_range = True
        self.omit_sink = True
        self.clear_sink_and_range = False
        # Same stove / microwave scale as boil_milk.
        self.range_scale_mult = float(
            self._cfg.get("range_scale_mult", self.RANGE_SCALE_MULT)
        )
        self.microwave_scale_mult = float(
            self._cfg.get("microwave_scale_mult", self.MICROWAVE_SCALE_MULT)
        )
        rel = self._cfg.get("range_xy", list(self.RANGE_REL_XY))
        self.range_position_override = [float(rel[0]), float(rel[1])]
        if "table_xy_bias" not in kwags and "table_xy_bias" in self._cfg:
            kwags["table_xy_bias"] = list(self._cfg["table_xy_bias"])

        # Guard early _update_kinematic_tasks during camera init.
        self._loaded = False
        self.stove_on = False
        self.turned_on_once = False
        self._liquid_entity = None
        self._burner_shapes: list[Any] = []
        self._ignore_knob = False
        self._expert_holding_knob = False
        self._prev_knob_pressed = False
        self._board_welded = False
        self._board_weld_offset = None
        self._veg_released = False
        self._veg_fallen = False
        self._pour_armed = False
        self._force_veg_hold = False
        self._veg_offsets: list[sapien.Pose] = []
        self.veggies: list[Any] = []
        self._veg_rigids: list[Any] = []
        self._board_rigid = None
        self.board = None
        self.pot = None
        self._ring_parts: list[Any] = []
        self._ring_shapes: list[Any] = []
        self._disc_parts: list[Any] = []
        self._disc_shapes: list[Any] = []
        self._handle_local = np.zeros(3)

        super().setup_demo(**kwags)
        self._configure_head_camera()

    def _configure_head_camera(self) -> None:
        """Closer head framing so the boil_milk-scale stove/pot read at true size."""
        cams = getattr(self, "cameras", None)
        if cams is None:
            return
        names = list(getattr(cams, "static_camera_name", []) or [])
        clist = list(getattr(cams, "static_camera_list", []) or [])
        if "head_camera" not in names:
            return
        camera = clist[names.index("head_camera")]
        rx, ry = getattr(self, "range_xy", (0.18, 0.22))
        # High and back far enough that the reaching arm never fills the frame
        # (the knob twist has to stay visible), only slightly tighter than the
        # stock KitchenS head view.
        cam_pos = np.array([float(rx) * 0.55, -0.98, 1.74], dtype=float)
        look_at = np.array([float(rx) * 0.70, float(ry) * 0.10, 0.83], dtype=float)
        forward = look_at - cam_pos
        forward /= np.linalg.norm(forward)
        left = np.cross(np.array([0.0, 0.0, 1.0], dtype=float), forward)
        if float(np.linalg.norm(left)) < 1e-6:
            left = np.array([-1.0, 0.0, 0.0], dtype=float)
        left /= np.linalg.norm(left)
        up = np.cross(forward, left)
        m = np.eye(4)
        m[:3, :3] = np.stack([forward, left, up], axis=1)
        m[:3, 3] = cam_pos
        camera.entity.set_pose(sapien.Pose(m))
        try:
            camera.set_fovy(float(np.deg2rad(52)))
        except Exception:
            try:
                camera.fovy = float(np.deg2rad(52))
            except Exception:
                pass

    # ---------------------------------------------------------------- actors
    def load_actors(self) -> None:
        cfg = self._cfg
        if not hasattr(self, "burner_positions"):
            raise UnStableError("cooking range missing — KitchenS base did not load a range")

        self.knob_contact_radius = float(
            cfg.get("knob_contact_radius", self.KNOB_CONTACT_RADIUS_DEFAULT)
        )
        self.tilt_hold_dot = float(cfg.get("tilt_hold_dot", self.TILT_HOLD_DOT))
        self.tilt_spill_dot = float(cfg.get("tilt_spill_dot", self.TILT_SPILL_DOT))
        self.pour_tip_rad = float(cfg.get("pour_tip_rad", self.POUR_TIP_RAD))
        self.water_level = float(cfg.get("water_level", self.WATER_LEVEL_DEFAULT))
        self.board_half = list(cfg.get("board_half", list(self.BOARD_HALF)))
        self.handle_half = list(cfg.get("handle_half", list(self.HANDLE_HALF)))
        self.cube_half = float(cfg.get("cube_half", self.CUBE_HALF))
        self.tomato_radius = float(cfg.get("tomato_radius", self.TOMATO_RADIUS))
        self.grasp_tcp_tol = float(cfg.get("grasp_tcp_tol", self.GRASP_TCP_TOL))

        self.stove_on = False
        self.turned_on_once = False
        self._liquid_entity = None
        self._ignore_knob = False
        self._expert_holding_knob = False
        self._prev_knob_pressed = False
        self._board_welded = False
        self._board_weld_offset = None
        self._veg_released = False
        self._veg_fallen = False
        self._pour_armed = False
        self._force_veg_hold = False
        self._veg_offsets = []
        self.veggies = []
        self._veg_rigids = []
        self._board_rigid = None
        self._ring_parts = []
        self._ring_shapes = []
        self._disc_parts = []
        self._disc_shapes = []

        bz = 0.74 + self.table_z_bias
        self.table_top = bz

        burner_name = str(cfg.get("burner", self.ACTIVE_BURNER)).strip().lower()
        if burner_name not in self.burner_positions:
            raise ValueError(
                f"make_soup.burner must be one of {list(self.burner_positions)}, got {burner_name!r}"
            )
        bx, by = self.burner_positions[burner_name]
        self.burner_name = burner_name
        self.burner_xy = (float(bx), float(by))

        # Keep the stock burner disc under the pot; fire ring sits around its base.
        self._burner_shapes = []
        if getattr(self, "active_burner", None) is not None:
            home = sapien.Pose(
                p=[float(bx), float(by), float(self.range_top_z) + 0.0015]
            )
            self._burner_home_pose = home
            try:
                self.active_burner.set_pose(home)
            except Exception:
                pass
            for c in self.active_burner.get_components():
                if isinstance(c, sapien.render.RenderBodyComponent):
                    self._burner_shapes = list(c.render_shapes)

        self._spawn_pot(float(bx), float(by), cfg)
        pot_r = float(self.pot_radius)
        self._build_stove_fire_ring(
            float(bx),
            float(by),
            float(self.range_top_z) + 0.0015,
            float(pot_r + 0.009),
            n=28,
            half_size=[0.007, 0.0035, 0.002],
        )
        self._rebuild_water(force=True)
        self._set_stove_fire(False)

        # Board on the right half of the apron for an easy right-arm grasp.
        board_x = float(cfg.get("board_x", 0.14))
        board_y = float(cfg.get("board_y", -0.10))
        self.board_xy = (board_x, board_y)
        self.board = self._spawn_board(board_x, board_y, bz)
        self.add_prohibit_area(self.board, padding=0.04)

        self._spawn_vegetables(board_x, board_y, bz + 2.0 * self.board_half[2])
        self._spawn_decor(bz)
        self.arm = ArmTag("right" if self.knob_xy[0] >= 0 else "left")
        self._loaded = True
        print(
            f"[make_soup] arm={self.arm} board={self.board_xy} pot={self.pot_xy} "
            f"range_scale={self.range_scale_mult} pot_r={self.pot_radius:.3f} "
            f"n_veg={len(self.veggies)}"
        )

    def _spawn_pot(self, cx: float, cy: float, cfg: dict[str, Any]) -> None:
        """Original soup pot: hollow cylinder with two short side handles."""
        pot_r = float(cfg.get("pot_radius", self.POT_RADIUS))
        pot_h = float(cfg.get("pot_height", self.POT_HEIGHT))
        wall = float(cfg.get("pot_wall", self.POT_WALL))
        pot_z0 = float(self.range_top_z)
        vertical_q = [0.70710678, 0.0, 0.70710678, 0.0]

        metal = sapien.render.RenderMaterial(base_color=[0.55, 0.55, 0.58, 1.0])
        metal.metallic = 0.7
        metal.roughness = 0.35

        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        builder.add_cylinder_collision(
            pose=sapien.Pose([0, 0, wall / 2], vertical_q),
            radius=pot_r,
            half_length=wall / 2,
            material=self.scene.default_physical_material,
        )
        builder.add_cylinder_visual(
            pose=sapien.Pose([0, 0, wall / 2], vertical_q),
            radius=pot_r,
            half_length=wall / 2,
            material=metal,
        )

        wall_segments = 20
        wall_radius = pot_r - wall / 2
        tangent_half = wall_radius * np.tan(np.pi / wall_segments) * 1.04
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
        self.pot = builder.build(name="soup_pot")

        # Two short lugs on the ±X sides, as in the original design.
        for sign, name in ((-1, "left"), (1, "right")):
            create_box(
                self.scene,
                sapien.Pose(p=[cx + sign * (pot_r + 0.015), cy, pot_z0 + pot_h * 0.65]),
                half_size=[0.012, 0.008, 0.006],
                color=(0.45, 0.45, 0.48),
                name=f"pot_handle_{name}",
                is_static=True,
            )

        self.pot_inner_radius = pot_r - 1.6 * wall
        self.pot_inner_height = pot_h - wall
        self.pot_bottom_z = pot_z0 + wall
        self.pot_rim_z = pot_z0 + pot_h
        self.pot_xy = (cx, cy)
        self.pot_radius = pot_r
        self.pot_height = pot_h

    def _spawn_board(self, x: float, y: float, table_z: float) -> Actor:
        """Wooden board with a small grasping cube on the robot-facing (−Y) end."""
        hx, hy, hz = [float(v) for v in self.board_half]
        hhx, hhy, hhz = [float(v) for v in self.handle_half]
        # Handle protrudes from the −Y edge so the gripper can pinch it cleanly.
        # Its underside is flush with the board's: any part hanging below would
        # spawn inside the counter and PhysX would launch the board off it.
        handle_local = np.array([0.0, -(hy + hhy), hhz - hz], dtype=float)
        self._handle_local = handle_local.copy()

        wood = sapien.render.RenderMaterial(base_color=[*self.BOARD_COLOR, 1.0])
        wood.roughness = 0.75
        wood.metallic = 0.0
        handle_mat = sapien.render.RenderMaterial(base_color=[*self.HANDLE_COLOR, 1.0])
        handle_mat.roughness = 0.7

        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("dynamic")
        # Grippy enough to stay put on the counter; produce still slides once
        # tipped because the pieces themselves are near-frictionless.
        board_mat = self.scene.create_physical_material(0.45, 0.35, 0.0)
        builder.add_box_collision(
            pose=sapien.Pose([0, 0, 0]),
            half_size=[hx, hy, hz],
            material=board_mat,
        )
        builder.add_box_visual(
            pose=sapien.Pose([0, 0, 0]),
            half_size=[hx, hy, hz],
            material=wood,
        )
        builder.add_box_collision(
            pose=sapien.Pose(handle_local.tolist()),
            half_size=[hhx, hhy, hhz],
            material=board_mat,
        )
        builder.add_box_visual(
            pose=sapien.Pose(handle_local.tolist()),
            half_size=[hhx, hhy, hhz],
            material=handle_mat,
        )
        builder.set_initial_pose(sapien.Pose([x, y, table_z + hz], [1, 0, 0, 0]))
        entity = builder.build(name="chopping_board")

        # Grasp frames on the handle top (meters; scale=1).
        top_z = float(handle_local[2] + hhz)
        hy_h = float(handle_local[1])
        data = {
            "center": [0, 0, 0],
            "extents": [hx * 2, (hy + 2 * hhy) * 2, max(hz, hhz) * 2],
            "scale": [1.0, 1.0, 1.0],
            "contact_points_pose": [
                # 4 yaw variants of top-down grasp centered on the handle top.
                [[0, 0, 1, 0.0], [1, 0, 0, hy_h], [0, 1, 0, top_z], [0, 0, 0, 1]],
                [[1, 0, 0, 0.0], [0, 0, -1, hy_h], [0, 1, 0, top_z], [0, 0, 0, 1]],
                [[-1, 0, 0, 0.0], [0, 0, 1, hy_h], [0, 1, 0, top_z], [0, 0, 0, 1]],
                [[0, 0, -1, 0.0], [-1, 0, 0, hy_h], [0, 1, 0, top_z], [0, 0, 0, 1]],
            ],
            "contact_points_group": [[0, 1, 2, 3]],
            "contact_points_mask": [True],
            "functional_matrix": [],
            "transform_matrix": np.eye(4).tolist(),
        }

        board = Actor(entity, data, mass=0.35)
        self._board_rigid = None
        for c in board.actor.get_components():
            if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                try:
                    c.set_linear_damping(6.0)
                    c.set_angular_damping(10.0)
                except Exception:
                    pass
                self._board_rigid = c
                break
        return board

    def _spawn_decor(self, bz: float) -> None:
        """Left-side static props: plate with bread, wine bottle, wine glass."""
        cfg = self._cfg
        plate_x = float(cfg.get("plate_x", -0.18))
        plate_y = float(cfg.get("plate_y", -0.08))
        plate_scale = float(cfg.get("plate_scale", 0.55))
        self.decor_plate = create_actor(
            self,
            pose=sapien.Pose([plate_x, plate_y, bz], list(self.DECOR_QPOS)),
            modelname="003_plate",
            model_id=0,
            convex=True,
            is_static=True,
            scale_mult=plate_scale,
        )
        self.decor_plate.set_name("003_plate")
        self.add_prohibit_area(self.decor_plate, padding=0.03)
        try:
            plate_top = float(self.decor_plate.get_functional_point(0)[2])
        except Exception:
            plate_top = bz + 0.02

        bread_scale = float(cfg.get("bread_scale", 1.0))
        self.decor_bread = create_actor(
            self,
            pose=sapien.Pose(
                [plate_x, plate_y, plate_top + 0.008], list(self.DECOR_QPOS)
            ),
            modelname="075_bread",
            model_id=int(np.random.choice([0, 1, 2])),
            convex=True,
            is_static=True,
            scale_mult=bread_scale,
        )
        self.decor_bread.set_name("075_bread")

        bottle_x = float(cfg.get("wine_x", -0.30))
        bottle_y = float(cfg.get("wine_y", -0.02))
        self.decor_wine = create_actor(
            self,
            pose=sapien.Pose([bottle_x, bottle_y, bz], list(self.DECOR_QPOS)),
            modelname="265_wine_bottle",
            model_id=0,
            convex=True,
            is_static=True,
            scale_mult=float(cfg.get("wine_scale", 1.0)),
        )
        self.decor_wine.set_name("265_wine_bottle")
        self.add_prohibit_area(self.decor_wine, padding=0.03)

        glass_x = float(cfg.get("glass_x", -0.22))
        glass_y = float(cfg.get("glass_y", -0.16))
        glass_scale = float(cfg.get("glass_scale", self.GLASS_SCALE))
        self.decor_glass = create_actor(
            self,
            pose=sapien.Pose([glass_x, glass_y, bz], list(self.DECOR_QPOS)),
            modelname="088_wineglass",
            model_id=0,
            convex=True,
            is_static=True,
            scale_mult=glass_scale,
        )
        self.decor_glass.set_name("088_wineglass")
        self.add_prohibit_area(self.decor_glass, padding=0.03)

    def _spawn_vegetables(self, bx: float, by: float, board_top: float) -> None:
        """Colored cubes + a small round tomato resting on the board."""
        # Two rows near the pour edge (−X). Spacing must exceed the largest
        # piece pair (tomato + cube = 27 mm) or PhysX depenetration launches
        # the board off the table during the settle phase.
        layout = [
            ("cube_orange", "cube", self.VEG_COLORS["orange"], (-0.065, -0.018)),
            ("cube_green", "cube", self.VEG_COLORS["green"], (-0.065, 0.018)),
            ("cube_purple", "cube", self.VEG_COLORS["purple"], (-0.030, -0.018)),
            ("tomato", "sphere", self.TOMATO_COLOR, (-0.030, 0.018)),
        ]
        jitter = 0.003
        for name, kind, color, (dx, dy) in layout:
            jx = float(np.random.uniform(-jitter, jitter))
            jy = float(np.random.uniform(-jitter, jitter))
            if kind == "cube":
                h = self.cube_half
                z = board_top + h + 0.001
                pose = sapien.Pose([bx + dx + jx, by + dy + jy, z], [1, 0, 0, 0])
                veg = create_box(
                    self,
                    pose=pose,
                    half_size=[h, h, h],
                    color=list(color),
                    name=name,
                    is_static=False,
                )
                veg.set_mass(0.012)
            else:
                r = self.tomato_radius
                z = board_top + r + 0.001
                pose = sapien.Pose([bx + dx + jx, by + dy + jy, z], [1, 0, 0, 0])
                entity = create_sphere(
                    self,
                    pose=pose,
                    radius=r,
                    color=list(color) + [1.0],
                    name=name,
                    is_static=False,
                )
                data = {
                    "center": [0, 0, 0],
                    "extents": [r, r, r],
                    "scale": [r, r, r],
                    "contact_points_pose": [],
                    "functional_matrix": [],
                    "transform_matrix": np.eye(4).tolist(),
                }
                veg = Actor(entity, data, mass=0.015)
            rigid = None
            for c in veg.actor.get_components():
                if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                    try:
                        c.set_linear_damping(0.25)
                        c.set_angular_damping(0.25)
                        for shape in c.get_collision_shapes():
                            m = shape.get_physical_material()
                            m.set_static_friction(0.10)
                            m.set_dynamic_friction(0.06)
                            m.set_restitution(0.0)
                    except Exception:
                        pass
                    rigid = c
                    break
            self.veggies.append(veg)
            self._veg_rigids.append(rigid)
            self.add_prohibit_area(veg, padding=0.015)

        # Soft-weld relative poses while the board is level.
        self._capture_veg_offsets()
        self._freeze_veggies_to_board()

    # ---------------------------------------------------------------- liquid / stove
    def _rebuild_water(self, force: bool = False) -> None:
        half_h = max(0.004, 0.5 * self.water_level * self.pot_inner_height)
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

        color = [0.25, 0.55, 0.92, 0.72]
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        mat = sapien.render.RenderMaterial(base_color=color)
        mat.metallic = 0.0
        mat.roughness = 0.12
        vertical_cyl_q = [0.70710678, 0.0, 0.70710678, 0.0]
        builder.add_cylinder_visual(
            pose=sapien.Pose([0, 0, 0], vertical_cyl_q),
            radius=self.pot_inner_radius,
            half_length=half_h,
            material=mat,
        )
        z = self.pot_bottom_z + half_h
        builder.set_initial_pose(sapien.Pose(p=[self.pot_xy[0], self.pot_xy[1], z]))
        self._liquid_entity = builder.build(name="pot_water")

    def _set_burner_glow(self, on: bool) -> None:
        self._set_stove_fire(bool(on), intensity=1.0 if on else 0.0)

    def _set_stove(self, on: bool) -> None:
        on = bool(on)
        if on == self.stove_on:
            return
        self.stove_on = on
        if on:
            self.turned_on_once = True
        self._set_burner_glow(on)

    # ---------------------------------------------------------------- board / veg physics
    def _get_rigid(self, entity: Any):
        obj = entity.actor if hasattr(entity, "actor") else entity
        for c in obj.get_components():
            if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                return c
        return None

    def _set_entity_pose(self, entity: Any, pose: sapien.Pose) -> None:
        obj = entity.actor if hasattr(entity, "actor") else entity
        obj.set_pose(pose)
        rigid = self._get_rigid(entity)
        if rigid is not None:
            try:
                rigid.set_kinematic_target(pose)
            except Exception:
                pass

    def _board_up_dot(self) -> float:
        """World +Z component of the board's local +Z (1 = flat, 0 = vertical)."""
        if self.board is None:
            return 1.0
        R = self.board.get_pose().to_transformation_matrix()[:3, :3]
        return float(R[:, 2] @ np.array([0.0, 0.0, 1.0]))

    def _capture_veg_offsets(self) -> None:
        if self.board is None:
            return
        board_pose = self.board.get_pose()
        self._veg_offsets = []
        for veg in self.veggies:
            self._veg_offsets.append(board_pose.inv() * veg.get_pose())

    def _freeze_veggies_to_board(self) -> None:
        for rigid in self._veg_rigids:
            if rigid is None:
                continue
            try:
                rigid.set_disable_gravity(True)
                rigid.set_kinematic(True)
                rigid.set_linear_velocity(np.zeros(3))
                rigid.set_angular_velocity(np.zeros(3))
            except Exception:
                pass

    def _release_veggies_physics(self, *, settle_on_board: bool = False) -> None:
        """Free produce as dynamic bodies so they can slide under gravity + contact.

        Soft-welded kinematics often sit slightly inside the board collider; releasing
        in place causes PhysX penetration explosions. Lift each piece a few mm along
        the board normal before enabling dynamics. Optionally settle while the board
        is still flat so contact is established before tipping.
        """
        if self._veg_released:
            return
        self._veg_released = True
        if self.board is None:
            return
        board_pose = self.board.get_pose()
        R = board_pose.to_transformation_matrix()[:3, :3]
        up = R[:, 2]
        sep = 0.003  # meters — clear the board surface without looking floated
        for veg, offset, rigid in zip(self.veggies, self._veg_offsets, self._veg_rigids):
            pose_v = board_pose * offset
            p = np.asarray(pose_v.p, dtype=float) + sep * up
            self._set_entity_pose(veg, sapien.Pose(p.tolist(), list(pose_v.q)))
            if rigid is None:
                continue
            try:
                rigid.set_kinematic(False)
                rigid.set_disable_gravity(False)
                rigid.set_linear_velocity(np.zeros(3))
                rigid.set_angular_velocity(np.zeros(3))
            except Exception:
                pass
        if settle_on_board:
            # Establish board contact before any tip motion.
            hold = self.board.get_pose()
            for _ in range(36):
                self._set_entity_pose(self.board, hold)
                self.scene.step()
        print(
            f"[make_soup] veggies free (board_up_dot={self._board_up_dot():.3f} "
            f"pour_armed={self._pour_armed})"
        )

    def _sync_veggies_to_board(self) -> None:
        if self._veg_released or self.board is None:
            return
        board_pose = self.board.get_pose()
        for veg, offset in zip(self.veggies, self._veg_offsets):
            self._set_entity_pose(veg, board_pose * offset)

    def _ee_pose(self, arm: ArmTag) -> sapien.Pose:
        p = self.get_arm_pose(str(arm))
        return sapien.Pose(list(p[:3]), list(p[3:7]))

    def _weld_board_to_ee(self, arm: ArmTag) -> None:
        if self.board is None:
            return
        if self._board_rigid is not None:
            try:
                self._board_rigid.set_disable_gravity(True)
                self._board_rigid.set_kinematic(True)
            except Exception:
                pass
        self._board_weld_offset = self._ee_pose(arm).inv() * self.board.get_pose()
        self._board_welded = True
        self._ignore_board_robot_collision()
        self._capture_veg_offsets()

    def _release_board_weld(self) -> None:
        self._board_welded = False
        self._board_weld_offset = None
        if self._board_rigid is not None:
            try:
                self._board_rigid.set_kinematic(False)
                self._board_rigid.set_disable_gravity(False)
            except Exception:
                pass

    def _sync_board_to_ee(self) -> None:
        """Keep the board fixed in the grasping hand (single arm, no reparent)."""
        if not self._board_welded or self._board_weld_offset is None:
            return
        pose = self._ee_pose(self.arm) * self._board_weld_offset
        self._set_entity_pose(self.board, pose)

    def _set_board_pose_keep_weld(self, pose: sapien.Pose) -> None:
        """Set board pose and rebuild the weld so it stays in the same hand."""
        self._set_entity_pose(self.board, pose)
        if self._board_welded:
            self._board_weld_offset = self._ee_pose(self.arm).inv() * self.board.get_pose()

    def _set_collision_ignore(self, entities: list[Any], ignore_bit: int, ignore_id: int) -> None:
        for ent in entities:
            if ent is None:
                continue
            try:
                shapes = []
                if hasattr(ent, "get_collision_shapes"):
                    shapes = list(ent.get_collision_shapes())
                elif hasattr(ent, "actor"):
                    for c in ent.actor.get_components():
                        if isinstance(
                            c,
                            (
                                sapien.physx.PhysxRigidDynamicComponent,
                                sapien.physx.PhysxRigidStaticComponent,
                            ),
                        ):
                            shapes.extend(c.get_collision_shapes())
                else:
                    for c in ent.get_components():
                        if isinstance(
                            c,
                            (
                                sapien.physx.PhysxRigidDynamicComponent,
                                sapien.physx.PhysxRigidStaticComponent,
                            ),
                        ):
                            shapes.extend(c.get_collision_shapes())
                for shape in shapes:
                    g0, g1, g2, g3 = shape.get_collision_groups()
                    shape.set_collision_groups(
                        [
                            int(g0),
                            int(g1),
                            int(g2) | ignore_bit,
                            (int(g3) & 0xFFFF0000) | ignore_id,
                        ]
                    )
            except Exception:
                pass

    def _ignore_board_robot_collision(self) -> None:
        ignore_bit, ignore_id = self.BOARD_IGNORE_BIT, self.BOARD_IGNORE_ID
        ents = [self.board.actor]
        try:
            ents += list(self.robot.left_entity.get_links()) + list(
                self.robot.right_entity.get_links()
            )
        except Exception:
            pass
        self._set_collision_ignore(ents, ignore_bit, ignore_id)

    def _ignore_board_pot_collision(self) -> None:
        """Keep the kinematic board from crushing produce against the pot walls."""
        if self.board is None or getattr(self, "pot", None) is None:
            return
        bit, gid = 1 << 15, 15
        self._set_collision_ignore([self.board.actor, self.pot], bit, gid)

    def _ignore_board_veg_collision(self) -> None:
        """Stop the board scooping settled produce back out as it rolls level."""
        if self.board is None:
            return
        bit, gid = 1 << 16, 16
        ents: list[Any] = [self.board.actor]
        for veg in self.veggies:
            ents.append(getattr(veg, "actor", veg))
        self._set_collision_ignore(ents, bit, gid)

    def _veg_in_pot(self, veg: Any) -> bool:
        p = np.asarray(veg.get_pose().p, dtype=float)
        dxy = float(np.linalg.norm(p[:2] - np.asarray(self.pot_xy)))
        return (
            dxy < self.pot_inner_radius * 0.95
            and float(p[2]) > self.pot_bottom_z - 0.01
            and float(p[2]) < self.pot_rim_z + 0.06
        )

    def _check_veg_fallen(self) -> None:
        """Mark failure if a piece left the board and is not heading into the pot."""
        if not self._veg_released:
            return
        pot_xy = np.asarray(self.pot_xy, dtype=float)
        for veg in self.veggies:
            if self._veg_in_pot(veg):
                continue
            p = np.asarray(veg.get_pose().p, dtype=float)
            # Still in flight above the pot / board — wait.
            if float(p[2]) > self.pot_rim_z + 0.02:
                continue
            # On / near the table and clear of the pot → spilled.
            if float(p[2]) < self.table_top + 0.06:
                d_pot = float(np.linalg.norm(p[:2] - pot_xy))
                if d_pot > self.pot_inner_radius + 0.03:
                    self._veg_fallen = True
                    return
            # Fell below the counter entirely.
            if float(p[2]) < self.table_top - 0.05:
                self._veg_fallen = True
                return

    # ---------------------------------------------------------------- per-step
    def _knob_is_pressed(self) -> bool:
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

    def _update_kinematic_tasks(self) -> None:
        super()._update_kinematic_tasks()
        if not getattr(self, "_loaded", False):
            return

        self._sync_board_to_ee()

        # Soft-weld while held (level carry + tip). Once freed, PhysX owns them.
        if not self._veg_released:
            if (
                self._pour_armed
                and not getattr(self, "_force_veg_hold", False)
                and self._board_up_dot() < self.tilt_hold_dot
            ):
                self._release_veggies_physics()
            else:
                self._sync_veggies_to_board()

        self._check_veg_fallen()

        if not getattr(self, "_ignore_knob", False):
            pressed = self._knob_is_pressed()
            if pressed and not self._prev_knob_pressed:
                self._set_stove(not self.stove_on)
            self._prev_knob_pressed = pressed
        else:
            self._prev_knob_pressed = False

    def _idle_steps(self, n_steps: int, until=None) -> None:
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

    # ---------------------------------------------------------------- expert motion
    def _knob_pose(self, offset, turn_angle: float) -> list[float]:
        base_q = np.asarray(GRASP_DIRECTION_DIC["front"], dtype=float)
        ee_p = np.asarray(self.knob_xyz, dtype=float) + np.asarray(offset, dtype=float)
        twist_q = np.array(
            [np.cos(turn_angle / 2), np.sin(turn_angle / 2), 0.0, 0.0],
            dtype=float,
        )
        ee_q = t3d.quaternions.qmult(base_q, twist_q)
        return [*ee_p.tolist(), *ee_q.tolist()]

    def _knob_turn_pose(self, standoff: float, turn_angle: float) -> list[float]:
        return self._knob_pose(
            [0.0, -(self.EE_TO_TCP + float(standoff)), 0.0], turn_angle
        )

    def _turn_knob_on(self) -> None:
        """Reach the front knob and twist the stove on (boil_milk corridor)."""
        arm = self.arm
        start_angle = -np.pi / 2 if self.stove_on else 0.0
        end_angle = -np.pi / 2
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
        self._set_stove(True)
        self._idle_steps(8)
        self.move(self.open_gripper(arm))
        for offset in reversed(path):
            self.move(self.move_to_pose(arm, self._knob_pose(offset, end_angle)))
        self._ignore_knob = False
        self._prev_knob_pressed = False

    def _top_down_pose(self, tcp_xyz) -> list[float]:
        return [
            float(tcp_xyz[0]),
            float(tcp_xyz[1]),
            float(tcp_xyz[2]) + self.EE_TO_TCP,
            *[float(v) for v in GRASP_DIRECTION_DIC["top_down"]],
        ]

    def _tcp_pos(self, arm: ArmTag) -> np.ndarray:
        p = (
            self.robot.get_right_tcp_pose()
            if str(arm) == "right"
            else self.robot.get_left_tcp_pose()
        )
        return np.asarray(p[:3], dtype=float)

    def _handle_world(self) -> np.ndarray:
        """World position of the board handle center (grasp target)."""
        pose = self.board.get_pose()
        R = pose.to_transformation_matrix()[:3, :3]
        local = np.asarray(self._handle_local, dtype=float)
        # Contact is on the handle top.
        local = local + np.array([0.0, 0.0, float(self.handle_half[2])], dtype=float)
        return np.asarray(pose.p, dtype=float) + R @ local

    def _grasp_board(self) -> bool:
        """Top-down grasp of the handle; weld stays on this arm for the episode."""
        arm = self.arm
        self.plan_success = True
        self.move(self.open_gripper(arm))

        handle = self._handle_world()
        # One planned hover straight above the handle — no angled / side pinch.
        # Several hover heights and a small inward nudge give the planner room
        # to find a wrist configuration it can descend from.
        hover = False
        for z_off, dx, dy in (
            (0.16, 0.0, 0.0),
            (0.20, 0.0, 0.0),
            (0.16, -0.02, 0.02),
            (0.24, 0.0, 0.0),
        ):
            target = np.array(
                [handle[0] + dx, handle[1] + dy, handle[2] + z_off], dtype=float
            )
            self.plan_success = True
            self.move(
                (arm, [Action(arm, "move", target_pose=self._top_down_pose(target))])
            )
            if self.plan_success:
                hover = True
                break
        if not hover:
            print("[make_soup] handle hover unreachable")
            return False

        # Descend on the handle with relative steps: a fresh absolute IK solve at
        # every height frequently fails from the hover configuration.
        goal = np.array(
            [handle[0], handle[1], float(handle[2]) + 0.012], dtype=float
        )
        for _ in range(4):
            delta = goal - self._tcp_pos(arm)
            if float(np.linalg.norm(delta)) < 0.008:
                break
            step = np.clip(delta, -0.07, 0.07)
            self.plan_success = True
            self.move(
                self.move_by_displacement(
                    arm,
                    x=float(step[0]),
                    y=float(step[1]),
                    z=float(step[2]),
                    quat=list(GRASP_DIRECTION_DIC["top_down"]),
                    move_axis="world",
                )
            )
            if not self.plan_success:
                print("[make_soup] handle descent failed")
                return False
        self.move(self.close_gripper(arm, pos=0.0))
        self._idle_steps(6)

        dist = float(np.linalg.norm(self._tcp_pos(arm) - self._handle_world()))
        if dist > self.grasp_tcp_tol * 2.5:
            print(f"[make_soup] refuse weld — TCP far from handle ({dist:.3f} m)")
            return False

        # Seat a level board under the TCP, then weld once to this arm.
        self._weld_board_to_ee(arm)
        self._seat_board_in_hand()
        self._capture_veg_offsets()
        self._sync_veggies_to_board()
        print(f"[make_soup] grasped handle tcp-dist={dist:.3f} arm={arm}")
        return True

    def _seat_board_in_hand(self) -> None:
        """Level board with handle under the TCP of the grasping arm."""
        if self.board is None or not self._board_welded:
            return
        tcp = self._tcp_pos(self.arm)
        local = np.asarray(self._handle_local, dtype=float) + np.array(
            [0.0, 0.0, float(self.handle_half[2])], dtype=float
        )
        board_p = tcp - local
        min_z = float(self.table_top) + float(self.board_half[2]) + 0.02
        board_p[2] = max(float(board_p[2]), min_z)
        self._set_board_pose_keep_weld(sapien.Pose(board_p.tolist(), [1, 0, 0, 0]))
        if not self._veg_released:
            self._sync_veggies_to_board()

    def _flatten_board(self) -> None:
        """Keep the board level in the same hand (no reparent / teleport)."""
        if self.board is None:
            return
        p = np.asarray(self.board.get_pose().p, dtype=float)
        self._set_board_pose_keep_weld(sapien.Pose(p.tolist(), [1, 0, 0, 0]))
        if not self._veg_released:
            self._sync_veggies_to_board()

    def _carry_board_level(self, target_xy, z: float) -> None:
        """Translate with top-down EE; board stays welded and level in-hand."""
        arm = self.arm
        bp = np.asarray(self.board.get_pose().p, dtype=float)
        dx = float(target_xy[0] - bp[0])
        dy = float(target_xy[1] - bp[1])
        dz = float(z - bp[2])
        steps = 5
        for _ in range(steps):
            self.plan_success = True
            self.move(
                self.move_by_displacement(
                    arm,
                    x=dx / steps,
                    y=dy / steps,
                    z=dz / steps,
                    quat=list(GRASP_DIRECTION_DIC["top_down"]),
                    move_axis="world",
                )
            )
            self._sync_board_to_ee()
            self._flatten_board()
            if self._veg_released:
                break

    def _nudge_board_to(self, target_xy, z: float, tol: float = 0.004) -> None:
        """Close a small carry error with pure translation (orientation frozen)."""
        arm = self.arm
        for _ in range(2):
            bp = np.asarray(self.board.get_pose().p, dtype=float)
            delta = np.array(
                [float(target_xy[0]) - bp[0], float(target_xy[1]) - bp[1], z - bp[2]],
                dtype=float,
            )
            if float(np.linalg.norm(delta)) < tol:
                return
            delta = np.clip(delta, -0.05, 0.05)
            self.plan_success = True
            self.move(
                self.move_by_displacement(
                    arm,
                    x=float(delta[0]),
                    y=float(delta[1]),
                    z=float(delta[2]),
                    quat=list(GRASP_DIRECTION_DIC["top_down"]),
                    move_axis="world",
                )
            )
            self.plan_success = True
            self._sync_board_to_ee()
            self._flatten_board()

    @staticmethod
    def _rot_about_y(pose: sapien.Pose, pivot, angle: float) -> sapien.Pose:
        """Rigidly rotate ``pose`` about the world-Y axis through ``pivot``."""
        piv = np.asarray(pivot, dtype=float)
        rot = sapien.Pose(
            [0.0, 0.0, 0.0],
            [float(v) for v in euler2quat(0.0, float(angle), 0.0, axes="sxyz")],
        )
        local = sapien.Pose(
            (np.asarray(pose.p, dtype=float) - piv).tolist(),
            [float(v) for v in pose.q],
        )
        out = rot * local
        return sapien.Pose(
            (np.asarray(out.p, dtype=float) + piv).tolist(),
            [float(v) for v in out.q],
        )

    def _pour_into_pot(self) -> None:
        """Carry the board level to the pot rim, then roll the wrist ~40°.

        Hand and board turn together about the board's pour edge, so the lip
        stays parked over the pot mouth and pieces slide off under PhysX.
        """
        arm = self.arm
        pot = np.asarray(self.pot_xy, dtype=float)
        hx = float(self.board_half[0])
        side_sign = 1.0 if str(arm) == "right" else -1.0
        tip_max = float(self.pour_tip_rad)

        # Board beside the pot: pour edge just over the pot center-line so a
        # 40° wrist tip drops pieces into the mouth.
        side_xy = np.array(
            [
                float(pot[0] + side_sign * (hx - 0.25 * float(self.pot_inner_radius))),
                float(pot[1]),
            ],
            dtype=float,
        )
        hover_z = float(self.pot_rim_z) + 0.028

        self._carry_board_level(side_xy, hover_z)
        self._flatten_board()
        # Close the residual carry error with plain top-down translation. Never
        # re-plan an absolute grasp pose here: that makes the arm swing to a new
        # IK branch and the board appears to spin on its way to the pot.
        self._nudge_board_to(side_xy, hover_z)
        self._flatten_board()
        pour_flat = self.board.get_pose()
        print(
            f"[make_soup] pour pose board={np.round(np.asarray(pour_flat.p), 4)} "
            f"edge={float(pour_flat.p[0]) - side_sign * hx:.3f} pot={pot} arm={arm}"
        )

        self._ignore_board_pot_collision()
        self._pour_armed = True
        self._force_veg_hold = False

        def in_pot() -> bool:
            return all(self._veg_in_pot(v) for v in self.veggies)

        # Free while flat so contact settles, then tip the wrist.
        self._release_veggies_physics(settle_on_board=True)

        # Tip: hand and board turn together as one rigid body about the board
        # center, so the lip dips toward the pot mouth and pieces slide down it.
        # Both transforms come from the same rotation, so the weld offset stays
        # exact — nothing to re-seat mid-air, and no arc for the arm to swing
        # through. The board is advanced a fraction of a degree per physics step
        # so it never teleports up into the food.
        ee_flat = self._ee_pose(arm)
        pivot = np.asarray(pour_flat.p, dtype=float)
        weld_offset = self._board_weld_offset
        tip_steps = 120
        wrist_at = (tip_steps // 2, tip_steps)
        hold_pose = pour_flat

        def _roll(frac: float) -> tuple[sapien.Pose, list[float]]:
            theta = -side_sign * tip_max * frac
            # Ease the lip in over the pot as it rolls.
            drift = np.array(
                [-side_sign * 0.010 * frac, 0.0, -0.008 * frac], dtype=float
            )
            board = self._rot_about_y(pour_flat, pivot, theta)
            ee = self._rot_about_y(ee_flat, pivot, theta)
            board = sapien.Pose(
                (np.asarray(board.p, dtype=float) + drift).tolist(),
                [float(v) for v in board.q],
            )
            ee = sapien.Pose(
                (np.asarray(ee.p, dtype=float) + drift).tolist(),
                [float(v) for v in ee.q],
            )
            return board, [*[float(v) for v in ee.p], *[float(v) for v in ee.q]]

        self._board_welded = False
        if self._board_rigid is not None:
            try:
                self._board_rigid.set_kinematic(True)
            except Exception:
                pass

        for i in range(1, tip_steps + 1):
            hold_pose, ee_target = _roll(i / tip_steps)
            self._set_entity_pose(self.board, hold_pose)
            if i in wrist_at:
                self.plan_success = True
                self.move(self.move_to_pose(arm, ee_target))
                self.plan_success = True
                self._set_entity_pose(self.board, hold_pose)
            self.scene.step()
            self._check_veg_fallen()
            if in_pot():
                break

        for _ in range(120):
            self._set_entity_pose(self.board, hold_pose)
            if in_pot():
                break
            self.scene.step()
            self._check_veg_fallen()

        print(
            f"[make_soup] after tip in_pot="
            f"{sum(self._veg_in_pot(v) for v in self.veggies)}/{len(self.veggies)}"
        )
        self._ignore_board_veg_collision()

        # Hand the board back to the weld (its pose already matches the tilted
        # EE exactly) and roll level on the same axis, so the arm reverses the
        # tip instead of hunting for a new configuration.
        self._board_weld_offset = weld_offset
        self._board_welded = True
        _, ee_level = _roll(0.0)
        self.plan_success = True
        self.move(self.move_to_pose(arm, ee_level))
        self.plan_success = True
        self._set_board_pose_keep_weld(pour_flat)
        self._idle_steps(4)
        self.move(
            self.move_by_displacement(
                arm,
                x=float(side_sign * 0.05),
                z=0.05,
                quat=list(GRASP_DIRECTION_DIC["top_down"]),
                move_axis="world",
            )
        )
        self.plan_success = True
        self._flatten_board()
        self._pour_armed = False
        self._force_veg_hold = False

    def _place_board_on_table(self) -> None:
        """Set the board down on the table away from the pot, then release."""
        arm = self.arm
        place_xy = np.array(
            [float(self.board_xy[0]), float(self.board_xy[1])], dtype=float
        )
        # Keep clear of the pot / stove apron.
        pot = np.asarray(self.pot_xy, dtype=float)
        if float(np.linalg.norm(place_xy - pot)) < 0.18:
            place_xy[1] = min(float(place_xy[1]), -0.10)
        self._carry_board_level(place_xy, self.table_top + 0.06)
        self._flatten_board()
        # Lower onto the table surface.
        self._carry_board_level(
            place_xy, self.table_top + float(self.board_half[2]) + 0.002
        )
        self._release_board_weld()
        # Pin board flat on the table so it does not bounce into the pot.
        if self.board is not None:
            pz = self.table_top + float(self.board_half[2])
            self._set_entity_pose(
                self.board,
                sapien.Pose(
                    [float(place_xy[0]), float(place_xy[1]), float(pz)],
                    [1, 0, 0, 0],
                ),
            )
            if self._board_rigid is not None:
                try:
                    self._board_rigid.set_kinematic(True)
                    self._board_rigid.set_disable_gravity(True)
                except Exception:
                    pass
        self.move(self.open_gripper(arm))
        self._idle_steps(8)
        # Retract so the knob reach is clear.
        self.move(
            self.move_by_displacement(arm, z=0.10, move_axis="world")
        )
        self.plan_success = True

    def play_once(self) -> dict[str, Any]:
        arm = self.arm
        self.plan_success = True

        if not self._grasp_board():
            self.plan_success = False
            return self.info

        # Level lift — veggies stay soft-welded on the board.
        self.move(
            self.move_by_displacement(
                arm,
                z=0.10,
                quat=list(GRASP_DIRECTION_DIC["top_down"]),
                move_axis="world",
            )
        )
        self._flatten_board()
        self._capture_veg_offsets()
        self._sync_veggies_to_board()
        if self._veg_released or self._veg_fallen:
            print("[make_soup] veggies fell during lift — fail")
            self.plan_success = False
            return self.info

        self._pour_into_pot()
        if self._veg_fallen or not all(self._veg_in_pot(v) for v in self.veggies):
            if not all(self._veg_in_pot(v) for v in self.veggies):
                self._idle_steps(40)
            if self._veg_fallen or not all(self._veg_in_pot(v) for v in self.veggies):
                print(
                    f"[make_soup] pour incomplete fallen={self._veg_fallen} "
                    f"in_pot={sum(self._veg_in_pot(v) for v in self.veggies)}/{len(self.veggies)}"
                )
                self.plan_success = False
                return self.info

        # Board goes back on the table — never left on the pot — then stove on.
        self._place_board_on_table()
        self._turn_knob_on()

        if self.check_success():
            self.plan_success = True

        self.info["info"] = {
            "{A}": "chopping_board",
            "{B}": "soup_pot",
            "{C}": "cooking_range",
            "{D}": "stove_knob",
            "{E}": "vegetables",
            "{a}": str(arm),
        }
        return self.info

    def check_success(self) -> bool:
        if not getattr(self, "_loaded", False):
            return False
        if self._veg_fallen:
            return False
        if not self.veggies:
            return False
        if not all(self._veg_in_pot(v) for v in self.veggies):
            return False
        if not self.stove_on or not self.turned_on_once:
            return False
        return True

    def get_obs(self) -> dict[str, Any]:
        obs = super().get_obs()
        n_in = sum(1 for v in self.veggies if self._veg_in_pot(v)) if self.veggies else 0
        obs["soup"] = {
            "stove_on": bool(self.stove_on),
            "veg_released": bool(self._veg_released),
            "veg_fallen": bool(self._veg_fallen),
            "board_up_dot": float(self._board_up_dot()),
            "n_veg": int(len(self.veggies)),
            "n_in_pot": int(n_in),
            "water_level": float(getattr(self, "water_level", 0.0)),
        }
        return obs
