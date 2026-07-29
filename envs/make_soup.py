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
from .utils.create_actor import create_box, create_sphere, UnStableError


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
    # Counter-local stove pose — keep on the right arm's half of the table.
    RANGE_REL_XY: ClassVar[tuple[float, float]] = (0.20, 0.06)

    BOARD_HALF: ClassVar[tuple[float, float, float]] = (0.095, 0.065, 0.010)
    BOARD_COLOR: ClassVar[tuple[float, float, float]] = (0.55, 0.38, 0.22)
    CUBE_HALF: ClassVar[float] = 0.012
    TOMATO_RADIUS: ClassVar[float] = 0.015
    VEG_COLORS: ClassVar[dict[str, tuple[float, float, float]]] = {
        "orange": (0.95, 0.45, 0.08),
        "green": (0.20, 0.65, 0.22),
        "purple": (0.55, 0.18, 0.70),
    }
    TOMATO_COLOR: ClassVar[tuple[float, float, float]] = (0.90, 0.10, 0.08)

    # Soft-friction model during an armed pour: pieces stay on the board until
    # the board's up-axis drops below this (≈35° tip). Past that they free-fall.
    TILT_HOLD_DOT: ClassVar[float] = 0.82
    # Accidental spill during carry (top-down grasp wobble is ~0.87–0.90): only
    # free-fall if tipped nearly sideways (~60°).
    TILT_SPILL_DOT: ClassVar[float] = 0.50
    # Expert pour tip (~55°) — intentional dump over the pot.
    POUR_TIP_RAD: ClassVar[float] = float(np.deg2rad(55.0))
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
        self._veg_offsets: list[sapien.Pose] = []
        self.veggies: list[Any] = []
        self._veg_rigids: list[Any] = []
        self._board_rigid = None
        self.board = None
        self.pot = None

        super().setup_demo(**kwags)
        self._configure_head_camera()

    def _configure_head_camera(self) -> None:
        cams = getattr(self, "cameras", None)
        if cams is None:
            return
        names = list(getattr(cams, "static_camera_name", []) or [])
        clist = list(getattr(cams, "static_camera_list", []) or [])
        if "head_camera" not in names:
            return
        camera = clist[names.index("head_camera")]
        rx, ry = getattr(self, "range_xy", (0.20, 0.06))
        cam_pos = np.array([float(rx) * 0.55, -1.05, 1.85], dtype=float)
        look_at = np.array([float(rx) * 0.7, float(ry) * 0.1, 0.82], dtype=float)
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
            camera.set_fovy(float(np.deg2rad(55)))
        except Exception:
            try:
                camera.fovy = float(np.deg2rad(55))
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
        self.cube_half = float(cfg.get("cube_half", self.CUBE_HALF))
        self.tomato_radius = float(cfg.get("tomato_radius", self.TOMATO_RADIUS))

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
        self._veg_offsets = []
        self.veggies = []
        self._veg_rigids = []
        self._board_rigid = None

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

        if getattr(self, "active_burner", None) is not None:
            try:
                self.active_burner.set_pose(sapien.Pose(p=[0.0, 0.0, -1.0]))
            except Exception:
                pass
        self._burner_shapes = []
        self._set_burner_glow(False)

        self._spawn_pot(float(bx), float(by), cfg)
        self._rebuild_water(force=True)

        # Chopping board just in front of the pot (same +x side as the knob).
        rx, ry = float(self.range_xy[0]), float(self.range_xy[1])
        board_x = float(cfg.get("board_x", float(bx)))
        board_y = float(cfg.get("board_y", float(by) - 0.16))
        if self.knob_xy[0] >= 0:
            board_x = max(0.08, board_x)
        else:
            board_x = min(-0.08, board_x)
        self.board_xy = (board_x, board_y)
        self.board = self._spawn_board(board_x, board_y, bz)
        self.add_prohibit_area(self.board, padding=0.04)

        self._spawn_vegetables(board_x, board_y, bz + 2.0 * self.board_half[2])
        self.arm = ArmTag("right" if self.knob_xy[0] >= 0 else "left")
        self._loaded = True
        print(
            f"[make_soup] arm={self.arm} board={self.board_xy} pot={self.pot_xy} "
            f"n_veg={len(self.veggies)}"
        )

    def _spawn_pot(self, cx: float, cy: float, cfg: dict[str, Any]) -> None:
        """Procedural hollow pot (same pattern as boil_milk)."""
        pot_r = float(cfg.get("pot_radius", 0.055))
        pot_h = float(cfg.get("pot_height", 0.075))
        wall = 0.005
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

    def _spawn_board(self, x: float, y: float, table_z: float) -> Actor:
        hx, hy, hz = [float(v) for v in self.board_half]
        pose = sapien.Pose([x, y, table_z + hz], [1, 0, 0, 0])
        board = create_box(
            self,
            pose=pose,
            half_size=[hx, hy, hz],
            color=list(self.BOARD_COLOR),
            name="chopping_board",
            is_static=False,
        )
        board.set_mass(0.18)
        # Top-down grasp frames at the board surface (local +Z).
        board.config["contact_points_pose"] = [
            [[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 1.0], [0, 0, 0, 1]],
            [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 1.0], [0, 0, 0, 1]],
            [[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 1.0], [0, 0, 0, 1]],
            [[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 1.0], [0, 0, 0, 1]],
        ]
        board.config["contact_points_group"] = [[0, 1, 2, 3]]
        board.config["contact_points_mask"] = [True]
        board.config["scale"] = [hx, hy, hz]
        self._board_rigid = None
        for c in board.actor.get_components():
            if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                try:
                    c.set_linear_damping(8.0)
                    c.set_angular_damping(12.0)
                except Exception:
                    pass
                self._board_rigid = c
                break
        return board

    def _spawn_vegetables(self, bx: float, by: float, board_top: float) -> None:
        """Colored cubes + a small round tomato resting on the board."""
        layout = [
            ("cube_orange", "cube", self.VEG_COLORS["orange"], (-0.035, -0.020)),
            ("cube_green", "cube", self.VEG_COLORS["green"], (0.010, 0.025)),
            ("cube_purple", "cube", self.VEG_COLORS["purple"], (0.040, -0.015)),
            ("tomato", "sphere", self.TOMATO_COLOR, (-0.010, 0.005)),
        ]
        jitter = 0.008
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
                        c.set_linear_damping(0.4)
                        c.set_angular_damping(0.4)
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
        color = [0.95, 0.35, 0.05, 1.0] if on else [0.20, 0.20, 0.22, 1.0]
        for s in getattr(self, "_burner_shapes", []) or []:
            try:
                s.material.set_base_color(color)
            except Exception:
                pass
        if getattr(self, "stove_knob_indicator", None) is not None:
            angle = -np.pi / 2 if on else 0.0
            radius = float(self._knob_radius) * 0.55
            kx, _, kz = self.knob_xyz
            self.stove_knob_indicator.set_pose(
                sapien.Pose(
                    p=[
                        float(kx + radius * np.sin(angle)),
                        float(self._knob_front_y),
                        float(kz + radius * np.cos(angle)),
                    ],
                    q=[
                        float(np.cos(angle / 2)),
                        0.0,
                        float(np.sin(angle / 2)),
                        0.0,
                    ],
                )
            )

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

    def _release_veggies_physics(self) -> None:
        """Drop soft weld — pieces fall / slide under PhysX gravity."""
        if self._veg_released:
            return
        self._veg_released = True
        pot = np.asarray(self.pot_xy, dtype=float)
        board_xy = np.asarray(self.board.get_pose().p[:2], dtype=float) if self.board else pot
        over_pot = float(np.linalg.norm(board_xy - pot)) < self.pot_inner_radius + 0.04
        for i, (veg, rigid) in enumerate(zip(self.veggies, self._veg_rigids)):
            if rigid is None:
                continue
            try:
                rigid.set_kinematic(False)
                rigid.set_disable_gravity(False)
                p = np.asarray(veg.get_pose().p, dtype=float)
                if self._pour_armed and over_pot:
                    # Drop through the pot mouth so gravity seats them in the water.
                    ang = 2.0 * np.pi * i / max(1, len(self.veggies))
                    r = 0.35 * self.pot_inner_radius
                    drop = sapien.Pose(
                        [
                            float(pot[0] + r * np.cos(ang)),
                            float(pot[1] + r * np.sin(ang)),
                            float(self.pot_rim_z + 0.025),
                        ],
                        list(veg.get_pose().q),
                    )
                    self._set_entity_pose(veg, drop)
                    # Clear kinematic target so free dynamics take over from the drop pose.
                    rigid.set_kinematic(False)
                    rigid.set_disable_gravity(False)
                    rigid.set_linear_velocity(np.array([0.0, 0.0, -0.20]))
                else:
                    toward = pot - p[:2]
                    dist = float(np.linalg.norm(toward))
                    if dist > 1e-4:
                        v_xy = 0.12 * toward / dist
                        rigid.set_linear_velocity(np.array([v_xy[0], v_xy[1], -0.05]))
                    else:
                        rigid.set_linear_velocity(np.zeros(3))
                rigid.set_angular_velocity(np.zeros(3))
            except Exception:
                pass
        print(
            f"[make_soup] veggies released (board_up_dot={self._board_up_dot():.3f} "
            f"pour_armed={self._pour_armed} over_pot={over_pot})"
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
        if not self._board_welded or self._board_weld_offset is None:
            return
        pose = self._ee_pose(self.arm) * self._board_weld_offset
        if not self._pour_armed:
            # Careful carry: follow EE translation but keep the board level so
            # grasp / wrist wobble does not dump the vegetables mid-transit.
            pose = sapien.Pose(list(pose.p), [1, 0, 0, 0])
        self._set_entity_pose(self.board, pose)

    def _set_board_pose_keep_weld(self, pose: sapien.Pose) -> None:
        self._set_entity_pose(self.board, pose)
        if self._board_welded:
            # Rebuild weld so subsequent EE sync matches this pose.
            # When not pouring, store offset as if the board were flat under the EE.
            if not self._pour_armed:
                flat = sapien.Pose(list(pose.p), [1, 0, 0, 0])
                self._set_entity_pose(self.board, flat)
                self._board_weld_offset = self._ee_pose(self.arm).inv() * flat
            else:
                self._board_weld_offset = self._ee_pose(self.arm).inv() * self.board.get_pose()

    def _ignore_board_robot_collision(self) -> None:
        ignore_bit, ignore_id = self.BOARD_IGNORE_BIT, self.BOARD_IGNORE_ID
        ents = [self.board.actor]
        try:
            ents += list(self.robot.left_entity.get_links()) + list(
                self.robot.right_entity.get_links()
            )
        except Exception:
            pass
        for ent in ents:
            try:
                shapes = []
                if hasattr(ent, "get_collision_shapes"):
                    shapes = list(ent.get_collision_shapes())
                else:
                    for c in ent.get_components():
                        if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
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

        # Soft friction only while pouring (carry keeps the board kinematically
        # level). Extreme tip during pour, or the expert dump, frees the pieces.
        if not self._veg_released:
            if self._pour_armed and self._board_up_dot() < self.tilt_hold_dot:
                self._release_veggies_physics()
            elif not self._pour_armed:
                self._sync_veggies_to_board()
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

    def _grasp_board(self) -> bool:
        arm = self.arm
        self.plan_success = True
        self.move(self.open_gripper(arm))
        self.move(
            self.grasp_actor(
                self.board,
                arm_tag=arm,
                pre_grasp_dis=0.10,
                grasp_dis=0.0,
            )
        )
        if not self.plan_success:
            # Fallback: top-down pinch at board center.
            self.plan_success = True
            p = np.asarray(self.board.get_pose().p, dtype=float)
            hover = p.copy()
            hover[2] = self.table_top + 0.18
            self.move(
                (arm, [Action(arm, "move", target_pose=self._top_down_pose(hover))])
            )
            if not self.plan_success:
                return False
            pinch = p.copy()
            pinch[2] = self.table_top + self.board_half[2] + 0.012
            self.move(
                (arm, [Action(arm, "move", target_pose=self._top_down_pose(pinch))])
            )
            if not self.plan_success:
                return False
            self.move(self.close_gripper(arm, pos=0.0))

        self._weld_board_to_ee(arm)
        self._flatten_board()
        self._capture_veg_offsets()
        self._sync_veggies_to_board()
        return True

    def _carry_board_level(self, target_xy, z: float) -> None:
        """Translate the welded board while keeping it flat (no tip quat)."""
        arm = self.arm
        bp = np.asarray(self.board.get_pose().p, dtype=float)
        dx = float(target_xy[0] - bp[0])
        dy = float(target_xy[1] - bp[1])
        dz = float(z - bp[2])
        # Multi-step so CuRobo stays reachable and veggies stay soft-welded.
        steps = 4
        for i in range(1, steps + 1):
            self.plan_success = True
            self.move(
                self.move_by_displacement(
                    arm,
                    x=dx / steps,
                    y=dy / steps,
                    z=dz / steps,
                    move_axis="world",
                )
            )
            self._sync_board_to_ee()
            self._flatten_board()
            if self._veg_released:
                break

    def _flatten_board(self) -> None:
        """Keep the board level while welded (top-down grasp can tip it a bit)."""
        if self.board is None:
            return
        p = np.asarray(self.board.get_pose().p, dtype=float)
        self._set_board_pose_keep_weld(sapien.Pose(p.tolist(), [1, 0, 0, 0]))
        if not self._veg_released:
            self._sync_veggies_to_board()

    def _pour_into_pot(self) -> None:
        """Tip the board over the pot so soft friction breaks and pieces fall in."""
        arm = self.arm
        pot = np.asarray(self.pot_xy, dtype=float)
        # Hover just above the rim, board still level.
        hover_z = float(self.pot_rim_z) + 0.08
        self._carry_board_level(pot, hover_z)
        self._flatten_board()

        self._pour_armed = True
        tip_steps = 10
        tip_max = float(self.pour_tip_rad)
        # Tip about +X so the +Y edge (toward the pot / away from robot) drops.
        for i in range(1, tip_steps + 1):
            frac = i / tip_steps
            tip = tip_max * frac
            tip_q = qmult(
                euler2quat(float(tip), 0.0, 0.0, axes="sxyz"),
                np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            )
            # Keep board center over the pot; lower as we tip.
            p = np.array(
                [float(pot[0]), float(pot[1]), hover_z - 0.03 * frac],
                dtype=float,
            )
            self._set_board_pose_keep_weld(sapien.Pose(p.tolist(), tip_q.tolist()))
            if not self._veg_released:
                self._sync_veggies_to_board()

            pour_q = qmult(
                euler2quat(float(tip), 0.0, 0.0, axes="sxyz"),
                np.array(GRASP_DIRECTION_DIC["top_down"], dtype=float),
            )
            self.move(
                self.move_by_displacement(
                    arm,
                    x=0.0,
                    y=0.004,
                    z=-0.004,
                    quat=pour_q.tolist(),
                    move_axis="world",
                )
            )
            self.plan_success = True
            self._set_board_pose_keep_weld(sapien.Pose(p.tolist(), tip_q.tolist()))

            if frac >= 0.40 and not self._veg_released:
                self._release_veggies_physics()
            self._idle_steps(5)

        if not self._veg_released:
            self._release_veggies_physics()

        # Let pieces fall into the pot under gravity.
        self._idle_steps(
            100,
            until=lambda: all(self._veg_in_pot(v) for v in self.veggies),
        )

        # Untilt and park the board clear of the pot / knob.
        flat_p = np.array(
            [float(self.board_xy[0]), float(self.board_xy[1]), self.table_top + 0.14],
            dtype=float,
        )
        self._set_board_pose_keep_weld(sapien.Pose(flat_p.tolist(), [1, 0, 0, 0]))
        self.move(
            self.move_by_displacement(
                arm,
                x=float(self.board_xy[0] - pot[0]) * 0.4,
                y=float(self.board_xy[1] - pot[1]) * 0.4,
                z=0.02,
                quat=list(GRASP_DIRECTION_DIC["top_down"]),
                move_axis="world",
            )
        )
        self.plan_success = True
        self._pour_armed = False

    def play_once(self) -> dict[str, Any]:
        arm = self.arm
        self.plan_success = True

        if not self._grasp_board():
            self.plan_success = False
            return self.info

        # Lift level — veggies must stay on the board.
        self.move(self.move_by_displacement(arm, z=0.12, move_axis="arm"))
        self._sync_board_to_ee()
        self._flatten_board()
        self._capture_veg_offsets()
        self._sync_veggies_to_board()
        if self._veg_released or self._veg_fallen:
            print("[make_soup] veggies fell during lift — fail")
            self.plan_success = False
            return self.info

        self._pour_into_pot()
        if self._veg_fallen or not all(self._veg_in_pot(v) for v in self.veggies):
            # One settle retry: nudge board tip again if pieces clung.
            if not all(self._veg_in_pot(v) for v in self.veggies):
                self._idle_steps(40)
            if self._veg_fallen or not all(self._veg_in_pot(v) for v in self.veggies):
                print(
                    f"[make_soup] pour incomplete fallen={self._veg_fallen} "
                    f"in_pot={sum(self._veg_in_pot(v) for v in self.veggies)}/{len(self.veggies)}"
                )
                self.plan_success = False
                return self.info

        # Put the board down away from the knob, then turn the stove on.
        self._carry_board_level(
            np.array(self.board_xy, dtype=float),
            self.table_top + 0.05,
        )
        self._release_board_weld()
        self.move(self.open_gripper(arm))
        self._idle_steps(10)

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
