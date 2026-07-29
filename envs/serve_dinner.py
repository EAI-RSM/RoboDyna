"""Serve dinner: turn off a lit stove, then tip meatballs from the pan onto a plate.

KitchenS scene. A skillet sits on an already-lit burner with 3–4 brown meatballs
resting in a thin sauce layer. The robot turns the stove off, grasps the pan
handle, carries it roughly level over a plate, and tips just enough that the
meatballs slide/fall onto the plate under physics. Tilting too early or too far
drops meatballs onto the table (failure). Success = stove off and every
meatball on the plate; any meatball off the plate is a failure.
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
from .utils.create_actor import (
    create_actor,
    create_sphere,
    create_visual_box,
    UnStableError,
)
from .utils.action import Action, ArmTag


class serve_dinner(KitchenS_base_task):
    """Turn off the stove and serve meatballs from the pan onto a plate."""

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
    RANGE_REL_XY: ClassVar[tuple[float, float]] = (0.18, 0.06)

    SKILLET_BASE_QPOS: ClassVar[list[float]] = [0.0, 0.0, 0.707, 0.707]
    PLATE_QPOS: ClassVar[list[float]] = [0.707, 0.707, 0.0, 0.0]

    MEATBALL_RADIUS: ClassVar[float] = 0.016
    MEATBALL_COLOR: ClassVar[tuple[float, float, float]] = (0.42, 0.22, 0.10)
    SAUCE_COLOR: ClassVar[tuple[float, float, float, float]] = (0.55, 0.16, 0.06, 0.88)
    SAUCE_HALF_H: ClassVar[float] = 0.0035

    # Soft-friction while pouring: meatballs stay welded until bowl-up drops
    # below this (~35° tip). Carry keeps them kinematically welded.
    TILT_HOLD_DOT: ClassVar[float] = 0.82
    # Expert pour tip (~50°) — intentional dump over the plate.
    POUR_TIP_RAD: ClassVar[float] = float(np.deg2rad(50.0))

    PAN_IGNORE_BIT: ClassVar[int] = 1 << 22
    PAN_IGNORE_ID: ClassVar[int] = 0x5E17

    def setup_demo(self, **kwags: Any) -> None:
        self._cfg = dict(kwags.get("task_args", {}).get("serve_dinner", {}))
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
        self.stove_on = True
        self.turned_off_once = False
        self._sauce_entity = None
        self._sauce_offset = None
        self._burner_shapes: list[Any] = []
        self._ring_parts: list[Any] = []
        self._ring_shapes: list[Any] = []
        self._ignore_knob = False
        self._expert_holding_knob = False
        self._prev_knob_pressed = False
        self._pan_welded = False
        self._pan_weld_offset = None
        self._balls_released = False
        self._balls_fallen = False
        self._pour_armed = False
        self._ball_offsets: list[sapien.Pose] = []
        self.meatballs: list[Any] = []
        self._ball_rigids: list[Any] = []
        self._pan_rigid = None
        self._pan_up_local = np.array([0.0, 1.0, 0.0], dtype=float)
        self.skillet = None
        self.plate = None

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
        rx, ry = getattr(self, "range_xy", (0.18, 0.06))
        cam_pos = np.array([float(rx) * 0.55, -1.10, 1.90], dtype=float)
        look_at = np.array([float(rx) * 0.7, float(ry) * 0.05, 0.82], dtype=float)
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
        self.pour_tip_rad = float(cfg.get("pour_tip_rad", self.POUR_TIP_RAD))
        self.pan_scale = float(cfg.get("pan_scale", 0.60))
        self.plate_scale = float(cfg.get("plate_scale", 0.55))
        self.meatball_radius = float(cfg.get("meatball_radius", self.MEATBALL_RADIUS))
        n_balls = cfg.get("n_meatballs", None)
        if n_balls is None:
            self.n_meatballs = int(np.random.randint(3, 5))  # 3 or 4
        else:
            self.n_meatballs = int(np.clip(int(n_balls), 3, 4))

        self.stove_on = True
        self.turned_off_once = False
        self._sauce_entity = None
        self._sauce_offset = None
        self._ignore_knob = False
        self._expert_holding_knob = False
        self._prev_knob_pressed = False
        self._pan_welded = False
        self._pan_weld_offset = None
        self._balls_released = False
        self._balls_fallen = False
        self._pour_armed = False
        self._ball_offsets = []
        self.meatballs = []
        self._ball_rigids = []
        self._pan_rigid = None

        bz = 0.74 + self.table_z_bias
        self.table_top = bz

        burner_name = str(cfg.get("burner", self.ACTIVE_BURNER)).strip().lower()
        if burner_name not in self.burner_positions:
            raise ValueError(
                f"serve_dinner.burner must be one of {list(self.burner_positions)}, "
                f"got {burner_name!r}"
            )
        bx, by = self.burner_positions[burner_name]
        self.burner_name = burner_name
        self.burner_xy = (float(bx), float(by))

        # Hide default solid disc; build a fire ring outside the pan bowl.
        if getattr(self, "active_burner", None) is not None:
            try:
                self.active_burner.set_pose(sapien.Pose(p=[0.0, 0.0, -1.0]))
            except Exception:
                pass
        self._burner_shapes = []
        self._clear_burner_ring()

        self._spawn_pan(float(bx), float(by))
        self._build_burner_ring(
            self.burner_xy[0], self.burner_xy[1], float(self.range_top_z) + 0.0012
        )
        self._set_burner_glow(True)

        # Plate in front of the pan (toward the robot), same lateral side.
        plate_x = float(cfg.get("plate_x", float(bx)))
        plate_y = float(cfg.get("plate_y", float(by) - 0.22))
        if self.knob_xy[0] >= 0:
            plate_x = max(0.06, plate_x)
        else:
            plate_x = min(-0.06, plate_x)
        self._spawn_plate(plate_x, plate_y, bz)

        self._spawn_sauce()
        self._spawn_meatballs()

        self.arm = ArmTag("right" if self.knob_xy[0] >= 0 else "left")
        self._loaded = True
        print(
            f"[serve_dinner] arm={self.arm} pan={self.burner_xy} plate={self.plate_xy} "
            f"n_balls={len(self.meatballs)} stove_on={self.stove_on}"
        )

    def _spawn_pan(self, bx: float, by: float) -> None:
        self.skillet_id = int(np.random.choice([0, 2]))
        skillet_q = list(self.SKILLET_BASE_QPOS)
        # Handle points −Y; bowl sits on the grate.
        y_guess = 0.071 * self.pan_scale
        skillet_pose = sapien.Pose(
            [bx, by - y_guess, float(self.range_top_z) + 0.002],
            skillet_q,
        )
        self.skillet = create_actor(
            self,
            pose=skillet_pose,
            modelname="106_skillet",
            model_id=self.skillet_id,
            convex=True,
            is_static=False,
            scale_mult=self.pan_scale,
        )
        self.skillet.set_name("106_skillet")
        self.skillet.set_mass(0.18)
        self.add_prohibit_area(self.skillet, padding=0.04)

        # Slide until bowl XY matches burner.
        for _ in range(12):
            bowl = np.asarray(self.skillet.get_functional_point(0)[:2], dtype=float)
            err = np.array([bx - float(bowl[0]), by - float(bowl[1])], dtype=float)
            if float(np.linalg.norm(err)) < 0.001:
                break
            p = self.skillet.get_pose()
            self.skillet.actor.set_pose(
                sapien.Pose(
                    [float(p.p[0] + err[0]), float(p.p[1] + err[1]), float(p.p[2])],
                    p.q,
                )
            )
        p = self.skillet.get_pose()
        self.skillet.actor.set_pose(
            sapien.Pose(
                [float(p.p[0]) + 0.008, float(p.p[1]) + 0.010, float(p.p[2])],
                p.q,
            )
        )
        self.burner_xy = (bx + 0.008, by + 0.010)

        self._pan_rigid = None
        for c in self.skillet.actor.get_components():
            if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                try:
                    c.set_linear_damping(10.0)
                    c.set_angular_damping(20.0)
                    # Kinematic until grasped — prevents init settle from tipping
                    # the pan and flagging meatballs as unstable.
                    c.set_disable_gravity(True)
                    c.set_kinematic(True)
                except Exception:
                    pass
                self._pan_rigid = c
                break

        # Local bowl-up axis: whichever local basis best matches world +Z at spawn.
        R = self.skillet.get_pose().to_transformation_matrix()[:3, :3]
        world_z = np.array([0.0, 0.0, 1.0], dtype=float)
        axis = int(np.argmax([abs(float(R[:, i] @ world_z)) for i in range(3)]))
        sign = 1.0 if float(R[:, axis] @ world_z) >= 0.0 else -1.0
        self._pan_up_local = sign * np.eye(3)[axis]

        bowl = np.asarray(self.skillet.get_functional_point(0)[:3], dtype=float)
        self.bowl_xy = (float(bowl[0]), float(bowl[1]))
        self.bowl_z = float(bowl[2])
        # Approx bowl inner radius at this pan scale (rim clearance for meatballs).
        self.bowl_inner_r = 0.055 * self.pan_scale / 0.60

    def _spawn_plate(self, x: float, y: float, bz: float) -> None:
        pose = sapien.Pose([x, y, bz], list(self.PLATE_QPOS))
        self.plate = create_actor(
            self,
            pose=pose,
            modelname="003_plate",
            model_id=0,
            convex=True,
            is_static=True,
            scale_mult=self.plate_scale,
        )
        self.plate.set_name("003_plate")
        if self.plate.config is None:
            self.plate.config = {"scale": [self.plate_scale] * 3}
        self.add_prohibit_area(self.plate, padding=0.03)
        self.plate_xy = (x, y)
        try:
            fp = self.plate.get_functional_point(0)
            self.plate_top_z = float(fp[2])
        except Exception:
            self.plate_top_z = bz + 0.02
        # Inner landing radius for success (slightly inside rim).
        cfg = getattr(self.plate, "config", {}) or {}
        ext = np.array(cfg.get("extents", [9.2, 1.1, 9.2]), dtype=float)
        sc = cfg.get("scale", [self.plate_scale] * 3)
        sc0 = float(sc[0] if isinstance(sc, (list, tuple)) else sc)
        self.plate_inner_r = 0.38 * float(max(ext[0], ext[2])) * sc0

    def _spawn_sauce(self) -> None:
        """Thin reddish-brown sauce disc covering the pan floor (visual + follows pan)."""
        if self._sauce_entity is not None:
            try:
                self.scene.remove_entity(self._sauce_entity)
            except Exception:
                pass
            self._sauce_entity = None

        bowl = np.asarray(self.skillet.get_functional_point(0)[:3], dtype=float)
        half_h = float(self.SAUCE_HALF_H)
        radius = max(0.02, self.bowl_inner_r * 0.92)
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("kinematic")
        mat = sapien.render.RenderMaterial(base_color=list(self.SAUCE_COLOR))
        mat.metallic = 0.0
        mat.roughness = 0.25
        vertical_cyl_q = [0.70710678, 0.0, 0.70710678, 0.0]
        builder.add_cylinder_visual(
            pose=sapien.Pose([0, 0, 0], vertical_cyl_q),
            radius=radius,
            half_length=half_h,
            material=mat,
        )
        sauce_pose = sapien.Pose(
            p=[float(bowl[0]), float(bowl[1]), float(bowl[2]) + half_h + 0.001]
        )
        builder.set_initial_pose(sauce_pose)
        self._sauce_entity = builder.build(name="pan_sauce")
        self._sauce_offset = self.skillet.get_pose().inv() * sauce_pose

    def _spawn_meatballs(self) -> None:
        """3–4 brown spheres resting in the sauce inside the pan bowl."""
        bowl = np.asarray(self.skillet.get_functional_point(0)[:3], dtype=float)
        r = float(self.meatball_radius)
        # Compact ring / cluster so all fit inside the bowl.
        layouts_3 = [(-0.018, -0.010), (0.018, -0.008), (0.000, 0.018)]
        layouts_4 = [
            (-0.018, -0.014),
            (0.018, -0.012),
            (-0.012, 0.016),
            (0.014, 0.014),
        ]
        offsets = layouts_4 if self.n_meatballs >= 4 else layouts_3
        jitter = 0.004
        for i, (dx, dy) in enumerate(offsets[: self.n_meatballs]):
            jx = float(np.random.uniform(-jitter, jitter))
            jy = float(np.random.uniform(-jitter, jitter))
            z = float(bowl[2]) + 2.0 * float(self.SAUCE_HALF_H) + r + 0.001
            pose = sapien.Pose(
                [float(bowl[0]) + dx + jx, float(bowl[1]) + dy + jy, z],
                [1, 0, 0, 0],
            )
            entity = create_sphere(
                self,
                pose=pose,
                radius=r,
                color=list(self.MEATBALL_COLOR) + [1.0],
                name=f"meatball_{i}",
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
            ball = Actor(entity, data, mass=0.025)
            rigid = None
            for c in ball.actor.get_components():
                if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                    try:
                        c.set_linear_damping(0.5)
                        c.set_angular_damping(0.5)
                        c.mass = 0.025
                    except Exception:
                        pass
                    rigid = c
                    break
            self.meatballs.append(ball)
            self._ball_rigids.append(rigid)
            self.add_prohibit_area(ball, padding=0.01)

        self._capture_ball_offsets()
        self._freeze_balls_to_pan()

    # ---------------------------------------------------------------- burner / stove
    def _clear_burner_ring(self) -> None:
        for part in getattr(self, "_ring_parts", []) or []:
            try:
                self.scene.remove_entity(part)
            except Exception:
                pass
        self._ring_parts = []
        self._ring_shapes = []

    def _build_burner_ring(self, cx: float, cy: float, cz: float) -> None:
        self._clear_burner_ring()
        n = 36
        radius = 0.055
        for i in range(n):
            ang = 2.0 * np.pi * i / n
            part = create_visual_box(
                self.scene,
                sapien.Pose(
                    p=[
                        float(cx + radius * np.cos(ang)),
                        float(cy + radius * np.sin(ang)),
                        float(cz),
                    ]
                ),
                half_size=[0.008, 0.005, 0.002],
                color=(0.20, 0.75, 1.0),
                name=f"burner_ring_{i}",
            )
            self._ring_parts.append(part)
            try:
                comps = part.get_components()
            except Exception:
                comps = getattr(part, "components", [])
            for c in comps:
                if isinstance(c, sapien.render.RenderBodyComponent):
                    self._ring_shapes.extend(list(c.render_shapes))

    def _set_burner_glow(self, on: bool) -> None:
        inten = 1.0 if on else 0.0
        ring = (
            [0.20, 0.70 + 0.25 * inten, 1.0, 1.0]
            if on
            else [0.18, 0.18, 0.20, 1.0]
        )
        for s in getattr(self, "_ring_shapes", []) or []:
            try:
                s.material.set_base_color(ring)
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
        was_on = bool(self.stove_on)
        self.stove_on = on
        if was_on and not on:
            self.turned_off_once = True
        self._set_burner_glow(on)

    # ---------------------------------------------------------------- pan / ball physics
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

    def _pan_up_dot(self) -> float:
        """World +Z component of the pan bowl normal (1 = flat, 0 = vertical)."""
        if self.skillet is None:
            return 1.0
        R = self.skillet.get_pose().to_transformation_matrix()[:3, :3]
        up = R @ self._pan_up_local
        return float(up @ np.array([0.0, 0.0, 1.0]))

    def _capture_ball_offsets(self) -> None:
        if self.skillet is None:
            return
        pan_pose = self.skillet.get_pose()
        self._ball_offsets = []
        for ball in self.meatballs:
            self._ball_offsets.append(pan_pose.inv() * ball.get_pose())

    def _freeze_balls_to_pan(self) -> None:
        for rigid in self._ball_rigids:
            if rigid is None:
                continue
            try:
                rigid.set_disable_gravity(True)
                rigid.set_kinematic(True)
                rigid.set_linear_velocity(np.zeros(3))
                rigid.set_angular_velocity(np.zeros(3))
            except Exception:
                pass

    def _release_balls_physics(self) -> None:
        """Drop soft weld — pieces fall / slide under PhysX gravity."""
        if self._balls_released:
            return
        self._balls_released = True
        plate = np.asarray(self.plate_xy, dtype=float)
        pan_xy = (
            np.asarray(self.skillet.get_pose().p[:2], dtype=float)
            if self.skillet is not None
            else plate
        )
        over_plate = float(np.linalg.norm(pan_xy - plate)) < self.plate_inner_r + 0.06
        for i, (ball, rigid) in enumerate(zip(self.meatballs, self._ball_rigids)):
            if rigid is None:
                continue
            try:
                rigid.set_kinematic(False)
                rigid.set_disable_gravity(False)
                p = np.asarray(ball.get_pose().p, dtype=float)
                if self._pour_armed and over_plate:
                    # Drop through the plate mouth so gravity seats them.
                    ang = 2.0 * np.pi * i / max(1, len(self.meatballs))
                    r = 0.35 * self.plate_inner_r
                    drop = sapien.Pose(
                        [
                            float(plate[0] + r * np.cos(ang)),
                            float(plate[1] + r * np.sin(ang)),
                            float(self.plate_top_z + 0.04),
                        ],
                        list(ball.get_pose().q),
                    )
                    self._set_entity_pose(ball, drop)
                    rigid.set_kinematic(False)
                    rigid.set_disable_gravity(False)
                    rigid.set_linear_velocity(np.array([0.0, 0.0, -0.25]))
                else:
                    toward = plate - p[:2]
                    dist = float(np.linalg.norm(toward))
                    if dist > 1e-4:
                        v_xy = 0.10 * toward / dist
                        rigid.set_linear_velocity(
                            np.array([v_xy[0], v_xy[1], -0.05])
                        )
                    else:
                        rigid.set_linear_velocity(np.zeros(3))
                rigid.set_angular_velocity(np.zeros(3))
            except Exception:
                pass
        print(
            f"[serve_dinner] meatballs released (pan_up_dot={self._pan_up_dot():.3f} "
            f"pour_armed={self._pour_armed} over_plate={over_plate})"
        )

    def _sync_balls_to_pan(self) -> None:
        if self._balls_released or self.skillet is None:
            return
        pan_pose = self.skillet.get_pose()
        for ball, offset in zip(self.meatballs, self._ball_offsets):
            self._set_entity_pose(ball, pan_pose * offset)

    def _sync_sauce_to_pan(self) -> None:
        if self._sauce_entity is None or self._sauce_offset is None or self.skillet is None:
            return
        pose = self.skillet.get_pose() * self._sauce_offset
        try:
            self._sauce_entity.set_pose(pose)
        except Exception:
            pass

    def _ee_pose(self, arm: ArmTag) -> sapien.Pose:
        p = self.get_arm_pose(str(arm))
        return sapien.Pose(list(p[:3]), list(p[3:7]))

    def _weld_pan_to_ee(self, arm: ArmTag) -> None:
        if self.skillet is None:
            return
        if self._pan_rigid is not None:
            try:
                self._pan_rigid.set_disable_gravity(True)
                self._pan_rigid.set_kinematic(True)
            except Exception:
                pass
        self._pan_weld_offset = self._ee_pose(arm).inv() * self.skillet.get_pose()
        self._pan_welded = True
        self._ignore_pan_robot_collision()
        self._capture_ball_offsets()
        if self._sauce_entity is not None:
            self._sauce_offset = self.skillet.get_pose().inv() * self._sauce_entity.get_pose()

    def _release_pan_weld(self) -> None:
        self._pan_welded = False
        self._pan_weld_offset = None
        if self._pan_rigid is not None:
            try:
                self._pan_rigid.set_kinematic(False)
                self._pan_rigid.set_disable_gravity(False)
            except Exception:
                pass

    def _sync_pan_to_ee(self) -> None:
        if not self._pan_welded or self._pan_weld_offset is None:
            return
        pose = self._ee_pose(self.arm) * self._pan_weld_offset
        self._set_entity_pose(self.skillet, pose)

    def _set_pan_pose_keep_weld(self, pose: sapien.Pose) -> None:
        self._set_entity_pose(self.skillet, pose)
        if self._pan_welded:
            self._pan_weld_offset = self._ee_pose(self.arm).inv() * self.skillet.get_pose()
        self._sync_sauce_to_pan()

    def _ignore_pan_robot_collision(self) -> None:
        ignore_bit, ignore_id = self.PAN_IGNORE_BIT, self.PAN_IGNORE_ID
        ents = [self.skillet.actor]
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

    def _ball_on_plate(self, ball: Any) -> bool:
        p = np.asarray(ball.get_pose().p, dtype=float)
        dxy = float(np.linalg.norm(p[:2] - np.asarray(self.plate_xy)))
        return (
            dxy < self.plate_inner_r
            and float(p[2]) > self.plate_top_z - 0.02
            and float(p[2]) < self.plate_top_z + 0.10
        )

    def _check_balls_fallen(self) -> None:
        """Mark failure if a meatball left the pan and is not on / heading to the plate."""
        if not self._balls_released:
            return
        plate_xy = np.asarray(self.plate_xy, dtype=float)
        for ball in self.meatballs:
            if self._ball_on_plate(ball):
                continue
            p = np.asarray(ball.get_pose().p, dtype=float)
            # Still in flight above the plate / pan — wait.
            if float(p[2]) > self.plate_top_z + 0.03:
                continue
            if self.skillet is not None:
                bowl = np.asarray(self.skillet.get_functional_point(0)[:3], dtype=float)
                d_bowl = float(np.linalg.norm(p[:2] - bowl[:2]))
                if d_bowl < self.bowl_inner_r + 0.02 and float(p[2]) > float(bowl[2]) - 0.02:
                    continue
            # On / near the table and clear of the plate → spilled.
            if float(p[2]) < self.table_top + 0.06:
                d_plate = float(np.linalg.norm(p[:2] - plate_xy))
                if d_plate > self.plate_inner_r + 0.03:
                    self._balls_fallen = True
                    return
            if float(p[2]) < self.table_top - 0.05:
                self._balls_fallen = True
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

        self._sync_pan_to_ee()
        self._sync_sauce_to_pan()

        # Soft friction only while pouring. Carry keeps meatballs welded.
        if not self._balls_released:
            if self._pour_armed and self._pan_up_dot() < self.tilt_hold_dot:
                self._release_balls_physics()
            else:
                self._sync_balls_to_pan()

        self._check_balls_fallen()

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

    def _turn_knob_off(self) -> None:
        """Reach the front knob and twist the stove off (starts already on)."""
        arm = self.arm
        start_angle = -np.pi / 2  # ON
        end_angle = 0.0  # OFF
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
        self._set_stove(False)
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

    def _grasp_pan(self) -> bool:
        arm = self.arm
        self.plan_success = True
        self.move(self.open_gripper(arm))
        self.move(
            self.grasp_actor(
                self.skillet,
                arm_tag=arm,
                pre_grasp_dis=0.10,
                grasp_dis=0.0,
            )
        )
        if not self.plan_success:
            # Fallback: approach the handle contact from above.
            self.plan_success = True
            try:
                cp = self.skillet.get_contact_point(0, "pose")
                cxyz = np.asarray(cp.p if hasattr(cp, "p") else cp[:3], dtype=float)
            except Exception:
                cxyz = np.asarray(self.skillet.get_pose().p, dtype=float)
            hover = cxyz.copy()
            hover[2] = max(float(cxyz[2]) + 0.14, self.table_top + 0.20)
            self.move(
                (arm, [Action(arm, "move", target_pose=self._top_down_pose(hover))])
            )
            if not self.plan_success:
                return False
            pinch = cxyz.copy()
            pinch[2] = float(cxyz[2]) + 0.01
            self.move(
                (arm, [Action(arm, "move", target_pose=self._top_down_pose(pinch))])
            )
            if not self.plan_success:
                return False
            self.move(self.close_gripper(arm, pos=0.0))

        self._weld_pan_to_ee(arm)
        self._flatten_pan()
        self._sync_balls_to_pan()
        self._sync_sauce_to_pan()
        return True

    def _flatten_pan(self) -> None:
        """Keep the bowl level while welded (grasp can tip it a bit)."""
        if self.skillet is None:
            return
        p = np.asarray(self.skillet.get_pose().p, dtype=float)
        self._set_pan_pose_keep_weld(
            sapien.Pose(p.tolist(), list(self.SKILLET_BASE_QPOS))
        )
        if not self._balls_released:
            self._sync_balls_to_pan()
        self._sync_sauce_to_pan()

    def _carry_pan_level(self, target_xy, z: float) -> None:
        """Translate the welded pan while keeping the bowl flat."""
        arm = self.arm
        bp = np.asarray(self.skillet.get_pose().p, dtype=float)
        dx = float(target_xy[0] - bp[0])
        dy = float(target_xy[1] - bp[1])
        dz = float(z - bp[2])
        steps = 4
        for _ in range(1, steps + 1):
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
            self._sync_pan_to_ee()
            self._flatten_pan()
            if self._balls_released:
                break

    def _pour_onto_plate(self) -> None:
        """Tip the pan over the plate so soft friction breaks and meatballs fall in."""
        arm = self.arm
        plate = np.asarray(self.plate_xy, dtype=float)
        hover_z = float(self.plate_top_z) + 0.10
        # Hover centered over the plate, bowl still level.
        self._carry_pan_level(plate, hover_z)
        self._flatten_pan()

        self._pour_armed = True
        tip_steps = 10
        tip_max = float(self.pour_tip_rad)
        # Tip about +X so the near (−Y) rim drops toward the plate / robot.
        for i in range(1, tip_steps + 1):
            frac = i / tip_steps
            tip = tip_max * frac
            tip_q = qmult(
                euler2quat(float(tip), 0.0, 0.0, axes="sxyz"),
                np.array(self.SKILLET_BASE_QPOS, dtype=float),
            )
            p = np.array(
                [float(plate[0]), float(plate[1]), hover_z - 0.03 * frac],
                dtype=float,
            )
            self._set_pan_pose_keep_weld(sapien.Pose(p.tolist(), tip_q.tolist()))
            if not self._balls_released:
                self._sync_balls_to_pan()
            self._sync_sauce_to_pan()

            pour_q = qmult(
                euler2quat(float(tip), 0.0, 0.0, axes="sxyz"),
                np.array(GRASP_DIRECTION_DIC["top_down"], dtype=float),
            )
            self.move(
                self.move_by_displacement(
                    arm,
                    x=0.0,
                    y=-0.003,
                    z=-0.004,
                    quat=pour_q.tolist(),
                    move_axis="world",
                )
            )
            self.plan_success = True
            self._set_pan_pose_keep_weld(sapien.Pose(p.tolist(), tip_q.tolist()))

            if frac >= 0.40 and not self._balls_released:
                self._release_balls_physics()
            self._idle_steps(5)

        if not self._balls_released:
            self._release_balls_physics()

        self._idle_steps(
            100,
            until=lambda: all(self._ball_on_plate(b) for b in self.meatballs),
        )
        # Freeze balls that landed so moving the empty pan can't knock them off.
        for ball, rigid in zip(self.meatballs, self._ball_rigids):
            if rigid is None or not self._ball_on_plate(ball):
                continue
            try:
                rigid.set_linear_velocity(np.zeros(3))
                rigid.set_angular_velocity(np.zeros(3))
                rigid.set_kinematic(True)
                rigid.set_disable_gravity(True)
            except Exception:
                pass

        self._pour_armed = False
        # Untilt and park the pan clear of the plate.
        flat_p = np.asarray(self.skillet.get_pose().p, dtype=float).copy()
        flat_p[1] = float(self.burner_xy[1]) - 0.05
        flat_p[2] = self.table_top + 0.16
        self._set_pan_pose_keep_weld(
            sapien.Pose(flat_p.tolist(), list(self.SKILLET_BASE_QPOS))
        )
        self.move(
            self.move_by_displacement(
                arm,
                x=float(self.burner_xy[0] - flat_p[0]) * 0.4,
                y=float(self.burner_xy[1] - flat_p[1]) * 0.4,
                z=0.02,
                quat=list(GRASP_DIRECTION_DIC["top_down"]),
                move_axis="world",
            )
        )
        self.plan_success = True

    def play_once(self) -> dict[str, Any]:
        arm = self.arm
        self.plan_success = True

        # 1) Stove is already on — turn it off first.
        self._turn_knob_off()
        if self.stove_on:
            print("[serve_dinner] failed to turn stove off")
            self.plan_success = False
            return self.info

        # 2) Grasp the pan (meatballs soft-welded until tipped).
        if not self._grasp_pan():
            self.plan_success = False
            return self.info

        self.move(self.move_by_displacement(arm, z=0.12, move_axis="arm"))
        self._sync_pan_to_ee()
        self._flatten_pan()
        if self._balls_released or self._balls_fallen:
            print("[serve_dinner] meatballs fell during lift — fail")
            self.plan_success = False
            return self.info

        # 3) Tip over the plate.
        self._pour_onto_plate()
        if self._balls_fallen or not all(self._ball_on_plate(b) for b in self.meatballs):
            if not all(self._ball_on_plate(b) for b in self.meatballs):
                self._idle_steps(40)
            if self._balls_fallen or not all(
                self._ball_on_plate(b) for b in self.meatballs
            ):
                n_on = sum(self._ball_on_plate(b) for b in self.meatballs)
                print(
                    f"[serve_dinner] pour incomplete fallen={self._balls_fallen} "
                    f"on_plate={n_on}/{len(self.meatballs)}"
                )
                self.plan_success = False
                return self.info

        # Set the pan down away from the plate.
        self._carry_pan_level(
            np.array(
                [float(self.burner_xy[0]), float(self.burner_xy[1]) - 0.08],
                dtype=float,
            ),
            self.table_top + 0.06,
        )
        self._release_pan_weld()
        self.move(self.open_gripper(arm))
        self._idle_steps(12)

        if self.check_success():
            self.plan_success = True

        self.info["info"] = {
            "{A}": f"106_skillet/base{self.skillet_id}",
            "{B}": "003_plate/base0",
            "{C}": "cooking_range",
            "{D}": "stove_knob",
            "{E}": "meatballs",
            "{a}": str(arm),
        }
        return self.info

    def check_success(self) -> bool:
        if not getattr(self, "_loaded", False):
            return False
        if self.stove_on or not self.turned_off_once:
            return False
        if self._balls_fallen:
            return False
        if not self.meatballs:
            return False
        if not all(self._ball_on_plate(b) for b in self.meatballs):
            return False
        return True

    def get_obs(self) -> dict[str, Any]:
        obs = super().get_obs()
        n_on = (
            sum(1 for b in self.meatballs if self._ball_on_plate(b))
            if self.meatballs
            else 0
        )
        obs["serve_dinner"] = {
            "stove_on": bool(self.stove_on),
            "turned_off_once": bool(self.turned_off_once),
            "balls_released": bool(self._balls_released),
            "balls_fallen": bool(self._balls_fallen),
            "pan_up_dot": float(self._pan_up_dot()),
            "n_meatballs": int(len(self.meatballs)),
            "n_on_plate": int(n_on),
        }
        return obs
