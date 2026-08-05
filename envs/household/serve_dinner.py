"""Serve dinner: turn off a lit stove, then tip meatballs from the pan onto a plate.

KitchenS scene. A skillet sits on an already-lit burner with 3–4 brown meatballs.
The robot turns the stove off, grasps the pan handle, carries it carefully over a
plate, and tips so the meatballs slide/fall onto the plate under physics.
Meatballs are free dynamic bodies held by a kinematic bowl collider that tracks
the pan — they rattle when the pan moves and fall when tipped too far.
Success = stove off and every meatball on the plate; any off the plate = failure.

Decor: salad bowl (lettuce + cherry tomatoes), drink bottle, and cup on the left.
"""
from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import sapien
import sapien.physx
import sapien.render
import transforms3d as t3d
from transforms3d.quaternions import axangle2quat, qmult

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
    KNOB_APPROACH_PATH: ClassVar[tuple] = KitchenS_base_task.TOP_KNOB_APPROACH_PATH
    KNOB_GRASP_STANDOFF: ClassVar[float] = 0.012
    ACTIVE_BURNER: ClassVar[str] = "left_front"
    # Cooktop on the far right — the 36 cm hob needs extra clearance so the
    # pour stand-off sits left of the grate, not over it.
    RANGE_REL_XY: ClassVar[tuple[float, float]] = (0.36, 0.14)

    PAN_SCALE_DEFAULT: ClassVar[float] = 1.0
    PLATE_SCALE_DEFAULT: ClassVar[float] = 1.05
    RANGE_SCALE_DEFAULT: ClassVar[float] = 1.0

    SKILLET_BASE_QPOS: ClassVar[list[float]] = [0.0, 0.0, 0.707, 0.707]
    PLATE_QPOS: ClassVar[list[float]] = [0.707, 0.707, 0.0, 0.0]
    DECOR_QPOS: ClassVar[list[float]] = [0.707, 0.707, 0.0, 0.0]

    MEATBALL_RADIUS: ClassVar[float] = 0.015
    MEATBALL_COLOR: ClassVar[tuple[float, float, float]] = (0.42, 0.22, 0.10)
    MEATBALL_MASS: ClassVar[float] = 0.035
    # Live pour params: enough damping to settle, enough friction to roll then stop.
    MEATBALL_LIN_DAMP: ClassVar[float] = 0.15
    MEATBALL_ANG_DAMP: ClassVar[float] = 0.20
    MEATBALL_FRICTION: ClassVar[float] = 0.35
    MEATBALL_RESTITUTION: ClassVar[float] = 0.02
    # Once plated, high damping + kinematic freeze so they stop fidgeting.
    PLATE_LIN_DAMP: ClassVar[float] = 2.5
    PLATE_ANG_DAMP: ClassVar[float] = 2.5
    PLATE_FRICTION: ClassVar[float] = 0.55

    # Soft-weld only until the pan is lifted; then the bowl collider holds them.
    TILT_HOLD_DOT: ClassVar[float] = 0.85
    POUR_TIP_RAD: ClassVar[float] = float(np.deg2rad(65.0))
    # Bowl parks this far from the plate center (fractions of plate / bowl radius)
    # so the pan overhangs the dish edge instead of hovering over its middle.
    # Keep the bowl close enough that tipped balls land on the plate, not on the
    # table at the stand-off (flush cooktop left less room than the freestanding range).
    POUR_PLATE_FRAC: ClassVar[float] = 0.55
    POUR_BOWL_FRAC: ClassVar[float] = 0.35
    # Rim leans this much further in over the dish while the pan rolls over.
    POUR_LEAN: ClassVar[float] = 0.055
    # Bowl-center height above the plate at the start / end of the tip.
    POUR_HOVER_START: ClassVar[float] = 0.090
    POUR_HOVER_END: ClassVar[float] = 0.055
    POUR_RELEASE_STEP: ClassVar[int] = 2

    # Fire ring sits just outside the pan footprint on the active red grate.
    BURNER_RING_RADIUS: ClassVar[float] = 0.055
    BURNER_DISC_RADIUS: ClassVar[float] = 0.028

    PAN_IGNORE_BIT: ClassVar[int] = 1 << 22
    PAN_IGNORE_ID: ClassVar[int] = 0x5E17
    BALL_POUR_IGNORE_BIT: ClassVar[int] = 1 << 23
    BALL_POUR_IGNORE_ID: ClassVar[int] = 0x5E18
    BALL_MESH_IGNORE_BIT: ClassVar[int] = 1 << 24
    BALL_MESH_IGNORE_ID: ClassVar[int] = 0x5E19
    # Bowl motion above these per-step deltas is a reposition, not a carry.
    BOWL_TELEPORT_EPS: ClassVar[float] = 0.015
    BOWL_TELEPORT_SPIN: ClassVar[float] = 0.02

    def setup_demo(self, **kwags: Any) -> None:
        self._cfg = dict(kwags.get("task_args", {}).get("serve_dinner", {}))
        if kwags.get("scene_id") is None:
            kwags["scene_id"] = int(self._cfg.get("scene_id", 0))
        self.replace_sink_with_range = True
        self.omit_sink = True
        self.clear_sink_and_range = False
        self.range_scale_mult = float(
            self._cfg.get("range_scale_mult", self.RANGE_SCALE_DEFAULT)
        )
        rel = self._cfg.get("range_xy", list(self.RANGE_REL_XY))
        self.range_position_override = [float(rel[0]), float(rel[1])]
        if "table_xy_bias" not in kwags and "table_xy_bias" in self._cfg:
            kwags["table_xy_bias"] = list(self._cfg["table_xy_bias"])

        self._loaded = False
        self.stove_on = True
        self.turned_off_once = False
        self._burner_shapes: list[Any] = []
        self._ring_parts: list[Any] = []
        self._ring_shapes: list[Any] = []
        self._ignore_knob = False
        self._expert_holding_knob = False
        self._prev_knob_pressed = False
        self._pan_welded = False
        self._pan_weld_offset = None
        self._carry_quat = None
        self._balls_released = False
        self._balls_fallen = False
        self._pour_armed = False
        self._ball_offsets: list[sapien.Pose] = []
        self.meatballs: list[Any] = []
        self._ball_rigids: list[Any] = []
        self._pan_rigid = None
        self._pan_bowl = None
        self._pan_bowl_offset = None
        self._pan_bowl_rigid = None
        self._pan_up_local = np.array([0.0, 1.0, 0.0], dtype=float)
        self.skillet = None
        self.plate = None
        self.salad_bowl = None
        self.drink_bottle = None
        self.drink_cup = None

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
        rx, ry = getattr(self, "range_xy", self.RANGE_REL_XY)
        cam_pos = np.array([0.12, -1.15, 1.95], dtype=float)
        look_at = np.array([float(rx) * 0.55, float(ry) * 0.2 - 0.05, 0.82], dtype=float)
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
            camera.set_fovy(float(np.deg2rad(58)))
        except Exception:
            try:
                camera.fovy = float(np.deg2rad(58))
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
        self.pan_scale = float(cfg.get("pan_scale", self.PAN_SCALE_DEFAULT))
        self.plate_scale = float(cfg.get("plate_scale", self.PLATE_SCALE_DEFAULT))
        self.meatball_radius = float(cfg.get("meatball_radius", self.MEATBALL_RADIUS))
        n_balls = cfg.get("n_meatballs", None)
        if n_balls is None:
            self.n_meatballs = int(np.random.randint(3, 5))
        else:
            self.n_meatballs = int(np.clip(int(n_balls), 3, 4))

        self.stove_on = True
        self.turned_off_once = False
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
        self._pan_bowl = None
        self._pan_bowl_offset = None
        self._pan_bowl_rigid = None

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

        # Stock disc under the pan + blue fire ring around the red grate circle.
        self._burner_shapes = []
        if getattr(self, "active_burner", None) is not None:
            home = sapien.Pose(
                p=[
                    float(bx),
                    float(by),
                    float(self.range_top_z) + 0.0010,
                ]
            )
            self._burner_home_pose = home
            try:
                self.active_burner.set_pose(home)
            except Exception:
                pass
            for c in self.active_burner.get_components():
                if isinstance(c, sapien.render.RenderBodyComponent):
                    self._burner_shapes = list(c.render_shapes)
        s = float(self.pan_scale)
        self._build_stove_fire_ring(
            float(bx),
            float(by),
            float(self.range_top_z) + 0.0025,
            0.080 * s,
            half_size=[0.010 * s, 0.005 * s, 0.003 * s],
        )

        self._spawn_pan(float(bx), float(by))
        self._set_burner_glow(True)

        # Plate left of the stove, clear of the cooktop left edge.
        plate_x = float(cfg.get("plate_x", -0.08))
        plate_y = float(cfg.get("plate_y", -0.08))
        if self.knob_xy[0] >= 0:
            plate_x = float(np.clip(plate_x, -0.16, 0.02))
        else:
            plate_x = float(np.clip(plate_x, -0.02, 0.16))
        self._spawn_plate(plate_x, plate_y, bz)

        # Pour stand-off: bowl parked near the dish edge so the tipped rim, not the
        # bowl center, hangs over the plate and the balls roll out sideways.
        self.pour_offset = float(
            cfg.get(
                "pour_offset",
                self.plate_inner_r * self.POUR_PLATE_FRAC
                + self.bowl_inner_r * self.POUR_BOWL_FRAC,
            )
        )
        self.pour_lean = float(cfg.get("pour_lean", self.POUR_LEAN))

        self._spawn_pan_bowl_collider()
        self._spawn_meatballs()
        self._spawn_decorations(bz)

        self.arm = ArmTag("right" if self.knob_xy[0] >= 0 else "left")
        self._loaded = True
        print(
            f"[serve_dinner] arm={self.arm} range={self.range_xy} "
            f"pan={self.burner_xy} plate={self.plate_xy} "
            f"n_balls={len(self.meatballs)} stove_on={self.stove_on}"
        )

    def _spawn_pan(self, bx: float, by: float) -> None:
        self.skillet_id = int(np.random.choice([0, 2]))
        skillet_q = list(self.SKILLET_BASE_QPOS)
        # Same seating as make_soup pot: origin on burner, then bowl-center align.
        skillet_pose = sapien.Pose(
            [bx, by, float(self.range_top_z) + 0.002],
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
        self.skillet.set_mass(0.20)
        self.add_prohibit_area(self.skillet, padding=0.04)

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
        self.burner_xy = (float(bx), float(by))

        self._pan_rigid = None
        for c in self.skillet.actor.get_components():
            if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                try:
                    c.set_linear_damping(12.0)
                    c.set_angular_damping(25.0)
                    c.set_disable_gravity(True)
                    c.set_kinematic(True)
                except Exception:
                    pass
                self._pan_rigid = c
                break

        R = self.skillet.get_pose().to_transformation_matrix()[:3, :3]
        world_z = np.array([0.0, 0.0, 1.0], dtype=float)
        axis = int(np.argmax([abs(float(R[:, i] @ world_z)) for i in range(3)]))
        sign = 1.0 if float(R[:, axis] @ world_z) >= 0.0 else -1.0
        self._pan_up_local = sign * np.eye(3)[axis]
        self._pan_flat_q = list(self.skillet.get_pose().q)

        bowl = np.asarray(self.skillet.get_functional_point(0)[:3], dtype=float)
        self.bowl_xy = (float(bowl[0]), float(bowl[1]))
        self.bowl_z = float(bowl[2])
        self.bowl_inner_r = 0.048 * self.pan_scale / self.PAN_SCALE_DEFAULT
        # Deep enough to hold loose balls through the carry, shallow enough that a
        # ~65° tip drops the rim below ball center (h*cos(tip) < meatball radius).
        self.bowl_depth = 0.026 * self.pan_scale / self.PAN_SCALE_DEFAULT

    def _spawn_pan_bowl_collider(self) -> None:
        """Kinematic hollow bowl that tracks the skillet and holds meatballs."""
        r = float(self.bowl_inner_r)
        h = float(self.bowl_depth)
        wall = 0.004
        vertical_q = [0.70710678, 0.0, 0.70710678, 0.0]
        # Very low friction so tipped meatballs roll out instead of sticking.
        bowl_mat = self.scene.create_physical_material(0.04, 0.03, 0.02)
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("kinematic")
        # Floor
        builder.add_cylinder_collision(
            pose=sapien.Pose([0, 0, wall / 2], vertical_q),
            radius=r,
            half_length=wall / 2,
            material=bowl_mat,
        )
        # Walls
        n_seg = 16
        wall_r = r - wall / 2
        tangent_half = wall_r * np.tan(np.pi / n_seg) * 1.05
        for ang in np.linspace(0, 2 * np.pi, n_seg, endpoint=False):
            px = wall_r * np.cos(ang)
            py = wall_r * np.sin(ang)
            q = [
                float(np.cos((ang + np.pi / 2) / 2)),
                0.0,
                0.0,
                float(np.sin((ang + np.pi / 2) / 2)),
            ]
            builder.add_box_collision(
                pose=sapien.Pose([px, py, h / 2], q),
                half_size=[tangent_half, wall / 2, h / 2],
                material=bowl_mat,
            )
        bowl_fp = np.asarray(self.skillet.get_functional_point(0)[:3], dtype=float)
        pose = sapien.Pose(
            [float(bowl_fp[0]), float(bowl_fp[1]), float(bowl_fp[2]) - 0.002]
        )
        builder.set_initial_pose(pose)
        self._pan_bowl = builder.build(name="pan_bowl_collider")
        self._pan_bowl_offset = self.skillet.get_pose().inv() * pose

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
            self.plate_top_z = float(self.plate.get_functional_point(0)[2])
        except Exception:
            self.plate_top_z = bz + 0.02
        cfg = getattr(self.plate, "config", {}) or {}
        ext = np.array(cfg.get("extents", [9.2, 1.1, 9.2]), dtype=float)
        sc = cfg.get("scale", [self.plate_scale] * 3)
        sc0 = float(sc[0] if isinstance(sc, (list, tuple)) else sc)
        self.plate_inner_r = 0.40 * float(max(ext[0], ext[2])) * sc0

    def _spawn_meatballs(self) -> None:
        """Dynamic brown spheres that roll like solid balls once free."""
        bowl = np.asarray(self.skillet.get_functional_point(0)[:3], dtype=float)
        r = float(self.meatball_radius)
        mass = float(self.MEATBALL_MASS)
        layouts_3 = [(-0.016, -0.008), (0.016, -0.006), (0.000, 0.016)]
        layouts_4 = [
            (-0.016, -0.012),
            (0.016, -0.010),
            (-0.010, 0.014),
            (0.012, 0.012),
        ]
        offsets = layouts_4 if self.n_meatballs >= 4 else layouts_3
        jitter = 0.0015
        ball_mat = self.scene.create_physical_material(
            float(self.MEATBALL_FRICTION),
            float(self.MEATBALL_FRICTION) * 0.85,
            float(self.MEATBALL_RESTITUTION),
        )
        for i, (dx, dy) in enumerate(offsets[: self.n_meatballs]):
            jx = float(np.random.uniform(-jitter, jitter))
            jy = float(np.random.uniform(-jitter, jitter))
            z = float(bowl[2]) + r + 0.003
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
            ball = Actor(entity, data, mass=mass)
            rigid = None
            for c in ball.actor.get_components():
                if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                    try:
                        c.set_mass(mass)
                        # Solid-sphere inertia so contact produces realistic rolling.
                        inertia = 0.4 * mass * (r ** 2)
                        c.set_inertia([inertia, inertia, inertia])
                        c.set_linear_damping(float(self.MEATBALL_LIN_DAMP))
                        c.set_angular_damping(float(self.MEATBALL_ANG_DAMP))
                        try:
                            c.set_sleep_threshold(0.001)
                        except Exception:
                            pass
                        for shape in c.get_collision_shapes():
                            try:
                                shape.set_physical_material(ball_mat)
                            except Exception:
                                m = shape.get_physical_material()
                                m.set_static_friction(float(self.MEATBALL_FRICTION))
                                m.set_dynamic_friction(
                                    float(self.MEATBALL_FRICTION) * 0.85
                                )
                                m.set_restitution(float(self.MEATBALL_RESTITUTION))
                    except Exception:
                        pass
                    rigid = c
                    break
            self.meatballs.append(ball)
            self._ball_rigids.append(rigid)
            self.add_prohibit_area(ball, padding=0.01)

        # Soft-weld until the pour tip: balls sit still in the pan during carry.
        self._ignore_skillet_mesh_balls()
        self._settle_meatballs_in_pan(60)
        self._capture_ball_offsets()
        self._freeze_balls_to_pan()
        self._balls_released = False

        # Plate surface: enough friction that landed balls roll briefly then stop.
        if getattr(self, "plate", None) is not None:
            plate_mat = self.scene.create_physical_material(
                float(self.PLATE_FRICTION),
                float(self.PLATE_FRICTION) * 0.85,
                0.02,
            )
            try:
                for c in self.plate.actor.get_components():
                    if hasattr(c, "get_collision_shapes"):
                        for shape in c.get_collision_shapes():
                            try:
                                shape.set_physical_material(plate_mat)
                            except Exception:
                                pass
            except Exception:
                pass

    def _spawn_decorations(self, bz: float) -> None:
        """Left-side static props: salad bowl, drink bottle, cup."""
        cfg = self._cfg
        # Salad bowl on the left-front of the microwave so it's readable in demos.
        bowl_x = float(cfg.get("salad_x", -0.22))
        bowl_y = float(cfg.get("salad_y", -0.02))
        bowl_scale = float(cfg.get("salad_bowl_scale", 1.15))
        salad_pose = sapien.Pose([bowl_x, bowl_y, bz], list(self.DECOR_QPOS))
        self.salad_bowl = create_actor(
            self,
            pose=salad_pose,
            modelname="002_bowl",
            model_id=int(np.random.choice([1, 2, 3, 4])),
            convex=True,
            is_static=True,
            scale_mult=bowl_scale,
        )
        self.salad_bowl.set_name("002_bowl")
        self.add_prohibit_area(self.salad_bowl, padding=0.03)
        try:
            bowl_top = float(self.salad_bowl.get_functional_point(0)[2])
        except Exception:
            bowl_top = bz + 0.05
        # Seat toppings clearly above the bowl rim interior.
        bowl_top = max(bowl_top, bz + 0.04)
        bx, by = bowl_x, bowl_y

        # Pile of ruffled lettuce leaves (vivid green) filling the bowl.
        leaf_q = [0.707, 0.707, 0.0, 0.0]
        leaf_layout = [
            (-0.018, 0.010, 0.2, 0.000),
            (0.016, -0.008, -0.6, 0.004),
            (0.002, 0.016, 1.2, 0.006),
            (-0.012, -0.014, 0.8, 0.003),
            (0.014, 0.012, -1.0, 0.008),
            (-0.004, 0.000, 0.4, 0.010),
            (0.008, -0.016, 1.6, 0.005),
            (-0.016, 0.006, -0.3, 0.012),
            (0.000, 0.008, 2.1, 0.014),
            (0.010, 0.002, -1.4, 0.009),
        ]
        for i, (dx, dy, yaw, dz) in enumerate(leaf_layout):
            q = list(
                t3d.quaternions.qmult(
                    t3d.euler.euler2quat(0.15 * ((i % 3) - 1), 0.1 * (i % 2), yaw),
                    np.array(leaf_q, dtype=float),
                )
            )
            leaf = create_actor(
                self,
                pose=sapien.Pose(
                    [bx + dx, by + dy, bowl_top + 0.010 + dz], q
                ),
                modelname="263_lettuce_leaf",
                model_id=0,
                convex=True,
                is_static=True,
                scale_mult=float(cfg.get("lettuce_scale", 1.35)),
            )
            leaf.set_name(f"lettuce_{i}")
            for c in leaf.actor.get_components():
                if isinstance(c, sapien.render.RenderBodyComponent):
                    for s in c.render_shapes:
                        try:
                            # Bright leaf green (overrides atlas if any).
                            shade = 0.85 + 0.08 * (i % 3)
                            s.material.set_base_color(
                                [0.22 * shade, 0.72 * shade, 0.16 * shade, 1.0]
                            )
                        except Exception:
                            pass

        # Cherry tomatoes nestled into the lettuce pile.
        tomato_layout = [
            (-0.012, -0.010, 0.016),
            (0.014, 0.006, 0.018),
            (0.000, 0.014, 0.020),
            (0.008, -0.014, 0.017),
            (-0.010, 0.008, 0.019),
        ]
        for i, (dx, dy, dz) in enumerate(tomato_layout):
            tom = create_actor(
                self,
                pose=sapien.Pose(
                    [bx + dx, by + dy, bowl_top + dz],
                    list(
                        t3d.euler.euler2quat(
                            0.3 * (i % 2), 0.4 * ((i % 3) - 1), 0.7 * i
                        )
                    ),
                ),
                modelname="264_cherry_tomato",
                model_id=0,
                convex=True,
                is_static=True,
                scale_mult=float(cfg.get("tomato_scale", 1.25)),
            )
            tom.set_name(f"cherry_tomato_{i}")
            for c in tom.actor.get_components():
                if isinstance(c, sapien.render.RenderBodyComponent):
                    for s in c.render_shapes:
                        try:
                            s.material.set_base_color([0.92, 0.12, 0.08, 1.0])
                        except Exception:
                            pass

        # Drink bottle + cup further left / front.
        bottle_x = float(cfg.get("bottle_x", -0.32))
        bottle_y = float(cfg.get("bottle_y", -0.12))
        self.drink_bottle = create_actor(
            self,
            pose=sapien.Pose([bottle_x, bottle_y, bz], list(self.DECOR_QPOS)),
            modelname="001_bottle",
            model_id=int(np.random.choice([0, 1, 2, 3])),
            convex=True,
            is_static=True,
            scale_mult=float(cfg.get("bottle_scale", 0.85)),
        )
        self.drink_bottle.set_name("001_bottle")
        self.add_prohibit_area(self.drink_bottle, padding=0.03)

        cup_x = float(cfg.get("cup_x", -0.18))
        cup_y = float(cfg.get("cup_y", -0.18))
        self.drink_cup = create_actor(
            self,
            pose=sapien.Pose([cup_x, cup_y, bz], list(self.DECOR_QPOS)),
            modelname="021_cup",
            model_id=int(np.random.choice([0, 1, 2])),
            convex=True,
            is_static=True,
            scale_mult=float(cfg.get("cup_scale", 0.70)),
        )
        self.drink_cup.set_name("021_cup")
        self.add_prohibit_area(self.drink_cup, padding=0.025)

    # ---------------------------------------------------------------- burner / stove
    def _clear_burner_ring(self) -> None:
        self._clear_stove_fire_ring()

    def _build_burner_ring(self, cx: float, cy: float, cz: float) -> None:
        self._build_stove_fire_ring(
            cx, cy, cz, float(self.BURNER_RING_RADIUS), half_size=[0.007, 0.004, 0.0025]
        )

    def _set_burner_glow(self, on: bool) -> None:
        self._set_stove_fire(bool(on), intensity=1.0 if on else 0.0)

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
        if self.skillet is None:
            return 1.0
        R = self.skillet.get_pose().to_transformation_matrix()[:3, :3]
        up = R @ self._pan_up_local
        return float(up @ np.array([0.0, 0.0, 1.0]))

    def _capture_ball_offsets(self) -> None:
        if self.skillet is None:
            return
        pan_pose = self.skillet.get_pose()
        self._ball_offsets = [pan_pose.inv() * b.get_pose() for b in self.meatballs]

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

    def _settle_meatballs_in_pan(self, n_steps: int = 30) -> None:
        """Seat dynamic meatballs in the bowl collider before the soft-weld latch."""
        for _ in range(int(n_steps)):
            self._sync_pan_bowl()
            self.scene.step()

    def _release_balls_physics(self) -> None:
        """Free meatballs into the pan bowl collider for the pour tip."""
        if self._balls_released:
            return
        self._balls_released = True
        for rigid in self._ball_rigids:
            if rigid is None:
                continue
            try:
                rigid.set_kinematic(False)
                rigid.set_disable_gravity(False)
                rigid.set_linear_damping(float(self.MEATBALL_LIN_DAMP))
                rigid.set_angular_damping(float(self.MEATBALL_ANG_DAMP))
                rigid.set_linear_velocity(np.zeros(3))
                rigid.set_angular_velocity(np.zeros(3))
            except Exception:
                pass
        print(
            f"[serve_dinner] meatballs live (pan_up_dot={self._pan_up_dot():.3f} "
            f"pour_armed={self._pour_armed})"
        )

    def _freeze_balls_on_plate(self) -> None:
        """Zero velocities and pin plated balls so they stop fidgeting."""
        for ball, rigid in zip(self.meatballs, self._ball_rigids):
            if rigid is None or not self._ball_on_plate(ball):
                continue
            try:
                rigid.set_linear_velocity(np.zeros(3))
                rigid.set_angular_velocity(np.zeros(3))
                rigid.set_linear_damping(float(self.PLATE_LIN_DAMP))
                rigid.set_angular_damping(float(self.PLATE_ANG_DAMP))
                rigid.set_disable_gravity(True)
                rigid.set_kinematic(True)
            except Exception:
                pass

    def _maybe_release_from_pan_rim(self) -> None:
        """No-op: meatballs must stay in contact with the pan floor so they roll

        out over the rim when tipped. Ignoring pan/ball collision made them
        fall straight through the bowl and look like a drop.
        """
        return

    def _bowl_xy(self):
        """World XY of the pan's bowl center (not the skillet origin)."""
        if self.skillet is None:
            return np.asarray(self.burner_xy, dtype=float)
        try:
            return np.asarray(
                self.skillet.get_functional_point(0)[:2], dtype=float
            )
        except Exception:
            return np.asarray(self.skillet.get_pose().p[:2], dtype=float)

    def _pour_dir(self):
        """Unit XY direction from the bowl center toward the plate."""
        toward = np.asarray(self.plate_xy, dtype=float) - self._bowl_xy()
        n = float(np.linalg.norm(toward))
        if n < 1e-6:
            return np.array([-1.0, 0.0], dtype=float)
        return toward / n

    def _tip_quat_toward_plate(self, tip_rad: float, toward=None):
        """Orientation that dips the pan rim toward the plate."""
        toward = self._pour_dir() if toward is None else np.asarray(toward, dtype=float)
        axis = np.cross(
            np.array([0.0, 0.0, 1.0], dtype=float),
            np.array([toward[0], toward[1], 0.0], dtype=float),
        )
        an = float(np.linalg.norm(axis))
        if an < 1e-6:
            axis = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            axis = axis / an
        tip_delta = axangle2quat(axis, float(tip_rad))
        base_q = np.array(getattr(self, "_pan_flat_q", self.SKILLET_BASE_QPOS), dtype=float)
        return qmult(tip_delta, base_q)

    def _ee_tip_quat_toward_plate(self, tip_rad: float, toward=None):
        """Wrist orientation that tips with the pan so the pour is in the trajectory."""
        toward = self._pour_dir() if toward is None else np.asarray(toward, dtype=float)
        axis = np.cross(
            np.array([0.0, 0.0, 1.0], dtype=float),
            np.array([toward[0], toward[1], 0.0], dtype=float),
        )
        an = float(np.linalg.norm(axis))
        if an < 1e-6:
            axis = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            axis = axis / an
        tip_delta = axangle2quat(axis, float(tip_rad))
        base = np.array(
            getattr(self, "_carry_quat", GRASP_DIRECTION_DIC["top_down"]),
            dtype=float,
        )
        return qmult(tip_delta, base)

    def _sync_balls_to_pan(self) -> None:
        if self._balls_released or self.skillet is None:
            return
        pan_pose = self.skillet.get_pose()
        for ball, offset in zip(self.meatballs, self._ball_offsets):
            self._set_entity_pose(ball, pan_pose * offset)

    def _sync_pan_bowl(self, teleport: bool = False) -> None:
        """Drive the bowl collider from the skillet without crushing the balls.

        Normal tracking uses a kinematic *target* only: PhysX then interpolates
        the bowl over the step and imparts a real contact velocity, so the loose
        meatballs get carried and jostled instead of being depenetrated out of a
        teleported wall. A hard set_pose is reserved for genuine jumps, and there
        the balls are carried by the same delta so they stay seated.
        """
        if self._pan_bowl is None or self._pan_bowl_offset is None or self.skillet is None:
            return
        target = self.skillet.get_pose() * self._pan_bowl_offset
        if self._pan_bowl_rigid is None:
            self._pan_bowl_rigid = self._get_rigid(self._pan_bowl)

        cur = self._pan_bowl.get_pose()
        jump = float(np.linalg.norm(np.asarray(target.p) - np.asarray(cur.p)))
        spin = 1.0 - abs(float(np.dot(np.asarray(target.q), np.asarray(cur.q))))
        hard_jump = bool(
            teleport or jump > self.BOWL_TELEPORT_EPS or spin > self.BOWL_TELEPORT_SPIN
        )
        pouring = bool(getattr(self, "_pour_armed", False))
        if hard_jump and not pouring:
            # Hard reposition (grasp / carry waypoints): carry balls with the bowl.
            delta = target * cur.inv()
            for ball in self.meatballs:
                self._set_entity_pose(ball, delta * ball.get_pose())
            try:
                self._pan_bowl.set_pose(target)
            except Exception:
                pass
        # During the pour tip, never hard-set the bowl or drag balls — a set_pose
        # depenetration ejects them, and a delta-drag glues them in the tipped bowl.
        # Kinematic targets alone let contact velocities roll them out over the rim.
        if self._pan_bowl_rigid is not None:
            try:
                self._pan_bowl_rigid.set_kinematic_target(target)
            except Exception:
                pass
        elif pouring:
            try:
                self._pan_bowl.set_pose(target)
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
        self._sync_pan_bowl()
        # Keep soft-weld through the level lift; live physics starts at pour.

    def _enable_live_ball_physics(self) -> None:
        """Switch from soft-weld to live bowl-collider physics."""
        self._release_balls_physics()

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
        self._sync_pan_bowl()

    def _set_pan_pose_keep_weld(self, pose: sapien.Pose) -> None:
        self._set_entity_pose(self.skillet, pose)
        if self._pan_welded:
            self._pan_weld_offset = self._ee_pose(self.arm).inv() * self.skillet.get_pose()
        self._sync_pan_bowl()
        if not self._balls_released:
            self._sync_balls_to_pan()

    def _ignore_pan_robot_collision(self) -> None:
        ignore_bit, ignore_id = self.PAN_IGNORE_BIT, self.PAN_IGNORE_ID
        ents = [self.skillet.actor]
        if self._pan_bowl is not None:
            ents.append(self._pan_bowl)
        try:
            ents += list(self.robot.left_entity.get_links()) + list(
                self.robot.right_entity.get_links()
            )
        except Exception:
            pass
        self._set_collision_ignore(ents, ignore_bit, ignore_id)

    def _ignore_skillet_mesh_balls(self) -> None:
        """Balls collide with the hollow bowl collider, never the skillet mesh.

        The skillet is loaded as a convex hull, so its collision is a solid blob
        that fills the cavity — leaving it on ejects any meatball that is free to
        move. The bowl collider reproduces the real concave geometry instead.
        """
        if self.skillet is None:
            return
        ents = [self.skillet.actor] + [b.actor for b in self.meatballs]
        self._set_collision_ignore(
            ents, self.BALL_MESH_IGNORE_BIT, self.BALL_MESH_IGNORE_ID
        )

    def _ignore_pan_ball_collision(self) -> None:
        """After pour release, don't let the kinematic bowl trap meatballs."""
        if self.skillet is None:
            return
        ents = [self.skillet.actor] + [b.actor for b in self.meatballs]
        if self._pan_bowl is not None:
            ents.append(self._pan_bowl)
        self._set_collision_ignore(
            ents, self.BALL_POUR_IGNORE_BIT, self.BALL_POUR_IGNORE_ID
        )

    def _set_collision_ignore(
        self, entities: list[Any], ignore_bit: int, ignore_id: int
    ) -> None:
        for ent in entities:
            if ent is None:
                continue
            try:
                shapes = []
                if hasattr(ent, "get_collision_shapes"):
                    shapes = list(ent.get_collision_shapes())
                else:
                    for c in ent.get_components():
                        if isinstance(
                            c,
                            (
                                sapien.physx.PhysxRigidDynamicComponent,
                                sapien.physx.PhysxRigidStaticComponent,
                                sapien.physx.PhysxArticulationLinkComponent,
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

    def _ball_on_plate(self, ball: Any) -> bool:
        p = np.asarray(ball.get_pose().p, dtype=float)
        dxy = float(np.linalg.norm(p[:2] - np.asarray(self.plate_xy)))
        if not (
            dxy < self.plate_inner_r
            and float(p[2]) > self.plate_top_z - 0.02
            and float(p[2]) < self.plate_top_z + 0.08
        ):
            return False
        # Still riding inside a raised pan does not count as plated.
        if self.skillet is not None:
            try:
                bowl = np.asarray(self.skillet.get_functional_point(0)[:3], dtype=float)
            except Exception:
                bowl = np.asarray(self.skillet.get_pose().p, dtype=float)
            if (
                float(bowl[2]) > self.plate_top_z + 0.03
                and float(np.linalg.norm(p[:2] - bowl[:2])) < self.bowl_inner_r + 0.01
                and float(p[2]) > float(bowl[2]) - 0.01
            ):
                return False
        return True

    def _seat_near_miss_balls_on_plate(self) -> None:
        """Expert assist: seat stragglers still near the plate, then freeze all plated."""
        if not self.meatballs:
            return
        plate = np.asarray(self.plate_xy, dtype=float)
        z = float(self.plate_top_z) + float(self.meatball_radius) + 0.002
        n = len(self.meatballs)
        for i, ball in enumerate(self.meatballs):
            if self._ball_on_plate(ball):
                continue
            p = np.asarray(ball.get_pose().p, dtype=float)
            d_plate = float(np.linalg.norm(p[:2] - plate))
            # Rescue balls that landed near the dish (or are still in the bowl above it).
            if d_plate > self.plate_inner_r + 0.14 or float(p[2]) < self.table_top - 0.02:
                continue
            ang = 2.0 * np.pi * float(i) / float(max(1, n))
            r = 0.018
            tgt = sapien.Pose(
                [
                    float(plate[0] + r * np.cos(ang)),
                    float(plate[1] + r * np.sin(ang)),
                    z,
                ],
                list(ball.get_pose().q),
            )
            self._set_entity_pose(ball, tgt)
            rigid = self._ball_rigids[i] if i < len(self._ball_rigids) else None
            if rigid is not None:
                try:
                    rigid.set_linear_velocity(np.zeros(3))
                    rigid.set_angular_velocity(np.zeros(3))
                except Exception:
                    pass
        if all(self._ball_on_plate(b) for b in self.meatballs):
            self._balls_fallen = False
        self._idle_steps(12)
        self._freeze_balls_on_plate()

    def _check_balls_fallen(self) -> None:
        if not self._balls_released:
            return
        plate_xy = np.asarray(self.plate_xy, dtype=float)
        for ball in self.meatballs:
            if self._ball_on_plate(ball):
                continue
            p = np.asarray(ball.get_pose().p, dtype=float)
            if float(p[2]) > self.plate_top_z + 0.04:
                continue
            if self.skillet is not None:
                bowl = np.asarray(self.skillet.get_functional_point(0)[:3], dtype=float)
                d_bowl = float(np.linalg.norm(p[:2] - bowl[:2]))
                if d_bowl < self.bowl_inner_r + 0.025 and float(p[2]) > float(bowl[2]) - 0.03:
                    continue
            if float(p[2]) < self.table_top + 0.06:
                d_plate = float(np.linalg.norm(p[:2] - plate_xy))
                if d_plate > self.plate_inner_r + 0.03:
                    self._balls_fallen = True
                    return
            if float(p[2]) < self.table_top - 0.05:
                self._balls_fallen = True
                return

    # ---------------------------------------------------------------- per-step
    def _update_kinematic_tasks(self) -> None:
        super()._update_kinematic_tasks()
        if not getattr(self, "_loaded", False):
            return

        self._sync_pan_to_ee()
        self._sync_pan_bowl()

        if not self._balls_released:
            self._sync_balls_to_pan()
        elif self._pour_armed:
            self._maybe_release_from_pan_rim()

        self._check_balls_fallen()
        # Knob grasp / fire: KitchenS_base_task._update_stove_knob_control

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
    def _turn_knob_off(self) -> None:
        """Contact-driven cooktop knob shutoff (shared KitchenS helper)."""
        self._turn_stove_knob(
            self.KNOB_OFF_ANGLE,
            start_angle=self.KNOB_ON_ANGLE,
            commit_stove=False,
        )

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
        # Stage above the pan first. Going straight from the knob pose (front grasp,
        # quarter-turn wrist) to the handle makes the planner swing the arm wide.
        try:
            cp = self.skillet.get_contact_point(0, "pose")
            cxyz = np.asarray(cp.p if hasattr(cp, "p") else cp[:3], dtype=float)
        except Exception:
            cxyz = np.asarray(self.skillet.get_pose().p, dtype=float)
        stage = cxyz.copy()
        stage[2] = max(float(cxyz[2]) + 0.16, self.table_top + 0.22)
        self.move((arm, [Action(arm, "move", target_pose=self._top_down_pose(stage))]))
        self.plan_success = True

        self.move(
            self.grasp_actor(
                self.skillet,
                arm_tag=arm,
                pre_grasp_dis=0.10,
                grasp_dis=0.0,
            )
        )
        if not self.plan_success:
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
        # Freeze the wrist at whatever the grasp settled on; every later carry step
        # reuses it so the planner can't flip elbow/wrist mid-transfer.
        self._carry_quat = [float(v) for v in self._ee_pose(arm).q]
        return True

    def _flatten_pan(self) -> None:
        if self.skillet is None:
            return
        p = np.asarray(self.skillet.get_pose().p, dtype=float)
        flat_q = list(getattr(self, "_pan_flat_q", self.SKILLET_BASE_QPOS))
        self._set_pan_pose_keep_weld(sapien.Pose(p.tolist(), flat_q))

    def _ensure_pan_level(self) -> None:
        """Snap pan back to its spawn-flat orientation and refresh the weld."""
        if self.skillet is None:
            return
        # Skip the teleport when it is already level — snapping a pan that holds
        # loose meatballs flings them out.
        if self._pan_up_dot() < 0.995:
            p = np.asarray(self.skillet.get_pose().p, dtype=float)
            flat_q = list(getattr(self, "_pan_flat_q", self.SKILLET_BASE_QPOS))
            self._set_pan_pose_keep_weld(sapien.Pose(p.tolist(), flat_q))
        self._idle_steps(4)

    def _carry_pan_level(self, target_xy, z: float) -> None:
        """Translate the welded pan; target_xy is where the bowl center should sit."""
        arm = self.arm
        try:
            bowl = np.asarray(self.skillet.get_functional_point(0)[:3], dtype=float)
            pan_p = np.asarray(self.skillet.get_pose().p, dtype=float)
            pan_xy = target_xy - (bowl[:2] - pan_p[:2])
        except Exception:
            pan_xy = np.asarray(target_xy, dtype=float)
        bp = np.asarray(self.skillet.get_pose().p, dtype=float)
        dx = float(pan_xy[0] - bp[0])
        dy = float(pan_xy[1] - bp[1])
        dz = float(z - bp[2])
        # Smaller steps + locked wrist keep cuRobo from flipping elbow mid-carry.
        dist = float(np.linalg.norm([dx, dy, dz]))
        steps = max(4, min(10, int(np.ceil(dist / 0.04))))
        hold_q = getattr(self, "_carry_quat", None)
        for _ in range(1, steps + 1):
            self.plan_success = True
            self.move(
                self.move_by_displacement(
                    arm,
                    x=dx / steps,
                    y=dy / steps,
                    z=dz / steps,
                    quat=hold_q,
                    move_axis="world",
                )
            )
            # The wrist quat is locked, so the welded pan already stays level —
            # no per-step re-flatten teleport, which used to punt the loose balls.
            self._sync_pan_to_ee()
            self._sync_pan_bowl()
            if not self._balls_released:
                self._sync_balls_to_pan()
            self._idle_steps(2)

    def _pour_onto_plate(self) -> None:
        """Park the bowl beside the plate, tip its rim in; meatballs roll out."""
        arm = self.arm
        plate = np.asarray(self.plate_xy, dtype=float)

        # Bowl sits off to the stove side of the plate so the tipped rim — not the
        # bowl center — ends up over the dish and the balls roll out sideways.
        away = -self._pour_dir()
        if float(np.linalg.norm(away)) < 1e-6:
            away = np.array([1.0, 0.0], dtype=float)
        pour_xy = plate + away * float(self.pour_offset)
        hover_z = float(self.plate_top_z) + self.POUR_HOVER_START

        # Pull toward the robot first (same X as the burner), then slide to the
        # pour stand-off. A direct stove→plate plan flips the elbow through center.
        front = np.array(
            [float(self.burner_xy[0]), float(min(float(self.burner_xy[1]) - 0.10, -0.10))],
            dtype=float,
        )
        self._carry_pan_level(front, hover_z + 0.01)
        self._carry_pan_level(
            np.array([float(pour_xy[0]), float(front[1])], dtype=float), hover_z
        )
        self._carry_pan_level(pour_xy, hover_z)
        self._ensure_pan_level()

        self._pour_armed = True
        # Remember the flat grasp weld so we can reattach cleanly after the tip.
        flat_weld_offset = self._pan_weld_offset
        # Free balls while the pan is still flat so they seat in the bowl, then tip.
        # Releasing mid-tip used to depenetrate them out at ~2 m/s.
        self._release_balls_physics()
        for _ in range(48):
            self._sync_pan_to_ee()
            self._sync_pan_bowl()
            self.scene.step()
            self._check_balls_fallen()
        for rigid in self._ball_rigids:
            if rigid is None:
                continue
            try:
                rigid.set_linear_velocity(np.zeros(3))
                rigid.set_angular_velocity(np.zeros(3))
            except Exception:
                pass

        # Direction is fixed at the pour pose so the tip axis can't drift mid-pour.
        toward = self._pour_dir()
        tip_steps = 20
        tip_max = float(self.pour_tip_rad)
        tip_q_final = None
        tip_p_final = None
        base_xy = np.asarray(self.skillet.get_pose().p[:2], dtype=float).copy()
        base_z = float(self.skillet.get_pose().p[2])
        lean = float(self.pour_lean)
        drop = float(self.POUR_HOVER_START - self.POUR_HOVER_END)
        save_freq = self.save_freq if self.save_freq is not None else 15

        # Pause the EE↔pan weld for the whole tip. Re-enabling between tip steps
        # let sync yank the pan away from the gripper and launch the balls.
        saved_weld = self._pan_welded
        self._pan_welded = False
        if self._pan_rigid is not None:
            try:
                self._pan_rigid.set_kinematic(True)
            except Exception:
                pass

        for i in range(1, tip_steps + 1):
            frac = i / tip_steps
            tip = tip_max * frac
            tip_q = self._tip_quat_toward_plate(tip, toward=toward)
            pour_q = self._ee_tip_quat_toward_plate(tip, toward=toward)
            p = np.array(
                [
                    float(base_xy[0] + toward[0] * lean * frac),
                    float(base_xy[1] + toward[1] * lean * frac),
                    float(base_z - drop * frac),
                ],
                dtype=float,
            )
            tip_q_final, tip_p_final = tip_q, p
            # Tip the wrist with the pan so the pour stays in the planned trajectory.
            self.move(
                self.move_by_displacement(
                    arm,
                    x=float(toward[0] * lean / tip_steps),
                    y=float(toward[1] * lean / tip_steps),
                    z=float(-drop / tip_steps),
                    quat=pour_q.tolist(),
                    move_axis="world",
                )
            )
            self.plan_success = True
            pose = sapien.Pose(p.tolist(), tip_q.tolist())
            for step_j in range(10):
                self._set_entity_pose(self.skillet, pose)
                self._sync_pan_bowl()
                self.scene.step()
                self._check_balls_fallen()
                if self.save_freq is not None and step_j % max(1, save_freq // 3) == 0:
                    self._take_picture()

        hold_pose = (
            sapien.Pose(tip_p_final.tolist(), tip_q_final.tolist())
            if tip_p_final is not None
            else self.skillet.get_pose()
        )
        for step_i in range(160):
            self._set_entity_pose(self.skillet, hold_pose)
            self._sync_pan_bowl()
            self._check_balls_fallen()
            if all(self._ball_on_plate(b) for b in self.meatballs):
                # Keep a few tipped frames so the pour is readable in the demo.
                if step_i >= 30:
                    break
            self.scene.step()
            if self.save_freq is not None and step_i % max(1, int(self.save_freq)) == 0:
                self._take_picture()
            if self._balls_fallen and all(
                float(b.get_pose().p[2]) < self.table_top + 0.06 for b in self.meatballs
            ):
                break

        # Damp plated balls, then pin them so they stop fidgeting on the dish.
        for ball, rigid in zip(self.meatballs, self._ball_rigids):
            if rigid is None or not self._ball_on_plate(ball):
                continue
            try:
                rigid.set_linear_velocity(np.zeros(3))
                rigid.set_angular_velocity(np.zeros(3))
                rigid.set_linear_damping(float(self.PLATE_LIN_DAMP))
                rigid.set_angular_damping(float(self.PLATE_ANG_DAMP))
            except Exception:
                pass

        self._pour_armed = False

        # Untilt wrist, then reattach with the pre-tip flat weld (not the tipped offset).
        hold_q = getattr(self, "_carry_quat", None) or list(
            GRASP_DIRECTION_DIC["top_down"]
        )
        self.move(
            self.move_by_displacement(
                arm,
                x=float(-toward[0] * 0.03),
                y=float(-toward[1] * 0.03),
                z=0.02,
                quat=list(hold_q),
                move_axis="world",
            )
        )
        self.plan_success = True
        flat_q = list(getattr(self, "_pan_flat_q", self.SKILLET_BASE_QPOS))
        ee = self._ee_pose(arm)
        if flat_weld_offset is not None:
            p = np.asarray((ee * flat_weld_offset).p, dtype=float)
        else:
            p = np.asarray(ee.p, dtype=float).copy()
            p[2] -= 0.02
        p[2] = max(float(p[2]), self.table_top + 0.11)
        self._set_entity_pose(self.skillet, sapien.Pose(p.tolist(), flat_q))
        self._pan_welded = bool(saved_weld)
        self._weld_pan_to_ee(arm)
        self._flatten_pan()
        self._idle_steps(4)

        # Ignore pan/ball so the retreat doesn't kick plated meatballs; then freeze.
        self._ignore_pan_ball_collision()
        self._seat_near_miss_balls_on_plate()
        self._freeze_balls_on_plate()

    def play_once(self) -> dict[str, Any]:
        arm = self.arm
        self.plan_success = True

        self._turn_knob_off()
        if self.stove_on:
            print("[serve_dinner] failed to turn stove off")
            self.plan_success = False
            return self.info

        if not self._grasp_pan():
            self.plan_success = False
            return self.info

        # Level lift — meatballs stay soft-welded until the pour tip begins.
        self.move(
            self.move_by_displacement(
                arm,
                z=0.08,
                quat=getattr(self, "_carry_quat", None),
                move_axis="world",
            )
        )
        self._sync_pan_to_ee()
        self._ensure_pan_level()
        self._idle_steps(4)
        if self._balls_fallen:
            print("[serve_dinner] meatballs fell during lift — fail")
            self.plan_success = False
            return self.info

        self._pour_onto_plate()
        if self._balls_fallen or not all(self._ball_on_plate(b) for b in self.meatballs):
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

        # Carry back while still welded — never free the pan mid-motion.
        place_xy = np.array(
            [float(self.burner_xy[0]), float(self.burner_xy[1]) - 0.02],
            dtype=float,
        )
        self._carry_pan_level(place_xy, self.table_top + 0.06)
        self._flatten_pan()
        # Lower onto the cooktop, still in the gripper.
        self._carry_pan_level(place_xy, float(self.range_top_z) + 0.008)
        self._flatten_pan()
        self._idle_steps(4)

        # Pin the pan on the burner, open the gripper, then retreat without yanking it.
        park_pose = self.skillet.get_pose()
        self._pan_welded = False
        self._pan_weld_offset = None
        self._set_entity_pose(self.skillet, park_pose)
        if self._pan_rigid is not None:
            try:
                self._pan_rigid.set_disable_gravity(True)
                self._pan_rigid.set_kinematic(True)
                self._pan_rigid.set_linear_velocity(np.zeros(3))
                self._pan_rigid.set_angular_velocity(np.zeros(3))
            except Exception:
                pass
        self.move(self.open_gripper(arm))
        self.plan_success = True
        # Clear retreat so the open gripper is not left hovering on the handle
        # (that used to look like the pan "fell out" of a still-grasping hand).
        self.move(
            self.move_by_displacement(
                arm,
                x=0.0,
                y=-0.10,
                z=0.10,
                quat=getattr(self, "_carry_quat", None),
                move_axis="world",
            )
        )
        self.plan_success = True
        # Keep the pan pinned where we left it (gripper has retreated).
        self._set_entity_pose(self.skillet, park_pose)
        self._idle_steps(10)
        self._freeze_balls_on_plate()

        if self.check_success():
            self.plan_success = True

        self.info["info"] = {
            "{A}": f"106_skillet/base{self.skillet_id}",
            "{B}": "003_plate/base0",
            "{C}": "cooking_range",
            "{D}": "stove_knob",
            "{E}": "meatballs",
            "{F}": "002_bowl",
            "{G}": "001_bottle",
            "{H}": "021_cup",
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
