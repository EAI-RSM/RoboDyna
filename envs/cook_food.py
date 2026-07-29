"""Cook food on a KitchenS stove, then pick it onto a plate.

KitchenS scene (no sink/tap): stove shifted left. Raw food starts on a chopping
board; the robot puts it in the pan, turns the stove knob up to 180° (intensity),
waits for a food-specific doneness, turns the knob back to zero, then picks the
cooked food out of the pan and places it on the plate.

Food types (``task_args.cook_food.food_type`` or random):
  - meat    — 200_steak; red → brown → black; ideal ~medium (0.5)
  - chicken — 258_chicken; white → lite pink → dark yellow → brown → black;
              ideal = dark yellow (~0.5)
  - sausage — 259_sausage; pink → red → dark red → black; ideal = dark red (~0.66)
"""
from __future__ import annotations

import json
import os
from typing import Any, ClassVar

import numpy as np
import sapien
import sapien.render
import transforms3d as t3d

from ._kitchens_base_task import KitchenS_base_task
from ._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
from .utils import *
from .utils.create_actor import create_actor, create_visual_box, UnStableError


class cook_food(KitchenS_base_task):
    """Cook one of three foods on the KitchenS range, then place onto a plate."""

    COOK_STEPS_DEFAULT: ClassVar[int] = 2000
    COOK_SPEED_JITTER_DEFAULT: ClassVar[float] = 0.10
    KNOB_CONTACT_RADIUS_DEFAULT: ClassVar[float] = 0.06
    EE_TO_TCP: ClassVar[float] = 0.12
    # Proven low corridor from boil_milk — must actually reach the knob.
    KNOB_APPROACH_PATH: ClassVar[tuple] = (
        (-0.13, -0.33, 0.06),
        (-0.08, -0.33, 0.00),
        (0.00, -0.25, 0.00),
    )
    KNOB_GRASP_STANDOFF: ClassVar[float] = -0.002  # slight push into the knob body
    KNOB_MAX_ANGLE: ClassVar[float] = float(np.pi)  # 180° clockwise = max fire
    SKILLET_BASE_QPOS: ClassVar[list[float]] = [0.0, 0.0, 0.707, 0.707]
    FOOD_QPOS: ClassVar[list[float]] = [0.707, 0.707, 0.0, 0.0]
    FOOD_TYPES: ClassVar[tuple[str, ...]] = ("meat", "chicken", "sausage")
    # Front-left burner: closest to the robot, handle points −Y.
    ACTIVE_BURNER: ClassVar[str] = "left_front"
    # Counter-local stove pose (scene_0 default is +0.42 — shift left into open space).
    RANGE_REL_XY: ClassVar[tuple[float, float]] = (0.02, 0.08)
    BOARD_QPOS: ClassVar[list[float]] = [0.707, 0.707, 0.0, 0.0]
    BOARD_SCALE_DEFAULT: ClassVar[float] = 0.07

    FOOD_SPECS: ClassVar[dict[str, dict[str, Any]]] = {
        "meat": {
            "modelname": "200_steak",
            "model_id": 0,
            "scale_mult": (0.85, 1.4, 0.85),
            "mass": 0.04,
            "target_doneness_range": (0.45, 0.55),
            "color_stops": [
                (0.0, [1.00, 0.12, 0.09]),
                (0.5, [0.66, 0.30, 0.14]),
                (1.0, [0.10, 0.05, 0.03]),
            ],
        },
        "chicken": {
            "modelname": "258_chicken",
            "model_id": 0,
            "scale_mult": 0.85,
            "mass": 0.04,
            "target_doneness_range": (0.45, 0.58),
            "color_stops": [
                (0.0, [0.96, 0.94, 0.88]),
                (0.25, [1.00, 0.72, 0.70]),
                (0.50, [0.78, 0.55, 0.12]),
                (0.75, [0.42, 0.22, 0.08]),
                (1.0, [0.06, 0.04, 0.03]),
            ],
        },
        "sausage": {
            "modelname": "259_sausage",
            "model_id": 0,
            "scale_mult": 1.0,
            "mass": 0.03,
            "target_doneness_range": (0.58, 0.72),
            "color_stops": [
                (0.0, [0.95, 0.55, 0.58]),
                (0.33, [0.85, 0.18, 0.16]),
                (0.66, [0.45, 0.06, 0.06]),
                (1.0, [0.05, 0.03, 0.03]),
            ],
        },
    }

    def setup_demo(self, **kwags: Any) -> None:
        self._cfg = dict(kwags.get("task_args", {}).get("cook_food", {}))
        if kwags.get("scene_id") is None:
            kwags["scene_id"] = int(self._cfg.get("scene_id", 0))
        self.replace_sink_with_range = True
        self.omit_sink = True
        self.clear_sink_and_range = False
        rel = self._cfg.get("range_xy", list(self.RANGE_REL_XY))
        self.range_position_override = [float(rel[0]), float(rel[1])]
        if "table_xy_bias" not in kwags and "table_xy_bias" in self._cfg:
            kwags["table_xy_bias"] = list(self._cfg["table_xy_bias"])

        self.stove_on = False
        self.knob_angle = 0.0
        self.fire_intensity = 0.0
        self.doneness = 0.0
        self.max_doneness = 0.0
        self._grasp_doneness = None
        self._food_locked = False
        self._food_rel = None
        self._food_shapes: list[Any] = []
        self._placed = False
        self._food_in_pan = False
        self.turned_on_once = False
        self.turned_off_after_cook = False
        self._ignore_knob = False
        self._expert_holding_knob = False
        self._cook_phase_done = False
        self._cooking_idle = False
        self._burner_shapes = []
        self._ring_shapes: list[Any] = []
        self._ring_parts: list[Any] = []
        self._skillet_home = None
        self._food_rigid = None
        self._food_welded = False
        self._food_weld_offset = None

        super().setup_demo(**kwags)
        self._configure_head_camera()

    def _configure_head_camera(self) -> None:
        """Pull the head camera well back/up and widen FOV so the stove is clear."""
        cams = getattr(self, "cameras", None)
        if cams is None:
            return
        names = list(getattr(cams, "static_camera_name", []) or [])
        clist = list(getattr(cams, "static_camera_list", []) or [])
        if "head_camera" not in names:
            return
        camera = clist[names.index("head_camera")]
        rx, ry = getattr(self, "range_xy", (0.22, 0.05))
        bx = float(getattr(self, "board_xy", (rx - 0.25, ry))[0])
        # Frame board + stove + plate (layout shifted left, no sink).
        center_x = 0.5 * (float(rx) + bx)
        cam_pos = np.array([center_x, -1.15, 1.95], dtype=float)
        look_at = np.array([center_x, float(ry) * 0.15, 0.82], dtype=float)
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

        self.cook_steps = self._sample_cook_steps(cfg)
        self.knob_contact_radius = float(
            cfg.get("knob_contact_radius", self.KNOB_CONTACT_RADIUS_DEFAULT)
        )
        self.pan_scale = float(cfg.get("pan_scale", 0.60))
        self.plate_scale = float(cfg.get("plate_scale", 0.55))
        self.board_scale = float(cfg.get("board_scale_mult", self.BOARD_SCALE_DEFAULT))
        self.shutoff_lead = float(cfg.get("shutoff_lead", 0.03))
        self.cook_intensity = float(np.clip(cfg.get("cook_intensity", 0.75), 0.15, 1.0))

        food_type = cfg.get("food_type", None)
        if food_type is None or str(food_type).lower() in ("random", "any", ""):
            food_type = str(np.random.choice(self.FOOD_TYPES))
        food_type = str(food_type).strip().lower()
        if food_type not in self.FOOD_SPECS:
            raise ValueError(
                f"cook_food.food_type must be one of {self.FOOD_TYPES}, got {food_type!r}"
            )
        self.food_type = food_type
        self.food_spec = self.FOOD_SPECS[food_type]
        self.color_stops = list(self.food_spec["color_stops"])

        range_cfg = cfg.get("target_doneness_range")
        if range_cfg is not None:
            lo, hi = float(range_cfg[0]), float(range_cfg[1])
        else:
            lo, hi = map(float, self.food_spec["target_doneness_range"])
        if not 0.0 <= lo <= hi <= 1.0:
            raise ValueError("cook_food target doneness range must satisfy 0 <= min <= max <= 1")
        self.target_doneness_range = (lo, hi)
        self.target_doneness = 0.5 * (lo + hi)

        self.stove_on = False
        self.knob_angle = 0.0
        self.fire_intensity = 0.0
        self.doneness = 0.0
        self.max_doneness = 0.0
        self._grasp_doneness = None
        self._food_locked = False
        self._food_rel = None
        self._placed = False
        self._food_in_pan = False
        self.turned_on_once = False
        self.turned_off_after_cook = False
        self._ignore_knob = False
        self._expert_holding_knob = False
        self._cook_phase_done = False
        self._cooking_idle = False
        self._food_welded = False
        self._food_weld_offset = None

        bz = 0.74 + self.table_z_bias
        burner_name = str(cfg.get("burner", self.ACTIVE_BURNER)).strip().lower()
        if burner_name not in self.burner_positions:
            raise ValueError(
                f"cook_food.burner must be one of {list(self.burner_positions)}, got {burner_name!r}"
            )
        bx, by = self.burner_positions[burner_name]
        self.burner_name = burner_name
        self.burner_xy = (float(bx), float(by))

        # Hide the solid orange disc — it shows through the open pan onto the food.
        if getattr(self, "active_burner", None) is not None:
            try:
                self.active_burner.set_pose(sapien.Pose(p=[0.0, 0.0, -1.0]))
            except Exception:
                pass
        self._burner_shapes = []
        self._ring_parts = []
        self._ring_shapes = []

        # Static pan seated on the burner (empty until food is placed from the board).
        self.skillet_id = int(np.random.choice([0, 2]))
        skillet_q = list(self.SKILLET_BASE_QPOS)
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
            is_static=True,
            scale_mult=self.pan_scale,
        )
        self.skillet.set_name("106_skillet")
        self.add_prohibit_area(self.skillet, padding=0.04)

        # Slide pan until bowl matches burner XY, then small visual nudge.
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
        self._build_burner_ring(
            self.burner_xy[0], self.burner_xy[1], float(self.range_top_z) + 0.0012
        )
        self._set_burner_visuals(0.0)
        self._skillet_home = self.skillet.get_pose()

        hx, hy = getattr(self, "range_half_size", (0.13, 0.15))
        rx, ry = float(self.range_xy[0]), float(self.range_xy[1])

        # Chopping board left of the stove (open counter between MW and range).
        # 104_board has no authored "scale" key → create_actor leaves config=None;
        # cook_meat restores extents/center so prohibit-area / Z math work.
        with open(
            "assets/objects/104_board/model_data0.json", encoding="utf-8"
        ) as board_data_file:
            board_data = json.load(board_data_file)
        # BOARD_QPOS is 90° about X → local Y (thin axis) becomes world Z thickness.
        board_th = float(board_data["extents"][1]) * self.board_scale
        board_x = rx - float(hx) - 0.14
        board_y = ry - 0.02
        board_pose = sapien.Pose(
            [board_x, board_y, bz + 0.5 * board_th], list(self.BOARD_QPOS)
        )
        self.board = create_actor(
            self,
            pose=board_pose,
            modelname="104_board",
            model_id=0,
            convex=True,
            is_static=True,
            scale_mult=self.board_scale,
        )
        self.board.set_name("104_board")
        self.board.config = {
            "scale": [self.board_scale, self.board_scale, self.board_scale],
            "extents": board_data["extents"],
            "center": board_data["center"],
        }
        self.add_prohibit_area(self.board, padding=0.03)
        self.board_xy = (board_x, board_y)
        self.board_top_z = bz + board_th

        # Plate in front of the board (toward the robot).
        plate_x = board_x
        plate_y = board_y - 0.16
        plate_pose = sapien.Pose([plate_x, plate_y, bz], [0.707, 0.707, 0.0, 0.0])
        self.plate = create_actor(
            self,
            pose=plate_pose,
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
        self.plate_xy = (plate_x, plate_y)
        try:
            self.plate_top_z = float(self.plate.get_functional_point(0)[2])
        except Exception:
            self.plate_top_z = bz + 0.02

        # Food starts on the chopping board (not in the pan).
        food_pose = sapien.Pose(
            [board_x, board_y, float(self.board_top_z) + 0.008],
            list(self.FOOD_QPOS),
        )
        self.food = create_actor(
            self,
            pose=food_pose,
            modelname=self.food_spec["modelname"],
            model_id=int(self.food_spec["model_id"]),
            convex=True,
            is_static=False,
            scale_mult=self.food_spec["scale_mult"],
        )
        self.food.set_name(self.food_spec["modelname"])
        self.food.set_mass(float(self.food_spec["mass"]))
        self._food_rigid = None
        for c in self.food.actor.get_components():
            if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                c.set_linear_damping(20.0)
                c.set_angular_damping(40.0)
                self._food_rigid = c
        self.add_prohibit_area(self.food, padding=0.02)

        self._food_shapes = []
        for c in self.food.actor.get_components():
            if isinstance(c, sapien.render.RenderBodyComponent):
                self._food_shapes = list(c.render_shapes)
        self._set_food_color(0.0)
        self._set_knob_angle(0.0)

        # Knob is near center-right of the shifted stove → right arm.
        # Food/board sit further left → left arm for pick/place.
        self.arm = ArmTag("right" if self.knob_xy[0] >= 0 else "left")
        self.food_arm = ArmTag("left" if float(self.board_xy[0]) < 0.0 else "right")

    def _clear_burner_ring(self) -> None:
        for part in getattr(self, "_ring_parts", []) or []:
            try:
                self.scene.remove_entity(part)
            except Exception:
                pass
        self._ring_parts = []
        self._ring_shapes = []

    def _build_burner_ring(self, cx: float, cy: float, cz: float) -> None:
        """Blue fire halo just outside the pan bowl (must clear the pan to be visible)."""
        self._clear_burner_ring()
        n = 36
        # Pan footprint at pan_scale≈0.6 covers ~r=0.045; ring must sit outside that.
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

    @classmethod
    def _sample_cook_steps(cls, config: dict[str, Any]) -> int:
        jitter = float(config.get("cook_speed_jitter", cls.COOK_SPEED_JITTER_DEFAULT))
        jitter = float(np.clip(abs(jitter), 0.0, 0.95))
        nom = float(config.get("cook_steps", cls.COOK_STEPS_DEFAULT))
        nom = max(1.0, nom)
        return max(
            1,
            int(round(float(np.random.uniform(nom * (1 - jitter), nom * (1 + jitter))))),
        )

    # ----------------------------------------------------------- visuals / cook
    def _set_food_color(self, doneness: float) -> None:
        d = float(np.clip(doneness, 0.0, 1.0))
        stops = self.color_stops
        rgb = stops[-1][1]
        for i in range(len(stops) - 1):
            d0, c0 = stops[i]
            d1, c1 = stops[i + 1]
            if d <= d1 or i == len(stops) - 2:
                t = 0.0 if d1 == d0 else (d - d0) / (d1 - d0)
                t = float(np.clip(t, 0.0, 1.0))
                rgb = [c0[k] + (c1[k] - c0[k]) * t for k in range(3)]
                break
        color = list(rgb) + [1.0]
        for s in self._food_shapes:
            try:
                s.material.set_base_color(color)
            except Exception:
                pass

    def _set_burner_visuals(self, intensity: float) -> None:
        inten = float(np.clip(intensity, 0.0, 1.0))
        on = inten > 0.02
        disc = [0.95, 0.35 + 0.2 * inten, 0.05, 1.0] if on else [0.20, 0.20, 0.22, 1.0]
        for s in getattr(self, "_burner_shapes", []) or []:
            try:
                s.material.set_base_color(disc)
            except Exception:
                pass
        ring = [0.20, 0.70 + 0.25 * inten, 1.0, 1.0] if on else [0.18, 0.18, 0.20, 1.0]
        for s in getattr(self, "_ring_shapes", []) or []:
            try:
                s.material.set_base_color(ring)
            except Exception:
                pass

    def _set_knob_angle(self, angle: float) -> None:
        """Set knob angle in [0, π]; 0 = off, π = max fire (180° to the right)."""
        angle = float(np.clip(angle, 0.0, self.KNOB_MAX_ANGLE))
        self.knob_angle = angle
        self.fire_intensity = angle / self.KNOB_MAX_ANGLE
        self.stove_on = self.fire_intensity > 0.02
        if self.stove_on:
            self.turned_on_once = True
        elif self.turned_on_once and self.max_doneness > 0.05:
            self.turned_off_after_cook = True

        if getattr(self, "stove_knob_indicator", None) is not None:
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
        self._set_burner_visuals(self.fire_intensity)

    def _lock_food_to_pan(self) -> None:
        pan = self.skillet.get_pose()
        food = self.food.get_pose()
        self._food_rel = pan.inv() * food
        self._food_locked = True

    def _unlock_food(self) -> None:
        self._food_locked = False
        self._food_rel = None

    def _sync_locked_food(self) -> None:
        if not self._food_locked or self._food_rel is None:
            return
        pose = self.skillet.get_pose() * self._food_rel
        self.food.actor.set_pose(pose)
        if self._food_rigid is not None:
            try:
                self._food_rigid.set_linear_velocity(np.zeros(3))
                self._food_rigid.set_angular_velocity(np.zeros(3))
            except Exception:
                pass

    def _ee_pose(self, arm: ArmTag) -> sapien.Pose:
        p = (
            self.robot.get_right_ee_pose()
            if str(arm) == "right"
            else self.robot.get_left_ee_pose()
        )
        return sapien.Pose(list(p[:3]), list(p[3:7]))

    def _tcp_pos(self, arm: ArmTag) -> np.ndarray:
        p = (
            self.robot.get_right_tcp_pose()
            if str(arm) == "right"
            else self.robot.get_left_tcp_pose()
        )
        return np.array(p[:3], dtype=float)

    def _weld_food_to_ee(self, arm: ArmTag) -> None:
        """Lock food at its current pose relative to the EE (no teleport/snap)."""
        if self._food_rigid is not None:
            try:
                self._food_rigid.set_disable_gravity(True)
                self._food_rigid.set_kinematic(True)
            except Exception:
                pass
        food_pose = self.food.get_pose()
        self._food_weld_offset = self._ee_pose(arm).inv() * food_pose
        self._food_welded = True

    def _release_food_weld(self) -> None:
        self._food_welded = False
        self._food_weld_offset = None
        if self._food_rigid is not None:
            try:
                self._food_rigid.set_kinematic(False)
                self._food_rigid.set_disable_gravity(False)
            except Exception:
                pass

    def _sync_welded_food(self) -> None:
        if not self._food_welded or self._food_weld_offset is None:
            return
        arm = getattr(self, "food_arm", self.arm)
        pose = self._ee_pose(arm) * self._food_weld_offset
        self.food.actor.set_pose(pose)
        try:
            if self._food_rigid is not None:
                self._food_rigid.set_kinematic_target(pose)
        except Exception:
            pass

    # ----------------------------------------------------------- per-step
    def _update_kinematic_tasks(self) -> None:
        super()._update_kinematic_tasks()
        if not getattr(self, "food", None):
            return

        self._sync_locked_food()
        self._sync_welded_food()
        if getattr(self, "_skillet_home", None) is not None:
            self.skillet.actor.set_pose(self._skillet_home)

        if getattr(self, "_expert_holding_knob", False) and hasattr(self, "robot"):
            try:
                ee = np.array(self.get_arm_pose(str(self.arm)), dtype=float)
                base_q = np.asarray(GRASP_DIRECTION_DIC["front"], dtype=float)
                rel = t3d.quaternions.qmult(
                    t3d.quaternions.qinverse(base_q), ee[3:7]
                )
                twist = 2.0 * np.arctan2(float(rel[1]), float(rel[0]))
                self._set_knob_angle(float(np.clip(twist, 0.0, self.KNOB_MAX_ANGLE)))
            except Exception:
                pass

        if self._cook_phase_done or self._grasp_doneness is not None:
            return
        if (
            self._cooking_idle
            and self.fire_intensity > 0.02
            and self._food_locked
        ):
            self.doneness = min(
                1.0,
                self.doneness + self.fire_intensity / max(1, self.cook_steps),
            )
            self.max_doneness = max(self.max_doneness, self.doneness)
            self._set_food_color(self.doneness)

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

    def _idle_until_doneness(self, level: float, max_steps: int | None = None) -> None:
        inten = max(0.05, float(self.fire_intensity))
        if max_steps is None:
            max_steps = int(round(float(level) * self.cook_steps / inten)) + 60
        self._idle_steps(
            max_steps,
            until=lambda: self.doneness >= float(level) or self.doneness >= 0.99,
        )

    # ----------------------------------------------------------- knob expert
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

    def _tcp_near_knob(self, tol: float = 0.045) -> bool:
        try:
            tcp = np.array(
                self.robot.get_right_tcp_pose()
                if str(self.arm) == "right"
                else self.robot.get_left_tcp_pose(),
                dtype=float,
            )[:3]
            return float(np.linalg.norm(tcp - np.asarray(self.knob_xyz))) < tol
        except Exception:
            return False

    def _set_knob_to(self, target_angle: float, approach: bool = True) -> None:
        """Grasp the knob (boil_milk corridor) and turn 0…180°."""
        arm = self.arm
        start = float(self.knob_angle)
        end = float(np.clip(target_angle, 0.0, self.KNOB_MAX_ANGLE))
        path = self.KNOB_APPROACH_PATH if approach else self.KNOB_APPROACH_PATH[-1:]

        self._ignore_knob = True
        self.move(self.open_gripper(arm))
        for offset in path:
            self.move(self.move_to_pose(arm, self._knob_pose(offset, start)))
        # Push in until jaws can close on the knob body.
        self.move(
            self.move_to_pose(arm, self._knob_turn_pose(self.KNOB_GRASP_STANDOFF, start))
        )
        if not self._tcp_near_knob(0.055):
            # One retry closer.
            self.move(self.move_to_pose(arm, self._knob_turn_pose(0.0, start)))
        self.move(self.close_gripper(arm))
        self._dbg(f"knob_grasp_near={self._tcp_near_knob()}")

        self._expert_holding_knob = True
        self.move(
            self.move_to_pose(arm, self._knob_turn_pose(self.KNOB_GRASP_STANDOFF, end))
        )
        self._expert_holding_knob = False
        self._set_knob_angle(end)
        self._idle_steps(6)

        self.move(self.open_gripper(arm))
        for offset in reversed(path):
            self.move(self.move_to_pose(arm, self._knob_pose(offset, end)))
        self._ignore_knob = False

    # ----------------------------------------------------------- place expert
    def _dbg(self, tag: str) -> None:
        if os.environ.get("COOK_DEBUG"):
            print(f"[cook_food] {tag}: plan_success={self.plan_success}", flush=True)

    def _food_held(self) -> bool:
        try:
            return len(self.get_gripper_actor_contact_position(self.food.get_name())) > 0
        except Exception:
            return False

    def _safe_grasp_actor(self, actor: Any, arm_tag: ArmTag, **kwargs: Any):
        pre_pose, grasp_pose = self.choose_grasp_pose(
            actor,
            arm_tag=arm_tag,
            pre_dis=float(kwargs.get("pre_grasp_dis", 0.1)),
            target_dis=float(kwargs.get("grasp_dis", 0.0)),
            contact_point_id=kwargs.get("contact_point_id"),
        )
        if pre_pose is None or grasp_pose is None:
            raise UnStableError("cook_food: no grasp pose — skip")
        return self.grasp_actor(actor, arm_tag=arm_tag, **kwargs)

    def _carry_food_by(self, food_xyz, arm: ArmTag | None = None) -> None:
        """Translate EE in world so welded food moves to ``food_xyz`` (keep orientation)."""
        arm = arm if arm is not None else self.food_arm
        cur = np.asarray(self.food.get_pose().p, dtype=float)
        delta = np.asarray(food_xyz, dtype=float) - cur
        self.move(
            self.move_by_displacement(
                arm_tag=arm,
                x=float(delta[0]),
                y=float(delta[1]),
                z=float(delta[2]),
                move_axis="world",
            )
        )

    def _pick_food_to_plate(self) -> None:
        """Grasp food, carry it with the arm, set it down on the plate (no teleport)."""
        arm = self.food_arm
        top = list(GRASP_DIRECTION_DIC["top_down"])
        self._unlock_food()
        if self._grasp_doneness is None:
            self._grasp_doneness = float(self.doneness)
        self._cook_phase_done = True

        # Lift food just above the rim so fingers can close around it (still in-bowl).
        bowl = np.asarray(self.skillet.get_functional_point(0)[:3], dtype=float)
        food_pose = sapien.Pose(
            [float(bowl[0]), float(bowl[1]), float(bowl[2]) + 0.022],
            list(self.FOOD_QPOS),
        )
        self.food.actor.set_pose(food_pose)
        if self._food_rigid is not None:
            try:
                self._food_rigid.set_linear_velocity(np.zeros(3))
                self._food_rigid.set_angular_velocity(np.zeros(3))
            except Exception:
                pass
        self._idle_steps(4)

        p = np.asarray(self.food.get_pose().p, dtype=float)
        self.move(self.open_gripper(arm))
        self.move(
            self.move_to_pose(
                arm, [float(p[0]), float(p[1]), float(p[2]) + 0.16, *top]
            )
        )
        self.move(
            self.move_to_pose(
                arm, [float(p[0]), float(p[1]), float(p[2]) + 0.018, *top]
            )
        )
        self.move(self.close_gripper(arm))
        self._idle_steps(4)
        tcp = self._tcp_pos(arm)
        near = float(np.linalg.norm(tcp[:2] - p[:2])) < 0.045
        if not (self._food_held() or near):
            raise UnStableError("cook_food: failed to grasp food from pan — skip")
        # Weld at the actual grasp pose — food rides with the gripper on the carry.
        self._weld_food_to_ee(arm)
        self.move(self.move_by_displacement(arm_tag=arm, z=0.12, move_axis="arm"))
        self._dbg("grasp_food")

        # Carry by world translation of the EE (food welded; no set_pose teleport).
        self._carry_food_by(
            [
                float(self.plate_xy[0]),
                float(self.plate_xy[1]),
                float(self.plate_top_z) + 0.12,
            ]
        )
        self._carry_food_by(
            [
                float(self.plate_xy[0]),
                float(self.plate_xy[1]),
                float(self.plate_top_z) + 0.018,
            ]
        )
        self._idle_steps(6)

        # Open while still welded (food stays put), then release onto the plate.
        self.move(self.open_gripper(arm))
        self._idle_steps(6)
        self._release_food_weld()
        self._idle_steps(12)
        self.move(self.move_by_displacement(arm_tag=arm, z=0.10, move_axis="arm"))
        self._idle_steps(6)

        food_xy = np.asarray(self.food.get_pose().p[:2], dtype=float)
        d_plate = float(np.linalg.norm(food_xy - np.asarray(self.plate_xy)))
        self._placed = d_plate < 0.08
        self._dbg(f"place_plate d={d_plate:.3f}")
        if not self._placed:
            raise UnStableError(
                f"cook_food: food missed plate (d={d_plate:.3f}) — skip"
            )

    # ----------------------------------------------------------- place into pan
    def _pan_place_target(self) -> list[float]:
        bowl = np.asarray(self.skillet.get_functional_point(0), dtype=float)
        return [
            float(bowl[0]),
            float(bowl[1]),
            float(bowl[2]) + 0.018,
            *self.FOOD_QPOS,
        ]

    def _seat_food_in_pan(self) -> None:
        """Snap food into the bowl and lock it for cooking."""
        bowl = np.asarray(self.skillet.get_functional_point(0)[:3], dtype=float)
        pose = sapien.Pose(
            [float(bowl[0]), float(bowl[1]), float(bowl[2]) + 0.012],
            list(self.FOOD_QPOS),
        )
        self.food.actor.set_pose(pose)
        if self._food_rigid is not None:
            try:
                self._food_rigid.set_linear_velocity(np.zeros(3))
                self._food_rigid.set_angular_velocity(np.zeros(3))
            except Exception:
                pass
        self._lock_food_to_pan()
        self._food_in_pan = True
        self._idle_steps(4)

    def _place_food_in_pan(self) -> None:
        """Pick food from the chopping board and put it in the pan.

        Top-down grasp + welded world carry (same pattern as pan→plate). Contact-point
        ``place_actor`` often fails once the board sits left of the shifted stove.
        """
        arm = self.food_arm
        top = list(GRASP_DIRECTION_DIC["top_down"])
        p = np.asarray(self.food.get_pose().p, dtype=float)
        self.move(self.open_gripper(arm))
        self.move(
            self.move_to_pose(
                arm, [float(p[0]), float(p[1]), float(p[2]) + 0.16, *top]
            )
        )
        self.move(
            self.move_to_pose(
                arm, [float(p[0]), float(p[1]), float(p[2]) + 0.016, *top]
            )
        )
        self.move(self.close_gripper(arm))
        self._idle_steps(4)
        tcp = self._tcp_pos(arm)
        near = float(np.linalg.norm(tcp[:2] - p[:2])) < 0.05
        if not (self._food_held() or near):
            # Fallback: contact-point grasp (works for left arm on board).
            self.move(self.open_gripper(arm))
            self.move(self._safe_grasp_actor(self.food, arm_tag=arm, pre_grasp_dis=0.10))
            self._idle_steps(4)
            if not self._food_held():
                raise UnStableError("cook_food: failed to grasp food from board — skip")
        self._weld_food_to_ee(arm)
        self.move(self.move_by_displacement(arm_tag=arm, z=0.12, move_axis="arm"))
        self._dbg("grasp_from_board")

        bowl = np.asarray(self.skillet.get_functional_point(0)[:3], dtype=float)
        self._carry_food_by(
            [float(bowl[0]), float(bowl[1]), float(bowl[2]) + 0.10], arm=arm
        )
        self._carry_food_by(
            [float(bowl[0]), float(bowl[1]), float(bowl[2]) + 0.018], arm=arm
        )
        self._idle_steps(6)
        self.move(self.open_gripper(arm))
        self._idle_steps(4)
        self._release_food_weld()
        self.move(self.move_by_displacement(arm_tag=arm, z=0.12, move_axis="arm"))
        # Park left arm so the right-arm knob corridor stays clear.
        try:
            self.move(self.back_to_origin(arm))
        except Exception:
            pass

        self._seat_food_in_pan()
        bowl_xy = np.asarray(bowl[:2], dtype=float)
        food_xy = np.asarray(self.food.get_pose().p[:2], dtype=float)
        if float(np.linalg.norm(food_xy - bowl_xy)) > 0.06:
            raise UnStableError("cook_food: food not in pan after place — skip")
        # Seating is kinematic; clear any prior motion-plan flag for the cook phase.
        self.plan_success = True
        self._dbg("food_in_pan")

    # ----------------------------------------------------------- policy
    def play_once(self) -> dict[str, Any]:
        # 1) Board → pan (left arm when board is left of center)
        self._place_food_in_pan()

        # 2) Cook via knob (right arm)
        target_on = float(self.cook_intensity) * self.KNOB_MAX_ANGLE
        self._set_knob_to(target_on, approach=True)
        self._dbg("stove_on")

        self._cooking_idle = True
        self._idle_until_doneness(self.target_doneness)
        self._cooking_idle = False
        self._cook_phase_done = True
        self._grasp_doneness = float(self.doneness)
        self._set_knob_to(0.0, approach=False)
        self._dbg("stove_off")

        # 3) Pan → plate
        self._pick_food_to_plate()

        self.info["info"] = {
            "{A}": f"{self.food_spec['modelname']}/base{self.food_spec['model_id']}",
            "{B}": f"106_skillet/base{self.skillet_id}",
            "{C}": "003_plate/base0",
            "{D}": "cooking_range",
            "{E}": "stove_knob",
            "{F}": self.food_type,
            "{G}": "104_board/base0",
            "{a}": str(self.food_arm),
        }
        return self.info

    # ----------------------------------------------------------- success / obs
    def _doneness_in_target_range(self, doneness: float) -> bool:
        lo, hi = self.target_doneness_range
        return float(lo) <= float(doneness) <= float(hi)

    def check_success(self) -> bool:
        if self._grasp_doneness is None:
            return False
        if not self._doneness_in_target_range(float(self._grasp_doneness)):
            return False
        if self.fire_intensity > 0.02 or self.knob_angle > 0.05:
            return False
        if not self.turned_on_once or not self.turned_off_after_cook:
            return False
        if not self._placed:
            return False
        food_p = self.food.get_pose().p
        food_xy = np.asarray(food_p[:2], dtype=float)
        plate_xy = np.asarray(self.plate_xy, dtype=float)
        pan_xy = np.asarray(self.burner_xy, dtype=float)
        d_plate = float(np.linalg.norm(food_xy - plate_xy))
        d_pan = float(np.linalg.norm(food_xy - pan_xy))
        on_plate = d_plate < 0.07
        off_pan = d_pan > 0.10
        above_plate = float(food_p[2]) > (self.plate_top_z - 0.03)
        return bool(on_plate and off_pan and above_plate)

    def get_obs(self) -> dict[str, Any]:
        obs = super().get_obs()
        obs["cooking"] = {
            "food_type": str(getattr(self, "food_type", "")),
            "doneness": float(getattr(self, "doneness", 0.0)),
            "max_doneness": float(getattr(self, "max_doneness", 0.0)),
            "grasp_doneness": (
                None
                if self._grasp_doneness is None
                else float(self._grasp_doneness)
            ),
            "target_doneness": float(getattr(self, "target_doneness", 0.5)),
            "target_doneness_range": list(
                getattr(self, "target_doneness_range", (0.45, 0.55))
            ),
            "cook_steps": float(getattr(self, "cook_steps", self.COOK_STEPS_DEFAULT)),
            "knob_angle": float(getattr(self, "knob_angle", 0.0)),
            "fire_intensity": float(getattr(self, "fire_intensity", 0.0)),
            "stove_on": bool(getattr(self, "stove_on", False)),
            "placed": bool(getattr(self, "_placed", False)),
            "burner": str(getattr(self, "burner_name", "")),
        }
        return obs
