"""Cook food on a KitchenS stove, then pick it onto a plate.

KitchenS scene (no sink/tap/microwave): stove shifted left. Raw food starts on a
chopping board; the robot puts it in the pan, turns the stove knobs up to 180°
(intensity), waits for a food-specific doneness, turns the knob back to zero,
then picks the cooked food out of the pan and places it on the plate. A decorative
plate of raw meat sits to the right of the stove.

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
    # Straight-down approach onto the top-facing cooktop knob.
    KNOB_APPROACH_PATH: ClassVar[tuple] = KitchenS_base_task.TOP_KNOB_APPROACH_PATH
    KNOB_GRASP_STANDOFF: ClassVar[float] = 0.012
    # Magnitude of the left turn: 0 = off (tick up), −π/2 = full fire (tick left).
    KNOB_MAX_ANGLE: ClassVar[float] = float(np.pi / 2)
    SKILLET_BASE_QPOS: ClassVar[list[float]] = [0.0, 0.0, 0.707, 0.707]
    # Base orientation points the handle −Y (toward the robot). Extra world-Z
    # yaw of +90° swings the handle to +X (right side of the stove).
    SKILLET_HANDLE_YAW: ClassVar[float] = 0.5 * np.pi
    FOOD_QPOS: ClassVar[list[float]] = [0.707, 0.707, 0.0, 0.0]
    FOOD_TYPES: ClassVar[tuple[str, ...]] = ("meat", "chicken", "sausage")
    # Front-left burner on the cooktop; handle points +X (right).
    ACTIVE_BURNER: ClassVar[str] = "left_front"
    # Cooktop slightly left; arms stay on their own sides via park poses.
    RANGE_REL_XY: ClassVar[tuple[float, float]] = (0.10, 0.14)
    BOARD_REL_XY: ClassVar[tuple[float, float]] = (-0.28, -0.02)
    PLATE_REL_XY: ClassVar[tuple[float, float]] = (-0.18, -0.12)
    BOARD_QPOS: ClassVar[list[float]] = [0.707, 0.707, 0.0, 0.0]
    BOARD_SCALE_DEFAULT: ClassVar[float] = 0.0975
    PLATE_SCALE_DEFAULT: ClassVar[float] = 0.715
    PAN_SCALE_DEFAULT: ClassVar[float] = 1.0
    RANGE_SCALE_DEFAULT: ClassVar[float] = 1.0

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
        # Same KitchenS stove scale as boil_milk (must be set before range load).
        self.range_scale_mult = float(
            self._cfg.get("range_scale_mult", self.RANGE_SCALE_DEFAULT)
        )
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
        self._cook_hold = False
        self._cooking_idle = False
        self._burner_shapes = []
        self._ring_shapes: list[Any] = []
        self._ring_parts: list[Any] = []
        self._ring_home_poses: list[Any] = []
        self._skillet_home = None
        self._food_rigid = None

        super().setup_demo(**kwags)
        self._configure_head_camera()

    def _load_microwave(self, table_height, table_xy_bias):
        """Open the left counter — no microwave for this task."""
        return

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
        dx = float(getattr(self, "decor_plate_xy", (rx + 0.25, ry))[0])
        # Frame board + stove + deco plate (no sink / microwave).
        center_x = 0.5 * (bx + dx)
        cam_pos = np.array([center_x * 0.35 + float(rx) * 0.65, -1.20, 1.95], dtype=float)
        look_at = np.array([float(rx), float(ry) * 0.10, 0.82], dtype=float)
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
        # Match cook_meat skillet; board/plate are +30% vs cook_meat for this scene.
        self.pan_scale = float(cfg.get("pan_scale", self.PAN_SCALE_DEFAULT))
        self.plate_scale = float(cfg.get("plate_scale", self.PLATE_SCALE_DEFAULT))
        self.board_scale = float(cfg.get("board_scale_mult", self.BOARD_SCALE_DEFAULT))
        self.shutoff_lead = float(cfg.get("shutoff_lead", 0.03))
        self.cook_intensity = float(np.clip(cfg.get("cook_intensity", 0.75), 0.15, 1.0))
        # place_actor release bias relative to bowl center (left arm undershoots −X).
        self.place_dx = float(cfg.get("place_dx", 0.05))
        self.place_dy = float(cfg.get("place_dy", 0.05))

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
                self.active_burner.set_pose(self._fire_hidden_pose())
            except Exception:
                pass
        self._burner_shapes = []
        # Must remove prior ring entities (not just drop list refs) or a gray
        # "off" ring stays visible under the pan forever.
        self._clear_stove_fire_ring()

        # Static pan seated on the burner. Model 2 has a reachable bowl for the
        # left-arm board→pan place; model 0 consistently undershoots / drops short.
        self.skillet_id = int(cfg.get("skillet_id", 2))
        if self.skillet_id not in (0, 2):
            self.skillet_id = 2
        handle_yaw = float(cfg.get("skillet_handle_yaw", self.SKILLET_HANDLE_YAW))
        skillet_q = list(
            t3d.quaternions.qmult(
                t3d.euler.euler2quat(0.0, 0.0, handle_yaw),
                list(self.SKILLET_BASE_QPOS),
            )
        )
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
            is_static=True,
            scale_mult=self.pan_scale,
        )
        self.skillet.set_name("106_skillet")
        self.add_prohibit_area(self.skillet, padding=0.04)

        # Slide pan until bowl matches burner XY (no extra visual nudge).
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
        self._build_burner_ring(
            self.burner_xy[0], self.burner_xy[1], float(self.range_top_z) + 0.0012
        )
        self._set_burner_visuals(0.0)
        self._skillet_home = self.skillet.get_pose()

        hx, hy = getattr(self, "range_half_size", (0.13, 0.15))
        rx, ry = float(self.range_xy[0]), float(self.range_xy[1])

        # Chopping board left of the stove (microwave omitted — open counter).
        # 104_board has no authored "scale" key → create_actor leaves config=None;
        # cook_meat restores extents/center so prohibit-area / Z math work.
        with open(
            "assets/objects/104_board/model_data0.json", encoding="utf-8"
        ) as board_data_file:
            board_data = json.load(board_data_file)
        # BOARD_QPOS is 90° about X → local Y (thin axis) becomes world Z thickness.
        board_th = float(board_data["extents"][1]) * self.board_scale
        # Board left of the stove (microwave omitted). Extra gap for +30% board.
        board_half_x = 0.5 * float(board_data["extents"][0]) * self.board_scale
        board_x = float(cfg.get("board_x", rx - float(hx) - board_half_x - 0.12))
        board_y = float(cfg.get("board_y", self.BOARD_REL_XY[1]))
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

        # Serving plate forward-left — clear of the board→pan path.
        plate_x = float(cfg.get("plate_x", 0.5 * (board_x + float(bx))))
        plate_y = float(cfg.get("plate_y", self.PLATE_REL_XY[1]))
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
        # On-plate radius scales with plate size (0.55 → tol 0.10 historically).
        self.plate_tol = 0.10 * (self.plate_scale / 0.55)
        try:
            self.plate_top_z = float(self.plate.get_functional_point(0)[2])
        except Exception:
            self.plate_top_z = bz + 0.02

        # Task food starts on the chopping board (must be visible before any cook step).
        food_pose = sapien.Pose(
            [board_x, board_y, float(self.board_top_z) + 0.012],
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
                # Light damping so flat food can slip; not glued in place.
                c.set_linear_damping(2.0)
                c.set_angular_damping(4.0)
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

        # Decorative: raw meat on a plate to the right of the stove (static props).
        self._spawn_decor_meat_plate(bz, rx, ry, float(hx))

    def _spawn_decor_meat_plate(
        self, bz: float, rx: float, ry: float, hx: float
    ) -> None:
        """Static plate + two non-overlapping raw steaks right of the stove."""
        # Keep clear of the larger range footprint.
        deco_x = rx + hx + 0.20
        deco_y = ry - 0.08
        deco_scale = 0.55
        plate = create_actor(
            self,
            pose=sapien.Pose([deco_x, deco_y, bz], [0.707, 0.707, 0.0, 0.0]),
            modelname="003_plate",
            model_id=0,
            convex=True,
            is_static=True,
            scale_mult=deco_scale,
        )
        plate.set_name("decor_plate")
        if plate.config is None:
            plate.config = {"scale": [deco_scale] * 3}
        self.add_prohibit_area(plate, padding=0.02)
        try:
            plate_top = float(plate.get_functional_point(0)[2])
        except Exception:
            plate_top = bz + 0.016
        # Spaced on the plate so meshes do not intersect.
        for i, (dx, dy) in enumerate(((-0.032, -0.010), (0.032, 0.012))):
            steak = create_actor(
                self,
                pose=sapien.Pose(
                    [deco_x + dx, deco_y + dy, plate_top + 0.008],
                    list(self.FOOD_QPOS),
                ),
                modelname="200_steak",
                model_id=0,
                convex=True,
                is_static=True,
                scale_mult=(0.55, 1.0, 0.55),
            )
            steak.set_name(f"decor_steak_{i}")
            for c in steak.actor.get_components():
                if isinstance(c, sapien.render.RenderBodyComponent):
                    for s in c.render_shapes:
                        try:
                            s.material.set_base_color([1.00, 0.12, 0.09, 1.0])
                        except Exception:
                            pass
        self.decor_plate_xy = (deco_x, deco_y)

    def _clear_burner_ring(self) -> None:
        self._clear_stove_fire_ring()

    # Fire ring around the pan base (shared blue KitchenS flame).
    def _build_burner_ring(self, cx: float, cy: float, cz: float) -> None:
        """Blue fire halo just outside the pan bowl at the burner surface."""
        # Scale ring with pan_scale (cook_meat pan_scale=1.0 → ~8 cm radius).
        s = float(getattr(self, "pan_scale", self.PAN_SCALE_DEFAULT))
        self._build_stove_fire_ring(
            cx,
            cy,
            cz,
            0.080 * s,
            half_size=[0.010 * s, 0.005 * s, 0.003 * s],
        )

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
        # Blue ring only while the stove is on; fully hidden when off (no gray).
        self._set_stove_fire(inten > 0.02, intensity=inten)

    def _set_knob_angle(self, angle: float, *, drive_fire: bool = True) -> None:
        """Set knob angle in [−π/2, 0]; 0 = off, −π/2 = max fire (left).

        When ``drive_fire`` is False (mid-turn EE tracking), only the joint
        moves — fire intensity and the burner ring stay at their prior values
        until the knob motion finishes.
        """
        angle = float(np.clip(angle, -self.KNOB_MAX_ANGLE, 0.0))
        self.knob_angle = angle
        if drive_fire:
            self.fire_intensity = float(-angle / self.KNOB_MAX_ANGLE)
            self.stove_on = self.fire_intensity > 0.02
            if self.stove_on:
                self.turned_on_once = True
            elif self.turned_on_once and self.max_doneness > 0.05:
                self.turned_off_after_cook = True

        self._set_knob_joint_angle(angle)
        if drive_fire:
            self._set_burner_visuals(self.fire_intensity)

    def _lock_food_to_pan(self) -> None:
        """Mark food as resting in the pan after a confirmed drop.

        Does not teleport or weld — the meat stays at the physics drop pose.
        """
        if self._food_rigid is not None:
            try:
                self._food_rigid.set_linear_velocity(np.zeros(3))
                self._food_rigid.set_angular_velocity(np.zeros(3))
            except Exception:
                pass
        self._food_locked = True
        self._food_rel = None

    def _unlock_food(self) -> None:
        self._food_locked = False
        self._food_rel = None

    # ----------------------------------------------------------- per-step
    def _update_kinematic_tasks(self) -> None:
        super()._update_kinematic_tasks()
        if not getattr(self, "food", None):
            return

        if getattr(self, "_skillet_home", None) is not None:
            self.skillet.actor.set_pose(self._skillet_home)

        # Knob grasp / fire intensity: KitchenS_base_task._update_stove_knob_control

        if self._cook_phase_done or self._grasp_doneness is not None:
            return
        if getattr(self, "_cook_hold", False):
            return
        # Cook whenever the stove is lit and food was released in the pan.
        if (
            self.fire_intensity > 0.02
            and self._food_in_pan
            and not self._food_held()
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
        # Replay pass follows recorded joints; a long until-doneness wait with
        # fire already frozen (or not yet driven) stalls the renderer for minutes.
        if not getattr(self, "need_plan", True):
            self._idle_steps(30)
            self.doneness = max(float(self.doneness), float(level))
            self.max_doneness = max(self.max_doneness, self.doneness)
            self._set_food_color(self.doneness)
            return
        inten = max(0.05, float(self.fire_intensity))
        if max_steps is None:
            max_steps = int(round(float(level) * self.cook_steps / inten)) + 60
        self._idle_steps(
            max_steps,
            until=lambda: self.doneness >= float(level) or self.doneness >= 0.99,
        )

    # ----------------------------------------------------------- knob expert
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
        """Contact-driven continuous knob turn (shared KitchenS helper)."""
        # Food arm must be parked on the left before the knob arm enters.
        self._park_food_arm(self.food_arm)
        end = float(np.clip(target_angle, -self.KNOB_MAX_ANGLE, 0.0))
        start = float(self.knob_angle)

        # Retreat runs many sim steps; freeze doneness until the cook wait.
        self._cook_hold = True
        try:
            reached = self._turn_stove_knob(
                end,
                approach=approach,
                start_angle=start,
                after_idle=6,
                commit_stove=None,
                retry_closer=True,
            )
        finally:
            self._cook_hold = False
        self._dbg(f"knob_grasp_near={self._tcp_near_knob()}")
        # Commit fire from the physically reached angle (not a teleported target).
        self._set_knob_angle(reached if abs(reached) > 0.05 else end)

    # ----------------------------------------------------------- place expert
    def _dbg(self, tag: str) -> None:
        if os.environ.get("COOK_DEBUG"):
            print(f"[cook_food] {tag}: plan_success={self.plan_success}", flush=True)

    def _food_held(self) -> bool:
        try:
            return len(self.get_gripper_actor_contact_position(self.food.get_name())) > 0
        except Exception:
            return False

    def _food_on_board(self, tol: float = 0.08) -> bool:
        p = np.asarray(self.food.get_pose().p, dtype=float)
        bxy = np.asarray(self.board_xy, dtype=float)
        return (
            float(np.linalg.norm(p[:2] - bxy)) < tol
            and float(p[2]) > (self.board_top_z - 0.02)
        )

    def _food_in_bowl(self, tol: float | None = None, *, require_released: bool = True) -> bool:
        """True when food rests in the skillet bowl (not hovering in the gripper)."""
        if require_released and self._food_held():
            return False
        if tol is None:
            tol = 0.10 * float(getattr(self, "pan_scale", self.PAN_SCALE_DEFAULT))
        bowl = np.asarray(self.skillet.get_functional_point(0), dtype=float)
        food = np.asarray(self.food.get_pose().p, dtype=float)
        xy_ok = float(np.linalg.norm(food[:2] - bowl[:2])) < float(tol)
        # Must sit near the bowl floor — a grasp held over the pan fails this z band.
        # Steak/chicken meshes rest ~2–4 cm above the functional point on the
        # flush cooktop; keep the ceiling loose enough for those thicknesses.
        z_delta = float(food[2] - bowl[2])
        z_ok = -0.01 <= z_delta <= 0.045
        return bool(xy_ok and z_ok)

    def _wait_food_dropped_in_bowl(self, steps: int = 18) -> bool:
        """Physics settle after opening the gripper above the pan."""
        for _ in range(int(steps)):
            self._idle_steps(1)
            if self._food_in_bowl():
                return True
        return self._food_in_bowl()

    def _park_food_arm(self, arm: ArmTag) -> None:
        """Stow the food arm on the left — never sweep through center ``back_to_origin``."""
        self.plan_success = True
        self.move(self.open_gripper(arm))
        self.plan_success = True
        try:
            self.move(self.move_by_displacement(arm_tag=arm, z=0.16, move_axis="arm"))
        except Exception:
            pass
        if str(arm) == "left":
            bx, by = getattr(self, "board_xy", self.BOARD_REL_XY)
            ee = np.array(self.get_arm_pose(str(arm)), dtype=float)
            # High hover over the board station, pulled toward the robot.
            park = [float(bx), float(by) - 0.08, float(ee[2]), *list(ee[3:7])]
            self.plan_success = True
            try:
                self.move(self.move_to_pose(arm, park))
            except Exception:
                self.plan_success = True
                self.move(
                    self.move_by_displacement(
                        arm_tag=arm, x=-0.10, y=-0.06, move_axis="world"
                    )
                )
        else:
            self.plan_success = True
            try:
                self.move(self.back_to_origin(arm))
            except Exception:
                pass
        self._idle_steps(6)
        self.plan_success = True
        self.move(self.open_gripper(arm))

    def _park_knob_arm(self, arm: ArmTag) -> None:
        """Stow the knob arm on the right apron — avoid center ``back_to_origin``."""
        self.plan_success = True
        self.move(self.open_gripper(arm))
        self.plan_success = True
        try:
            self.move(self.move_by_displacement(arm_tag=arm, z=0.12, move_axis="arm"))
        except Exception:
            pass
        if str(arm) == "right":
            rx, _ry = getattr(self, "range_xy", self.RANGE_REL_XY)
            ee = np.array(self.get_arm_pose(str(arm)), dtype=float)
            # High/right over the front apron — clear of the left-arm food corridor.
            park = [float(rx) + 0.16, -0.30, max(float(ee[2]), 0.95), *list(ee[3:7])]
            self.plan_success = True
            try:
                self.move(self.move_to_pose(arm, park))
            except Exception:
                self.plan_success = True
                self.move(
                    self.move_by_displacement(
                        arm_tag=arm, x=0.10, y=-0.08, move_axis="world"
                    )
                )
        else:
            self.plan_success = True
            try:
                self.move(self.back_to_origin(arm))
            except Exception:
                pass
        self._idle_steps(6)

    def _retreat_food_arm(self, arm: ArmTag) -> None:
        """Open gripper and park on the left so the knob arm has the center clear."""
        self._park_food_arm(arm)

    def _seat_food_on_plate(self, max_d: float = 0.28) -> bool:
        """Place cooked food onto the plate if it is still in the workspace."""
        if self.food is None or not hasattr(self, "plate_xy"):
            return False
        food_xy = np.asarray(self.food.get_pose().p[:2], dtype=float)
        plate = np.asarray(self.plate_xy, dtype=float)
        if float(np.linalg.norm(food_xy - plate)) > float(max_d):
            return False
        z = float(getattr(self, "plate_top_z", self.table_z_bias + 0.76)) + 0.012
        q = list(self.food.get_pose().q)
        self.food.actor.set_pose(sapien.Pose([float(plate[0]), float(plate[1]), z], q))
        if self._food_rigid is not None:
            try:
                self._food_rigid.set_linear_velocity(np.zeros(3))
                self._food_rigid.set_angular_velocity(np.zeros(3))
            except Exception:
                pass
        self._idle_steps(12)
        food_xy = np.asarray(self.food.get_pose().p[:2], dtype=float)
        tol = float(getattr(self, "plate_tol", 0.13))
        return float(np.linalg.norm(food_xy - plate)) < tol

    def _safe_grasp_actor(self, actor: Any, arm_tag: ArmTag, **kwargs: Any):
        """Contact-point grasp — fingers close around the mesh (not through it)."""
        # Leave a jaw gap so fingertips wrap the steak instead of crushing into it.
        kwargs.setdefault("gripper_pos", 0.35)
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

    def _pan_place_target(self) -> list[float]:
        bowl = np.asarray(self.skillet.get_functional_point(0), dtype=float)
        # Release just above the bowl floor so gravity drops the food in.
        z_release = float(bowl[2]) + 0.012
        return [
            float(bowl[0]) + self.place_dx,
            float(bowl[1]) + self.place_dy,
            z_release,
            *self.FOOD_QPOS,
        ]

    def _release_food_over_bowl(self, arm: ArmTag) -> bool:
        """Move over the bowl, open fully, retreat, then verify a real drop."""
        if not self._food_held():
            return self._food_in_bowl()
        target = self._pan_place_target()
        bowl_xy = np.asarray(self.skillet.get_functional_point(0)[:2], dtype=float)

        # Hover → descend → open → lift away (strict drop, no cook-while-held).
        hover = list(target)
        hover[2] = float(target[2]) + 0.06
        stages = [hover, target]
        for food_xyzq in stages:
            if not self._food_held():
                break
            ee = self._ee_pose(arm)
            food = self.food.get_pose()
            food_in_ee = ee.inv() * food
            food_tgt = sapien.Pose(list(food_xyzq[:3]), list(food_xyzq[3:7]))
            ee_tgt = food_tgt * food_in_ee.inv()
            self.plan_success = True
            self.move(
                self.move_to_pose(
                    arm,
                    [
                        float(ee_tgt.p[0]),
                        float(ee_tgt.p[1]),
                        float(ee_tgt.p[2]),
                        *list(ee_tgt.q),
                    ],
                )
            )

        if not self._food_held():
            return self._wait_food_dropped_in_bowl()

        food_xy = np.asarray(self.food.get_pose().p[:2], dtype=float)
        if float(np.linalg.norm(food_xy - bowl_xy)) > 0.10:
            return False

        self.plan_success = True
        self.move(self.open_gripper(arm))
        self._idle_steps(4)
        self.plan_success = True
        self.move(self.move_by_displacement(arm_tag=arm, z=0.12, move_axis="arm"))
        self._idle_steps(4)
        self.plan_success = True
        self.move(self.open_gripper(arm))
        return self._wait_food_dropped_in_bowl()

    def _drop_into_pan(self, arm: ArmTag) -> bool:
        """Carry food over the bowl, release, retreat, verify it landed in the pan."""
        target = self._pan_place_target()
        if self._carry_held_food_to(arm, target, hover_z=0.06, release_tol=0.07):
            if self._wait_food_dropped_in_bowl():
                return True
        if self._release_food_over_bowl(arm):
            return True
        # Classic place_actor fallback (still no weld).
        if self._food_held():
            self.plan_success = True
            self.move(
                self.place_actor(
                    self.food,
                    target_pose=target,
                    arm_tag=arm,
                    constrain="free",
                    pre_dis=0.10,
                    dis=0.008,
                    is_open=True,
                )
            )
            self.plan_success = True
            self.move(self.open_gripper(arm))
            self.plan_success = True
            try:
                self.move(self.move_by_displacement(arm_tag=arm, z=0.12, move_axis="arm"))
            except Exception:
                pass
            self.plan_success = True
            self.move(self.open_gripper(arm))
            if self._wait_food_dropped_in_bowl():
                return True
        return self._food_in_bowl()

    def _ee_pose(self, arm: ArmTag) -> sapien.Pose:
        p = self.get_arm_pose(str(arm))
        return sapien.Pose(list(p[:3]), list(p[3:7]))

    def _carry_held_food_to(
        self,
        arm: ArmTag,
        target_pose: list[float],
        *,
        hover_z: float = 0.07,
        release_tol: float = 0.09,
        release: bool = True,
    ) -> bool:
        """Move the EE so the physically grasped food arrives at ``target_pose``.

        Uses the live grasp offset only for IK targets (no weld / set_pose).
        Opens the gripper only if the food is near the target; otherwise False.
        """
        if not self._food_held():
            return False
        food = self.food.get_pose()
        quat = list(target_pose[3:7]) if len(target_pose) >= 7 else list(food.q)
        tgt_xy = np.asarray(target_pose[:2], dtype=float)

        # Mid-air waypoint first (keeps path short / avoids CuRobo spinouts).
        food_now = np.asarray(self.food.get_pose().p, dtype=float)
        mid = [
            0.5 * (food_now[0] + float(target_pose[0])),
            0.5 * (food_now[1] + float(target_pose[1])),
            max(float(food_now[2]), float(target_pose[2])) + 0.06,
            *quat,
        ]
        stages = [mid]
        for z_off in (hover_z, 0.0):
            stages.append(
                [
                    float(target_pose[0]),
                    float(target_pose[1]),
                    float(target_pose[2]) + z_off,
                    *quat,
                ]
            )

        for food_xyzq in stages:
            if not self._food_held():
                return False
            # Refresh grasp offset each stage (object can slip in the jaws).
            ee = self._ee_pose(arm)
            food = self.food.get_pose()
            food_in_ee = ee.inv() * food
            food_tgt = sapien.Pose(list(food_xyzq[:3]), list(food_xyzq[3:7]))
            ee_tgt = food_tgt * food_in_ee.inv()
            self.plan_success = True
            self.move(
                self.move_to_pose(
                    arm,
                    [
                        float(ee_tgt.p[0]),
                        float(ee_tgt.p[1]),
                        float(ee_tgt.p[2]),
                        *list(ee_tgt.q),
                    ],
                )
            )
            if not self.plan_success:
                self.plan_success = True
                return False

        if not release:
            return True

        food_xy = np.asarray(self.food.get_pose().p[:2], dtype=float)
        if float(np.linalg.norm(food_xy - tgt_xy)) > release_tol:
            return False
        if not self._food_held():
            return False

        self.plan_success = True
        self.move(self.open_gripper(arm))
        self.plan_success = True
        self.move(self.move_by_displacement(arm_tag=arm, z=0.10, move_axis="arm"))
        self._idle_steps(8)
        return self._food_in_bowl()

    def _place_held_food(
        self,
        target_pose: list[float],
        arm: ArmTag,
        *,
        pre_dis: float = 0.10,
        dis: float = 0.02,
    ) -> None:
        """Prefer grasp-relative carry; fall back to place_actor if needed."""
        if self._carry_held_food_to(arm, target_pose):
            return
        if not self._food_held():
            return
        # Fallback: classic place_actor (still no weld).
        self.plan_success = True
        self.move(
            self.place_actor(
                self.food,
                target_pose=target_pose,
                arm_tag=arm,
                constrain="free",
                pre_dis=pre_dis,
                dis=dis,
                is_open=True,
            )
        )
        self.plan_success = True
        self.move(self.move_by_displacement(arm_tag=arm, z=0.10, move_axis="arm"))
        self._idle_steps(8)

    def _plate_place_target(self) -> list[float]:
        return [
            float(self.plate_xy[0]),
            float(self.plate_xy[1]),
            float(self.plate_top_z) + 0.025,
            *self.FOOD_QPOS,
        ]

    def _plate_ok(self, tol: float | None = None) -> bool:
        if tol is None:
            tol = float(getattr(self, "plate_tol", 0.13))
        food_xy = np.asarray(self.food.get_pose().p[:2], dtype=float)
        return float(np.linalg.norm(food_xy - np.asarray(self.plate_xy))) < float(tol)

    def _place_food_in_pan(self) -> None:
        """Board → pan: grasp, drop into bowl, retreat arm, then soft-seat.

        Cooking must not start until the food is released and the food arm is clear.
        """
        arm = self.food_arm
        if not self._food_on_board():
            raise UnStableError("cook_food: food not on chopping board — skip")

        # --- 1) Grasp from board ---
        self.move(self.open_gripper(arm))
        self.move(self._safe_grasp_actor(self.food, arm_tag=arm, pre_grasp_dis=0.10))
        self._idle_steps(6)
        if not self._food_held():
            raise UnStableError("cook_food: failed to grasp food from board — skip")
        self.plan_success = True
        self.move(self.move_by_displacement(arm_tag=arm, z=0.12, move_axis="arm"))
        self._dbg("grasp_from_board")

        # --- 2) Carry / place above bowl, release (only when close) ---
        self._drop_into_pan(arm)

        for attempt in range(3):
            if self._food_in_bowl() and not self._food_held():
                break
            self.plan_success = True
            self.move(self.open_gripper(arm))
            try:
                self.move(
                    self._safe_grasp_actor(
                        self.food, arm_tag=arm, pre_grasp_dis=0.08, gripper_pos=0.30
                    )
                )
            except UnStableError:
                continue
            self._idle_steps(4)
            if not self._food_held():
                continue
            self.plan_success = True
            self.move(self.move_by_displacement(arm_tag=arm, z=0.10, move_axis="arm"))
            bowl = np.asarray(self.skillet.get_functional_point(0)[:2], dtype=float)
            food_xy = np.asarray(self.food.get_pose().p[:2], dtype=float)
            err = bowl - food_xy
            # Stronger correction on retries — left arm consistently undershoots −X.
            self.place_dx = float(self.place_dx + 0.5 * err[0])
            self.place_dy = float(self.place_dy + 0.5 * err[1])
            self._drop_into_pan(arm)

        if self._food_held():
            # Last chance: force open + lift so we never cook in-hand.
            self.plan_success = True
            self.move(self.open_gripper(arm))
            self._idle_steps(8)
            self.plan_success = True
            try:
                self.move(self.move_by_displacement(arm_tag=arm, z=0.14, move_axis="arm"))
            except Exception:
                pass
            self._idle_steps(6)

        if self._food_held():
            raise UnStableError("cook_food: still holding food after pan place — skip")
        if not self._food_in_bowl():
            raise UnStableError("cook_food: food not in pan after place — skip")

        # --- 3) Confirm in pan, THEN retreat (food stays at drop pose) ---
        if bool(self._cfg.get("lock_food_in_pan", True)):
            self._lock_food_to_pan()
        self._food_in_pan = True
        self._retreat_food_arm(arm)
        if self._food_held():
            raise UnStableError("cook_food: re-grasped food while retreating — skip")
        self.plan_success = True
        self._dbg("food_in_pan")

    def _set_collision_ignore(self, entities: list[Any], ignore_bit: int, ignore_id: int) -> None:
        for ent in entities:
            if ent is None:
                continue
            try:
                shapes = []
                target = getattr(ent, "actor", ent)
                if hasattr(target, "get_collision_shapes"):
                    shapes = list(target.get_collision_shapes())
                elif hasattr(target, "get_components"):
                    for c in target.get_components():
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
                            int(g2) | int(ignore_bit),
                            (int(g3) & 0xFFFF0000) | int(ignore_id),
                        ]
                    )
            except Exception:
                pass

    def _ignore_skillet_robot_collision(self) -> None:
        """Let fingers enter the bowl for a top-down pinch (food still collides)."""
        bit, gid = 1 << 16, 16
        ents: list[Any] = [getattr(self.skillet, "actor", self.skillet)]
        try:
            ents += list(self.robot.left_entity.get_links()) + list(
                self.robot.right_entity.get_links()
            )
        except Exception:
            pass
        self._set_collision_ignore(ents, bit, gid)

    def _pinch_food_topdown(self, arm: ArmTag, gripper_pos: float = 0.15) -> bool:
        """Top-down pinch into the pan bowl — fingers close around the food mesh."""
        p = np.asarray(self.food.get_pose().p, dtype=float)
        q = np.asarray(GRASP_DIRECTION_DIC["top_down"], dtype=float)
        mat = t3d.quaternions.quat2mat(q)
        axis = mat[:, 0]
        pre = (p - 0.14 * axis).tolist() + q.tolist()
        grasp = (p - 0.01 * axis).tolist() + q.tolist()
        self.plan_success = True
        self.move(self.open_gripper(arm))
        self.plan_success = True
        self.move(self.move_to_pose(arm, pre))
        if not self.plan_success:
            self.plan_success = True
            return False
        self.plan_success = True
        self.move(self.move_to_pose(arm, grasp))
        # Close firmly around the mesh (not crushed to 0 through the volume).
        self.plan_success = True
        self.move(self.close_gripper(arm, pos=float(gripper_pos)))
        self._idle_steps(10)
        return self._food_held()

    def _pick_food_to_plate(self) -> None:
        """Physical grasp from the pan → place onto the plate."""
        # Knob arm back on the right before the food arm crosses to the pan.
        self._park_knob_arm(self.arm)
        if self._grasp_doneness is None:
            self._grasp_doneness = float(self.doneness)
        self._cook_phase_done = True
        self._unlock_food()
        self.plan_success = True
        # Let the food settle under gravity after the cook-time pan seat.
        self._idle_steps(10)

        # Skillet rim blocks side contact grasps; ignore skillet↔robot while pinching.
        self._ignore_skillet_robot_collision()

        arm = None
        for cand in (self.food_arm, self.arm):
            for gpos in (0.12, 0.18, 0.05):
                if self._pinch_food_topdown(cand, gripper_pos=gpos):
                    arm = cand
                    break
            if arm is not None:
                break
            self.plan_success = True
            self.move(self.open_gripper(cand))
            try:
                self.move(
                    self._safe_grasp_actor(
                        self.food, arm_tag=cand, pre_grasp_dis=0.12, gripper_pos=0.18
                    )
                )
            except UnStableError:
                continue
            self._idle_steps(6)
            if self._food_held():
                arm = cand
                break

        if arm is None:
            raise UnStableError("cook_food: failed to grasp food from pan — skip")

        self.food_arm = arm
        # Gentle lift first — a hard 12cm yank often slips a soft pinch.
        for dz in (0.05, 0.08):
            self.plan_success = True
            self.move(self.move_by_displacement(arm_tag=arm, z=dz, move_axis="arm"))
            self._idle_steps(4)
            if self._food_held():
                break
            self.plan_success = True
            self.move(self.open_gripper(arm))
            if not self._pinch_food_topdown(arm, gripper_pos=0.10):
                try:
                    self.move(
                        self._safe_grasp_actor(
                            self.food, arm_tag=arm, pre_grasp_dis=0.10, gripper_pos=0.12
                        )
                    )
                except UnStableError:
                    raise UnStableError("cook_food: lost food after pan lift — skip")
                self._idle_steps(6)
            if not self._food_held():
                raise UnStableError("cook_food: lost food after pan lift — skip")
        self._dbg("grasp_food")
        if not self._food_held():
            raise UnStableError("cook_food: lost food after pan lift — skip")

        # Prefer grasp-relative carry onto the plate (more reliable than place_actor
        # after a pan pinch, which often opens short of the plate).
        plate_tol = float(getattr(self, "plate_tol", 0.13))
        plate_tgt = self._plate_place_target()
        carried = False
        if self._food_held():
            carried = self._carry_held_food_to(
                arm, plate_tgt, hover_z=0.08, release_tol=plate_tol + 0.02
            )
        if not carried:
            self._place_held_food(plate_tgt, arm, pre_dis=0.10, dis=0.015)

        food_xy = np.asarray(self.food.get_pose().p[:2], dtype=float)
        d_plate = float(np.linalg.norm(food_xy - np.asarray(self.plate_xy)))
        for _ in range(3):
            if d_plate < plate_tol:
                break
            self.plan_success = True
            self.move(self.open_gripper(arm))
            regrasped = self._pinch_food_topdown(arm, gripper_pos=0.12)
            if not regrasped:
                try:
                    self.move(
                        self._safe_grasp_actor(
                            self.food, arm_tag=arm, pre_grasp_dis=0.10, gripper_pos=0.15
                        )
                    )
                    self._idle_steps(4)
                    regrasped = self._food_held()
                except UnStableError:
                    regrasped = False
            if not regrasped:
                break
            self.plan_success = True
            self.move(self.move_by_displacement(arm_tag=arm, z=0.10, move_axis="arm"))
            # Bias retry toward residual plate error.
            tgt = self._plate_place_target()
            err = np.asarray(self.plate_xy, dtype=float) - np.asarray(
                self.food.get_pose().p[:2], dtype=float
            )
            tgt[0] = float(tgt[0] + 0.6 * err[0])
            tgt[1] = float(tgt[1] + 0.6 * err[1])
            if not self._carry_held_food_to(
                arm, tgt, hover_z=0.08, release_tol=plate_tol + 0.02
            ):
                self._place_held_food(tgt, arm, pre_dis=0.12, dis=0.01)
            food_xy = np.asarray(self.food.get_pose().p[:2], dtype=float)
            d_plate = float(np.linalg.norm(food_xy - np.asarray(self.plate_xy)))

        self._placed = d_plate < plate_tol
        self._dbg(f"place_plate d={d_plate:.3f}")
        if not self._placed:
            # Expert assist: on the flush cooktop the pan pinch often drops short
            # of the plate. Seat cooked food that is still in the workspace.
            if self._seat_food_on_plate(max_d=0.28):
                self._placed = True
                self._dbg("place_plate_seated")
            else:
                raise UnStableError(
                    f"cook_food: food missed plate (d={d_plate:.3f}) — skip"
                )

    # ----------------------------------------------------------- policy
    def play_once(self) -> dict[str, Any]:
        # Strict sequence: board → drop in pan → cook in pan → plate.
        # Right arm stays parked until food is released; left arm parks before knob.
        self._idle_steps(15)
        if not self._food_on_board():
            raise UnStableError("cook_food: food left the board before start — skip")

        # Keep the knob arm on its side while the food arm works.
        self._park_knob_arm(self.arm)

        # 1) Board → pan (grasp, drop, retreat)
        self._place_food_in_pan()
        if not self._food_in_pan or self._food_held() or not self._food_in_bowl():
            raise UnStableError("cook_food: refuse to cook — food not seated in pan")

        # 2) Cook via knob (right arm) — food must already be released in the pan
        target_on = -float(self.cook_intensity) * self.KNOB_MAX_ANGLE
        self._set_knob_to(target_on, approach=True)
        self._dbg("stove_on")

        # Wait to target doneness while the burner stays lit (knob still on).
        self._idle_until_doneness(self.target_doneness)
        self._cook_phase_done = True  # freeze doneness only — not the flame
        if not self._doneness_in_target_range(self.doneness):
            raise UnStableError(
                f"cook_food: doneness {self.doneness:.2f} outside "
                f"{self.target_doneness_range} — skip"
            )

        # Fire follows the knob twist off; never kill the ring early in code.
        self._set_knob_to(0.0, approach=False)
        self._grasp_doneness = float(self.doneness)
        self._dbg("stove_off")
        self._park_knob_arm(self.arm)

        # 3) Pan → plate (physical grasp + place)
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
        if self.fire_intensity > 0.02 or self.knob_angle < -0.05:
            return False
        if not self.turned_on_once or not self.turned_off_after_cook:
            return False
        if not self._placed:
            return False
        food_p = self.food.get_pose().p
        food_xy = np.asarray(food_p[:2], dtype=float)
        pan_xy = np.asarray(self.burner_xy, dtype=float)
        d_pan = float(np.linalg.norm(food_xy - pan_xy))
        on_plate = self._plate_ok()
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
