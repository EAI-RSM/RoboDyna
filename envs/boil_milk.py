"""Boil milk on a KitchenS cooking range without letting it overflow.

KitchenS scene with a center sink+tap and a cooking range on the side. A pot of
milk sits on a burner. The robot turns the stove on; the milk level rises while
boiling. If left on, it would spill. The robot must turn the stove off before
overflow; once off, the milk settles back to its resting level.
"""
from __future__ import annotations

import numpy as np
import sapien
import sapien.render
import transforms3d as t3d

from ._kitchens_base_task import KitchenS_base_task
from ._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
from .utils import *
from .utils.create_actor import create_box, UnStableError


class boil_milk(KitchenS_base_task):
    """Turn the range on, let the milk rise, then shut it off before it spills."""

    BOIL_STEPS_DEFAULT = 6000         # steps for liquid_level 0→1 while stove is on
    SETTLE_STEPS_DEFAULT = 1200       # steps for liquid_level → baseline while stove is off
    BASELINE_LEVEL_DEFAULT = 0.35     # resting fill fraction of pot height
    EXPERT_SHUTOFF_LEVEL_DEFAULT = 0.62  # expert starts the shutoff reach here
    OVERFLOW_LEVEL_DEFAULT = 0.95     # spill threshold (failure if reached while on)
    KNOB_CONTACT_RADIUS_DEFAULT = 0.06   # pinch-to-knob distance that counts as held
    EE_TO_TCP = 0.12                  # EE frame → TCP offset (cook_meat / dispense_gummy)
    # Waypoints relative to the knob centre: drop the wrist beside the counter,
    # then track in horizontally at knob height. See ``_turn_knob``.
    KNOB_APPROACH_PATH = (
        (-0.13, -0.33, 0.06),
        (-0.08, -0.33, 0.00),
        (0.00, -0.25, 0.00),
    )
    KNOB_GRASP_STANDOFF = 0.015       # jaw clearance short of the knob centre

    def setup_demo(self, **kwags):
        self._cfg = kwags.get("task_args", {}).get("boil_milk", {})
        # Prefer scene_0 so the range is on the right (reachable by the right arm).
        if kwags.get("scene_id") is None:
            kwags["scene_id"] = int(self._cfg.get("scene_id", 0))
        self.replace_sink_with_range = True

        # Per-step state must exist before any early _update_kinematic_tasks call.
        self.stove_on = False
        self.liquid_level = float(self._cfg.get("baseline_level", self.BASELINE_LEVEL_DEFAULT))
        self.baseline_level = float(self.liquid_level)
        self.max_liquid_level = float(self.liquid_level)
        self.overflowed = False
        self.turned_on_once = False
        self.turned_off_after_boil = False
        self._liquid_entity = None
        self._burner_shapes = []
        self._knob_press_latched = False
        self._prev_knob_pressed = False
        self._ignore_knob = False
        self._expert_holding_knob = False

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

        if not hasattr(self, "burner_xy"):
            raise UnStableError("cooking range missing — KitchenS base did not load a range")

        # Procedural hollow pot (URDF 060_kitchenpot uses non-unit scale and
        # trips mplib/CuRobo). Build an open metal vessel from a bottom disc
        # and segmented side walls so the liquid surface is directly visible.
        self.pot_id = int(cfg.get("pot_id", 0))
        pot_r = float(cfg.get("pot_radius", 0.055))
        pot_h = float(cfg.get("pot_height", 0.075))
        wall = 0.005
        pot_z0 = float(self.range_top_z)
        cx, cy = float(self.burner_xy[0]), float(self.burner_xy[1])
        vertical_q = [0.70710678, 0.0, 0.70710678, 0.0]

        metal = sapien.render.RenderMaterial(base_color=[0.55, 0.55, 0.58, 1.0])
        metal.metallic = 0.7
        metal.roughness = 0.35

        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        # Floor (opaque metal)
        builder.add_cylinder_collision(
            pose=sapien.Pose([0, 0, wall / 2], vertical_q),
            radius=pot_r, half_length=wall / 2,
            material=self.scene.default_physical_material,
        )
        builder.add_cylinder_visual(
            pose=sapien.Pose([0, 0, wall / 2], vertical_q),
            radius=pot_r, half_length=wall / 2, material=metal,
        )

        # Faceted cylindrical wall. Each box is tangent to the circle; together
        # they form a genuinely hollow, open-top pot rather than a solid cylinder.
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
        self.pot = builder.build(name="kitchen_pot")
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
        self.pot_xy = (cx, cy)

        self._rebuild_liquid(force=True)
        self._set_burner_glow(False)

        # Arm that can reach the knob (range is always on +x in scene_0 / scene_2).
        self.arm = ArmTag("right" if self.knob_xy[0] >= 0 else "left")

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
        color = [0.95, 0.35, 0.05, 1.0] if on else [0.20, 0.20, 0.22, 1.0]
        for s in getattr(self, "_burner_shapes", []) or []:
            try:
                s.material.set_base_color(color)
            except Exception:
                pass
        # Rotate the knob's indicator instead of changing its color: off is
        # 12 o'clock and on is 3 o'clock, like a physical range control.
        if getattr(self, "stove_knob_indicator", None) is not None:
            angle = -np.pi / 2 if on else 0.0
            radius = float(self._knob_radius) * 0.55
            kx, _, kz = self.knob_xyz
            indicator_pose = sapien.Pose(
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
            self.stove_knob_indicator.set_pose(indicator_pose)

    def _set_stove(self, on: bool):
        on = bool(on)
        if on == self.stove_on:
            return
        self.stove_on = on
        if on:
            self.turned_on_once = True
        elif self.turned_on_once and self.max_liquid_level > self.baseline_level + 0.05:
            self.turned_off_after_boil = True
        self._set_burner_glow(on)

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

        # Boiling is driven purely by the burner state: the level keeps climbing
        # while the arm reaches for the knob, and only stops once it is turned off.
        if self.stove_on:
            self.liquid_level = min(
                1.0, self.liquid_level + 1.0 / max(1, self.boil_steps)
            )
            self.max_liquid_level = max(self.max_liquid_level, self.liquid_level)
            if self.liquid_level >= self.overflow_level:
                self.overflowed = True
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
        """Grasp pose whose jaws close around the knob, ``standoff`` short of it."""
        return self._knob_pose(
            [0.0, -(self.EE_TO_TCP + float(standoff)), 0.0], turn_angle
        )

    def _turn_knob(
        self,
        want_on: bool,
        approach_from_far: bool = True,
        retreat_far: bool = True,
    ):
        """Reach in along the low corridor, grasp the knob, and twist the wrist.

        The knob sits on the range's front panel, below the height the arm can
        reach by diving straight down onto it: from the home pose the planner
        bottoms out ~7 cm high. Dropping the wrist clear of the counter first and
        then sliding in horizontally keeps the elbow in the low branch that can.

        Once the hand is already in front of the panel that detour is redundant,
        so the caller can ask for the short in/out — which matters here, because
        the pot keeps boiling for every step the arm spends travelling.
        """
        arm = self.arm
        start_angle = -np.pi / 2 if self.stove_on else 0.0
        end_angle = -np.pi / 2 if bool(want_on) else 0.0
        path_in = self.KNOB_APPROACH_PATH if approach_from_far else self.KNOB_APPROACH_PATH[-1:]
        path_out = self.KNOB_APPROACH_PATH if retreat_far else self.KNOB_APPROACH_PATH[-1:]

        self._ignore_knob = True
        self.move(self.open_gripper(arm))
        for offset in path_in:
            self.move(self.move_to_pose(arm, self._knob_pose(offset, start_angle)))
        self.move(self.move_to_pose(
            arm, self._knob_turn_pose(self.KNOB_GRASP_STANDOFF, start_angle)
        ))
        self.move(self.close_gripper(arm))

        # The wrist twist is the actual control gesture. Latch contact while
        # moving so simulator stepping cannot create an accidental extra toggle.
        self._expert_holding_knob = True
        self.move(self.move_to_pose(
            arm, self._knob_turn_pose(self.KNOB_GRASP_STANDOFF, end_angle)
        ))
        self._expert_holding_knob = False
        self._set_stove(bool(want_on))
        self._idle_steps(10)

        self.move(self.open_gripper(arm))
        for offset in reversed(path_out):
            self.move(self.move_to_pose(arm, self._knob_pose(offset, end_angle)))
        self._ignore_knob = False
        self._prev_knob_pressed = False

    def play_once(self):
        arm = self.arm

        # 1) Turn the stove ON, then hold station in front of the control panel.
        self._turn_knob(want_on=True, retreat_far=False)

        # 2) Watch the pot. The level keeps climbing through the whole reach for
        #    the knob, so the expert commits to the shutoff early enough that the
        #    motion itself still finishes below the spill line.
        self._idle_until_level(self.expert_shutoff_level)

        # 3) Turn the stove OFF before overflow.
        self._turn_knob(want_on=False, approach_from_far=False)

        # 4) Let the liquid settle back toward baseline (visible in the video).
        settle = max(0.0, self.liquid_level - self.baseline_level)
        steps_settle = int(round(settle * self.settle_steps))
        self._idle_steps(max(1, min(steps_settle, self.settle_steps)))

        self.info["info"] = {
            "{A}": "kitchen_pot",
            "{B}": "cooking_range",
            "{C}": "stove_knob",
            "{a}": str(arm),
        }
        return self.info

    def check_success(self):
        """Success: stove was turned on (liquid rose), then off before spill, and is off."""
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
        }
        return obs
