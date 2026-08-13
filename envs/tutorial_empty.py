"""Empty table + dual UR5 playground for the interactive tutorial.

Parts 1–2 leave the table bare. Part 3 spawns one prop at a time
(cube, hold button, on/off switch, push box) via ``tutorial_set_stage``.
Part 4 reuses the same hook for advanced actions (rolling ball, stove
knob, mallet, multi-stage force key).
"""
from __future__ import annotations

import sapien
import sapien.physx
import sapien.render
import numpy as np

from ._base_task import Base_Task
from ._kitchens_base_task import KitchenS_base_task
from .utils.actor_utils import Actor
from .utils.create_actor import create_box, create_sphere, preprocess
from .utils.reactive_button import ReactivePushButtons, add_key_base_border


class _ForceKeyBank(ReactivePushButtons):
    """fill_coffee-style spring key that reports live force and allows full travel."""

    def live_force(self, button_id=0) -> float:
        idx = self.resolve_index(button_id)
        home_xy = np.asarray(self.home_poses[idx].p[:2], dtype=float)
        top_z = float(self.tops_z[idx])
        engage_z = top_z + self.force_engage_slack
        best = 0.0
        contact = self._contact_force(idx)
        for side in self._sides_for_button(idx):
            tip = self._tip_xyz(side)
            if tip is None:
                continue
            if float(np.linalg.norm(tip[:2] - home_xy)) > self.xy_tol:
                continue
            spring = self.force_stiffness * max(0.0, engage_z - float(tip[2]))
            best = max(best, spring, contact)
        return float(best)

    def min_ee_z_over_key(self, xy, *, margin: float = 0.003) -> float | None:
        """Floor Q at full force (level 4), not the first trigger depth."""
        xy = np.asarray(xy, dtype=float)[:2]
        floor = None
        for i in range(self._n):
            home_xy = np.asarray(self.home_poses[i].p[:2], dtype=float)
            if float(np.linalg.norm(xy - home_xy)) > self.xy_tol:
                continue
            ee_floor = self.tip_z_at_full_press(i) - float(margin) + self.ee_to_tcp
            floor = ee_floor if floor is None else min(floor, ee_floor)
        return floor


class tutorial_empty(Base_Task):
    """Bare tabletop with both arms at home. Used by tutorial parts 1–4."""

    # Left-side workspace so the selected left arm can reach every prop.
    PROP_X = -0.16
    PROP_Y = -0.08
    TABLE_Z = 0.74  # create_box adds table_z_bias
    EE_TO_TCP = 0.12

    CUBE_HALF = 0.022
    CUBE_COLOR = (0.95, 0.55, 0.12)
    CUBE_MASS = 0.05
    CUBE_LIFT = 0.035  # m above rest before a pick counts
    GRASP_CONFIRM_STEPS = 20

    KEY_HALF = (0.020, 0.020, 0.014)
    KEY_COLOR_UP = (0.18, 0.78, 0.28)
    KEY_COLOR_DOWN = (0.85, 0.10, 0.10)
    KEY_XY_TOL = 0.06
    HOLD_MIN_STEPS = 60  # ~0.24 s at 250 Hz — a tap does not count
    HOLD_RELEASE_STEPS = 12

    PUSH_HALF = (0.035, 0.035, 0.022)
    PUSH_COLOR = (0.28, 0.68, 0.82)
    PUSH_MASS = 0.10
    PUSH_Y = -0.14
    PUSH_GOAL_DY = 0.10
    TARGET_COLOR = (0.15, 0.85, 0.25)
    PUSH_CONFIRM_STEPS = 20
    SETTLE_STEPS = 45  # ignore success for a beat after a new prop appears

    # Part 4 — rolling ball (toward the robot, −Y).
    BALL_RADIUS = 0.024
    BALL_COLOR = (0.92, 0.18, 0.16)
    BALL_MASS = 0.04
    BALL_X = -0.16
    BALL_Y0 = 0.16
    BALL_VY = -0.12
    BALL_LIFT = 0.030
    BALL_TABLE_X_LIM = 0.42
    BALL_TABLE_Y_MIN = -0.32
    BALL_TABLE_Y_MAX = 0.34
    BALL_RESPAWN_SETTLE = 20

    # Part 4 — KitchenS cooktop (knob on the left workspace).
    STOVE_XY = (-0.28, 0.10)
    STOVE_BURNER = "left_front"
    KNOB_MAX_ANGLE = float(np.pi / 2)
    STOVE_ON_FRAC = 0.25
    STOVE_OFF_FRAC = 0.08

    # Part 4 — mallet (same geometry as whack_moles).
    MALLET_HEAD_RADIUS = 0.050
    MALLET_HEAD_HALF_X = 0.030
    MALLET_HANDLE_RADIUS = 0.012
    MALLET_HANDLE_HALF_Y = 0.130
    MALLET_HEAD_Y = -0.105
    MALLET_HANDLE_CENTER_Y = -0.025
    MALLET_WOOD_COLOR = [0.48, 0.25, 0.10]
    MALLET_REST_HEIGHT = 0.115
    MALLET_LEVEL_Q = [0.0, 0.70710678, 0.0, 0.70710678]
    MALLET_REST_POST_Y = (-0.065, 0.035)
    MALLET_MASS = 0.06
    MALLET_LIFT = 0.040

    # Part 4 — fill_coffee force key (four thresholds, target = level 2).
    FORCE_THRESHOLDS = (3.0, 6.0, 10.0, 14.0)
    FORCE_TARGET_LEVEL = 2
    FORCE_KEY_COLOR = (0.08, 0.36, 0.95)
    FORCE_LEVEL_COLORS = (
        (0.08, 0.36, 0.95),
        (0.18, 0.78, 0.28),
        (0.92, 0.78, 0.12),
        (0.95, 0.50, 0.10),
        (0.85, 0.10, 0.10),
    )
    DISPENSE_Z_DOWN_SCALE = 0.22
    # Park unused props off-table. Never scene.remove_entity at runtime —
    # deleting a body still in gripper contact segfaults PhysX (exit -11).
    HIDE_XY = (1.85, 1.85)

    def setup_demo(self, **kwags):
        self._tutorial_complete = False
        self._tutorial_stage = None
        self._props_ready = False
        self.arm = "left"
        self._tutorial_actors = []
        self._reactive_buttons = None
        self._cube = None
        self._cube_rest_z = None
        self._cube_table_pose = None
        self._push_box = None
        self._push_goal_y = None
        self._push_table_pose = None
        self._push_goal = None
        self._push_goal_table_pose = None
        self._key_actor = None
        self._key_bezel = None
        self._key_table_pose = None
        self._bezel_table_poses = []
        self._key_shapes = []
        self._key_color_down = None
        self._hold_was_pressed = False
        self._hold_press_steps = 0
        self._hold_release_steps = 0
        self._hold_complete = False
        self._switch_on = False
        self._switch_complete = False
        self._switch_touch_latched = False
        self._grasp_hold_steps = 0
        self._grasp_complete = False
        self._push_hold_steps = 0
        self._push_complete = False
        self._stage_settle = 0
        self._defer_hide_key = False
        self._ball = None
        self._ball_rigid = None
        self._ball_start = None
        self._ball_rest_z = None
        self._ball_complete = False
        self._ball_ever_lifted = False
        self._mallet = None
        self._mallet_rests = []
        self._mallet_table_pose = None
        self._mallet_rest_poses = []
        self._mallet_rest_z = None
        self._mallet_complete = False
        self._force_key_actor = None
        self._force_key_bezel = None
        self._force_key_table_pose = None
        self._force_key_bezel_poses = []
        self._stove_loaded = False
        self._stove_range_home = None
        self._stove_knob_home = None
        self._stove_cover_homes = {}
        self._stove_complete = False
        self._stove_turned_on = False
        self._stove_turned_off = False
        self.stove_on = False
        self.knob_angle = 0.0
        self.fire_intensity = 0.0
        self.stove_knob_articulation = None
        self._knob_grasp_active = False
        self._knob_clutch_engaged = False
        self._ignore_knob = False
        self._expert_holding_knob = False
        self._policy_controlling_knob = False
        self._knob_grasp_arm = None
        self._last_committed_knob_angle = None
        self._force_complete = False
        self._force_target_level = int(self.FORCE_TARGET_LEVEL)
        self._key_force_n = 0.0
        self._press_peak_force = 0.0
        self._press_active = False
        self._force_feedback = ""
        super()._init_task_env_(**kwags)

    def load_actors(self):
        return

    def play_once(self):
        pass

    def check_success(self):
        return bool(getattr(self, "_tutorial_complete", False))

    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
        self._maybe_park_key()
        stage = getattr(self, "_tutorial_stage", None)
        if not stage:
            return
        if stage == "grasp":
            self._tick_grasp()
        elif stage in ("hold", "switch"):
            self._tick_buttons()
        elif stage == "push":
            self._tick_push()
        elif stage == "ball":
            self._tick_ball()
        elif stage == "stove":
            self._tick_stove()
        elif stage == "mallet":
            self._tick_mallet()
        elif stage == "force_key":
            self._tick_force_key()

    # ---------------------------------------------------------- stage props
    def tutorial_set_stage(self, stage: str | None) -> None:
        """Show the next prop. Hold → switch keeps the same key in place."""
        prev = self._tutorial_stage
        self._reactive_buttons = None
        self._hold_was_pressed = False
        self._hold_press_steps = 0
        self._hold_release_steps = 0
        self._hold_complete = False
        self._switch_on = False
        self._switch_complete = False
        self._switch_touch_latched = False
        self._grasp_hold_steps = 0
        self._grasp_complete = False
        self._push_hold_steps = 0
        self._push_complete = False
        self._ball_complete = False
        self._ball_ever_lifted = False
        self._mallet_complete = False
        self._stove_complete = False
        self._stove_turned_on = False
        self._stove_turned_off = False
        self.stove_on = False
        self.knob_angle = 0.0
        self.fire_intensity = 0.0
        self._knob_grasp_active = False
        self._knob_clutch_engaged = False
        self._ignore_knob = False
        self._expert_holding_knob = False
        self._policy_controlling_knob = False
        self._knob_grasp_arm = None
        self._last_committed_knob_angle = None
        self._force_complete = False
        self._key_force_n = 0.0
        self._press_peak_force = 0.0
        self._press_active = False
        self._force_feedback = ""
        self._stage_settle = int(self.SETTLE_STEPS) if stage else 0
        self._tutorial_stage = stage
        if stage in ("grasp", "hold", "switch", "push"):
            self._ensure_props()
        # Do not teleport the key out from under the gripper — that PhysX
        # contact break is the green-screen SIGSEGV (-11) after a hold release.
        if stage == "switch" and prev == "hold" and self._key_actor is not None:
            self._init_key_bank("switch")
            self._set_key_color(False)
            return
        if stage == "push" and prev in ("hold", "switch") and self._key_actor is not None:
            self._show_push_box()
            self._defer_hide_key = True
            return
        self._defer_hide_key = False
        self._hide_all_props()
        if stage == "grasp":
            self._show_cube()
        elif stage == "hold":
            self._show_key("hold")
        elif stage == "switch":
            self._show_key("switch")
        elif stage == "push":
            self._show_push_box()
        elif stage == "ball":
            self._ensure_ball()
            self._respawn_ball()
        elif stage == "stove":
            self._ensure_stove()
            self._show_stove()
        elif stage == "mallet":
            self._ensure_mallet()
            self._show_mallet()
        elif stage == "force_key":
            self._ensure_force_key()
            self._show_force_key()

    def tutorial_stage_complete(self) -> bool:
        stage = self._tutorial_stage
        if stage == "grasp":
            return bool(self._grasp_complete)
        if stage == "hold":
            return bool(self._hold_complete)
        if stage == "switch":
            return bool(self._switch_complete)
        if stage == "push":
            return bool(self._push_complete)
        if stage == "ball":
            return bool(self._ball_complete)
        if stage == "stove":
            return bool(self._stove_complete)
        if stage == "mallet":
            return bool(self._mallet_complete)
        if stage == "force_key":
            return bool(self._force_complete)
        return False

    def _ensure_props(self) -> None:
        """Spawn every prop once, then hide all but the active stage."""
        if self._props_ready:
            return
        self._spawn_key_hidden()
        self._spawn_push_hidden()
        self._spawn_cube()
        self._props_ready = True

    def _iter_entities(self, obj):
        if obj is None:
            return
        if isinstance(obj, (list, tuple)):
            for item in obj:
                yield from self._iter_entities(item)
            return
        yield obj.actor if hasattr(obj, "actor") else obj

    def _pose_copy(self, ent) -> sapien.Pose:
        pose = ent.get_pose()
        return sapien.Pose(list(pose.p), list(pose.q))

    def _hide_pose(self, ent, slot: int = 0) -> sapien.Pose:
        z = 0.05 + 0.04 * slot
        return sapien.Pose(
            [self.HIDE_XY[0] + 0.08 * slot, self.HIDE_XY[1], z], [1, 0, 0, 0]
        )

    def _zero_velocity(self, ent) -> None:
        try:
            for comp in ent.get_components():
                if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent):
                    comp.set_linear_velocity([0.0, 0.0, 0.0])
                    comp.set_angular_velocity([0.0, 0.0, 0.0])
        except Exception:
            pass

    def _hide_entity(self, ent, slot: int = 0) -> None:
        try:
            ent.set_pose(self._hide_pose(ent, slot))
        except Exception:
            return
        self._zero_velocity(ent)

    def _hide_obj(self, obj, slot0: int = 0) -> None:
        for i, ent in enumerate(self._iter_entities(obj)):
            self._hide_entity(ent, slot0 + i)

    def _apply_pose(self, obj, pose: sapien.Pose | None) -> None:
        if obj is None or pose is None:
            return
        ent = next(self._iter_entities(obj), None)
        if ent is None:
            return
        try:
            ent.set_pose(pose)
        except Exception:
            return
        self._zero_velocity(ent)

    def _hide_all_props(self) -> None:
        self._hide_obj(self._cube, 0)
        self._hide_obj(self._key_actor, 1)
        self._hide_obj(self._key_bezel, 2)
        self._hide_obj(self._push_box, 8)
        self._hide_obj(self._push_goal, 9)
        self._hide_obj(self._ball, 10)
        self._hide_obj(self._mallet, 11)
        self._hide_obj(getattr(self, "_mallet_rests", None), 12)
        self._hide_obj(getattr(self, "_force_key_actor", None), 16)
        self._hide_obj(getattr(self, "_force_key_bezel", None), 17)
        hide_stove = getattr(self, "_hide_stove", None)
        if callable(hide_stove):
            hide_stove()

    def _gripper_near_key(self) -> bool:
        if self._key_table_pose is None:
            return False
        home = np.asarray(self._key_table_pose.p[:2], dtype=float)
        robot = getattr(self, "robot", None)
        if robot is None:
            return False
        for getter in (
            getattr(robot, "get_left_ee_pose", None),
            getattr(robot, "get_right_ee_pose", None),
        ):
            if getter is None:
                continue
            try:
                tip = np.asarray(getter(), dtype=float)[:3].copy()
            except Exception:
                continue
            tip[2] -= float(self.EE_TO_TCP)
            if float(np.linalg.norm(tip[:2] - home)) < float(self.KEY_XY_TOL) + 0.03:
                return True
        return False

    def _maybe_park_key(self) -> None:
        if not getattr(self, "_defer_hide_key", False):
            return
        if self._gripper_near_key():
            return
        self._hide_obj(self._key_actor, 1)
        self._hide_obj(self._key_bezel, 2)
        self._defer_hide_key = False

    def _show_cube(self) -> None:
        self._apply_pose(self._cube, self._cube_table_pose)
        if self._cube is not None:
            self._cube_rest_z = float(self._cube.get_pose().p[2])

    def _show_key(self, button_id: str) -> None:
        self._apply_pose(self._key_actor, self._key_table_pose)
        if self._key_bezel is not None and self._bezel_table_poses:
            for ent, pose in zip(
                self._iter_entities(self._key_bezel), self._bezel_table_poses
            ):
                try:
                    ent.set_pose(pose)
                except Exception:
                    pass
        self._init_key_bank(button_id)
        self._key_color_down = None
        self._set_key_color(False)

    def _show_push_box(self) -> None:
        self._apply_pose(self._push_box, self._push_table_pose)
        self._apply_pose(self._push_goal, self._push_goal_table_pose)

    def _init_key_bank(self, button_id: str) -> None:
        key = self._key_actor
        if key is None:
            self._reactive_buttons = None
            return
        live = self._pose_copy(next(self._iter_entities(key)))
        hz = float(self.KEY_HALF[2])
        self._reactive_buttons = ReactivePushButtons(
            self,
            actors=[key],
            home_poses=[live],
            max_depth=hz,
            ids=[button_id],
            press_arms=(("left", "right"),),
            xy_tol=float(self.KEY_XY_TOL),
        )
        self._reactive_buttons.set_tops_z([float(live.p[2]) + hz])

    def _spawn_cube(self) -> None:
        hz = float(self.CUBE_HALF)
        cube = create_box(
            self,
            pose=sapien.Pose(
                [self.PROP_X, self.PROP_Y, self.TABLE_Z + hz], [1, 0, 0, 0]
            ),
            half_size=[hz, hz, hz],
            color=list(self.CUBE_COLOR),
            name="tutorial_cube",
            is_static=False,
        )
        cube.set_name("tutorial_cube")
        cube.set_mass(float(self.CUBE_MASS))
        self._cube = cube
        self._cube_table_pose = self._pose_copy(cube.actor)
        self._cube_rest_z = float(self._cube_table_pose.p[2])

    def _spawn_key_hidden(self) -> None:
        hz = float(self.KEY_HALF[2])
        x, y = float(self.PROP_X), float(self.PROP_Y)
        z0 = float(self.TABLE_Z)
        bezel = add_key_base_border(
            self,
            x,
            y,
            z0,
            self.KEY_HALF,
            name_prefix="tutorial_key_base",
        )
        home = sapien.Pose([x, y, z0 + hz], [1, 0, 0, 0])
        key = create_box(
            self,
            pose=home,
            half_size=list(self.KEY_HALF),
            color=list(self.KEY_COLOR_UP),
            name="tutorial_key",
            is_static=True,
        )
        self._key_actor = key
        self._key_bezel = bezel
        self._key_table_pose = self._pose_copy(key.actor)
        self._bezel_table_poses = [
            self._pose_copy(ent) for ent in self._iter_entities(bezel)
        ]
        self._key_shapes = self._render_shapes(key)
        self._hide_obj(key, 1)
        self._hide_obj(bezel, 2)

    def _spawn_push_hidden(self) -> None:
        hx, hy, hz = (float(v) for v in self.PUSH_HALF)
        y0 = float(self.PUSH_Y)
        goal_y = y0 + float(self.PUSH_GOAL_DY)
        box = create_box(
            self,
            pose=sapien.Pose(
                [self.PROP_X, y0, self.TABLE_Z + hz], [1, 0, 0, 0]
            ),
            half_size=[hx, hy, hz],
            color=list(self.PUSH_COLOR),
            name="tutorial_push_box",
            is_static=False,
        )
        box.set_name("tutorial_push_box")
        box.set_mass(float(self.PUSH_MASS))
        self._push_box = box
        self._push_table_pose = self._pose_copy(box.actor)
        self._push_goal_y = goal_y
        goal = self._spawn_visual_box(
            sapien.Pose(
                [self.PROP_X, goal_y, self.TABLE_Z + 0.0015], [1, 0, 0, 0]
            ),
            half_size=[hx + 0.04, 0.008, 0.0015],
            color=list(self.TARGET_COLOR),
            name="tutorial_push_goal",
        )
        self._push_goal = goal
        self._push_goal_table_pose = self._pose_copy(goal)
        self._hide_obj(box, 8)
        self._hide_obj(goal, 9)

    def _spawn_visual_box(self, pose, half_size, color, name: str):
        scene, pose = preprocess(self, pose)
        entity = sapien.Entity()
        entity.set_name(name)
        entity.set_pose(pose)
        material = sapien.render.RenderMaterial(base_color=[*list(color)[:3], 1])
        render = sapien.render.RenderBodyComponent()
        render.attach(sapien.render.RenderShapeBox(list(half_size), material))
        entity.add_component(render)
        scene.add_entity(entity)
        entity.set_pose(pose)
        return entity

    # ------------------------------------------------------- success ticks
    def _tick_grasp(self) -> None:
        if self._stage_settle > 0:
            self._stage_settle -= 1
            return
        if self._cube_picked_up():
            self._grasp_hold_steps += 1
            if self._grasp_hold_steps >= int(self.GRASP_CONFIRM_STEPS):
                self._grasp_complete = True
        else:
            self._grasp_hold_steps = 0

    def _cube_picked_up(self) -> bool:
        cube = self._cube
        rest = self._cube_rest_z
        if cube is None or rest is None:
            return False
        if float(cube.get_pose().p[2]) < float(rest) + float(self.CUBE_LIFT):
            return False
        closing = False
        try:
            robot = self.robot
            closing = (
                float(robot.left_gripper_val) < 0.55
                or float(robot.right_gripper_val) < 0.55
            )
        except Exception:
            try:
                closing = bool(
                    self.is_left_gripper_close() or self.is_right_gripper_close()
                )
            except Exception:
                return False
        if not closing:
            return False
        try:
            contacts = self.get_gripper_actor_contact_position(cube.get_name())
        except Exception:
            return False
        return len(contacts) > 0

    def _tick_push(self) -> None:
        if self._stage_settle > 0:
            self._stage_settle -= 1
            return
        box = self._push_box
        goal = self._push_goal_y
        if box is None or goal is None:
            return
        if float(box.get_pose().p[1]) >= float(goal) - 0.01:
            self._push_hold_steps += 1
            if self._push_hold_steps >= int(self.PUSH_CONFIRM_STEPS):
                self._push_complete = True
        else:
            self._push_hold_steps = 0

    def _tick_buttons(self) -> None:
        bank = self._reactive_buttons
        if bank is None:
            return
        stage = self._tutorial_stage
        if self._stage_settle > 0:
            self._stage_settle -= 1
            if stage == "switch":
                bank.set_forced("switch", False)
            bank.update()
            if stage == "switch":
                still = False
                try:
                    still = bool(bank.is_engaged("switch"))
                except Exception:
                    still = False
                self._switch_touch_latched = still or self._key_tip_pressing("switch")
            return
        if stage == "hold":
            bank.update()
            engaged = bool(bank.is_engaged("hold"))
            self._set_key_color(engaged or float(bank.visual_depth[0]) > 1e-4)
            if engaged:
                self._hold_was_pressed = True
                self._hold_press_steps += 1
                self._hold_release_steps = 0
            elif self._hold_was_pressed and not self._hold_complete:
                self._hold_release_steps += 1
                held_long_enough = self._hold_press_steps >= int(self.HOLD_MIN_STEPS)
                if (
                    held_long_enough
                    and self._hold_release_steps >= int(self.HOLD_RELEASE_STEPS)
                ):
                    self._hold_complete = True
            return

        # Latch on/off switch (cook_meat default): press → ON (stays down,
        # red); press again → OFF (springs up, green).
        bank.set_forced("switch", self._switch_on)
        triggered = set(bank.update())
        touching = self._key_tip_pressing("switch")
        if not self._switch_on:
            if "switch" in triggered:
                self._switch_on = True
                self._switch_touch_latched = True
            else:
                self._switch_touch_latched = touching
        else:
            if touching and not self._switch_touch_latched:
                self._switch_on = False
                self._switch_complete = True
                bank.set_forced("switch", False)
            self._switch_touch_latched = touching
        down = bool(self._switch_on)
        if bank.visual_depth:
            down = down or float(bank.visual_depth[0]) > 1e-4
        self._set_key_color(down)

    def _key_tip_pressing(self, button_id: str) -> bool:
        bank = self._reactive_buttons
        if bank is None:
            return False
        try:
            idx = bank.resolve_index(button_id)
        except Exception:
            return False
        tip = None
        for side in bank._sides_for_button(idx):
            candidate = bank._tip_xyz(side)
            if candidate is None:
                continue
            home_xy = np.asarray(bank.home_poses[idx].p[:2], dtype=float)
            if float(np.linalg.norm(candidate[:2] - home_xy)) > float(bank.xy_tol):
                continue
            tip = candidate
            break
        if tip is None:
            return False
        top_z = float(bank.tops_z[idx])
        force = float(bank.force_stiffness) * max(
            0.0, top_z + float(bank.force_engage_slack) - float(tip[2])
        )
        engage = float(bank.force_full) * (
            float(bank.trigger_depth) / max(float(bank.max_depth), 1e-6)
        )
        return force >= engage

    def _render_shapes(self, actor) -> list:
        entity = actor.actor if hasattr(actor, "actor") else actor
        shapes = []
        try:
            for comp in entity.get_components():
                if isinstance(comp, sapien.render.RenderBodyComponent):
                    shapes.extend(list(comp.render_shapes))
        except Exception:
            return []
        return shapes

    def _set_key_color(self, down: bool) -> None:
        down = bool(down)
        if self._key_color_down is not None and bool(self._key_color_down) == down:
            return
        self._key_color_down = down
        rgb = self.KEY_COLOR_DOWN if down else self.KEY_COLOR_UP
        color = list(rgb) + [1.0]
        for shape in self._key_shapes:
            try:
                shape.material.set_base_color(color)
            except Exception:
                pass

    def _get_rigid(self, obj):
        ent = obj.actor if hasattr(obj, "actor") else obj
        try:
            for comp in ent.get_components():
                if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent):
                    return comp
        except Exception:
            return None
        return None

    def _object_name(self, obj) -> str:
        if obj is None:
            return ""
        try:
            if hasattr(obj, "get_name"):
                return str(obj.get_name() or "")
        except Exception:
            pass
        ent = obj.actor if hasattr(obj, "actor") else obj
        try:
            return str(ent.get_name() or "")
        except Exception:
            return ""

    def _gripper_holding(self, obj, rest_z, lift: float) -> bool:
        if obj is None or rest_z is None:
            return False
        try:
            z = float(obj.get_pose().p[2])
        except Exception:
            return False
        if z < float(rest_z) + float(lift):
            return False
        closing = False
        try:
            robot = self.robot
            closing = (
                float(robot.left_gripper_val) < 0.55
                or float(robot.right_gripper_val) < 0.55
            )
        except Exception:
            try:
                closing = bool(
                    self.is_left_gripper_close() or self.is_right_gripper_close()
                )
            except Exception:
                return False
        if not closing:
            return False
        name = self._object_name(obj)
        if not name:
            return False
        try:
            contacts = self.get_gripper_actor_contact_position(name)
        except Exception:
            return False
        return len(contacts) > 0

    # ------------------------------------------------------- part 4: ball
    def _ensure_ball(self) -> None:
        if self._ball is not None:
            return
        r = float(self.BALL_RADIUS)
        ball = create_sphere(
            self,
            pose=sapien.Pose(
                [float(self.BALL_X), float(self.BALL_Y0), float(self.TABLE_Z) + r]
            ),
            radius=r,
            color=list(self.BALL_COLOR),
            is_static=False,
            name="tutorial_ball",
        )
        try:
            ball.set_name("tutorial_ball")
        except Exception:
            pass
        self._ball = ball
        self._ball_start = self._pose_copy(ball)
        self._ball_rest_z = float(self._ball_start.p[2])
        rigid = self._get_rigid(ball)
        self._ball_rigid = rigid
        if rigid is not None:
            try:
                rigid.mass = float(self.BALL_MASS)
            except Exception:
                pass

    def _launch_ball(self) -> None:
        ball = self._ball
        start = self._ball_start
        if ball is None or start is None:
            return
        try:
            ball.set_pose(start)
        except Exception:
            return
        self._zero_velocity(ball)
        rigid = self._ball_rigid
        if rigid is None:
            return
        vy = float(self.BALL_VY)
        r = max(float(self.BALL_RADIUS), 1e-4)
        try:
            rigid.set_linear_velocity([0.0, vy, 0.0])
            rigid.set_angular_velocity([vy / r, 0.0, 0.0])
        except Exception:
            pass

    def _respawn_ball(self) -> None:
        self._ball_ever_lifted = False
        self._grasp_hold_steps = 0
        self._stage_settle = int(self.BALL_RESPAWN_SETTLE)
        self._launch_ball()

    def _ball_off_table(self) -> bool:
        ball = self._ball
        if ball is None:
            return False
        p = np.asarray(ball.get_pose().p, dtype=float)
        table_z = float(self._ball_rest_z or (self.TABLE_Z + self.BALL_RADIUS))
        if float(p[2]) < table_z - 0.06:
            return True
        if abs(float(p[0])) > float(self.BALL_TABLE_X_LIM):
            return True
        if float(p[1]) < float(self.BALL_TABLE_Y_MIN) or float(p[1]) > float(
            self.BALL_TABLE_Y_MAX
        ):
            return True
        return False

    def _ball_stopped(self) -> bool:
        rigid = self._ball_rigid
        ball = self._ball
        if rigid is None or ball is None or self._ball_rest_z is None:
            return False
        try:
            speed = float(np.linalg.norm(rigid.get_linear_velocity()))
            z = float(ball.get_pose().p[2])
        except Exception:
            return False
        on_table = z < float(self._ball_rest_z) + 0.012
        return on_table and speed < 0.025

    def _tick_ball(self) -> None:
        if self._stage_settle > 0:
            self._stage_settle -= 1
            return
        if self._gripper_holding(self._ball, self._ball_rest_z, self.BALL_LIFT):
            self._ball_ever_lifted = True
            self._grasp_hold_steps += 1
            if self._grasp_hold_steps >= int(self.GRASP_CONFIRM_STEPS):
                self._ball_complete = True
            return
        self._grasp_hold_steps = 0
        dropped = bool(self._ball_ever_lifted)
        if dropped or self._ball_off_table() or self._ball_missed():
            self._respawn_ball()

    def _ball_missed(self) -> bool:
        """True when the ball rolled past the workspace without being picked up."""
        ball = self._ball
        if ball is None or self._ball_ever_lifted:
            return False
        try:
            y = float(ball.get_pose().p[1])
        except Exception:
            return False
        if y < float(self.PROP_Y) - 0.05:
            return True
        return bool(self._ball_stopped() and y < float(self.BALL_Y0) - 0.08)

    # ------------------------------------------------------- part 4: stove
    def _ensure_stove(self) -> None:
        if self._stove_loaded:
            return
        self.range_scale_mult = 1.0
        self.range_position_override = list(self.STOVE_XY)
        self.stove_side = "left"
        self.knob_contact_radius = float(self.KNOB_CONTACT_RADIUS_DEFAULT)
        table_h = float(self.TABLE_Z) + float(getattr(self, "table_z_bias", 0.0))
        self._load_cooking_range(table_h, [0.0, 0.0])
        name = str(self.STOVE_BURNER)
        if name not in getattr(self, "burner_positions", {}):
            name = "left_front"
        self.burner_name = name
        bx, by = self.burner_positions[name]
        self.burner_xy = (float(bx), float(by))
        self._burner_home_pose = sapien.Pose(
            p=[float(bx), float(by), float(self.range_top_z) + 0.002]
        )
        self._build_stove_fire_ring(
            float(bx),
            float(by),
            float(self.range_top_z) + 0.004,
            0.058,
            half_size=[0.010, 0.005, 0.004],
        )
        self._set_stove_fire(False, intensity=0.0)
        if getattr(self, "range_body", None) is not None:
            self._stove_range_home = self._pose_copy(self.range_body.actor)
        art = getattr(self, "stove_knob_articulation", None)
        if art is not None:
            try:
                self._stove_knob_home = sapien.Pose(
                    list(art.get_pose().p), list(art.get_pose().q)
                )
            except Exception:
                try:
                    pose = art.get_root_pose()
                    self._stove_knob_home = sapien.Pose(list(pose.p), list(pose.q))
                except Exception:
                    self._stove_knob_home = None
        self._stove_cover_homes = dict(getattr(self, "_burner_cover_home_poses", {}) or {})
        self._set_knob_joint_angle(float(self.KNOB_OFF_ANGLE), hard=True)
        self._stove_loaded = True

    def _hide_stove(self) -> None:
        if not self._stove_loaded:
            return
        try:
            self._set_stove_fire(False, intensity=0.0)
        except Exception:
            pass
        hidden = sapien.Pose([0.0, 0.0, -1.6], [1, 0, 0, 0])
        try:
            if getattr(self, "range_body", None) is not None:
                self.range_body.actor.set_pose(hidden)
        except Exception:
            pass
        art = getattr(self, "stove_knob_articulation", None)
        if art is not None:
            try:
                art.set_pose(hidden)
            except Exception:
                try:
                    art.set_root_pose(hidden)
                except Exception:
                    pass
        for cover in getattr(self, "_burner_covers", []) or []:
            try:
                cover.set_pose(hidden)
            except Exception:
                pass
        burner = getattr(self, "active_burner", None)
        if burner is not None:
            try:
                burner.set_pose(hidden)
            except Exception:
                pass

    def _show_stove(self) -> None:
        if not self._stove_loaded:
            return
        if self._stove_range_home is not None and getattr(self, "range_body", None):
            try:
                self.range_body.actor.set_pose(self._stove_range_home)
            except Exception:
                pass
        art = getattr(self, "stove_knob_articulation", None)
        if art is not None and self._stove_knob_home is not None:
            try:
                art.set_pose(self._stove_knob_home)
            except Exception:
                try:
                    art.set_root_pose(self._stove_knob_home)
                except Exception:
                    pass
        self._set_knob_joint_angle(float(self.KNOB_OFF_ANGLE), hard=True)
        self.knob_angle = float(self.KNOB_OFF_ANGLE)
        self.fire_intensity = 0.0
        self.stove_on = False
        self._set_stove_fire(False, intensity=0.0)

    def _set_knob_angle(self, angle: float, *, drive_fire: bool = True) -> None:
        """cook_food mapping: 0 = off, −π/2 = full fire (tick left)."""
        angle = float(np.clip(angle, -float(self.KNOB_MAX_ANGLE), 0.0))
        self.knob_angle = angle
        if not drive_fire:
            return
        frac = float(-angle / max(float(self.KNOB_MAX_ANGLE), 1e-6))
        self.fire_intensity = float(np.clip(frac, 0.0, 1.0))
        self.stove_on = self.fire_intensity > 0.02
        if self.fire_intensity >= float(self.STOVE_ON_FRAC):
            self._stove_turned_on = True
        if self._stove_turned_on and self.fire_intensity <= float(self.STOVE_OFF_FRAC):
            self._stove_turned_off = True
        self._set_knob_joint_angle(
            angle, hard=not bool(getattr(self, "_knob_grasp_active", False))
        )
        self._set_stove_fire(self.stove_on, intensity=self.fire_intensity)

    def _tick_stove(self) -> None:
        if getattr(self, "stove_knob_articulation", None) is None:
            return
        self._update_knob_from_physics()
        self._update_stove_knob_control()
        if self._stage_settle > 0:
            self._stage_settle -= 1
            return
        if self._stove_turned_on and self._stove_turned_off:
            self._stove_complete = True

    # ------------------------------------------------------- part 4: mallet
    def _ensure_mallet(self) -> None:
        if self._mallet is not None:
            return
        x, y = float(self.PROP_X), float(self.PROP_Y)
        rest_h = float(self.MALLET_REST_HEIGHT)
        table_z = float(self.TABLE_Z) + float(getattr(self, "table_z_bias", 0.0))
        rails = []
        rest_poses = []
        for post_idx, y_offset in enumerate(self.MALLET_REST_POST_Y):
            rail = create_box(
                self,
                sapien.Pose(
                    [x, y + float(y_offset), self.TABLE_Z + 0.5 * rest_h]
                ),
                half_size=[0.016, 0.010, 0.5 * rest_h],
                color=[0.32, 0.30, 0.28],
                is_static=True,
                name=f"tutorial_mallet_rest_{post_idx}",
            )
            rails.append(rail)
            rest_poses.append(self._pose_copy(next(self._iter_entities(rail))))
        z = table_z + rest_h + float(self.MALLET_HANDLE_RADIUS)
        mallet = self._build_mallet(
            sapien.Pose([x, y, z], list(self.MALLET_LEVEL_Q)),
            name="tutorial_mallet",
        )
        mallet.set_name("tutorial_mallet")
        mallet.set_mass(float(self.MALLET_MASS))
        self._mallet = mallet
        self._mallet_rests = rails
        self._mallet_table_pose = self._pose_copy(mallet.actor)
        self._mallet_rest_poses = rest_poses
        self._mallet_rest_z = float(self._mallet_table_pose.p[2])

    def _build_mallet(self, pose, name: str):
        """T mallet: handle along Y, cylindrical head across X (whack_moles)."""
        builder = self.scene.create_actor_builder()
        mat = self.scene.default_physical_material
        wood = sapien.render.RenderMaterial(
            base_color=[*self.MALLET_WOOD_COLOR, 1.0]
        )
        builder.add_cylinder_collision(
            pose=sapien.Pose([0.0, self.MALLET_HEAD_Y, 0.0]),
            radius=self.MALLET_HEAD_RADIUS,
            half_length=self.MALLET_HEAD_HALF_X,
            material=mat,
        )
        builder.add_cylinder_visual(
            pose=sapien.Pose([0.0, self.MALLET_HEAD_Y, 0.0]),
            radius=self.MALLET_HEAD_RADIUS,
            half_length=self.MALLET_HEAD_HALF_X,
            material=wood,
        )
        handle_pose = sapien.Pose(
            [0.0, self.MALLET_HANDLE_CENTER_Y, 0.0],
            [0.70710678, 0.0, 0.0, 0.70710678],
        )
        builder.add_cylinder_collision(
            pose=handle_pose,
            radius=self.MALLET_HANDLE_RADIUS,
            half_length=self.MALLET_HANDLE_HALF_Y,
            material=mat,
        )
        builder.add_cylinder_visual(
            pose=handle_pose,
            radius=self.MALLET_HANDLE_RADIUS,
            half_length=self.MALLET_HANDLE_HALF_Y,
            material=wood,
        )
        builder.set_initial_pose(pose)
        entity = builder.build(name=name)
        data = {
            "center": [0.0, 0.0, 0.0],
            "extents": [
                self.MALLET_HEAD_HALF_X,
                self.MALLET_HANDLE_HALF_Y + abs(self.MALLET_HEAD_Y),
                self.MALLET_HEAD_RADIUS,
            ],
            "scale": [
                self.MALLET_HANDLE_RADIUS,
                self.MALLET_HANDLE_HALF_Y,
                self.MALLET_HANDLE_RADIUS,
            ],
            "target_pose": [np.eye(4).tolist()],
            "contact_points_pose": [[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]],
            "transform_matrix": np.eye(4).tolist(),
            "functional_matrix": [],
        }
        return Actor(entity, data, mass=float(self.MALLET_MASS))

    def _show_mallet(self) -> None:
        self._apply_pose(self._mallet, self._mallet_table_pose)
        if self._mallet_rests and self._mallet_rest_poses:
            for ent, pose in zip(
                self._iter_entities(self._mallet_rests), self._mallet_rest_poses
            ):
                try:
                    ent.set_pose(pose)
                except Exception:
                    pass
        if self._mallet is not None:
            self._mallet_rest_z = float(self._mallet.get_pose().p[2])

    def _tick_mallet(self) -> None:
        if self._stage_settle > 0:
            self._stage_settle -= 1
            return
        if self._gripper_holding(self._mallet, self._mallet_rest_z, self.MALLET_LIFT):
            self._grasp_hold_steps += 1
            if self._grasp_hold_steps >= int(self.GRASP_CONFIRM_STEPS):
                self._mallet_complete = True
        else:
            self._grasp_hold_steps = 0

    # ------------------------------------------------------- part 4: force key
    def _ensure_force_key(self) -> None:
        if self._force_key_actor is not None:
            return
        hz = float(self.KEY_HALF[2])
        x, y = float(self.PROP_X), float(self.PROP_Y)
        z0 = float(self.TABLE_Z)
        bezel = add_key_base_border(
            self,
            x,
            y,
            z0,
            self.KEY_HALF,
            name_prefix="tutorial_force_key_base",
        )
        home = sapien.Pose([x, y, z0 + hz], [1, 0, 0, 0])
        key = create_box(
            self,
            pose=home,
            half_size=list(self.KEY_HALF),
            color=list(self.FORCE_KEY_COLOR),
            name="tutorial_force_key",
            is_static=True,
        )
        self._force_key_actor = key
        self._force_key_bezel = bezel
        self._force_key_table_pose = self._pose_copy(key.actor)
        self._force_key_bezel_poses = [
            self._pose_copy(ent) for ent in self._iter_entities(bezel)
        ]
        self._key_shapes = self._render_shapes(key)
        self._hide_obj(key, 16)
        self._hide_obj(bezel, 17)

    def _show_force_key(self) -> None:
        self._apply_pose(self._force_key_actor, self._force_key_table_pose)
        if self._force_key_bezel is not None and self._force_key_bezel_poses:
            for ent, pose in zip(
                self._iter_entities(self._force_key_bezel),
                self._force_key_bezel_poses,
            ):
                try:
                    ent.set_pose(pose)
                except Exception:
                    pass
        key = self._force_key_actor
        if key is None:
            self._reactive_buttons = None
            return
        live = self._pose_copy(next(self._iter_entities(key)))
        hz = float(self.KEY_HALF[2])
        self._key_shapes = self._render_shapes(key)
        self._key_color_down = None
        self._set_force_key_color(0.0)
        self._reactive_buttons = _ForceKeyBank(
            self,
            actors=[key],
            home_poses=[live],
            max_depth=hz,
            ids=["force"],
            press_arms=(("left", "right"),),
            xy_tol=float(self.KEY_XY_TOL),
            force_full=float(self.FORCE_THRESHOLDS[-1]),
        )
        self._reactive_buttons.set_tops_z([float(live.p[2]) + hz])

    def _force_level(self, force_n: float) -> int:
        level = 0
        for i, thr in enumerate(self.FORCE_THRESHOLDS):
            if float(force_n) >= float(thr):
                level = i + 1
        return int(level)

    def _set_force_key_color(self, force_n: float) -> None:
        level = self._force_level(force_n)
        rgb = self.FORCE_LEVEL_COLORS[min(level, len(self.FORCE_LEVEL_COLORS) - 1)]
        if self._key_color_down is not None and self._key_color_down == level:
            return
        self._key_color_down = level
        color = list(rgb) + [1.0]
        for shape in self._key_shapes:
            try:
                shape.material.set_base_color(color)
            except Exception:
                pass

    def _tick_force_key(self) -> None:
        bank = self._reactive_buttons
        if bank is None or not isinstance(bank, _ForceKeyBank):
            return
        if self._stage_settle > 0:
            self._stage_settle -= 1
            bank.update()
            return
        bank.update()
        force = float(bank.live_force("force"))
        self._key_force_n = force
        self._set_force_key_color(force)
        engaged = bool(bank.is_engaged("force"))
        if engaged:
            self._press_active = True
            self._press_peak_force = max(float(self._press_peak_force), force)
            return
        if not self._press_active:
            return
        peak = float(self._press_peak_force)
        level = self._force_level(peak)
        target = int(self._force_target_level)
        self._press_active = False
        self._press_peak_force = 0.0
        lo = float(self.FORCE_THRESHOLDS[target - 1])
        hi = (
            float(self.FORCE_THRESHOLDS[target])
            if target < len(self.FORCE_THRESHOLDS)
            else 1e9
        )
        if level == target:
            self._force_complete = True
            self._force_feedback = f"correct ({peak:.1f} N)"
        elif level < target:
            self._force_feedback = f"too light ({peak:.1f} N, need {lo:.0f}–{hi:.0f} N)"
        else:
            self._force_feedback = f"too hard ({peak:.1f} N, need {lo:.0f}–{hi:.0f} N)"

    def interactive_ee_z_floor(self, side, pose):
        if getattr(self, "_tutorial_stage", None) != "force_key":
            return None
        bank = self._reactive_buttons
        if not isinstance(bank, _ForceKeyBank):
            return None
        try:
            return bank.min_ee_z_over_key(np.asarray(pose[:2], dtype=float))
        except Exception:
            return None

    def interactive_teleop_z_speed_scale(self, side, pose, z_delta: float):
        if getattr(self, "_tutorial_stage", None) != "force_key":
            return None
        if float(z_delta) >= 0.0:
            return None
        bank = self._reactive_buttons
        if not isinstance(bank, _ForceKeyBank) or not bank.home_poses:
            return None
        xy = np.asarray(pose[:2], dtype=float)
        home = np.asarray(bank.home_poses[0].p[:2], dtype=float)
        if float(np.linalg.norm(xy - home)) > float(bank.xy_tol) * 1.35:
            return None
        return float(self.DISPENSE_Z_DOWN_SCALE)


# Reuse KitchenS cooktop / knob helpers without swapping the empty table scene.
_KS_ATTRS = (
    "COOKTOP_ASSET",
    "RANGE_BURNER_OFFSETS",
    "KNOB_LOCAL_XY",
    "KNOB_RADIUS",
    "KNOB_HEIGHT",
    "KNOB_OFF_ANGLE",
    "KNOB_ON_ANGLE",
    "KNOB_CONTACT_RADIUS_DEFAULT",
    "KNOB_GRASP_GRIPPER_MAX",
    "EE_TO_TCP_DEFAULT",
    "FIRE_BLUE",
    "FIRE_DISC_BLUE",
    "TOP_KNOB_APPROACH_PATH",
    "KNOB_APPROACH_PATH",
    "KNOB_GRASP_STANDOFF",
)
_KS_METHODS = (
    "_load_cooking_range",
    "_dim_cooktop_burner_materials",
    "_fire_hidden_pose",
    "_clear_stove_fire_ring",
    "_build_stove_fire_ring",
    "_bump_render_bodies",
    "_flush_stove_fire_viewer",
    "_set_stove_fire",
    "_get_knob_joint_angle",
    "_set_knob_articulation_qpos",
    "_hold_knob_joint",
    "_set_knob_joint_angle",
    "_ee_knob_twist",
    "_boost_gripper_knob_friction",
    "_knob_candidate_arms",
    "_begin_knob_turn",
    "_end_knob_turn",
    "_coupled_knob_angle",
    "_update_knob_from_physics",
    "_knob_pinch_near",
    "_knob_gripper_closed",
    "_knob_has_gripper_contact",
    "_knob_is_grasped",
    "_commit_stove_from_knob_angle",
    "_update_stove_knob_control",
)
for _name in _KS_ATTRS:
    setattr(tutorial_empty, _name, getattr(KitchenS_base_task, _name))
for _name in _KS_METHODS:
    setattr(tutorial_empty, _name, getattr(KitchenS_base_task, _name))

