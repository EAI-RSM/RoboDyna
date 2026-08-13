"""Empty table + dual UR5 playground for the interactive tutorial.

Parts 1–2 and 4 leave the table bare. Part 3 spawns one prop at a time
(cube, hold button, on/off switch, push box) via ``tutorial_set_stage``.
"""
from __future__ import annotations

import sapien
import sapien.render
import numpy as np

from ._base_task import Base_Task
from .utils.create_actor import create_box
from .utils.reactive_button import ReactivePushButtons, add_key_base_border


class tutorial_empty(Base_Task):
    """Bare tabletop with both arms at home. Used by tutorial parts 1–4."""

    # Left-side workspace so the selected left arm can reach every prop.
    PROP_X = -0.16
    PROP_Y = -0.08
    TABLE_Z = 0.74  # create_box adds table_z_bias

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

    def setup_demo(self, **kwags):
        self._tutorial_complete = False
        self._tutorial_stage = None
        self._tutorial_actors = []
        self._reactive_buttons = None
        self._cube = None
        self._cube_rest_z = None
        self._push_box = None
        self._push_goal_y = None
        self._key_actor = None
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
        super()._init_task_env_(**kwags)

    def load_actors(self):
        return

    def play_once(self):
        pass

    def check_success(self):
        return bool(getattr(self, "_tutorial_complete", False))

    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
        stage = getattr(self, "_tutorial_stage", None)
        if not stage:
            return
        if stage == "grasp":
            self._tick_grasp()
        elif stage in ("hold", "switch"):
            self._tick_buttons()
        elif stage == "push":
            self._tick_push()

    # ---------------------------------------------------------- stage props
    def tutorial_set_stage(self, stage: str | None) -> None:
        """Despawn the current prop and spawn the next part-3 object."""
        self._clear_stage_actors()
        self._tutorial_stage = stage
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
        self._stage_settle = int(self.SETTLE_STEPS) if stage else 0
        if stage == "grasp":
            self._spawn_cube()
        elif stage == "hold":
            self._spawn_key("hold")
        elif stage == "switch":
            self._spawn_key("switch")
        elif stage == "push":
            self._spawn_push_box()

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
        return False

    def _clear_stage_actors(self) -> None:
        self._reactive_buttons = None
        for obj in list(self._tutorial_actors):
            self._remove_actor(obj)
        self._tutorial_actors = []
        self._cube = None
        self._push_box = None
        self._key_actor = None
        self._key_shapes = []
        self._key_color_down = None

    def _remove_actor(self, obj) -> None:
        if obj is None:
            return
        if isinstance(obj, (list, tuple)):
            for item in obj:
                self._remove_actor(item)
            return
        ent = obj.actor if hasattr(obj, "actor") else obj
        try:
            self.scene.remove_entity(ent)
        except Exception:
            pass

    def _track(self, obj):
        self._tutorial_actors.append(obj)
        return obj

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
        self._cube = self._track(cube)
        self._cube_rest_z = float(cube.get_pose().p[2])

    def _spawn_key(self, button_id: str) -> None:
        hx, hy, hz = (float(v) for v in self.KEY_HALF)
        x, y = float(self.PROP_X), float(self.PROP_Y)
        z0 = float(self.TABLE_Z)
        bezel = add_key_base_border(
            self,
            x,
            y,
            z0,
            self.KEY_HALF,
            name_prefix=f"tutorial_{button_id}_base",
        )
        self._track(bezel)
        home = sapien.Pose([x, y, z0 + hz], [1, 0, 0, 0])
        key = create_box(
            self,
            pose=home,
            half_size=list(self.KEY_HALF),
            color=list(self.KEY_COLOR_UP),
            name=f"tutorial_{button_id}_key",
            is_static=True,
        )
        self._key_actor = self._track(key)
        self._key_shapes = self._render_shapes(key)
        self._key_color_down = None
        self._set_key_color(False)
        top_z = float(key.get_pose().p[2]) + hz
        self._reactive_buttons = ReactivePushButtons(
            self,
            actors=[key],
            home_poses=[home],
            max_depth=hz,
            ids=[button_id],
            press_arms=(("left", "right"),),
            xy_tol=float(self.KEY_XY_TOL),
        )
        self._reactive_buttons.set_tops_z([top_z])

    def _spawn_push_box(self) -> None:
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
        self._push_box = self._track(box)
        self._push_goal_y = goal_y
        # Thin green goal strip on the table — push the box onto / past it.
        strip_hz = 0.002
        strip = create_box(
            self,
            pose=sapien.Pose(
                [self.PROP_X, goal_y, self.TABLE_Z + strip_hz], [1, 0, 0, 0]
            ),
            half_size=[hx + 0.04, 0.008, strip_hz],
            color=list(self.TARGET_COLOR),
            name="tutorial_push_goal",
            is_static=True,
        )
        self._track(strip)

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
        closed = False
        try:
            closed = bool(self.is_left_gripper_close() or self.is_right_gripper_close())
        except Exception:
            return False
        if not closed:
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
