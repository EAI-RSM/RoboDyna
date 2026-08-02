"""Shared interactive runner for the household task sandboxes.

The household environments deliberately keep their normal physics and success
checks.  This module only adds the same viewer/arm teleoperation used by
``script_exp``: arrows move the selected end-effector in XY, Q/E move it in Z,
and 1/2/3 select the left/right/both arms.  Space grasps/releases the task's
primary prop and F invokes a task-specific control (turn a knob, press a
dispenser, release a moving object, ...).
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

import numpy as np
import sapien
import sapien.physx
from transforms3d.quaternions import axangle2quat, qmult

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BENCH = ROOT / "script" / "bench_script"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))
if str(ROOT / "script_exp") not in sys.path:
    sys.path.insert(0, str(ROOT / "script_exp"))

from script_exp._interactive_common import (  # noqa: E402
    configure_task,
    make_viewer_view_toggle,
    print_mode_controls,
    report_task_result,
)


PROFILES = {
    "trap_bug": ("envs.trap_bug", "trap_bug", "demo_dynamic", "trap"),
    "boil_milk": ("envs.boil_milk", "boil_milk", "demo_dynamic", None),
    "fill_coffee_jar": ("envs.fill_coffee_jar", "fill_coffee_jar", "demo_dynamic", None),
    "pour_beer": ("envs.pour_beer", "pour_beer", "demo_dynamic", None),
    "cook_food": ("envs.cook_food", "cook_food", "demo_dynamic", "food"),
    "measure_ingredient": ("envs.measure_ingredient", "measure_ingredient", "demo_dynamic", "jar"),
    "make_soup": ("envs.make_soup", "make_soup", "demo_dynamic", "board"),
    "catch_cup": ("envs.catch_cup", "catch_cup", "demo_dynamic", "pillow"),
    "mouse_object_drop": ("envs.mouse_object_drop", "mouse_object_drop", "demo_dynamic", "basket"),
    "stop_ball": ("envs.stop_ball", "stop_ball", "demo_dynamic", None),
    "clean_table": ("envs.clean_table", "clean_table", "demo_dynamic", "sponge"),
}


class _Edge:
    def __init__(self):
        self.down = False

    def poll(self, value):
        value = bool(value)
        edge = value and not self.down
        self.down = value
        return edge


def _rigid(actor):
    obj = getattr(actor, "actor", actor)
    for comp in getattr(obj, "get_components", lambda: ())():
        if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent):
            return comp
    return None


def _set_pose(actor, xyz, quat=None, kinematic=True):
    """Set an Actor pose while preserving its dynamic/kinematic state."""
    if actor is None:
        return
    pose = actor.get_pose()
    q = pose.q if quat is None else quat
    target = sapien.Pose([float(v) for v in xyz], list(q))
    obj = getattr(actor, "actor", actor)
    obj.set_pose(target)
    body = _rigid(actor)
    if body is None:
        return
    try:
        body.set_linear_velocity([0, 0, 0])
        body.set_angular_velocity([0, 0, 0])
        body.set_disable_gravity(bool(kinematic))
        body.set_kinematic(bool(kinematic))
        if kinematic and hasattr(body, "set_kinematic_target"):
            body.set_kinematic_target(target)
    except Exception:
        pass


def _arm_tag(env):
    from envs.utils.action import ArmTag
    selected = tuple(getattr(env, "_interactive_selected_arms", ()) or ())
    if selected:
        return ArmTag(selected[0])
    return ArmTag(str(getattr(env, "arm_side", "right")))


class HouseholdController:
    """Small adapter around each task's existing actors and expert hooks."""

    FILL_PRESS_STEP = 0.006
    FILL_MAX_PRESS_DEPTH = 0.12
    FILL_CONTACT_DZ = 0.006
    FILL_STALL_DZ = 0.012
    FILL_XY_TOL = 0.040
    FILL_RETURN_STEPS = 25

    def __init__(self, env, task, robot=False):
        self.env, self.task, self.robot = env, task, robot
        actor_attr = PROFILES[task][3]
        self.actor = getattr(env, actor_attr, None) if actor_attr else None
        self.holding = False
        self.space = _Edge()
        self.f = _Edge()
        self._fill_space_started_at = None
        self._fill_space_was_down = False
        self._fill_press_state = None
        self.board_over_pot = False
        self.trap_released = False
        self.scenario_started = False
        self._failure_visual_until = 0.0
        self._failure_visual_original = {}
        if task == "fill_coffee_jar" and robot:
            self._close_fill_coffee_grippers_at_start()
        if not robot and self.actor is not None:
            _set_pose(self.actor, self.actor.get_pose().p, kinematic=True)

    def _close_fill_coffee_grippers_at_start(self):
        """Start fill_coffee_jar robot mode with both grippers closed."""
        from envs.utils.action import ArmTag

        for side in ("left", "right"):
            try:
                arm = ArmTag(side)
                self.env.plan_success = True
                moved = self.env.move(self.env.close_gripper(arm))
                if moved is False or not bool(getattr(self.env, "plan_success", True)):
                    print(f"[fill_coffee_jar] could not pre-close {side} gripper")
            except Exception as exc:
                print(f"[fill_coffee_jar] could not pre-close {side} gripper: {exc}")
        self.env.plan_success = True

    def _keyboard_action(self):
        """Operate task state directly without moving either robot arm."""
        e, t = self.env, self.task
        try:
            if t == "boil_milk":
                e._set_stove(not bool(e.stove_on))
            elif t == "fill_coffee_jar":
                self._fill_coffee_press(1)
            elif t == "pour_beer":
                target = 0.0 if e._lever_open_frac() > 0.05 else 0.70
                e._apply_lever_pose(target * float(e.lever_open_rad))
                print(f"[pour_beer] lever {'closed' if target == 0 else 'opened'}")
            elif t == "cook_food":
                angle = (
                    -float(getattr(e, "cook_intensity", 1.0))
                    * float(getattr(e, "KNOB_MAX_ANGLE", np.pi / 2))
                )
                e._set_knob_angle(0.0 if bool(e.stove_on) else angle)
            elif t == "measure_ingredient":
                e._set_tab_open(not bool(e.tab_open))
            elif t == "make_soup":
                if not bool(e.stove_on):
                    e._set_stove(True)
                    print("[make_soup] burner on; hold/move the board, then press F to pour")
                elif self.holding:
                    target = [float(e.pot_xy[0]), float(e.pot_xy[1]), float(e.pot_rim_z + 0.12)]
                    e._set_entity_pose(e.board, sapien.Pose(target, [1, 0, 0, 0]))
                    self.board_over_pot = True
                    print("[make_soup] board is over the pot; hold R/T to tilt")
            elif t == "clean_table":
                if not bool(e.cup_tipped):
                    e._animate_tip()
                else:
                    cleared = e._try_clear_spots_under_sponge()
                    print(f"[clean_table] cleared {cleared} spot(s) under the sponge")
        except Exception as exc:
            print(f"[{t}] action unavailable: {exc}")

    def _fill_coffee_press(self, force_level):
        """Dispense at the elapsed-contact force level after an in-place press."""
        e = self.env
        level = max(1, min(4, int(force_level)))
        try:
            e._start_press()
            if not bool(getattr(e, "_press_active", False)):
                return
            e._interactive_force_level_override = level
            e._press_peak_force = float(e.force_thresholds[level - 1])
            e._press_force_level = level
            e._end_press()
            print(f"[fill_coffee_jar] completed force level {level} press")
        except Exception as exc:
            print(f"[fill_coffee_jar] level {level} press unavailable: {exc}")
        finally:
            if hasattr(e, "_interactive_force_level_override"):
                del e._interactive_force_level_override

    @staticmethod
    def _fill_force_level(held_seconds):
        """Use one force level per 0.5 s, with level 4 at 2 s and above."""
        return max(1, min(4, int(np.ceil(max(0.0, float(held_seconds)) / 0.5))))

    def _fill_tcp_pose(self, side):
        getter = (
            self.env.robot.get_left_tcp_pose
            if side == "left"
            else self.env.robot.get_right_tcp_pose
        )
        return np.asarray(getter(), dtype=np.float64)

    def _fill_arm_qpos(self, side):
        joints = (
            self.env.robot.left_arm_joints
            if side == "left"
            else self.env.robot.right_arm_joints
        )
        entity = (
            self.env.robot.left_entity
            if side == "left"
            else self.env.robot.right_entity
        )
        try:
            qpos = np.asarray(entity.get_qpos(), dtype=np.float64)
            active = list(entity.get_active_joints())
            return np.asarray([qpos[active.index(joint)] for joint in joints], dtype=np.float64)
        except Exception:
            return np.asarray(
                [float(np.asarray(joint.get_drive_target()).reshape(-1)[0]) for joint in joints],
                dtype=np.float64,
            )

    def _fill_over_dispenser(self, side):
        tcp = self._fill_tcp_pose(side)
        distance = float(np.linalg.norm(tcp[:2] - np.asarray(self.env.touch_xy, dtype=float)))
        return distance <= self.FILL_XY_TOL

    def _fill_contact(self, state):
        """Detect real contact, or a collision-limited stop at the button top."""
        side = state["side"]
        if not self._fill_over_dispenser(side):
            return False
        tcp_z = float(self._fill_tcp_pose(side)[2])
        dz = tcp_z - float(self.env.touch_top_z)
        real_contact = float(self.env._lid_contact_force()) > 0.5
        geometric_contact = dz <= self.FILL_CONTACT_DZ
        stalled = (
            state["descended"] >= 0.010
            and dz <= self.FILL_STALL_DZ
            and abs(tcp_z - state["last_tcp_z"]) < 0.0008
        )
        state["last_tcp_z"] = tcp_z
        return bool(real_contact or geometric_contact or stalled)

    def _start_fill_press(self):
        selected = tuple(getattr(self.env, "_interactive_selected_arms", ()) or ())
        if len(selected) != 1:
            print("[fill_coffee_jar] select one arm, move it above the dispenser, then hold Space")
            return
        side = selected[0]
        from envs.utils.action import ArmTag

        arm = ArmTag(side)
        return_qpos = self._fill_arm_qpos(side)
        self.env._pressing_arm_side = side
        self.env._interactive_teleop_locked = True
        high_dis = float(getattr(self.env, "KEY_HOVER_DIS", 0.06)) + 0.08
        hover_dis = float(getattr(self.env, "KEY_HOVER_DIS", 0.06))
        self.env.plan_success = True
        moved = self.env.move(self.env.move_to_pose(arm, self.env._touch_tip_pose(high_dis)))
        if moved is not False and bool(getattr(self.env, "plan_success", True)):
            self.env.move(self.env.move_by_displacement(arm, z=-(high_dis - hover_dis)))
        if moved is False or not bool(getattr(self.env, "plan_success", True)):
            self.env.plan_success = True
            moved = self.env.move(self.env.move_to_pose(arm, self.env._touch_tip_pose(hover_dis)))
        if moved is False or not bool(getattr(self.env, "plan_success", True)):
            self.env._interactive_teleop_locked = False
            print("[fill_coffee_jar] could not move selected arm above the blue button")
            return
        tcp_z = float(self._fill_tcp_pose(side)[2])
        self._fill_press_state = {
            "side": side,
            "arm": arm,
            "return_qpos": return_qpos,
            "descended": 0.0,
            "last_tcp_z": tcp_z,
            "contact_started_at": None,
            "descent_stopped": False,
        }
    def _hold_fill_press(self):
        state = self._fill_press_state
        if state is None or state["descent_stopped"]:
            return
        remaining = self.FILL_MAX_PRESS_DEPTH - float(state["descended"])
        if remaining <= 1e-6:
            state["descent_stopped"] = True
            print("[fill_coffee_jar] press-depth cap reached without dispenser contact")
            return
        contacting = self._fill_contact(state)
        if contacting and state["contact_started_at"] is None:
            state["contact_started_at"] = time.perf_counter()
            print("[fill_coffee_jar] dispenser contact; button is physically pressed")
        step = min(self.FILL_PRESS_STEP, remaining)
        if contacting:
            step = min(max(step * 0.5, 0.002), remaining)
        self.env.plan_success = True
        moved = self.env.move(self.env.move_by_displacement(state["arm"], z=-step))
        if moved is False or not bool(getattr(self.env, "plan_success", True)):
            state["descent_stopped"] = True
            print("[fill_coffee_jar] downward press plan stopped; release Space to return")
            return
        state["descended"] += step
        if state["descended"] >= self.FILL_MAX_PRESS_DEPTH - 1e-6:
            state["descent_stopped"] = True
            print("[fill_coffee_jar] press-depth cap reached; release Space to return")

    def _return_fill_arm(self, state):
        side = state["side"]
        target = np.asarray(state["return_qpos"], dtype=np.float64)
        start = self._fill_arm_qpos(side)
        for index in range(1, self.FILL_RETURN_STEPS + 1):
            alpha = index / float(self.FILL_RETURN_STEPS)
            smooth = alpha * alpha * (3.0 - 2.0 * alpha)
            qpos = start + (target - start) * smooth
            velocity = (target - start) / max(self.FILL_RETURN_STEPS, 1)
            self.env.robot.set_arm_joints(qpos, velocity, side)
            if hasattr(self.env, "_update_kinematic_tasks"):
                self.env._update_kinematic_tasks()
            self.env.scene.step()
        self.env.robot.set_arm_joints(target, np.zeros_like(target), side)
        self.env.plan_success = True

    def _finish_fill_press(self):
        state = self._fill_press_state
        self._fill_press_state = None
        self._fill_space_started_at = None
        if state is None:
            self.env._interactive_teleop_locked = False
            return
        try:
            self._return_fill_arm(state)
        finally:
            self.env._interactive_teleop_locked = False
        print("[fill_coffee_jar] released; dispensing is driven by measured button press")

    def _robot_action(self):
        """Use the environment's physical robot helper for discrete actions."""
        e, t, arm = self.env, self.task, _arm_tag(self.env)
        try:
            # A failed plan latches this flag false. Each deliberate key press
            # starts a fresh attempt so the same action can be retried.
            e.plan_success = True
            if t == "boil_milk":
                self._turn_boil_milk_knob()
            elif t == "fill_coffee_jar":
                e._press_dispenser(arm, force_level=4)
            elif t == "pour_beer":
                if not self.holding:
                    self.holding = bool(e._grasp_lever(arm))
                elif e._lever_open_frac() < 0.10:
                    e._sweep_lever_to(arm, 0.70, n_steps=8, stop_on_foam=True)
                else:
                    e._sweep_lever_to(arm, 0.0, n_steps=5)
                    e._release_lever(arm)
                    self.holding = False
            elif t == "cook_food":
                self._turn_cook_food_knob()
            elif t == "measure_ingredient":
                e._press_switch(arm, not bool(e.tab_open))
            elif t == "make_soup":
                if not bool(e.stove_on):
                    e._turn_knob_on()
                elif self.holding:
                    e._pour_into_pot()
            elif t == "clean_table":
                if not bool(e.cup_tipped):
                    e._animate_tip()
                else:
                    spot = e._next_dirty_spot()
                    if spot is not None:
                        e._dab_spot(spot)
        except Exception as exc:
            print(f"[{t}] action unavailable: {exc}")

    def _return_arm_after_failure(self, arm):
        """Best-effort recovery for a failed scripted reach."""
        e = self.env
        try:
            e.plan_success = True
            e.move(e.open_gripper(arm))
        except Exception as exc:
            print(f"[boil_milk] could not open {arm} gripper during recovery: {exc}")
        try:
            # A failed gripper command must not prevent the return plan.
            e.plan_success = True
            e.move(e.back_to_origin(arm))
        except Exception as exc:
            print(f"[boil_milk] could not return {arm} arm to origin: {exc}")

    def _restore_failure_visual(self):
        for material, color in self._failure_visual_original.values():
            try:
                material.set_base_color(color)
                material.base_color = color
            except Exception:
                pass
        self._failure_visual_original.clear()

    def _show_failed_arm_red(self, arm):
        """Tint the failed arm's gripper red for two seconds."""
        self._restore_failure_visual()
        side = str(arm)
        robot = getattr(self.env, "robot", None)
        articulation = getattr(robot, f"{side}_entity", None)
        if articulation is None:
            return
        gripper_links = {
            "wsg_50_base_link", "gripper_left", "gripper_right",
            "finger_left", "finger_right",
        }
        red = [0.92, 0.04, 0.03, 1.0]
        for link in articulation.get_links():
            if link.get_name() not in gripper_links:
                continue
            for component in link.entity.get_components():
                if not isinstance(component, sapien.render.RenderBodyComponent):
                    continue
                for shape in component.render_shapes:
                    material = shape.material
                    key = id(material)
                    if key not in self._failure_visual_original:
                        self._failure_visual_original[key] = (
                            material, list(material.base_color)
                        )
                    try:
                        material.set_base_color(red)
                        material.base_color = red
                    except Exception:
                        pass
        self._failure_visual_until = time.perf_counter() + 2.0

    def _update_failure_visual(self):
        if self._failure_visual_original and time.perf_counter() >= self._failure_visual_until:
            self._restore_failure_visual()
            self._failure_visual_until = 0.0

    def _turn_boil_milk_knob(self):
        """Turn the knob with exactly the arm selected by interactive controls."""
        e = self.env
        selected = tuple(getattr(e, "_interactive_selected_arms", ()) or ())
        if len(selected) != 1:
            print("[boil_milk] select one arm with 1 (left) or 2 (right) before pressing Space")
            return

        from envs.utils.action import ArmTag

        arm = ArmTag(selected[0])
        previous_arm = getattr(e, "arm", None)
        stove_was_on = bool(getattr(e, "stove_on", False))
        wanted_on = not stove_was_on
        action_error = None
        try:
            e.arm = arm
            e.plan_success = True
            e._turn_knob(wanted_on)
        except Exception as exc:
            action_error = exc
        finally:
            e.arm = previous_arm
            e._ignore_knob = False

        # The physical stove state is authoritative. If it did not reach the
        # requested state, the next Space press requests that same action again.
        reached_target = bool(getattr(e, "stove_on", False)) == wanted_on
        if reached_target:
            print(f"[boil_milk] knob turned with {arm} arm")
            return

        if action_error is not None:
            print(f"[boil_milk] {arm} arm could not turn the knob: {action_error}")
        print(f"[boil_milk] {arm} arm failed; returning it to origin")
        self._return_arm_after_failure(arm)
        self._show_failed_arm_red(arm)

    def _turn_cook_food_knob(self):
        """Toggle cook_food heat with the selected arm, matching main intensity."""
        e = self.env
        selected = tuple(getattr(e, "_interactive_selected_arms", ()) or ())
        if len(selected) > 1:
            print("[cook_food] select one arm before pressing F to turn the knob")
            return

        from envs.utils.action import ArmTag

        arm = ArmTag(selected[0]) if selected else getattr(e, "arm", _arm_tag(e))
        previous_arm = getattr(e, "arm", None)
        wanted_on = not bool(getattr(e, "stove_on", False))
        on_angle = (
            -float(getattr(e, "cook_intensity", 1.0))
            * float(getattr(e, "KNOB_MAX_ANGLE", np.pi / 2))
        )
        target = on_angle if wanted_on else 0.0
        action_error = None
        try:
            e.arm = arm
            e.plan_success = True
            e._set_knob_to(target)
        except Exception as exc:
            action_error = exc
        finally:
            e.arm = previous_arm
            e._ignore_knob = False

        reached_target = bool(getattr(e, "stove_on", False)) == wanted_on
        if reached_target:
            state = "on" if wanted_on else "off"
            print(f"[cook_food] stove {state}; knob turned with {arm} arm")
            return

        if action_error is not None:
            print(f"[cook_food] {arm} arm could not turn the knob: {action_error}")
        else:
            print("[cook_food] knob did not reach the heat threshold; stove state unchanged")

    def _task_action(self):
        if self.robot:
            self._robot_action()
        else:
            self._keyboard_action()

    def _grasp_or_release(self):
        if self.task == "pour_beer":
            self._task_action()
            return
        if self.task in ("boil_milk", "fill_coffee_jar"):
            self._task_action()
            return
        if self.task == "stop_ball":
            return
        if not self.robot:
            if self.actor is not None:
                self.holding = not self.holding
                body = _rigid(self.actor)
                if self.holding:
                    _set_pose(self.actor, self.actor.get_pose().p, kinematic=True)
                    print(f"[{self.task}] prop held; arrows/Q/E move it")
                else:
                    if self.task == "trap_bug":
                        # Trap stays kinematic; release arms evaluation + kinematic fall.
                        self.env.release_trap()
                        self.trap_released = True
                    else:
                        try:
                            body.set_kinematic(False)
                            body.set_disable_gravity(False)
                        except Exception:
                            pass
                    print(f"[{self.task}] prop released")
            return
        if self.actor is None:
            self._task_action()
            return
        arm = _arm_tag(self.env)
        try:
            if not self.holding:
                self.env.plan_success = True
                if self.task == "make_soup":
                    grasped = bool(self.env._grasp_board())
                elif self.task == "clean_table":
                    grasped = bool(self.env._grasp_sponge())
                else:
                    moved = self.env.move(
                        self.env.grasp_actor(self.actor, arm_tag=arm, pre_grasp_dis=0.08)
                    )
                    grasped = moved is not False and bool(
                        getattr(self.env, "plan_success", True)
                    )
                    if grasped and self.task == "trap_bug":
                        self.env.weld_trap_to_gripper(arm)
                self.holding = grasped
                result = "ok" if self.holding else "failed; press Space to retry"
                print(f"[{self.task}] grasp {result}")
            else:
                self.env.plan_success = True
                moved = self.env.move(self.env.open_gripper(arm))
                released = moved is not False and bool(
                    getattr(self.env, "plan_success", True)
                )
                if not released:
                    print(f"[{self.task}] release failed; press Space to retry")
                    return
                if self.task == "make_soup":
                    self.env._release_board_weld()
                elif self.task == "clean_table":
                    self.env._sponge_welded = False
                    self.env._sponge_weld_offset = None
                elif self.task == "trap_bug":
                    self.env.release_trap()
                    self.trap_released = True
                self.holding = False
                print(f"[{self.task}] released")
        except Exception as exc:
            print(f"[{self.task}] grasp/release unavailable: {exc}")

    def update(self, window):
        self._update_failure_visual()
        space_down = bool(window.key_down("space"))
        if self.task == "fill_coffee_jar":
            # Space is intentionally unused for fill_coffee_jar.  The user
            # presses the blue key by moving the selected closed gripper in Z;
            # env._detect_lid_touch converts that physical engagement to force.
            self._fill_space_was_down = space_down
        elif self.space.poll(space_down):
            self._grasp_or_release()
        if self.f.poll(window.key_down("f")) and self.task not in ("boil_milk", "fill_coffee_jar"):
            self._task_action()
        if not self.robot and self.holding and self.actor is not None:
            p = np.asarray(self.actor.get_pose().p, dtype=float)
            step = 0.012
            dz = step * (bool(window.key_down("q")) - bool(window.key_down("e")))
            p += [step * (bool(window.key_down("right")) - bool(window.key_down("left"))),
                  step * (bool(window.key_down("up")) - bool(window.key_down("down"))), dz]
            if np.any(np.asarray([window.key_down(k) for k in ("left", "right", "up", "down", "q", "e")])):
                _set_pose(self.actor, p, kinematic=True)
            if self.task == "make_soup" and self.board_over_pot and not getattr(self.env, "_veg_released", False):
                tilt = 0.018 * (
                    bool(window.key_down("t")) - bool(window.key_down("r"))
                )
                if abs(tilt) > 0.0:
                    pose = self.actor.get_pose()
                    q = qmult(axangle2quat([0.0, 1.0, 0.0], tilt), list(pose.q))
                    _set_pose(self.actor, pose.p, quat=q, kinematic=True)
                    # Once the board is substantially tipped, let PhysX carry
                    # the vegetables into the pot instead of teleporting them.
                    if float(self.env._board_up_dot()) < float(self.env.tilt_hold_dot):
                        self.env._pour_armed = True
                        self.env._release_veggies_physics()
                        print("[make_soup] vegetables released from tilted board")

    def start_scenario(self):
        """Start time-sensitive scene motion after the first rendered frame."""
        if self.scenario_started:
            return
        self.scenario_started = True
        e = self.env
        try:
            if self.task == "trap_bug":
                e._start_bug()
            elif self.task == "catch_cup":
                e._release_cup()
            elif self.task == "mouse_object_drop":
                e._activate_target()
                e._release_mouse()
            elif self.task == "stop_ball":
                e._release_ball()
            elif self.task == "clean_table" and not bool(e.cup_tipped):
                e._animate_tip()
            else:
                return
            print(f"[{self.task}] scenario started automatically")
        except Exception as exc:
            print(f"[{self.task}] automatic start failed: {exc}")

    def after_step(self):
        """Ensure release is armed; landing freeze is handled in the env (pose as-is)."""
        if self.task != "trap_bug" or not self.trap_released:
            return
        env = self.env
        if env.trap is None:
            return
        if not bool(getattr(env, "_trap_released", False)):
            env.release_trap()
        if bool(getattr(env, "_trap_anchored", False)) and not getattr(self, "_trap_land_logged", False):
            self._trap_land_logged = True
            print("[trap_bug] trap landed; pose frozen as-is")


def _terminal_failure(env, task):
    """Return an irreversible task failure reason, or ``None`` while playable."""
    if task == "trap_bug":
        if bool(getattr(env, "_bug_escaped", False)):
            return "bug went back into hiding"
        if bool(getattr(env, "_trap_anchored", False)):
            return "trap missed the bug"
    elif task in ("boil_milk", "pour_beer", "measure_ingredient"):
        if bool(getattr(env, "overflowed", False)):
            return "liquid overflowed"
        if task == "boil_milk":
            turned_on = bool(getattr(env, "turned_on_once", False))
            stove_off = not bool(getattr(env, "stove_on", False))
            reached_target = bool(getattr(env, "reached_target", False))
            max_level = float(getattr(env, "max_liquid_level", 0.0))
            target_level = float(getattr(env, "target_level", 1.0))
            if turned_on and stove_off and not reached_target and max_level < target_level - 1e-3:
                return "stove turned off before milk reached the target"
        if task == "measure_ingredient" and float(getattr(env, "spill_amount", 0.0)) > 1e-4:
            return "ingredient spilled"
    elif task == "fill_coffee_jar":
        try:
            _lo, hi = env._fill_band()
            if float(env._current_fill()) > float(hi) + 1e-3:
                return "coffee jar overfilled"
        except Exception:
            pass
    elif task == "cook_food":
        try:
            if float(env.doneness) > float(env.target_doneness_range[1]) + 1e-3:
                return "food overcooked"
            grasped = getattr(env, "_grasp_doneness", None)
            if grasped is not None and not env._doneness_in_target_range(float(grasped)):
                return "food removed outside target doneness"
        except Exception:
            pass
    elif task == "make_soup":
        if bool(getattr(env, "_veg_fallen", False)):
            return "vegetables spilled outside the pot"
    elif task == "catch_cup":
        if bool(getattr(env, "_fell_on_table", False)) or getattr(env, "_cup_state", "") == "fallen":
            return "cup fell on the table"
    elif task == "mouse_object_drop":
        if bool(getattr(env, "_fell_on_table", False)) or getattr(env, "_obj_state", "") == "fallen":
            return "object missed the basket"
    elif task == "stop_ball":
        if bool(getattr(env, "_fell_off", False)) or getattr(env, "_ball_state", "") == "fallen":
            return "ball fell off the table"
    elif task == "clean_table":
        if bool(getattr(env, "laptop_reached", False)):
            return "spill reached the laptop"
    return None


def run_task(task, args, keyboard_controls, robot_controls, post_setup=None):
    module_name, class_name, _, _ = PROFILES[task]
    module = importlib.import_module(module_name)
    task_cls = getattr(module, class_name)
    from envs import CONFIGS_PATH
    globals()["CONFIGS_PATH"] = CONFIGS_PATH
    config = configure_task(task, args.config, args.seed, args.control == "robot")
    task_args = config.setdefault("task_args", {}).setdefault(task, {})
    for item in args.task_arg:
        if "=" not in item:
            raise SystemExit(f"--task-arg expects key=value, got: {item}")
        key, raw = item.split("=", 1)
        if raw.lower() in ("true", "false"):
            value = raw.lower() == "true"
        else:
            try:
                value = float(raw) if any(c in raw for c in ".eE") else int(raw)
            except ValueError:
                value = raw
        task_args[key.strip()] = value
    env = task_cls()
    env.setup_demo(**config)
    if post_setup is not None:
        post_setup(env)
    env._interactive_robot_mode = args.control == "robot"
    print_mode_controls(task, args.control, keyboard=keyboard_controls, robot=robot_controls)
    controller = HouseholdController(env, task, robot=args.control == "robot")
    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    views = make_viewer_view_toggle(env, viewer)
    rendered_frames = 0
    terminal_result = None  # True=success, False=failure, None=manual close/smoke
    terminal_started_at = None
    try:
        while not viewer.closed:
            views.update(viewer.window)
            start = time.perf_counter()
            controller.update(viewer.window)
            if hasattr(env, "_update_kinematic_tasks"):
                env._update_kinematic_tasks()
            env.scene.step()
            controller.after_step()
            env.scene.update_render()
            viewer.render()
            rendered_frames += 1
            if rendered_frames == 1 and not args.smoke_test:
                controller.start_scenario()
            if viewer.window.key_down("escape"):
                break
            if args.smoke_test and rendered_frames >= 3:
                print(f"[{task}] smoke test rendered {rendered_frames} frames")
                break
            if terminal_started_at is not None:
                if time.perf_counter() - terminal_started_at >= 2.0:
                    print(f"[{task}] closing after 2-second terminal-result display")
                    break
                remaining = float(env.scene.get_timestep()) - (time.perf_counter() - start)
                if remaining > 0:
                    time.sleep(remaining)
                continue
            try:
                succeeded = bool(env.check_success())
            except Exception as exc:
                print(f"[{task}] success check failed: {exc}")
                succeeded = False
            failure = None if succeeded else _terminal_failure(env, task)
            if succeeded:
                print(f"[{task}] terminal result: SUCCESS")
                terminal_result = True
                terminal_started_at = time.perf_counter()
            if failure is not None:
                print(f"[{task}] terminal result: FAILURE ({failure})")
                terminal_result = False
                terminal_started_at = time.perf_counter()
            remaining = float(env.scene.get_timestep()) - (time.perf_counter() - start)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        try:
            report_task_result(env)
        finally:
            try:
                viewer.close()
            except Exception:
                pass
            env.close_env()
    return terminal_result


def make_parser(task, description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=PROFILES[task][2], help="task_config name without .yml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--control", choices=("keyboard", "robot"), default="robot")
    p.add_argument("--robot-motion", choices=("planner", "interpolate"), default="interpolate")
    p.add_argument("--task-arg", action="append", default=[], help="override task_args entry (key=value)")
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="initialize, render three frames, report state, and exit",
    )
    return p
