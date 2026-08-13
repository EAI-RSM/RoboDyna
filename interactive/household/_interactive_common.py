"""Shared interactive runner for the household task sandboxes.

The household environments deliberately keep their normal physics and success
checks.  This module only adds the same viewer/arm teleoperation used by
``interactive/_interactive_common.py``: arrows move the selected end-effector in XY, Q/E move it in Z,
F/G tip it left/right about world Y, and 1/2/3 select the left/right/both arms.
Space opens/closes the selected gripper(s) only (shared ``ViewerViewToggle``);
V cycles head_camera ↔ gripper views (default head framing matches base suite
GUI snapshots — top-down is not available).  Space never auto-grasps, teleports,
or runs planner pick/place shortcuts.
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BENCH = ROOT / "script" / "bench_script"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))
from interactive._interactive_common import (  # noqa: E402
    RealtimePhysicsPacer,
    action_failed,
    configure_task,
    flash_gripper_failure,
    gripper_failure_feedback,
    make_viewer_view_toggle,
    print_episode_condition,
    print_failure,
    print_instructions,
    print_mode_controls,
    print_success,
    report_task_result,
    require_selected_arms,
    resolve_action_arm,
)


PROFILES = {
    "trap_bug": ("envs.trap_bug", "trap_bug", "demo_dynamic", "trap"),
    "boil_milk": ("envs.boil_milk", "boil_milk", "demo_dynamic", None),
    "fill_coffee_jar": ("envs.fill_coffee_jar", "fill_coffee_jar", "demo_dynamic", None),
    "pour_beer": ("envs.pour_beer", "pour_beer", "demo_dynamic", None),
    "cook_food": ("envs.cook_food", "cook_food", "demo_dynamic", "food"),
    "cook_food_timer": ("envs.cook_food_timer", "cook_food_timer", "demo_dynamic", "food"),
    "measure_ingredient": ("envs.measure_ingredient", "measure_ingredient", "demo_dynamic", "jar"),
    "make_soup": ("envs.make_soup", "make_soup", "demo_dynamic", "board"),
    "catch_cup": ("envs.catch_cup", "catch_cup", "demo_dynamic", "pillow"),
    "catch_mouse_object_drop": ("envs.catch_mouse_object_drop", "catch_mouse_object_drop", "demo_dynamic", "basket"),
    "stop_ball": ("envs.stop_ball", "stop_ball", "demo_dynamic", None),
    "clean_table": ("envs.clean_table", "clean_table", "demo_dynamic", "sponge"),
}


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


def _arm_tag(env, *, exactly_one: bool = True):
    """Return the highlighted arm, or ``None`` when no action should run."""
    from envs.utils.action import ArmTag
    return resolve_action_arm(env, ArmTag, exactly_one=exactly_one)


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
        self._fill_press_state = None
        self.trap_released = False
        self.scenario_started = False
        if task in (
            "fill_coffee_jar", "pour_beer", "measure_ingredient", "catch_cup",
        ) and robot:
            self._close_grippers_at_start()
        elif task == "stop_ball" and robot:
            self._open_stop_ball_grippers_at_start()
        elif task in ("clean_table", "make_soup") and robot:
            # Start open so the WSG can close around the handle cube.
            self._open_clean_table_grippers_at_start()
        # catch_cup / make_soup: keep the prop dynamic so gripper contact is real.
        # Other keyboard tasks start kinematic so arrows can teleport the prop.
        if not robot and self.actor is not None and task not in (
            "catch_cup",
            "make_soup",
        ):
            _set_pose(self.actor, self.actor.get_pose().p, kinematic=True)
        elif task == "catch_cup" and self.actor is not None:
            try:
                self.env._enable_pillow_physics()
            except Exception:
                pass

    def _close_grippers_at_start(self):
        """Start robot mode with both grippers closed at once."""
        try:
            self.env.plan_success = True
            self.env.together_close_gripper(save_freq=None)
        except Exception as exc:
            print(f"[{self.task}] could not pre-close grippers: {exc}")
        self.env.plan_success = True

    def _open_stop_ball_grippers_at_start(self):
        """stop_ball: start open so the fingers can catch the rolling ball."""
        try:
            self.env.plan_success = True
            opener = getattr(self.env, "together_open_gripper", None)
            if callable(opener):
                opener(save_freq=None, left_pos=1.0, right_pos=1.0)
            else:
                from envs.utils.action import ArmTag

                for side in ("left", "right"):
                    self.env.plan_success = True
                    self.env.move(self.env.open_gripper(ArmTag(side), pos=1.0))
        except Exception as exc:
            print(f"[stop_ball] could not pre-open grippers: {exc}")
        self.env.plan_success = True

    def _open_clean_table_grippers_at_start(self):
        """Start robot mode with both grippers fully open (handle pinch tasks)."""
        if self.task not in ("clean_table", "make_soup"):
            return
        try:
            self.env.plan_success = True
            opener = getattr(self.env, "together_open_gripper", None)
            if callable(opener):
                opener(save_freq=None, left_pos=1.0, right_pos=1.0)
            else:
                from envs.utils.action import ArmTag

                for side in ("left", "right"):
                    self.env.plan_success = True
                    self.env.move(self.env.open_gripper(ArmTag(side), pos=1.0))
        except Exception as exc:
            print(f"[{self.task}] could not pre-open grippers: {exc}")
        self.env.plan_success = True

    def _keyboard_action(self):
        """Operate task state directly without moving either robot arm."""
        e, t = self.env, self.task
        try:
            if t == "boil_milk":
                # No keyboard snap — fire follows a physical gripper twist only.
                print(
                    "[boil_milk] turn the stove with the gripper on the knob "
                    "(close on the knob and twist with teleop)"
                )
            elif t == "fill_coffee_jar":
                self._fill_coffee_press(1)
            elif t in ("cook_food", "cook_food_timer"):
                # No keyboard snap — fire follows a physical gripper twist only.
                print(
                    f"[{t}] turn the stove with the gripper on the knob "
                    "(close on the knob and twist with teleop)"
                )
            elif t == "measure_ingredient":
                # No C/keyboard proxy — oil key is pressed by lowering the gripper.
                print(
                    "[measure_ingredient] push jar under nozzle, then press the "
                    "green key (ON/OFF); success is checked after OFF"
                )
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
        selected = require_selected_arms(self.env, exactly_one=True)
        if not selected:
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
            action_failed(
                self.env, (side,),
                detail="could not move above the blue button",
            )
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
        e, t = self.env, self.task
        arm = None
        try:
            # A failed plan latches this flag false. Each deliberate key press
            # starts a fresh attempt so the same action can be retried.
            e.plan_success = True
            if t == "boil_milk":
                self._turn_boil_milk_knob()
            elif t == "fill_coffee_jar":
                arm = _arm_tag(e)
                if arm is None:
                    return
                e._press_dispenser(arm, force_level=4)
                if not bool(getattr(e, "plan_success", True)):
                    action_failed(e, (str(arm),), detail="dispenser press failed")
            elif t == "measure_ingredient":
                # No C assist — oil key is pressed by lowering the closed gripper.
                print(
                    "[measure_ingredient] push jar under nozzle, then lower onto "
                    "the green key; success is checked after the key turns OFF"
                )
        except Exception as exc:
            if arm is not None:
                action_failed(e, (str(arm),), detail=f"action unavailable: {exc}")
            else:
                print(f"[{t}] action unavailable: {exc}")

    def _snap_stove_knob(self, want_on: bool, *, continuous_angle: float | None = None):
        """Instant interactive knob/fire update (no planner reach).

        Matches the snappy base-suite control feel: teleop stays on UniversalRobotControls;
        Space/C only flips task state instead of running a multi-second expert path.
        """
        e = self.env
        # Drop debounce caches so the snap is applied even if the joint was near
        # the previous committed angle.
        e._last_committed_knob_angle = None
        e._stove_fire_visual = None
        if continuous_angle is not None and callable(getattr(e, "_set_knob_angle", None)):
            e._set_knob_angle(0.0 if not want_on else float(continuous_angle))
            return
        angle = e.KNOB_ON_ANGLE if want_on else e.KNOB_OFF_ANGLE
        if callable(getattr(e, "_set_knob_joint_angle", None)):
            e._set_knob_joint_angle(angle, hard=True)
        if callable(getattr(e, "_set_stove", None)):
            e._set_stove(bool(want_on))

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

    def _show_failed_arm_red(self, arm):
        """Tint the failed arm's gripper red for two seconds."""
        flash_gripper_failure(self.env, (str(arm),))

    def _update_failure_visual(self):
        gripper_failure_feedback(self.env).update()

    def _turn_boil_milk_knob(self):
        """Turn the knob with exactly the arm selected by interactive controls."""
        e = self.env
        arm = _arm_tag(e)
        if arm is None:
            return

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

        detail = (
            f"could not turn the knob: {action_error}"
            if action_error is not None
            else "could not turn the knob"
        )
        print(f"[boil_milk] {arm} arm failed; returning it to origin")
        self._return_arm_after_failure(arm)
        action_failed(e, (str(arm),), detail=detail)

    def _task_action(self):
        if self.robot:
            self._robot_action()
        else:
            self._keyboard_action()

    def update(self, window):
        del window  # Space / teleop owned by shared ViewerViewToggle + UniversalRobotControls
        self._update_failure_visual()
        # Space opens/closes grippers only (ViewerViewToggle). No auto-grasp,
        # teleport hold, or planner pick/place shortcuts from this controller.

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
            elif self.task == "catch_mouse_object_drop":
                # Release shelf objects and start the mouse immediately (no
                # stand-off wait for basket placement).
                e._activate_target()
                e._allow_shove = True
                e._release_mouse()
            elif self.task == "stop_ball":
                # Ball goes live at handoff; a miss keeps rolling until fall-off.
                e._release_ball()
            elif self.task == "clean_table" and not bool(e.cup_tipped):
                e._animate_tip()
            else:
                return
            print(f"[{self.task}] scenario started automatically")
        except Exception as exc:
            print(f"[{self.task}] automatic start failed: {exc}")

    def after_step(self):
        """Post-physics hooks (release arming, catch/miss latch, trap freeze)."""
        env = self.env
        if self.task == "catch_mouse_object_drop":
            # Catch/miss is evaluated before scene.step inside
            # _update_kinematic_tasks; re-check after the step so a table
            # landing latches immediately for terminal failure.
            try:
                env._update_catch_state()
            except Exception:
                pass
            return
        if self.task != "trap_bug":
            return
        if env.trap is None:
            return
        # Gripper open / slip is detected in env._maybe_auto_weld_or_release.
        if bool(getattr(env, "_trap_released", False)):
            self.trap_released = True
        if bool(getattr(env, "_trap_anchored", False)) and not getattr(self, "_trap_land_logged", False):
            self._trap_land_logged = True
            print("[trap_bug] trap landed; pose frozen as-is")


def _fill_level_detail(env, task: str) -> str:
    """Compact fill readout for terminal SUCCESS/FAILURE lines."""
    if task == "pour_beer":
        liq = 100.0 * float(getattr(env, "liquid_level", 0.0))
        foam = 100.0 * float(getattr(env, "foam_level", 0.0))
        try:
            total = 100.0 * float(env._total_fill())
        except Exception:
            total = liq + foam
        tgt = 100.0 * float(getattr(env, "target_liquid", 0.90))
        band = f"need>{tgt:.0f}%"
        if bool(getattr(env, "overflowed", False)):
            return f"OVERFLOW beer={liq:.0f}% foam={foam:.0f}% total={total:.0f}%"
        return f"beer={liq:.0f}% foam={foam:.0f}% total={total:.0f}% {band}"
    if task == "boil_milk":
        lvl = 100.0 * float(getattr(env, "liquid_level", 0.0))
        tgt = 100.0 * float(getattr(env, "target_level", 1.0))
        if bool(getattr(env, "overflowed", False)):
            return f"OVERFLOW level={lvl:.0f}%"
        return f"level={lvl:.0f}% target={tgt:.0f}%"
    if task == "measure_ingredient":
        lvl = 100.0 * float(getattr(env, "liquid_level", 0.0))
        try:
            lo, hi = env._fill_band()
            lo_pct, hi_pct = 100.0 * float(lo), 100.0 * float(hi)
        except Exception:
            tgt = 100.0 * float(getattr(env, "target_fill", 0.0))
            tol = 100.0 * float(getattr(env, "fill_tol", 0.05))
            lo_pct, hi_pct = tgt - tol, tgt + tol
        if bool(getattr(env, "overflowed", False)):
            return f"OVERFLOW fill={lvl:.0f}%"
        return f"fill={lvl:.0f}% target={lo_pct:.0f}–{hi_pct:.0f}%"
    if task == "fill_coffee_jar":
        try:
            fill = 100.0 * float(env._current_fill())
        except Exception:
            fill = 0.0
        tgt = 100.0 * float(getattr(env, "target_fill", 0.0))
        tol = 100.0 * float(getattr(env, "fill_tol", 0.05))
        idle = float(getattr(env, "_press_idle_s", 0.0))
        need = float(getattr(env, "idle_score_sec", 3.0))
        return (
            f"fill={fill:.0f}% target={tgt - tol:.0f}–{tgt + tol:.0f}% "
            f"idle={idle:.1f}/{need:.0f}s"
        )
    return ""


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
        # pour_beer: score only when the finish bell is pressed.
        if task == "pour_beer" and bool(getattr(env, "_bell_pressed", False)):
            try:
                if bool(env._pour_quality_ok()):
                    return None
            except Exception:
                pass
            liq = 100.0 * float(getattr(env, "liquid_level", 0.0))
            tgt = 100.0 * float(getattr(env, "target_liquid", 0.85))
            if not bool(getattr(env, "opened_once", False)):
                return "finish bell pressed before pouring"
            if liq + 1e-3 <= tgt:
                return f"finish bell pressed with underfill beer={liq:.0f}% need>{tgt:.0f}%"
            return "finish bell pressed without meeting pour criteria"
        if task == "boil_milk":
            turned_on = bool(getattr(env, "turned_on_once", False))
            stove_off = not bool(getattr(env, "stove_on", False))
            reached_target = bool(getattr(env, "reached_target", False))
            max_level = float(getattr(env, "max_liquid_level", 0.0))
            target_level = float(getattr(env, "target_level", 1.0))
            if turned_on and stove_off and not reached_target and max_level < target_level - 1e-3:
                return "stove turned off before milk reached the target"
        if task == "measure_ingredient":
            if float(getattr(env, "spill_amount", 0.0)) > 1e-4:
                return "ingredient spilled"
            # Score only after the nozzle key is turned OFF (success is checked
            # first in the viewer loop; this path is the fail branch).
            if bool(getattr(env, "closed_after_pour", False)):
                lvl = float(getattr(env, "liquid_level", 0.0))
                try:
                    under = bool(env._jar_under_nozzle())
                except Exception:
                    under = False
                if not under:
                    return "switch turned off with jar not under the nozzle"
                try:
                    lo, hi = env._fill_band()
                except Exception:
                    tgt = float(getattr(env, "target_fill", 0.0))
                    tol = float(getattr(env, "fill_tol", 0.05))
                    lo, hi = tgt - tol, tgt + tol
                if lvl + 1e-3 < float(lo) or lvl - 1e-3 > float(hi):
                    return "switch turned off with incorrect fill level"
                return "switch turned off without meeting success criteria"
    elif task == "fill_coffee_jar":
        try:
            if not bool(env._fill_ready_to_score()):
                return None
            fill = float(env._current_fill())
            lo, hi = env._fill_band()
            if fill > float(hi) + 1e-3:
                return "coffee jar overfilled after idle"
            if fill + 1e-3 < float(lo):
                return "coffee jar underfilled after idle"
            return "fill outside target after idle"
        except Exception:
            pass
    elif task in ("cook_food", "cook_food_timer"):
        try:
            lo, hi = env.target_doneness_range
            # Overcook is irreversible — fail as soon as doneness passes the band.
            if float(env.doneness) > float(hi) + 1e-3:
                return "food overcooked"
            # Under/out-of-band after shutoff only (success is checked first).
            stove_off = (
                not bool(getattr(env, "stove_on", False))
                and float(getattr(env, "fire_intensity", 1.0)) <= 0.02
                and float(getattr(env, "knob_angle", -1.0)) >= -0.05
            )
            if not stove_off or not bool(getattr(env, "turned_off_after_cook", False)):
                return None
            grasped = getattr(env, "_grasp_doneness", None)
            score = float(env.doneness) if grasped is None else float(grasped)
            if score + 1e-3 < float(lo):
                return "food undercooked"
            if not env._doneness_in_target_range(score):
                return "food outside target doneness"
        except Exception:
            pass
    elif task == "make_soup":
        if bool(getattr(env, "_arm_veg_contact", False)):
            return getattr(env, "_fail_reason", None) or "arm contacted vegetables"
        if bool(getattr(env, "_veg_fallen", False)):
            return getattr(env, "_fail_reason", None) or "vegetables dropped on the table"
        # Live re-check so a mid-episode spill latches before the next success poll.
        try:
            if callable(getattr(env, "_check_veg_fallen", None)):
                env._check_veg_fallen()
            if callable(getattr(env, "_check_arm_veg_contact", None)):
                env._check_arm_veg_contact()
        except Exception:
            pass
        if bool(getattr(env, "_arm_veg_contact", False)):
            return getattr(env, "_fail_reason", None) or "arm contacted vegetables"
        if bool(getattr(env, "_veg_fallen", False)):
            return getattr(env, "_fail_reason", None) or "vegetables dropped on the table"
    elif task == "catch_cup":
        if bool(getattr(env, "_fell_on_table", False)) or getattr(env, "_cup_state", "") == "fallen":
            return "cup fell on the table"
    elif task == "catch_mouse_object_drop":
        # Live re-check: tipped landings used to miss pose-based contact, so
        # latch failure from the AABB table test before reading the flags.
        try:
            if callable(getattr(env, "_object_touches_table", None)) and env._object_touches_table():
                env._fell_on_table = True
                env._caught = False
                env._obj_state = "fallen"
        except Exception:
            pass
        if bool(getattr(env, "_fell_on_table", False)) or getattr(env, "_obj_state", "") == "fallen":
            return "object fell on the table"
    elif task == "stop_ball":
        if bool(getattr(env, "_fell_off", False)) or getattr(env, "_ball_state", "") == "fallen":
            return "ball fell off the table without being stopped"
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
    env._interactive_robot_mode = args.control == "robot"
    # Viewer sessions (keyboard or robot teleop) — tasks use this for live physics
    # handoff / miss-continues-until-fall behavior (e.g. stop_ball). Set before
    # setup_demo so interactive warning suppression applies during load.
    env._interactive_session = True
    env.setup_demo(**config)
    if post_setup is not None:
        post_setup(env)
    print_episode_condition(env, task)
    print_mode_controls(task, args.control, keyboard=keyboard_controls, robot=robot_controls)
    controller = HouseholdController(env, task, robot=args.control == "robot")
    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    views = make_viewer_view_toggle(env, viewer)
    rendered_frames = 0
    terminal_result = None  # True=success, False=failure, None=manual close/smoke
    terminal_started_at = None
    terminal_fill_detail = ""
    terminal_failure_reason = None
    # Match interactive/_interactive_common run_viewer_loop: teleop once per display frame, then
    # fixed-dt physics catch-up so 60 Hz / 240 Hz monitors feel the same speed.
    # Success checks every few *physics* steps keep kitchen eval cost down.
    SUCCESS_CHECK_EVERY = 5
    pacer = RealtimePhysicsPacer(env)
    physics_steps = 0
    try:
        while not viewer.closed:
            n_steps = pacer.begin_frame()
            views.update(viewer.window)
            if n_steps == 0:
                env.scene.update_render()
                viewer.render()
                if viewer.window.key_down("escape"):
                    break
                if terminal_started_at is not None and time.perf_counter() - terminal_started_at >= 2.0:
                    print(f"[{task}] closing after 2-second terminal-result display")
                    break
                continue

            controller.update(viewer.window)
            for _ in range(n_steps):
                if hasattr(env, "_update_kinematic_tasks"):
                    env._update_kinematic_tasks()
                env.scene.step()
                controller.after_step()
                physics_steps += 1

                if terminal_started_at is not None:
                    continue
                if physics_steps % SUCCESS_CHECK_EVERY == 0:
                    try:
                        succeeded = bool(env.check_success())
                    except Exception as exc:
                        print_failure(f"[{task}] success check failed: {exc}")
                        succeeded = False
                    failure = None if succeeded else _terminal_failure(env, task)
                    fill = _fill_level_detail(env, task)
                    if fill:
                        terminal_fill_detail = fill
                    if succeeded:
                        msg = f"[{task}] terminal result: SUCCESS"
                        if fill:
                            msg = f"{msg} ({fill})"
                        print_success(msg)
                        terminal_result = True
                        terminal_started_at = time.perf_counter()
                    if failure is not None:
                        if fill and fill not in failure:
                            terminal_failure_reason = f"{failure}; {fill}"
                        else:
                            terminal_failure_reason = failure
                        print_failure(
                            f"[{task}] terminal result: FAILURE ({terminal_failure_reason})"
                        )
                        terminal_result = False
                        terminal_started_at = time.perf_counter()

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
            if terminal_started_at is not None and time.perf_counter() - terminal_started_at >= 2.0:
                print(f"[{task}] closing after 2-second terminal-result display")
                break
    finally:
        try:
            if not terminal_fill_detail:
                terminal_fill_detail = _fill_level_detail(env, task)
            if terminal_result is False:
                detail = terminal_failure_reason or terminal_fill_detail or None
            else:
                detail = terminal_fill_detail or None
            report_task_result(env, detail=detail)
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
