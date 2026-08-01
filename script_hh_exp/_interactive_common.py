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
    "boil_milk": ("envs.boil_milk", "boil_milk", "demo_kitchens", None),
    "fill_coffee_jar": ("envs.fill_coffee_jar", "fill_coffee_jar", "demo_clean", None),
    "pour_beer": ("envs.pour_beer", "pour_beer", "demo_pour_beer", None),
    "cook_food": ("envs.cook_food", "cook_food", "demo_kitchens", "food"),
    "measure_ingredient": ("envs.measure_ingredient", "measure_ingredient", "demo_kitchens", "jar"),
    "make_soup": ("envs.make_soup", "make_soup", "demo_kitchens", "board"),
    "catch_cup": ("envs.catch_cup", "catch_cup", "demo_dynamic", "pillow"),
    "mouse_object_drop": ("envs.mouse_object_drop", "mouse_object_drop", "demo_dynamic", "basket"),
    "stop_ball": ("envs.stop_ball", "stop_ball", "demo_dynamic", None),
    "clean_table": ("envs.clean_table", "clean_table", "demo_clean_table", "sponge"),
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

    def __init__(self, env, task, robot=False):
        self.env, self.task, self.robot = env, task, robot
        actor_attr = PROFILES[task][3]
        self.actor = getattr(env, actor_attr, None) if actor_attr else None
        self.holding = False
        self.space = _Edge()
        self.f = _Edge()
        self.board_over_pot = False
        self.trap_released = False
        if not robot and self.actor is not None:
            _set_pose(self.actor, self.actor.get_pose().p, kinematic=True)

    def _keyboard_action(self):
        """Operate task state directly without moving either robot arm."""
        e, t = self.env, self.task
        try:
            if t == "trap_bug":
                e._start_bug()
            elif t == "boil_milk":
                e._set_stove(not bool(e.stove_on))
            elif t == "fill_coffee_jar":
                remaining = max(0, int(e._beans_needed()) - int(e._effective_bean_count()))
                count = min(remaining, int(e.beans_per_force_level[-1]))
                for _ in range(count):
                    e._spawn_one_dispensed_bean()
                e.beans_in_jar += count
                e.press_count += 1
                e._sync_fill_visual()
                print(f"[fill_coffee_jar] dispensed {count} beans; fill={e._current_fill():.0%}")
            elif t == "pour_beer":
                target = 0.0 if e._lever_open_frac() > 0.05 else 0.70
                e._apply_lever_pose(target * float(e.lever_open_rad))
                print(f"[pour_beer] lever {'closed' if target == 0 else 'opened'}")
            elif t == "cook_food":
                angle = -float(getattr(e, "KNOB_MAX_ANGLE", np.pi / 2))
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
            elif t == "catch_cup":
                e._release_cup()
            elif t == "mouse_object_drop":
                e._activate_target()
                e._release_mouse()
            elif t == "stop_ball":
                e._release_ball()
            elif t == "clean_table":
                if not bool(e.cup_tipped):
                    e._animate_tip()
                else:
                    cleared = e._try_clear_spots_under_sponge()
                    print(f"[clean_table] cleared {cleared} spot(s) under the sponge")
        except Exception as exc:
            print(f"[{t}] action unavailable: {exc}")

    def _robot_action(self):
        """Use the environment's physical robot helper for discrete actions."""
        e, t, arm = self.env, self.task, _arm_tag(self.env)
        try:
            if t == "trap_bug":
                e._start_bug()
            elif t == "boil_milk":
                e._turn_knob(not bool(e.stove_on))
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
                angle = -float(getattr(e, "KNOB_MAX_ANGLE", np.pi / 2))
                e._set_knob_to(0.0 if bool(e.stove_on) else angle)
            elif t == "measure_ingredient":
                e._press_switch(arm, not bool(e.tab_open))
            elif t == "make_soup":
                if not bool(e.stove_on):
                    e._turn_knob_on()
                elif self.holding:
                    e._pour_into_pot()
            elif t == "catch_cup":
                e._release_cup()
            elif t == "mouse_object_drop":
                e._activate_target()
                e._release_mouse()
            elif t == "stop_ball":
                e._release_ball()
            elif t == "clean_table":
                if not bool(e.cup_tipped):
                    e._animate_tip()
                else:
                    spot = e._next_dirty_spot()
                    if spot is not None:
                        e._dab_spot(spot)
        except Exception as exc:
            print(f"[{t}] action unavailable: {exc}")

    def _task_action(self):
        if self.robot:
            self._robot_action()
        else:
            self._keyboard_action()

    def _grasp_or_release(self):
        if self.task == "pour_beer":
            self._task_action()
            return
        if self.task in ("boil_milk", "fill_coffee_jar", "stop_ball"):
            self._task_action()
            return
        if not self.robot:
            if self.actor is not None:
                self.holding = not self.holding
                body = _rigid(self.actor)
                if self.holding:
                    _set_pose(self.actor, self.actor.get_pose().p, kinematic=True)
                    print(f"[{self.task}] prop held; arrows/Q/E move it")
                else:
                    try:
                        body.set_kinematic(False)
                        body.set_disable_gravity(False)
                    except Exception:
                        pass
                    print(f"[{self.task}] prop released")
                    if self.task == "trap_bug":
                        self.trap_released = True
            return
        if self.actor is None:
            self._task_action()
            return
        arm = _arm_tag(self.env)
        try:
            if not self.holding:
                if self.task == "make_soup":
                    self.holding = bool(self.env._grasp_board())
                elif self.task == "clean_table":
                    self.holding = bool(self.env._grasp_sponge())
                else:
                    self.env.move(self.env.grasp_actor(self.actor, arm_tag=arm, pre_grasp_dis=0.08))
                    self.holding = bool(getattr(self.env, "plan_success", True))
                print(f"[{self.task}] grasp {'ok' if self.holding else 'failed'}")
            else:
                self.env.move(self.env.open_gripper(arm))
                if self.task == "make_soup":
                    self.env._release_board_weld()
                elif self.task == "clean_table":
                    self.env._sponge_welded = False
                    self.env._sponge_weld_offset = None
                self.holding = False
                if self.task == "trap_bug":
                    self.trap_released = True
                print(f"[{self.task}] released")
        except Exception as exc:
            print(f"[{self.task}] grasp/release unavailable: {exc}")

    def update(self, window):
        if self.space.poll(window.key_down("space")):
            self._grasp_or_release()
        if self.f.poll(window.key_down("f")):
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

    def after_step(self):
        """Freeze a released trap only after it physically reaches the table."""
        if self.task != "trap_bug" or not self.trap_released:
            return
        env = self.env
        if getattr(env, "_trap_anchored", False) or env.trap is None:
            return
        seated_z = float(env.table_top + env.trap_half[2])
        trap_z = float(env.trap.get_pose().p[2])
        if trap_z <= seated_z + 0.008:
            env._anchor_trap()
            print("[trap_bug] trap reached the table and is now static")


def _terminal_failure(env, task):
    """Return an irreversible task failure reason, or ``None`` while playable."""
    if task == "trap_bug":
        if bool(getattr(env, "_trap_anchored", False)):
            return "trap missed the bug"
    elif task in ("boil_milk", "pour_beer", "measure_ingredient"):
        if bool(getattr(env, "overflowed", False)):
            return "liquid overflowed"
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
            if viewer.window.key_down("escape"):
                break
            if args.smoke_test and rendered_frames >= 3:
                print(f"[{task}] smoke test rendered {rendered_frames} frames")
                break
            try:
                succeeded = bool(env.check_success())
            except Exception as exc:
                print(f"[{task}] success check failed: {exc}")
                succeeded = False
            failure = None if succeeded else _terminal_failure(env, task)
            if succeeded:
                print(f"[{task}] terminal result: SUCCESS")
                terminal_result = True
                break
            if failure is not None:
                print(f"[{task}] terminal result: FAILURE ({failure})")
                terminal_result = False
                break
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
