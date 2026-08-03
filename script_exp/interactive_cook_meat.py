#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive viewer for ``cook_meat``.

Run from any directory:

    /path/to/RoboDynaExp/script_exp/interactive_cook_meat.py --control keyboard
    /path/to/RoboDynaExp/script_exp/interactive_cook_meat.py --control robot
    /path/to/RoboDynaExp/script_exp/interactive_cook_meat.py --control robot --robot-motion interpolate

Respects ``cook_button_enabled`` from the selected config. When enabled, keyboard
latches ``station["_expert_key_held"]`` and robot mode presses the matching key.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import sapien
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "script" / "bench_script"))
sys.path.insert(0, str(REPO_ROOT / "script_exp"))

from _interactive_common import (  # noqa: E402
    action_failed,
    make_viewer_view_toggle,
    add_robot_motion_arg,
    make_button_controller,
    report_task_result,
    print_mode_controls,
    require_selected_arms,
)


CONTROLS_KEYBOARD = """
  Hold Space       →  cook (all stations / primary)
  Hold Q           →  cook LEFT station only (dual)
  Hold E           →  cook RIGHT station only (dual)
  G                →  toggle steak(s): board ↔ pan
  P                →  snap steak(s) onto pan(s)
  B                →  snap steak(s) back to board(s)

  Cooking advances only while the steak is on the pan and the key is held.
  Latches station["_expert_key_held"] directly (no arm).
  V                 toggle view: top-down ↔ head_camera
  Close the viewer window to quit.
"""

CONTROLS_ROBOT = """
  Hold Space       →  cook with selected arm(s)
  G                →  robot toggles steak(s): board → pan, then pan → board

  Cooking advances only while the steak is on the pan and the key is held.
  Selected arm(s) press the cook key while Space is held.
  --robot-motion planner|interpolate
  V                 toggle view: top-down ↔ head_camera
  Close the viewer window to quit.
"""


def _embodiment_config(robot_file):
    with open(Path(robot_file) / "config.yml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _configure_task(config_name: str, seed: int, use_robot: bool = False):
    config_path = REPO_ROOT / "task_config" / f"{config_name}.yml"
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    config.update(
        task_name="cook_meat",
        render_freq=1,
        now_ep_num=0,
        seed=seed,
        need_plan=use_robot,
        save_data=False,
    )

    with open(Path(CONFIGS_PATH) / "_embodiment_config.yml", "r", encoding="utf-8") as handle:
        embodiments = yaml.safe_load(handle)
    embodiment_names = config.get("embodiment", ["aloha-agilex"])
    if len(embodiment_names) == 1:
        left_name = right_name = embodiment_names[0]
        config["dual_arm_embodied"] = True
    elif len(embodiment_names) == 3:
        left_name, right_name, config["embodiment_dis"] = embodiment_names
        config["dual_arm_embodied"] = False
    else:
        raise SystemExit("Expected one embodiment or [left_embodiment, right_embodiment, separation].")

    config["left_robot_file"] = embodiments[left_name]["file_path"]
    config["right_robot_file"] = embodiments[right_name]["file_path"]
    config["left_embodiment_config"] = _embodiment_config(config["left_robot_file"])
    config["right_embodiment_config"] = _embodiment_config(config["right_robot_file"])
    return config


def _stations_by_arm(env):
    left = right = None
    for st in getattr(env, "stations", []) or []:
        if str(st["arm"]) == "left":
            left = st
        else:
            right = st
    return left, right


def _clear_cook_latches(env):
    for st in getattr(env, "stations", []) or []:
        st["_expert_key_held"] = False


def _active_stations(env, mode):
    """Stations whose arm matches ``mode`` (highlighted gripper).

    Never fall back to the other arm's station when the selected gripper has
    no matching station — that would move an unselected gripper.
    """
    left_st, right_st = _stations_by_arm(env)
    if mode == "all":
        return list(env.stations)
    if mode == "left" and left_st is not None:
        return [left_st]
    if mode == "right" and right_st is not None:
        return [right_st]
    return []


def _arms_for_mode(env, mode):
    stations = _active_stations(env, mode)
    return tuple(str(st["arm"]) for st in stations if st.get("cook_key") is not None)


def _station_for_side(env, side):
    """Station owned by ``side`` only — never the other arm's station."""
    left_st, right_st = _stations_by_arm(env)
    return left_st if side == "left" else right_st


def _snap_steaks_to_pans(env):
    for st in getattr(env, "stations", []) or []:
        pan = list(env._pan_place_target(st))
        pan[2] += 0.012
        q = st["steak"].get_pose().q
        st["steak"].set_pose(sapien.Pose(pan[:3], q))
        try:
            rigid = st["steak"].actor.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent)
            if rigid is not None:
                rigid.set_linear_velocity([0, 0, 0])
                rigid.set_angular_velocity([0, 0, 0])
        except Exception:
            pass
        if not env.use_cook_button:
            st["cooking_active"] = True
    if env.use_cook_button:
        print("Snapped steak(s) onto pan(s). Hold Space/Q/E to cook.")
    else:
        print("Snapped steak(s) onto pan(s); contact cooking is active.")


def _snap_steaks_to_boards(env):
    for st in getattr(env, "stations", []) or []:
        p = st["board"].get_pose().p
        z = float(st.get("board_top", p[2] + 0.02)) + 0.02
        q = st["steak"].get_pose().q
        st["steak"].set_pose(sapien.Pose([float(p[0]), float(p[1]), z], q))
        env._latch_grasp_doneness(st, force=True)
    print("Snapped steak(s) to board(s); doneness latched.")


def _steaks_on_pans(env):
    """Return whether every interactive station currently has its steak on its pan."""

    return bool(env.stations) and all(env._steak_on_pan_station(st) for st in env.stations)


def _stations_for_selected(env):
    """Stations owned by the currently highlighted gripper(s)."""
    selected = require_selected_arms(env, exactly_one=False)
    if not selected:
        return []
    selected_set = set(selected)
    stations = [
        st for st in (getattr(env, "stations", None) or [])
        if str(st["arm"]) in selected_set
    ]
    if not stations:
        action_failed(
            env, selected,
            detail="has no cook station / steak to transfer",
        )
    return stations


def _place_selected_steaks_on_pans(env, stations):
    """Place already-held steaks for ``stations`` only (no other-arm motion)."""
    for st in stations:
        arm = st["arm"]
        pan_target = env._pan_place_target(st)
        env.plan_success = True
        env.move(
            env.place_actor(
                st["steak"],
                target_pose=pan_target,
                arm_tag=arm,
                constrain="free",
                pre_dis=0.10,
                dis=0.02,
                is_open=True,
            )
        )
        if not env.plan_success:
            return False
        env.move(env.move_by_displacement(arm_tag=arm, z=0.10, move_axis="arm"))
        if not env.plan_success:
            return False
        if not env._steak_on_pan_station(st):
            env.plan_success = True
            env.move(
                env.place_actor(
                    st["steak"],
                    target_pose=pan_target,
                    arm_tag=arm,
                    constrain="free",
                    pre_dis=0.10,
                    dis=0.02,
                    is_open=True,
                )
            )
        if not env._steak_on_pan_station(st):
            env.plan_success = False
            return False
    return True


def _return_selected_steaks_to_boards(env, stations):
    """Return cooked steaks for ``stations`` only (no other-arm motion)."""
    for st in stations:
        arm = st["arm"]
        p = st["board"].get_pose().p
        board_target = [float(p[0]), float(p[1]), float(st["board_top"]) + 0.03]
        st["awaiting_return_grasp"] = True
        env.plan_success = True
        env.move(env.open_gripper(arm))
        if not env.plan_success:
            return False
        env.move(env._safe_grasp_actor(st["steak"], arm_tag=arm, pre_grasp_dis=0.1))
        if not env.plan_success:
            return False
        env._latch_grasp_doneness(st, force=True)
        env.move(env.move_by_displacement(arm_tag=arm, z=0.12, move_axis="arm"))
        if not env.plan_success:
            return False
        env.move(
            env.place_actor(
                st["steak"],
                target_pose=board_target,
                arm_tag=arm,
                constrain="free",
                pre_dis=0.10,
                dis=0.015,
                is_open=True,
            )
        )
        if not env.plan_success:
            return False
        env.move(env.move_by_displacement(arm_tag=arm, z=0.08))
    return True


def _toggle_steak_transfer(env, *, robot: bool):
    """Move steaks between boards and pans for the selected arm(s) only."""

    _clear_cook_latches(env)
    if not robot:
        if _steaks_on_pans(env):
            _snap_steaks_to_boards(env)
        else:
            _snap_steaks_to_pans(env)
        return

    stations = _stations_for_selected(env)
    if not stations:
        return
    arms = [str(st["arm"]) for st in stations]
    stations = sorted(stations, key=lambda st: str(st["arm"]))

    on_pans = all(env._steak_on_pan_station(st) for st in stations)
    if on_pans:
        env.plan_success = True
        try:
            if len(stations) == len(env.stations):
                env._return_steaks_to_boards()
                ok = bool(env.plan_success)
            else:
                ok = _return_selected_steaks_to_boards(env, stations)
        except Exception as exc:
            action_failed(env, arms, detail=f"could not return steak(s): {exc}")
            return
        if not ok or not env.plan_success:
            action_failed(env, arms, detail="could not return steak(s) to board(s)")
            return
        print(f"Robot returned steak(s) with {'+'.join(arms)} arm(s).")
        return

    env.plan_success = True
    env.move(*[env.open_gripper(st["arm"]) for st in stations])
    if not env.plan_success:
        action_failed(env, arms, detail="could not open gripper before steak grasp")
        return
    env.move(*[
        env._safe_grasp_actor(st["steak"], arm_tag=st["arm"], pre_grasp_dis=0.10)
        for st in stations
    ])
    if not env.plan_success:
        action_failed(
            env, arms,
            detail="could not grasp steak (out of reach or plan failed)",
        )
        return
    env.move(*[
        env.move_by_displacement(st["arm"], z=0.10, move_axis="arm")
        for st in stations
    ])
    if not env.plan_success:
        action_failed(env, arms, detail="could not lift steak after grasp")
        return
    try:
        if len(stations) == len(env.stations):
            env._place_steaks_on_pans()
            ok = bool(env.plan_success)
        else:
            ok = _place_selected_steaks_on_pans(env, stations)
    except Exception as exc:
        action_failed(env, arms, detail=f"could not place steak(s): {exc}")
        return
    if not ok or not env.plan_success:
        action_failed(env, arms, detail="could not place steak(s) on pan(s)")
        return
    if not env.use_cook_button:
        for st in stations:
            st["cooking_active"] = True
    print(f"Robot moved steak(s) to pan(s) with {'+'.join(arms)} arm(s).")


class KeyboardState:
    def __init__(self):
        self.prev_p = False
        self.prev_b = False
        self.prev_g = False

    def update(self, env, window):
        left_st, right_st = _stations_by_arm(env)
        space = window.key_down("space")
        q = window.key_down("q")
        e = window.key_down("e")
        _clear_cook_latches(env)
        if space:
            for st in env.stations:
                st["_expert_key_held"] = True
        else:
            if q and left_st is not None:
                left_st["_expert_key_held"] = True
            if e and right_st is not None:
                right_st["_expert_key_held"] = True
            # Single-station: Q/E/Space all map to the only cook key.
            if len(env.stations) == 1 and (q or e):
                env.stations[0]["_expert_key_held"] = True

        p = window.key_down("p")
        if p and not self.prev_p:
            _snap_steaks_to_pans(env)
        self.prev_p = p
        b = window.key_down("b")
        if b and not self.prev_b:
            _snap_steaks_to_boards(env)
        self.prev_b = b
        g = window.key_down("g")
        if g and not self.prev_g:
            _toggle_steak_transfer(env, robot=False)
        self.prev_g = g


def _requested_cook_mode(window):
    if window.key_down("space"):
        return "all"
    q = window.key_down("q")
    e = window.key_down("e")
    if q and not e:
        return "left"
    if e and not q:
        return "right"
    return None


def _selected_cook_mode(env, window):
    if not window.key_down("space"):
        env._interactive_cook_fail_latched = False
        return None
    selected = tuple(getattr(env, "_interactive_selected_arms", ()) or ())
    if not selected:
        return None
    if len(selected) == 2:
        return "all"
    mode = selected[0]
    # Selected gripper has no station → flash red once per Space hold.
    if not _active_stations(env, mode):
        if not getattr(env, "_interactive_cook_fail_latched", False):
            action_failed(
                env, selected,
                detail="has no cook station for this action",
            )
            env._interactive_cook_fail_latched = True
        return None
    env._interactive_cook_fail_latched = False
    return mode


def _station_cook_finished(env, st):
    """True once this station's steak has been grasped for its board return."""
    return st.get("grasp_doneness") is not None


def _episode_done(env):
    """Finish after all steaks return, or immediately on definite overcooking."""
    stations = getattr(env, "stations", None) or []
    if not stations:
        return False, None
    doneness = [round(float(st["doneness"]), 2) for st in stations]
    grasps = [
        None if st.get("grasp_doneness") is None else round(float(st["grasp_doneness"]), 2)
        for st in stations
    ]
    detail = (
        f"doneness={doneness} grasp={grasps} "
        f"target={float(env.target_doneness_range[0]):.2f}-"
        f"{float(env.target_doneness_range[1]):.2f}"
    )
    target_max = float(env.target_doneness_range[1])
    if any(float(st.get("doneness", 0.0)) > target_max for st in stations):
        return True, f"overcooked; {detail}"
    if all(_station_cook_finished(env, st) for st in stations):
        return True, detail
    return False, None


class CookKeyController:
    """Responsive fixed-pose press with planner-validated cached targets."""

    TRANSITION_SECONDS = 0.12
    RELEASE_CLEARANCE = 0.04

    def __init__(self, env, arm_tag):
        self.env = env
        self.arm_tag = arm_tag
        self.mode = None
        self.hover_qpos = {}
        self.press_qpos = {}
        self._starts = {}
        self._targets = {}
        self._started_at = None
        self.prepare()

    def _drive_qpos(self, side):
        joints = (
            self.env.robot.left_arm_joints
            if side == "left"
            else self.env.robot.right_arm_joints
        )
        return np.asarray([joint.get_drive_target()[0] for joint in joints], dtype=np.float64)

    def _move_for_stations(self, stations, action_fn):
        actions = [action_fn(st) for st in stations]
        if not actions:
            return
        self.env.plan_success = True
        self.env._last_plan_fail = None
        self.env.move(*actions)

    def prepare(self):
        """Move to hover once and cache a validated press target for each arm."""

        _clear_cook_latches(self.env)
        self.mode = None
        self._started_at = None
        stations = [st for st in self.env.stations if st.get("cook_key") is not None]
        configured_hover = float(
            getattr(self.env, "key_hover_dis", self.env.KEY_HOVER_DIS_DEFAULT)
        )
        depth = float(getattr(self.env, "key_press_depth", self.env.KEY_PRESS_DEPTH_DEFAULT))
        # cook_meat detects presses from EE height, while key_hover_dis is a TCP
        # clearance. The configured 6 cm TCP hover leaves the EE just inside the
        # default 20 cm active band, so explicitly clear that band on release.
        hover = max(
            configured_hover,
            float(self.env.key_press_dz) - float(self.env.EE_TO_TCP) + self.RELEASE_CLEARANCE,
        )
        press_above = max(0.0, configured_hover - depth)
        self._move_for_stations(
            stations, lambda st: self.env.close_gripper(self.arm_tag(str(st["arm"])))
        )
        if not self.env.plan_success:
            return
        self._move_for_stations(
            stations,
            lambda st: self.env.move_to_pose(
                self.arm_tag(str(st["arm"])), self.env._cook_key_tip_pose(st, hover)
            ),
        )
        if not self.env.plan_success:
            return

        self.hover_qpos.clear()
        self.press_qpos.clear()
        for st in stations:
            side = str(st["arm"])
            hover_q = self._drive_qpos(side)
            press_pose = self.env._cook_key_tip_pose(st, press_above)
            planner = (
                self.env.robot.left_plan_path
                if side == "left"
                else self.env.robot.right_plan_path
            )
            result = planner(press_pose, last_qpos=np.asarray(hover_q, dtype=np.float32))
            if result is None or result.get("status") != "Success":
                reason = "no result" if result is None else result.get("reason", "unknown")
                raise RuntimeError(f"Could not prepare {side} cook-key press: {reason}")
            self.hover_qpos[side] = hover_q
            self.press_qpos[side] = np.asarray(result["position"][-1], dtype=np.float64)
        print(
            f"Cook-key arms ready {hover * 100:.1f} cm above key; "
            "press/release transitions are non-blocking."
        )

    def _begin_transition(self, requested_mode):
        _clear_cook_latches(self.env)
        active = {str(st["arm"]) for st in _active_stations(self.env, requested_mode)}
        previous = {str(st["arm"]) for st in _active_stations(self.env, self.mode)}
        moving = (active | previous) & self.hover_qpos.keys()
        self._starts = {side: self._drive_qpos(side) for side in moving}
        self._targets = {
            side: self.press_qpos[side] if side in active else self.hover_qpos[side]
            for side in moving
        }
        self._started_at = time.perf_counter() if moving else None
        self.mode = requested_mode

    def _advance(self):
        if self._started_at is None:
            return
        progress = min(
            1.0,
            (time.perf_counter() - self._started_at) / self.TRANSITION_SECONDS,
        )
        smooth = progress * progress * (3.0 - 2.0 * progress)
        for side, target in self._targets.items():
            start = self._starts[side]
            position = start + (target - start) * smooth
            velocity = (
                (target - start) / self.TRANSITION_SECONDS
                if progress < 1.0
                else np.zeros_like(target)
            )
            self.env.robot.set_arm_joints(position, velocity, side)
        if progress >= 1.0:
            self._started_at = None

    def update(self, requested_mode):
        if requested_mode is not None:
            active = {
                str(st["arm"])
                for st in _active_stations(self.env, requested_mode)
            } & self.hover_qpos.keys()
            if not active:
                # Highlighted arm(s) cannot cook (no station / not prepared).
                if requested_mode != self.mode and not getattr(
                    self.env, "_interactive_cook_fail_latched", False
                ):
                    selected = tuple(
                        getattr(self.env, "_interactive_selected_arms", ()) or ()
                    )
                    action_failed(
                        self.env, selected or (requested_mode,),
                        detail="cannot cook with this gripper",
                    )
                    self.env._interactive_cook_fail_latched = True
                _clear_cook_latches(self.env)
                return
        if requested_mode != self.mode:
            # A release immediately reverses any incomplete downward transition.
            self._begin_transition(requested_mode)
        self._advance()
        if requested_mode is not None and self._started_at is None:
            for st in _active_stations(self.env, requested_mode):
                st["_expert_key_held"] = True
        else:
            _clear_cook_latches(self.env)

    def release(self):
        _clear_cook_latches(self.env)
        self.mode = None
        self._started_at = None


def _make_robot_controller(env, arm_tag, robot_motion):
    def get_button(e, side):
        st = _station_for_side(e, side)
        if st is not None and st.get("cook_key") is not None:
            return st["cook_key"]
        return e.cook_key

    def get_top_z(e, side):
        st = _station_for_side(e, side)
        if st is not None and st.get("key_top_z") is not None:
            return float(st["key_top_z"])
        return float(e._key_top_z)

    def set_latch(e, mode):
        _clear_cook_latches(e)
        for st in _active_stations(e, mode):
            st["_expert_key_held"] = True

    sides = tuple(
        str(st["arm"])
        for st in env.stations
        if st.get("cook_key") is not None
    ) or ("left", "right")

    if robot_motion == "planner":
        return CookKeyController(env, arm_tag)

    return make_button_controller(
        env,
        arm_tag,
        robot_motion,
        get_button=get_button,
        get_top_z=get_top_z,
        set_latch=set_latch,
        clear_latch=_clear_cook_latches,
        arms_for_mode=lambda m: _arms_for_mode(env, m),
        hold=True,
        active_dz=float(getattr(env, "key_press_dz", 0.20)),
        sides=sides,
    )


def main():
    parser = argparse.ArgumentParser(description="Interactive cook_meat viewer")
    parser.add_argument("--config", default="demo_dynamic", help="Task config name without .yml")
    parser.add_argument("--seed", type=int, default=0, help="Scene randomization seed")
    add_robot_motion_arg(parser)
    args = parser.parse_args()

    from envs import CONFIGS_PATH
    from envs.cook_meat import cook_meat
    from envs.utils.action import ArmTag
    globals()["CONFIGS_PATH"] = CONFIGS_PATH

    print_mode_controls("cook_meat", args.control, keyboard=CONTROLS_KEYBOARD, robot=CONTROLS_ROBOT)

    env = cook_meat()
    env.setup_demo(**_configure_task(args.config, args.seed, use_robot=args.control == "robot"))
    # Match the main cook_meat rollout: open fingers before approaching steak.
    env.together_open_gripper(save_freq=None)
    _clear_cook_latches(env)
    if not env.use_cook_button:
        print("cook_button_enabled=false: no cook button; meat cooks by pan contact.")

    # Keyboard sandbox starts with steaks on pans so Space can cook immediately.
    if args.control == "keyboard":
        _snap_steaks_to_pans(env)

    keyboard = KeyboardState()
    robot_controller = (
        _make_robot_controller(env, ArmTag, args.robot_motion)
        if args.control == "robot" and env.use_cook_button
        else None
    )

    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    views = make_viewer_view_toggle(env, viewer)

    n = len(env.stations)
    print(
        f"Cook-button sandbox ready ({n} station(s)). "
        f"Control={args.control}. robot-motion={args.robot_motion}."
    )

    last_status = None
    try:
        while not viewer.closed:
            views.update(viewer.window)
            frame_start = time.perf_counter()
            if args.control == "keyboard":
                keyboard.update(env, viewer.window)
            else:
                if robot_controller is not None:
                    robot_controller.update(_selected_cook_mode(env, viewer.window))
                if viewer.window.key_press("g"):
                    if robot_controller is not None:
                        robot_controller.release()
                    _toggle_steak_transfer(env, robot=True)
                    if robot_controller is not None and hasattr(robot_controller, "prepare"):
                        robot_controller.prepare()
            env._update_kinematic_tasks()
            env.scene.step()
            env.scene.update_render()
            viewer.render()
            if viewer.window.key_down("escape"):
                break
            doneness = [round(float(st["doneness"]), 2) for st in env.stations]
            target_range = env.target_doneness_range
            status = (
                f"doneness={doneness} "
                f"target={float(target_range[0]):.2f}-{float(target_range[1]):.2f}"
            )
            if status != last_status and any(d > 0 for d in doneness):
                print(status)
                last_status = status
            done, detail = _episode_done(env)
            if done:
                report_task_result(env, detail)
                break
            remaining = float(env.scene.get_timestep()) - (time.perf_counter() - frame_start)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        try:
            if robot_controller is not None:
                robot_controller.release()
        finally:
            env.close_env()


if __name__ == "__main__":
    main()
