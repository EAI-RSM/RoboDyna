#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive viewer for ``cook_meat`` (Opt1 cook-button sandbox).

Run from any directory:

    /path/to/RoboDynaExp/script_exp/interactive_cook_meat.py --control keyboard
    /path/to/RoboDynaExp/script_exp/interactive_cook_meat.py --control robot
    /path/to/RoboDynaExp/script_exp/interactive_cook_meat.py --control robot --robot-motion interpolate

Forces ``cook_button_enabled=true``. Keyboard latches ``station["_expert_key_held"]``.
Robot mode presses the cook key with the matching arm. Sandbox only.
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
    make_viewer_view_toggle,
    add_robot_motion_arg,
    make_button_controller,
    report_task_result,
)


CONTROLS = """
============================================================
  cook_meat — interactive controls (cook button Opt1)
============================================================
  Hold Space       →  cook (all stations / primary)
  Hold Q           →  cook LEFT station only (dual)
  Hold E           →  cook RIGHT station only (dual)
  P                →  snap steak(s) onto pan(s) (keyboard helper)
  B                →  snap steak(s) back to board(s) (keyboard helper)

  Cooking advances only while the steak is on the pan and the key is held.
  --control keyboard : latch station["_expert_key_held"]
  --control robot    : arm presses the cook key (hold Space / Q / E)
  --robot-motion planner|interpolate  (robot key-press routine)
  V                 toggle view: top-down ↔ head_camera
  Close the viewer window to quit.
============================================================
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

    task_args = config.setdefault("task_args", {}).setdefault("cook_meat", {})
    task_args["cook_button_enabled"] = True

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
    left_st, right_st = _stations_by_arm(env)
    if mode == "all":
        return list(env.stations)
    if mode == "left" and left_st is not None:
        return [left_st]
    if mode == "right" and right_st is not None:
        return [right_st]
    if mode in ("left", "right") and len(env.stations) == 1:
        return [env.stations[0]]
    return []


def _arms_for_mode(env, mode):
    stations = _active_stations(env, mode)
    return tuple(str(st["arm"]) for st in stations if st.get("cook_key") is not None)


def _station_for_side(env, side):
    left_st, right_st = _stations_by_arm(env)
    st = left_st if side == "left" else right_st
    if st is None and len(getattr(env, "stations", []) or []) == 1:
        st = env.stations[0]
    return st


def _snap_steaks_to_pans(env):
    for st in getattr(env, "stations", []) or []:
        pan = list(st["skillet"].get_functional_point(0))
        pan[0] += float(getattr(env, "place_dx", 0.0))
        pan[1] += float(getattr(env, "place_dy", 0.0))
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
    print("Snapped steak(s) onto pan(s). Hold Space/Q/E to cook.")


def _snap_steaks_to_boards(env):
    for st in getattr(env, "stations", []) or []:
        p = st["board"].get_pose().p
        z = float(st.get("board_top", p[2] + 0.02)) + 0.02
        q = st["steak"].get_pose().q
        st["steak"].set_pose(sapien.Pose([float(p[0]), float(p[1]), z], q))
        env._latch_grasp_doneness(st, force=True)
    print("Snapped steak(s) to board(s); doneness latched.")


class KeyboardState:
    def __init__(self):
        self.prev_p = False
        self.prev_b = False

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


def _station_cook_finished(env, st):
    """True when this station's cook window is definitively over."""
    tol = float(getattr(env, "cook_doneness_tol", 0.08))
    target = float(env.target_doneness)
    doneness = float(st.get("doneness", 0.0))
    grasp = st.get("grasp_doneness")
    # Overcooked past tolerance (or fully maxed) → episode can end.
    if doneness > target + tol or doneness >= 0.999:
        return True
    # Cooking stopped / latched (B snap or grasp); prefer steak off pan if detectable.
    if grasp is not None:
        try:
            on_pan = bool(env._steak_on_pan_station(st))
        except Exception:
            on_pan = None
        if on_pan is False or on_pan is None:
            return True
        # Still on pan after latch — cooking is frozen; count as finished.
        return True
    # Reached target and cook key released (window closed without board return).
    if doneness >= target and not bool(st.get("_expert_key_held")):
        return True
    return False


def _episode_done(env):
    """Return ``(done, detail)`` once every station has finished cooking."""
    stations = getattr(env, "stations", None) or []
    if not stations:
        return False, None
    if not all(_station_cook_finished(env, st) for st in stations):
        return False, None
    doneness = [round(float(st["doneness"]), 2) for st in stations]
    grasps = [
        None if st.get("grasp_doneness") is None else round(float(st["grasp_doneness"]), 2)
        for st in stations
    ]
    detail = (
        f"doneness={doneness} grasp={grasps} "
        f"target={float(env.target_doneness):.2f}±{float(getattr(env, 'cook_doneness_tol', 0.08)):.2f}"
    )
    return True, detail


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

    print(CONTROLS)

    env = cook_meat()
    env.setup_demo(**_configure_task(args.config, args.seed, use_robot=args.control == "robot"))
    env.together_close_gripper(save_freq=None)
    _clear_cook_latches(env)
    if not env.use_cook_button:
        raise SystemExit("cook_button_enabled did not activate; check task_args.")

    # Keyboard sandbox starts with steaks on pans so Space can cook immediately.
    if args.control == "keyboard":
        _snap_steaks_to_pans(env)

    keyboard = KeyboardState()
    robot_controller = (
        _make_robot_controller(env, ArmTag, args.robot_motion)
        if args.control == "robot"
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
            elif robot_controller is not None:
                robot_controller.update(_requested_cook_mode(viewer.window))
            env._update_kinematic_tasks()
            env.scene.step()
            env.scene.update_render()
            viewer.render()
            doneness = [round(float(st["doneness"]), 2) for st in env.stations]
            status = f"doneness={doneness} target={env.target_doneness:.2f}"
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
