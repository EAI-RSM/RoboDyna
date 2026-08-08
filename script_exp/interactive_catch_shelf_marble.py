#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive viewer for ``catch_shelf_marble``.

Run from any directory:

    /path/to/RoboDynaExp/script_exp/interactive_catch_shelf_marble.py --control keyboard
    /path/to/RoboDynaExp/script_exp/interactive_catch_shelf_marble.py --control robot

Keyboard mode latches bowl keys directly via arrows. Robot mode: select an arm,
move over the bowl key, lower with Q to press (gripper-Z / ReactivePushButtons). Sandbox only — not data collection.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "script" / "bench_script"))
sys.path.insert(0, str(REPO_ROOT / "script_exp"))

from _interactive_common import (  # noqa: E402
    print_instructions,
    UniversalRobotControls,
    make_viewer_view_toggle,
    add_robot_motion_arg,
    report_task_result,
    sleep_to_timestep,
    terminal_hold_should_close,
    print_mode_controls,
    print_episode_condition,
)


CONTROLS_KEYBOARD = """
  Left Arrow        move bowl left
  Right Arrow       move bowl right
"""

CONTROLS_ROBOT = """
  Select an arm, move over a key, lower with Q to press (E to raise).
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
        task_name="catch_shelf_marble",
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


def _requested_side(window):
    # Arrows only — letter-key aliases also orbit the SAPIEN viewer camera.
    left = window.key_down("left")
    right = window.key_down("right")
    if left and not right:
        return "left"
    if right and not left:
        return "right"
    return None


def _update_keyboard(env, window):
    side = _requested_side(window)
    env._expert_hold = side
    env._bowl_force_stop = False
    if side is not None and getattr(env, "_marble_state", None) == "parked":
        env._release_marble()


def main():
    parser = argparse.ArgumentParser(description="Interactive catch_shelf_marble viewer")
    parser.add_argument("--config", default="demo_dynamic", help="Task config name without .yml")
    parser.add_argument("--seed", type=int, default=0, help="Scene randomization seed")
    add_robot_motion_arg(parser, robot_motion_default="interpolate")
    args = parser.parse_args()

    from envs import CONFIGS_PATH
    from envs.catch_shelf_marble import catch_shelf_marble
    globals()["CONFIGS_PATH"] = CONFIGS_PATH

    print_mode_controls("catch_shelf_marble", args.control, keyboard=CONTROLS_KEYBOARD, robot=CONTROLS_ROBOT)

    env = catch_shelf_marble()
    # Always enable arm teleop: presses are gripper-Z only (no Space).
    env._interactive_robot_mode = True
    # Raster viewer: pour_beer-style plain-alpha shelves (transmission is invisible here).
    env._plain_glass = True
    env.setup_demo(**_configure_task(args.config, args.seed, use_robot=True))
    env.together_close_gripper(save_freq=None)
    print_episode_condition(env)
    env._osc_armed = True
    env._bowl_force_stop = False
    env._expert_hold = None

    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    views = make_viewer_view_toggle(env, viewer)
    if views.robot_controls is None:
        views.robot_controls = UniversalRobotControls(env)

    if args.control == "keyboard":
        print_instructions("Keyboard arrows still latch bowl motion as a sandbox shortcut.")

    terminal_started_at = None

    try:
        while not viewer.closed:
            views.update(viewer.window)
            frame_start = time.perf_counter()
            if args.control == "keyboard":
                _update_keyboard(env, viewer.window)
            env._update_kinematic_tasks()
            env.scene.step()
            env.scene.update_render()
            viewer.render()
            if viewer.window.key_down("escape"):
                break

            if terminal_started_at is not None:
                if terminal_hold_should_close(terminal_started_at):
                    break
                sleep_to_timestep(env, frame_start)
                continue
            if getattr(env, "_marble_state", None) == "landed":
                if getattr(env, "_marble_result", None) is None:
                    env._resolve_marble()
                report_task_result(env, env._marble_result)
                terminal_started_at = time.perf_counter()
            remaining = float(env.scene.get_timestep()) - (time.perf_counter() - frame_start)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        env.close_env()


if __name__ == "__main__":
    main()
    # household_task_gui convention: 0=SUCCESS, 10=FAILURE, 2=no result
    from _interactive_common import task_result_exit_code
    raise SystemExit(task_result_exit_code())
