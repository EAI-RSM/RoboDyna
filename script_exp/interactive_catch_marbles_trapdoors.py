#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive viewer for ``catch_marbles_trapdoors``.

Run from any directory:

    /path/to/RoboDynaExp/script_exp/interactive_catch_marbles_trapdoors.py --control robot
    /path/to/RoboDynaExp/script_exp/interactive_catch_marbles_trapdoors.py --control robot --seed 3

Select an arm, move over the matching colored key, then lower with Q to press.
The keycap depresses and springs back like ``fill_coffee_jar``; the trapdoor
opens briefly on the press edge (then auto-closes). Holding does not keep it
open — release the key fully and press again to reopen.
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
    RealtimePhysicsPacer,
    begin_interactive_frame,
    UniversalRobotControls,
    add_robot_motion_arg,
    make_viewer_view_toggle,
    print_instructions,
    print_mode_controls,
    report_task_result,
    terminal_hold_should_close,
    print_episode_condition,
)


CONTROLS_KEYBOARD = """
  Prefer --control robot: select an arm, move over the matching colored key, lower with Q to press.
  The keycap springs back when you raise the gripper.
"""

CONTROLS_ROBOT = """
  Select left (1) or right (2) arm, move over the matching colored key, then lower with Q to press (E to raise).
  Left arm covers left-half keys; right arm covers right-half keys.
  Door opens when the keycap is pushed down past its trigger depth.
  Release fully and press again to reopen a door.
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
        task_name="catch_marbles_trapdoors",
        render_freq=1,
        now_ep_num=0,
        seed=seed,
        need_plan=use_robot,
        save_data=False,
    )
    # Interactive sandbox: swing doors open quickly (~0.1 s) so they don't lag
    # behind the arm press. Demo collection still uses demo_dynamic.yml as-is.
    task_args = config.setdefault("task_args", {}).setdefault("catch_marbles_trapdoors", {})
    task_args["door_open_speed_deg"] = max(float(task_args.get("door_open_speed_deg", 220.0)), 1200.0)

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


def _print_color_map(env):
    names = list(getattr(env, "button_color_names", []) or [])
    target = int(getattr(env, "target_button_idx", -1))
    mapping = ", ".join(f"{i}:{c}" for i, c in enumerate(names))
    target_name = names[target] if 0 <= target < len(names) else "?"
    left_keys = [f"{i}:{c}" for i, c in enumerate(names) if str(env._arm_for_door(i)) == "left"]
    right_keys = [f"{i}:{c}" for i, c in enumerate(names) if str(env._arm_for_door(i)) == "right"]
    print(f"Buttons L→R: {mapping}")
    print(f"Left-arm keys: {', '.join(left_keys) or '(none)'} | Right-arm keys: {', '.join(right_keys) or '(none)'}")
    print(f"Target marble color: {target_name} (index {target})")


def main():
    parser = argparse.ArgumentParser(description="Interactive catch_marbles_trapdoors viewer")
    parser.add_argument("--config", default="demo_dynamic", help="Task config name without .yml")
    parser.add_argument("--seed", type=int, default=0, help="Scene randomization seed")
    add_robot_motion_arg(parser)
    args = parser.parse_args()

    from envs import CONFIGS_PATH
    from envs.catch_marbles_trapdoors import catch_marbles_trapdoors
    globals()["CONFIGS_PATH"] = CONFIGS_PATH

    print_mode_controls(
        "catch_marbles_trapdoors",
        args.control,
        keyboard=CONTROLS_KEYBOARD,
        robot=CONTROLS_ROBOT,
    )

    env = catch_marbles_trapdoors()
    # Always enable arm teleop: presses are gripper-Z only (no Space).
    env._interactive_robot_mode = True
    env.setup_demo(**_configure_task(args.config, args.seed, use_robot=True))
    env.together_close_gripper(save_freq=None)
    print_episode_condition(env)
    _print_color_map(env)

    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    views = make_viewer_view_toggle(env, viewer)
    if views.robot_controls is None:
        views.robot_controls = UniversalRobotControls(env)


    left_track_since = None
    settle_s = 0.6
    terminal_started_at = None
    pacer = RealtimePhysicsPacer(env)

    try:
        while not viewer.closed:
            n_steps = begin_interactive_frame(views, pacer, viewer.window)
            if n_steps == 0:
                env.scene.update_render()
                viewer.render()
                if viewer.window.key_down("escape"):
                    break
                if terminal_started_at is not None and terminal_hold_should_close(terminal_started_at):
                    break
                continue

            for _ in range(n_steps):
                env._update_kinematic_tasks()
                env.scene.step()
            env.scene.update_render()
            viewer.render()
            if viewer.window.key_down("escape"):
                break

            if terminal_started_at is not None:
                if terminal_hold_should_close(terminal_started_at):
                    break
                continue
            mode = str(getattr(env, "_ball_mode", "track"))
            if mode != "track":
                if left_track_since is None:
                    left_track_since = time.perf_counter()
                elif time.perf_counter() - left_track_since >= settle_s:
                    report_task_result(env, f"ball_mode={mode}")
                    terminal_started_at = time.perf_counter()
            else:
                left_track_since = None
    finally:
        env.close_env()


if __name__ == "__main__":
    main()
    # household_task_gui convention: 0=SUCCESS, 10=FAILURE, 2=no result
    from _interactive_common import task_result_exit_code
    raise SystemExit(task_result_exit_code())
