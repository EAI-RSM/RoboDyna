#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive viewer for ``marble_shelf_maze``.

Run from any directory:

    /path/to/RoboDynaExp/interactive/base/interactive_marble_shelf_maze.py --control keyboard
    /path/to/RoboDynaExp/interactive/base/interactive_marble_shelf_maze.py --control robot

Robot mode: select an arm, move over a shelf key, lower with Q to press
(``ReactivePushButtons`` spring keycap). Hold keeps the active shelf tilting;
release springs the key up and the shelf returns to flat.

Keyboard mode: hold Left/Right arrows to latch the matching key (sandbox
shortcut — same hold-to-tilt / release-to-return behavior). Sandbox only.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "script" / "bench_script"))
sys.path.insert(0, str(REPO_ROOT / "interactive"))

from _interactive_common import (  # noqa: E402
    print_instructions,
    UniversalRobotControls,
    make_viewer_view_toggle,
    add_robot_motion_arg,
    report_task_result,
    RealtimePhysicsPacer,
    terminal_hold_should_close,
    print_mode_controls,
    print_episode_condition,
)


CONTROLS_KEYBOARD = """
  Left Arrow        hold left key (tilt active shelf left while held)
  Right Arrow       hold right key (tilt active shelf right while held)

  Release to spring the key up and return the shelf flat.
  If the marble already landed below, release also switches control to that next shelf.
"""

CONTROLS_ROBOT = """
  Select left (1) or right (2) arm, move over the matching shelf button, then lower with Q to press (E to raise).
  Hold the key to tilt the active shelf (slower sweep); release to let it return flat.
  After the marble lands on the next shelf, keep holding to leave the previous shelf active; release to switch control to the next level.
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
        task_name="marble_shelf_maze",
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
    left = window.key_down("left")
    right = window.key_down("right")
    if left and not right:
        return "left"
    if right and not left:
        return "right"
    return None


def _update_keyboard(env, window):
    """Latch arrow keys onto the spring buttons (same hold-to-tilt path as Q)."""
    env._expert_hold = _requested_side(window)


def main():
    parser = argparse.ArgumentParser(description="Interactive marble_shelf_maze viewer")
    parser.add_argument("--config", default="demo_dynamic", help="Task config name without .yml")
    parser.add_argument("--seed", type=int, default=0, help="Scene randomization seed")
    add_robot_motion_arg(parser)
    args = parser.parse_args()

    from envs import CONFIGS_PATH
    from envs.marble_shelf_maze import marble_shelf_maze
    globals()["CONFIGS_PATH"] = CONFIGS_PATH

    print_mode_controls("marble_shelf_maze", args.control, keyboard=CONTROLS_KEYBOARD, robot=CONTROLS_ROBOT)

    env = marble_shelf_maze()
    # Always enable arm teleop: button presses are gripper-Z; Space opens/closes grippers.
    env._interactive_robot_mode = True
    # Raster viewer: pour_beer-style plain-alpha shelves/panes (transmission is invisible here).
    env._plain_glass = True
    env.setup_demo(**_configure_task(args.config, args.seed, use_robot=True))
    env.together_close_gripper(save_freq=None)
    print_episode_condition(env)
    env._bowl_armed = bool(getattr(env, "osc_bowl_enabled", False))
    env.plan_success = True
    env._expert_hold = None

    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    views = make_viewer_view_toggle(env, viewer)
    if views.robot_controls is None:
        views.robot_controls = UniversalRobotControls(env)

    dirs = list(getattr(env, "correct_dir", []) or [])
    print(f"Shelves={env.n_shelves}. Suggested directions top→bottom: {dirs}")
    if args.control == "keyboard":
        print_instructions("Keyboard arrows latch the spring keys (hold-to-tilt) as a sandbox shortcut.")

    terminal_started_at = None
    pacer = RealtimePhysicsPacer(env)

    try:
        while not viewer.closed:
            n_steps = pacer.begin_frame()
            views.update(viewer.window)
            if args.control == "keyboard":
                _update_keyboard(env, viewer.window)
            else:
                env._expert_hold = None

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
            mode = str(getattr(env, "_ball_mode", ""))
            if mode in ("done", "missed") and int(getattr(env, "active_shelf_idx", 0)) < 0:
                report_task_result(env, f"ball_mode={mode}")
                terminal_started_at = time.perf_counter()
    finally:
        env.close_env()


if __name__ == "__main__":
    main()
    # household_task_gui convention: 0=SUCCESS, 10=FAILURE, 2=no result
    from _interactive_common import task_result_exit_code
    raise SystemExit(task_result_exit_code())
