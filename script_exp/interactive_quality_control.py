#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive viewer for ``quality_control``.

Run from any directory:

    /path/to/RoboDynaExp/script_exp/interactive_quality_control.py --control keyboard
    /path/to/RoboDynaExp/script_exp/interactive_quality_control.py --control robot
    /path/to/RoboDynaExp/script_exp/interactive_quality_control.py --control robot --robot-motion interpolate

Keyboard mode calls ``_press_key`` directly. Robot mode makes the matching arm
tap ``keys[red/green]``. Skip black tiles (do not press).
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
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
 quality_control — interactive controls
============================================================
  R or Left Arrow   stamp RED   (left key)
  G or Right Arrow  stamp GREEN (right key)
  Skip BLACK tiles — do not press while they are under the stamp
  V                 toggle view: top-down ↔ head_camera
  Q / Escape         quit
------------------------------------------------------------
  --control keyboard  direct stamp via _press_key (default)
  --control robot     arm taps the matching colored key
  --robot-motion planner|interpolate  (robot mode; interpolate is faster test)
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

    config.update(
        task_name="quality_control",
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


def _requested_color(window):
    if window.key_down("r") or window.key_down("left"):
        return "red"
    if window.key_down("g") or window.key_down("right"):
        return "green"
    return None


class EdgePress:
    """Fire once per key-hold edge."""

    def __init__(self):
        self._prev = None

    def poll(self, color):
        edge = color is not None and color != self._prev
        self._prev = color
        return color if edge else None


def _start_belt(env):
    env._belt_running = True
    env._stamp_active = True


def _finalize_departed_tiles(env, last_under):
    """When a tile leaves the stamp without a mark, record skip/miss for scoring."""

    current = env._tile_under_stamp(require_unhandled=True)
    if last_under is None or last_under == current:
        return current
    i = last_under
    if env.tile_hidden[i] or env.tile_marked[i] or env.tile_skipped[i] or env.tile_missed[i]:
        return current
    if env.tile_colors[i] == "black":
        env.tile_skipped[i] = True
        print(f"Black tile {i} skipped (no press).")
    else:
        env._mark_missed_tile(i)
        print(f"Tile {i} ({env.tile_colors[i]}) missed — not stamped in time.")
    return current


def _episode_done(env):
    for i, color in enumerate(env.tile_colors):
        if env.tile_hidden[i]:
            continue
        if color == "black":
            if not (env.tile_skipped[i] or env.tile_marked[i]):
                return False
        else:
            if not (env.tile_marked[i] or env.tile_missed[i]):
                return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Interactive quality_control viewer")
    parser.add_argument("--config", default="demo_dynamic", help="Task config name without .yml")
    parser.add_argument("--seed", type=int, default=0, help="Scene randomization seed")
    add_robot_motion_arg(parser)
    args = parser.parse_args()

    from envs import CONFIGS_PATH
    from envs.quality_control import quality_control
    from envs.utils.action import ArmTag
    globals()["CONFIGS_PATH"] = CONFIGS_PATH

    print(CONTROLS)

    env = quality_control()
    env.setup_demo(**_configure_task(args.config, args.seed, use_robot=args.control == "robot"))
    env.together_close_gripper(save_freq=None)
    _start_belt(env)

    robot_controller = None
    if args.control == "robot":
        def arms_for_mode(m):
            if m == "red":
                return ("left",)
            if m == "green":
                return ("right",)
            return ()

        def get_button(e, side):
            return e.keys["red" if side == "left" else "green"]

        def get_top_z(e, side):
            color = "red" if side == "left" else "green"
            return float(e.keys[color].get_pose().p[2]) + float(e.KEY_HALF[2])

        robot_controller = make_button_controller(
            env, ArmTag, args.robot_motion,
            get_button=get_button,
            get_top_z=get_top_z,
            arms_for_mode=arms_for_mode,
            on_press=lambda e, m: e._press_key(m),
            hold=False,
            sides=("left", "right"),
        )

    edge = EdgePress()
    last_under = None

    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    views = make_viewer_view_toggle(env, viewer)

    if args.control == "robot":
        print(
            f"Robot mode ready (motion={args.robot_motion}). "
            "Tap R/Left or G/Right to press the matching key."
        )
    else:
        print("Keyboard mode ready. Tap R/Left or G/Right when the matching tile is under the stamp.")
    print(f"Tile colors: {env.tile_colors}")

    try:
        while not viewer.closed:
            views.update(viewer.window)
            frame_start = time.perf_counter()
            requested = _requested_color(viewer.window)
            color = edge.poll(requested)
            if args.control == "keyboard":
                if color is not None:
                    env._press_key(color)
                    print(f"Stamp requested: {color}")
            elif robot_controller is not None and color is not None:
                robot_controller.update(color)

            last_under = _finalize_departed_tiles(env, last_under)
            under = env._tile_under_stamp(require_unhandled=True)
            if under is not None:
                last_under = under

            env._update_kinematic_tasks()
            env.scene.step()
            env.scene.update_render()
            viewer.render()

            if viewer.window.key_down("q") or viewer.window.key_down("escape"):
                break

            if _episode_done(env):
                report_task_result(
                    env,
                    f"correct={sum(1 for c in env.tile_correct if c)}, "
                    f"missed={sum(1 for m in env.tile_missed if m)}, "
                    f"black_press={env.black_press}",
                )
                time.sleep(1.5)
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
