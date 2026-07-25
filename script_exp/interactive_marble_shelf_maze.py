#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive viewer for ``marble_shelf_maze``.

Run from any directory:

    /path/to/RoboDynaExp/script_exp/interactive_marble_shelf_maze.py --control keyboard
    /path/to/RoboDynaExp/script_exp/interactive_marble_shelf_maze.py --control robot
    /path/to/RoboDynaExp/script_exp/interactive_marble_shelf_maze.py --control robot --robot-motion interpolate

Keyboard mode tilts the active shelf via ``_press_tilt_direct`` (no arm).
Robot mode taps the matching side button then tilts. Sandbox only.
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
    print_mode_controls,
)


CONTROLS_KEYBOARD = """
  Left Arrow / A   →  tilt active shelf LEFT  (red / left button)
  Right Arrow / D  →  tilt active shelf RIGHT (right button)

  One press tilts the current shelf, rolls the marble off, and
  waits for the fall/settle before accepting another press.
  Hint: correct_dir for each shelf is printed at startup.

  Keys call the tilt API directly (no arm motion).
  V                 toggle view: top-down ↔ head_camera
  Close the viewer window to quit.
"""

CONTROLS_ROBOT = """
  Left Arrow / A   →  tilt active shelf LEFT  (red / left button)
  Right Arrow / D  →  tilt active shelf RIGHT (right button)

  One press tilts the current shelf, rolls the marble off, and
  waits for the fall/settle before accepting another press.
  Hint: correct_dir for each shelf is printed at startup.

  Matching arm taps the side button, then tilts.
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


def _requested_direction(window):
    left = window.key_down("left") or window.key_down("a")
    right = window.key_down("right") or window.key_down("d")
    if left and not right:
        return "left"
    if right and not left:
        return "right"
    return None


class EdgeDirection:
    def __init__(self):
        self.prev = None

    def edge(self, window):
        cur = _requested_direction(window)
        fired = cur if cur is not None and cur != self.prev else None
        self.prev = cur
        return fired


def _with_live_viewer(env, viewer, fn):
    """Run a blocking tilt/press while refreshing the GUI each physics step."""
    original_step = env.scene.step

    def stepped():
        original_step()
        if viewer is not None and not viewer.closed:
            env.scene.update_render()
            viewer.render()

    env.scene.step = stepped
    try:
        return fn()
    finally:
        env.scene.step = original_step


def main():
    parser = argparse.ArgumentParser(description="Interactive marble_shelf_maze viewer")
    parser.add_argument("--config", default="demo_dynamic", help="Task config name without .yml")
    parser.add_argument("--seed", type=int, default=0, help="Scene randomization seed")
    add_robot_motion_arg(parser)
    args = parser.parse_args()

    from envs import CONFIGS_PATH
    from envs.marble_shelf_maze import marble_shelf_maze
    from envs.utils.action import ArmTag
    globals()["CONFIGS_PATH"] = CONFIGS_PATH

    print_mode_controls("marble_shelf_maze", args.control, keyboard=CONTROLS_KEYBOARD, robot=CONTROLS_ROBOT)

    env = marble_shelf_maze()
    env.setup_demo(**_configure_task(args.config, args.seed, use_robot=args.control == "robot"))
    env.together_close_gripper(save_freq=None)
    env._bowl_armed = bool(getattr(env, "osc_bowl_enabled", False))
    env.plan_success = True

    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    views = make_viewer_view_toggle(env, viewer)

    dirs = list(getattr(env, "correct_dir", []) or [])
    print(f"Shelves={env.n_shelves}. Suggested directions top→bottom: {dirs}")
    motion = f", robot-motion={args.robot_motion}" if args.control == "robot" else ""
    print(f"Control={args.control}{motion}. Tap Left/A or Right/D to tilt.")

    edges = EdgeDirection()
    busy = False
    robot_controller = None
    if args.control == "robot":
        def get_button(e, side):
            return e.left_button if side == "left" else e.right_button

        def get_top_z(e, side):
            btn = e.left_button if side == "left" else e.right_button
            return float(btn.get_pose().p[2]) + float(e.button_half[2])

        def on_press(e, m):
            ok = _with_live_viewer(e, viewer, lambda: e._press_tilt_direct(m))
            print(f"Tilt done. ball_mode={e._ball_mode} ok={ok}")

        robot_controller = make_button_controller(
            env, ArmTag, args.robot_motion,
            get_button=get_button,
            get_top_z=get_top_z,
            arms_for_mode=lambda m: (m,) if m else (),
            on_press=on_press,
            hold=False,
            sides=("left", "right"),
        )

    try:
        while not viewer.closed:
            views.update(viewer.window)
            frame_start = time.perf_counter()
            direction = edges.edge(viewer.window)
            mode = str(getattr(env, "_ball_mode", ""))
            can_tilt = (
                direction is not None
                and not busy
                and env.active_shelf_idx >= 0
                and mode not in ("sliding", "falling", "missed", "done")
            )
            if can_tilt:
                busy = True
                idx = env.active_shelf_idx
                try:
                    if args.control == "keyboard":
                        print(f"Tilting shelf {idx} {direction} (keyboard)...")
                        ok = _with_live_viewer(env, viewer, lambda: env._press_tilt_direct(direction))
                        print(f"Tilt done. ball_mode={env._ball_mode} ok={ok}")
                    elif robot_controller is not None:
                        print(f"Robot tapping {direction} button for shelf {idx}...")
                        robot_controller.update(direction)
                        print(
                            f"Press done. ball_mode={env._ball_mode} "
                            f"plan_success={env.plan_success}"
                        )
                finally:
                    busy = False
            env._update_kinematic_tasks()
            env.scene.step()
            env.scene.update_render()
            viewer.render()
            mode = str(getattr(env, "_ball_mode", ""))
            if mode in ("done", "missed") and int(getattr(env, "active_shelf_idx", 0)) < 0:
                report_task_result(env, f"ball_mode={mode}")
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
