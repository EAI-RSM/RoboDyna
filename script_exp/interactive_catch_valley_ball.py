#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive viewer for ``catch_valley_ball`` (push custom catch box).

Run from any directory:

    /path/to/RoboDynaExp/script_exp/interactive_catch_valley_ball.py --control keyboard
    /path/to/RoboDynaExp/script_exp/interactive_catch_valley_ball.py --control robot

Keyboard / robot sandbox: arrows nudge the box on the table; Space freezes it
past the red line. The scripted expert uses a closed-gripper contact push.
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

from _interactive_common import make_viewer_view_toggle, report_task_result, print_mode_controls  # noqa: E402


CONTROLS_KEYBOARD = """
  Arrows            slide custom catch box on the table
  Space             freeze box at current XY (must clear red line)
  V                 toggle view: top-down ↔ head_camera
  Escape            quit
------------------------------------------------------------
  Success: red ball in box, box fully past red line
"""

CONTROLS_ROBOT = """
  Arrows / E/Q      nudge selected arm (teleop)
  Space             freeze the box at its current table pose
  V                 toggle view: top-down ↔ head_camera
  Escape            quit
------------------------------------------------------------
  Expert policy pushes the box with a closed gripper (see play_once).
  Success: red ball in box, box fully past red line
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
        task_name="catch_valley_ball",
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


def _target_xy(env):
    landing = np.asarray(env.landing, dtype=float)
    return float(env._catch_target_x(landing[0])), float(landing[1])


def _nudge_from_keys(window, step=0.008):
    dx = dy = 0.0
    if window.key_down("left"):
        dx -= step
    if window.key_down("right"):
        dx += step
    if window.key_down("up"):
        dy += step
    if window.key_down("down"):
        dy -= step
    return dx, dy


class EdgeKey:
    def __init__(self):
        self._prev = False

    def poll(self, down):
        edge = bool(down) and not self._prev
        self._prev = bool(down)
        return edge


class KeyboardBoxController:
    def __init__(self, env):
        self.env = env
        self.placed = False
        self._space = EdgeKey()

    def update(self, window):
        if not self.placed:
            dx, dy = _nudge_from_keys(window)
            if dx or dy:
                p = np.asarray(self.env.bowl.get_pose().p, dtype=float)
                x = float(self.env._catch_target_x(p[0] + dx))
                y = float(np.clip(p[1] + dy, -0.50, 0.25))
                self.env._freeze_box(self.env._box_pose_at([x, y]))
        if self._space.poll(window.key_down("space")):
            p = np.asarray(self.env.bowl.get_pose().p, dtype=float)
            x = float(self.env._catch_target_x(p[0]))
            y = float(np.clip(p[1], -0.50, 0.25))
            self.env._freeze_box(self.env._box_pose_at([x, y]))
            self.env._bowl_ready = True
            self.placed = True
            print(f"Box frozen at ({x:.3f}, {y:.3f}) past red line.")


def main():
    parser = argparse.ArgumentParser(description="Interactive catch_valley_ball viewer")
    parser.add_argument("--config", default="demo_dynamic", help="Task config name without .yml")
    parser.add_argument("--seed", type=int, default=0, help="Scene randomization seed")
    parser.add_argument(
        "--control",
        choices=("keyboard", "robot"),
        default="keyboard",
        help="Interaction method (default: keyboard nudge)",
    )
    parser.add_argument(
        "--robot-motion",
        choices=("planner", "interpolate"),
        default="planner",
        help="Unused for keyboard nudge; kept for GUI compatibility",
    )
    args = parser.parse_args()

    from envs import CONFIGS_PATH
    from envs.catch_valley_ball import catch_valley_ball
    globals()["CONFIGS_PATH"] = CONFIGS_PATH

    print_mode_controls(
        "catch_valley_ball",
        args.control,
        keyboard=CONTROLS_KEYBOARD,
        robot=CONTROLS_ROBOT,
    )

    env = catch_valley_ball()
    env.setup_demo(**_configure_task(args.config, args.seed, use_robot=False))
    env._interactive_selected_arms = ("left" if env.mirrored else "right",)
    x, y = _target_xy(env)
    print(
        f"Predicted catch target ≈ ({x:.3f}, {y:.3f}); red_line_x={env.red_line_x:.3f}; "
        f"mirrored={env.mirrored}; catcher={env.catcher_model}/base{env.bowl_id}."
    )

    controller = KeyboardBoxController(env)
    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    views = make_viewer_view_toggle(env, viewer)

    print("Arrows nudge the box; Space freezes it past the red line.")
    settle_after = None
    try:
        while not viewer.closed:
            views.update(viewer.window)
            frame_start = time.perf_counter()
            controller.update(viewer.window)

            env._update_kinematic_tasks()
            env.scene.step()
            env.scene.update_render()
            viewer.render()

            if viewer.window.key_down("escape"):
                break

            if getattr(env, "_ball_phase", None) == "released":
                if settle_after is None:
                    settle_after = time.perf_counter()
                    print("Ball left the valley exit; waiting to settle…")
                elif time.perf_counter() - settle_after >= 2.5:
                    report_task_result(env)
                    break

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
