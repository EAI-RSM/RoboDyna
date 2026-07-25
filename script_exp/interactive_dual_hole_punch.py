#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive viewer for ``dual_hole_punch``.

Run from any directory:

    /path/to/RoboDynaExp/script_exp/interactive_dual_hole_punch.py --control keyboard
    /path/to/RoboDynaExp/script_exp/interactive_dual_hole_punch.py --control robot
    /path/to/RoboDynaExp/script_exp/interactive_dual_hole_punch.py --control robot --robot-motion interpolate

Keyboard mode calls ``_fire_punch``. Robot mode taps the side button(s) then fires.
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
 dual_hole_punch — interactive controls
============================================================
  A                 punch LEFT belt
  D                 punch RIGHT belt
  S                 punch BOTH belts
  Press when a card is under the stamp head
  V                 toggle view: top-down ↔ head_camera
  Q / Escape         quit
------------------------------------------------------------
  --control keyboard  direct _fire_punch (default)
  --control robot     arms tap the side buttons, then fire punch
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
        task_name="dual_hole_punch",
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


def _requested_sides(window):
    if window.key_down("s"):
        return ("left", "right")
    left = window.key_down("a")
    right = window.key_down("d")
    if left and right:
        return ("left", "right")
    if left:
        return ("left",)
    if right:
        return ("right",)
    return ()


def _sides_to_mode(sides):
    if sides == ("left", "right"):
        return "both"
    if sides == ("left",):
        return "left"
    if sides == ("right",):
        return "right"
    return None


class EdgeSides:
    def __init__(self):
        self._prev = ()

    def poll(self, sides):
        key = tuple(sides)
        edge = bool(key) and key != self._prev
        self._prev = key
        return key if edge else ()


def _start_belts(env):
    env._belt_active = True
    # Keep belts advancing every physics step in the interactive sandbox.
    env._belt_running = True
    if not getattr(env, "belt_continous_motion", False):
        # Discrete configs still advance while _belt_running is held True.
        pass


def _all_pages_resolved(env):
    for side in ("left", "right"):
        for k in range(env.n_pages):
            if env.page_missing[side][k]:
                continue
            if not env.page_punched[side][k]:
                return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Interactive dual_hole_punch viewer")
    parser.add_argument("--config", default="demo_dynamic", help="Task config name without .yml")
    parser.add_argument("--seed", type=int, default=0, help="Scene randomization seed")
    add_robot_motion_arg(parser)
    args = parser.parse_args()

    from envs import CONFIGS_PATH
    from envs.dual_hole_punch import dual_hole_punch
    from envs.utils.action import ArmTag
    globals()["CONFIGS_PATH"] = CONFIGS_PATH

    print(CONTROLS)

    env = dual_hole_punch()
    env.setup_demo(**_configure_task(args.config, args.seed, use_robot=args.control == "robot"))
    env.together_close_gripper(save_freq=None)
    _start_belts(env)

    def arms_for_mode(m):
        if m in ("both", "dump"):
            return ("left", "right")
        return (m,) if m else ()

    robot_controller = None
    if args.control == "robot":
        def get_button(e, side):
            return e.button[side]

        def get_top_z(e, side):
            return float(e.button[side].get_pose().p[2]) + float(e.BUTTON_HALF[2])

        def on_press(e, m):
            punched = []
            for side in arms_for_mode(m):
                e._fire_punch(side)
                punched.append(side)
            print(f"Robot punched: {', '.join(punched) if punched else 'none'}.")

        robot_controller = make_button_controller(
            env, ArmTag, args.robot_motion,
            get_button=get_button,
            get_top_z=get_top_z,
            arms_for_mode=arms_for_mode,
            on_press=on_press,
            hold=False,
            sides=("left", "right"),
        )

    edge = EdgeSides()

    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    viewer.set_camera_xyz(0.0, 0.0, 2.1)
    viewer.set_camera_rpy(0.0, -np.pi / 2.0, -np.pi / 2.0)
    views = make_viewer_view_toggle(env, viewer)

    if args.control == "robot":
        print(
            f"Robot mode ready (motion={args.robot_motion}). "
            "Tap A/D/S to press the side button(s)."
        )
    else:
        print("Keyboard mode ready. Tap A/D/S when a card is under the stamp.")

    try:
        while not viewer.closed:
            views.update(viewer.window)
            frame_start = time.perf_counter()
            sides = _requested_sides(viewer.window)
            fired = edge.poll(sides)
            if args.control == "keyboard":
                for side in fired:
                    env._fire_punch(side)
                if fired:
                    print(f"Punch fired: {', '.join(fired)}")
            elif robot_controller is not None:
                mode = _sides_to_mode(fired)
                if mode is not None:
                    robot_controller.update(mode)

            # Mark pages that slid past the stamp without a punch.
            env._mark_overdue_pages()

            env._update_kinematic_tasks()
            env.scene.step()
            env.scene.update_render()
            viewer.render()

            if viewer.window.key_down("q") or viewer.window.key_down("escape"):
                break

            if _all_pages_resolved(env):
                report_task_result(
                    env,
                    f"L={env.punch_score_L:.2f} R={env.punch_score_R:.2f} "
                    f"empty_press={env.invalid_empty_press}",
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
