#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive viewer for ``dispense_gummy``.

Run from any directory:

    /path/to/RoboDynaExp/script_exp/interactive_dispense_gummy.py --control keyboard
    /path/to/RoboDynaExp/script_exp/interactive_dispense_gummy.py --control robot
    /path/to/RoboDynaExp/script_exp/interactive_dispense_gummy.py --control robot --robot-motion interpolate

Keyboard mode forces belt-key / dispense latches. Robot mode: right arm for belt
keys, left arm for the dispense key. Sandbox only — not data collection.
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
  dispense_gummy — interactive controls
============================================================
  Left Arrow / A   →  move bowl LEFT  (right-arm belt key)
  Right Arrow / D  →  move bowl RIGHT (right-arm belt key)
  Space            →  dispense (left-arm key)

  Continuous belt (Opt 2): hold A/D to slide.
  Discrete belt (default): tap A/D to hop one station.

  --control keyboard : force key presses / dispense request
  --control robot    : arms physically press the keys
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

    config.update(
        task_name="dispense_gummy",
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


def _belt_side(window):
    left = window.key_down("left") or window.key_down("a")
    right = window.key_down("right") or window.key_down("d")
    if left and not right:
        return "left"
    if right and not left:
        return "right"
    return None


def _remaining_in_tubes(env):
    left = list(env._tube_stack_colors["left"][env._dispensed_count["left"]:])
    right = list(env._tube_stack_colors["right"][env._dispensed_count["right"]:])
    return left, right


def _unrecoverable(env):
    """Any target miss or distractor catch ends the episode as a failure."""
    target_missed = env.yellow_missed if env.target_color == "yellow" else env.blue_dropped
    distractor_caught = env.blue_caught if env.target_color == "yellow" else env.yellow_caught
    return int(target_missed) > 0 or int(distractor_caught) > 0


def _episode_done(env):
    """Return ``(done, detail)`` for a definitive success/failure state."""
    if bool(getattr(env, "invalid_pattern", False)):
        return True, "invalid_pattern"
    if _unrecoverable(env):
        target_missed = env.yellow_missed if env.target_color == "yellow" else env.blue_dropped
        distractor_caught = env.blue_caught if env.target_color == "yellow" else env.yellow_caught
        return True, f"unrecoverable (target_missed={target_missed}, distractor_caught={distractor_caught})"
    left, right = _remaining_in_tubes(env)
    if not left and not right and not getattr(env, "_active_drops", None):
        return True, "tubes empty"
    return False, None


class KeyboardState:
    def __init__(self):
        self.prev_space = False

    def update(self, env, window):
        env._expert_belt_hold = _belt_side(window)
        env._bowl_force_stop = False
        space = window.key_down("space")
        if space and not self.prev_space:
            env._expert_dispense = True
        self.prev_space = space


class RobotGummyController:
    """Right-arm belt hold/tap + left-arm dispense tap via shared button controllers.

    Belt left/right keys both use the right arm, so each key gets its own
    controller instance (needed for correct interpolate precompute targets).
    """

    def __init__(self, env, arm_tag, robot_motion):
        self.env = env
        self.continuous = bool(getattr(env, "belt_continuous_motion", False))
        self.prev_space = False
        self._prev_belt = None
        self._active_belt = None
        active_dz = float(getattr(env, "belt_key_press_dz", 0.17))

        def set_belt_latch(e, mode):
            e._expert_belt_hold = mode
            e._bowl_force_stop = False

        def clear_belt_latch(e):
            e._expert_belt_hold = None

        self.belt = {}
        for belt_side in ("left", "right"):
            self.belt[belt_side] = make_button_controller(
                env,
                arm_tag,
                robot_motion,
                get_button=lambda e, _s, bs=belt_side: e.belt_keys[bs],
                get_top_z=lambda e, _s: e.belt_key_top_z,
                set_latch=set_belt_latch,
                clear_latch=clear_belt_latch,
                arms_for_mode=lambda m, bs=belt_side: ("right",) if m == bs else (),
                hold=self.continuous,
                active_dz=active_dz,
                sides=("right",),
            )
        self.dispense = make_button_controller(
            env,
            arm_tag,
            robot_motion,
            get_button=lambda e, _s: e.dispense_key,
            get_top_z=lambda e, _s: e.dispense_key_top_z,
            set_latch=lambda e, _m: None,
            clear_latch=lambda e: None,
            arms_for_mode=lambda m: ("left",) if m == "dispense" else (),
            on_press=lambda e, _m: setattr(e, "_expert_dispense", True),
            hold=False,
            active_dz=active_dz,
            sides=("left",),
        )

    def _update_belt(self, requested):
        if self.continuous:
            if requested != self._active_belt:
                if self._active_belt is not None:
                    self.belt[self._active_belt].update(None)
                self._active_belt = requested
                if requested is not None:
                    self.belt[requested].update(requested)
            elif requested is not None:
                self.belt[requested].update(requested)
            return
        # Discrete: edge-triggered shared tap (hold=False).
        if requested is not None and requested != self._prev_belt:
            self.belt[requested].update(requested)
        self._prev_belt = requested

    def update(self, window):
        self._update_belt(_belt_side(window))
        space = window.key_down("space")
        if space and not self.prev_space:
            self.dispense.update("dispense")
        self.prev_space = space

    def release(self):
        for ctrl in self.belt.values():
            ctrl.release()
        self._active_belt = None
        self.dispense.release()


def main():
    parser = argparse.ArgumentParser(description="Interactive dispense_gummy viewer")
    parser.add_argument("--config", default="demo_dynamic", help="Task config name without .yml")
    parser.add_argument("--seed", type=int, default=0, help="Scene randomization seed")
    add_robot_motion_arg(parser)
    args = parser.parse_args()

    from envs import CONFIGS_PATH
    from envs.dispense_gummy import dispense_gummy
    from envs.utils.action import ArmTag
    globals()["CONFIGS_PATH"] = CONFIGS_PATH

    print(CONTROLS)

    env = dispense_gummy()
    env.setup_demo(**_configure_task(args.config, args.seed, use_robot=args.control == "robot"))
    env.together_close_gripper(save_freq=None)
    env._expert_belt_hold = None
    env._expert_dispense = False
    env._bowl_force_stop = False

    keyboard = KeyboardState()
    robot_controller = (
        RobotGummyController(env, ArmTag, args.robot_motion)
        if args.control == "robot"
        else None
    )

    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    views = make_viewer_view_toggle(env, viewer)

    mode = "continuous" if getattr(env, "belt_continuous_motion", False) else "discrete"
    print(f"Belt mode: {mode}. Control={args.control}. robot-motion={args.robot_motion}.")

    try:
        while not viewer.closed:
            views.update(viewer.window)
            frame_start = time.perf_counter()
            if args.control == "keyboard":
                keyboard.update(env, viewer.window)
            elif robot_controller is not None:
                robot_controller.update(viewer.window)
            env._update_kinematic_tasks()
            env.scene.step()
            env.scene.update_render()
            viewer.render()
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
