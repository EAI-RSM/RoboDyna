#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive viewer for ``hit_target``.

Run from any directory:

    /path/to/RoboDynaExp/script_exp/interactive_hit_target.py --control keyboard
    /path/to/RoboDynaExp/script_exp/interactive_hit_target.py --control robot

Keyboard mode aims the dart tip with arrows and drives it into the board.
Robot mode: grasp the dart with Space, aim with the movement keys, then jab the
tip into the yellow center by teleop.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import sapien
import sapien.physx
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "script" / "bench_script"))
sys.path.insert(0, str(REPO_ROOT / "script_exp"))

from _interactive_common import (  # noqa: E402
    make_viewer_view_toggle,
    print_mode_controls,
    report_task_result,
    RealtimePhysicsPacer,
    terminal_hold_should_close,
    print_episode_condition,
)


CONTROLS_KEYBOARD = """
  Arrow keys        aim dart tip (L/R = x, U/D = y / depth)
  E / Q             raise/lower dart tip
  Drive the tip into the yellow center to stick.
"""

CONTROLS_ROBOT = """
  Space             open / close gripper to grasp the dart
  Drive the tip into the yellow center with teleop to jab / stick.
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
        task_name="hit_target",
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


def _get_rigid(actor):
    for comp in actor.actor.get_components():
        if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent):
            return comp
    return None


def _tip(env):
    return np.asarray(env.dart.get_functional_point(0, "list")[:3], dtype=float)


def _set_tip_xyz(env, tip_xyz, kinematic=True):
    """Translate the dart so its tip lands at tip_xyz (keeps orientation)."""
    tip = _tip(env)
    body = np.asarray(env.dart.get_pose().p, dtype=float)
    offset = tip - body
    new_body = np.asarray(tip_xyz, dtype=float) - offset
    pose = env.dart.get_pose()
    new_pose = sapien.Pose(new_body.tolist(), pose.q)
    # ``create_actor`` returns an Actor wrapper; only its entity can be posed.
    entity = getattr(env.dart, "actor", env.dart)
    entity.set_pose(new_pose)
    rigid = env._dart_rigid or _get_rigid(env.dart)
    env._dart_rigid = rigid
    if rigid is None:
        return
    try:
        rigid.set_linear_velocity(np.zeros(3))
        rigid.set_angular_velocity(np.zeros(3))
        if kinematic:
            rigid.set_kinematic(True)
            rigid.set_kinematic_target(new_pose)
        else:
            rigid.set_kinematic(False)
    except Exception:
        pass


def _nudge_from_keys(window, step=0.008):
    dx = dy = dz = 0.0
    if window.key_down("left"):
        dx -= step
    if window.key_down("right"):
        dx += step
    if window.key_down("up"):
        dy += step
    if window.key_down("down"):
        dy -= step
    if window.key_down("q"):
        dz += step
    if window.key_down("e"):
        dz -= step
    return dx, dy, dz


class KeyboardDartController:
    def __init__(self, env):
        self.env = env
        tip = _tip(env)
        _set_tip_xyz(env, tip, kinematic=True)

    def update(self, window):
        if self.env._stuck or self.env._hit_blocker:
            return
        dx, dy, dz = _nudge_from_keys(window)
        if dx or dy or dz:
            tip = _tip(self.env)
            _set_tip_xyz(self.env, tip + np.array([dx, dy, dz]), kinematic=True)


class RobotDartController:
    """Teleop aim only — Space gripper toggle is shared; jab by driving tip in."""

    def __init__(self, env, ArmTag, robot_motion="interpolate"):
        self.env = env
        self.ArmTag = ArmTag
        self.arm = ArmTag("right" if env.dart_side > 0 else "left")
        self.busy = False
        self.robot_motion = robot_motion

    def update(self, window):
        # Universal viewer controls own arrow/E/Q / Space gripper motion.
        del window


def main():
    parser = argparse.ArgumentParser(description="Interactive hit_target viewer")
    parser.add_argument("--config", default="demo_dynamic", help="Task config name without .yml")
    parser.add_argument("--seed", type=int, default=0, help="Scene randomization seed")
    parser.add_argument(
        "--control",
        choices=("keyboard", "robot"),
        default="robot",
        help="Interaction method (default: robot)",
    )
    parser.add_argument(
        "--robot-motion",
        choices=("planner", "interpolate"),
        default="interpolate",
        help="Robot motion backend for aim nudges (default: interpolate)",
    )
    args = parser.parse_args()

    from envs import CONFIGS_PATH
    from envs.hit_target import hit_target
    from envs.utils.action import ArmTag
    globals()["CONFIGS_PATH"] = CONFIGS_PATH

    print_mode_controls("hit_target", args.control, keyboard=CONTROLS_KEYBOARD, robot=CONTROLS_ROBOT)
    env = hit_target()
    env.setup_demo(**_configure_task(args.config, args.seed, use_robot=args.control == "robot"))
    print_episode_condition(env)
    print(
        f"Arm={'right' if env.dart_side > 0 else 'left'}; "
        f"blocker_static={env.blocker_enabled}; blocker_dyn={env.blocker_dynamic}."
    )

    controller = (
        RobotDartController(env, ArmTag, args.robot_motion) if args.control == "robot"
        else KeyboardDartController(env)
    )

    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    views = make_viewer_view_toggle(env, viewer)

    settle_after = None
    terminal_started_at = None
    pacer = RealtimePhysicsPacer(env)

    try:
        while not viewer.closed:
            n_steps = pacer.begin_frame()
            views.update(viewer.window)
            controller.update(viewer.window)

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
                if not env._stuck and not env._hit_blocker:
                    env._try_form_stick(any_ring=True, exact_pose=True)
            env.scene.update_render()
            viewer.render()

            if viewer.window.key_down("escape"):
                break

            if terminal_started_at is not None:
                if terminal_hold_should_close(terminal_started_at):
                    break
                continue

            if env._hit_blocker:
                report_task_result(env, env.hit_result_detail())
                terminal_started_at = time.perf_counter()
                continue
            if env._stuck or env._hit_color is not None:
                if settle_after is None:
                    settle_after = time.perf_counter()
                elif time.perf_counter() - settle_after >= 1.0:
                    if not env._hit_blocker and env._hit_color is None:
                        env._record_board_hit()
                    report_task_result(env, env.hit_result_detail())
                    terminal_started_at = time.perf_counter()
                    continue
    finally:
        env.close_env()


if __name__ == "__main__":
    main()
    # household_task_gui convention: 0=SUCCESS, 10=FAILURE, 2=no result
    from _interactive_common import task_result_exit_code
    raise SystemExit(task_result_exit_code())
