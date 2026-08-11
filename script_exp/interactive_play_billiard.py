#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive viewer for ``play_billiard``.

Run from any directory:

    /path/to/RoboDynaExp/script_exp/interactive_play_billiard.py --control keyboard
    /path/to/RoboDynaExp/script_exp/interactive_play_billiard.py --control robot

Keyboard mode aims the cue; slide the tip into the ball to hit (blue tip only,
one contact). Robot mode: grasp/release with Space, aim with arrows/E/Q/R/T, and
drive the blue tip into the ball for a single hit.
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
    UniversalRobotControls,
)


CONTROLS_KEYBOARD = """
  Left / Right      rotate aim direction
  Up / Down         slide tip along aim (approach / retreat)
  (hit)             drive the blue tip into the ball — one tip contact only
"""

CONTROLS_ROBOT = """
  Space             open / close gripper to grasp or release the cue
  (hit)             move the blue tip into the ball — one tip contact only
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
        task_name="play_billiard",
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


def _default_aim(env):
    pocket, pid = env._choose_pocket()
    if not env.specific_hole:
        env._target_pocket = pocket
        env._target_pocket_id = pid
        env._target_pocket_name = env.POCKET_NAMES[pid]
    ball_xy = env._ball_xy(env.primary_ball)
    aim = pocket[:2] - ball_xy
    n = float(np.linalg.norm(aim))
    if n < 1e-6:
        aim = np.array([0.0, 1.0])
        n = 1.0
    env._aim_dir = aim / n
    return env._aim_dir


def _place_cue_for_aim(env, gap=None):
    """Park the cue tip behind the ball along ``_aim_dir`` (keyboard sandbox)."""
    if gap is None:
        gap = float(env.APPROACH_GAP)
    ball = np.asarray(env.primary_ball.get_pose().p, dtype=float)
    aim = np.asarray(env._aim_dir, dtype=float)
    tip_xy = ball[:2] - aim * gap
    tip = np.array([tip_xy[0], tip_xy[1], float(env.ball_z)], dtype=float)
    # Cue body: tip along +local X; park body so tip ≈ ball − aim*gap.
    yaw = float(np.arctan2(aim[1], aim[0]))
    q = _yaw_quat(yaw)
    half = float(env.CUE_HALF_LEN)
    new_body = tip - np.array([aim[0] * half, aim[1] * half, 0.0])
    new_pose = sapien.Pose(new_body.tolist(), q)
    # ``cue`` is the task's Actor wrapper; pose updates belong to its
    # underlying SAPIEN actor (matching the main task's cue placement code).
    env.cue.actor.set_pose(new_pose)
    rigid = _get_rigid(env.cue)
    if rigid is not None:
        try:
            rigid.set_kinematic(True)
            rigid.set_linear_velocity(np.zeros(3))
            rigid.set_angular_velocity(np.zeros(3))
            rigid.set_kinematic_target(new_pose)
        except Exception:
            pass
    return tip


def _yaw_quat(yaw):
    """Quaternion (wxyz) for a yaw about +Z."""
    half = 0.5 * yaw
    return [float(np.cos(half)), 0.0, 0.0, float(np.sin(half))]


class KeyboardCueController:
    def __init__(self, env):
        self.env = env
        self.struck = False
        self.gap = float(env.APPROACH_GAP)
        _default_aim(env)
        _place_cue_for_aim(env, gap=self.gap)
        print(
            f"Aim → {env._target_pocket_name}; tip parked behind the red ball. "
            "Slide the blue tip into the ball to hit (one contact)."
        )

    def update(self, window):
        if self.env._strike_done or self.env._primary_pocketed:
            self.struck = True
            return
        # Rotate aim
        rot = 0.0
        if window.key_down("left"):
            rot += 0.04
        if window.key_down("right"):
            rot -= 0.04
        if rot:
            c, s = np.cos(rot), np.sin(rot)
            ax, ay = self.env._aim_dir
            self.env._aim_dir = np.array([c * ax - s * ay, s * ax + c * ay], dtype=float)
            n = float(np.linalg.norm(self.env._aim_dir))
            self.env._aim_dir /= max(n, 1e-6)
            _place_cue_for_aim(self.env, gap=self.gap)
        if window.key_down("up"):
            self.gap = max(0.018, self.gap - 0.004)
            _place_cue_for_aim(self.env, gap=self.gap)
        if window.key_down("down"):
            self.gap = min(0.12, self.gap + 0.004)
            _place_cue_for_aim(self.env, gap=self.gap)


class RobotCueController:
    """Space physics grasp; shared teleop aims; tip-only one-shot hit."""

    def __init__(self, env, ArmTag):
        self.env = env
        self.ArmTag = ArmTag
        self.arm = ArmTag(env._arm_side)
        self.struck = False
        _default_aim(env)

    def update(self, window):
        if self.env._strike_done or self.env._primary_pocketed:
            if not self.struck:
                print(
                    "Tip hit registered."
                    if self.env._strike_done
                    else "Ball pocketed."
                )
            self.struck = True


def main():
    parser = argparse.ArgumentParser(description="Interactive play_billiard viewer")
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
        default="planner",
        help="Robot motion backend (interpolate = faster joint interp when supported; default planner)",
    )
    args = parser.parse_args()

    from envs import CONFIGS_PATH
    from envs.play_billiard import play_billiard
    from envs.utils.action import ArmTag
    globals()["CONFIGS_PATH"] = CONFIGS_PATH

    print_mode_controls("play_billiard", args.control, keyboard=CONTROLS_KEYBOARD, robot=CONTROLS_ROBOT)
    if args.robot_motion == "interpolate":
        print(
            "Note: --robot-motion interpolate uses planner motions for this teleop task "
            "(key-press sandboxes use joint interpolation)."
        )

    env = play_billiard()
    # Always enable arm teleop + Space grasp/release; hit by driving the tip into the ball.
    env._interactive_robot_mode = True
    env.setup_demo(**_configure_task(args.config, args.seed, use_robot=True))
    env._interactive_selected_arms = (env._arm_side,)
    print_episode_condition(env)
    print(
        f"Arm={env._arm_side}; target pocket={env._target_pocket_name}; "
        f"specific_hole={env.specific_hole}; distractors={env.enable_distractors}. "
        "Blue tip only; one contact."
    )

    controller = (
        RobotCueController(env, ArmTag) if args.control == "robot"
        else KeyboardCueController(env)
    )

    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    views = make_viewer_view_toggle(env, viewer)
    if views.robot_controls is None:
        views.robot_controls = UniversalRobotControls(env)

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
            env.scene.update_render()
            viewer.render()

            if viewer.window.key_down("escape"):
                break

            if terminal_started_at is not None:
                if terminal_hold_should_close(terminal_started_at):
                    break
                continue

            if env._robot_ball_contact:
                report_task_result(env, "robot touched ball")
                terminal_started_at = time.perf_counter()
                continue
            if env._strike_done or env._primary_pocketed:
                if settle_after is None:
                    settle_after = time.perf_counter()
                    print("Ball in motion; settling…")
                elif time.perf_counter() - settle_after >= 3.0:
                    report_task_result(env)
                    terminal_started_at = time.perf_counter()
                    continue
    finally:
        env.close_env()


if __name__ == "__main__":
    main()
    # household_task_gui convention: 0=SUCCESS, 10=FAILURE, 2=no result
    from _interactive_common import task_result_exit_code
    raise SystemExit(task_result_exit_code())
