#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive viewer for ``save_goal``.

Run from any directory:

    /path/to/RoboDynaExp/interactive/base/interactive_save_goal.py --control keyboard
    /path/to/RoboDynaExp/interactive/base/interactive_save_goal.py --control robot

Place the square keeper in the green zone before the red line so the solid
keeper can bounce the ball from any angle (mass-aware: ball 100 g, keeper 500 g).
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "script" / "bench_script"))
sys.path.insert(0, str(REPO_ROOT / "interactive"))

from _interactive_common import (  # noqa: E402
    add_record_data_arg,
    make_viewer_view_toggle,
    print_mode_controls,
    report_task_result,
    RealtimePhysicsPacer,
    terminal_hold_should_close,
    print_episode_condition,
)


CONTROLS_KEYBOARD = """
  Arrow keys        nudge keeper XY (stay inside the green zone)

  Place the keeper before the ball crosses the red line.
"""

CONTROLS_ROBOT = """
  Arrow keys        nudge keeper XY (stay inside the green zone)

  Place the keeper before the ball crosses the red line.
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
        task_name="save_goal",
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


def _clip_keeper_xy(env, x, y):
    x_min = float(env.green_area_x_min + env.keeper_half_x)
    x_max = float(env.green_area_x_max - env.keeper_half_x)
    y_min = float(env.green_area_y_min + env.keeper_half_y)
    y_max = float(env.green_area_y_max - env.keeper_half_y)
    return float(np.clip(x, x_min, x_max)), float(np.clip(y, y_min, y_max))


def _set_keeper_xy(env, x, y, clip=True):
    import sapien
    if clip:
        x, y = _clip_keeper_xy(env, x, y)
    z = float(env.table_top_z + env.keeper_half_z)
    q = list(env.goalkeeper.get_pose().q)
    pose = sapien.Pose([x, y, z], q)
    try:
        env.goalkeeper.set_pose(pose)
    except Exception:
        env.goalkeeper.actor.set_pose(pose)
    rigid = env._get_rigid(env.goalkeeper)
    if rigid is not None:
        try:
            rigid.set_linear_velocity(np.zeros(3))
            rigid.set_angular_velocity(np.zeros(3))
            rigid.set_kinematic(True)
            rigid.set_kinematic_target(pose)
        except Exception:
            pass
    return x, y


def _start_shot(env):
    import sapien
    env._ball_step = 0
    env._ball_blocked = False
    env._ball_live = False
    env._block_was_legal = False
    env._goal_conceded = False
    env._late_failure = False
    env._ball_crossed_goal = False
    env._keeper_deployed = False
    env._keeper_drop_pose = None
    env._ball_motion_active = True
    if hasattr(env, "_set_collision_enabled"):
        env._set_collision_enabled(env.ball, False)
    start = getattr(env, "ball_start_pose", None)
    if start is not None and env.ball is not None:
        pose = sapien.Pose(np.asarray(start, dtype=float).tolist(), [1, 0, 0, 0])
        env.ball.set_pose(pose)
        rigid = getattr(env, "_ball_rigid", None)
        if rigid is not None:
            try:
                rigid.set_kinematic(True)
                rigid.set_kinematic_target(pose)
                rigid.set_linear_velocity(np.zeros(3))
                rigid.set_angular_velocity(np.zeros(3))
            except Exception:
                pass


class KeyboardKeeperController:
    def __init__(self, env):
        self.env = env
        self.deployed = False

    def update(self, window):
        if self.deployed:
            return
        dx, dy = _nudge_from_keys(window)
        if dx or dy:
            p = np.asarray(self.env.goalkeeper.get_pose().p, dtype=float)
            _set_keeper_xy(self.env, p[0] + dx, p[1] + dy, clip=True)


class RobotKeeperController:
    """Teleop only — Space opens/closes the gripper only."""

    def __init__(self, env, ArmTag):
        self.env = env
        self.ArmTag = ArmTag

    def update(self, window):
        # Shared viewer Space / arrows / E/Q own gripper and arm motion.
        return


def main():
    parser = argparse.ArgumentParser(description="Interactive save_goal viewer")
    parser.add_argument("--config", default="demo_dynamic", help="Task config name without .yml")
    parser.add_argument("--seed", type=int, default=0, help="Scene randomization seed")
    parser.add_argument(
        "--control",
        choices=("keyboard", "keyboard+mouse", "robot"),
        default="robot",
        help="Interaction method (default: robot)",
    )
    parser.add_argument(
        "--robot-motion",
        choices=("planner", "interpolate"),
        default="planner",
        help="Robot motion backend (interpolate = faster joint interp when supported; default planner)",
    )
    add_record_data_arg(parser)
    args = parser.parse_args()

    from envs import CONFIGS_PATH
    from envs.save_goal import save_goal
    from envs.utils.action import ArmTag
    globals()["CONFIGS_PATH"] = CONFIGS_PATH

    print_mode_controls("save_goal", args.control, keyboard=CONTROLS_KEYBOARD, robot=CONTROLS_ROBOT)
    if args.robot_motion == "interpolate":
        print(
            "Note: --robot-motion interpolate uses planner motions for this teleop task "
            "(key-press sandboxes use joint interpolation)."
        )

    env = save_goal()
    env.setup_demo(**_configure_task(args.config, args.seed, use_robot=args.control == "robot"))
    print_episode_condition(env)
    _start_shot(env)
    target = env.goalkeeper_target_pose.p if env.goalkeeper_target_pose is not None else [0, 0, 0]
    print(
        f"Shot started. Intercept target ≈ ({target[0]:.3f}, {target[1]:.3f}); "
        f"red_line_x={env.red_line_x:.3f}; mirrored={env.mirrored}."
    )

    controller = (
        RobotKeeperController(env, ArmTag) if args.control == "robot" else KeyboardKeeperController(env)
    )

    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    views = make_viewer_view_toggle(env, viewer)

    done_since = None
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

            shot_done = (not getattr(env, "_ball_motion_active", False)) or env._goal_conceded
            if shot_done:
                if done_since is None:
                    done_since = time.perf_counter()
                elif time.perf_counter() - done_since >= 1.0:
                    report_task_result(
                        env,
                        f"in_zone={env._keeper_in_zone()}, legal_block={getattr(env, '_block_was_legal', False)}, "
                        f"blocked={env._ball_blocked}, late={env._late_failure}, conceded={env._goal_conceded}",
                    )
                    terminal_started_at = time.perf_counter()
    finally:
        env.close_env()


if __name__ == "__main__":
    main()
    # household_task_gui convention: 0=SUCCESS, 10=FAILURE, 2=no result
    from _interactive_common import task_result_exit_code
    raise SystemExit(task_result_exit_code())
