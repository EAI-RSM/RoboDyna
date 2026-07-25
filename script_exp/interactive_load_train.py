#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive sandbox for ``load_train``.

Pick the ball, hover over the near rail, release into an open wagon.

Run from any directory:

    /path/to/RoboDynaExp/script_exp/interactive_load_train.py --control keyboard
    /path/to/RoboDynaExp/script_exp/interactive_load_train.py --control robot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _interactive_common import (  # noqa: E402
    add_robot_motion_arg,
    arrow_nudge_xy,
    bootstrap_repo,
    configure_task,
    edge_pressed,
    hold_dynamic_at,
    print_banner,
    release_dynamic,
    run_viewer_loop,
)

bootstrap_repo()


def _release_z(env) -> float:
    return float(
        env.car_floor_z + env.car_floor_h + env.car_wall_h
        + env.ball_radius + max(env.transport_clearance_z, env.release_clearance_z)
    )


def _prepare_keyboard_hold(env):
    """Teleport the ball to the rail drop station and pin it for teleop."""
    z = _release_z(env)
    xy = np.asarray(env._drop_target_xy, dtype=np.float64).copy()
    hold_dynamic_at(env._ball_rigid, env.ball, [xy[0], xy[1], z])
    if env._ball_rigid is not None:
        try:
            env._ball_rigid.set_kinematic(True)
        except Exception:
            pass
    env._ball_released = False
    env._interactive_hold_xy = xy
    env._interactive_hold_z = z
    env._interactive_holding = True
    env._interactive_released = False


def _prepare_robot_hold(env):
    """Grasp the ball and carry it to the drop station (timing left to the user)."""
    from envs.utils.action import ArmTag

    arm_name = "left" if env.ball_side == "left" else "right"
    arm_tag = ArmTag(arm_name)
    env.selected_arm = arm_name
    z = _release_z(env)

    def _try_grasp(tag):
        env.plan_success = True
        env.move(env.grasp_actor(env.ball, arm_tag=tag, pre_grasp_dis=0.1))
        if not env.plan_success:
            return False
        env._move_ball_to_height(arm_tag=tag, target_z=z)
        return float(env.ball.get_pose().p[2]) >= z - 0.08

    if not _try_grasp(arm_tag):
        try:
            env.move(env.back_to_origin(arm_tag))
        except Exception:
            pass
        alt = "right" if arm_name == "left" else "left"
        env.plan_success = True
        if _try_grasp(ArmTag(alt)):
            arm_name, arm_tag = alt, ArmTag(alt)
            env.selected_arm = arm_name

    if env._ball_rigid is not None:
        try:
            env._ball_rigid.set_disable_gravity(True)
            env._ball_rigid.set_linear_velocity([0, 0, 0])
            env._ball_rigid.set_angular_velocity([0, 0, 0])
        except Exception:
            pass

    reach_tol = float(getattr(env, "drop_reach_tol", env.DROP_REACH_TOL_DEFAULT))
    for _ in range(6):
        ball_xy = np.array(env.ball.get_pose().p[:2], dtype=np.float64)
        delta = env._drop_target_xy - ball_xy
        if float(np.linalg.norm(delta)) <= reach_tol:
            break
        env.move(env.move_by_displacement(
            arm_tag=arm_tag, x=float(delta[0]), y=float(delta[1]),
        ))
        env._move_ball_to_height(arm_tag=arm_tag, target_z=z)

    env._interactive_arm = arm_tag
    env._interactive_holding = True
    env._interactive_released = False
    env._ball_released = False


def _do_release(env, use_robot: bool):
    if getattr(env, "_interactive_released", False):
        return
    env._interactive_released = True
    env._interactive_holding = False
    if env._ball_rigid is not None:
        try:
            env._ball_rigid.set_disable_gravity(False)
        except Exception:
            pass
    release_dynamic(env._ball_rigid)
    if use_robot and getattr(env, "_interactive_arm", None) is not None:
        env.move(env.open_gripper(env._interactive_arm))
        try:
            env.move(env.move_by_displacement(
                arm_tag=env._interactive_arm, z=0.12, move_axis="world",
            ))
        except Exception:
            pass
    env._ball_released = True
    env._bed_contact_steps = 0
    print("Released ball — watch for wagon latch.")


def main():
    parser = argparse.ArgumentParser(description="Interactive load_train viewer")
    parser.add_argument("--config", default="demo_dynamic", help="Task config name without .yml")
    parser.add_argument("--seed", type=int, default=0)
    add_robot_motion_arg(parser)
    args = parser.parse_args()

    from envs.load_train import load_train

    use_robot = args.control == "robot"
    env = load_train()
    env.setup_demo(**configure_task("load_train", args.config, args.seed, use_robot=use_robot))
    env._train_running = True

    print_banner(
        "load_train — interactive controls",
        [
            f"Mode: {args.control}  |  robot-motion: {args.robot_motion}  |  "
            f"config: {args.config}  |  seed: {args.seed}",
            "Goal: drop the ball into an open wagon as it passes under the near rail.",
            "Space  — release / open gripper when a wagon is aligned",
            "Arrows — nudge hold XY (keyboard teleop, or robot displace)",
            "V — toggle view: top-down ↔ head_camera",
            "Q / Esc — close the viewer window to quit",
            "Robot mode starts with grasp+hover already done; time Space for the drop.",
            "Keyboard mode pins the ball over the rail; time Space for the drop.",
            "--robot-motion planner|interpolate",
        ],
    )
    if args.robot_motion == "interpolate":
        print(
            "Note: --robot-motion interpolate uses planner motions for this teleop task "
            "(key-press sandboxes use joint interpolation)."
        )

    if use_robot:
        print("Robot: grasping and carrying to the drop station…")
        _prepare_robot_hold(env)
        print("Ready over the rail. Press Space when an open wagon is underneath.")
    else:
        _prepare_keyboard_hold(env)
        print("Ball pinned over the rail. Nudge with arrows; Space to release.")

    keys_prev: dict = {}
    post_release = 0

    def on_step(window, step):
        nonlocal post_release
        if edge_pressed(window, "space", keys_prev) and not env._interactive_released:
            _do_release(env, use_robot)
        nudge = arrow_nudge_xy(window, step=0.0025)
        if float(np.linalg.norm(nudge)) > 0 and env._interactive_holding and not env._interactive_released:
            if use_robot and getattr(env, "_interactive_arm", None) is not None:
                # Edge-ish: only displace occasionally so the planner is not flooded.
                if step % 8 == 0:
                    env.move(env.move_by_displacement(
                        arm_tag=env._interactive_arm,
                        x=float(nudge[0]) * 4,
                        y=float(nudge[1]) * 4,
                    ))
                    env._move_ball_to_height(arm_tag=env._interactive_arm, target_z=_release_z(env))
            else:
                env._interactive_hold_xy = env._interactive_hold_xy + nudge
                hold_dynamic_at(
                    env._ball_rigid, env.ball,
                    [env._interactive_hold_xy[0], env._interactive_hold_xy[1], env._interactive_hold_z],
                )
        elif env._interactive_holding and not env._interactive_released and not use_robot:
            hold_dynamic_at(
                env._ball_rigid, env.ball,
                [env._interactive_hold_xy[0], env._interactive_hold_xy[1], env._interactive_hold_z],
            )
        if env._interactive_released:
            post_release += 1

    def is_done(step):
        if env._interactive_released and post_release > 400:
            return True, f"ball_in_train={env.ball_in_train}"
        return False

    run_viewer_loop(env, on_step, is_done=is_done, max_steps=20000)


if __name__ == "__main__":
    main()
