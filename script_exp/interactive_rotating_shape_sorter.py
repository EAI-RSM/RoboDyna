#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive sandbox for ``rotating_shape_sorter``.

Hold the ball over the drop station; release when the hole aligns underneath.

Run from any directory:

    /path/to/RoboDynaExp/script_exp/interactive_rotating_shape_sorter.py --control keyboard
    /path/to/RoboDynaExp/script_exp/interactive_rotating_shape_sorter.py --control robot
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
        env.cap_z + env.cap_thickness + env.ball_radius + env.release_clearance_z
    )


def _prepare_keyboard_hold(env):
    z = _release_z(env)
    xy = np.asarray(env._drop_target_xy, dtype=np.float64).copy()
    hold_dynamic_at(env._ball_rigid, env.ball, [xy[0], xy[1], z])
    if env._ball_rigid is not None:
        try:
            env._ball_rigid.set_kinematic(True)
        except Exception:
            pass
    env._cap_tracking = True
    env.ball_released = False
    env._interactive_hold_xy = xy
    env._interactive_hold_z = z
    env._interactive_holding = True
    env._interactive_released = False


def _prepare_robot_hold(env):
    from envs.utils.action import ArmTag

    arm_name = "left" if env.ball_side == "left" else "right"
    arm_tag = ArmTag(arm_name)
    env.selected_arm = arm_name
    env._cap_tracking = True
    env.move(env.grasp_actor(env.ball, arm_tag=arm_tag, pre_grasp_dis=0.08))
    transport_z = float(
        env.cap_z + env.cap_thickness + env.ball_radius + env.transport_clearance_z
    )
    release_z = _release_z(env)
    env._move_ball_to_height(arm_tag=arm_tag, target_z=transport_z)
    ball_xy = np.array(env.ball.get_pose().p[:2], dtype=np.float64)
    delta = env._drop_target_xy - ball_xy
    if np.linalg.norm(delta) > 1e-4:
        env.move(env.move_by_displacement(
            arm_tag=arm_tag, x=float(delta[0]), y=float(delta[1]),
        ))
    env._move_ball_to_height(arm_tag=arm_tag, target_z=release_z)
    env._interactive_arm = arm_tag
    env._interactive_holding = True
    env._interactive_released = False
    env.ball_released = False


def _do_release(env, use_robot: bool):
    if getattr(env, "_interactive_released", False):
        return
    env._interactive_released = True
    env._interactive_holding = False
    release_dynamic(env._ball_rigid)
    if use_robot and getattr(env, "_interactive_arm", None) is not None:
        env.move(env.open_gripper(env._interactive_arm))
        try:
            env.move(env.move_by_displacement(
                arm_tag=env._interactive_arm, z=0.08, move_axis="arm",
            ))
        except Exception:
            pass
    env.ball_released = True
    print("Released ball — hope the hole was aligned.")


def main():
    parser = argparse.ArgumentParser(description="Interactive rotating_shape_sorter viewer")
    parser.add_argument("--config", default="demo_dynamic")
    parser.add_argument("--seed", type=int, default=0)
    add_robot_motion_arg(parser)
    args = parser.parse_args()

    from envs.rotating_shape_sorter import rotating_shape_sorter

    use_robot = args.control == "robot"
    env = rotating_shape_sorter()
    env.setup_demo(**configure_task(
        "rotating_shape_sorter", args.config, args.seed, use_robot=use_robot,
    ))

    print_banner(
        "rotating_shape_sorter — interactive controls",
        [
            f"Mode: {args.control}  |  robot-motion: {args.robot_motion}  |  "
            f"config: {args.config}  |  seed: {args.seed}",
            "Goal: drop the ball through the rotating corner hole into the box.",
            "Space  — open gripper / release when the hole is under the ball",
            "Arrows — nudge hold XY",
            "V — toggle view: top-down ↔ head_camera",
            "Q / Esc — close the viewer window to quit",
            "Watch the spinning platform; release only when the hole passes under.",
            "--robot-motion planner|interpolate",
        ],
    )
    if args.robot_motion == "interpolate":
        print(
            "Note: --robot-motion interpolate uses planner motions for this teleop task "
            "(key-press sandboxes use joint interpolation)."
        )

    if use_robot:
        print("Robot: grasping and moving to the release station…")
        _prepare_robot_hold(env)
        print("Holding over the station. Press Space when the hole aligns.")
    else:
        _prepare_keyboard_hold(env)
        print("Ball pinned over the station. Nudge with arrows; Space to release.")

    keys_prev: dict = {}
    post_release = 0

    def on_step(window, step):
        nonlocal post_release
        if edge_pressed(window, "space", keys_prev) and not env._interactive_released:
            _do_release(env, use_robot)
        nudge = arrow_nudge_xy(window, step=0.002)
        if float(np.linalg.norm(nudge)) > 0 and env._interactive_holding and not env._interactive_released:
            if use_robot and getattr(env, "_interactive_arm", None) is not None:
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
        if env._interactive_released and post_release > getattr(env, "post_release_steps", 220) + 40:
            return True, f"ball_in_box={env.ball_in_box}, stuck={env.ball_stuck_on_platform}"
        return False

    run_viewer_loop(env, on_step, is_done=is_done, max_steps=20000)


if __name__ == "__main__":
    main()
