#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive sandbox for ``drop_ball_hole``.

Hold the ball over the drop station; release when the hole aligns underneath.

Run from any directory:

    /path/to/RoboDynaExp/script_exp/interactive_drop_ball_hole.py --control keyboard
    /path/to/RoboDynaExp/script_exp/interactive_drop_ball_hole.py --control robot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _interactive_common import (  # noqa: E402
    print_instructions,
    add_robot_motion_arg,
    bootstrap_repo,
    configure_task,
    gripper_width,
    hold_dynamic_at,
    print_banner,
    release_dynamic,
    run_viewer_loop,
    print_episode_condition,
    selected_robot_arms,
)

bootstrap_repo()


def _nudge_from_keys(window, xy_step=0.022, z_step=0.016):
    """Map arrows to world XY and E/Q to vertical hold movement."""
    dx = xy_step * (window.key_down("right") - window.key_down("left"))
    dy = xy_step * (window.key_down("up") - window.key_down("down"))
    dz = z_step * (window.key_down("e") - window.key_down("q"))
    return float(dx), float(dy), float(dz)


def main():
    parser = argparse.ArgumentParser(description="Interactive drop_ball_hole viewer")
    parser.add_argument("--config", default="demo_dynamic")
    parser.add_argument("--seed", type=int, default=0)
    add_robot_motion_arg(parser)
    args = parser.parse_args()

    from envs.drop_ball_hole import drop_ball_hole

    use_robot = args.control == "robot"
    env = drop_ball_hole()
    env.setup_demo(**configure_task(
        "drop_ball_hole", args.config, args.seed, use_robot=use_robot,
    ))
    print_episode_condition(env)
    env._interactive_released = False
    # Spin the platform for the interactive session.
    env._cap_tracking = True

    print_banner(
        "drop_ball_hole — interactive controls",
        [
            f"Mode: {args.control}  |  robot-motion: {args.robot_motion}  |  "
            f"config: {args.config}  |  seed: {args.seed}",
            "Goal: drop the ball through the rotating corner hole into the box.",
            "Arrows — move the ball in XY",
            "E / Q — move the ball up / down",
            "V — cycle view: head_camera ↔ gripper(s)",
            "Esc — close the viewer window to quit",
            "Use teleop / Space to grasp and release.",
            "Watch the spinning platform; release only when the hole passes under.",
            "Default / Opt2: after release the ball has 2s to fall into the box.",
            "--robot-motion planner|interpolate",
        ],
    )
    print_instructions(
        "Robot: close Space on the ball to grasp, open Space to drop. "
        "Keyboard: arrows/E/Q to move; release keys to drop."
    )
    if env._uses_drop_timeout():
        print(
            f"Drop timeout armed: {float(env.drop_timeout_s):.0f}s after release "
            "(stick_to_surface off)."
        )

    if not use_robot:
        p = np.asarray(env.ball.get_pose().p, dtype=np.float64)
        env._interactive_hold_xy = p[:2].copy()
        env._interactive_hold_z = float(p[2])

    prev_gripper_closed = False
    grasped_ball = False
    keyboard_moved = False

    def _tcp_xyz(side: str) -> np.ndarray | None:
        robot = getattr(env, "robot", None)
        if robot is None:
            return None
        getter = robot.get_left_tcp_pose if side == "left" else robot.get_right_tcp_pose
        try:
            return np.asarray(getter()[:3], dtype=np.float64)
        except Exception:
            return None

    def _ball_near_gripper(side: str, tol: float = 0.09) -> bool:
        if getattr(env, "ball", None) is None:
            return False
        tcp = _tcp_xyz(side)
        if tcp is None:
            return False
        ball = np.asarray(env.ball.get_pose().p[:3], dtype=np.float64)
        return float(np.linalg.norm(ball - tcp)) <= float(tol)

    def _mark_released(message: str):
        if bool(getattr(env, "ball_released", False)):
            return
        if hasattr(env, "mark_ball_released"):
            env.mark_ball_released()
        else:
            env.ball_released = True
        env._interactive_released = True
        # Keyboard kinematic pin only — robot grasp is PhysX / gripper contact.
        if not use_robot:
            release_dynamic(env._ball_rigid)
        print(message)

    def on_step(window, step):
        nonlocal prev_gripper_closed, grasped_ball, keyboard_moved
        # Robot: mark release only after a real manual grasp (closed near ball),
        # then Space-open. Opening before pickup must not start the drop timeout.
        if use_robot and not bool(getattr(env, "ball_released", False)):
            arms = selected_robot_arms(env)
            if arms:
                closed_now = all(float(gripper_width(env, side)) < 0.5 for side in arms)
                near_now = any(_ball_near_gripper(side) for side in arms)
                if closed_now and near_now:
                    grasped_ball = True
                opening = (not closed_now) and prev_gripper_closed
                if grasped_ball and opening:
                    if env._uses_drop_timeout():
                        _mark_released(
                            f"Released ball — {float(env.drop_timeout_s):.0f}s to land in the box "
                            "(default / opt2)."
                        )
                    else:
                        _mark_released("Released ball — hope the hole was aligned.")
                    grasped_ball = False
                prev_gripper_closed = closed_now

        # Keyboard: arrows / E/Q still move the ball; key-up drops it.
        if not use_robot and not env._interactive_released:
            dx, dy, dz = _nudge_from_keys(window)
            keys_down = any(
                bool(window.key_down(k))
                for k in ("left", "right", "up", "down", "e", "q")
            )
            if keys_down:
                keyboard_moved = True
                if dx or dy or dz:
                    env._interactive_hold_xy = env._interactive_hold_xy + np.array([dx, dy])
                    env._interactive_hold_z += dz
                hold_dynamic_at(
                    env._ball_rigid, env.ball,
                    [env._interactive_hold_xy[0], env._interactive_hold_xy[1], env._interactive_hold_z],
                )
            elif keyboard_moved:
                if env._uses_drop_timeout():
                    _mark_released(
                        f"Released ball — {float(env.drop_timeout_s):.0f}s to land in the box "
                        "(default / opt2)."
                    )
                else:
                    _mark_released("Released ball — hope the hole was aligned.")

    def is_done(step):
        if env.check_success():
            return True, "ball in box"
        if getattr(env, "_ball_fell_off_table", False):
            return True, "ball dropped off the table"
        if getattr(env, "_drop_timed_out", False):
            return True, (
                f"drop timeout ({float(env.drop_timeout_s):.0f}s) — ball not in box"
            )
        if env.ball_stuck_on_platform:
            return True, "ball stuck on platform"
        return False
    run_viewer_loop(env, on_step, is_done=is_done, max_steps=20000)


if __name__ == "__main__":
    main()
    # household_task_gui convention: 0=SUCCESS, 10=FAILURE, 2=no result
    from _interactive_common import task_result_exit_code
    raise SystemExit(task_result_exit_code())
