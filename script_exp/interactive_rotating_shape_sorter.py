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
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _interactive_common import (  # noqa: E402
    add_robot_motion_arg,
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


INTERACTIVE_ROBOT_HOLD_CLEARANCE = 0.025
INTERACTIVE_CYLINDER_SIDE_CLEARANCE = 0.035


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


def _nudge_from_keys(window, xy_step=0.022, z_step=0.016):
    """Map arrows to world XY and E/Q to vertical hold movement."""
    dx = xy_step * (window.key_down("right") - window.key_down("left"))
    dy = xy_step * (window.key_down("up") - window.key_down("down"))
    dz = z_step * (window.key_down("e") - window.key_down("q"))
    return float(dx), float(dy), float(dz)


def _prepare_robot_hold(env, selected_arm=None):
    from envs.utils.action import ArmTag

    arm_name = selected_arm or ("left" if env.ball_side == "left" else "right")
    arm_tag = ArmTag(arm_name)
    env.selected_arm = arm_name
    env._cap_tracking = True
    env.move(env.grasp_actor(env.ball, arm_tag=arm_tag, pre_grasp_dis=0.08))
    # Clear the container before translating, then stage just beside it. Do
    # not pre-align to the moving hole: the user positions the ball with keys.
    transport_z = float(
        env.cap_z + env.cap_thickness + env.ball_radius + env.transport_clearance_z
    )
    hold_z = float(
        env.cap_z + env.cap_thickness + env.ball_radius
        + INTERACTIVE_ROBOT_HOLD_CLEARANCE
    )
    env._move_ball_to_height(arm_tag=arm_tag, target_z=transport_z)
    side_sign = -1.0 if env.ball_side == "left" else 1.0
    container_radius = float(getattr(env, "cap_radius", env.cap_half_extent))
    side_xy = np.array([
        float(env.bucket_center[0]) + side_sign * (
            container_radius + float(env.ball_radius) + INTERACTIVE_CYLINDER_SIDE_CLEARANCE
        ),
        float(env.bucket_center[1]),
    ])
    ball_xy = np.asarray(env.ball.get_pose().p[:2], dtype=np.float64)
    delta = side_xy - ball_xy
    if np.linalg.norm(delta) > 1e-4:
        env.move(env.move_by_displacement(
            arm_tag=arm_tag, x=float(delta[0]), y=float(delta[1]), move_axis="world",
        ))
    env._move_ball_to_height(arm_tag=arm_tag, target_z=hold_z)
    env._interactive_arm = arm_tag
    env._interactive_holding = True
    env._interactive_released = False
    env.ball_released = False


class RobotHoldMotion:
    """Move a held ball without blocking the viewer on every nudge."""

    DURATION = 0.06

    def __init__(self, env, arm_tag, motion):
        self.env = env
        self.arm_tag = arm_tag
        self.motion = motion
        self.side = str(arm_tag)
        self._start = None
        self._target = None
        self._started_at = None

    def _drive_qpos(self):
        joints = self.env.robot.left_arm_joints if self.side == "left" else self.env.robot.right_arm_joints
        return np.asarray([joint.get_drive_target()[0] for joint in joints], dtype=np.float64)

    def _ee_pose(self):
        get_pose = self.env.robot.get_left_ee_pose if self.side == "left" else self.env.robot.get_right_ee_pose
        return np.asarray(get_pose(), dtype=np.float64)

    def _plan_target(self, dx, dy, dz):
        pose = self._ee_pose().copy()
        pose[:3] += np.array([dx, dy, dz], dtype=np.float64)
        planner = self.env.robot.left_plan_path if self.side == "left" else self.env.robot.right_plan_path
        result = planner(pose, last_qpos=np.asarray(self._drive_qpos(), dtype=np.float32))
        if result is None or result.get("status") != "Success":
            return None
        return np.asarray(result["position"][-1], dtype=np.float64)

    def nudge(self, dx, dy, dz):
        if self.motion == "planner":
            self.env.move(self.env.move_by_displacement(
                arm_tag=self.arm_tag, x=dx, y=dy, z=dz, move_axis="world",
            ))
            return
        # Do not restart an in-flight curve every simulation frame. Holding a
        # key schedules a sequence of complete, visibly smooth short moves.
        if self._started_at is not None:
            return
        target = self._plan_target(dx, dy, dz)
        if target is None:
            return
        self._start = self._drive_qpos()
        self._target = target
        self._started_at = time.perf_counter()

    def update(self):
        if self._started_at is None:
            return
        progress = min(1.0, (time.perf_counter() - self._started_at) / self.DURATION)
        smooth = progress * progress * (3.0 - 2.0 * progress)
        position = self._start + (self._target - self._start) * smooth
        velocity = (
            (self._target - self._start) / self.DURATION
            if progress < 1.0 else np.zeros_like(self._target)
        )
        self.env.robot.set_arm_joints(position, velocity, self.side)
        if progress >= 1.0:
            self._started_at = None


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
    env._interactive_selected_arms = (
        "left" if env.ball_side == "left" else "right",
    )
    # Space performs the initial pickup, so the idle state must exist before
    # the viewer loop evaluates release/hold conditions.
    env._interactive_holding = False
    env._interactive_released = False

    print_banner(
        "rotating_shape_sorter — interactive controls",
        [
            f"Mode: {args.control}  |  robot-motion: {args.robot_motion}  |  "
            f"config: {args.config}  |  seed: {args.seed}",
            "Goal: drop the ball through the rotating corner hole into the box.",
            "Space  — pick up the ball; press again to release",
            "Arrows — move the held ball in XY",
            "E / Q — move the held ball up / down",
            "V — toggle view: top-down ↔ head_camera",
            "Esc — close the viewer window to quit",
            "Watch the spinning platform; release only when the hole passes under.",
            "--robot-motion planner|interpolate",
        ],
    )
    print("Press Space to pick up the ball, then position it over the moving hole.")

    keys_prev: dict = {}
    post_release = 0
    hold_motion = None

    def on_step(window, step):
        nonlocal post_release, hold_motion
        if edge_pressed(window, "space", keys_prev):
            if not getattr(env, "_interactive_holding", False):
                if use_robot:
                    print("Robot: picking up the ball…")
                    selected = tuple(getattr(env, "_interactive_selected_arms", ()))
                    _prepare_robot_hold(env, selected[0] if selected else None)
                    hold_motion = RobotHoldMotion(env, env._interactive_arm, args.robot_motion)
                else:
                    _prepare_keyboard_hold(env)
                print("Holding ball. Use arrows/E/Q to position it; Space releases.")
            elif not env._interactive_released:
                _do_release(env, use_robot)

        dx, dy, dz = _nudge_from_keys(window) if not use_robot else (0.0, 0.0, 0.0)
        if (dx or dy or dz) and getattr(env, "_interactive_holding", False) and not env._interactive_released:
            if use_robot and hold_motion is not None:
                # Planner mode executes robust Cartesian moves; interpolate mode
                # retargets a smooth joint trajectory without blocking rendering.
                if args.robot_motion == "interpolate" or step % 20 == 0:
                    hold_motion.nudge(dx, dy, dz)
            else:
                env._interactive_hold_xy = env._interactive_hold_xy + np.array([dx, dy])
                env._interactive_hold_z += dz
                hold_dynamic_at(
                    env._ball_rigid, env.ball,
                    [env._interactive_hold_xy[0], env._interactive_hold_xy[1], env._interactive_hold_z],
                )
        if hold_motion is not None:
            hold_motion.update()
        if getattr(env, "_interactive_holding", False) and not env._interactive_released and not use_robot:
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
