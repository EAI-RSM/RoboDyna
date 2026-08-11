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
    print_instructions,
    action_failed,
    try_interactive_grasp,
    add_robot_motion_arg,
    arrow_nudge_xy,
    bootstrap_repo,
    configure_task,
    edge_pressed,
    hold_dynamic_at,
    print_banner,
    release_dynamic,
    require_selected_arms,
    run_viewer_loop,
    print_episode_condition,
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
    return True


def _prepare_robot_hold(env, selected_arm=None):
    """Run load_train's standard grasp/carry sequence on the user's request."""
    from envs.utils.action import ArmTag

    selected = require_selected_arms(env, exactly_one=True)
    if not selected:
        return False
    arm_name = selected[0]
    arm_tag = ArmTag(arm_name)
    env.selected_arm = arm_name
    z = _release_z(env)

    if not try_interactive_grasp(env, env.ball, arm_tag, pre_grasp_dis=0.1):
        env._interactive_holding = False
        return False
    env._move_ball_to_height(arm_tag=arm_tag, target_z=z)

    # Same authoritative hold check as load_train.play_once().
    if float(env.ball.get_pose().p[2]) < z - 0.08:
        action_failed(env, (arm_name,), detail="grasp failed")
        env._interactive_holding = False
        return False

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
    return True


def _drive_qpos(env, arm_name: str) -> np.ndarray:
    joints = env.robot.left_arm_joints if arm_name == "left" else env.robot.right_arm_joints
    return np.asarray([joint.get_drive_target()[0] for joint in joints], dtype=np.float64)


def _begin_interpolated_robot_nudge(env, arm_tag, dx: float, dy: float):
    """Create a nearby joint-space nudge that the GUI loop advances per frame."""
    arm_name = str(arm_tag)
    ee_pose = np.asarray(
        env.robot.get_left_ee_pose() if arm_name == "left" else env.robot.get_right_ee_pose(),
        dtype=np.float64,
    )
    ee_pose[:3] += np.asarray([dx, dy, 0.0], dtype=np.float64)
    plan = env.robot.left_plan_path if arm_name == "left" else env.robot.right_plan_path
    result = plan(ee_pose.tolist(), constraint_pose=[1, 1, 1, 0, 0, 0])
    if result is None or result.get("status") != "Success":
        return None
    positions = result.get("position")
    if positions is None or len(positions) == 0:
        return None
    start = _drive_qpos(env, arm_name)
    target = np.asarray(positions[-1], dtype=np.float64).reshape(-1)
    if target.shape != start.shape or float(np.max(np.abs(target - start))) > 0.70:
        return None
    return {"arm": arm_name, "start": start, "target": target, "index": 0, "steps": 18}


def _advance_interpolated_robot_nudge(env, motion) -> bool:
    """Advance one eased joint target; ``run_viewer_loop`` performs the physics step."""
    motion["index"] += 1
    alpha = motion["index"] / float(motion["steps"])
    smooth = alpha * alpha * (3.0 - 2.0 * alpha)
    delta = motion["target"] - motion["start"]
    position = motion["start"] + delta * smooth
    velocity = delta / float(motion["steps"])
    if motion["index"] >= motion["steps"]:
        position = motion["target"]
        velocity = np.zeros_like(delta)
    env.robot.set_arm_joints(position, velocity, motion["arm"])
    if motion["index"] >= motion["steps"]:
        env.plan_success = True
        return True
    return False


def _ball_held(env) -> bool:
    """True while fingers still contact the ball (shared G-close grasp)."""
    if getattr(env, "ball", None) is None:
        return False
    try:
        return len(env.get_gripper_actor_contact_position(env.ball.get_name())) > 0
    except Exception:
        return False


class BallReleaseMonitor:
    """Drive hold/release off the gripper so a manual G grasp is recognised.

    The viewer used to latch release only through the Space helper, so a player
    who picked the ball up with the shared teleop controls (G) and dropped it
    into a wagon never set ``_ball_released`` -- the env never ran its latch
    check and ``is_done`` never fired, so no SUCCESS/FAILURE was ever printed.
    """

    def __init__(self, env):
        self.env = env
        self.holding = False
        self._hold_contact_seen = False
        self._no_contact_steps = 0
        self._slip_no_contact_steps = 8

    def update(self):
        if getattr(self.env, "_interactive_released", False):
            return
        if _ball_held(self.env):
            if not self.holding:
                self.holding = True
                self.env._interactive_holding = True
                print("Ball grasped — carry it over a wagon, then G to open / release.")
            self._hold_contact_seen = True
            self._no_contact_steps = 0
            return
        if not self.holding:
            return
        self._no_contact_steps += 1
        limit = (
            self._slip_no_contact_steps
            if self._hold_contact_seen
            else self._slip_no_contact_steps * 4
        )
        if self._no_contact_steps < limit:
            return
        self.holding = False
        self.env._interactive_holding = False
        self.env._interactive_released = True
        if self.env._ball_rigid is not None:
            try:
                self.env._ball_rigid.set_disable_gravity(False)
            except Exception:
                pass
        release_dynamic(self.env._ball_rigid)
        self.env._ball_released = True
        self.env._bed_contact_steps = 0
        print("Ball released — watch for wagon latch.")


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
    env._ball_released = True
    env._bed_contact_steps = 0
    print("Released ball — watch for wagon latch.")


def main():
    parser = argparse.ArgumentParser(description="Interactive load_train viewer")
    parser.add_argument("--config", default="demo_dynamic", help="Task config name without .yml")
    parser.add_argument("--seed", type=int, default=0)
    add_robot_motion_arg(parser, robot_motion_default="interpolate")
    args = parser.parse_args()

    from envs.load_train import load_train

    use_robot = args.control == "robot"
    env = load_train()
    env.setup_demo(**configure_task("load_train", args.config, args.seed, use_robot=use_robot))
    env._interactive_selected_arms = (
        "left" if env.ball_side == "left" else "right",
    )
    print_episode_condition(env)
    env._train_running = True

    print_banner(
        "load_train — interactive controls",
        [
            f"Mode: {args.control}  |  robot-motion: {args.robot_motion}  |  "
            f"config: {args.config}  |  seed: {args.seed}",
            "Goal: drop the ball into an open wagon as it passes under the near rail.",
            "Opt 1 (target wagon): ONLY the RED wagon counts — gray ones are distractors.",
            "G — close on the ball to pick it up, open again over a wagon to drop it",
            "Arrows — nudge the held ball in XY (robot supports smooth interpolation)",
            "Space — optional shortcut: auto pick / auto release (keyboard mode)",
            "V — cycle view: head_camera ↔ gripper(s)",
            "Esc — close the viewer window to quit",
            "--robot-motion planner|interpolate",
        ],
    )
    env._interactive_holding = False
    env._interactive_released = False
    env._ball_released = False
    release_monitor = BallReleaseMonitor(env)
    print_instructions(
        "Ball ready. Grab it with G (or press Space to auto-pick), then drop it into a wagon."
    )

    keys_prev: dict = {}
    post_release = 0
    interpolate_motion = None

    def on_step(window, step):
        nonlocal post_release, interpolate_motion
        # Recognise a manual G grasp / open even when Space is never pressed.
        release_monitor.update()
        if edge_pressed(window, "space", keys_prev) and not env._interactive_released:
            if not env._interactive_holding:
                if use_robot:
                    print("Robot: picking up the ball and carrying it to the drop station…")
                    if _prepare_robot_hold(env):
                        print("Ball picked up. Press Space again when a wagon is aligned.")
                    else:
                        print("Ball pickup failed. Press Space to try again.")
                else:
                    _prepare_keyboard_hold(env)
                    print("Ball picked up. Nudge with arrows; press Space again to release.")
            else:
                _do_release(env, use_robot)
                interpolate_motion = None
        if interpolate_motion is not None:
            if _advance_interpolated_robot_nudge(env, interpolate_motion):
                interpolate_motion = None

        nudge = (arrow_nudge_xy(window, step=0.0025) if not use_robot
                 else np.zeros(2, dtype=np.float64))
        if float(np.linalg.norm(nudge)) > 0 and env._interactive_holding and not env._interactive_released:
            if use_robot and getattr(env, "_interactive_arm", None) is not None:
                # Avoid sending a movement request every viewer frame.
                if step % 8 == 0 and interpolate_motion is None:
                    dx, dy = float(nudge[0]) * 4, float(nudge[1]) * 4
                    if args.robot_motion == "interpolate":
                        interpolate_motion = _begin_interpolated_robot_nudge(
                            env, env._interactive_arm, dx, dy
                        )
                        if interpolate_motion is None:
                            print("Ball nudge could not reach the requested pose.")
                    else:
                        env.move(env.move_by_displacement(
                            arm_tag=env._interactive_arm, x=dx, y=dy,
                        ))
                        env._move_ball_to_height(
                            arm_tag=env._interactive_arm, target_z=_release_z(env)
                        )
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

    def _outcome_detail():
        """Explain the result in wagon terms.

        Opt 1 (``target_wagon_mode``) only counts the red target wagon, so a ball
        sitting in a gray distractor is a legitimate failure -- say so, instead of
        a bare ``ball_in_train=False`` that reads like a bug when the ball is
        visibly inside a wagon.
        """
        landed = getattr(env, "_latched_car_idx", None)
        target = getattr(env, "target_wagon_idx", None)
        if not getattr(env, "target_wagon_mode", False) or target is None:
            return "ball in a wagon" if env.ball_in_train else "ball not in any wagon"
        if landed is None:
            return f"ball not in any wagon (target was wagon {int(target)}, the red one)"
        if int(landed) == int(target):
            return f"ball in target wagon {int(target)}"
        return (
            f"ball landed in wagon {int(landed)} (gray distractor); "
            f"target was wagon {int(target)}, the red one"
        )

    def is_done(step):
        if env._interactive_released and post_release > 400:
            return True, _outcome_detail()
        return False

    run_viewer_loop(env, on_step, is_done=is_done, max_steps=20000)


if __name__ == "__main__":
    main()
    # household_task_gui convention: 0=SUCCESS, 10=FAILURE, 2=no result
    from _interactive_common import task_result_exit_code
    raise SystemExit(task_result_exit_code())
