#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive sandbox for ``pick_ripe_apple``.

Arms stay at their original EE XYZ and only reorient to the frozen front
pre-grasp quat (horizontal pinch, parallel to the table). Teleop with arrows /
E / Q keeps that orientation; Space opens/closes the gripper to pinch and release.
No auto-grasp / auto-drop beyond Space gripper.

Run from any directory:

    /path/to/RoboDynaExp/interactive/base/interactive_pick_ripe_apple.py --control keyboard
    /path/to/RoboDynaExp/interactive/base/interactive_pick_ripe_apple.py --control robot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _interactive_common import (  # noqa: E402
    add_robot_motion_arg,
    bootstrap_repo,
    configure_task,
    gripper_width,
    print_banner,
    print_episode_condition,
    run_viewer_loop,
)

bootstrap_repo()


def _move_arms_to_pre_grasp_orientation(env) -> None:
    """Reorient both open grippers to the front pre-grasp quat at home XYZ.

    Keeps each arm at its original EE position — does not approach the tree or
    apples. Orientation matches ``_try_front_grasp`` (horizontal pinch, parallel
    to the table) so teleop translates without tipping unless F/G/R/T are used.
    """
    from envs._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
    from envs.utils.action import ArmTag

    env.together_open_gripper(save_freq=None)
    for side_name in ("left", "right"):
        arm = ArmTag(side_name)
        if side_name == "left":
            origin = np.asarray(env.robot.left_original_pose, dtype=np.float64)
        else:
            origin = np.asarray(env.robot.right_original_pose, dtype=np.float64)
        xyz = origin[:3].tolist()
        moved = False
        for key in env._front_grasp_quat_keys(arm):
            quat = list(GRASP_DIRECTION_DIC[key])
            pose = list(xyz) + quat
            env.plan_success = True
            env._last_plan_fail = None
            env.move(env.move_to_pose(arm, pose))
            if env.plan_success:
                print(f"{side_name} arm at home XYZ with front pre-grasp orientation ({key}).")
                moved = True
                break
        if not moved:
            detail = getattr(env, "_last_plan_fail", None) or "unreachable"
            print(f"Warning: {side_name} arm could not reach pre-grasp orientation ({detail}).")


class ApplePinchMonitor:
    """Detach a hanging apple when the matching arm's gripper pinches it."""

    # TCP↔center radius to count as a pinch (looser than expert GRASP_TCP_ERR_MAX).
    PINCH_DIST = 0.06

    def __init__(self, env):
        self.env = env
        self._gravity_at_step: dict[float, int] = {}
        self._announced: set[float] = set()

    def _still_attached(self, side: float) -> bool:
        if abs(float(side) - float(self.env.apple_side)) < 0.5:
            return bool(getattr(self.env, "_apple_attached", False))
        return bool(getattr(self.env, "_spoiled_attached", False))

    def update(self, step: int) -> None:
        env = self.env
        for side, at in list(self._gravity_at_step.items()):
            if step >= at:
                env._enable_held_apple_gravity(rigid=env._apple_rigids.get(side))
                del self._gravity_at_step[side]

        for side, apple in (getattr(env, "apples", {}) or {}).items():
            if not self._still_attached(side):
                continue
            arm_name = "left" if float(side) < 0 else "right"
            if gripper_width(env, arm_name) > 0.45:
                continue
            tcp = env._tcp_pos(arm_name)
            center = env._apple_grasp_center(apple)
            near = float(np.linalg.norm(tcp - center)) <= self.PINCH_DIST
            contacting = False
            try:
                contacting = len(
                    env.get_gripper_actor_contact_position(apple.get_name())
                ) > 0
            except Exception:
                pass
            if not (near or contacting):
                continue

            env._detach_apple(side=side)
            self._gravity_at_step[float(side)] = (
                step + int(getattr(env, "GRASP_SETTLE_STEPS", 25))
            )
            if float(side) in self._announced:
                continue
            self._announced.add(float(side))
            if abs(float(side) - float(env.apple_side)) < 0.5:
                tol = float(getattr(env, "red_tol", env.RED_TOLERANCE_DEFAULT))
                in_window = abs(float(env.ripeness) - float(env.red_window)) <= tol
                print(
                    f"Detached good apple at ripeness={env.ripeness:.3f} "
                    f"(window={env.red_window:.3f}±{tol:.3f}, "
                    f"{'in window' if in_window else 'OUTSIDE window'}). "
                    "Carry over the basket, then Space to open / release."
                )
            else:
                print(
                    f"Detached spoiled apple at ripeness={env.spoiled_ripeness:.3f} — "
                    "putting it in the basket fails. Space opens to release."
                )


def main():
    parser = argparse.ArgumentParser(description="Interactive pick_ripe_apple viewer")
    parser.add_argument("--config", default="demo_dynamic")
    parser.add_argument("--seed", type=int, default=0)
    add_robot_motion_arg(parser)
    args = parser.parse_args()

    from envs.pick_ripe_apple import pick_ripe_apple

    # Planning is required for the initial orientation park at home XYZ.
    env = pick_ripe_apple()
    env.setup_demo(**configure_task(
        "pick_ripe_apple", args.config, args.seed, use_robot=True,
    ))
    print_episode_condition(env)
    env._ripen_started = True

    _move_arms_to_pre_grasp_orientation(env)

    print_banner(
        "pick_ripe_apple — interactive controls",
        [
            f"Mode: {args.control}  |  robot-motion: {args.robot_motion}  |  "
            f"config: {args.config}  |  seed: {args.seed}",
            "Goal: pinch the GOOD (red-path) apple near peak red; drop in the basket.",
            "      Do NOT pick the spoiled/yellow apple (Opt1).",
            "Arms keep home XYZ with front pre-grasp orientation (gripper level).",
            "1 / 2 / 3 — select left / right / both arms",
            "Space — close to pinch / open to release into the basket",
            "Arrows / E / Q — teleop selected arm(s); orientation stays level",
            "V — cycle view: head_camera ↔ gripper(s)",
            "Esc — close the viewer window to quit",
            "Success needs BOTH: grasp inside the ripeness window AND apple in the basket.",
            "--robot-motion planner|interpolate",
        ],
    )
    if args.robot_motion == "interpolate":
        print(
            "Note: --robot-motion interpolate uses planner motions for the initial "
            "orientation park; teleop still uses seeded IK."
        )
    n_apples = len(getattr(env, "apples", {}) or {})
    print(
        f"Good side={'left' if env.apple_side < 0 else 'right'}  "
        f"apples={n_apples}"
        + (
            f"  spoiled_side={'left' if env.spoiled_side < 0 else 'right'}"
            if n_apples > 1 else ""
        )
        + f"  red_window={env.red_window:.3f}"
        f"±{float(getattr(env, 'red_tol', env.RED_TOLERANCE_DEFAULT)):.3f}  "
        f"ripen_steps={env.ripen_steps}"
    )

    pinch = ApplePinchMonitor(env)
    in_basket_since = None
    on_table_since = None

    def on_step(window, step):
        nonlocal in_basket_since, on_table_since
        pinch.update(step)

        # Heartbeat while the good apple is still on the tree.
        if getattr(env, "_apple_attached", False) and step % 150 == 0:
            tol = float(getattr(env, "red_tol", env.RED_TOLERANCE_DEFAULT))
            in_window = abs(float(env.ripeness) - float(env.red_window)) <= tol
            print(
                f"[ripeness] good={env.ripeness:.3f}  "
                f"window={env.red_window:.3f}±{tol:.3f}  "
                f"{'IN window' if in_window else 'outside window'}"
                + (
                    f"  spoiled={env.spoiled_ripeness:.3f}"
                    if getattr(env, "spoiled_apple", None) is not None
                    else ""
                )
            )

        bp = np.array(env.basket.get_pose().p)
        basket_xy = np.array([bp[0], bp[1]], dtype=np.float64)
        good_in = bool(
            env.r_grasp is not None
            and env._pose_in_basket(np.array(env.apple.get_pose().p), basket_xy)
        )
        spoiled = getattr(env, "spoiled_apple", None)
        spoiled_in = bool(
            spoiled is not None
            and env._pose_in_basket(np.array(spoiled.get_pose().p), basket_xy)
        )
        if good_in or spoiled_in:
            if in_basket_since is None:
                in_basket_since = step
            on_table_since = None
        else:
            in_basket_since = None

        if env._good_apple_dropped_on_table():
            if on_table_since is None:
                on_table_since = step
                print("Good apple on table (missed basket) — terminating…")
        else:
            on_table_since = None

    def is_done(step):
        settle = max(1, int(getattr(env, "DROP_SETTLE_STEPS", 80)))
        if on_table_since is not None and step - on_table_since >= settle:
            return True, "good apple dropped on table (missed basket)"
        if in_basket_since is not None and step - in_basket_since >= settle:
            return True, (
                f"r_grasp={env.r_grasp}, window={env.red_window:.3f}"
                f"±{float(getattr(env, 'red_tol', env.RED_TOLERANCE_DEFAULT)):.3f}, "
                f"ripeness_ok={env._grasp_in_red_window()}, "
                f"ripeness_score={env._ripeness_score():.3f}"
            )
        if (
            getattr(env, "_apple_attached", False)
            and float(env.ripeness) >= 0.95
        ):
            return True, "apple overripe (black) without a grasp"
        return False

    run_viewer_loop(env, on_step, is_done=is_done, max_steps=30000)


if __name__ == "__main__":
    main()
    # household_task_gui convention: 0=SUCCESS, 10=FAILURE, 2=no result
    from _interactive_common import task_result_exit_code
    raise SystemExit(task_result_exit_code())
