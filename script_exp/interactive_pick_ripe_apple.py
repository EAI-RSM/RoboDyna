#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive sandbox for ``pick_ripe_apple``.

Wait for the red ripeness window, trigger the frozen front-pinch grasp, then
drop into the basket. Grasp / hang geometry is FROZEN — this script only
triggers existing ``play_once`` motion steps.

Run from any directory:

    /path/to/RoboDynaExp/script_exp/interactive_pick_ripe_apple.py --control keyboard
    /path/to/RoboDynaExp/script_exp/interactive_pick_ripe_apple.py --control robot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _interactive_common import (  # noqa: E402
    bootstrap_repo,
    configure_task,
    edge_pressed,
    print_banner,
    run_viewer_loop,
)

bootstrap_repo()


def _arm_for_good(env):
    from envs.utils.action import ArmTag

    return ArmTag("left" if env.apple_side < 0 else "right")


def _do_grasp(env):
    """Trigger the frozen front-grasp + clear/lift (no geometry retune)."""
    arm = _arm_for_good(env)
    target = max(0.18, env.red_window - env.GRASP_LEAD_STEPS / max(1, env.ripen_steps))
    if env._apple_attached and env.ripeness < target - 0.02:
        print(
            f"Too early: ripeness={env.ripeness:.3f} < trigger≈{target:.3f} "
            f"(red_window={env.red_window:.3f}). Wait for the apple to redden."
        )
        return False

    print(f"Grasping good apple at ripeness={env.ripeness:.3f} (red_window={env.red_window:.3f})…")
    env.move(env.open_gripper(arm))
    env.plan_success = True
    if not env._try_front_grasp(arm, grasp_dis=env.FRONT_GRASP_DIS, gripper_pos=0.0):
        print(f"Grasp failed (plan={env.plan_success}, "
              f"fail={getattr(env, '_last_plan_fail', None)}).")
        return False

    # Frozen clear-then-lift (same as play_once).
    clear_x = float(env.CLEAR_LATERAL * env.apple_side)
    env.move(env.move_by_displacement(arm_tag=arm, x=clear_x, move_axis="world"))
    env.move(env.move_by_displacement(arm_tag=arm, z=env.CLEAR_LIFT_Z, move_axis="world"))

    # Carry toward predicted basket (Opt2 keeps moving).
    ap = np.array(env.apple.get_pose().p)
    pred_x = env._predict_basket_x(env.DROP_PREDICT_TIME)
    hover_z = max(env.basket_top_z + 0.05, float(ap[2]))
    env.move(env.move_by_displacement(
        arm_tag=arm,
        x=float(pred_x - ap[0]),
        y=float(env.basket_y - ap[1]),
        z=float(hover_z - ap[2]),
        move_axis="world",
    ))
    env._interactive_arm = arm
    env._interactive_phase = "hold"
    print(
        f"Holding over basket (r_grasp={env.r_grasp}). "
        "Press Space again to drop when aligned."
    )
    return True


def _do_drop(env):
    arm = getattr(env, "_interactive_arm", None) or _arm_for_good(env)
    hold_x = float(env.apple.get_pose().p[0])
    if getattr(env, "basket_move_enabled", False):
        aligned = env._wait_basket_align(hold_x)
        print(f"Basket align={'ok' if aligned else 'timeout'}; opening gripper.")
    env.move(env.open_gripper(arm))
    for j in range(int(env.DROP_SETTLE_STEPS)):
        env._update_kinematic_tasks()
        env.scene.step()
    try:
        env.move(env.move_by_displacement(arm_tag=arm, z=0.08, move_axis="arm"))
    except Exception:
        pass
    env._interactive_phase = "done"
    print(
        f"Drop complete (r_grasp={env.r_grasp}, "
        f"ripeness_score={env._ripeness_score():.3f}); settling…"
    )
    return True


def main():
    parser = argparse.ArgumentParser(description="Interactive pick_ripe_apple viewer")
    parser.add_argument("--config", default="demo_dynamic")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--control",
        choices=("keyboard", "robot"),
        default="keyboard",
        help="Both modes trigger the same frozen grasp/drop motions via Space.",
    )
    parser.add_argument(
        "--robot-motion",
        choices=("planner", "interpolate"),
        default="planner",
        help="Robot motion backend (interpolate = faster joint interp when supported; default planner)",
    )
    args = parser.parse_args()

    from envs.pick_ripe_apple import pick_ripe_apple

    # Keyboard still invokes planned frozen grasp helpers — planning must be on.
    use_robot = True
    env = pick_ripe_apple()
    env.setup_demo(**configure_task(
        "pick_ripe_apple", args.config, args.seed, use_robot=use_robot,
    ))
    env._interactive_phase = "wait"  # wait → hold → done
    env._ripen_started = True

    print_banner(
        "pick_ripe_apple — interactive controls",
        [
            f"Mode: {args.control}  |  robot-motion: {args.robot_motion}  |  "
            f"config: {args.config}  |  seed: {args.seed}",
            "Goal: pinch the GOOD (red-path) apple near peak red; drop in the basket.",
            "      Do NOT pick the spoiled/yellow apple (Opt1).",
            "Space  — (1) grasp when red enough   (2) drop when over the basket",
            "V — toggle view: top-down ↔ head_camera",
            "Q / Esc — close the viewer window to quit",
            "Grasp / hang / clear geometry is FROZEN — Space only triggers existing motions.",
            "Watch the apple color: green → red → black. Act near vivid red.",
            "--robot-motion planner|interpolate",
        ],
    )
    if args.robot_motion == "interpolate":
        print(
            "Note: --robot-motion interpolate uses planner motions for this teleop task "
            "(key-press sandboxes use joint interpolation)."
        )
    print(
        f"Good side={'left' if env.apple_side < 0 else 'right'}  "
        f"red_window={env.red_window:.3f}  ripen_steps={env.ripen_steps}"
    )

    keys_prev: dict = {}
    done_since = None

    def on_step(window, step):
        nonlocal done_since
        if edge_pressed(window, "space", keys_prev):
            if env._interactive_phase == "wait":
                _do_grasp(env)
            elif env._interactive_phase == "hold":
                _do_drop(env)
                done_since = step
        if env._interactive_phase == "done" and done_since is None:
            done_since = step
        # Status heartbeat while waiting for red.
        if env._interactive_phase == "wait" and step % 150 == 0:
            target = max(
                0.18,
                env.red_window - env.GRASP_LEAD_STEPS / max(1, env.ripen_steps),
            )
            print(
                f"[ripeness] good={env.ripeness:.3f}  trigger≥{target:.3f}  "
                f"red_window={env.red_window:.3f}"
                + (
                    f"  spoiled={env.spoiled_ripeness:.3f}"
                    if getattr(env, "spoiled_apple", None) is not None
                    else ""
                )
            )

    def is_done(step):
        if done_since is not None and step - done_since > 80:
            return True, (
                f"r_grasp={env.r_grasp}, ripeness_score={env._ripeness_score():.3f}"
            )
        # Overripe with no grasp — definitive FAILURE.
        if (
            env._interactive_phase == "wait"
            and env._apple_attached
            and env.ripeness >= 0.95
        ):
            return True, "apple overripe (black) without a grasp"
        return False

    run_viewer_loop(env, on_step, is_done=is_done, max_steps=30000)


if __name__ == "__main__":
    main()
