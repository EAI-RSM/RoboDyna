#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive sandbox for ``packing``.

Dual belts: pack apples into the left basket and oranges into the right.
Opt2 black distractors are never packed.

Run from any directory:

    /path/to/RoboDynaExp/script_exp/interactive_packing.py --control keyboard
    /path/to/RoboDynaExp/script_exp/interactive_packing.py --control robot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _interactive_common import (  # noqa: E402
    add_robot_motion_arg,
    bootstrap_repo,
    configure_task,
    edge_pressed,
    print_banner,
    run_viewer_loop,
)

bootstrap_repo()


def _keyboard_pack_idx(env, idx: int):
    """Teleport a ready fruit into its color-matched basket (no planner)."""
    if idx is None:
        return False
    if env._packed[idx] or env._missed[idx] or env._item_y[idx] is None:
        return False
    ftype = env.item_types[idx]
    target = env._basket_target_xy(idx)
    z = float(env.basket_base_z[ftype] + 0.05)
    env._set_fruit_pose(idx, float(target[0]), float(target[1]), z)
    rigid = env._item_comps[idx]
    if rigid is not None:
        try:
            rigid.set_kinematic(True)
            rigid.set_disable_gravity(True)
            rigid.set_linear_velocity([0, 0, 0])
            rigid.set_angular_velocity([0, 0, 0])
        except Exception:
            pass
    env._mark_packed(idx)
    print(f"Packed {ftype}_{idx} → {ftype} basket (keyboard teleport).")
    return True


def _ready_for_arm(env, arm: str):
    """Oldest ready fruit assigned to this arm, regardless of belt side."""
    latest_start = env.pick_station_y + 0.08
    for idx in range(env.n_items):
        if (
            env._spawned_mask[idx]
            and not env._packed[idx]
            and not env._missed[idx]
            and not env._welded[idx]
            and env._item_y[idx] is not None
            and env.item_arms[idx] == arm
            and latest_start <= env._item_y[idx] <= env.pick_y
        ):
            return idx
    return None


def _dispatch_arm(env, arm: str, use_robot: bool) -> bool:
    """Pack one ready fruit with its color-assigned arm."""
    if env._grasping_idxs:
        print("Arm busy — wait for the current pack to finish.")
        return False
    idx = _ready_for_arm(env, arm)
    if idx is None:
        print(f"No ready fruit for the {arm} arm (in the pick window).")
        return False

    print(f"Packing {env.item_types[idx]}_{idx} with {arm} arm…")
    if use_robot:
        env._pack_item(idx)
    else:
        _keyboard_pack_idx(env, idx)
    return True


def _dispatch_both(env, use_robot: bool) -> bool:
    """Pack a ready apple/orange pair with left and right arms simultaneously."""
    if env._grasping_idxs:
        print("Arms busy — wait for the current pack to finish.")
        return False
    idx_l = _ready_for_arm(env, "left")
    idx_r = _ready_for_arm(env, "right")
    if idx_l is None or idx_r is None:
        return False
    print(f"Packing both: apple_{idx_l} (left) + orange_{idx_r} (right)…")
    if use_robot:
        env._pack_pair(idx_l, idx_r)
    else:
        _keyboard_pack_idx(env, idx_l)
        _keyboard_pack_idx(env, idx_r)
    return True


def main():
    parser = argparse.ArgumentParser(description="Interactive packing viewer")
    parser.add_argument("--config", default="demo_dynamic")
    parser.add_argument("--seed", type=int, default=0)
    add_robot_motion_arg(parser)
    args = parser.parse_args()

    from envs.packing import packing

    use_robot = args.control == "robot"
    env = packing()
    env.setup_demo(**configure_task("packing", args.config, args.seed, use_robot=use_robot))
    env._belt_running = True

    print_banner(
        "packing — interactive controls",
        [
            f"Mode: {args.control}  |  robot-motion: {args.robot_motion}  |  "
            f"config: {args.config}  |  seed: {args.seed}",
            "Goal: apple → left basket, orange → right basket. Never pack black (Opt2).",
            "Left Arrow  — left arm packs a ready red apple",
            "Right Arrow — right arm packs a ready orange",
            "Up Arrow    — both arms pack a ready apple + orange together",
            "V — toggle view: top-down ↔ head_camera",
            "Q / Esc   — close the viewer window to quit",
            "Keyboard: teleports ready fruit into the matching basket.",
            "Robot: queues an arm intercept → carry → drop path.",
            "Press a key any time; it waits for its required fruit(s).",
            "--robot-motion planner|interpolate",
        ],
    )
    if args.robot_motion == "interpolate":
        print(
            "Note: --robot-motion interpolate uses planner motions for this teleop task "
            "(key-press sandboxes use joint interpolation)."
        )

    keys_prev: dict = {}
    settle_after_done = None
    pending_mode = None

    def on_step(window, step):
        nonlocal settle_after_done, pending_mode
        # Start belts every step (safe if already True).
        env._belt_running = True

        if edge_pressed(window, "left", keys_prev):
            pending_mode = "left"
            print("Queued left-arm apple pack action.")
        if edge_pressed(window, "right", keys_prev):
            pending_mode = "right"
            print("Queued right-arm orange pack action.")
        if edge_pressed(window, "up", keys_prev):
            pending_mode = "both"
            print("Queued two-arm apple + orange pack action.")

        # Retain the user's request until its fruit(s) enter the short live
        # pick window, rather than silently dropping an early key press.
        if pending_mode is not None and not env._grasping_idxs:
            if pending_mode == "both":
                if (
                    _ready_for_arm(env, "left") is not None
                    and _ready_for_arm(env, "right") is not None
                    and _dispatch_both(env, use_robot)
                ):
                    pending_mode = None
            elif (
                _ready_for_arm(env, pending_mode) is not None
                and _dispatch_arm(env, pending_mode, use_robot)
            ):
                pending_mode = None

        if settle_after_done is None:
            all_done = (
                env._spawned >= env.n_items
                and all(env._packed[i] or env._missed[i] for i in range(env.n_items))
            )
            if all_done:
                settle_after_done = step
                print("All fruits resolved — settling…")

    def is_done(step):
        if settle_after_done is not None and step - settle_after_done > 120:
            n_ok = sum(1 for i in range(env.n_items) if env._fruit_in_basket(i))
            return True, f"{n_ok}/{env.n_items} in correct baskets"
        return False

    run_viewer_loop(env, on_step, is_done=is_done, max_steps=40000)


if __name__ == "__main__":
    main()
