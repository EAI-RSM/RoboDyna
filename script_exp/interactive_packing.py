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
    selected_robot_arms,
)

bootstrap_repo()


ARM_BASKET = {"left": "apple", "right": "orange"}


def _on_belt_by_side(env):
    """Oldest live fruit per belt (anywhere on the stream, not just at the station)."""
    on_belt = {}
    for idx in range(env.n_items):
        if (
            env._spawned_mask[idx]
            and not env._packed[idx]
            and not env._missed[idx]
            and not env._welded[idx]
            and env._item_y[idx] is not None
            and env._item_y[idx] >= env.pick_y_end
        ):
            on_belt.setdefault(env.item_sides[idx], idx)
    return on_belt


def _picks_for_selection(env, selection):
    """``(fruit index, arm)`` pairs for the current 1/2/3 selection.

    Starts as soon as fruit is on the belt so the arm can hover and wait while
    the stream keeps rolling — no freeze at the grasp. The selected arm owns
    the pick; its basket owns the drop (wrong arm → wrong basket → fail).
    """
    on_belt = _on_belt_by_side(env)
    if not on_belt:
        return None
    if selection == "both":
        return [(on_belt[side], side) for side in ("left", "right") if side in on_belt]
    other = "right" if selection == "left" else "left"
    for side in (selection, other):
        if side in on_belt:
            return [(on_belt[side], selection)]
    return None


def _keyboard_pack(env, picks):
    """Teleport each fruit into the selected arm's basket in one shot."""
    wrong = []
    for idx, arm_side in picks:
        basket = ARM_BASKET[str(arm_side)]
        ftype = env.item_types[idx]
        target = env._basket_target_xy(idx, basket=basket)
        env._set_fruit_pose(
            idx, float(target[0]), float(target[1]),
            float(env.basket_base_z[basket] + 0.05),
        )
        env._mark_packed(idx)
        if basket != ftype:
            wrong.append(idx)
    return wrong


def _robot_pack(env, picks):
    """One-shot intercept → grasp → drop into the selected arm's basket.

    The belt stays running the whole time: hover/wait uses ``_belt_dwell``, and
    arm moves call ``_update_kinematic_tasks`` every physics step. Dropping into
    the arm's own basket (not the fruit's color) makes a wrong-arm choice a
    permanent mis-pack.
    """
    from envs.utils.action import ArmTag

    if env._grasping_idxs:
        return None
    indices = [idx for idx, _arm in picks]
    env._belt_running = True
    env._begin_spawn_hold()
    env._grasping_idxs.update(indices)
    held = None

    def failed():
        for idx in indices:
            env._grasping_idxs.discard(idx)
        env._end_spawn_hold()
        return None

    try:
        if len(picks) == 1:
            idx, arm_side = picks[0]
            arm = ArmTag(arm_side)
            if not env._intercept_and_grasp(idx, arm, env.item_sides[idx]):
                return failed()
            held = [(idx, arm)]
        else:
            idx_l = next(idx for idx, arm_side in picks if arm_side == "left")
            idx_r = next(idx for idx, arm_side in picks if arm_side == "right")
            arm_l, arm_r = ArmTag("left"), ArmTag("right")
            env.plan_success = True
            env.move(env.open_gripper(arm_l), env.open_gripper(arm_r))
            pre_l = env._plan_station_pre(idx_l, arm_l)
            pre_r = env._plan_station_pre(idx_r, arm_r)
            if pre_l is None or pre_r is None:
                return failed()
            env.plan_success = True
            if env.move(env.move_to_pose(arm_l, pre_l),
                        env.move_to_pose(arm_r, pre_r)) is False:
                return failed()
            if not env._wait_pair_at_station(idx_l, idx_r):
                return failed()
            got_l, got_r = env._reach_and_attach_pair(idx_l, idx_r, arm_l, arm_r)
            if not (got_l and got_r):
                return failed()
            held = [(idx_l, arm_l), (idx_r, arm_r)]

        wrong, aborted = [], []
        for idx, arm in held:
            basket = ARM_BASKET[str(arm)]
            target = env._basket_target_xy(idx, basket=basket)
            if not env._carry_and_drop(idx, arm, target, resend_on_miss=False):
                aborted.append(idx)
            elif not env._fruit_in_basket(idx):
                wrong.append(idx)
        if aborted:
            names = ", ".join(f"{env.item_types[i]}_{i}" for i in aborted)
            print(f"Could not reach the basket with {names} — back on the belt.")
            return None
        return wrong
    finally:
        for idx in indices:
            env._grasping_idxs.discard(idx)
        env._end_spawn_hold()
        env._belt_running = True


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
            "Space — arm approaches while the belt keeps moving, then grasps and "
            "drops into that arm's basket in one shot",
            "V — toggle view: top-down ↔ head_camera",
            "Esc — close the viewer window to quit",
            "Pick apples with 1 and oranges with 2 — the wrong arm mis-packs "
            "into its own basket and fails the episode.",
            "--robot-motion planner|interpolate",
        ],
    )
    keys_prev: dict = {}
    settle_after_done = None
    selected = None
    pending_pack = False
    busy = False

    def on_step(window, step):
        nonlocal settle_after_done, selected, pending_pack, busy
        # Keep the stream rolling every viewer tick (also restored after packs).
        env._belt_running = True

        # Gripper highlighting belongs to the shared 1/2/3 controls; a second
        # recolor here would cache the already-tinted color and leave the
        # previously selected arm lit.
        arms = selected_robot_arms(env, fallback=("left",))
        new_selected = "both" if len(arms) == 2 else arms[0]
        if new_selected != selected:
            selected = new_selected

        if busy:
            return

        if edge_pressed(window, "space", keys_prev):
            pending_pack = True
            print(f"Armed: {selected} arm will pack the next fruit on the belt.")

        # Start as soon as fruit is on the belt so the arm can hover/wait while
        # the stream keeps advancing — no stop-and-go freeze at the grasp.
        if pending_pack and not env._grasping_idxs:
            picks = _picks_for_selection(env, selected)
            if picks is not None:
                busy = True
                pending_pack = False
                for idx, arm in picks:
                    ftype = env.item_types[idx]
                    basket = ARM_BASKET[str(arm)]
                    warn = "" if basket == ftype else f"  WRONG ARM → {basket} basket!"
                    print(f"Packing {ftype}_{idx} with the {arm} arm.{warn}")
                try:
                    if use_robot:
                        wrong = _robot_pack(env, picks)
                    else:
                        wrong = _keyboard_pack(env, picks)
                finally:
                    busy = False
                    env._belt_running = True
                if wrong is None:
                    print("Pack failed; press Space to try again.")
                    pending_pack = True
                elif wrong:
                    names = ", ".join(f"{env.item_types[i]}_{i}" for i in wrong)
                    print(f"Mis-packed {names} — episode will fail.")
                else:
                    print("Packed into the matching basket(s).")

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
