#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive sandbox for ``pack_fruits``.

Pack red apples into the left basket and green apples into the right
(when both colors are present). Opt2 black distractors are never packed.
Colored apples may appear on either belt in every scenario.

Physical grasp (same pattern as pick_ripe_apple — no teleport / no EE weld):
  Teleop over a belt apple → Space closes the gripper. The apple keeps riding
  the belt until a real pinch is confirmed, then frees in place for a friction
  hold → lift with E → Space opens → falls under gravity.

Run from any directory:

    /path/to/RoboDynaExp/script_exp/interactive_pack_fruits.py --control keyboard
    /path/to/RoboDynaExp/script_exp/interactive_pack_fruits.py --control robot
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
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


# Pinch window (looser than the expert attach thresholds).
PINCH_XY = 0.055
PINCH_Z_MAX = 0.08
GRASP_SETTLE_STEPS = 28
DROP_SETTLE_STEPS = 90
CLOSE_WIDTH = 0.45
OPEN_WIDTH = 0.55


def _tcp_fruit_gap(env, idx, arm_name: str):
    tcp = env._tcp_pos(arm_name)
    fp = np.array(env.items[idx].get_pose().p, dtype=float)
    xy = float(np.linalg.norm(fp[:2] - tcp[:2]))
    dz = float(tcp[2] - fp[2])
    return xy, dz


def _nearest_graspable(env, arm_name: str):
    """Closest belt apple under the gripper (never black distractors / held)."""
    best = None
    best_score = 1e9
    for idx in range(env.n_items):
        if (
            not env._spawned_mask[idx]
            or env._packed[idx]
            or env._missed[idx]
            or env._welded[idx]
            or idx in env._grasping_idxs
            or env._item_y[idx] is None
        ):
            continue
        xy, dz = _tcp_fruit_gap(env, idx, arm_name)
        if xy > PINCH_XY or dz < -0.02 or dz > PINCH_Z_MAX:
            continue
        score = xy + 0.5 * max(0.0, dz)
        if score < best_score:
            best_score = score
            best = idx
    return best


def _near_fruit(env, idx, arm_name: str) -> bool:
    xy, dz = _tcp_fruit_gap(env, idx, arm_name)
    return xy <= PINCH_XY and -0.02 <= dz <= PINCH_Z_MAX


class FruitPinchMonitor:
    """Detach a belt apple only when a real pinch is confirmed.

    Closing early near an apple does **not** pull it off the belt — it keeps
    riding at its current lateral pose. Only confirmed contact/proximity frees
    it for a friction hold (stacking/drop unchanged).
    """

    def __init__(self, env):
        self.env = env
        self._prev_width = {"left": 1.0, "right": 1.0}
        self._held = {}  # arm_name -> fruit idx
        # arm -> (idx, step_when_fingers_should_be_closed)
        self._pending = {}
        self._settle = None  # (idx, arm, steps_left) after release
        self._announced = set()

    def _try_pinch(self, arm_name: str, step: int) -> None:
        if arm_name in self._held or arm_name in self._pending:
            return
        idx = _nearest_graspable(self.env, arm_name)
        if idx is None:
            return
        # Stay on the kinematic belt stream while the jaws close.
        self._pending[arm_name] = (idx, step + GRASP_SETTLE_STEPS)

    def _confirm_or_abort_pinch(self, arm_name: str, idx: int) -> None:
        held = self.env._fruit_held_by_gripper(idx)
        near = _near_fruit(self.env, idx, arm_name)
        closed = gripper_width(self.env, arm_name) <= CLOSE_WIDTH
        on_belt = self.env._item_y[idx] is not None

        if closed and (held or near):
            # Real pinch — only now leave the belt, at the apple's current pose.
            self.env._free_fruit_for_physical_grasp(idx)
            self.env._grasping_idxs.add(idx)
            self.env._enable_fruit_gravity(idx)
            self._held[arm_name] = idx
            if idx not in self._announced:
                self._announced.add(idx)
                ftype = self.env.item_types[idx]
                print(
                    f"Pinched {ftype}_{idx} with {arm_name}. "
                    f"Lift with E over the "
                    f"{'red' if ftype == 'apple' else 'green'} basket, "
                    f"Space to release."
                )
            return

        # Missed / early close: apple never left the belt — keep riding.
        if on_belt:
            return
        self.env._reseat_on_belt(idx)

    def _begin_release(self, arm_name: str) -> None:
        idx = self._held.pop(arm_name, None)
        if idx is None:
            return
        self._pending.pop(arm_name, None)
        self.env._calm_fruit(idx, damping=(2.5, 12.0))
        self.env._enable_fruit_gravity(idx)
        self._settle = (idx, arm_name, DROP_SETTLE_STEPS)
        print(f"Released fruit_{idx} — dropping under gravity…")

    def _tick_settle(self) -> None:
        if self._settle is None:
            return
        idx, arm_name, left = self._settle
        if left > DROP_SETTLE_STEPS - 12:
            self.env._calm_fruit(idx, damping=(3.0, 14.0))
        left -= 1
        if left > 0:
            self._settle = (idx, arm_name, left)
            return
        self._settle = None
        self.env._grasping_idxs.discard(idx)

        if self.env._fruit_held_by_gripper(idx) or _near_fruit(self.env, idx, arm_name):
            if gripper_width(self.env, arm_name) <= CLOSE_WIDTH:
                self._held[arm_name] = idx
                self.env._grasping_idxs.add(idx)
                print(f"fruit_{idx} still in the jaws — open Space fully to drop.")
                return

        if self.env._fruit_in_basket(idx):
            self.env._mark_packed(idx, freeze=False)
            print(f"Packed {self.env.item_types[idx]}_{idx} into the matching basket.")
            return

        p = np.array(self.env.items[idx].get_pose().p, dtype=float)
        in_any = any(
            self.env._xy_inside_basket(p[:2], btype) for btype in self.env.baskets
        )
        if in_any:
            self.env._mark_packed(idx, freeze=False)
            print(f"{self.env.item_types[idx]}_{idx} landed in the wrong basket.")
            return

        if self.env._fruit_over_belt(idx) is not None:
            self.env._reseat_on_belt(idx)
            print(f"{self.env.item_types[idx]}_{idx} back on the belt — keep packing.")
            return

        self.env._mark_table_rest(idx)
        print(f"{self.env.item_types[idx]}_{idx} left on the table.")

    def update(self, step: int) -> None:
        env = self.env
        self._tick_settle()

        for arm, (idx, at) in list(self._pending.items()):
            if step < at:
                continue
            del self._pending[arm]
            if arm in self._held:
                continue
            self._confirm_or_abort_pinch(arm, idx)

        for arm in ("left", "right"):
            w = gripper_width(env, arm)
            prev = self._prev_width[arm]
            closing = prev > CLOSE_WIDTH and w <= CLOSE_WIDTH
            opening = prev <= CLOSE_WIDTH and w > OPEN_WIDTH
            self._prev_width[arm] = w

            if self._settle is not None:
                continue

            if closing:
                self._try_pinch(arm, step)
            elif opening and arm in self._held:
                self._begin_release(arm)
            elif opening:
                self._pending.pop(arm, None)


def main():
    parser = argparse.ArgumentParser(description="Interactive pack_fruits viewer")
    parser.add_argument("--config", default="demo_dynamic")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--scenario",
        choices=("default", "opt1", "opt2", "opt1+2"),
        default=None,
        help="Apply GUI scenario flags (opt1 = red+green with both baskets)",
    )
    add_robot_motion_arg(parser)
    args = parser.parse_args()

    from envs.pack_fruits import pack_fruits

    scenario_overrides = {
        "default": {"two_colors_enabled": False, "distractor_enabled": False},
        "opt1": {"two_colors_enabled": True, "distractor_enabled": False},
        "opt2": {"two_colors_enabled": False, "distractor_enabled": True},
        "opt1+2": {"two_colors_enabled": True, "distractor_enabled": True},
    }
    scenario = args.scenario or os.environ.get("ROBODYNA_SCENARIO") or None
    overrides = scenario_overrides.get(scenario) if scenario else None

    use_robot = args.control == "robot"
    config = configure_task(
        "pack_fruits", args.config, args.seed, use_robot=use_robot,
        task_arg_overrides=overrides,
    )
    # Prefer CLI/env scenario; fall back to top-level key from GUI temp yml.
    scenario = scenario or config.get("interactive_scenario")
    if scenario in scenario_overrides:
        config.setdefault("task_args", {}).setdefault("pack_fruits", {}).update(
            scenario_overrides[scenario]
        )
        config["interactive_scenario"] = scenario
        config["scenario"] = scenario
    print(
        f"[interactive_pack_fruits] scenario={scenario!r} "
        f"two_colors={config.get('task_args', {}).get('pack_fruits', {}).get('two_colors_enabled')}",
        flush=True,
    )
    env = pack_fruits()
    env.setup_demo(**config)
    print_episode_condition(env)
    env._belt_running = True

    if env.two_colors_enabled:
        # Opt1 / Opt1+2: red→left basket, green→right — both arms needed.
        env._interactive_selected_arms = ("left", "right")
    else:
        env._interactive_selected_arms = (
            ("left",) if env.active_colors[0] == "apple" else ("right",)
        )

    for side in ("left", "right"):
        try:
            env.robot.set_gripper(1.0, side, gripper_eps=0.0)
        except Exception:
            pass

    if env.two_colors_enabled:
        goal = (
            f"Opt1-style: {env.n_apple} red → left basket, "
            f"{env.n_green} green → right basket (one apple at a time)."
        )
    else:
        color = "red" if env.active_colors[0] == "apple" else "green"
        side = "left" if color == "red" else "right"
        goal = f"Default/Opt2: {env.n_items} {color} apples → {side} basket."

    print_banner(
        "pack_fruits — interactive controls",
        [
            f"Mode: {args.control}  |  robot-motion: {args.robot_motion}  |  "
            f"config: {args.config}  |  seed: {args.seed}"
            + (f"  |  scenario: {args.scenario}" if args.scenario else ""),
            goal,
            "Never pack black distractors (Opt2).",
            "1 / 2 / 3 — select left / right / both arms",
            "Arrows / E / Q — teleop the selected arm(s)",
            "Space — close to pinch; apple keeps moving until a real grasp",
            "V — cycle view: head_camera ↔ gripper(s)",
            "Esc — close the viewer window to quit",
            "--scenario default|opt1|opt2|opt1+2",
            "--robot-motion planner|interpolate",
        ],
    )

    settle_after_done = None
    pinch = FruitPinchMonitor(env) if use_robot else None

    def on_step(window, step):
        nonlocal settle_after_done
        env._belt_running = True

        if pinch is not None:
            pinch.update(step)

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
    from _interactive_common import task_result_exit_code
    raise SystemExit(task_result_exit_code())
