#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive sandbox for ``pack_fruits``.

Pack red apples into the left basket and green apples into the right
(when both colors are present). Opt2 black distractors are never packed.

Keyboard+mouse: click a fruit (highlights lighter), then click a basket.
Robot: physical pinch grasp (no teleport / no EE weld).

Run from any directory:

    /path/to/RoboDynaExp/interactive/base/interactive_pack_fruits.py --control keyboard
    /path/to/RoboDynaExp/interactive/base/interactive_pack_fruits.py --control robot
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _interactive_common import (  # noqa: E402
    actor_scene_id,
    add_robot_motion_arg,
    bootstrap_repo,
    click_hits_actor_map,
    configure_task,
    gripper_width,
    is_robot_control,
    prepare_interactive_control,
    print_banner,
    print_episode_condition,
    print_instructions,
    run_viewer_loop,
    table_xy_from_click,
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


def _lighter_rgb(rgb):
    return [min(1.0, 0.45 * float(c) + 0.55) for c in list(rgb)[:3]]


class FruitClickController:
    """Click fruit (highlight) then click a basket to pack it there."""

    def __init__(self, env, viewer):
        self.env = env
        self.viewer = viewer
        self.selected = None
        self._dropping = {}  # idx -> steps remaining
        self._basket_ids = {}
        for ftype, basket in (getattr(env, "baskets", {}) or {}).items():
            sid = actor_scene_id(basket)
            if sid is not None:
                self._basket_ids[int(sid)] = str(ftype)

    def _fruit_ids(self):
        ids = {}
        for idx in range(self.env.n_items):
            if (
                not self.env._spawned_mask[idx]
                or self.env._packed[idx]
                or self.env._missed[idx]
                or idx in self._dropping
            ):
                continue
            sid = actor_scene_id(self.env.items[idx])
            if sid is not None:
                ids[int(sid)] = int(idx)
        return ids

    def _restore_color(self, idx):
        ftype = self.env.item_types[idx]
        self.env._recolor(self.env.items[idx], self.env._type_rgb(ftype))

    def _highlight(self, idx):
        ftype = self.env.item_types[idx]
        self.env._recolor(self.env.items[idx], _lighter_rgb(self.env._type_rgb(ftype)))

    def _select(self, idx):
        if self.selected is not None and self.selected != idx:
            if not self.env._packed[self.selected] and not self.env._missed[self.selected]:
                self._restore_color(self.selected)
        self.selected = int(idx)
        self._highlight(idx)
        ftype = self.env.item_types[idx]
        print(f"Selected {ftype}_{idx} — click a basket to pack it.")

    def _drop_xy(self, basket_type: str, pixel_x, pixel_y):
        """World XY over the basket mouth (click if inside, else center)."""
        env = self.env
        c = np.array(env.basket_centers[basket_type], dtype=float)
        half_x, half_y = env.basket_half_xy.get(basket_type, env.BASKET_HALF_XY)
        r = float(env.fruit_r)
        hit = table_xy_from_click(
            self.viewer, pixel_x, pixel_y,
            float(env.basket_top_z[basket_type]),
        )
        if hit is not None:
            x, y = float(hit[0]), float(hit[1])
        else:
            x, y = float(c[0]), float(c[1])
        x = float(np.clip(x, c[0] - half_x + r, c[0] + half_x - r))
        y = float(np.clip(y, c[1] - half_y + r, c[1] + half_y - r))
        return x, y

    def _pack_into(self, basket_type: str, pixel_x, pixel_y):
        idx = self.selected
        if idx is None:
            print("Select a fruit first.")
            return
        if self.env._packed[idx] or self.env._missed[idx] or idx in self._dropping:
            self.selected = None
            return
        x, y = self._drop_xy(basket_type, pixel_x, pixel_y)
        # Release above the rim so the apple falls in; do not snap to a seated pose.
        z = float(self.env.basket_top_z[basket_type]) + float(self.env.fruit_r) + 0.04
        self.env._item_y[idx] = None
        self.env._grasping_idxs.add(idx)  # keep orphan cleanup from reseating mid-drop
        self.env._set_fruit_collision_enabled(idx, True)
        self.env._set_fruit_pose(idx, x, y, z)
        self.env._enable_fruit_gravity(idx)
        self.env._calm_fruit(idx, damping=(4.0, 16.0))
        self._restore_color(idx)
        self._dropping[idx] = int(DROP_SETTLE_STEPS)
        want = self.env.item_types[idx]
        print(
            f"Dropping {want}_{idx} into the {basket_type} basket…"
            + ("" if basket_type == want else " (wrong color).")
        )
        self.selected = None

    def update(self) -> None:
        for idx in list(self._dropping):
            left = self._dropping[idx] - 1
            if left > DROP_SETTLE_STEPS - 12:
                self.env._calm_fruit(idx, damping=(4.0, 16.0))
            if left > 0:
                self._dropping[idx] = left
                continue
            del self._dropping[idx]
            self.env._grasping_idxs.discard(idx)
            if self.env._fruit_in_basket(idx):
                self.env._mark_packed(idx, freeze=False)
                print(f"Packed {self.env.item_types[idx]}_{idx} into the matching basket.")
                continue
            p = np.array(self.env.items[idx].get_pose().p, dtype=float)
            in_any = any(
                self.env._xy_inside_basket(p[:2], btype) for btype in self.env.baskets
            )
            if in_any:
                self.env._mark_packed(idx, freeze=False)
                print(f"{self.env.item_types[idx]}_{idx} landed in the wrong basket.")
                continue
            if self.env._fruit_over_belt(idx) is not None:
                self.env._reseat_on_belt(idx)
                print(f"{self.env.item_types[idx]}_{idx} missed — back on the belt.")
                continue
            self.env._mark_table_rest(idx)
            print(f"{self.env.item_types[idx]}_{idx} left on the table.")

    def on_click(self, viewer, pixel_x, pixel_y):
        fruit_hit = click_hits_actor_map(
            viewer, pixel_x, pixel_y, self._fruit_ids()
        )
        if fruit_hit is not None:
            self._select(int(fruit_hit))
            return True
        basket_hit = click_hits_actor_map(
            viewer, pixel_x, pixel_y, self._basket_ids
        )
        if basket_hit is not None:
            self._pack_into(str(basket_hit), pixel_x, pixel_y)
            return True
        return False


class FruitPinchMonitor:
    """Detach a belt apple only when a real pinch is confirmed."""

    def __init__(self, env):
        self.env = env
        self._prev_width = {"left": 1.0, "right": 1.0}
        self._held = {}  # arm_name -> fruit idx
        self._pending = {}
        self._settle = None
        self._announced = set()

    def _try_pinch(self, arm_name: str, step: int) -> None:
        if arm_name in self._held or arm_name in self._pending:
            return
        idx = _nearest_graspable(self.env, arm_name)
        if idx is None:
            return
        self._pending[arm_name] = (idx, step + GRASP_SETTLE_STEPS)

    def _confirm_or_abort_pinch(self, arm_name: str, idx: int) -> None:
        held = self.env._fruit_held_by_gripper(idx)
        near = _near_fruit(self.env, idx, arm_name)
        closed = gripper_width(self.env, arm_name) <= CLOSE_WIDTH
        on_belt = self.env._item_y[idx] is not None

        if closed and (held or near):
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

    use_robot = is_robot_control(args.control)
    config = configure_task(
        "pack_fruits", args.config, args.seed, use_robot=use_robot,
        task_arg_overrides=overrides,
    )
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
    prepare_interactive_control(env, args.control)
    print_episode_condition(env)
    env._belt_running = True

    if use_robot:
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

    if use_robot:
        print_banner(
            "pack_fruits — interactive controls",
            [
                f"Mode: {args.control}  |  robot-motion: {args.robot_motion}  |  "
                f"config: {args.config}  |  seed: {args.seed}"
                + (f"  |  scenario: {args.scenario}" if args.scenario else ""),
                goal,
                "Never pack black distractors (Opt2).",
                "1 / 2 / 3 — select left / right / both arms (selected gripper turns green)",
                "Arrows / E / Q — teleop the selected arm(s)",
                "Space — close to pinch; apple keeps moving until a real grasp",
                "V — cycle view: head_camera ↔ gripper(s)",
                "Esc — close the viewer window to quit",
                "--scenario default|opt1|opt2|opt1+2",
                "--robot-motion planner|interpolate",
            ],
        )
    else:
        print_banner(
            "pack_fruits — keyboard+mouse",
            [
                f"Mode: {args.control}  |  config: {args.config}  |  seed: {args.seed}"
                + (f"  |  scenario: {args.scenario}" if args.scenario else ""),
                goal,
                "Click a fruit to select (highlights lighter), then click a basket to drop it in.",
                "Never pack black distractors (Opt2).",
                "Esc — quit",
            ],
        )

    settle_after_done = None
    pinch = FruitPinchMonitor(env) if use_robot else None
    clicker = None
    if not use_robot:
        viewer = env.viewer
        if viewer is None:
            raise SystemExit("Viewer was not created; ensure a graphical display is available.")
        clicker = FruitClickController(env, viewer)
        viewer.register_click_handler(clicker.on_click)
        print_instructions("Click a fruit, then click a basket to drop it in.")

    def on_step(window, step):
        nonlocal settle_after_done
        del window
        env._belt_running = True

        if pinch is not None:
            pinch.update(step)
        if clicker is not None:
            clicker.update()

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
