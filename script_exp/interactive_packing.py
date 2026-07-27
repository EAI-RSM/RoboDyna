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

import sapien

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


_GRIPPER_LINK_NAMES = (
    "wsg_50_base_link",
    "gripper_left",
    "gripper_right",
    "finger_left",
    "finger_right",
)
_ARM_HIGHLIGHT = {
    "left": [1.0, 0.85, 0.10, 1.0],   # yellow
    "right": [0.15, 0.75, 1.0, 1.0],  # cyan
}


class ArmGripperHighlight:
    """Recolor selected gripper meshes, matching interactive_catch_rat."""

    def __init__(self, env):
        self._orig = {}
        robot = env.robot
        self._entities = {
            "left": self._gripper_entities(robot.left_entity),
            "right": self._gripper_entities(robot.right_entity),
        }

    @staticmethod
    def _gripper_entities(articulation):
        return [
            link.entity for link in articulation.get_links()
            if link.get_name() in _GRIPPER_LINK_NAMES
        ]

    @staticmethod
    def _iter_materials(entity):
        for component in entity.get_components():
            if not isinstance(component, sapien.render.RenderBodyComponent):
                continue
            for shape in component.render_shapes:
                try:
                    yield shape.material
                except Exception:
                    continue

    def _remember(self, material):
        if id(material) in self._orig:
            return
        try:
            color = list(material.base_color)
        except Exception:
            color = [0.75, 0.75, 0.75, 1.0]
        self._orig[id(material)] = (material, color)

    def clear(self):
        for material, color in self._orig.values():
            try:
                material.set_base_color(color)
                material.base_color = color
            except Exception:
                pass

    def set_selected(self, selection):
        self.clear()
        sides = ("left", "right") if selection == "both" else (selection,)
        for side in sides:
            if side not in _ARM_HIGHLIGHT:
                continue
            for entity in self._entities[side]:
                for material in self._iter_materials(entity):
                    self._remember(material)
                    try:
                        material.set_base_color_texture(None)
                    except Exception:
                        pass
                    try:
                        material.set_base_color(_ARM_HIGHLIGHT[side])
                        material.base_color = _ARM_HIGHLIGHT[side]
                    except Exception:
                        pass


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


def _indices_for_selection(env, selection: str):
    arms = ("left", "right") if selection == "both" else (selection,)
    indices = [_ready_for_arm(env, arm) for arm in arms]
    return None if any(idx is None for idx in indices) else indices


def _keyboard_pick(env, indices):
    """Freeze selected fruits in a virtual gripper until the next Space press."""
    for idx in indices:
        rigid = env._item_comps[idx]
        if rigid is not None:
            try:
                rigid.set_kinematic(True)
                rigid.set_disable_gravity(True)
                rigid.set_linear_velocity([0, 0, 0])
                rigid.set_angular_velocity([0, 0, 0])
            except Exception:
                pass
        env._item_y[idx] = None
    return [(idx, None) for idx in indices]


def _keyboard_drop(env, held):
    for idx, _arm in held:
        ftype = env.item_types[idx]
        target = env._basket_target_xy(idx)
        env._set_fruit_pose(idx, float(target[0]), float(target[1]), float(env.basket_base_z[ftype] + 0.05))
        env._mark_packed(idx)


def _robot_pick(env, indices):
    """Use the packing task's normal intercept/grasp stage, but defer the drop."""
    from envs.utils.action import ArmTag

    if env._grasping_idxs:
        return None
    env._begin_spawn_hold()
    env._grasping_idxs.update(indices)

    def failed():
        for idx in indices:
            env._grasping_idxs.discard(idx)
        env._end_spawn_hold()
        return None

    if len(indices) == 1:
        idx = indices[0]
        arm = ArmTag(env.item_arms[idx])
        if not env._intercept_and_grasp(idx, arm, env.item_sides[idx]):
            return failed()
        return [(idx, arm)]

    idx_l = next(idx for idx in indices if env.item_arms[idx] == "left")
    idx_r = next(idx for idx in indices if env.item_arms[idx] == "right")
    arm_l, arm_r = ArmTag("left"), ArmTag("right")
    env.plan_success = True
    env.move(env.open_gripper(arm_l), env.open_gripper(arm_r))
    pre_l = env._plan_station_pre(idx_l, arm_l)
    pre_r = env._plan_station_pre(idx_r, arm_r)
    if pre_l is None or pre_r is None:
        return failed()
    env.plan_success = True
    if env.move(env.move_to_pose(arm_l, pre_l), env.move_to_pose(arm_r, pre_r)) is False:
        return failed()
    if not env._wait_pair_at_station(idx_l, idx_r):
        return failed()
    got_l, got_r = env._reach_and_attach_pair(idx_l, idx_r, arm_l, arm_r)
    if not (got_l and got_r):
        return failed()
    return [(idx_l, arm_l), (idx_r, arm_r)]


def _robot_drop(env, held):
    """Carry the already-grasped fruit(s) to baskets and release them."""
    try:
        for idx, arm in held:
            env._carry_and_drop(idx, arm, env._basket_target_xy(idx))
    finally:
        for idx, _arm in held:
            env._grasping_idxs.discard(idx)
        env._end_spawn_hold()


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
            "1 / 2 / 3 — select left / right / both arms",
            "Space — first press picks up selected fruit(s); second press drops them",
            "V — toggle view: top-down ↔ head_camera",
            "Esc — close the viewer window to quit",
            "Select an arm, then press Space; it waits for the required fruit(s).",
            "--robot-motion planner|interpolate",
        ],
    )
    keys_prev: dict = {}
    settle_after_done = None
    selected = None
    pending_pick = False
    held = None
    highlight = ArmGripperHighlight(env) if use_robot else None

    def on_step(window, step):
        nonlocal settle_after_done, selected, pending_pick, held
        # Start belts every step (safe if already True).
        env._belt_running = True

        if held is None:
            selected_arms = tuple(getattr(env, "_interactive_selected_arms", ()))
            if use_robot and selected_arms:
                new_selected = "both" if len(selected_arms) == 2 else selected_arms[0]
                if new_selected != selected:
                    selected = new_selected
                    if highlight is not None:
                        highlight.set_selected(selected)
            if edge_pressed(window, "space", keys_prev):
                if selected is None:
                    print("Select an arm first: 1 (left), 2 (right), or 3 (both).")
                else:
                    pending_pick = True
                    print(f"Waiting to pick with {selected} selection…")
        elif edge_pressed(window, "space", keys_prev):
            if use_robot:
                _robot_drop(env, held)
            else:
                _keyboard_drop(env, held)
            print("Dropped selected fruit(s) into their baskets.")
            held = None
            selected = None
            if highlight is not None:
                highlight.clear()

        # Keep a Space pickup request until its selected fruit(s) enter the
        # short live pick window, then leave them held for the next Space.
        if pending_pick and held is None and selected is not None and not env._grasping_idxs:
            indices = _indices_for_selection(env, selected)
            if indices is not None:
                held = _robot_pick(env, indices) if use_robot else _keyboard_pick(env, indices)
                pending_pick = False
                if held is None:
                    print("Pickup failed; press Space to try again.")
                else:
                    names = ", ".join(f"{env.item_types[idx]}_{idx}" for idx, _arm in held)
                    print(f"Holding {names}. Press Space again to drop.")

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
