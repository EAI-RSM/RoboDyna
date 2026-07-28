#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive sandbox for ``pick_ripe_apple``.

Space triggers the frozen front-pinch grasp immediately (any ripeness), then
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
import sapien

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _interactive_common import (  # noqa: E402
    bootstrap_repo,
    arrow_nudge_xy,
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
    """Recolor the selected gripper so the active pickup arm is obvious."""

    def __init__(self, env):
        self._orig = {}
        self._selected = None
        self._entities = {
            "left": self._gripper_entities(env.robot.left_entity),
            "right": self._gripper_entities(env.robot.right_entity),
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
        self._selected = None

    def set_selected(self, arm):
        if arm == self._selected:
            return
        self.clear()
        for entity in self._entities[arm]:
            for material in self._iter_materials(entity):
                self._remember(material)
                try:
                    material.set_base_color_texture(None)
                except Exception:
                    pass
                try:
                    material.set_base_color(_ARM_HIGHLIGHT[arm])
                    material.base_color = _ARM_HIGHLIGHT[arm]
                except Exception:
                    pass
        self._selected = arm


def _arm_for_good(env):
    from envs.utils.action import ArmTag

    return ArmTag("left" if env.apple_side < 0 else "right")


def _do_grasp(env, arm):
    """Trigger the frozen front-grasp + clear/lift immediately on Space."""
    tol = float(getattr(env, "red_tol", env.RED_TOLERANCE_DEFAULT))
    in_window = abs(float(env.ripeness) - float(env.red_window)) <= tol
    print(
        f"Grasping good apple at ripeness={env.ripeness:.3f} "
        f"(window={env.red_window:.3f}±{tol:.3f}, "
        f"{'in window' if in_window else 'OUTSIDE window — will fail'})…"
    )
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

    env._interactive_arm = arm
    env._interactive_phase = "hold"
    print(
        f"Holding apple (r_grasp={env.r_grasp}). "
        "Use arrows to carry it over the basket, then press Space to drop."
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
        default="robot",
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
    selected_arm = "left" if env.apple_side < 0 else "right"
    env._interactive_selected_arms = (selected_arm,)
    highlight = ArmGripperHighlight(env)
    highlight.set_selected(selected_arm)

    print_banner(
        "pick_ripe_apple — interactive controls",
        [
            f"Mode: {args.control}  |  robot-motion: {args.robot_motion}  |  "
            f"config: {args.config}  |  seed: {args.seed}",
            "Goal: pinch the GOOD (red-path) apple near peak red; drop in the basket.",
            "      Do NOT pick the spoiled/yellow apple (Opt1).",
            "1 / 2 / 3 — select left / right / both arms",
            "Space — (1) grasp immediately   (2) drop when over the basket",
            "Arrows — move the held apple/arm in world XY over the basket",
            "V — toggle view: top-down ↔ head_camera",
            "Esc — close the viewer window to quit",
            "Grasp / hang / clear geometry is FROZEN — Space only triggers existing motions.",
            "Success needs BOTH: grasp inside the ripeness window AND apple in the basket.",
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
        f"red_window={env.red_window:.3f}"
        f"±{float(getattr(env, 'red_tol', env.RED_TOLERANCE_DEFAULT)):.3f}  "
        f"ripen_steps={env.ripen_steps}"
    )

    keys_prev: dict = {}
    done_since = None

    def on_step(window, step):
        nonlocal done_since, selected_arm
        if env._interactive_phase == "wait":
            selected = tuple(getattr(env, "_interactive_selected_arms", ()))
            if len(selected) == 1 and selected[0] != selected_arm:
                selected_arm = selected[0]
                highlight.set_selected(selected_arm)

        if edge_pressed(window, "space", keys_prev):
            if env._interactive_phase == "wait":
                from envs.utils.action import ArmTag

                _do_grasp(env, ArmTag(selected_arm))
            elif env._interactive_phase == "hold":
                _do_drop(env)
                highlight.clear()
                done_since = step

        # Carry at the frozen post-grasp height.  Throttle discrete plans so
        # holding an arrow does not enqueue a new planner call every frame.
        nudge = (np.zeros(2, dtype=np.float64) if args.control == "robot"
                 else arrow_nudge_xy(window, step=0.0025))
        if (
            env._interactive_phase == "hold"
            and float(np.linalg.norm(nudge)) > 0.0
            and step % 8 == 0
        ):
            arm = env._interactive_arm
            env.plan_success = True
            env.move(env.move_by_displacement(
                arm_tag=arm,
                x=float(nudge[0]) * 4.0,
                y=float(nudge[1]) * 4.0,
                move_axis="world",
            ))
            if not env.plan_success:
                print("Arm nudge could not reach the requested position.")

        if env._interactive_phase == "done" and done_since is None:
            done_since = step
        # Status heartbeat while the apple is still on the tree.
        if env._interactive_phase == "wait" and step % 150 == 0:
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

    def is_done(step):
        if done_since is not None and step - done_since > 80:
            return True, (
                f"r_grasp={env.r_grasp}, window={env.red_window:.3f}"
                f"±{float(getattr(env, 'red_tol', env.RED_TOLERANCE_DEFAULT)):.3f}, "
                f"ripeness_ok={env._grasp_in_red_window()}, "
                f"ripeness_score={env._ripeness_score():.3f}"
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
