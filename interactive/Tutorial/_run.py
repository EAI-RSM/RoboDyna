"""Shared empty-scene viewer for tutorial parts 1–4."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_INTERACTIVE = _HERE.parent
sys.path.insert(0, str(_INTERACTIVE))
sys.path.insert(0, str(_HERE))
from _interactive_common import (  # noqa: E402
    add_robot_motion_arg,
    bootstrap_repo,
    configure_task,
    print_banner,
    print_instructions,
    print_mode_controls,
    run_viewer_loop,
    task_result_exit_code,
)

bootstrap_repo()

PART_TITLES = {
    1: "Part 1 — Arm selection and camera",
    2: "Part 2 — Base controls",
    3: "Part 3 — Basic actions",
    4: "Part 4 — Practice",
}

CONTROLS_KEYBOARD = """
  This tutorial part is robot teleop (choose Control: robot in the GUI).
  Esc               close the viewer
"""

CONTROLS_ROBOT = """
  Arrow keys        move selected arm(s) in world XY
  E / Q             raise / lower selected arm(s)
  F / G             tip gripper left / right (world Y)
  R / T             yaw gripper CCW / CW (world Z)
  1 / 2 / 3         select left / right / both arms (selected gripper turns green)
  O                 return selected arm(s) to original position
  Space             open / close selected gripper(s)
  V                 cycle view: head_camera ↔ gripper(s)
  Esc               close the viewer
"""


def main(part: int | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tutorial empty-scene viewer")
    parser.add_argument("--config", default="demo_dynamic")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--part", type=int, default=part or 1, choices=(1, 2, 3, 4))
    add_robot_motion_arg(parser)
    args = parser.parse_args()
    part_id = int(args.part)
    title = PART_TITLES[part_id]

    from envs.tutorial_empty import tutorial_empty

    use_robot = args.control == "robot"
    env = tutorial_empty()
    env.setup_demo(
        **configure_task("tutorial_empty", args.config, args.seed, use_robot=use_robot)
    )

    print_mode_controls(
        "tutorial_empty",
        args.control,
        keyboard=CONTROLS_KEYBOARD,
        robot=CONTROLS_ROBOT,
    )
    if part_id == 1:
        print_banner(
            f"Tutorial {title}",
            [
                f"Mode: {args.control}  |  config: {args.config}  |  seed: {args.seed}",
                "Top-right: press 1, 2, and 3 to select left / right / both arms.",
                "The selected gripper turns green.",
                "Then press V to cycle head_camera ↔ gripper views.",
                "Esc — close the viewer window to quit",
            ],
        )
        print_instructions(
            "Press keys 1, 2, and 3 (figures at the top right). "
            "The selected gripper turns green. "
            "When those are done, the overlay switches to V."
        )
        from _part1 import run_part1

        return run_part1(env)

    if part_id == 2:
        print_banner(
            f"Tutorial {title}",
            [
                f"Mode: {args.control}  |  config: {args.config}  |  seed: {args.seed}",
                "Left arm starts selected (green). 1 / 2 / 3 still switch arms.",
                "Top-right overlay: arrows, then E/Q, R/T, F/G, then Space.",
                "Esc — close the viewer window to quit",
            ],
        )
        print_instructions(
            "Key figures at the top right walk through the base controls. "
            "The left arm is selected so the keys move it immediately."
        )
        from _part2 import run_part2

        return run_part2(env)

    if part_id == 3:
        print_banner(
            f"Tutorial {title}",
            [
                f"Mode: {args.control}  |  config: {args.config}  |  seed: {args.seed}",
                "Left arm starts selected (green). 1 / 2 / 3 still switch arms.",
                "Top-right overlay: grasp cube, hold button, on/off switch, then push box.",
                "Esc — close the viewer window to quit",
            ],
        )
        print_instructions(
            "Key figures at the top right show which keys to use for each action. "
            "Finish one object to spawn the next."
        )
        from _part3 import run_part3

        return run_part3(env)

    print_banner(
        f"Tutorial {title}",
        [
            f"Mode: {args.control}  |  config: {args.config}  |  seed: {args.seed}",
            "Empty table — no task objects yet.",
            "V — cycle view: head_camera ↔ gripper(s)",
            "Esc — close the viewer window to quit",
        ],
    )
    print_instructions(
        "Practice looking around (V) and moving the arms. "
        "Close the viewer or press Esc when finished."
    )

    run_viewer_loop(env, on_step=None, is_done=None)
    return task_result_exit_code()
