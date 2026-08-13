#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive sandbox for ``place_block_belt``.

Grasp the tall block with Space, teleop over the belt, open Space to drop it so the
belt can carry it. Space opens/closes the gripper only.

Run from any directory:

    /path/to/RoboDynaExp/script_exp/interactive_place_block_belt.py --control keyboard
    /path/to/RoboDynaExp/script_exp/interactive_place_block_belt.py --control robot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _interactive_common import (  # noqa: E402
    print_instructions,
    add_robot_motion_arg,
    bootstrap_repo,
    configure_task,
    print_banner,
    run_viewer_loop,
    print_episode_condition,
)

bootstrap_repo()


def _block_held(env) -> bool:
    """True while fingers still contact the block (Space-close grasp)."""
    if getattr(env, "block", None) is None:
        return False
    try:
        return len(env.get_gripper_actor_contact_position(env.block.get_name())) > 0
    except Exception:
        return False


class BlockReleaseMonitor:
    """When the block leaves the hand, mark release so the belt drive can engage."""

    def __init__(self, env):
        self.env = env
        self.holding = False
        self._hold_contact_seen = False
        self._no_contact_steps = 0
        self._slip_no_contact_steps = 8

    def update(self):
        if getattr(self.env, "_interactive_released", False):
            return
        held = _block_held(self.env)
        if held:
            if not self.holding:
                self.holding = True
                print("Block grasped — teleop over the belt, then Space to open / release.")
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
        self.env._interactive_released = True
        self.env._interactive_holding = False
        self.env._release_q = [1.0, 0.0, 0.0, 0.0]
        self.env._released = True
        self.env._release_delay_left = int(self.env.belt_release_delay_steps)
        print("Block released — belt will engage after release delay.")


def main():
    parser = argparse.ArgumentParser(description="Interactive place_block_belt viewer")
    parser.add_argument("--config", default="demo_dynamic")
    parser.add_argument("--seed", type=int, default=0)
    add_robot_motion_arg(parser)
    args = parser.parse_args()

    from envs.place_block_belt import place_block_belt

    use_robot = args.control == "robot"
    env = place_block_belt()
    env.setup_demo(**configure_task(
        "place_block_belt", args.config, args.seed, use_robot=use_robot,
    ))
    print_episode_condition(env)

    print_banner(
        "place_block_belt — interactive controls",
        [
            f"Mode: {args.control}  |  robot-motion: {args.robot_motion}  |  "
            f"config: {args.config}  |  seed: {args.seed}",
            "Goal: place the tall block on the belt BEFORE the red place line;",
            "      stay in the clear lane if a blocker is present.",
            "1 / 2 / 3 — select left / right / both arms (robot mode)",
            "Space — close to grasp / open to release (drop onto the belt)",
            "Arrows / E / Q — teleop the selected arm(s)",
            "V — cycle view: head_camera ↔ gripper(s)",
            "Esc — close the viewer window to quit",
            "Release too late (past the red line) or into the blocker → failure.",
            "--robot-motion planner|interpolate",
        ],
    )
    if args.robot_motion == "interpolate":
        print(
            "Note: --robot-motion interpolate uses planner motions for this teleop task "
            "(key-press sandboxes use joint interpolation)."
        )

    suggested_arm = "right" if env.block.get_pose().p[0] > 0 else "left"
    env._interactive_holding = False
    env._interactive_released = False
    env._released = False
    env._belt_active = False
    env._release_delay_left = 0
    print_instructions(
        f"Press 1/2/3 to select an arm (block is on the {suggested_arm}). "
        "Space closes/opens the gripper to grasp/release the block. "
        "When the block leaves the fingers on the belt, the conveyor engages."
    )

    release_monitor = BlockReleaseMonitor(env)
    off_belt_since = None
    settle_steps = max(1, int(round(2.0 / float(env.scene.get_timestep()))))

    def on_step(window, step):
        nonlocal off_belt_since
        release_monitor.update()

        if env._interactive_released:
            if env._release_delay_left > 0:
                env._release_delay_left -= 1
                if env._release_delay_left == 0:
                    env._belt_active = True
                    print("Belt drive engaged.")

            # ``_block_dropped`` is latched by the task exactly when the
            # conveyor carries the block past its exit.  The fallback also
            # catches a block that falls off after a verified belt contact.
            left_belt = bool(getattr(env, "_block_dropped", False)) or (
                bool(getattr(env, "placed_on_belt", False)) and not env._on_belt()
            )
            if left_belt and off_belt_since is None:
                off_belt_since = step
                print("Block has left the belt — settling for 2 seconds…")

    def is_done(step):
        if off_belt_since is not None and step - off_belt_since >= settle_steps:
            return True, (
                f"left_belt=True, on_belt={env.placed_on_belt}, in_bowl={env.in_bowl}, "
                f"avoided_blocker={env.avoided_blocker}"
            )
        return False

    run_viewer_loop(env, on_step, is_done=is_done, max_steps=25000)


if __name__ == "__main__":
    main()
    # household_task_gui convention: 0=SUCCESS, 10=FAILURE, 2=no result
    from _interactive_common import task_result_exit_code
    raise SystemExit(task_result_exit_code())
