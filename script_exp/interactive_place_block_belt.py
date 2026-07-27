#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive sandbox for ``place_block_belt``.

Grasp the tall block, hover/match over the belt, release before the place line.

Run from any directory:

    /path/to/RoboDynaExp/script_exp/interactive_place_block_belt.py --control keyboard
    /path/to/RoboDynaExp/script_exp/interactive_place_block_belt.py --control robot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _interactive_common import (  # noqa: E402
    add_robot_motion_arg,
    arrow_nudge_xy,
    bootstrap_repo,
    configure_task,
    edge_pressed,
    hold_dynamic_at,
    print_banner,
    release_dynamic,
    run_viewer_loop,
)

bootstrap_repo()


def _match_geometry(env):
    match_dist = max(0.02, env.belt_speed * 0.5)
    max_match = max(
        0.01,
        abs(float(env.place_line_x) - float(env.belt_x_start)) - (env.block_half_w + 0.01),
    )
    match_dist = float(min(match_dist, max_match))
    hover_x = float(env.belt_x_start)
    release_x = hover_x + env.belt_dir * match_dist
    preferred_y = (
        float(env.bowl.get_pose().p[1])
        if getattr(env, "bowl", None) is not None and not getattr(env, "bowl_move_enabled", False)
        else float(env.belt_y)
    )
    lane_y = env._choose_release_lane_y(preferred_y)
    return match_dist, hover_x, release_x, lane_y


def _hover_z(env, clearance: float = 0.015) -> float:
    return float(env.belt_surface_z + clearance + env.block_half_h)


def _prepare_keyboard_hold(env):
    match_dist, hover_x, release_x, lane_y = _match_geometry(env)
    z = _hover_z(env)
    # Start at hover (belt load); user can nudge toward release_x before Space.
    xy = np.array([hover_x, lane_y], dtype=np.float64)
    hold_dynamic_at(env._block_dyn, env.block, [xy[0], xy[1], z], quat=[1, 0, 0, 0])
    if env._block_dyn is not None:
        try:
            env._block_dyn.set_kinematic(True)
        except Exception:
            pass
    env._interactive_hold_xy = xy
    env._interactive_hold_z = z
    env._interactive_match_dist = match_dist
    env._interactive_release_x = release_x
    env._interactive_holding = True
    env._interactive_released = False
    env._released = False
    env._belt_active = False
    env._release_delay_left = 0


def _prepare_robot_hold(env, arm_tag):
    """Pick the block, then carry it to the normal belt hover position."""
    match_dist, hover_x, release_x, lane_y = _match_geometry(env)
    env.move(env.grasp_actor(env.block, arm_tag=arm_tag, pre_grasp_dis=0.1))
    env.move(env.move_by_displacement(arm_tag=arm_tag, z=0.14, move_axis="arm"))
    dx = hover_x - float(env.block.get_pose().p[0])
    dy = lane_y - float(env.block.get_pose().p[1])
    env.move(env.move_by_displacement(arm_tag=arm_tag, x=dx, y=dy))
    env._move_block_to_belt_clearance(arm_tag=arm_tag, clearance=0.015)
    env._interactive_arm = arm_tag
    env._interactive_match_dist = match_dist
    env._interactive_release_x = release_x
    env._interactive_holding = True
    env._interactive_released = False
    env._interactive_matched = False
    env._released = False
    env._belt_active = False
    env._release_delay_left = 0


def _do_release(env, use_robot: bool):
    if getattr(env, "_interactive_released", False):
        return
    env._interactive_released = True
    env._interactive_holding = False
    match_dist = float(env._interactive_match_dist)

    if use_robot and getattr(env, "_interactive_arm", None) is not None:
        # Match stroke then open (expert timing mechanic).
        env.move(env.move_by_displacement(
            arm_tag=env._interactive_arm, x=env.belt_dir * match_dist,
        ))
        env._release_q = [1.0, 0.0, 0.0, 0.0]
        env._released = True
        env.move(env.open_gripper(env._interactive_arm))
        env._release_delay_left = int(env.belt_release_delay_steps)
        print("Match stroke + open — belt will engage after release delay.")
        return

    # Keyboard: drop onto the belt surface near the current hold / release_x.
    x = float(np.clip(
        env._interactive_hold_xy[0],
        min(env.belt_x_start, env._interactive_release_x) - 0.02,
        max(env.belt_x_start, env._interactive_release_x) + 0.02,
    ))
    y = float(env._interactive_hold_xy[1])
    z = float(env.belt_surface_z + env.block_half_h + 0.002)
    release_dynamic(env._block_dyn)
    hold_dynamic_at(env._block_dyn, env.block, [x, y, z], quat=[1, 0, 0, 0])
    release_dynamic(env._block_dyn)
    # Impart approximate belt-direction momentum.
    if env._block_dyn is not None:
        try:
            env._block_dyn.set_linear_velocity([env.belt_dir * env.belt_speed, 0.0, 0.0])
        except Exception:
            pass
    env._release_q = [1.0, 0.0, 0.0, 0.0]
    env._released = True
    env._release_delay_left = int(env.belt_release_delay_steps)
    print("Released block onto belt — ride to the bowl.")


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

    print_banner(
        "place_block_belt — interactive controls",
        [
            f"Mode: {args.control}  |  robot-motion: {args.robot_motion}  |  "
            f"config: {args.config}  |  seed: {args.seed}",
            "Goal: place the tall block on the belt BEFORE the red place line;",
            "      match belt velocity, stay in the clear lane if a blocker is present.",
            "1 / 2 / 3 — select left / right / both arms (robot mode)",
            "Space  — (1) pick + lift above the belt  (2) match + release",
            "Arrows — nudge hold XY before release",
            "V — toggle view: top-down ↔ head_camera",
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

    selected_arm = "right" if env.block.get_pose().p[0] > 0 else "left"
    env._interactive_selected_arms = (selected_arm,)
    env._interactive_holding = False
    env._interactive_released = False
    env._released = False
    env._belt_active = False
    env._release_delay_left = 0
    print(
        f"Selected {selected_arm} arm. Press Space to pick the block and lift it over the belt."
        if use_robot else "Press Space to lift the block into the virtual hold over the belt."
    )

    keys_prev: dict = {}
    off_belt_since = None
    settle_steps = max(1, int(round(2.0 / float(env.scene.get_timestep()))))

    def on_step(window, step):
        nonlocal off_belt_since, selected_arm
        selected = tuple(getattr(env, "_interactive_selected_arms", ()))
        if not env._interactive_holding and selected:
            selected_arm = selected[0]

        if edge_pressed(window, "space", keys_prev) and not env._interactive_released:
            if not env._interactive_holding:
                if use_robot:
                    from envs.utils.action import ArmTag

                    print(f"Robot: picking with {selected_arm} arm and lifting over the belt…")
                    _prepare_robot_hold(env, ArmTag(selected_arm))
                else:
                    _prepare_keyboard_hold(env)
                print("Holding above the belt. Nudge with arrows, then press Space to release.")
            else:
                _do_release(env, use_robot)

        nudge = (arrow_nudge_xy(window, step=0.0025) if not use_robot
                 else np.zeros(2, dtype=np.float64))
        if float(np.linalg.norm(nudge)) > 0 and env._interactive_holding and not env._interactive_released:
            if use_robot and getattr(env, "_interactive_arm", None) is not None:
                if step % 8 == 0:
                    env.move(env.move_by_displacement(
                        arm_tag=env._interactive_arm,
                        x=float(nudge[0]) * 4,
                        y=float(nudge[1]) * 4,
                    ))
            else:
                env._interactive_hold_xy = env._interactive_hold_xy + nudge
                hold_dynamic_at(
                    env._block_dyn, env.block,
                    [env._interactive_hold_xy[0], env._interactive_hold_xy[1], env._interactive_hold_z],
                    quat=[1, 0, 0, 0],
                )
        elif env._interactive_holding and not env._interactive_released and not use_robot:
            hold_dynamic_at(
                env._block_dyn, env.block,
                [env._interactive_hold_xy[0], env._interactive_hold_xy[1], env._interactive_hold_z],
                quat=[1, 0, 0, 0],
            )

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
