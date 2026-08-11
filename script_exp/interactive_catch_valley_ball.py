#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive viewer for ``catch_valley_ball`` (PhysX push catch box).

Run from any directory:

    /path/to/RoboDynaExp/script_exp/interactive_catch_valley_ball.py --control robot

The catch box is an ordinary dynamic body (same mechanism as ``catch_cup``'s
pillow): it only moves when the closed gripper shoves it. There is no keyboard
teleport of the box.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "script" / "bench_script"))
sys.path.insert(0, str(REPO_ROOT / "script_exp"))

from _interactive_common import (  # noqa: E402
    UniversalRobotControls,
    add_robot_motion_arg,
    make_viewer_view_toggle,
    print_mode_controls,
    report_task_result,
    RealtimePhysicsPacer,
    terminal_hold_should_close,
    print_episode_condition,
)


CONTROLS_KEYBOARD = """
  Prefer --control robot. The box is PhysX-dynamic — shove it with the closed gripper.
"""

CONTROLS_ROBOT = """
  The catch box is PhysX-dynamic — it moves only under gripper contact (no teleport).
  Close the gripper (Space), then shove the box into the catch zone.
"""


def _embodiment_config(robot_file):
    with open(Path(robot_file) / "config.yml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _configure_task(config_name: str, seed: int, use_robot: bool = True):
    config_path = REPO_ROOT / "task_config" / f"{config_name}.yml"
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    config.update(
        task_name="catch_valley_ball",
        render_freq=1,
        now_ep_num=0,
        seed=seed,
        need_plan=use_robot,
        save_data=False,
    )

    with open(Path(CONFIGS_PATH) / "_embodiment_config.yml", "r", encoding="utf-8") as handle:
        embodiments = yaml.safe_load(handle)
    embodiment_names = config.get("embodiment", ["aloha-agilex"])
    if len(embodiment_names) == 1:
        left_name = right_name = embodiment_names[0]
        config["dual_arm_embodied"] = True
    elif len(embodiment_names) == 3:
        left_name, right_name, config["embodiment_dis"] = embodiment_names
        config["dual_arm_embodied"] = False
    else:
        raise SystemExit("Expected one embodiment or [left_embodiment, right_embodiment, separation].")

    config["left_robot_file"] = embodiments[left_name]["file_path"]
    config["right_robot_file"] = embodiments[right_name]["file_path"]
    config["left_embodiment_config"] = _embodiment_config(config["left_robot_file"])
    config["right_embodiment_config"] = _embodiment_config(config["right_robot_file"])
    return config


def main():
    parser = argparse.ArgumentParser(description="Interactive catch_valley_ball viewer")
    parser.add_argument("--config", default="demo_dynamic", help="Task config name without .yml")
    parser.add_argument("--seed", type=int, default=0, help="Scene randomization seed")
    add_robot_motion_arg(parser, robot_motion_default="interpolate")
    args = parser.parse_args()

    from envs import CONFIGS_PATH
    from envs.catch_valley_ball import catch_valley_ball
    globals()["CONFIGS_PATH"] = CONFIGS_PATH

    print_mode_controls(
        "catch_valley_ball",
        args.control,
        keyboard=CONTROLS_KEYBOARD,
        robot=CONTROLS_ROBOT,
    )

    env = catch_valley_ball()
    env._interactive_robot_mode = True
    env.setup_demo(**_configure_task(args.config, args.seed, use_robot=True))

    # Same as catch_cup pillow: after settle, hand the box to PhysX. Block any
    # freeze/teleport helpers while the interactive push session is active.
    env._enable_box_physics()
    print_episode_condition(env)
    env._push_active = True
    env._bowl_ready = False

    catcher = "left" if env.mirrored else "right"
    env._interactive_selected_arms = (catcher,)
    env.together_close_gripper(save_freq=None)

    landing = env.landing
    print(
        f"Catch arm={catcher}; predicted landing ≈ "
        f"({float(landing[0]):.3f}, {float(landing[1]):.3f}); "
        f"red_line_x={env.red_line_x:.3f}; mirrored={env.mirrored}. "
        f"Shove the box with the closed gripper (PhysX)."
    )

    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    views = make_viewer_view_toggle(env, viewer)
    if views.robot_controls is None:
        views.robot_controls = UniversalRobotControls(env)

    settle_after = None
    terminal_started_at = None
    pacer = RealtimePhysicsPacer(env)

    # Start the ball's pre-roll hold now, not at setup: the freeze is meant to give
    # the player time once the window is live, so it must not tick during settle.
    env._ball_freeze_armed = True

    try:
        while not viewer.closed:
            n_steps = pacer.begin_frame()
            views.update(viewer.window)

            # Keep the box dynamic every frame (in case settle helpers re-freeze).
            if not bool(getattr(env, "_push_active", False)):
                env._enable_box_physics()
                env._push_active = True

            if n_steps == 0:
                env.scene.update_render()
                viewer.render()
                if viewer.window.key_down("escape"):
                    break
                if terminal_started_at is not None and terminal_hold_should_close(terminal_started_at):
                    break
                continue

            for _ in range(n_steps):
                env._update_kinematic_tasks()
                env.scene.step()
            env.scene.update_render()
            viewer.render()

            if viewer.window.key_down("escape"):
                break

            if terminal_started_at is not None:
                if terminal_hold_should_close(terminal_started_at):
                    break
                continue

            if getattr(env, "_ball_phase", None) == "released":
                if settle_after is None:
                    settle_after = time.perf_counter()
                    print("Ball left the valley exit; waiting to settle…")
                elif time.perf_counter() - settle_after >= 2.5:
                    report_task_result(env)
                    terminal_started_at = time.perf_counter()
    finally:
        env.close_env()


if __name__ == "__main__":
    main()
    # household_task_gui convention: 0=SUCCESS, 10=FAILURE, 2=no result
    from _interactive_common import task_result_exit_code
    raise SystemExit(task_result_exit_code())
