#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive viewer for ``put_cup_belt``.

Run from any directory:

    /path/to/RoboDynaExp/script_exp/interactive_put_cup_belt.py --control keyboard
    /path/to/RoboDynaExp/script_exp/interactive_put_cup_belt.py --control robot

Grasp / release with Space (open/close selected gripper). Teleop the cup onto the
belt gap; when the cup leaves the fingers (open gripper or slip), landing pose
is scored after a short settle — gripper open is not required for success.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import sapien
import sapien.physx
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "script" / "bench_script"))
sys.path.insert(0, str(REPO_ROOT / "script_exp"))

from _interactive_common import (  # noqa: E402
    UniversalRobotControls,
    make_viewer_view_toggle,
    print_instructions,
    print_mode_controls,
    report_task_result,
    RealtimePhysicsPacer,
    terminal_hold_should_close,
    print_episode_condition,
)


# Standalone-demo-only world-Y placement adjustment.  The task environment and
# its shared configuration files remain unchanged.  Negative Y moves the setup
# toward the robot/front of the scene.
INTERACTIVE_SETUP_Y_OFFSET = -0.06
# Gap between the belt's south edge and the cup's north edge.
INTERACTIVE_CUP_SOUTH_CLEARANCE = 0.05


CONTROLS_KEYBOARD = """
  (Same as robot) Space grasp/release; arrows / E / Q teleop the arm.
"""

CONTROLS_ROBOT = """
  Space             open / close selected gripper to grasp or release the cup
  1 / 2 / 3         select left / right / both arms
  Arrows / E / Q    teleop the selected arm(s)
  When the cup leaves the fingers, landing is scored after a short settle.
"""


def _embodiment_config(robot_file):
    with open(Path(robot_file) / "config.yml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _configure_task(config_name: str, seed: int, use_robot: bool = False):
    config_path = REPO_ROOT / "task_config" / f"{config_name}.yml"
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    config.update(
        task_name="put_cup_belt",
        render_freq=1,
        now_ep_num=0,
        seed=seed,
        need_plan=use_robot,
        save_data=False,
    )
    table_xy_bias = list(config.get("table_xy_bias", [0.0, 0.0]))
    if len(table_xy_bias) < 2:
        table_xy_bias = [0.0, 0.0]
    table_xy_bias[1] = float(table_xy_bias[1]) + INTERACTIVE_SETUP_Y_OFFSET
    config["table_xy_bias"] = table_xy_bias

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


def _get_rigid(actor):
    for comp in actor.actor.get_components():
        if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent):
            return comp
    return None


def _set_cup_pose(env, x, y, z=None, kinematic=True, quat=None):
    pose = env.cup.get_pose()
    if z is None:
        z = float(pose.p[2])
    if quat is None:
        quat = pose.q
    new_pose = sapien.Pose([float(x), float(y), float(z)], quat)
    env.cup.actor.set_pose(new_pose)
    rigid = _get_rigid(env.cup)
    if rigid is None:
        return
    try:
        rigid.set_linear_velocity(np.zeros(3))
        rigid.set_angular_velocity(np.zeros(3))
        if kinematic:
            rigid.set_kinematic(True)
            rigid.set_kinematic_target(new_pose)
        else:
            rigid.set_kinematic(False)
    except Exception:
        pass


def _place_cup_south_of_belt(env):
    """Place the cup 5 cm clear of the belt's south edge (dynamic for Space grasp)."""
    pose = env.cup.get_pose()
    belt_half_y = float(getattr(env, "belt_plate_half_size", [0.0, 0.0])[1])
    cup_half_y = 0.5 * float(env._actor_world_size(env.cup)[1])
    target_y = (
        float(env.belt_y)
        - belt_half_y
        - cup_half_y
        - INTERACTIVE_CUP_SOUTH_CLEARANCE
    )
    _set_cup_pose(
        env,
        float(pose.p[0]),
        target_y,
        0.74 + float(env.table_z_bias),
        kinematic=False,
        quat=env.CUP_UPRIGHT_QPOS,
    )
    env.cup_y = target_y
    print(f"Starting cup 5 cm south of belt at y={target_y:.3f}.")


def _mark_deposit(env):
    env._attempt_active = True
    env._deposit_step = int(getattr(env, "_kin_step", 0))
    env._slot_x_at_deposit = float(env.slot_x())


class CupReleaseMonitor:
    """Watch finger contact; when the cup leaves the hand, score the landing.

    Mirrors the previous slip / Space-release path: mark deposit on detach,
    settle briefly, then ``check_success`` / placement score.
    """

    def __init__(self, env):
        self.env = env
        self.holding = False
        self.placed = False
        # Require sustained contact before treating a later drop as detach
        # (avoids one-frame PhysX flicker right after grasp).
        self._hold_contact_seen = False
        self._no_contact_steps = 0
        self._slip_no_contact_steps = 8

    def update(self, _window=None):
        if self.placed:
            return
        held = bool(self.env._cup_held())
        if held:
            if not self.holding:
                self.holding = True
                self.env._attempt_active = True
                print("Cup grasped — teleop to the belt gap, then Space to open / release.")
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
        p = np.asarray(self.env.cup.get_pose().p, dtype=float)
        _mark_deposit(self.env)
        self.holding = False
        self.placed = True
        print(
            f"Cup detached at ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}); "
            "evaluating landing…"
        )


def main():
    parser = argparse.ArgumentParser(description="Interactive put_cup_belt viewer")
    parser.add_argument("--config", default="demo_dynamic", help="Task config name without .yml")
    parser.add_argument("--seed", type=int, default=0, help="Scene randomization seed")
    parser.add_argument(
        "--control",
        choices=("keyboard", "robot"),
        default="robot",
        help="Interaction method (default: robot)",
    )
    parser.add_argument(
        "--robot-motion",
        choices=("planner", "interpolate"),
        default="interpolate",
        help="Retained for compatibility; arm teleop uses UniversalRobotControls.",
    )
    args = parser.parse_args()

    from envs import CONFIGS_PATH
    from envs.put_cup_belt import put_cup_belt
    globals()["CONFIGS_PATH"] = CONFIGS_PATH

    print_mode_controls("put_cup_belt", args.control, keyboard=CONTROLS_KEYBOARD, robot=CONTROLS_ROBOT)

    env = put_cup_belt()
    # Always enable arm teleop + Space grasp/release.
    env._interactive_robot_mode = True
    env.setup_demo(**_configure_task(args.config, args.seed, use_robot=True))
    print_episode_condition(env)
    _place_cup_south_of_belt(env)
    print(
        f"Side={'left' if env.mirrored else 'right'}; "
        f"curtains={env.blue_curtains_enabled}; "
        f"gap x≈{env.slot_x():.3f}."
    )

    release_monitor = CupReleaseMonitor(env)

    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    views = make_viewer_view_toggle(env, viewer)
    if views.robot_controls is None:
        views.robot_controls = UniversalRobotControls(env)

    print_instructions(
        "Space opens/closes the gripper to grasp/release the cup. "
        "When the cup leaves the fingers, landing is scored."
    )

    placed_since = None
    terminal_started_at = None
    pacer = RealtimePhysicsPacer(env)

    try:
        while not viewer.closed:
            n_steps = pacer.begin_frame()
            views.update(viewer.window)
            release_monitor.update(viewer.window)

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

            if getattr(env, "_curtain_hit", False) and placed_since is None:
                report_task_result(env, "curtain contact")
                terminal_started_at = time.perf_counter()
                continue
            if release_monitor.placed:
                if placed_since is None:
                    placed_since = time.perf_counter()
                    print("Cup detached; settling…")
                elif time.perf_counter() - placed_since >= 2.0:
                    hit = bool(getattr(env, "_curtain_hit", False))
                    detail = f"score={env.placement_score():.2f}"
                    if hit:
                        detail = f"curtain hit; {detail}"
                    report_task_result(env, detail)
                    terminal_started_at = time.perf_counter()
                    continue
    finally:
        env.close_env()


if __name__ == "__main__":
    main()
    # household_task_gui convention: 0=SUCCESS, 10=FAILURE, 2=no result
    from _interactive_common import task_result_exit_code
    raise SystemExit(task_result_exit_code())
