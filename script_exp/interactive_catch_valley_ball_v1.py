#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive viewer for ``catch_valley_ball_v1``.

Run from any directory:

    /path/to/RoboDynaExp/script_exp/interactive_catch_valley_ball_v1.py --control keyboard
    /path/to/RoboDynaExp/script_exp/interactive_catch_valley_ball_v1.py --control robot

Keyboard mode freezes the catcher on Space.
Robot mode: Space picks up the catcher, Space again drops it in place.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "script" / "bench_script"))
sys.path.insert(0, str(REPO_ROOT / "script_exp"))

from _interactive_common import (  # noqa: E402
    action_failed,
    make_viewer_view_toggle,
    print_instructions,
    print_mode_controls,
    report_task_result,
    sleep_to_timestep,
    terminal_hold_should_close,
    resolve_action_arm,
    print_episode_condition,
)


CONTROLS_KEYBOARD = """
  Space             freeze/place catcher at current XY
"""

CONTROLS_ROBOT = """
  Space             pick up catcher; press again to drop it

  Flow: Space to pick up → move with arrows / E/Q → Space to drop.
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
        task_name="catch_valley_ball_v1",
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


def _target_xy(env):
    landing = np.asarray(env.landing, dtype=float)
    return float(env._catch_target_x(landing[0])), float(landing[1])


def _bowl_place_z(env):
    offset = float(getattr(env, "BOWL_PLACE_Z_OFFSET", -0.020))
    return float(env.table_top + offset)


def _get_rigid(actor):
    import sapien
    for comp in actor.actor.get_components():
        if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent):
            return comp
    return None


def _set_bowl_xy(env, x, y, z=None):
    import sapien
    pose = env.bowl.get_pose()
    if z is None:
        z = _bowl_place_z(env)
    new_pose = sapien.Pose([float(x), float(y), float(z)], pose.q)
    # ``create_actor`` returns an Actor wrapper; only its entity can be posed.
    entity = getattr(env.bowl, "actor", env.bowl)
    entity.set_pose(new_pose)
    rigid = _get_rigid(env.bowl)
    if rigid is not None:
        try:
            rigid.set_disable_gravity(True)
            rigid.set_kinematic(True)
            rigid.set_linear_velocity(np.zeros(3))
            rigid.set_angular_velocity(np.zeros(3))
            rigid.set_kinematic_target(new_pose)
        except Exception:
            pass


def _nudge_from_keys(window, step=0.008):
    dx = dy = 0.0
    if window.key_down("left"):
        dx -= step
    if window.key_down("right"):
        dx += step
    if window.key_down("up"):
        dy += step
    if window.key_down("down"):
        dy -= step
    return dx, dy


def _clamp_table_xy(env, x, y):
    """Snap X past the red line (success requires this); keep Y on the table."""
    x = float(env._catch_target_x(x))
    y = float(np.clip(y, -0.50, 0.25))
    return x, y


class EdgeKey:
    def __init__(self):
        self._prev = False

    def poll(self, down):
        edge = bool(down) and not self._prev
        self._prev = bool(down)
        return edge


class KeyboardBowlController:
    def __init__(self, env):
        self.env = env
        self.placed = False
        self._space = EdgeKey()

    def update(self, window):
        if not self.placed:
            dx, dy = _nudge_from_keys(window)
            if dx or dy:
                p = np.asarray(self.env.bowl.get_pose().p, dtype=float)
                x, y = _clamp_table_xy(self.env, p[0] + dx, p[1] + dy)
                _set_bowl_xy(self.env, x, y)
        if self._space.poll(window.key_down("space")):
            p = np.asarray(self.env.bowl.get_pose().p, dtype=float)
            x, y = _clamp_table_xy(self.env, p[0], p[1])
            _set_bowl_xy(self.env, x, y, _bowl_place_z(self.env))
            self.env._fix_bowl_at_placed_pose()
            self.env._bowl_ready = True
            self.placed = True
            print(f"Bowl placed at ({x:.3f}, {y:.3f}) behind red line.")


class RobotBowlController:
    def __init__(self, env, ArmTag):
        self.env = env
        self.ArmTag = ArmTag
        self.arm = None
        self.holding = False
        self.placed = False
        self.busy = False
        self._space = EdgeKey()

    def _choose_arm(self):
        return resolve_action_arm(self.env, self.ArmTag, exactly_one=True)

    def grasp(self):
        self.busy = True
        self.arm = self._choose_arm()
        if self.arm is None:
            self.busy = False
            return
        self.env.move(self.env.grasp_actor(self.env.bowl, arm_tag=self.arm, pre_grasp_dis=0.10))
        if self.env.plan_success:
            self.env._weld_bowl_to_end_effector(self.arm)
            self.env.move(self.env.move_by_displacement(self.arm, z=0.05, move_axis="arm"))
            self.holding = True
            print(f"Picked up bowl with {self.arm} arm. Move, then Space to drop.")
        else:
            action_failed(self.env, (str(self.arm),), detail="grasp failed")
        self.busy = False

    def drop(self):
        if not self.holding or self.arm is None or self.placed:
            return
        self.busy = True
        self.env._unweld_bowl()
        self.env.move(self.env.open_gripper(self.arm))
        for _ in range(8):
            self.env._update_kinematic_tasks()
            self.env.scene.step()
        self.env._fix_bowl_at_placed_pose()
        p = np.asarray(self.env.bowl.get_pose().p, dtype=float)
        self.env._bowl_ready = True
        self.holding = False
        self.placed = True
        print(f"Dropped bowl at ({p[0]:.3f}, {p[1]:.3f}).")
        self.busy = False

    def update(self, window):
        if self.busy or self.placed:
            return
        if self._space.poll(window.key_down("space")):
            if not self.holding:
                self.grasp()
            else:
                self.drop()


def main():
    parser = argparse.ArgumentParser(description="Interactive catch_valley_ball_v1 viewer")
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
        default="planner",
        help="Robot motion backend (interpolate = faster joint interp when supported; default planner)",
    )
    args = parser.parse_args()

    from envs import CONFIGS_PATH
    from envs.catch_valley_ball_v1 import catch_valley_ball_v1
    from envs.utils.action import ArmTag
    globals()["CONFIGS_PATH"] = CONFIGS_PATH

    print_mode_controls("catch_valley_ball_v1", args.control, keyboard=CONTROLS_KEYBOARD, robot=CONTROLS_ROBOT)
    if args.robot_motion == "interpolate":
        print(
            "Note: --robot-motion interpolate uses planner motions for this teleop task "
            "(key-press sandboxes use joint interpolation)."
        )

    env = catch_valley_ball_v1()
    env.setup_demo(**_configure_task(args.config, args.seed, use_robot=args.control == "robot"))
    env._interactive_selected_arms = (
        "left" if env.mirrored else "right",
    )
    print_episode_condition(env)
    x, y = _target_xy(env)
    print(
        f"Predicted catch target ≈ ({x:.3f}, {y:.3f}); red_line_x={env.red_line_x:.3f}; "
        f"mirrored={env.mirrored}."
    )

    controller = (
        RobotBowlController(env, ArmTag) if args.control == "robot" else KeyboardBowlController(env)
    )

    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    views = make_viewer_view_toggle(env, viewer)

    if args.control == "robot":
        print_instructions("Space picks up the catcher; Space again drops it.")
    else:
        print_instructions("Space places the catcher at its current XY.")

    settle_after = None
    terminal_started_at = None

    try:
        while not viewer.closed:
            views.update(viewer.window)
            frame_start = time.perf_counter()
            controller.update(viewer.window)

            env._update_kinematic_tasks()
            env.scene.step()
            env.scene.update_render()
            viewer.render()

            if viewer.window.key_down("escape"):
                break

            if terminal_started_at is not None:
                if terminal_hold_should_close(terminal_started_at):
                    break
                sleep_to_timestep(env, frame_start)
                continue

            if getattr(env, "_ball_phase", None) == "released":
                if settle_after is None:
                    settle_after = time.perf_counter()
                    print("Ball left the valley exit; waiting to settle…")
                elif time.perf_counter() - settle_after >= 2.5:
                    report_task_result(env)
                    terminal_started_at = time.perf_counter()

            remaining = float(env.scene.get_timestep()) - (time.perf_counter() - frame_start)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        env.close_env()


if __name__ == "__main__":
    main()
    # household_task_gui convention: 0=SUCCESS, 10=FAILURE, 2=no result
    from _interactive_common import task_result_exit_code
    raise SystemExit(task_result_exit_code())
