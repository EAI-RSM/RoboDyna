#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive viewer for ``catch_ramp_ball``.

Run from any directory:

    /path/to/RoboDynaExp/script_exp/interactive_catch_ramp_ball.py --control keyboard
    /path/to/RoboDynaExp/script_exp/interactive_catch_ramp_ball.py --control robot

Keyboard mode nudges the cup with arrows and freezes it on Space.
Robot mode: Space picks up the cup, Space again drops it in place.
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

from _interactive_common import make_viewer_view_toggle, report_task_result, print_mode_controls  # noqa: E402


CONTROLS_KEYBOARD = """
  Space             freeze/place cup at current XY
  V                 toggle view: top-down ↔ head_camera
  Escape            quit
------------------------------------------------------------
  Flow: nudge with arrows → Space to place
  Success: red ball lands in the cup (not the distractor)
"""

CONTROLS_ROBOT = """
  Space             first press picks up the cup; second drops it
  V                 toggle view: top-down ↔ head_camera
  Escape            quit
------------------------------------------------------------
  Flow: Space to pick up → move with arrows / E/Q → Space to drop
  Success: red ball lands in the cup (not the distractor)
  --robot-motion planner|interpolate
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
        task_name="catch_ramp_ball",
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


def _aim_xy(env):
    landing, _ = env._predict_landing()
    aim = np.asarray(getattr(env, "catch_aim", landing), dtype=float)
    return float(aim[0]), float(aim[1])


def _set_cup_xy(env, x, y, z=None):
    import sapien
    pose = env.cup.get_pose()
    if z is None:
        z = float(env.table_top + env.CUP_CENTER_Z)
    new_pose = sapien.Pose([float(x), float(y), float(z)], pose.q)
    # ``create_actor`` returns an Actor wrapper; only its entity can be posed.
    entity = getattr(env.cup, "actor", env.cup)
    entity.set_pose(new_pose)
    rigid = env._cup_comp()
    if rigid is not None:
        try:
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
    """Keep the cup on a usable patch of the table near the ramp exit."""
    x = float(np.clip(x, -0.40, 0.40))
    y = float(np.clip(y, -0.50, 0.25))
    return x, y


class EdgeKey:
    def __init__(self):
        self._prev = False

    def poll(self, down):
        edge = bool(down) and not self._prev
        self._prev = bool(down)
        return edge


class KeyboardCupController:
    def __init__(self, env):
        self.env = env
        self.placed = False
        self._space = EdgeKey()

    def update(self, window):
        if not self.placed:
            dx, dy = _nudge_from_keys(window)
            if dx or dy:
                p = np.asarray(self.env.cup.get_pose().p, dtype=float)
                x, y = _clamp_table_xy(self.env, p[0] + dx, p[1] + dy)
                _set_cup_xy(self.env, x, y)
        if self._space.poll(window.key_down("space")):
            p = np.asarray(self.env.cup.get_pose().p, dtype=float)
            x, y = _clamp_table_xy(self.env, p[0], p[1])
            _set_cup_xy(self.env, x, y, self.env.table_top + self.env.CUP_CENTER_Z)
            self.env._cup_ready = True
            self.placed = True
            print(f"Cup placed at ({x:.3f}, {y:.3f}).")


class RobotCupController:
    def __init__(self, env, ArmTag):
        self.env = env
        self.ArmTag = ArmTag
        self.arm = None
        self.holding = False
        self.placed = False
        self.busy = False
        self._space = EdgeKey()

    def _choose_arm(self):
        selected = tuple(getattr(self.env, "_interactive_selected_arms", ()))
        if selected:
            return self.ArmTag(selected[0])
        x, _ = _aim_xy(self.env)
        return self.ArmTag("right" if float(x) > 0 else "left")

    def grasp(self):
        self.busy = True
        self.arm = self._choose_arm()
        self.env.move(self.env.grasp_actor(self.env.cup, arm_tag=self.arm, pre_grasp_dis=0.08))
        if self.env.plan_success:
            self.env.move(self.env.move_by_displacement(self.arm, z=0.12, move_axis="arm"))
            self.holding = True
            print(f"Picked up cup with {self.arm} arm. Move, then Space to drop.")
        else:
            print("Grasp failed; planner disabled further robot actions.")
        self.busy = False

    def drop(self):
        if not self.holding or self.arm is None or self.placed:
            return
        self.busy = True
        self.env.move(self.env.open_gripper(self.arm))
        for _ in range(8):
            self.env._update_kinematic_tasks()
            self.env.scene.step()
        p = np.asarray(self.env.cup.get_pose().p, dtype=float)
        x, y = _clamp_table_xy(self.env, p[0], p[1])
        _set_cup_xy(self.env, x, y, self.env.table_top + self.env.CUP_CENTER_Z)
        self.env._cup_ready = True
        self.holding = False
        self.placed = True
        print(f"Dropped cup at ({x:.3f}, {y:.3f}).")
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
    parser = argparse.ArgumentParser(description="Interactive catch_ramp_ball viewer")
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
    from envs.catch_ramp_ball import catch_ramp_ball
    from envs.utils.action import ArmTag
    globals()["CONFIGS_PATH"] = CONFIGS_PATH

    print_mode_controls("catch_ramp_ball", args.control, keyboard=CONTROLS_KEYBOARD, robot=CONTROLS_ROBOT)
    if args.robot_motion == "interpolate":
        print(
            "Note: --robot-motion interpolate uses planner motions for this teleop task "
            "(key-press sandboxes use joint interpolation)."
        )

    env = catch_ramp_ball()
    env.setup_demo(**_configure_task(args.config, args.seed, use_robot=args.control == "robot"))
    env._interactive_selected_arms = (
        "right" if float(_aim_xy(env)[0]) > 0 else "left",
    )
    env._start_ball_motion(expert_demo=False)
    x, y = _aim_xy(env)
    print(f"Predicted catch aim ≈ ({x:.3f}, {y:.3f}). Ball is rolling.")

    controller = (
        RobotCupController(env, ArmTag) if args.control == "robot" else KeyboardCupController(env)
    )

    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    views = make_viewer_view_toggle(env, viewer)

    if args.control == "robot":
        print("Space picks up the cup; Space again drops it.")
    else:
        print("Arrows nudge the cup; Space places it.")

    settle_after = None
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

            if getattr(env, "_ball_phase", None) == "released":
                if settle_after is None:
                    settle_after = time.perf_counter()
                    print("Ball released from ramp lip; waiting to settle…")
                elif time.perf_counter() - settle_after >= 2.5:
                    report_task_result(env)
                    break

            remaining = float(env.scene.get_timestep()) - (time.perf_counter() - frame_start)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        env.close_env()


if __name__ == "__main__":
    main()
