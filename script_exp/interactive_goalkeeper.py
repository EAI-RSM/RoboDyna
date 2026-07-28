#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive viewer for ``goalkeeper``.

Run from any directory:

    /path/to/RoboDynaExp/script_exp/interactive_goalkeeper.py --control keyboard
    /path/to/RoboDynaExp/script_exp/interactive_goalkeeper.py --control robot

Place the square keeper in the green zone before the red line, then release so
its front face can stop the ball.
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
  Arrow keys        nudge keeper XY (stay inside the green zone)
  Space             deploy / freeze keeper in place
  V                 toggle view: top-down ↔ head_camera
  Escape             quit
------------------------------------------------------------
  Success: keeper in green zone, front-face save, grippers open
  Place BEFORE the ball crosses the red line
"""

CONTROLS_ROBOT = """
  Arrow keys        nudge keeper XY (stay inside the green zone)
  Space             grasp, then release
  V                 toggle view: top-down ↔ head_camera
  Escape             quit
------------------------------------------------------------
  Success: keeper in green zone, front-face save, grippers open
  Place BEFORE the ball crosses the red line
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
        task_name="goalkeeper",
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


def _clip_keeper_xy(env, x, y):
    x_min = float(env.green_area_x_min + env.keeper_half_x)
    x_max = float(env.green_area_x_max - env.keeper_half_x)
    y_min = float(env.green_area_y_min + env.keeper_half_y)
    y_max = float(env.green_area_y_max - env.keeper_half_y)
    return float(np.clip(x, x_min, x_max)), float(np.clip(y, y_min, y_max))


def _set_keeper_xy(env, x, y, clip=True):
    import sapien
    if clip:
        x, y = _clip_keeper_xy(env, x, y)
    z = float(env.table_top_z + env.keeper_half_z)
    pose = sapien.Pose([x, y, z], [1, 0, 0, 0])
    try:
        env.goalkeeper.set_pose(pose)
    except Exception:
        env.goalkeeper.actor.set_pose(pose)
    rigid = env._get_rigid(env.goalkeeper)
    if rigid is not None:
        try:
            rigid.set_linear_velocity(np.zeros(3))
            rigid.set_angular_velocity(np.zeros(3))
            rigid.set_kinematic(True)
            rigid.set_kinematic_target(pose)
        except Exception:
            pass
    return x, y


def _start_shot(env):
    import sapien
    env._ball_step = 0
    env._ball_blocked = False
    env._goal_conceded = False
    env._late_failure = False
    env._ball_crossed_goal = False
    env._keeper_deployed = False
    env._keeper_drop_pose = None
    env._ball_motion_active = True
    start = getattr(env, "ball_start_pose", None)
    if start is not None and env.ball is not None:
        pose = sapien.Pose(np.asarray(start, dtype=float).tolist(), [1, 0, 0, 0])
        env.ball.set_pose(pose)
        rigid = getattr(env, "_ball_rigid", None)
        if rigid is not None:
            try:
                rigid.set_kinematic(True)
                rigid.set_kinematic_target(pose)
            except Exception:
                pass


class EdgeKey:
    def __init__(self):
        self._prev = False

    def poll(self, down):
        edge = bool(down) and not self._prev
        self._prev = bool(down)
        return edge


class KeyboardKeeperController:
    def __init__(self, env):
        self.env = env
        self.deployed = False
        self._space = EdgeKey()

    def update(self, window):
        if not self.deployed:
            dx, dy = _nudge_from_keys(window)
            if dx or dy:
                p = np.asarray(self.env.goalkeeper.get_pose().p, dtype=float)
                _set_keeper_xy(self.env, p[0] + dx, p[1] + dy, clip=True)
        if self._space.poll(window.key_down("space")):
            p = np.asarray(self.env.goalkeeper.get_pose().p, dtype=float)
            x, y = _set_keeper_xy(self.env, p[0], p[1], clip=True)
            self.env._freeze_keeper_in_place()
            self.env._keeper_deployed = True
            self.deployed = True
            in_zone = self.env._keeper_in_zone()
            print(f"Keeper deployed at ({x:.3f}, {y:.3f}); in_zone={in_zone}.")


class RobotKeeperController:
    def __init__(self, env, ArmTag):
        self.env = env
        self.ArmTag = ArmTag
        self.arm = None
        self.holding = False
        self.deployed = False
        self.busy = False
        self._space = EdgeKey()

    def _choose_arm(self):
        selected = tuple(getattr(self.env, "_interactive_selected_arms", ()))
        side = selected[0] if selected else ("left" if self.env.mirrored else "right")
        return self.ArmTag(side)

    def grasp(self):
        self.busy = True
        self.arm = self._choose_arm()
        self.env.move(self.env.close_gripper(self.arm, pos=0.6))
        self.env.move(self.env.grasp_actor(
            self.env.goalkeeper,
            arm_tag=self.arm,
            pre_grasp_dis=0.10,
            grasp_dis=0.0,
            contact_point_id=[0, 1, 2, 3],
        ))
        if self.env.plan_success:
            self.env.move(self.env.move_by_displacement(self.arm, z=0.12, move_axis="arm"))
            self.holding = True
            print(f"Grasped keeper with {self.arm} arm. Arrows nudge; Space releases.")
        else:
            print("Grasp failed; planner disabled further robot actions.")
        self.busy = False

    def release(self):
        if not self.holding:
            return
        self.busy = True
        self.env.move(self.env.open_gripper(self.arm))
        for _ in range(8):
            self.env._update_kinematic_tasks()
            self.env.scene.step()
        self.env._freeze_keeper_in_place()
        self.env._keeper_deployed = True
        self.env._hold_keeper_kinematic()
        self.holding = False
        self.deployed = True
        print(f"Keeper released; in_zone={self.env._keeper_in_zone()}.")
        self.busy = False

    def nudge(self, window):
        if self.busy or not self.holding:
            return
        dx, dy = _nudge_from_keys(window, step=0.02)
        if not (dx or dy):
            return
        self.busy = True
        self.env.move(self.env.move_by_displacement(
            arm_tag=self.arm, x=dx, y=dy, move_axis="world",
        ))
        self.busy = False

    def update(self, window):
        if self.busy:
            return
        if self._space.poll(window.key_down("space")):
            if not self.holding:
                self.grasp()
            else:
                self.release()
            return
        # Universal viewer controls own arrow/E/Q motion.


def main():
    parser = argparse.ArgumentParser(description="Interactive goalkeeper viewer")
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
    from envs.goalkeeper import goalkeeper
    from envs.utils.action import ArmTag
    globals()["CONFIGS_PATH"] = CONFIGS_PATH

    print_mode_controls("goalkeeper", args.control, keyboard=CONTROLS_KEYBOARD, robot=CONTROLS_ROBOT)
    if args.robot_motion == "interpolate":
        print(
            "Note: --robot-motion interpolate uses planner motions for this teleop task "
            "(key-press sandboxes use joint interpolation)."
        )

    env = goalkeeper()
    env.setup_demo(**_configure_task(args.config, args.seed, use_robot=args.control == "robot"))
    env._interactive_selected_arms = (
        "left" if env.mirrored else "right",
    )
    _start_shot(env)
    target = env.goalkeeper_target_pose.p if env.goalkeeper_target_pose is not None else [0, 0, 0]
    print(
        f"Shot started. Intercept target ≈ ({target[0]:.3f}, {target[1]:.3f}); "
        f"red_line_x={env.red_line_x:.3f}; mirrored={env.mirrored}."
    )

    controller = (
        RobotKeeperController(env, ArmTag) if args.control == "robot" else KeyboardKeeperController(env)
    )

    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    views = make_viewer_view_toggle(env, viewer)

    done_since = None
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

            shot_done = (not getattr(env, "_ball_motion_active", False)) or env._goal_conceded or env._ball_blocked
            if shot_done:
                if done_since is None:
                    done_since = time.perf_counter()
                elif time.perf_counter() - done_since >= 1.0:
                    report_task_result(
                        env,
                        f"in_zone={env._keeper_in_zone()}, blocked={env._ball_blocked}, "
                        f"late={env._late_failure}, conceded={env._goal_conceded}",
                    )
                    break

            remaining = float(env.scene.get_timestep()) - (time.perf_counter() - frame_start)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        env.close_env()


if __name__ == "__main__":
    main()
