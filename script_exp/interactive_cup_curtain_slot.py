#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive viewer for ``cup_curtain_slot``.

Run from any directory:

    /path/to/RoboDynaExp/script_exp/interactive_cup_curtain_slot.py --control keyboard
    /path/to/RoboDynaExp/script_exp/interactive_cup_curtain_slot.py --control robot

Keyboard mode nudges the cup to track the moving yellow gap. Robot mode grasps
the cup, tracks with arrow keys, then places on Space.
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

from _interactive_common import make_viewer_view_toggle, report_task_result, print_mode_controls  # noqa: E402


CONTROLS_KEYBOARD = """
  Arrow keys        track gap: nudge cup XY (L/R = x, U/D = y)
  Space             place/release cup at current pose
  T                 snap cup X to current yellow-gap center
  V                 toggle view: top-down ↔ head_camera
  Q / Escape         quit
------------------------------------------------------------
  Success: cup seated between yellow sticks; no curtain touch
"""

CONTROLS_ROBOT = """
  Arrow keys        track gap: nudge cup XY (L/R = x, U/D = y)
  Space             grasp cup, then place/release
  T                 snap cup X to current yellow-gap center
  V                 toggle view: top-down ↔ head_camera
  Q / Escape         quit
------------------------------------------------------------
  Success: cup seated between yellow sticks; no curtain touch
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
        task_name="cup_curtain_slot",
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


def _get_rigid(actor):
    for comp in actor.actor.get_components():
        if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent):
            return comp
    return None


def _set_cup_pose(env, x, y, z=None, kinematic=True):
    pose = env.cup.get_pose()
    if z is None:
        z = float(pose.p[2])
    new_pose = sapien.Pose([float(x), float(y), float(z)], pose.q)
    env.cup.set_pose(new_pose)
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


def _mark_deposit(env):
    env._attempt_active = True
    env._deposit_step = int(getattr(env, "_kin_step", 0))
    env._slot_x_at_deposit = float(env.slot_x())


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
        self._track = EdgeKey()
        # Hold cup kinematic until place.
        p = np.asarray(env.cup.get_pose().p, dtype=float)
        _set_cup_pose(env, p[0], p[1], p[2], kinematic=True)

    def update(self, window):
        if self._track.poll(window.key_down("t")) and not self.placed:
            p = np.asarray(self.env.cup.get_pose().p, dtype=float)
            _set_cup_pose(self.env, self.env.slot_x(), p[1], p[2], kinematic=True)
            print(f"Snapped cup X to gap ({self.env.slot_x():.3f}).")
        if not self.placed:
            dx, dy = _nudge_from_keys(window)
            if dx or dy:
                p = np.asarray(self.env.cup.get_pose().p, dtype=float)
                _set_cup_pose(self.env, p[0] + dx, p[1] + dy, p[2], kinematic=True)
        if self._space.poll(window.key_down("space")) and not self.placed:
            p = np.asarray(self.env.cup.get_pose().p, dtype=float)
            place_z = float(self.env.slot_z) + float(getattr(self.env, "cup_height", 0.04)) * 0.5
            _set_cup_pose(self.env, p[0], float(self.env.belt_y), place_z, kinematic=False)
            _mark_deposit(self.env)
            self.placed = True
            print(f"Cup placed at ({p[0]:.3f}, {self.env.belt_y:.3f}).")


class RobotCupController:
    def __init__(self, env, ArmTag):
        self.env = env
        self.ArmTag = ArmTag
        self.arm = ArmTag("left" if env.mirrored else "right")
        self.holding = False
        self.placed = False
        self.busy = False
        self._space = EdgeKey()
        self._track = EdgeKey()

    def grasp(self):
        self.busy = True
        contact_id, pre = self.env._find_cup_grasp(self.arm)
        if pre is None:
            print("No cup grasp pose found.")
            self.busy = False
            return
        self.env.move(self.env.close_gripper(self.arm, pos=0.6))
        self.env.move(self.env.grasp_actor(
            self.env.cup, arm_tag=self.arm, pre_grasp_dis=pre,
            gripper_pos=0.0, contact_point_id=contact_id,
        ))
        if self.env.plan_success:
            half = 0.5 * float(self.env.lift_z)
            self.env.move(self.env.move_by_displacement(self.arm, z=half))
            self.env.move(self.env.move_by_displacement(self.arm, z=self.env.lift_z - half))
            self.holding = True
            self.env._attempt_active = True
            print(f"Grasped cup with {self.arm}. Arrows track; T snaps X; Space places.")
        else:
            print("Grasp failed; planner disabled further robot actions.")
        self.busy = False

    def place(self):
        if not self.holding:
            return
        self.busy = True
        # Track gap then lower onto the belt row.
        self.env._nudge_x_to_slot(self.arm, max_step=0.08, lead_steps=20)
        cup_p = np.asarray(self.env.cup.get_pose().p, dtype=float)
        dy = float(self.env.belt_y - cup_p[1])
        if abs(dy) > 0.01:
            self.env.move(self.env.move_by_displacement(
                self.arm, y=float(np.clip(dy, -0.08, 0.08)), move_axis="world",
            ))
        target_z = float(self.env.slot_z) + 0.008
        dz = float(target_z - self.env.cup.get_pose().p[2])
        if abs(dz) > 0.005:
            self.env.move(self.env.move_by_displacement(self.arm, z=dz, move_axis="world"))
        self.env._nudge_x_to_slot(self.arm, max_step=0.05, lead_steps=10)
        _mark_deposit(self.env)
        self.env.move(self.env.open_gripper(self.arm))
        self.env.move(self.env.move_by_displacement(
            self.arm, z=float(self.env.post_place_lift_z), move_axis="world",
        ))
        self.holding = False
        self.placed = True
        print("Cup placed (robot).")
        self.busy = False

    def update(self, window):
        if self.busy:
            return
        if self._space.poll(window.key_down("space")):
            if not self.holding and not self.placed:
                self.grasp()
            elif self.holding:
                self.place()
            return
        if not self.holding or self.placed:
            return
        if self._track.poll(window.key_down("t")):
            self.env._nudge_x_to_slot(self.arm, max_step=0.10, lead_steps=25)
            print(f"Tracked gap x≈{self.env.slot_x():.3f}.")
            return
        dx, dy = _nudge_from_keys(window, step=0.02)
        if dx or dy:
            self.busy = True
            self.env.move(self.env.move_by_displacement(
                self.arm, x=dx, y=dy, move_axis="world",
            ))
            self.busy = False


def main():
    parser = argparse.ArgumentParser(description="Interactive cup_curtain_slot viewer")
    parser.add_argument("--config", default="demo_dynamic", help="Task config name without .yml")
    parser.add_argument("--seed", type=int, default=0, help="Scene randomization seed")
    parser.add_argument(
        "--control",
        choices=("keyboard", "robot"),
        default="keyboard",
        help="Interaction method (default: keyboard)",
    )
    parser.add_argument(
        "--robot-motion",
        choices=("planner", "interpolate"),
        default="planner",
        help="Robot motion backend (interpolate = faster joint interp when supported; default planner)",
    )
    args = parser.parse_args()

    from envs import CONFIGS_PATH
    from envs.cup_curtain_slot import cup_curtain_slot
    from envs.utils.action import ArmTag
    globals()["CONFIGS_PATH"] = CONFIGS_PATH

    print_mode_controls("cup_curtain_slot", args.control, keyboard=CONTROLS_KEYBOARD, robot=CONTROLS_ROBOT)
    if args.robot_motion == "interpolate":
        print(
            "Note: --robot-motion interpolate uses planner motions for this teleop task "
            "(key-press sandboxes use joint interpolation)."
        )

    env = cup_curtain_slot()
    env.setup_demo(**_configure_task(args.config, args.seed, use_robot=args.control == "robot"))
    print(
        f"Side={'left' if env.mirrored else 'right'}; "
        f"curtains={env.blue_curtains_enabled}; "
        f"gap x≈{env.slot_x():.3f}."
    )

    controller = (
        RobotCupController(env, ArmTag) if args.control == "robot"
        else KeyboardCupController(env)
    )

    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    views = make_viewer_view_toggle(env, viewer)

    placed_since = None
    try:
        while not viewer.closed:
            views.update(viewer.window)
            frame_start = time.perf_counter()
            controller.update(viewer.window)

            env._update_kinematic_tasks()
            env.scene.step()
            env.scene.update_render()
            viewer.render()

            if viewer.window.key_down("q") or viewer.window.key_down("escape"):
                break

            if getattr(controller, "placed", False):
                if placed_since is None:
                    placed_since = time.perf_counter()
                    print("Cup released; settling…")
                elif time.perf_counter() - placed_since >= 2.0:
                    hit = bool(getattr(env, "_curtain_hit", False))
                    detail = f"score={env.placement_score():.2f}"
                    if hit:
                        detail = f"curtain hit; {detail}"
                    report_task_result(env, detail)
                    break
            if getattr(env, "_curtain_hit", False) and placed_since is None:
                report_task_result(env, "curtain contact")
                break

            remaining = float(env.scene.get_timestep()) - (time.perf_counter() - frame_start)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        env.close_env()


if __name__ == "__main__":
    main()
