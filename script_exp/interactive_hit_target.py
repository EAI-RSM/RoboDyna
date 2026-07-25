#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive viewer for ``hit_target``.

Run from any directory:

    /path/to/RoboDynaExp/script_exp/interactive_hit_target.py --control keyboard
    /path/to/RoboDynaExp/script_exp/interactive_hit_target.py --control robot

Keyboard mode aims the dart tip with arrows and thrusts on Space. Robot mode
grasps the dart, aims, then jabs toward the yellow center.
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

from _interactive_common import make_viewer_view_toggle, report_task_result  # noqa: E402


CONTROLS = """
============================================================
 hit_target — interactive controls
============================================================
  Arrow keys        aim dart tip (L/R = x, U/D = y / depth)
  Space             keyboard: thrust tip at yellow center
                    robot: grasp dart, then jab / thrust
  T                 snap tip X to live bullseye
  V                 toggle view: top-down ↔ head_camera
  Q / Escape         quit
------------------------------------------------------------
  Success: tip sticks in yellow center; never hit a blocker
  --control keyboard  direct dart tip teleop (default)
  --control robot     grasp dart + jab helpers
  --robot-motion planner|interpolate
============================================================
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
        task_name="hit_target",
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


def _tip(env):
    return np.asarray(env.dart.get_functional_point(0, "list")[:3], dtype=float)


def _set_tip_xyz(env, tip_xyz, kinematic=True):
    """Translate the dart so its tip lands at tip_xyz (keeps orientation)."""
    tip = _tip(env)
    body = np.asarray(env.dart.get_pose().p, dtype=float)
    offset = tip - body
    new_body = np.asarray(tip_xyz, dtype=float) - offset
    pose = env.dart.get_pose()
    new_pose = sapien.Pose(new_body.tolist(), pose.q)
    env.dart.set_pose(new_pose)
    rigid = env._dart_rigid or _get_rigid(env.dart)
    env._dart_rigid = rigid
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
    dx = dy = dz = 0.0
    if window.key_down("left"):
        dx -= step
    if window.key_down("right"):
        dx += step
    if window.key_down("up"):
        dy += step
    if window.key_down("down"):
        dy -= step
    return dx, dy, dz


class EdgeKey:
    def __init__(self):
        self._prev = False

    def poll(self, down):
        edge = bool(down) and not self._prev
        self._prev = bool(down)
        return edge


class KeyboardDartController:
    def __init__(self, env):
        self.env = env
        self.done = False
        self._space = EdgeKey()
        self._snap = EdgeKey()
        tip = _tip(env)
        _set_tip_xyz(env, tip, kinematic=True)

    def update(self, window):
        if self.done or self.env._stuck or self.env._hit_blocker:
            return
        if self._snap.poll(window.key_down("t")):
            c = self.env._target_center_world()
            tip = _tip(self.env)
            _set_tip_xyz(self.env, [c[0], tip[1], c[2]], kinematic=True)
            print(f"Snapped tip X/Z to bullseye ({c[0]:.3f}, {c[2]:.3f}).")
        dx, dy, _ = _nudge_from_keys(window)
        if dx or dy:
            tip = _tip(self.env)
            _set_tip_xyz(self.env, tip + np.array([dx, dy, 0.0]), kinematic=True)
        if self._space.poll(window.key_down("space")):
            c = self.env._target_center_world()
            # Thrust tip onto the board plane at the live center.
            _set_tip_xyz(self.env, [c[0], c[1] - 0.005, c[2]], kinematic=True)
            self.env._check_blocker_hit()
            if not self.env._hit_blocker:
                self.env._try_form_stick()
            self.done = True
            status = (
                "STUCK on center" if self.env._stuck else
                "BLOCKER HIT" if self.env._hit_blocker else
                "missed"
            )
            print(f"Thrust: {status}.")


class RobotDartController:
    def __init__(self, env, ArmTag):
        self.env = env
        self.ArmTag = ArmTag
        self.arm = ArmTag("right" if env.dart_side > 0 else "left")
        self.side = float(env.dart_side)
        self.holding = False
        self.done = False
        self.busy = False
        self._space = EdgeKey()
        self._snap = EdgeKey()

    def grasp(self):
        self.busy = True
        self.env.move(self.env.grasp_actor(
            self.env.dart, arm_tag=self.arm, pre_grasp_dis=0.08, contact_point_id=0,
        ))
        if self.env.plan_success:
            self.env.move(self.env.move_by_displacement(
                self.arm, z=0.10, move_axis="world",
            ))
            self.env._align_tip_z(self.arm)
            self.env._go_to_standoff(self.arm)
            self.holding = True
            print(f"Grasped dart with {self.arm}. Arrows aim; T snaps X; Space jabs.")
        else:
            print("Dart grasp failed.")
        self.busy = False

    def jab(self):
        if not self.holding:
            return
        self.busy = True
        self.env._align_tip_z(self.arm)
        c = self.env._target_center_world()
        tip = _tip(self.env)
        # Close in Y toward the board while recentering X/Z.
        self.env._move_tip_x(self.arm, c[0], self.side, clear_blockers=True)
        tip = _tip(self.env)
        dy = float(c[1] - 0.012 - tip[1])
        if abs(dy) > 0.004:
            self.env.move(self.env.move_by_displacement(
                self.arm, y=float(np.clip(dy, -0.06, 0.06)), move_axis="world",
            ))
        self.env._dwell(30)
        self.done = True
        status = (
            "STUCK on center" if self.env._stuck else
            "BLOCKER HIT" if self.env._hit_blocker else
            "missed / settle"
        )
        print(f"Jab: {status}.")
        self.busy = False

    def update(self, window):
        if self.busy or self.done or self.env._stuck or self.env._hit_blocker:
            return
        if self._space.poll(window.key_down("space")):
            if not self.holding:
                self.grasp()
            else:
                self.jab()
            return
        if not self.holding:
            return
        if self._snap.poll(window.key_down("t")):
            c = self.env._target_center_world()
            self.env._align_tip_z(self.arm)
            self.env._move_tip_x(self.arm, c[0], self.side, clear_blockers=True)
            print(f"Snapped tip toward bullseye x={c[0]:.3f}.")
            return
        dx, dy, _ = _nudge_from_keys(window, step=0.02)
        if dx or dy:
            self.busy = True
            self.env.move(self.env.move_by_displacement(
                self.arm, x=dx, y=dy, move_axis="world",
            ))
            self.busy = False


def main():
    parser = argparse.ArgumentParser(description="Interactive hit_target viewer")
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
    from envs.hit_target import hit_target
    from envs.utils.action import ArmTag
    globals()["CONFIGS_PATH"] = CONFIGS_PATH

    print(CONTROLS)
    if args.robot_motion == "interpolate":
        print(
            "Note: --robot-motion interpolate uses planner motions for this teleop task "
            "(key-press sandboxes use joint interpolation)."
        )

    env = hit_target()
    env.setup_demo(**_configure_task(args.config, args.seed, use_robot=args.control == "robot"))
    print(
        f"Arm={'right' if env.dart_side > 0 else 'left'}; "
        f"blocker_static={env.blocker_enabled}; blocker_dyn={env.blocker_dynamic}."
    )

    controller = (
        RobotDartController(env, ArmTag) if args.control == "robot"
        else KeyboardDartController(env)
    )

    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    viewer.set_camera_xyz(0.0, -0.15, 1.7)
    viewer.set_camera_rpy(0.0, -0.9, -np.pi / 2.0)
    views = make_viewer_view_toggle(env, viewer)

    settle_after = None
    try:
        while not viewer.closed:
            views.update(viewer.window)
            frame_start = time.perf_counter()
            controller.update(viewer.window)

            env._update_kinematic_tasks()
            env.scene.step()
            if not env._stuck and not env._hit_blocker:
                env._try_form_stick()
            env.scene.update_render()
            viewer.render()

            if viewer.window.key_down("q") or viewer.window.key_down("escape"):
                break

            if env._hit_blocker:
                report_task_result(env, "blocker hit")
                break
            if env._stuck or getattr(controller, "done", False):
                if settle_after is None:
                    settle_after = time.perf_counter()
                elif time.perf_counter() - settle_after >= 1.0:
                    report_task_result(env)
                    break

            remaining = float(env.scene.get_timestep()) - (time.perf_counter() - frame_start)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        env.close_env()


if __name__ == "__main__":
    main()
