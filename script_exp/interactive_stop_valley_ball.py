#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive viewer for ``stop_valley_ball``.

Run from any directory:

    /path/to/RoboDynaExp/script_exp/interactive_stop_valley_ball.py --control keyboard
    /path/to/RoboDynaExp/script_exp/interactive_stop_valley_ball.py --control robot

Press Space to grasp the bat, then use arrow keys for world XY and E/Q for
height while the gripper holds it.
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
    print_mode_controls,
    report_task_result,
    resolve_action_arm,
)


CONTROLS_KEYBOARD = """
  Space             place bat at the predicted intercept, then arm it
  Arrow keys        move bat in world XY
  E / Q             raise / lower bat height
  V                 toggle view: top-down ↔ head_camera
  Escape             quit
------------------------------------------------------------
  Flow: Space → use arrows/E/Q to aim
  Success: red ball hits the red circular bat head before
           falling to the table; handle contact does not count.
"""

CONTROLS_ROBOT = """
  Space             fast grasp and move bat to predicted intercept
  Arrow keys        move the held bat in world XY
  E / Q             raise / lower the held bat
  V                 toggle view: top-down ↔ head_camera
  Escape             quit
------------------------------------------------------------
  Flow: Space → use arrows/E/Q to aim
  Success: red ball hits the red circular bat head before
           falling to the table; handle contact does not count.
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
        task_name="stop_valley_ball",
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


def _intercept_xyz(env):
    if getattr(env, "intercept", None) is None:
        env._compute_intercept()
    return np.asarray(env.intercept, dtype=float)


def _get_rigid(actor):
    import sapien
    for comp in actor.actor.get_components():
        if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent):
            return comp
    return None


def _bat_face_quat(env):
    if hasattr(env, "_bat_face_quat"):
        return list(env._bat_face_quat())
    return [1.0, 0.0, 0.0, 0.0]


def _clamp_bat_xyz(env, x, y, z):
    """Keep the bat mid-air past the red line and within a reachable band."""
    min_clear = float(getattr(env, "INTERCEPT_MIN_CLEARANCE_DEFAULT", 0.06))
    min_z = float(env.table_top + min_clear + 0.5 * float(env.panel_radius))
    max_z = float(env.table_top + 0.28)
    z = float(np.clip(z, min_z, max_z))
    y = float(np.clip(y, -0.45, 0.30))
    # Stay on the exit side of the red line (same spirit as catch_valley place).
    side = float(getattr(env, "side", 1.0))
    red = float(env.red_line_x)
    if side > 0.0:
        x = float(np.clip(x, red + 0.02, 0.38))
    else:
        x = float(np.clip(x, -0.38, red - 0.02))
    return x, y, z


def _set_bat_xyz(env, x, y, z=None):
    import sapien
    pose = env.panel.get_pose()
    if z is None:
        z = float(pose.p[2])
    x, y, z = _clamp_bat_xyz(env, x, y, z)
    new_pose = sapien.Pose([float(x), float(y), float(z)], _bat_face_quat(env))
    try:
        env.panel.set_pose(new_pose)
    except Exception:
        env.panel.actor.set_pose(new_pose)
    # Keep aliases in sync (weld helpers use self.bowl).
    env.bowl = env.panel
    rigid = _get_rigid(env.panel)
    if rigid is not None:
        try:
            rigid.set_disable_gravity(True)
            rigid.set_kinematic(True)
            rigid.set_linear_velocity(np.zeros(3))
            rigid.set_angular_velocity(np.zeros(3))
            rigid.set_kinematic_target(new_pose)
        except Exception:
            pass
    return x, y, z


def _nudge_from_keys(window, xy_step=0.045, z_step=0.030):
    """Map arrows/E/Q to world-frame held-bat movement."""

    dx = xy_step * (window.key_down("right") - window.key_down("left"))
    dy = xy_step * (window.key_down("up") - window.key_down("down"))
    dz = z_step * (window.key_down("e") - window.key_down("q"))
    return float(dx), float(dy), float(dz)


class EdgeKey:
    def __init__(self):
        self._prev = False

    def poll(self, down):
        edge = bool(down) and not self._prev
        self._prev = bool(down)
        return edge


class KeyboardBatController:
    def __init__(self, env):
        self.env = env
        self.ready = False
        self._space = EdgeKey()
        # Start at the intercept plane; arrow keys provide all adjustments.
        ix, iy, iz = _intercept_xyz(env)
        _set_bat_xyz(env, ix, iy, iz)

    def update(self, window):
        if not self.ready:
            dx, dy, dz = _nudge_from_keys(window)
            if dx or dy or dz:
                p = np.asarray(self.env.panel.get_pose().p, dtype=float)
                _set_bat_xyz(self.env, p[0] + dx, p[1] + dy, p[2] + dz)
        if self._space.poll(window.key_down("space")):
            p = np.asarray(self.env.panel.get_pose().p, dtype=float)
            x, y, z = _set_bat_xyz(self.env, p[0], p[1], p[2])
            self.env._bowl_ready = True
            self.ready = True
            print(f"Bat armed mid-air at ({x:.3f}, {y:.3f}, {z:.3f}).")


class RobotBatMotion:
    """Fast non-blocking joint interpolation for a bat welded to one gripper."""

    DURATION = 0.04

    def __init__(self, env, arm):
        self.env = env
        self.arm = arm
        self.side = str(arm)
        self._start = None
        self._target = None
        self._started_at = None

    def _drive_qpos(self):
        joints = self.env.robot.left_arm_joints if self.side == "left" else self.env.robot.right_arm_joints
        return np.asarray([joint.get_drive_target()[0] for joint in joints], dtype=np.float64)

    def _ee_pose(self):
        get_pose = self.env.robot.get_left_ee_pose if self.side == "left" else self.env.robot.get_right_ee_pose
        return np.asarray(get_pose(), dtype=np.float64)

    def move_bat_to(self, x, y, z):
        """Queue a short world-frame move; ignore new keys until it completes."""

        if self._started_at is not None:
            return False
        x, y, z = _clamp_bat_xyz(self.env, x, y, z)
        panel = np.asarray(self.env.panel.get_pose().p, dtype=np.float64)
        pose = self._ee_pose().copy()
        pose[:3] += np.asarray([x, y, z], dtype=np.float64) - panel
        planner = self.env.robot.left_plan_path if self.side == "left" else self.env.robot.right_plan_path
        result = planner(pose.tolist(), last_qpos=np.asarray(self._drive_qpos(), dtype=np.float32))
        if result is None or result.get("status") != "Success":
            return False
        self._start = self._drive_qpos()
        self._target = np.asarray(result["position"][-1], dtype=np.float64)
        self._started_at = time.perf_counter()
        return True

    def update(self):
        if self._started_at is None:
            return
        progress = min(1.0, (time.perf_counter() - self._started_at) / self.DURATION)
        smooth = progress * progress * (3.0 - 2.0 * progress)
        delta = self._target - self._start
        position = self._start + delta * smooth
        velocity = delta / self.DURATION if progress < 1.0 else np.zeros_like(delta)
        self.env.robot.set_arm_joints(position, velocity, self.side)
        if progress >= 1.0:
            self._started_at = None

    @property
    def moving(self):
        return self._started_at is not None


class RobotBatController:
    def __init__(self, env, ArmTag):
        self.env = env
        self.ArmTag = ArmTag
        self.arm = None
        self.holding = False
        self.busy = False
        self._space = EdgeKey()
        self.motion = None

    def _choose_arm(self):
        return resolve_action_arm(self.env, self.ArmTag, exactly_one=True)

    def grasp(self):
        self.busy = True
        self.arm = self._choose_arm()
        if self.arm is None:
            self.busy = False
            return
        # Preserve the task's normal grasp sequence; it establishes the weld
        # offset used by the held-bat controller.
        self.env.move(self.env.grasp_actor(
            self.env.panel, arm_tag=self.arm, pre_grasp_dis=0.025, grasp_dis=0.025,
        ))
        if self.env.plan_success:
            self.env._weld_bowl_to_end_effector(self.arm)
            self.holding = True
            self.env._bowl_ready = True
            self.motion = RobotBatMotion(self.env, self.arm)
            panel = np.asarray(self.env.panel.get_pose().p, dtype=np.float64)
            # Clear the holder/table, then leave all lateral placement to teleop.
            self.motion.move_bat_to(panel[0], panel[1], panel[2] + 0.10)
            print(f"Picked up bat with {self.arm} arm. Use arrows/E/Q to adjust it.")
        else:
            action_failed(self.env, (str(self.arm),), detail="grasp failed")
        self.busy = False

    def nudge(self, window):
        if self.busy or not self.holding or self.motion is None:
            return
        dx, dy, dz = _nudge_from_keys(window)
        if not (dx or dy or dz):
            return
        p = np.asarray(self.env.panel.get_pose().p, dtype=np.float64)
        self.motion.move_bat_to(p[0] + dx, p[1] + dy, p[2] + dz)

    def update(self, window):
        if self.motion is not None:
            self.motion.update()
        if self.busy:
            return
        if self._space.poll(window.key_down("space")):
            if not self.holding:
                self.grasp()
            return
        # Universal viewer controls own arrow/E/Q motion.


def main():
    parser = argparse.ArgumentParser(description="Interactive stop_valley_ball viewer")
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
        help="Retained for compatibility; held-bat teleoperation uses fast interpolation.",
    )
    parser.add_argument(
        "--task-arg",
        action="append",
        default=[],
        help="Override task_args.stop_valley_ball entry, e.g. --task-arg wall_bounce_enabled=true",
    )
    args = parser.parse_args()

    from envs import CONFIGS_PATH
    from envs.stop_valley_ball import stop_valley_ball
    from envs.utils.action import ArmTag
    globals()["CONFIGS_PATH"] = CONFIGS_PATH

    print_mode_controls("stop_valley_ball", args.control, keyboard=CONTROLS_KEYBOARD, robot=CONTROLS_ROBOT)
    config = _configure_task(args.config, args.seed, use_robot=args.control == "robot")
    # Optional option toggles (same as record_demo --task-arg).
    targs = config.setdefault("task_args", {}).setdefault("stop_valley_ball", {})
    for item in args.task_arg:
        if "=" not in item:
            raise SystemExit(f"--task-arg expects key=value, got: {item}")
        key, raw = item.split("=", 1)
        key = key.strip()
        raw = raw.strip()
        if raw.lower() in ("true", "false"):
            targs[key] = raw.lower() == "true"
        else:
            try:
                targs[key] = int(raw) if raw.isdigit() else float(raw)
            except ValueError:
                targs[key] = raw

    env = stop_valley_ball()
    env.setup_demo(**config)
    env._interactive_selected_arms = (
        "left" if env.mirrored else "right",
    )
    # setup_demo already starts ball motion with expert_demo=False.
    ix, iy, iz = _intercept_xyz(env)
    print(
        f"Predicted intercept ≈ ({ix:.3f}, {iy:.3f}, {iz:.3f}); "
        f"red_line_x={env.red_line_x:.3f}; mirrored={env.mirrored}."
    )

    controller = (
        RobotBatController(env, ArmTag) if args.control == "robot" else KeyboardBatController(env)
    )

    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    views = make_viewer_view_toggle(env, viewer)

    print("Press Space to grasp the bat; arrows move XY and E/Q move height.")

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
                    print("Ball left the valley exit; waiting to settle…")
                elif time.perf_counter() - settle_after >= 2.5:
                    report_task_result(env)
                    break
            # Early fail if the ball hit the table before a head contact.
            if getattr(env, "_ball_table_before_hit", False) and not getattr(env, "_panel_hit", False):
                report_task_result(env)
                break

            remaining = float(env.scene.get_timestep()) - (time.perf_counter() - frame_start)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        env.close_env()


if __name__ == "__main__":
    main()
