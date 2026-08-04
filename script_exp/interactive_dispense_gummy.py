#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive viewer for ``dispense_gummy``.

Run from any directory:

    /path/to/RoboDynaExp/script_exp/interactive_dispense_gummy.py --control keyboard
    /path/to/RoboDynaExp/script_exp/interactive_dispense_gummy.py --control robot

Keyboard mode forces belt-key latches via arrows. Robot mode: select an arm,
move over a key, lower with Q to press (left → red dispense; right → belt keys).
Space is unused. Sandbox only — not data collection.
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
    UniversalRobotControls,
    make_viewer_view_toggle,
    add_robot_motion_arg,
    report_task_result,
    print_mode_controls,
)


CONTROLS_KEYBOARD = """
  Left Arrow       →  move bowl LEFT  (right-arm belt key)
  Right Arrow      →  move bowl RIGHT (right-arm belt key)
  Space is unused. Prefer --control robot: select arm, move over key, lower with Q
  (left arm → red dispense; right arm → belt keys).

  Continuous belt (Opt 2): hold an arrow key to slide.
  Discrete belt (default): tap an arrow key to hop one station.

  Forces belt-key latches directly (no arm). Dispense via gripper-Z in robot mode.
  V                 toggle view: top-down ↔ head_camera
  Close the viewer window to quit.
"""

CONTROLS_ROBOT = """
  Select left (1) or right (2) arm, move over a key, lower with Q to press
  (E to raise). Space is unused.

    Left arm  — hover over the red dispense key, lower with Q
    Right arm — hover over a left/right belt key, lower with Q

  Continuous belt (Opt 2): hold the gripper down on a belt key to slide.
  Discrete belt (default): press edge hops one station.

  Gripper-Z / ReactivePushButtons drives keys (no Space latch).
  V                 toggle view: top-down ↔ head_camera
  Close the viewer window to quit.
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
        task_name="dispense_gummy",
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


def _belt_side(window):
    left = window.key_down("left")
    right = window.key_down("right")
    if left and not right:
        return "left"
    if right and not left:
        return "right"
    return None


def _remaining_in_tubes(env):
    left = list(env._tube_stack_colors["left"][env._dispensed_count["left"]:])
    right = list(env._tube_stack_colors["right"][env._dispensed_count["right"]:])
    return left, right


def _unrecoverable(env):
    """Any target miss or distractor catch ends the episode as a failure."""
    target_missed = env.yellow_missed if env.target_color == "yellow" else env.blue_dropped
    distractor_caught = env.blue_caught if env.target_color == "yellow" else env.yellow_caught
    return int(target_missed) > 0 or int(distractor_caught) > 0


def _episode_done(env):
    """Return ``(done, detail)`` for a definitive success/failure state."""
    if bool(getattr(env, "invalid_pattern", False)):
        return True, "invalid_pattern"
    if _unrecoverable(env):
        target_missed = env.yellow_missed if env.target_color == "yellow" else env.blue_dropped
        distractor_caught = env.blue_caught if env.target_color == "yellow" else env.yellow_caught
        return True, f"unrecoverable (target_missed={target_missed}, distractor_caught={distractor_caught})"
    left, right = _remaining_in_tubes(env)
    if not left and not right and not getattr(env, "_active_drops", None):
        return True, "tubes empty"
    return False, None


def _tcp_xy(env, side: str) -> np.ndarray:
    getter = env.robot.get_left_tcp_pose if side == "left" else env.robot.get_right_tcp_pose
    return np.asarray(getter()[:2], dtype=np.float64)


def _key_xy(env, name: str) -> np.ndarray:
    if name == "dispense":
        return np.asarray([env.key_x, env.key_y], dtype=np.float64)
    return np.asarray(env.belt_key_xy[name], dtype=np.float64)


def _key_top_z(env, name: str) -> float:
    if name == "dispense":
        return float(env.dispense_key_top_z)
    return float(env.belt_key_top_z)


def _arm_for_key(name: str) -> str:
    return "left" if name == "dispense" else "right"


def _nearest_key_for_arm(env, side: str, max_dist: float = _KEY_XY_TOL):
    """Name of the nearest key under ``side``'s TCP, or None if too far."""
    tcp = _tcp_xy(env, side)
    candidates = ("dispense",) if side == "left" else ("left", "right")
    best_name, best_d = None, float(max_dist)
    for name in candidates:
        d = float(np.linalg.norm(_key_xy(env, name) - tcp))
        if d < best_d:
            best_d, best_name = d, name
    return best_name


class KeyboardState:
    """Arrow latches for belt only — dispense is gripper-Z (no Space)."""

    def update(self, env, window):
        env._expert_belt_hold = _belt_side(window)
        env._bowl_force_stop = False


class SmoothGummyPressController:
    """Non-blocking vertical key press from the current arm pose.

    Timed on the simulation clock (same approach as catch_marbles_trapdoors):
    the viewer advances a few ms of physics per frame, so wall-timed ramps
    finish before the fingertip arrives.
    """

    PRESS_SPEED = 0.70
    RAISE_SPEED = 1.00
    MIN_TRANSITION_SECONDS = 0.10
    MAX_TRANSITION_SECONDS = 0.90
    MIN_HOLD_SECONDS = 0.05
    MAX_HOLD_SECONDS = 0.25
    # Aim TCP slightly above the keycap so EE enters the task press band.
    TOUCH_DZ = 0.020
    MIN_DESCENT = 0.010
    MAX_DESCENT = 0.50
    MAX_PRESS_JOINT_TRAVEL = 1.80

    def __init__(self, env):
        self.env = env
        self.continuous = bool(getattr(env, "belt_continuous_motion", False))
        self.phase = "idle"
        self.side = None
        self.key = None
        self.start_qpos = None
        self.hover_qpos = None
        self.press_qpos = None
        self.started_at = None
        self.holding_from = None
        self.holding_until = None
        self.transition_seconds = self.MIN_TRANSITION_SECONDS
        self.descent = 0.0
        self._clock = 0.0
        self._hold_while_space = False
        self._space_held = False

    @property
    def busy(self):
        return self.phase != "idle"

    def _drive_qpos(self, side):
        joints = (
            self.env.robot.left_arm_joints
            if side == "left"
            else self.env.robot.right_arm_joints
        )
        return np.asarray(
            [joint.get_drive_target()[0] for joint in joints],
            dtype=np.float64,
        )

    def _tcp_z(self):
        getter = (
            self.env.robot.get_left_tcp_pose
            if self.side == "left"
            else self.env.robot.get_right_tcp_pose
        )
        return float(getter()[2])

    def _ik_joints(self, ee_pose7):
        solver = arm_ik(self.env, self.side)
        if solver is None:
            return None
        solution = solver.solve(ee_pose7)
        return None if solution is None else solution[0]

    def _plan_press_target(self):
        get_ee = (
            self.env.robot.get_left_ee_pose
            if self.side == "left"
            else self.env.robot.get_right_ee_pose
        )
        pose = np.asarray(get_ee(), dtype=np.float64).copy()
        desired_tcp_z = _key_top_z(self.env, self.key) + self.TOUCH_DZ
        descent = float(
            np.clip(
                self._tcp_z() - desired_tcp_z,
                self.MIN_DESCENT,
                self.MAX_DESCENT,
            )
        )
        pose[2] -= descent
        q = self._ik_joints(pose)
        if q is None:
            return None, 0.0
        start = self.hover_qpos
        target = np.asarray(q[: len(start)], dtype=np.float64)
        if float(np.max(np.abs(target - start))) > self.MAX_PRESS_JOINT_TRAVEL:
            return None, 0.0
        return target, descent

    def request(self, key: str, *, hold_while_space: bool = False):
        if self.busy:
            return False
        self.key = str(key)
        self.side = _arm_for_key(self.key)
        self.hover_qpos = self._drive_qpos(self.side)
        self.press_qpos, descent = self._plan_press_target()
        if self.press_qpos is None:
            print("Could not plan a smooth vertical key press.")
            self._reset()
            return False
        self.descent = descent
        self._hold_while_space = bool(hold_while_space)
        self._space_held = True
        self.env._interactive_teleop_locked = True
        self.env._expert_belt_hold = None
        self.env._expert_dispense = False
        self._begin_transition("pressing", self.press_qpos, self.PRESS_SPEED)
        return True

    def _begin_transition(self, phase, target, speed):
        self.phase = phase
        self.start_qpos = self._drive_qpos(self.side)
        self.target_qpos = np.asarray(target, dtype=np.float64)
        self.transition_seconds = float(np.clip(
            self.descent / speed,
            self.MIN_TRANSITION_SECONDS,
            self.MAX_TRANSITION_SECONDS,
        ))
        self.started_at = self._clock

    def _finish_transition(self, now):
        if self.phase == "pressing":
            self.phase = "holding"
            self.started_at = None
            self.holding_from = now
            if self._hold_while_space:
                self.holding_until = None
            else:
                self.holding_until = now + self.MAX_HOLD_SECONDS
        elif self.phase == "raising":
            self._reset()

    def set_space_held(self, held: bool):
        self._space_held = bool(held)

    def update(self):
        if self.phase == "idle":
            return
        self._clock += float(self.env.scene.get_timestep())
        now = self._clock
        if self.phase == "holding":
            self.env.robot.set_arm_joints(
                self.press_qpos,
                np.zeros_like(self.press_qpos),
                self.side,
            )
            if self._hold_while_space:
                if not self._space_held:
                    self._begin_transition("raising", self.hover_qpos, self.RAISE_SPEED)
                return
            settled = now >= self.holding_until
            if settled and now - self.holding_from >= self.MIN_HOLD_SECONDS:
                self._begin_transition("raising", self.hover_qpos, self.RAISE_SPEED)
            return

        progress = min(
            1.0,
            (now - self.started_at) / self.transition_seconds,
        )
        smooth = progress * progress * (3.0 - 2.0 * progress)
        delta = self.target_qpos - self.start_qpos
        velocity = (
            delta / self.transition_seconds
            if progress < 1.0
            else np.zeros_like(delta)
        )
        self.env.robot.set_arm_joints(
            self.start_qpos + delta * smooth,
            velocity,
            self.side,
        )
        if progress >= 1.0:
            self._finish_transition(now)

    def _reset(self):
        self.env._interactive_teleop_locked = False
        self.phase = "idle"
        self.side = None
        self.key = None
        self.start_qpos = None
        self.hover_qpos = None
        self.press_qpos = None
        self.started_at = None
        self.holding_from = None
        self.holding_until = None
        self._hold_while_space = False
        self._space_held = False

    def release(self):
        if self.busy and self.hover_qpos is not None and self.side is not None:
            self.env.robot.set_arm_joints(
                self.hover_qpos,
                np.zeros_like(self.hover_qpos),
                self.side,
            )
        self.env._expert_belt_hold = None
        self.env._expert_dispense = False
        self._reset()


def main():
    parser = argparse.ArgumentParser(description="Interactive dispense_gummy viewer")
    parser.add_argument("--config", default="demo_dynamic", help="Task config name without .yml")
    parser.add_argument("--seed", type=int, default=0, help="Scene randomization seed")
    add_robot_motion_arg(parser)
    args = parser.parse_args()

    from envs import CONFIGS_PATH
    from envs.dispense_gummy import dispense_gummy
    globals()["CONFIGS_PATH"] = CONFIGS_PATH

    print_mode_controls("dispense_gummy", args.control, keyboard=CONTROLS_KEYBOARD, robot=CONTROLS_ROBOT)

    env = dispense_gummy()
    # Always enable arm teleop: presses are gripper-Z only (no Space).
    env._interactive_robot_mode = True
    env.setup_demo(**_configure_task(args.config, args.seed, use_robot=True))
    env.together_close_gripper(save_freq=None)
    env._expert_belt_hold = None
    env._expert_dispense = False
    env._bowl_force_stop = False

    keyboard = KeyboardState() if args.control == "keyboard" else None

    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    views = make_viewer_view_toggle(env, viewer)
    if views.robot_controls is None:
        views.robot_controls = UniversalRobotControls(env)

    mode = "continuous" if getattr(env, "belt_continuous_motion", False) else "discrete"
    print(f"Belt mode: {mode}.")
    print(
        "Control=robot teleop. Select an arm (1/2), move over a key, lower with Q to press. "
        "Space is unused."
    )
    if args.control == "keyboard":
        print("Keyboard arrows still latch belt motion as a sandbox shortcut.")

    try:
        while not viewer.closed:
            views.update(viewer.window)
            frame_start = time.perf_counter()
            if keyboard is not None:
                keyboard.update(env, viewer.window)
            env._update_kinematic_tasks()
            env.scene.step()
            env.scene.update_render()
            viewer.render()
            if viewer.window.key_down("escape"):
                break
            done, detail = _episode_done(env)
            if done:
                report_task_result(env, detail)
                break
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
