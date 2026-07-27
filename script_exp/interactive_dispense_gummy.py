#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive viewer for ``dispense_gummy``.

Run from any directory:

    /path/to/RoboDynaExp/script_exp/interactive_dispense_gummy.py --control keyboard
    /path/to/RoboDynaExp/script_exp/interactive_dispense_gummy.py --control robot
    /path/to/RoboDynaExp/script_exp/interactive_dispense_gummy.py --control robot --robot-motion interpolate

Keyboard mode forces belt-key / dispense latches. Robot mode: right arm for belt
keys, left arm for the dispense key. Sandbox only — not data collection.
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
    make_viewer_view_toggle,
    add_robot_motion_arg,
    report_task_result,
    print_mode_controls,
)


CONTROLS_KEYBOARD = """
  Left Arrow       →  move bowl LEFT  (right-arm belt key)
  Right Arrow      →  move bowl RIGHT (right-arm belt key)
  Space            →  dispense (left-arm key)

  Continuous belt (Opt 2): hold an arrow key to slide.
  Discrete belt (default): tap an arrow key to hop one station.

  Forces key presses / dispense request directly (no arm).
  V                 toggle view: top-down ↔ head_camera
  Close the viewer window to quit.
"""

CONTROLS_ROBOT = """
  Z                →  move bowl LEFT  (right-arm belt key)
  C                →  move bowl RIGHT (right-arm belt key)
  Space            →  dispense (left-arm key)

  Continuous belt (Opt 2): hold an arrow key to slide.
  Discrete belt (default): tap an arrow key to hop one station.

  Arms physically press the keys.
  --robot-motion planner|interpolate
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


class KeyboardState:
    def __init__(self):
        self.prev_space = False

    def update(self, env, window):
        env._expert_belt_hold = _belt_side(window)
        env._bowl_force_stop = False
        space = window.key_down("space")
        if space and not self.prev_space:
            env._expert_dispense = True
        self.prev_space = space


class RobotGummyController:
    """Cook-meat-style hover -> vertical press -> hover button controller."""

    TRANSITION_SECONDS = 0.12
    PRESS_HOLD_SECONDS = 0.10
    RELEASE_CLEARANCE = 0.04

    def __init__(self, env, arm_tag, robot_motion):
        self.env = env
        self.arm_tag = arm_tag
        self.robot_motion = robot_motion  # Kept for CLI compatibility.
        self.continuous = bool(getattr(env, "belt_continuous_motion", False))
        self.prev_space = False
        self._prev_belt = None
        self._continuous_belt = None
        self._starts = {}
        self._targets = {}
        self._started_at = {}
        self._holding_until = {}
        self._phase = {"left": "idle", "right": "idle"}
        self._queued_taps = {"left": [], "right": []}
        self._active_tap = {"left": None, "right": None}
        self.hover_qpos = {}
        self.press_qpos = {}
        self.prepare()

    def _drive_qpos(self, side):
        joints = self.env.robot.left_arm_joints if side == "left" else self.env.robot.right_arm_joints
        return np.asarray([joint.get_drive_target()[0] for joint in joints], dtype=np.float64)

    def _plan(self, side, pose, last_qpos=None):
        planner = self.env.robot.left_plan_path if side == "left" else self.env.robot.right_plan_path
        result = planner(pose, last_qpos=None if last_qpos is None else np.asarray(last_qpos, dtype=np.float32))
        if result is None or result.get("status") != "Success":
            reason = "no result" if result is None else result.get("reason", "unknown reason")
            raise RuntimeError(f"Could not prepare {side} gummy-key press: {reason}")
        return np.asarray(result["position"][-1], dtype=np.float64)

    @staticmethod
    def _arm_for_button(name):
        return "left" if name == "dispense" else "right"

    def _tip_pose(self, name, above):
        if name == "dispense":
            return self.env._key_tip_pose(above)
        return self.env._belt_key_tip_pose(name, above)

    def prepare(self):
        """Move to hover and cache planner-validated hover/press targets."""
        configured_hover = float(self.env.key_hover_dis)
        depth = float(self.env.key_press_depth)
        # Identical clearance calculation to CookKeyController: gummy also
        # detects presses from EE height while these helpers take TCP clearance.
        hover = max(
            configured_hover,
            float(self.env.belt_key_press_dz)
            - float(self.env.EE_TO_TCP)
            + self.RELEASE_CLEARANCE,
        )
        press_above = max(0.0, configured_hover - depth)

        for name in ("left", "right", "dispense"):
            side = self._arm_for_button(name)
            hover_q = self._plan(side, self._tip_pose(name, hover))
            self.hover_qpos[name] = hover_q
            self.press_qpos[name] = self._plan(
                side, self._tip_pose(name, press_above), last_qpos=hover_q
            )

        # CookKeyController closes the fingers and physically moves to hover
        # before enabling its cached non-blocking transitions. Do the same for
        # the right belt arm and left dispense arm.
        self.env.plan_success = True
        self.env._last_plan_fail = None
        self.env.move(
            self.env.close_gripper(self.arm_tag("right")),
            self.env.close_gripper(self.arm_tag("left")),
        )
        if self.env.plan_success:
            self.env.move(
                self.env.move_to_pose(self.arm_tag("right"), self._tip_pose("left", hover)),
                self.env.move_to_pose(self.arm_tag("left"), self._tip_pose("dispense", hover)),
            )
        if not self.env.plan_success:
            detail = getattr(self.env, "_last_plan_fail", None) or "unknown planner failure"
            raise RuntimeError(f"Could not prepare gummy-key hover poses: {detail}")

        # Capture the executed hover targets exactly, as cook_meat does.
        self.hover_qpos["left"] = self._drive_qpos("right")
        self.hover_qpos["dispense"] = self._drive_qpos("left")
        print(
            f"Gummy-key arms ready {hover * 100:.1f} cm above keys; "
            "each tap moves over the key, presses vertically, then raises."
        )

    def _begin_transition(self, side, phase, target):
        self._starts[side] = self._drive_qpos(side)
        self._targets[side] = np.asarray(target, dtype=np.float64)
        self._started_at[side] = time.perf_counter()
        self._phase[side] = phase

    def _tap(self, name):
        side = self._arm_for_button(name)
        self._queued_taps[side].append(name)
        self._start_next_tap(side)

    def _start_next_tap(self, side):
        if self._phase[side] != "idle" or not self._queued_taps[side]:
            return
        name = self._queued_taps[side].pop(0)
        self._active_tap[side] = name
        # Always go to the selected button's clear hover first. This is the
        # missing step that previously sent the arm diagonally toward press.
        self._begin_transition(side, "to_hover", self.hover_qpos[name])

    def _finish_transition(self, side, now):
        name = self._active_tap[side]
        phase = self._phase[side]
        if name is None:
            self._phase[side] = "idle"
            return
        if phase == "to_hover":
            self._begin_transition(side, "pressing", self.press_qpos[name])
        elif phase == "pressing":
            self._phase[side] = "holding"
            if not (
                self.continuous
                and side == "right"
                and name == self._continuous_belt
            ):
                self._holding_until[side] = now + self.PRESS_HOLD_SECONDS
        elif phase == "raising":
            self._phase[side] = "idle"
            self._active_tap[side] = None
            self._start_next_tap(side)

    def _advance(self):
        now = time.perf_counter()
        # Opt 2 matches cook_meat's held-key behavior: once the downward
        # transition finishes, keep commanding the press target until the
        # user releases the arrow key.
        if (
            self.continuous
            and self._phase["right"] == "holding"
            and "right" not in self._holding_until
            and self._active_tap["right"] is not None
        ):
            target = self.press_qpos[self._active_tap["right"]]
            self.env.robot.set_arm_joints(target, np.zeros_like(target), "right")

        for side in tuple(self._holding_until):
            self.env.robot.set_arm_joints(
                self.press_qpos[self._active_tap[side]],
                np.zeros_like(self.press_qpos[self._active_tap[side]]),
                side,
            )
            if now < self._holding_until[side]:
                continue
            del self._holding_until[side]
            name = self._active_tap[side]
            self._begin_transition(side, "raising", self.hover_qpos[name])

        for side in tuple(self._started_at):
            progress = min(
                1.0,
                (now - self._started_at[side]) / self.TRANSITION_SECONDS,
            )
            smooth = progress * progress * (3.0 - 2.0 * progress)
            start = self._starts[side]
            target = self._targets[side]
            velocity = (
                (target - start) / self.TRANSITION_SECONDS
                if progress < 1.0
                else np.zeros_like(target)
            )
            self.env.robot.set_arm_joints(
                start + (target - start) * smooth, velocity, side
            )
            if progress >= 1.0:
                del self._started_at[side]
                self._finish_transition(side, now)

    def _update_continuous_belt(self, requested):
        if requested == self._continuous_belt:
            return
        self._continuous_belt = requested
        side = "right"
        active = self._active_tap[side]
        self._queued_taps[side].clear()

        if active is not None:
            # Release the current key first, even when switching directly from
            # left to right. This resets the physical edge latch in the task.
            self._started_at.pop(side, None)
            self._holding_until.pop(side, None)
            if requested is not None:
                self._queued_taps[side].append(requested)
            self._begin_transition(side, "raising", self.hover_qpos[active])
        elif requested is not None:
            self._queued_taps[side].append(requested)
            self._start_next_tap(side)

    def update(self, window):
        requested = None
        if window.key_down("z"):
            requested = "left"
        elif window.key_down("c"):
            requested = "right"
        self.env._bowl_force_stop = False
        # Robot mode uses genuine EE contact; no expert latch is needed.
        self.env._expert_belt_hold = None
        if self.continuous:
            self._update_continuous_belt(requested)
        elif requested is not None and requested != self._prev_belt:
            self._tap(requested)
        self._prev_belt = requested

        space = window.key_down("space")
        if space and not self.prev_space:
            self._tap("dispense")
        self.prev_space = space
        self._advance()

    def release(self):
        self.env._expert_belt_hold = None
        self.env._expert_dispense = False
        self._started_at.clear()
        self._holding_until.clear()
        self._queued_taps = {"left": [], "right": []}
        self._active_tap = {"left": None, "right": None}
        self._phase = {"left": "idle", "right": "idle"}
        self._continuous_belt = None


def main():
    parser = argparse.ArgumentParser(description="Interactive dispense_gummy viewer")
    parser.add_argument("--config", default="demo_dynamic", help="Task config name without .yml")
    parser.add_argument("--seed", type=int, default=0, help="Scene randomization seed")
    add_robot_motion_arg(parser)
    args = parser.parse_args()

    from envs import CONFIGS_PATH
    from envs.dispense_gummy import dispense_gummy
    from envs.utils.action import ArmTag
    globals()["CONFIGS_PATH"] = CONFIGS_PATH

    print_mode_controls("dispense_gummy", args.control, keyboard=CONTROLS_KEYBOARD, robot=CONTROLS_ROBOT)

    env = dispense_gummy()
    env.setup_demo(**_configure_task(args.config, args.seed, use_robot=args.control == "robot"))
    env.together_close_gripper(save_freq=None)
    env._expert_belt_hold = None
    env._expert_dispense = False
    env._bowl_force_stop = False

    keyboard = KeyboardState()
    robot_controller = (
        RobotGummyController(env, ArmTag, args.robot_motion)
        if args.control == "robot"
        else None
    )

    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    views = make_viewer_view_toggle(env, viewer)

    mode = "continuous" if getattr(env, "belt_continuous_motion", False) else "discrete"
    print(f"Belt mode: {mode}. Control={args.control}. robot-motion={args.robot_motion}.")

    try:
        while not viewer.closed:
            views.update(viewer.window)
            frame_start = time.perf_counter()
            if args.control == "keyboard":
                keyboard.update(env, viewer.window)
            elif robot_controller is not None:
                robot_controller.update(viewer.window)
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
        try:
            if robot_controller is not None:
                robot_controller.release()
        finally:
            env.close_env()


if __name__ == "__main__":
    main()
