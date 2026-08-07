#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive viewer for ``control_quality``.

Run from any directory:

    /path/to/RoboDynaExp/script_exp/interactive_control_quality.py
    /path/to/RoboDynaExp/script_exp/interactive_control_quality.py --control robot

Select an arm, move over the red/green key, lower with Q to press. Skip black
tiles (do not press).
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
    print_instructions,
    UniversalRobotControls,
    make_viewer_view_toggle,
    report_task_result,
    sleep_to_timestep,
    terminal_hold_should_close,
    print_mode_controls,
)


CONTROLS_KEYBOARD = """
  Left Arrow        stamp RED (left arm)
  Right Arrow       stamp GREEN (right arm)

  Skip BLACK tiles — do not press while they are under the stamp.
"""

CONTROLS_ROBOT = """
  Select an arm (1/2), move over the matching colored key, lower with Q to press (E to raise).
  Skip BLACK tiles — do not press while they are under the stamp.
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
        task_name="control_quality",
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


class KeyboardStampController:
    """Non-blocking hover -> press -> lift animation for arrow-key taps."""

    TRANSITION_SECONDS = 0.12
    PRESS_HOLD_SECONDS = 0.10

    def __init__(self, env, arm_tag):
        self.env = env
        self.arm_tag = arm_tag
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

    @staticmethod
    def _side(color):
        return "left" if color == "red" else "right"

    def _drive_qpos(self, side):
        joints = self.env.robot.left_arm_joints if side == "left" else self.env.robot.right_arm_joints
        return np.asarray([joint.get_drive_target()[0] for joint in joints], dtype=np.float64)

    def _plan(self, side, pose, last_qpos=None):
        planner = self.env.robot.left_plan_path if side == "left" else self.env.robot.right_plan_path
        result = planner(pose, last_qpos=None if last_qpos is None else np.asarray(last_qpos, dtype=np.float32))
        if result is None or result.get("status") != "Success":
            reason = "no result" if result is None else result.get("reason", "unknown reason")
            raise RuntimeError(f"Could not prepare {side} quality-control key press: {reason}")
        return np.asarray(result["position"][-1], dtype=np.float64)

    def prepare(self):
        hover = float(self.env.KEY_HOVER_DIS)
        # Keep the TCP just above the key cap; the task's stamp fires at the
        # end of this downward transition rather than blocking the viewer loop.
        press_above = max(0.002, hover - float(self.env.KEY_PRESS_DEPTH))
        for color in ("red", "green"):
            side = self._side(color)
            hover_q = self._plan(side, self.env._key_tip_pose(color, hover))
            self.hover_qpos[color] = hover_q
            self.press_qpos[color] = self._plan(
                side, self.env._key_tip_pose(color, press_above), last_qpos=hover_q
            )

        # Ready hover is applied once before the belt starts (see main).
        self.hover_qpos["red"] = self._drive_qpos("left")
        self.hover_qpos["green"] = self._drive_qpos("right")
        print("Arrow-key stamp arms ready; each press animates a smooth key tap.")

    def _begin_transition(self, side, phase, target):
        self._starts[side] = self._drive_qpos(side)
        self._targets[side] = np.asarray(target, dtype=np.float64)
        self._started_at[side] = time.perf_counter()
        self._phase[side] = phase

    def tap(self, color):
        side = self._side(color)
        self._queued_taps[side].append(color)
        self._start_next_tap(side)

    def _start_next_tap(self, side):
        if self._phase[side] != "idle" or not self._queued_taps[side]:
            return
        color = self._queued_taps[side].pop(0)
        self._active_tap[side] = color
        self._begin_transition(side, "to_hover", self.hover_qpos[color])

    def _finish_transition(self, side, now):
        color = self._active_tap[side]
        phase = self._phase[side]
        if phase == "to_hover":
            self._begin_transition(side, "pressing", self.press_qpos[color])
        elif phase == "pressing":
            self.env._press_key(color)
            self._phase[side] = "holding"
            self._holding_until[side] = now + self.PRESS_HOLD_SECONDS
        elif phase == "raising":
            self._phase[side] = "idle"
            self._active_tap[side] = None
            self._start_next_tap(side)

    def update(self):
        now = time.perf_counter()
        for side in tuple(self._holding_until):
            color = self._active_tap[side]
            self.env.robot.set_arm_joints(self.press_qpos[color], np.zeros_like(self.press_qpos[color]), side)
            if now >= self._holding_until[side]:
                del self._holding_until[side]
                self._begin_transition(side, "raising", self.hover_qpos[color])
        for side in tuple(self._started_at):
            progress = min(1.0, (now - self._started_at[side]) / self.TRANSITION_SECONDS)
            smooth = progress * progress * (3.0 - 2.0 * progress)
            start, target = self._starts[side], self._targets[side]
            velocity = (target - start) / self.TRANSITION_SECONDS if progress < 1.0 else np.zeros_like(target)
            self.env.robot.set_arm_joints(start + (target - start) * smooth, velocity, side)
            if progress >= 1.0:
                del self._started_at[side]
                self._finish_transition(side, now)

    def release(self):
        self._starts.clear()
        self._targets.clear()
        self._started_at.clear()
        self._holding_until.clear()
        self._queued_taps = {"left": [], "right": []}


class ArrowPresses:
    """Emit one stamp request for each Left/Right Arrow press edge."""

    def __init__(self):
        self._previous = {"left": False, "right": False}

    def update(self, window, controller):
        for key, color in (("left", "red"), ("right", "green")):
            down = bool(window.key_down(key))
            if down and not self._previous[key]:
                controller.tap(color)
                print(f"Stamp requested: {color}")
            self._previous[key] = down


def _move_arms_to_ready(env):
    """Park both grippers above the keys before the belt starts rolling."""
    env._move_arms_to_ready()
    if not env.plan_success:
        detail = getattr(env, "_last_plan_fail", None) or "unknown planner failure"
        raise RuntimeError(f"Could not reach quality-control key ready pose: {detail}")
    print("Arms ready above the quality-control keys.")


def _start_belt(env):
    env._belt_running = True
    env._stamp_active = True


def _finalize_departed_tiles(env, last_under):
    """When a tile leaves the stamp without a mark, record skip/miss for scoring."""

    current = env._tile_under_stamp(require_unhandled=True)
    if last_under is None or last_under == current:
        return current
    i = last_under
    if env.tile_hidden[i] or env.tile_marked[i] or env.tile_skipped[i] or env.tile_missed[i]:
        return current
    if env.tile_colors[i] == "black":
        env.tile_skipped[i] = True
        print(f"Black tile {i} skipped (no press).")
    else:
        env._mark_missed_tile(i)
        print(f"Tile {i} ({env.tile_colors[i]}) missed — not stamped in time.")
    return current


def _episode_done(env):
    for i, color in enumerate(env.tile_colors):
        if env.tile_hidden[i]:
            continue
        if color == "black":
            if not (env.tile_skipped[i] or env.tile_marked[i]):
                return False
        else:
            if not (env.tile_marked[i] or env.tile_missed[i]):
                return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Interactive control_quality viewer")
    parser.add_argument("--config", default="demo_dynamic", help="Task config name without .yml")
    parser.add_argument("--seed", type=int, default=0, help="Scene randomization seed")
    parser.add_argument(
        "--control",
        choices=("keyboard", "robot"),
        default="robot",
        help="Interaction method (default: robot)",
    )
    args = parser.parse_args()

    from envs import CONFIGS_PATH
    from envs.control_quality import control_quality
    from envs.utils.action import ArmTag
    globals()["CONFIGS_PATH"] = CONFIGS_PATH

    print_mode_controls("control_quality", args.control, keyboard=CONTROLS_KEYBOARD, robot=CONTROLS_ROBOT)

    env = control_quality()
    env._interactive_robot_mode = True
    # Arrow presses / gripper teleop need planner support.
    env.setup_demo(**_configure_task(args.config, args.seed, use_robot=True))
    env.enable_interactive_tile_pause()
    # Keep the belt frozen while arms move to the key hover poses. Starting it
    # earlier burns the first tile's pause window before the operator can act.
    _move_arms_to_ready(env)

    stamp_controller = None
    arrow_presses = None
    if args.control == "keyboard":
        stamp_controller = KeyboardStampController(env, ArmTag)
        arrow_presses = ArrowPresses()
    last_under = None

    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    views = make_viewer_view_toggle(env, viewer)
    if views.robot_controls is None:
        views.robot_controls = UniversalRobotControls(env)
    # Start rolling only after both grippers are parked above the keys.
    _start_belt(env)
    last_frame_start = time.perf_counter()
    simulation_credit = 0.0

    print(f"Tile colors: {env.tile_colors}")
    if args.control == "keyboard":
        print_instructions("Keyboard arrows still animate key taps as a sandbox shortcut.")

    terminal_started_at = None

    try:
        while not viewer.closed:
            views.update(viewer.window)
            frame_start = time.perf_counter()
            if stamp_controller is not None and arrow_presses is not None:
                arrow_presses.update(viewer.window, stamp_controller)
                stamp_controller.update()

            last_under = _finalize_departed_tiles(env, last_under)
            under = env._tile_under_stamp(require_unhandled=True)
            if under is not None:
                last_under = under

            # The task uses a 250 Hz physics clock. Catch it up to elapsed wall
            # time so a 2-second, 500-step pause visibly lasts two seconds,
            # rather than 500 rendered frames.
            simulation_credit += frame_start - last_frame_start
            last_frame_start = frame_start
            timestep = float(env.scene.get_timestep())
            steps = min(32, int(simulation_credit / timestep))
            for _ in range(steps):
                env._update_kinematic_tasks()
                env.scene.step()
            simulation_credit -= steps * timestep
            env.scene.update_render()
            viewer.render()

            if viewer.window.key_down("escape"):
                break

            if terminal_started_at is not None:
                if terminal_hold_should_close(terminal_started_at):
                    break
                sleep_to_timestep(env, frame_start)
                continue

            if _episode_done(env):
                report_task_result(
                    env,
                    f"correct={sum(1 for c in env.tile_correct if c)}, "
                    f"missed={sum(1 for m in env.tile_missed if m)}, "
                    f"black_press={env.black_press}",
                )
                terminal_started_at = time.perf_counter()

            remaining = (1.0 / 60.0) - (time.perf_counter() - frame_start)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        try:
            if stamp_controller is not None:
                stamp_controller.release()
        finally:
            env.close_env()


if __name__ == "__main__":
    main()
    # household_task_gui convention: 0=SUCCESS, 10=FAILURE, 2=no result
    from _interactive_common import task_result_exit_code
    raise SystemExit(task_result_exit_code())
