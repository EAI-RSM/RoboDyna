#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive viewer for ``marble_shelf_maze``.

Run from any directory:

    /path/to/RoboDynaExp/script_exp/interactive_marble_shelf_maze.py --control keyboard
    /path/to/RoboDynaExp/script_exp/interactive_marble_shelf_maze.py --control robot

Keyboard mode tilts the active shelf via ``_press_tilt_direct`` (no arm).
Robot mode: select an arm, move over the shelf button, lower with Q to press.
Queued tilts run via ``consume_pending_tilt``. Space is unused. Sandbox only.
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
  Left Arrow       →  tilt active shelf LEFT  (red / left button)
  Right Arrow      →  tilt active shelf RIGHT (right button)
  Space is unused. Prefer --control robot: select arm, move over key, lower with Q.

  One press tilts the current shelf, rolls the marble off, and
  waits for the fall/settle before accepting another press.
  Hint: correct_dir for each shelf is printed at startup.

  Keys call the tilt API directly (no arm motion).
  V                 toggle view: top-down ↔ head_camera
  Close the viewer window to quit.
"""

CONTROLS_ROBOT = """
  Select left (1) or right (2) arm, move over the matching shelf button,
  then lower with Q to press (E to raise). Space is unused.

  One press tilts the current shelf, rolls the marble off, and
  waits for the fall/settle before accepting another press.
  Hint: correct_dir for each shelf is printed at startup.

  Gripper-Z depresses the key; reactive edge queues a tilt.
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
        task_name="marble_shelf_maze",
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


def _requested_direction(window):
    left = window.key_down("left")
    right = window.key_down("right")
    if left and not right:
        return "left"
    if right and not left:
        return "right"
    return None


class EdgeDirection:
    def __init__(self):
        self.prev = None

    def edge(self, window):
        cur = _requested_direction(window)
        fired = cur if cur is not None and cur != self.prev else None
        self.prev = cur
        return fired


def _with_live_viewer(env, viewer, fn):
    """Run a blocking tilt/press while refreshing the GUI each physics step."""
    original_step = env.scene.step

    def stepped():
        original_step()
        if viewer is not None and not viewer.closed:
            env.scene.update_render()
            viewer.render()

    env.scene.step = stepped
    try:
        return fn()
    finally:
        env.scene.step = original_step


class RobotShelfKeyController:
    """Per-frame hover → press → tilt → hover controller for the shelf keys."""

    TRANSITION_SECONDS = 0.14
    PRESS_HOLD_SECONDS = 0.08

    def __init__(self, env, arm_tag, viewer):
        self.env = env
        self.arm_tag = arm_tag
        self.viewer = viewer
        self.hover_qpos = {}
        self.press_qpos = {}
        self.active = None
        self.phase = "idle"
        self.start = None
        self.target = None
        self.started_at = None
        self.hold_until = None
        self._prepare()

    @property
    def busy(self):
        return self.phase != "idle"

    def _button(self, side):
        return self.env.left_button if side == "left" else self.env.right_button

    def _drive_qpos(self, side):
        joints = self.env.robot.left_arm_joints if side == "left" else self.env.robot.right_arm_joints
        return np.asarray([joint.get_drive_target()[0] for joint in joints], dtype=np.float64)

    def _plan(self, side, pose, last_qpos=None):
        planner = self.env.robot.left_plan_path if side == "left" else self.env.robot.right_plan_path
        result = planner(
            np.asarray(pose, dtype=np.float64).tolist(),
            last_qpos=None if last_qpos is None else np.asarray(last_qpos, dtype=np.float32),
        )
        if result is None or result.get("status") != "Success":
            reason = "no result" if result is None else result.get("reason", "unknown reason")
            raise RuntimeError(f"Could not prepare {side} shelf-key pose: {reason}")
        return np.asarray(result["position"][-1], dtype=np.float64)

    def _prepare(self):
        """Use the task's normal button approach, then cache its press endpoint."""
        for side in ("left", "right"):
            self.env.plan_success = True
            self.env.move(self.env.grasp_actor(
                self._button(side), arm_tag=self.arm_tag(side),
                pre_grasp_dis=0.09, grasp_dis=0.09,
                contact_point_id=0, gripper_pos=0.5,
            ))
            if not self.env.plan_success:
                detail = getattr(self.env, "_last_plan_fail", None) or "unknown planner failure"
                raise RuntimeError(f"Could not approach {side} shelf key: {detail}")
            hover = self._drive_qpos(side)
            ee_pose = np.asarray(
                self.env.robot.get_left_ee_pose() if side == "left" else self.env.robot.get_right_ee_pose(),
                dtype=np.float64,
            )
            ee_pose[2] -= float(self.env.button_press_depth)
            self.hover_qpos[side] = hover
            self.press_qpos[side] = self._plan(side, ee_pose, last_qpos=hover)
        print("Shelf-key arms ready; arrow taps descend, tilt, then return to hover.")

    def _begin(self, phase, target):
        side = self.active
        self.start = self._drive_qpos(side)
        self.target = np.asarray(target, dtype=np.float64)
        self.started_at = time.perf_counter()
        self.phase = phase

    def tap(self, direction):
        if self.busy:
            return False
        self.active = direction
        self._begin("pressing", self.press_qpos[direction])
        return True

    def _finish_transition(self, now):
        if self.phase == "pressing":
            self.phase = "holding"
            self.hold_until = now + self.PRESS_HOLD_SECONDS
        elif self.phase == "raising":
            self.phase = "idle"
            self.active = None
            self.start = self.target = self.started_at = self.hold_until = None

    def update(self):
        if self.phase == "idle":
            return
        now = time.perf_counter()
        if self.phase == "holding":
            self.env.robot.set_arm_joints(
                self.press_qpos[self.active], np.zeros_like(self.press_qpos[self.active]), self.active,
            )
            if now < self.hold_until:
                return
            # The direct API is the task's key-actuation behavior; wrap it so
            # the viewer stays live during the shelf tilt and marble landing.
            ok = _with_live_viewer(
                self.env, self.viewer,
                lambda: self.env._press_tilt_direct(self.active),
            )
            print(f"Tilt done. ball_mode={self.env._ball_mode} ok={ok}")
            self._begin("raising", self.hover_qpos[self.active])
            return

        progress = min(1.0, (now - self.started_at) / self.TRANSITION_SECONDS)
        smooth = progress * progress * (3.0 - 2.0 * progress)
        delta = self.target - self.start
        velocity = delta / self.TRANSITION_SECONDS if progress < 1.0 else np.zeros_like(delta)
        self.env.robot.set_arm_joints(self.start + delta * smooth, velocity, self.active)
        if progress >= 1.0:
            self._finish_transition(now)

    def release(self):
        self.phase = "idle"
        self.active = None


def main():
    parser = argparse.ArgumentParser(description="Interactive marble_shelf_maze viewer")
    parser.add_argument("--config", default="demo_dynamic", help="Task config name without .yml")
    parser.add_argument("--seed", type=int, default=0, help="Scene randomization seed")
    add_robot_motion_arg(parser)
    args = parser.parse_args()

    from envs import CONFIGS_PATH
    from envs.marble_shelf_maze import marble_shelf_maze
    globals()["CONFIGS_PATH"] = CONFIGS_PATH

    print_mode_controls("marble_shelf_maze", args.control, keyboard=CONTROLS_KEYBOARD, robot=CONTROLS_ROBOT)

    env = marble_shelf_maze()
    env._interactive_robot_mode = True
    # Raster viewer: pour_beer-style plain-alpha shelves/panes (transmission is invisible here).
    env._plain_glass = True
    env.setup_demo(**_configure_task(args.config, args.seed, use_robot=True))
    env.together_close_gripper(save_freq=None)
    env._bowl_armed = bool(getattr(env, "osc_bowl_enabled", False))
    env.plan_success = True

    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    views = make_viewer_view_toggle(env, viewer)
    if views.robot_controls is None:
        views.robot_controls = UniversalRobotControls(env)

    dirs = list(getattr(env, "correct_dir", []) or [])
    print(f"Shelves={env.n_shelves}. Suggested directions top→bottom: {dirs}")
    print(
        "Control=robot teleop. Select an arm (1/2), move over a key, lower with Q to press. "
        "Space is unused."
    )
    if args.control == "keyboard":
        print("Keyboard arrows still call _press_tilt_direct as a sandbox shortcut.")

    edges = EdgeDirection()

    try:
        while not viewer.closed:
            views.update(viewer.window)
            frame_start = time.perf_counter()
            if args.control == "keyboard":
                direction = edges.edge(viewer.window)
                mode = str(getattr(env, "_ball_mode", ""))
                can_tilt = (
                    direction is not None
                    and env.active_shelf_idx >= 0
                    and mode not in ("sliding", "falling", "missed", "done")
                )
                if can_tilt:
                    idx = env.active_shelf_idx
                    print(f"Tilting shelf {idx} {direction} (keyboard)...")
                    ok = _with_live_viewer(env, viewer, lambda: env._press_tilt_direct(direction))
                    print(f"Tilt done. ball_mode={env._ball_mode} ok={ok}")
            env._update_kinematic_tasks()
            if hasattr(env, "consume_pending_tilt") and getattr(env, "_pending_tilt_dir", None):
                _with_live_viewer(env, viewer, env.consume_pending_tilt)
            env.scene.step()
            env.scene.update_render()
            viewer.render()
            if viewer.window.key_down("escape"):
                break
            mode = str(getattr(env, "_ball_mode", ""))
            if mode in ("done", "missed") and int(getattr(env, "active_shelf_idx", 0)) < 0:
                report_task_result(env, f"ball_mode={mode}")
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
