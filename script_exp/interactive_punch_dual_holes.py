#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive viewer for ``dual_hole_punch``.

Run from any directory:

    /path/to/RoboDynaExp/script_exp/interactive_dual_hole_punch.py --control keyboard
    /path/to/RoboDynaExp/script_exp/interactive_dual_hole_punch.py --control robot
    /path/to/RoboDynaExp/script_exp/interactive_dual_hole_punch.py --control robot --robot-motion interpolate

Keyboard mode calls ``_fire_punch``. Robot mode taps the side button(s) then fires.
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
    edge_pressed,
    make_viewer_view_toggle,
    add_robot_motion_arg,
    report_task_result,
    print_mode_controls,
    require_selected_arms,
)


CONTROLS_KEYBOARD = """
  Left Arrow        punch LEFT belt
  Right Arrow       punch RIGHT belt
  Up Arrow          punch BOTH belts
  Counts only when half the stamp head is on the card; otherwise missed
  V                 toggle view: front ↔ top-down
  Escape             quit
------------------------------------------------------------
  Punches via direct _fire_punch (no arm motion).
"""

CONTROLS_ROBOT = """
  Space             punch with selected arm(s)
  Counts only when half the stamp head is on the card; otherwise missed
  V                 toggle view: front ↔ top-down
  Escape             quit
------------------------------------------------------------
  Arms tap the side buttons, then fire punch.
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
        task_name="dual_hole_punch",
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


def _requested_sides(window):
    if window.key_down("up"):
        return ("left", "right")
    left = window.key_down("left")
    right = window.key_down("right")
    if left and right:
        return ("left", "right")
    if left:
        return ("left",)
    if right:
        return ("right",)
    return ()


class EdgeSides:
    def __init__(self):
        self._prev = ()

    def poll(self, sides):
        key = tuple(sides)
        edge = bool(key) and key != self._prev
        self._prev = key
        return key if edge else ()


class RobotPunchController:
    """Dispense-gummy-style hover -> press -> hold -> raise controller."""

    TRANSITION_SECONDS = 0.12
    PRESS_HOLD_SECONDS = 0.10
    PRESS_DEPTH = 0.05

    def __init__(self, env, arm_tag, robot_motion):
        self.env = env
        self.arm_tag = arm_tag
        self.robot_motion = robot_motion  # Retained for CLI compatibility.
        self.hover_qpos = {}
        self.press_qpos = {}
        self._starts = {}
        self._targets = {}
        self._started_at = {}
        self._holding_until = {}
        self._phase = {"left": "idle", "right": "idle"}
        self._queued = {"left": 0, "right": 0}
        self.prepare()

    def _drive_qpos(self, side):
        joints = (
            self.env.robot.left_arm_joints
            if side == "left"
            else self.env.robot.right_arm_joints
        )
        return np.asarray(
            [joint.get_drive_target()[0] for joint in joints], dtype=np.float64
        )

    def _ee_pose(self, side):
        get_pose = (
            self.env.robot.get_left_ee_pose
            if side == "left"
            else self.env.robot.get_right_ee_pose
        )
        return np.asarray(get_pose(), dtype=np.float64)

    def _plan(self, side, pose, last_qpos):
        planner = (
            self.env.robot.left_plan_path
            if side == "left"
            else self.env.robot.right_plan_path
        )
        result = planner(pose, last_qpos=np.asarray(last_qpos, dtype=np.float32))
        if result is None or result.get("status") != "Success":
            reason = "no result" if result is None else result.get("reason", "unknown")
            raise RuntimeError(f"Could not prepare {side} punch-button press: {reason}")
        return np.asarray(result["position"][-1], dtype=np.float64)

    def prepare(self):
        """Execute the task's real hover routine, then cache a 5 cm press."""
        self.env.plan_success = True
        self.env._last_plan_fail = None
        self.env.move(
            self.env._hover_button("left"),
            self.env._hover_button("right"),
        )
        if not self.env.plan_success:
            detail = getattr(self.env, "_last_plan_fail", None) or "unknown planner failure"
            raise RuntimeError(f"Could not prepare punch-button hovers: {detail}")

        for side in ("left", "right"):
            hover = self._drive_qpos(side)
            press_pose = self._ee_pose(side)
            press_pose[2] -= self.PRESS_DEPTH
            self.hover_qpos[side] = hover
            self.press_qpos[side] = self._plan(side, press_pose, hover)
        print(
            "Punch arms ready above the buttons; each command presses 5.0 cm, "
            "fires on contact, then raises."
        )

    def _begin(self, side, phase, target):
        self._starts[side] = self._drive_qpos(side)
        self._targets[side] = np.asarray(target, dtype=np.float64)
        self._started_at[side] = time.perf_counter()
        self._phase[side] = phase

    def request(self, sides):
        for side in sides:
            self._queued[side] += 1
            if self._phase[side] == "idle":
                self._start_next(side)

    def _start_next(self, side):
        if self._phase[side] != "idle" or self._queued[side] <= 0:
            return
        self._queued[side] -= 1
        # Return directly above the button before every vertical press.
        self._begin(side, "to_hover", self.hover_qpos[side])

    def _finish_transition(self, side, now):
        phase = self._phase[side]
        if phase == "to_hover":
            self._begin(side, "pressing", self.press_qpos[side])
        elif phase == "pressing":
            self.env._fire_punch(side)
            print(f"Robot punched: {side}.")
            self._phase[side] = "holding"
            self._holding_until[side] = now + self.PRESS_HOLD_SECONDS
        elif phase == "raising":
            self._phase[side] = "idle"
            self._start_next(side)

    def update(self):
        now = time.perf_counter()
        for side in tuple(self._holding_until):
            target = self.press_qpos[side]
            self.env.robot.set_arm_joints(target, np.zeros_like(target), side)
            if now < self._holding_until[side]:
                continue
            del self._holding_until[side]
            self._begin(side, "raising", self.hover_qpos[side])

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

    def release(self):
        self._started_at.clear()
        self._holding_until.clear()
        self._queued = {"left": 0, "right": 0}
        self._phase = {"left": "idle", "right": "idle"}


def _start_belts(env):
    env._belt_active = True
    # Continuous mode advances from the task's mode flag.  Discrete mode is
    # started by _update_interactive_belt below and pauses at each punch stop.
    env._belt_running = bool(getattr(env, "belt_continous_motion", False))
    env._interactive_discrete_stop_started = None


def _update_interactive_belt(env):
    """Drive the belt according to the task's configured motion mode.

    ``dual_hole_punch`` uses ``_belt_running`` only as an explicit dwell-loop
    override. Keeping it true every frame turns the discrete option into a
    continuously moving belt, which is not the behavior of demo_dynamic.yml.
    """
    if bool(getattr(env, "belt_continous_motion", False)):
        env._belt_running = True
        return

    ready = env._ready_pages_at_current_step()
    if ready:
        now = time.perf_counter()
        if env._interactive_discrete_stop_started is None:
            env._interactive_discrete_stop_started = now
        # Match the normal discrete rollout: stop and center each ready page
        # under its punch head until the user presses the corresponding key.
        for side, page_idx in ready.items():
            env._align_page_under_punch(side, page_idx)
        pause_s = max(0.0, float(getattr(env, "tile_pause_s", 2.0)))
        if now - env._interactive_discrete_stop_started >= pause_s:
            # Match _handle_discrete_stamp_stop: an unpunched tile is a miss
            # when the finite pause window expires, then the belt resumes.
            for side, page_idx in ready.items():
                env._mark_missed_page(side, page_idx)
            env._interactive_discrete_stop_started = None
            env._belt_running = True
        else:
            env._belt_running = False
    else:
        env._interactive_discrete_stop_started = None
        env._belt_running = True


def _all_pages_resolved(env):
    for side in ("left", "right"):
        for k in range(env.n_pages):
            if env.page_missing[side][k]:
                continue
            if not env.page_punched[side][k]:
                return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Interactive dual_hole_punch viewer")
    parser.add_argument("--config", default="demo_dynamic", help="Task config name without .yml")
    parser.add_argument("--seed", type=int, default=0, help="Scene randomization seed")
    add_robot_motion_arg(parser)
    args = parser.parse_args()

    from envs import CONFIGS_PATH
    from envs.dual_hole_punch import dual_hole_punch
    from envs.utils.action import ArmTag
    globals()["CONFIGS_PATH"] = CONFIGS_PATH

    print_mode_controls("dual_hole_punch", args.control, keyboard=CONTROLS_KEYBOARD, robot=CONTROLS_ROBOT)

    env = dual_hole_punch()
    env.setup_demo(**_configure_task(args.config, args.seed, use_robot=args.control == "robot"))
    env.together_close_gripper(save_freq=None)

    robot_controller = None
    if args.control == "robot":
        robot_controller = RobotPunchController(env, ArmTag, args.robot_motion)
    # Match the main task: approach the buttons while belts are inactive, then
    # begin continuous motion or the first discrete run-to-stop.
    _start_belts(env)

    edge = EdgeSides()
    keys_prev: dict = {}

    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    views = make_viewer_view_toggle(env, viewer)
    # Start in the same front/head-camera view normally reached with V.
    if getattr(views, "_head", None) is not None:
        views.mode = "head"
        views.apply(announce=False)

    if args.control == "robot":
        print(
            f"Robot mode ready (motion={args.robot_motion}). "
            "Select arms with 1/2/3 and press Space to punch."
        )
    else:
        print("Keyboard mode ready. Tap Left/Right/Up Arrow when a card is under the stamp.")

    try:
        while not viewer.closed:
            views.update(viewer.window)
            frame_start = time.perf_counter()
            if args.control == "robot":
                if viewer.window.key_down("space"):
                    selected = tuple(getattr(env, "_interactive_selected_arms", ()))
                    if not selected and edge_pressed(viewer.window, "space", keys_prev):
                        require_selected_arms(env, exactly_one=False)
                    sides = selected
                else:
                    sides = ()
            else:
                sides = _requested_sides(viewer.window)
            fired = edge.poll(sides)
            if args.control == "keyboard":
                for side in fired:
                    env._fire_punch(side)
                if fired:
                    print(f"Punch fired: {', '.join(fired)}")
            elif robot_controller is not None:
                if fired:
                    robot_controller.request(fired)
                robot_controller.update()

            _update_interactive_belt(env)
            # Mark pages that slid past the stamp without a punch.
            env._mark_overdue_pages()

            env._update_kinematic_tasks()
            env.scene.step()
            env.scene.update_render()
            viewer.render()

            if viewer.window.key_down("escape"):
                break

            if _all_pages_resolved(env):
                # check_success() initializes punch_score_L/R, but the detail
                # string is evaluated before report_task_result calls it.
                left_score = env._side_score("left")
                right_score = env._side_score("right")
                report_task_result(
                    env,
                    f"L={left_score:.2f} R={right_score:.2f} "
                    f"empty_press={env.invalid_empty_press}",
                )
                time.sleep(1.5)
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
