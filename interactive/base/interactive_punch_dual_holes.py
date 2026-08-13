#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive viewer for ``punch_dual_holes``.

Run from any directory:

    /path/to/RoboDynaExp/interactive/base/interactive_punch_dual_holes.py --control keyboard
    /path/to/RoboDynaExp/interactive/base/interactive_punch_dual_holes.py --control robot

Keyboard mode calls ``_fire_punch`` via arrows. Robot mode: select an arm, move
over the key, lower with Q to press (ReactivePushButtons fires the punch).
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "script" / "bench_script"))
sys.path.insert(0, str(REPO_ROOT / "interactive"))

from _interactive_common import (  # noqa: E402
    print_instructions,
    UniversalRobotControls,
    make_viewer_view_toggle,
    add_robot_motion_arg,
    report_task_result,
    RealtimePhysicsPacer,
    terminal_hold_should_close,
    print_mode_controls,
    print_episode_condition,
)


CONTROLS_KEYBOARD = """
  Left Arrow / Right Arrow
                    fire punch on the matching belt

  Prefer --control robot for gripper-Z key presses.
"""

CONTROLS_ROBOT = """
  Select an arm, move over a key, lower with Q to press (E to raise).
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
        task_name="punch_dual_holes",
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
        _move_arms_to_ready(self.env)

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


def _move_arms_to_ready(env):
    """Hover both grippers above their buttons while belts stay inactive.

    Mirrors ``punch_dual_holes.play_once`` / ``_hover_button``: grasp the button
    top-down contact at ``pre_grasp_dis=grasp_dis=0.09`` so the closed gripper
    parks above the keycap without pressing. Belts remain off because
    ``_belt_active`` is still False and ``advance_belts=False``.
    """
    env.plan_success = True
    env._last_plan_fail = None
    env._move_with_belt_motion(
        env._hover_button("left"),
        env._hover_button("right"),
        advance_belts=False,
    )
    if not env.plan_success:
        detail = getattr(env, "_last_plan_fail", None) or "unknown planner failure"
        raise RuntimeError(f"Could not reach punch-button ready pose: {detail}")
    print("Arms ready above the punch buttons.")


def _start_belts(env):
    env._belt_active = True
    # Continuous mode advances from the task's mode flag.  Discrete mode is
    # started by _update_interactive_belt below and pauses at each punch stop.
    continuous = bool(getattr(env, "belt_continous_motion", False))
    env._belt_running = continuous
    env._interactive_discrete_stop_started = None
    env._interactive_discrete_stop_pages = None
    if continuous:
        print("Belts started (continuous motion).")
    else:
        pause = float(getattr(env, "tile_pause_s", 2.0))
        print(
            f"Belts started (discrete: hold under stamp ≤{pause:.2f}s "
            "or until stamped, whichever first)."
        )


def _update_interactive_belt(env):
    """Drive the belt according to the task's configured motion mode.

    Discrete (default): advance to the next tile arrival step, snap under the
    stamp, and HOLD until every ready tile is stamped OR ``tile_pause_s``
    elapses (miss then resume) — whichever comes first.

    Must run once per physics step so multi-step display frames cannot skip
    the single-step arrival window.
    """
    if bool(getattr(env, "belt_continous_motion", False)):
        env._belt_running = True
        return

    stop = getattr(env, "_interactive_discrete_stop_pages", None)
    if stop:
        env._belt_running = False
        for side, page_idx in list(stop.items()):
            if not env.page_punched[side][page_idx]:
                env._align_page_under_punch(side, page_idx)

        # Stamp-first: resume as soon as every tile in this stop is punched.
        if all(bool(env.page_punched[side][k]) for side, k in stop.items()):
            env._interactive_discrete_stop_pages = None
            env._interactive_discrete_stop_started = None
            env._belt_running = True
            return

        pause_s = max(0.0, float(getattr(env, "tile_pause_s", 2.0)))
        started = env._interactive_discrete_stop_started
        if started is not None and (time.perf_counter() - started) >= pause_s:
            for side, page_idx in stop.items():
                if not env.page_punched[side][page_idx]:
                    env._mark_missed_page(side, page_idx)
            env._interactive_discrete_stop_pages = None
            env._interactive_discrete_stop_started = None
            env._belt_running = True
        return

    # Cruise until the next unpunched tile's arrival step (play_once style).
    next_steps = []
    for side in ("left", "right"):
        k = env._next_unpunched_page(side)
        if k is not None:
            next_steps.append(env._page_arrival_step(side, k))
    if not next_steps:
        # All tiles stamped/missed — keep advancing so the last cards clear
        # the stamp (matches play_once final belt runout).
        env._belt_running = True
        return

    target = int(min(next_steps))
    if int(env._belt_step) < target:
        env._belt_running = True
        return

    ready = {}
    for side in ("left", "right"):
        k = env._next_unpunched_page(side)
        if k is None:
            continue
        if int(env._page_arrival_step(side, k)) == target:
            ready[side] = k
    if not ready:
        # Fallback: overlap-based ready (should be rare once arrival is hit).
        ready = dict(env._ready_pages_at_current_step())
    if not ready:
        env._belt_running = True
        return

    for side, page_idx in ready.items():
        env._align_page_under_punch(side, page_idx)
    env._interactive_discrete_stop_pages = dict(ready)
    env._interactive_discrete_stop_started = time.perf_counter()
    env._belt_running = False


def _all_pages_resolved(env):
    for side in ("left", "right"):
        for k in range(env.n_pages):
            if env.page_missing[side][k]:
                continue
            if not env.page_punched[side][k]:
                return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Interactive punch_dual_holes viewer")
    parser.add_argument("--config", default="demo_dynamic", help="Task config name without .yml")
    parser.add_argument("--seed", type=int, default=0, help="Scene randomization seed")
    add_robot_motion_arg(parser)
    args = parser.parse_args()

    from envs import CONFIGS_PATH
    from envs.punch_dual_holes import punch_dual_holes
    globals()["CONFIGS_PATH"] = CONFIGS_PATH

    print_mode_controls("punch_dual_holes", args.control, keyboard=CONTROLS_KEYBOARD, robot=CONTROLS_ROBOT)

    env = punch_dual_holes()
    env._interactive_robot_mode = True
    env.setup_demo(**_configure_task(args.config, args.seed, use_robot=True))
    env.together_close_gripper(save_freq=None)
    print_episode_condition(env)

    # Match play_once: approach buttons with belts inactive, then start motion.
    _move_arms_to_ready(env)
    _start_belts(env)

    edge = EdgeSides()

    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    views = make_viewer_view_toggle(env, viewer)
    if views.robot_controls is None:
        views.robot_controls = UniversalRobotControls(env)
    # Start in the same front/head-camera view normally reached with V.
    if getattr(views, "_head", None) is not None:
        views.mode = "head"
        views.apply(announce=False)

    print_instructions(
        "Arms start above the buttons; belts begin after that ready pose. "
        "Select an arm (1/2/3), move over a key, lower with Q to press. "
    )
    if args.control == "keyboard":
        print_instructions("Keyboard arrows still call _fire_punch directly as a sandbox shortcut.")

    terminal_started_at = None
    runout_start_step = None
    pacer = RealtimePhysicsPacer(env)

    try:
        while not viewer.closed:
            n_steps = pacer.begin_frame()
            views.update(viewer.window)
            if args.control == "keyboard":
                fired = edge.poll(_requested_sides(viewer.window))
                for side in fired:
                    env._fire_punch(side)
                if fired:
                    print(f"Punch fired: {', '.join(fired)}")

            continuous = bool(getattr(env, "belt_continous_motion", False))

            if n_steps == 0:
                _update_interactive_belt(env)
                env.scene.update_render()
                viewer.render()
                if viewer.window.key_down("escape"):
                    break
                if terminal_started_at is not None and terminal_hold_should_close(terminal_started_at):
                    break
                continue

            for _ in range(n_steps):
                # Discrete: decide stop/go every physics step so multi-step
                # display frames cannot skip the single-step arrival window.
                _update_interactive_belt(env)
                env._update_kinematic_tasks()
                env.scene.step()
            # Continuous only: miss tiles that slid past without a punch.
            # Discrete misses are handled by the stop timeout.
            if continuous:
                env._mark_overdue_pages()
            env.scene.update_render()
            viewer.render()

            if viewer.window.key_down("escape"):
                break

            if terminal_started_at is not None:
                if terminal_hold_should_close(terminal_started_at):
                    break
                continue

            if not _all_pages_resolved(env):
                runout_start_step = None
                continue

            # After the last tile, keep the belt moving (play_once runout) so
            # stamped cards clear the heads before reporting the result.
            if runout_start_step is None:
                runout_start_step = int(env._belt_step)
                print("Last tile resolved — running belt clear…")
                continue
            if int(env._belt_step) - runout_start_step < int(env._final_belt_runout_steps()):
                continue

            left_score = env._side_score("left")
            right_score = env._side_score("right")
            report_task_result(
                env,
                f"L={left_score:.2f} R={right_score:.2f} "
                f"empty_press={env.invalid_empty_press}",
            )
            terminal_started_at = time.perf_counter()
    finally:
        env.close_env()


if __name__ == "__main__":
    main()
    # household_task_gui convention: 0=SUCCESS, 10=FAILURE, 2=no result
    from _interactive_common import task_result_exit_code
    raise SystemExit(task_result_exit_code())
