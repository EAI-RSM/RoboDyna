#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive viewer for ``play_billiard``.

Run from any directory:

    /path/to/RoboDynaExp/script_exp/interactive_play_billiard.py --control keyboard
    /path/to/RoboDynaExp/script_exp/interactive_play_billiard.py --control robot

Keyboard mode aims the cue and fires a strike impulse on Space. Robot mode
picks up the cue and runs a planned aim+strike on Space.
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
  Left / Right      rotate aim direction
  Up / Down         slide tip along aim (approach / retreat)
  Space             fire strike impulse along aim
  V                 toggle view: top-down ↔ head_camera
  Q / Escape         quit
------------------------------------------------------------
  Success: red ball in an allowed pocket; no distractor sink
"""

CONTROLS_ROBOT = """
  Left / Right      rotate aim direction
  Up / Down         slide tip along aim (approach / retreat)
  Space             pick up cue, then planned strike
  V                 toggle view: top-down ↔ head_camera
  Q / Escape         quit
------------------------------------------------------------
  Success: red ball in an allowed pocket; no distractor sink
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
        task_name="play_billiard",
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


def _default_aim(env):
    pocket, pid = env._choose_pocket()
    if not env.specific_hole:
        env._target_pocket = pocket
        env._target_pocket_id = pid
        env._target_pocket_name = env.POCKET_NAMES[pid]
    ball_xy = env._ball_xy(env.primary_ball)
    aim = pocket[:2] - ball_xy
    n = float(np.linalg.norm(aim))
    if n < 1e-6:
        aim = np.array([0.0, 1.0])
        n = 1.0
    env._aim_dir = aim / n
    return env._aim_dir


def _place_cue_for_aim(env, gap=None):
    """Park the cue tip behind the ball along ``_aim_dir`` (keyboard sandbox)."""
    if gap is None:
        gap = float(env.APPROACH_GAP)
    ball = np.asarray(env.primary_ball.get_pose().p, dtype=float)
    aim = np.asarray(env._aim_dir, dtype=float)
    tip_xy = ball[:2] - aim * gap
    tip = np.array([tip_xy[0], tip_xy[1], float(env.ball_z)], dtype=float)
    # Cue body: tip along +local X; park body so tip ≈ ball − aim*gap.
    yaw = float(np.arctan2(aim[1], aim[0]))
    q = _yaw_quat(yaw)
    half = float(env.CUE_HALF_LEN)
    new_body = tip - np.array([aim[0] * half, aim[1] * half, 0.0])
    new_pose = sapien.Pose(new_body.tolist(), q)
    env.cue.set_pose(new_pose)
    rigid = _get_rigid(env.cue)
    if rigid is not None:
        try:
            rigid.set_kinematic(True)
            rigid.set_linear_velocity(np.zeros(3))
            rigid.set_angular_velocity(np.zeros(3))
            rigid.set_kinematic_target(new_pose)
        except Exception:
            pass
    return tip


def _yaw_quat(yaw):
    """Quaternion (wxyz) for a yaw about +Z."""
    half = 0.5 * yaw
    return [float(np.cos(half)), 0.0, 0.0, float(np.sin(half))]


def _fire_impulse(env):
    rigid = env._primary_rigid or _get_rigid(env.primary_ball)
    env._primary_rigid = rigid
    if rigid is None:
        return False
    direction = np.array([env._aim_dir[0], env._aim_dir[1], 0.0], dtype=float)
    n = float(np.linalg.norm(direction))
    if n < 1e-6:
        return False
    direction /= n
    # Seat tip near the ball so the kinematic update path stays consistent.
    contact_gap = env.ball_radius + 2.0 * env.CUE_RADIUS + 0.002
    _place_cue_for_aim(env, gap=contact_gap)
    try:
        rigid.set_linear_velocity(direction * float(env.strike_impulse))
        rigid.set_angular_velocity(np.zeros(3))
        rigid.wake_up()
    except Exception:
        return False
    env._wake_all_balls()
    env._strike_done = True
    env._strike_armed = False
    return True


class EdgeKey:
    def __init__(self):
        self._prev = False

    def poll(self, down):
        edge = bool(down) and not self._prev
        self._prev = bool(down)
        return edge


class KeyboardCueController:
    def __init__(self, env):
        self.env = env
        self.struck = False
        self.gap = float(env.APPROACH_GAP)
        self._space = EdgeKey()
        _default_aim(env)
        _place_cue_for_aim(env, gap=self.gap)
        print(f"Aim → {env._target_pocket_name}; tip parked behind the red ball.")

    def update(self, window):
        if self.struck:
            return
        # Rotate aim
        rot = 0.0
        if window.key_down("left"):
            rot += 0.04
        if window.key_down("right"):
            rot -= 0.04
        if rot:
            c, s = np.cos(rot), np.sin(rot)
            ax, ay = self.env._aim_dir
            self.env._aim_dir = np.array([c * ax - s * ay, s * ax + c * ay], dtype=float)
            n = float(np.linalg.norm(self.env._aim_dir))
            self.env._aim_dir /= max(n, 1e-6)
            _place_cue_for_aim(self.env, gap=self.gap)
        if window.key_down("up"):
            self.gap = max(0.018, self.gap - 0.004)
            _place_cue_for_aim(self.env, gap=self.gap)
        if window.key_down("down"):
            self.gap = min(0.12, self.gap + 0.004)
            _place_cue_for_aim(self.env, gap=self.gap)
        if self._space.poll(window.key_down("space")):
            ok = _fire_impulse(self.env)
            self.struck = True
            print("Strike fired." if ok else "Strike failed (no rigid).")


class RobotCueController:
    def __init__(self, env, ArmTag):
        self.env = env
        self.ArmTag = ArmTag
        self.arm = ArmTag(env._arm_side)
        self.ready = False
        self.struck = False
        self.busy = False
        self._space = EdgeKey()
        _default_aim(env)

    def pickup_and_aim(self):
        self.busy = True
        if not self.env._pick_up_cue(self.arm):
            print("Cue pickup failed.")
            self.busy = False
            return
        hover_z = float(self.env.felt_top + self.env.HOVER_CLEARANCE)
        self.env._move_tip_z_to(self.arm, hover_z, max_step=0.09)
        self.env._seat_cue_for_strike(self.arm)
        self.env._move_tip_z_to(self.arm, hover_z, max_step=0.09)
        # Yaw tip toward aim.
        tip = self.env._tip_xyz()
        cue_p = np.asarray(self.env.cue.get_pose().p, dtype=float)
        tip_dir = tip[:2] - cue_p[:2]
        if float(np.linalg.norm(tip_dir)) < 1e-4:
            tip_ang = 0.0
        else:
            tip_ang = float(np.arctan2(tip_dir[1], tip_dir[0]))
        aim_ang = float(np.arctan2(self.env._aim_dir[1], self.env._aim_dir[0]))
        yaw_delta = (aim_ang - tip_ang + np.pi) % (2.0 * np.pi) - np.pi
        import transforms3d as t3d
        cur_q = np.array(self.env.get_arm_pose(str(self.arm))[3:], dtype=float)
        new_q = t3d.quaternions.qmult(
            t3d.quaternions.axangle2quat([0, 0, 1], float(np.clip(yaw_delta, -2.0, 2.0))),
            cur_q,
        )
        self.env.move(self.env.move_by_displacement(
            self.arm, quat=list(new_q), move_axis="world",
        ))
        ball_xy = self.env._ball_xy(self.env.primary_ball)
        behind = ball_xy - self.env._aim_dir * self.env.APPROACH_GAP
        tip = self.env._tip_xyz()
        self.env.move(self.env.move_by_displacement(
            self.arm,
            x=float(behind[0] - tip[0]),
            y=float(behind[1] - tip[1]),
            z=float(hover_z - tip[2]),
            move_axis="world",
        ))
        self.env._move_tip_z_to(self.arm, float(self.env.ball_z), max_step=0.05)
        self.env._strike_armed = True
        self.env._strike_done = False
        self.ready = True
        print("Cue ready behind the ball. Space strikes.")
        self.busy = False

    def strike(self):
        if not self.ready:
            return
        self.busy = True
        ball_xy = self.env._ball_xy(self.env.primary_ball)
        through = ball_xy + self.env._aim_dir * self.env.STRIKE_PUSH
        tip = self.env._tip_xyz()
        self.env.move(self.env.move_by_displacement(
            self.arm,
            x=float(through[0] - tip[0]),
            y=float(through[1] - tip[1]),
            move_axis="world",
        ))
        self.env._dwell(20)
        # Fallback impulse if contact did not fire.
        if not self.env._strike_done:
            _fire_impulse(self.env)
        self.struck = True
        print("Robot strike complete.")
        self.busy = False

    def update(self, window):
        if self.busy or self.struck:
            return
        if self._space.poll(window.key_down("space")):
            if not self.ready:
                self.pickup_and_aim()
            else:
                self.strike()
            return
        if not self.ready:
            return
        rot = 0.0
        if window.key_down("left"):
            rot += 0.03
        if window.key_down("right"):
            rot -= 0.03
        if rot:
            c, s = np.cos(rot), np.sin(rot)
            ax, ay = self.env._aim_dir
            self.env._aim_dir = np.array([c * ax - s * ay, s * ax + c * ay], dtype=float)
            self.env._aim_dir /= max(float(np.linalg.norm(self.env._aim_dir)), 1e-6)


def main():
    parser = argparse.ArgumentParser(description="Interactive play_billiard viewer")
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
    from envs.play_billiard import play_billiard
    from envs.utils.action import ArmTag
    globals()["CONFIGS_PATH"] = CONFIGS_PATH

    print_mode_controls("play_billiard", args.control, keyboard=CONTROLS_KEYBOARD, robot=CONTROLS_ROBOT)
    if args.robot_motion == "interpolate":
        print(
            "Note: --robot-motion interpolate uses planner motions for this teleop task "
            "(key-press sandboxes use joint interpolation)."
        )

    env = play_billiard()
    env.setup_demo(**_configure_task(args.config, args.seed, use_robot=args.control == "robot"))
    print(
        f"Arm={env._arm_side}; target pocket={env._target_pocket_name}; "
        f"specific_hole={env.specific_hole}; distractors={env.enable_distractors}."
    )

    controller = (
        RobotCueController(env, ArmTag) if args.control == "robot"
        else KeyboardCueController(env)
    )

    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    views = make_viewer_view_toggle(env, viewer)

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

            if viewer.window.key_down("q") or viewer.window.key_down("escape"):
                break

            if getattr(controller, "struck", False) or env._strike_done or env._primary_pocketed:
                if settle_after is None:
                    settle_after = time.perf_counter()
                    print("Ball in motion; settling…")
                elif time.perf_counter() - settle_after >= 3.0:
                    report_task_result(env)
                    break
            if env._robot_ball_contact:
                report_task_result(env, "robot touched ball")
                break

            remaining = float(env.scene.get_timestep()) - (time.perf_counter() - frame_start)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        env.close_env()


if __name__ == "__main__":
    main()
