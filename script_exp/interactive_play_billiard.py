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

from _interactive_common import (  # noqa: E402
    action_failed,
    try_interactive_grasp,
    make_viewer_view_toggle,
    print_mode_controls,
    report_task_result,
    RealtimePhysicsPacer,
    terminal_hold_should_close,
    resolve_action_arm,
    print_episode_condition,
)


CONTROLS_KEYBOARD = """
  Left / Right      rotate aim direction
  Up / Down         slide tip along aim (approach / retreat)
  Space             fire strike impulse along aim
"""

CONTROLS_ROBOT = """
  R / T             rotate gripper clockwise / counter-clockwise
  Space             pick up cue, then strike in the cue's current direction
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
    # ``cue`` is the task's Actor wrapper; pose updates belong to its
    # underlying SAPIEN actor (matching the main task's cue placement code).
    env.cue.actor.set_pose(new_pose)
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

    def _sync_selected_arm(self):
        selected = tuple(getattr(self.env, "_interactive_selected_arms", ()))
        if not self.ready and len(selected) == 1:
            self.arm = self.ArmTag(selected[0])

    def pickup_and_aim(self):
        """Pick only; the user positions and orients the cue afterwards."""
        self.busy = True
        arm = resolve_action_arm(self.env, self.ArmTag, exactly_one=True)
        if arm is None:
            self.busy = False
            return
        self.arm = arm
        # Keep the grasp point 3 cm higher than the prior fingertip-height
        # attempt.  Both values must move together because the task helper
        # applies a final finger-pad-height correction before closing.
        tip_clearance = float(self.env.CUE_RADIUS + 0.03)
        pad_to_ee_z = float(
            self.env.CUE_FINGER_EE_Z - self.env.CUE_TIP_GRASP_CLEARANCE
        )
        self.env.CUE_TIP_GRASP_CLEARANCE = tip_clearance
        self.env.CUE_FINGER_EE_Z = pad_to_ee_z + tip_clearance
        try:
            picked = bool(self.env._pick_up_cue(self.arm))
        except AssertionError:
            picked = False
            self.env.plan_success = False
        if not picked:
            action_failed(self.env, (str(self.arm),), detail="cue pickup failed")
            self.busy = False
            return
        self.ready = True
        print("Cue picked up. Position it with arrows/E/Q and rotate it with R/T; Space strikes.")
        self.busy = False

    def _move_gripper(self, x=0.0, y=0.0, z=0.0, quat=None):
        """Execute one small world-space manual motion without queued auto-aim."""
        # Block planned -Z once the welded cue is already on the felt.
        if float(z) < 0.0 and getattr(self.env, "_cue_welded", False):
            ee = np.asarray(self.env.get_arm_pose(str(self.arm)), dtype=float)
            trial = ee.copy()
            trial[2] += float(z)
            z_floor = self.env.interactive_ee_z_floor(str(self.arm), trial)
            if z_floor is not None:
                z = max(float(z), float(z_floor) - float(ee[2]))
                if z >= -1e-5:
                    return False
        self.env.plan_success = True
        ok = self.env.move(self.env.move_by_displacement(
            self.arm, x=x, y=y, z=z, quat=quat, move_axis="world",
        ))
        if ok is False or not self.env.plan_success:
            print("Requested gripper motion is unreachable.")
            self.env.plan_success = True
            return False
        self._lift_arm_off_felt_if_needed()
        return True

    def _lift_arm_off_felt_if_needed(self):
        """If a rotate/slide dipped the cue into the felt, raise the arm with it."""
        if not getattr(self.env, "_cue_welded", False):
            return
        ee = np.asarray(self.env.get_arm_pose(str(self.arm)), dtype=float)
        z_floor = self.env.interactive_ee_z_floor(str(self.arm), ee)
        if z_floor is None:
            return
        dz = float(z_floor) - float(ee[2])
        if dz <= 1e-4:
            return
        self.env.plan_success = True
        self.env.move(self.env.move_by_displacement(
            self.arm, z=dz, move_axis="world",
        ))
        self.env.plan_success = True

    def _rotate_gripper(self, yaw):
        import transforms3d as t3d

        cur_q = np.asarray(self.env.get_arm_pose(str(self.arm))[3:], dtype=float)
        turn = t3d.quaternions.axangle2quat([0.0, 0.0, 1.0], float(yaw))
        # Premultiplication applies the turn around the fixed world-Z axis.
        new_q = t3d.quaternions.qmult(turn, cur_q)
        self._move_gripper(quat=list(new_q))

    def strike(self):
        if not self.ready:
            return
        self.busy = True
        # The user has aimed manually, so strike forward along the actual cue
        # direction rather than moving it toward the task's automatic pocket aim.
        cue_dir = self.env._stick_tip_dir_xy()
        self.env._strike_armed = True
        self.env._strike_done = False
        self._move_gripper(
            x=float(cue_dir[0] * self.env.STRIKE_PUSH),
            y=float(cue_dir[1] * self.env.STRIKE_PUSH),
        )
        self.env._dwell(20)
        self.struck = True
        print("Robot strike complete." if self.env._strike_done else "Robot strike missed the ball.")
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
        # Clockwise is negative yaw when viewing the table from above.
        rot = 0.0
        if window.key_down("r"):
            rot -= 0.08
        if window.key_down("t"):
            rot += 0.08
        if rot:
            self.busy = True
            self._rotate_gripper(rot)
            self.busy = False


def main():
    parser = argparse.ArgumentParser(description="Interactive play_billiard viewer")
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
    env._interactive_selected_arms = (env._arm_side,)
    print_episode_condition(env)
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
    terminal_started_at = None
    pacer = RealtimePhysicsPacer(env)

    try:
        while not viewer.closed:
            n_steps = pacer.begin_frame()
            views.update(viewer.window)
            controller.update(viewer.window)

            if n_steps == 0:
                env.scene.update_render()
                viewer.render()
                if viewer.window.key_down("escape"):
                    break
                if terminal_started_at is not None and terminal_hold_should_close(terminal_started_at):
                    break
                continue

            for _ in range(n_steps):
                env._update_kinematic_tasks()
                env.scene.step()
            env.scene.update_render()
            viewer.render()

            if viewer.window.key_down("escape"):
                break

            if terminal_started_at is not None:
                if terminal_hold_should_close(terminal_started_at):
                    break
                continue

            if env._robot_ball_contact:
                report_task_result(env, "robot touched ball")
                terminal_started_at = time.perf_counter()
                continue
            if getattr(controller, "struck", False) or env._strike_done or env._primary_pocketed:
                if settle_after is None:
                    settle_after = time.perf_counter()
                    print("Ball in motion; settling…")
                elif time.perf_counter() - settle_after >= 3.0:
                    report_task_result(env)
                    terminal_started_at = time.perf_counter()
                    continue
    finally:
        env.close_env()


if __name__ == "__main__":
    main()
    # household_task_gui convention: 0=SUCCESS, 10=FAILURE, 2=no result
    from _interactive_common import task_result_exit_code
    raise SystemExit(task_result_exit_code())
