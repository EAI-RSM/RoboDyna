#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive viewer for ``put_cup_belt``.

Run from any directory:

    /path/to/RoboDynaExp/script_exp/interactive_put_cup_belt.py --control keyboard
    /path/to/RoboDynaExp/script_exp/interactive_put_cup_belt.py --control robot
    /path/to/RoboDynaExp/script_exp/interactive_put_cup_belt.py --control robot --robot-motion planner

Keyboard mode moves the cup in X/Y/Z. Robot mode grasps the cup on the first
Space press, moves it with the controls, then releases it at its current pose
on the second Space press. If the cup slips off the gripper, landing pose is
still scored (gripper open is not required).
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
    make_viewer_view_toggle,
    print_mode_controls,
    report_task_result,
    RealtimePhysicsPacer,
    terminal_hold_should_close,
    resolve_action_arm,
    try_interactive_grasp,
    print_episode_condition,
)


# Standalone-demo-only world-Y placement adjustment.  The task environment and
# its shared configuration files remain unchanged.  Negative Y moves the setup
# toward the robot/front of the scene.
INTERACTIVE_SETUP_Y_OFFSET = -0.06
# Gap between the belt's south edge and the cup's north edge.
INTERACTIVE_CUP_SOUTH_CLEARANCE = 0.05


CONTROLS_KEYBOARD = """
  Left / Right      move cup left/right (world X)
  Up / Down         move cup forward/backward (world Y)
  E / Q             raise/lower cup (world Z)
  Space             release cup at its current position
"""

CONTROLS_ROBOT = """
  Space             first press: grasp; second press: release at current pose
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
        task_name="put_cup_belt",
        render_freq=1,
        now_ep_num=0,
        seed=seed,
        need_plan=use_robot,
        save_data=False,
    )
    table_xy_bias = list(config.get("table_xy_bias", [0.0, 0.0]))
    if len(table_xy_bias) < 2:
        table_xy_bias = [0.0, 0.0]
    table_xy_bias[1] = float(table_xy_bias[1]) + INTERACTIVE_SETUP_Y_OFFSET
    config["table_xy_bias"] = table_xy_bias

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


def _set_cup_pose(env, x, y, z=None, kinematic=True, quat=None):
    pose = env.cup.get_pose()
    if z is None:
        z = float(pose.p[2])
    if quat is None:
        quat = pose.q
    new_pose = sapien.Pose([float(x), float(y), float(z)], quat)
    env.cup.actor.set_pose(new_pose)
    rigid = _get_rigid(env.cup)
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


def _place_cup_south_of_belt(env, hold_kinematic):
    """Place the cup 5 cm clear of the belt's south edge."""
    pose = env.cup.get_pose()
    belt_half_y = float(getattr(env, "belt_plate_half_size", [0.0, 0.0])[1])
    cup_half_y = 0.5 * float(env._actor_world_size(env.cup)[1])
    target_y = (
        float(env.belt_y)
        - belt_half_y
        - cup_half_y
        - INTERACTIVE_CUP_SOUTH_CLEARANCE
    )
    _set_cup_pose(
        env,
        float(pose.p[0]),
        target_y,
        0.74 + float(env.table_z_bias),
        kinematic=hold_kinematic,
        quat=env.CUP_UPRIGHT_QPOS,
    )
    env.cup_y = target_y
    print(f"Starting cup 5 cm south of belt at y={target_y:.3f}.")


def _nudge_from_keys(
    window, lateral_step=0.012, longitudinal_step=0.012, vertical_step=0.018
):
    """Return the requested (world X, world Y, world Z) displacement."""
    dx = 0.0
    if window.key_down("left"):
        dx -= lateral_step
    if window.key_down("right"):
        dx += lateral_step
    dy = 0.0
    if window.key_down("up"):
        dy += longitudinal_step
    if window.key_down("down"):
        dy -= longitudinal_step
    dz = 0.0
    if window.key_down("q"):
        dz += vertical_step
    if window.key_down("e"):
        dz -= vertical_step
    return dx, dy, dz


def _mark_deposit(env):
    env._attempt_active = True
    env._deposit_step = int(getattr(env, "_kin_step", 0))
    env._slot_x_at_deposit = float(env.slot_x())


class EdgeKey:
    def __init__(self):
        self._prev = False

    def poll(self, down):
        edge = bool(down) and not self._prev
        self._prev = bool(down)
        return edge


class KeyboardCupController:
    def __init__(self, env):
        self.env = env
        self.placed = False
        self._space = EdgeKey()
        # Hold cup kinematic until place.
        p = np.asarray(env.cup.get_pose().p, dtype=float)
        _set_cup_pose(env, p[0], p[1], p[2], kinematic=True)

    def update(self, window):
        if not self.placed:
            dx, dy, dz = _nudge_from_keys(window)
            if dx or dy or dz:
                p = np.asarray(self.env.cup.get_pose().p, dtype=float)
                _set_cup_pose(
                    self.env, p[0] + dx, p[1] + dy, p[2] + dz, kinematic=True
                )
        if self._space.poll(window.key_down("space")) and not self.placed:
            p = np.asarray(self.env.cup.get_pose().p, dtype=float)
            _set_cup_pose(self.env, p[0], p[1], p[2], kinematic=False)
            _mark_deposit(self.env)
            self.placed = True
            print(f"Cup released at ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}).")


class RobotCupController:
    DIRECT_CARTESIAN_STEP = 0.035
    DIRECT_JOINT_STEPS = 10
    MAX_LOCAL_JOINT_DELTA = 0.70

    def __init__(self, env, ArmTag, robot_motion="interpolate"):
        self.env = env
        self.ArmTag = ArmTag
        self.arm = ArmTag("left" if env.mirrored else "right")
        self.side = str(self.arm)
        self.robot_motion = robot_motion
        self.holding = False
        self.placed = False
        self.busy = False
        self._space = EdgeKey()
        # Require sustained loss of finger contact before treating a drop as a slip
        # (avoids one-frame PhysX contact flicker right after grasp).
        self._hold_contact_seen = False
        self._no_contact_steps = 0
        self._slip_no_contact_steps = 8

    def _drive_qpos(self):
        joints = (
            self.env.robot.left_arm_joints
            if self.side == "left"
            else self.env.robot.right_arm_joints
        )
        return np.asarray(
            [joint.get_drive_target()[0] for joint in joints], dtype=np.float64
        )

    def _render_physics_step(self):
        self.env._update_kinematic_tasks()
        self.env.scene.step()
        viewer = getattr(self.env, "viewer", None)
        if viewer is not None:
            self.env.scene.update_render()
            viewer.render()

    def _direct_joint_target(self, ee_pose):
        """Plan a nearby endpoint, rejecting IK branches that would flip the arm."""
        planner = (
            self.env.robot.left_plan_path
            if self.side == "left"
            else self.env.robot.right_plan_path
        )
        result = planner(
            np.asarray(ee_pose, dtype=np.float64).tolist(),
            constraint_pose=[1, 1, 1, 0, 0, 0],
        )
        if result is None or result.get("status") != "Success":
            reason = "no result" if result is None else result.get("reason", "unknown")
            print(f"Direct placement endpoint failed: {reason}.")
            return None
        positions = result.get("position")
        if positions is None or len(positions) == 0:
            print("Direct placement endpoint returned no joint target.")
            return None
        start = self._drive_qpos()
        target = np.asarray(positions[-1], dtype=np.float64).reshape(-1)
        if target.shape != start.shape:
            print(
                f"Direct placement joint mismatch: start={start.shape}, "
                f"target={target.shape}."
            )
            return None
        largest_delta = float(np.max(np.abs(target - start)))
        if largest_delta > self.MAX_LOCAL_JOINT_DELTA:
            print(
                "Direct placement refused an IK branch change "
                f"({largest_delta:.2f} rad joint jump)."
            )
            return None
        return start, target

    def _interpolate_to_ee_pose(self, ee_pose):
        endpoints = self._direct_joint_target(ee_pose)
        if endpoints is None:
            return False
        start, target = endpoints
        delta = target - start
        for index in range(1, self.DIRECT_JOINT_STEPS + 1):
            alpha = index / float(self.DIRECT_JOINT_STEPS)
            smooth = alpha * alpha * (3.0 - 2.0 * alpha)
            position = start + delta * smooth
            velocity = delta / float(self.DIRECT_JOINT_STEPS)
            self.env.robot.set_arm_joints(position, velocity, self.side)
            self._render_physics_step()
        self.env.robot.set_arm_joints(target, np.zeros_like(target), self.side)
        for _ in range(2):
            self._render_physics_step()
        self.env.plan_success = True
        self.env._last_plan_fail = None
        return True

    def _interpolate_cup_axis(self, axis, target_value, selected_x, tolerance):
        """Move the held cup along one world axis using short local IK segments."""
        axis_index = {"y": 1, "z": 2}[axis]
        max_segments = 10
        for _ in range(max_segments):
            cup_p = np.asarray(self.env.cup.get_pose().p, dtype=np.float64)
            remaining = float(target_value - cup_p[axis_index])
            x_error = float(selected_x - cup_p[0])
            if abs(remaining) <= tolerance and abs(x_error) <= 0.004:
                return True
            ee_pose = np.asarray(
                self.env.robot.get_left_ee_pose()
                if self.side == "left"
                else self.env.robot.get_right_ee_pose(),
                dtype=np.float64,
            ).copy()
            ee_pose[0] += x_error
            ee_pose[axis_index] += float(
                np.clip(
                    remaining,
                    -self.DIRECT_CARTESIAN_STEP,
                    self.DIRECT_CARTESIAN_STEP,
                )
            )
            if not self._interpolate_to_ee_pose(ee_pose):
                return False
        cup_value = float(self.env.cup.get_pose().p[axis_index])
        print(
            f"Direct placement could not reach cup {axis}={target_value:.3f}; "
            f"stopped at {cup_value:.3f}."
        )
        return False

    def _interpolate_ee_lift(self, distance):
        remaining = float(distance)
        while remaining > 0.003:
            step = min(remaining, self.DIRECT_CARTESIAN_STEP)
            ee_pose = np.asarray(
                self.env.robot.get_left_ee_pose()
                if self.side == "left"
                else self.env.robot.get_right_ee_pose(),
                dtype=np.float64,
            ).copy()
            ee_pose[2] += step
            if not self._interpolate_to_ee_pose(ee_pose):
                return False
            remaining -= step
        return True

    def _interpolate_ee_nudge(self, dx, dy, dz):
        """Move the held cup responsively with one nearby IK endpoint."""
        ee_pose = np.asarray(
            self.env.robot.get_left_ee_pose()
            if self.side == "left"
            else self.env.robot.get_right_ee_pose(),
            dtype=np.float64,
        ).copy()
        ee_pose[:3] += np.asarray([dx, dy, dz], dtype=np.float64)
        return self._interpolate_to_ee_pose(ee_pose)

    def grasp(self):
        self.busy = True
        self.arm = resolve_action_arm(self.env, self.ArmTag, exactly_one=True)
        if self.arm is None:
            self.busy = False
            return
        self.side = str(self.arm)
        contact_id, pre = self.env._find_cup_grasp(self.arm)
        if pre is None:
            action_failed(self.env, (self.side,), detail="no cup grasp pose")
            self.busy = False
            return
        self.env.move(self.env.close_gripper(self.arm, pos=0.6))
        if try_interactive_grasp(
            self.env, self.env.cup, self.arm, pre_grasp_dis=pre,
            gripper_pos=0.0, contact_point_id=contact_id,
        ):
            half = 0.5 * float(self.env.lift_z)
            self.env.move(self.env.move_by_displacement(self.arm, z=half))
            self.env.move(self.env.move_by_displacement(self.arm, z=self.env.lift_z - half))
            self.holding = True
            self._hold_contact_seen = bool(self.env._cup_held())
            self._no_contact_steps = 0
            self.env._attempt_active = True
            print(
                f"Grasped cup with {self.arm}. Use arrows for X/Y, E/Q for Z; "
                "Space releases here."
            )
        self.busy = False

    def place(self):
        if not self.holding:
            return
        self.busy = True
        # Do not reposition automatically: release at exactly the pose selected
        # by the user with the universal Cartesian controls.
        released_pose = np.asarray(self.env.cup.get_pose().p, dtype=float)
        _mark_deposit(self.env)
        self.env.move(self.env.open_gripper(self.arm))
        self.holding = False
        self.placed = True
        if self.robot_motion == "interpolate":
            if not self._interpolate_ee_lift(float(self.env.post_place_lift_z)):
                print("Cup released, but the direct post-place lift could not finish.")
        else:
            self.env.move(self.env.move_by_displacement(
                self.arm, z=float(self.env.post_place_lift_z), move_axis="world",
            ))
        print(
            f"Cup released at ({released_pose[0]:.3f}, {released_pose[1]:.3f}, "
            f"{released_pose[2]:.3f})."
        )
        self.busy = False

    def update(self, window):
        if self.busy:
            return
        # Slip-off: sustained loss of finger contact (gripper need not open).
        if self.holding and not self.placed:
            if self.env._cup_held():
                self._hold_contact_seen = True
                self._no_contact_steps = 0
            else:
                self._no_contact_steps += 1
                # After a real pinch, a short no-contact streak means slip; if grasp
                # never registered contact, wait longer before giving up.
                limit = (
                    self._slip_no_contact_steps
                    if self._hold_contact_seen
                    else self._slip_no_contact_steps * 4
                )
                if self._no_contact_steps >= limit:
                    p = np.asarray(self.env.cup.get_pose().p, dtype=float)
                    _mark_deposit(self.env)
                    self.holding = False
                    self.placed = True
                    print(
                        f"Cup slipped from gripper at ({p[0]:.3f}, {p[1]:.3f}, "
                        f"{p[2]:.3f}); evaluating landing…"
                    )
                    return
        if self._space.poll(window.key_down("space")):
            if not self.holding and not self.placed:
                self.grasp()
            elif self.holding:
                self.place()
            return
        # Universal viewer controls own arrow/E/Q motion.


def main():
    parser = argparse.ArgumentParser(description="Interactive put_cup_belt viewer")
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
        help="Backend for vertical nudges and post-release arm lift (default: interpolate)",
    )
    args = parser.parse_args()

    from envs import CONFIGS_PATH
    from envs.put_cup_belt import put_cup_belt
    from envs.utils.action import ArmTag
    globals()["CONFIGS_PATH"] = CONFIGS_PATH

    print_mode_controls("put_cup_belt", args.control, keyboard=CONTROLS_KEYBOARD, robot=CONTROLS_ROBOT)
    if args.control == "robot" and args.robot_motion == "interpolate":
        print(
            "X/Y/Z nudges and the post-release lift use short joint interpolations "
            "with orientation-constrained IK endpoints."
        )

    env = put_cup_belt()
    env.setup_demo(**_configure_task(args.config, args.seed, use_robot=args.control == "robot"))
    env._interactive_selected_arms = (
        "left" if env.mirrored else "right",
    )
    print_episode_condition(env)
    _place_cup_south_of_belt(env, hold_kinematic=args.control == "keyboard")
    print(
        f"Side={'left' if env.mirrored else 'right'}; "
        f"curtains={env.blue_curtains_enabled}; "
        f"gap x≈{env.slot_x():.3f}."
    )

    controller = (
        RobotCupController(env, ArmTag, args.robot_motion) if args.control == "robot"
        else KeyboardCupController(env)
    )

    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    views = make_viewer_view_toggle(env, viewer)

    placed_since = None
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

            if getattr(env, "_curtain_hit", False) and placed_since is None:
                report_task_result(env, "curtain contact")
                terminal_started_at = time.perf_counter()
                continue
            if getattr(controller, "placed", False):
                if placed_since is None:
                    placed_since = time.perf_counter()
                    print("Cup detached; settling…")
                elif time.perf_counter() - placed_since >= 2.0:
                    hit = bool(getattr(env, "_curtain_hit", False))
                    detail = f"score={env.placement_score():.2f}"
                    if hit:
                        detail = f"curtain hit; {detail}"
                    report_task_result(env, detail)
                    terminal_started_at = time.perf_counter()
                    continue
    finally:
        env.close_env()


if __name__ == "__main__":
    main()
    # household_task_gui convention: 0=SUCCESS, 10=FAILURE, 2=no result
    from _interactive_common import task_result_exit_code
    raise SystemExit(task_result_exit_code())
