#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive viewer for ``whack_a_mole``.

Run from any directory:

    /path/to/RoboDynaExp/script_exp/interactive_whack_a_mole.py --control keyboard
    /path/to/RoboDynaExp/script_exp/interactive_whack_a_mole.py --control robot

Pick up a side-staged mallet, then jab moles mid-rise. Avoid rabbits (Opt1).
Success = all moles hit and no rabbit touch.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import sapien
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "script" / "bench_script"))
sys.path.insert(0, str(REPO_ROOT / "script_exp"))

from _interactive_common import make_viewer_view_toggle, report_task_result, print_mode_controls  # noqa: E402


CONTROLS_KEYBOARD = """
  Q / E             select previous / next unhit mole
  1 .. N            select mole index directly
  Space             jab selected mole (cube press)
  V                 toggle view: top-down ↔ head_camera
  Escape            quit
------------------------------------------------------------
  Success: all moles hit (green), no rabbit touched
"""

CONTROLS_ROBOT = """
  Q / E             select left / right arm (highlighted yellow / cyan)
  Arrow keys        move selected arm in world XY
  T / R             raise / lower selected arm
  Space             pick selected side's mallet, then strike and return to hover
  V                 toggle view: top-down ↔ head_camera
  Escape            quit
------------------------------------------------------------
  Success: all moles hit (green), no rabbit touched
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
        task_name="whack_a_mole",
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


class EdgeKey:
    def __init__(self):
        self._prev = False

    def poll(self, down):
        edge = bool(down) and not self._prev
        self._prev = bool(down)
        return edge


_GRIPPER_LINK_NAMES = (
    "wsg_50_base_link", "gripper_left", "gripper_right", "finger_left", "finger_right",
)
_ARM_HIGHLIGHT = {
    "left": [1.0, 0.85, 0.10, 1.0],
    "right": [0.15, 0.75, 1.0, 1.0],
}


class ArmGripperHighlight:
    """Recolor the selected gripper so Q/E selection is unambiguous."""

    def __init__(self, env):
        self._orig = {}
        self._entities = {
            side: [link.entity for link in articulation.get_links()
                   if link.get_name() in _GRIPPER_LINK_NAMES]
            for side, articulation in (("left", env.robot.left_entity), ("right", env.robot.right_entity))
        }

    def set_selected(self, side):
        for material, color in self._orig.values():
            try:
                material.set_base_color(color)
                material.base_color = color
            except Exception:
                pass
        for entity in self._entities[side]:
            for component in entity.get_components():
                if not isinstance(component, sapien.render.RenderBodyComponent):
                    continue
                for shape in component.render_shapes:
                    material = shape.material
                    if id(material) not in self._orig:
                        self._orig[id(material)] = (material, list(material.base_color))
                    try:
                        material.set_base_color_texture(None)
                        material.set_base_color(_ARM_HIGHLIGHT[side])
                        material.base_color = _ARM_HIGHLIGHT[side]
                    except Exception:
                        pass


def _arm_for_mole(env, idx, ArmTag):
    return env._arm_for_hole(env.mole_holes[idx])


def _set_cube_over_hole(env, arm_name, hole_xy, z=None):
    cube = env.hammer_cubes.get(arm_name)
    if cube is None:
        return
    if z is None:
        z = float(env.board_top_z + env.mole_height + float(env.cube_half) + 0.02)
    pose = sapien.Pose([float(hole_xy[0]), float(hole_xy[1]), float(z)], cube.get_pose().q)
    cube.actor.set_pose(pose)
    rigid = env._cube_comps.get(arm_name)
    if rigid is not None:
        try:
            rigid.set_kinematic_target(pose)
        except Exception:
            pass
    # Keep weld consistent with teleported cube for subsequent kinematics.
    if arm_name in env._cube_weld:
        ee = np.array(env.get_arm_pose(arm_name), dtype=float)
        # Leave weld; cube will re-sync from EE in _update_hammer_cubes.
        # For keyboard teleop, suppress EE weld by storing identity-ish local.
        pass


def _jab_cube_on_mole(env, idx, ArmTag):
    """Lower the correct-side cube onto the mole crown and register a hit."""
    arm = _arm_for_mole(env, idx, ArmTag)
    arm_name = str(arm)
    hole = env.holes[env.mole_holes[idx]]
    # Hover then press.
    hover_z = float(env.board_top_z + env.mole_height + float(env.cube_half) + 0.025)
    _set_cube_over_hole(env, arm_name, hole, z=hover_z)
    # Aim at current crown.
    top = env._critter_top_z(env.moles, env._mole_state, idx)
    press_z = float(top - float(env.cube_half) + 0.002)
    _set_cube_over_hole(env, arm_name, hole, z=press_z)
    # Geometric / contact hit, or force mark if cube is over a rising mole.
    if env._mole_above_surface(idx) and not env.touched[idx]:
        if env._cube_bottom_contact_with_critter(env.moles, env._mole_state, idx):
            env._mark_touched(idx)
        else:
            cube_p = np.array(env.hammer_cubes[arm_name].get_pose().p, dtype=float)
            if float(np.linalg.norm(cube_p[:2] - hole[:2])) < 0.05:
                env._mark_touched(idx)
    # Lift back.
    _set_cube_over_hole(env, arm_name, hole, z=hover_z)
    return bool(env.touched[idx])


class KeyboardMoleController:
    def __init__(self, env, ArmTag):
        self.env = env
        self.ArmTag = ArmTag
        self.selected = 0
        self._space = EdgeKey()
        self._q = EdgeKey()
        self._e = EdgeKey()
        self._digit = {str(i): EdgeKey() for i in range(1, 10)}
        self._select_next(0)

    def _unhit(self):
        return [i for i in range(self.env.num_moles) if not self.env.touched[i]]

    def _select_next(self, delta):
        unhit = self._unhit()
        if not unhit:
            return
        if self.selected not in unhit:
            self.selected = unhit[0]
        else:
            k = unhit.index(self.selected)
            self.selected = unhit[(k + delta) % len(unhit)]
        arm = _arm_for_mole(self.env, self.selected, self.ArmTag)
        hole = self.env.holes[self.env.mole_holes[self.selected]]
        _set_cube_over_hole(self.env, str(arm), hole)
        print(f"Selected mole {self.selected} ({arm} arm).")

    def update(self, window):
        if self.env.distractor_hit:
            return
        if self._q.poll(window.key_down("q")):
            self._select_next(-1)
        if self._e.poll(window.key_down("e")):
            self._select_next(+1)
        for d, edge in self._digit.items():
            if edge.poll(window.key_down(d)):
                idx = int(d) - 1
                if 0 <= idx < self.env.num_moles and not self.env.touched[idx]:
                    self.selected = idx
                    arm = _arm_for_mole(self.env, idx, self.ArmTag)
                    hole = self.env.holes[self.env.mole_holes[idx]]
                    _set_cube_over_hole(self.env, str(arm), hole)
                    print(f"Selected mole {idx} ({arm} arm).")
        # Keep cube hovering over selection while waiting (unless weld rewrites it).
        if not self.env.touched[self.selected]:
            arm = _arm_for_mole(self.env, self.selected, self.ArmTag)
            hole = self.env.holes[self.env.mole_holes[self.selected]]
            # Only re-seat when mole is rising so the player can time the jab.
            if self.env._mole_is_rising(self.selected) or self.env._mole_above_surface(self.selected):
                hover_z = float(self.env.board_top_z + self.env.mole_height + float(self.env.cube_half) + 0.03)
                _set_cube_over_hole(self.env, str(arm), hole, z=hover_z)
        if self._space.poll(window.key_down("space")):
            if self.env.touched[self.selected]:
                self._select_next(+1)
                return
            ok = _jab_cube_on_mole(self.env, self.selected, self.ArmTag)
            print(f"Jab mole {self.selected}: {'HIT' if ok else 'miss'} "
                  f"(rising={self.env._mole_is_rising(self.selected)}).")
            if ok:
                unhit = self._unhit()
                if unhit:
                    self.selected = unhit[0]
                    arm = _arm_for_mole(self.env, self.selected, self.ArmTag)
                    hole = self.env.holes[self.env.mole_holes[self.selected]]
                    _set_cube_over_hole(self.env, str(arm), hole)


class RobotMoleController:
    XY_STEP = 0.045
    Z_STEP = 0.030
    DURATION = 0.04
    MAX_RAISE_ABOVE_HOVER = 0.12
    MAX_JOINT_DELTA = 0.45

    def __init__(self, env, ArmTag):
        self.env = env
        self.ArmTag = ArmTag
        self.selected_arm = "left"
        self.busy = False
        self._space = EdgeKey()
        self._q = EdgeKey()
        self._e = EdgeKey()
        self.highlight = ArmGripperHighlight(env)
        self.highlight.set_selected(self.selected_arm)
        self._start = None
        self._target = None
        self._started_at = None
        self._moving_arm = None

    def _arm(self):
        return self.ArmTag(self.selected_arm)

    def _drive_qpos(self, side):
        joints = self.env.robot.left_arm_joints if side == "left" else self.env.robot.right_arm_joints
        return np.asarray([joint.get_drive_target()[0] for joint in joints], dtype=np.float64)

    def _ee_pose(self, side):
        getter = self.env.robot.get_left_ee_pose if side == "left" else self.env.robot.get_right_ee_pose
        return np.asarray(getter(), dtype=np.float64)

    def _advance_motion(self):
        if self._started_at is None:
            return
        progress = min(1.0, (time.perf_counter() - self._started_at) / self.DURATION)
        smooth = progress * progress * (3.0 - 2.0 * progress)
        delta = self._target - self._start
        position = self._start + delta * smooth
        velocity = delta / self.DURATION if progress < 1.0 else np.zeros_like(delta)
        self.env.robot.set_arm_joints(position, velocity, self._moving_arm)
        if progress >= 1.0:
            self._started_at = None
            self._moving_arm = None

    def _move_selected_arm(self, window):
        if self._started_at is not None:
            return
        dx = self.XY_STEP * (window.key_down("right") - window.key_down("left"))
        dy = self.XY_STEP * (window.key_down("up") - window.key_down("down"))
        dz = self.Z_STEP * (window.key_down("t") - window.key_down("r"))
        if not (dx or dy or dz):
            return
        side = self.selected_arm
        pose = self._ee_pose(side).copy()
        pose[:3] += np.asarray([dx, dy, dz], dtype=np.float64)
        # The upper edge of reach can make IK switch to a radically different
        # elbow configuration. Keep manual Z motion in the mallet's safe band.
        hover_z = float(self.env._hover_ee_z(self._arm()))
        pose[2] = np.clip(pose[2], hover_z, hover_z + self.MAX_RAISE_ABOVE_HOVER)
        planner = self.env.robot.left_plan_path if side == "left" else self.env.robot.right_plan_path
        start = self._drive_qpos(side)
        result = planner(pose.tolist(), last_qpos=np.asarray(start, dtype=np.float32))
        if result is None or result.get("status") != "Success":
            return
        target = np.asarray(result["position"][-1], dtype=np.float64)
        if float(np.max(np.abs(target - start))) > self.MAX_JOINT_DELTA:
            print("Requested arm move is outside the safe teleoperation range.")
            return
        self._start = start
        self._target = target
        self._moving_arm = side
        self._started_at = time.perf_counter()

    def jab(self):
        self.busy = True
        arm = self._arm()
        self.env.plan_success = True
        if self.selected_arm not in self.env.hammer_cubes:
            self.env.pickup_mallet_to_ready(arm)
            print(f"Picked up {self.selected_arm} mallet; it is ready at hover height.")
            self.busy = False
            return
        cube_p = self.env._mallet_head_center(arm)
        idx = next(
            (i for i, hole_idx in enumerate(self.env.mole_holes)
             if not self.env.touched[i]
             and float(np.linalg.norm(cube_p[:2] - self.env.holes[hole_idx][:2])) < 0.07),
            None,
        )
        if idx is None:
            print("No unhit mole under the selected cube.")
            self.busy = False
            return
        self.env._mole_state[idx]["freeze_bob"] = True
        self.env.move(self.env._press_down(arm, mole_idx=idx))
        self.env._dwell(10)
        self.env._mole_state[idx]["freeze_bob"] = False
        if not self.env.touched[idx] and self.env._mole_above_surface(idx):
            hole = self.env.holes[self.env.mole_holes[idx]]
            cube_p = np.array(self.env.hammer_cubes[str(arm)].get_pose().p, dtype=float)
            if float(np.linalg.norm(cube_p[:2] - hole[:2])) < 0.08:
                self.env._mark_touched(idx)
        self.env.plan_success = True
        # Every strike returns to the task's safe hover/rest height.
        self.env.move(self.env._press_up(arm))
        print(f"Robot jab mole {idx}: {'HIT' if self.env.touched[idx] else 'miss'}.")
        self.busy = False

    def update(self, window):
        if self.busy or self.env.distractor_hit:
            return
        self._advance_motion()
        if self._q.poll(window.key_down("q")):
            self.selected_arm = "left"
            self.highlight.set_selected(self.selected_arm)
            print("Selected left arm.")
        if self._e.poll(window.key_down("e")):
            self.selected_arm = "right"
            self.highlight.set_selected(self.selected_arm)
            print("Selected right arm.")
        self._move_selected_arm(window)
        if self._space.poll(window.key_down("space")):
            self.jab()


def main():
    parser = argparse.ArgumentParser(description="Interactive whack_a_mole viewer")
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
    from envs.whack_a_mole import whack_a_mole
    from envs.utils.action import ArmTag
    globals()["CONFIGS_PATH"] = CONFIGS_PATH

    print_mode_controls("whack_a_mole", args.control, keyboard=CONTROLS_KEYBOARD, robot=CONTROLS_ROBOT)
    if args.robot_motion == "interpolate":
        print(
            "Note: --robot-motion interpolate uses planner motions for this teleop task "
            "(key-press sandboxes use joint interpolation)."
        )

    env = whack_a_mole()
    env.setup_demo(**_configure_task(args.config, args.seed, use_robot=args.control == "robot"))
    print(
        f"moles={env.num_moles}; distractors={env.num_distractors}; "
        f"relocating={env.relocating_moles}; difficulty={env.difficulty}."
    )

    controller = (
        RobotMoleController(env, ArmTag) if args.control == "robot"
        else KeyboardMoleController(env, ArmTag)
    )

    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    views = make_viewer_view_toggle(env, viewer)

    # For keyboard teleop, prevent EE weld from yanking cubes every step.
    if args.control == "keyboard":
        env._cube_weld = {}

    done_since = None
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

            if env.distractor_hit:
                report_task_result(env, "rabbit touched")
                break
            if env.check_success():
                if done_since is None:
                    done_since = time.perf_counter()
                    print("All moles hit; wrapping up…")
                elif time.perf_counter() - done_since >= 1.0:
                    report_task_result(env)
                    break

            remaining = float(env.scene.get_timestep()) - (time.perf_counter() - frame_start)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        env.close_env()


if __name__ == "__main__":
    main()
