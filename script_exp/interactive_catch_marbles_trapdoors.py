#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive viewer for ``catch_marbles_trapdoors``.

Run from any directory:

    /path/to/RoboDynaExp/script_exp/interactive_catch_marbles_trapdoors.py --control keyboard
    /path/to/RoboDynaExp/script_exp/interactive_catch_marbles_trapdoors.py --control robot
    /path/to/RoboDynaExp/script_exp/interactive_catch_marbles_trapdoors.py --control robot --robot-motion interpolate

Keyboard mode opens trapdoors directly (cycle + Space). Robot mode: select an
arm, move over the matching colored key, then Space to press. Sandbox only.
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
    action_failed,
    arm_ik,
    make_viewer_view_toggle,
    RobotButtonController,
    add_robot_motion_arg,
    edge_pressed,
    report_task_result,
    print_mode_controls,
    require_selected_arms,
)


CONTROLS_KEYBOARD = """
  Left / Right Arrow  cycle selected trapdoor button
  Space               open the selected trapdoor (direct, no arm)

  Colors are printed at startup (left→right order may shuffle).
  Open the button whose color matches the moving marble.

  V                 toggle view: top-down ↔ head_camera
  Close the viewer window to quit.
"""

CONTROLS_ROBOT = """
  Space            →  press the key under the selected arm

  Select left (1) or right (2) arm, move over the matching colored key,
  then press Space. Left arm covers left-half keys; right arm covers
  right-half keys. Door opens on fingertip contact.

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
        task_name="catch_marbles_trapdoors",
        render_freq=1,
        now_ep_num=0,
        seed=seed,
        need_plan=use_robot,
        save_data=False,
    )
    # Interactive sandbox: swing doors open quickly (~0.1 s) so they don't lag
    # behind the arm press. Demo collection still uses demo_dynamic.yml as-is.
    task_args = config.setdefault("task_args", {}).setdefault("catch_marbles_trapdoors", {})
    task_args["door_open_speed_deg"] = max(float(task_args.get("door_open_speed_deg", 220.0)), 1200.0)

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


# Max TCP→key XY distance (m) to count as "over" a key (button half is ~2 cm).
_KEY_XY_TOL = 0.055


def _tcp_xy(env, side: str) -> np.ndarray:
    getter = env.robot.get_left_tcp_pose if side == "left" else env.robot.get_right_tcp_pose
    return np.asarray(getter()[:2], dtype=np.float64)


def _nearest_button_for_arm(env, side: str, max_dist: float = _KEY_XY_TOL):
    """Index of the nearest key under ``side``'s TCP, or None if too far."""
    tcp = _tcp_xy(env, side)
    best_i, best_d = None, float(max_dist)
    for i in range(int(env.n_buttons)):
        if str(env._arm_for_door(i)) != side:
            continue
        p = np.asarray(env.buttons[i].get_pose().p[:2], dtype=np.float64)
        d = float(np.linalg.norm(p - tcp))
        if d < best_d:
            best_d, best_i = d, i
    return best_i


class KeyboardDoorSelect:
    """Left/Right cycle a selected trapdoor; Space opens it directly."""

    def __init__(self, n_buttons: int):
        self.selected = 0
        self.n = max(1, int(n_buttons))
        self._prev = {}

    def update(self, window, env):
        if edge_pressed(window, "left", self._prev):
            self.selected = (self.selected - 1) % self.n
            color = env.button_color_names[self.selected]
            print(f"Selected button {self.selected} ({color}).")
        if edge_pressed(window, "right", self._prev):
            self.selected = (self.selected + 1) % self.n
            color = env.button_color_names[self.selected]
            print(f"Selected button {self.selected} ({color}).")
        if edge_pressed(window, "space", self._prev):
            idx = self.selected
            if env._open_door_direct(idx):
                color = env.button_color_names[idx]
                print(f"Opened trapdoor {idx} ({color}).")
            else:
                print(f"Could not open trapdoor {idx} (locked/already open).")


class TrapdoorPlannerButtonController(RobotButtonController):
    """Trapdoors-only: open the door on the first physics frame of fingertip contact.

    Kept out of ``_interactive_common`` so other tasks keep the shared press path.
    """

    PRESS_HOLD_STEPS = 2
    # Hover clearly above the key so we don't open before the press starts.
    HOVER_DIS = 0.055
    # For ur5-wsg, TCP plateaus ~2.1–2.8 cm above the geometric key top when the
    # fingertips are on the key (collision). Must stay in that band — tighter
    # values (e.g. 2.2 cm) miss contact and the door never opens.
    TOUCH_DZ = 0.028
    # Stall / end-of-path open: slightly above TOUCH so a jammed fingertip still counts.
    STALL_DZ = 0.035
    PRESS_DEPTH = 0.05
    # After a tap, clear well above the key (shared default is only ~4 cm).
    CLEAR_ABOVE_ACTIVE = 0.10
    POST_PRESS_HOVER = 0.12

    def __init__(self, env, arm_tag, *, single_press_depth: float, **kwargs):
        hover = float(kwargs.pop("grasp_dis", self.HOVER_DIS))
        kwargs["grasp_dis"] = hover
        kwargs["pre_grasp_dis"] = kwargs.get("pre_grasp_dis", hover + 0.02)
        kwargs["max_press_depth"] = kwargs.get(
            "max_press_depth", max(float(single_press_depth), self.PRESS_DEPTH)
        )
        kwargs["active_dz"] = kwargs.get("active_dz", self.TOUCH_DZ)
        super().__init__(env, arm_tag, **kwargs)
        self.single_press_depth = max(float(single_press_depth), self.PRESS_DEPTH)
        self._opened_this_press = False

    def press_in_place(self, mode):
        """Descend from the current pose (no approach); open on fingertip contact."""
        self._opened_this_press = False
        sides = tuple(self.arms_for_mode(mode))
        if not sides:
            return
        self._hover_qpos = {side: self._drive_qpos(side) for side in sides}
        if not self._press_and_open(mode):
            print("Could not reach key; returning to hover (door stays closed).")
            self._interpolate_to_qpos(self._hover_qpos)
            self.clear_latch(self.env)
            return
        viewer = getattr(self.env, "viewer", None)
        for _ in range(self.PRESS_HOLD_STEPS):
            self.set_latch(self.env, mode)
            self.env._update_kinematic_tasks()
            self.env.scene.step()
            if viewer is not None:
                self.env.scene.update_render()
                viewer.render()
        print(f"Robot tapped {mode}.")
        self.clear_latch(self.env)
        self._lift_from_buttons(mode)
        self.mode = None

    def _tcp_near_key(self, mode, dz: float | None = None) -> bool:
        band = float(self.TOUCH_DZ if dz is None else dz)
        sides = tuple(self.arms_for_mode(mode))
        if not sides:
            return False
        for side in sides:
            tcp_z = self._tcp_z(side)
            if tcp_z is None:
                return False
            top_z = float(self.get_top_z(self.env, side))
            if float(tcp_z) > top_z + band:
                return False
        return True

    def _force_open(self, mode) -> bool:
        """Open immediately (bypass shared active-zone gate)."""
        if self._opened_this_press:
            return True
        self.set_latch(self.env, mode)
        if self.on_press is not None:
            try:
                self.on_press(self.env, mode)
            except Exception as exc:
                print(f"on_press callback failed: {exc}")
                return False
        self._opened_this_press = True
        self.env.plan_success = True
        self.env._last_plan_fail = None
        return True

    def _plan_press_path(self, side, depth: float):
        get_ee = (
            self.env.robot.get_left_ee_pose if side == "left"
            else self.env.robot.get_right_ee_pose
        )
        planner = (
            self.env.robot.left_plan_path if side == "left"
            else self.env.robot.right_plan_path
        )
        pose = np.asarray(get_ee(), dtype=np.float64).copy()
        pose[2] -= float(depth)
        result = planner(pose.tolist())
        if result is None or result.get("status") != "Success":
            return None
        positions = result.get("position")
        if positions is None or len(positions) == 0:
            return None
        return np.asarray(positions, dtype=np.float64)

    def _side_dz(self, side) -> float | None:
        tcp_z = self._tcp_z(side)
        if tcp_z is None:
            return None
        return float(tcp_z) - float(self.get_top_z(self.env, side))

    def _contact_open(self, mode, hover_dz: float, dz: float, prev_dz: float, min_drop: float) -> bool:
        dropped = hover_dz - dz
        if dropped >= min_drop and dz <= self.TOUCH_DZ:
            return self._force_open(mode)
        # TCP stopped against the key after a real descent (collision floor).
        if dropped >= min_drop and dz <= self.STALL_DZ and abs(prev_dz - dz) < 0.0008:
            return self._force_open(mode)
        return False

    def _ik_joints_for_ee(self, side, ee_pose7):
        """Curobo solve_ik (not trajopt) for a planning-EE pose."""
        import torch  # noqa: F401
        from curobo.types.math import Pose as CuroboPose

        planner = (
            self.env.robot.left_planner if side == "left"
            else self.env.robot.right_planner
        )
        trans_target = self.env.robot._trans_from_gripper_to_endlink(
            list(ee_pose7), arm_tag=side,
        )
        world_target = np.concatenate([
            np.asarray(trans_target.p, dtype=float),
            np.asarray(trans_target.q, dtype=float),
        ])
        world_base = np.concatenate([
            np.asarray(planner.robot_origion_pose.p, dtype=float),
            np.asarray(planner.robot_origion_pose.q, dtype=float),
        ])
        tp_p, tp_q = planner._trans_from_world_to_base(world_base, world_target)
        tp_p = np.asarray(tp_p, dtype=float)
        tp_q = np.asarray(tp_q, dtype=float)
        if "aloha-agilex" not in str(getattr(planner, "yml_path", "")):
            tp_p = tp_p + np.asarray(planner.frame_bias, dtype=float)
        goal = CuroboPose.from_list(list(tp_p) + list(tp_q))
        ik = planner.motion_gen.solve_ik(goal, return_seeds=1)
        if not bool(ik.success.reshape(-1)[0].item()):
            return None
        return ik.solution.detach().cpu().numpy().reshape(-1).astype(float)

    def _apply_arm_qpos(self, side, q) -> None:
        """Write joint targets and teleport active qpos (drive-only stalls in contact)."""
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        planner = (
            self.env.robot.left_planner if side == "left"
            else self.env.robot.right_planner
        )
        entity = (
            self.env.robot.left_entity if side == "left"
            else self.env.robot.right_entity
        )
        active = entity.get_active_joints()
        name_to_i = {j.get_name(): i for i, j in enumerate(active)}
        qpos = np.asarray(entity.get_qpos(), dtype=np.float64)
        for j, jn in enumerate(planner.active_joints_name):
            if j >= len(q):
                break
            if jn in name_to_i:
                qpos[name_to_i[jn]] = q[j]
        entity.set_qpos(qpos)
        self.env.robot.set_arm_joints(q, np.zeros_like(q), side)

    def _arm_qpos(self, side) -> np.ndarray:
        """Current active-arm joint vector (entity qpos, not stale drive targets)."""
        planner = (
            self.env.robot.left_planner if side == "left"
            else self.env.robot.right_planner
        )
        entity = (
            self.env.robot.left_entity if side == "left"
            else self.env.robot.right_entity
        )
        active = entity.get_active_joints()
        name_to_i = {j.get_name(): i for i, j in enumerate(active)}
        qpos = np.asarray(entity.get_qpos(), dtype=np.float64)
        out = []
        for jn in planner.active_joints_name:
            out.append(float(qpos[name_to_i[jn]]) if jn in name_to_i else 0.0)
        return np.asarray(out[:6], dtype=np.float64)

    def _descend_ik(self, mode, sides, hover_dz: float, min_drop: float) -> bool:
        """IK + set_qpos interpolate -Z when trajopt press paths stall high.

        Common on a second same-arm tap: plan_path reports Success but barely
        lowers TCP. solve_ik reaches the band only if we also teleport qpos
        each step (drive targets alone stall against collision).
        """
        # Aim just through the fingertip band so the first contact frame is
        # near touch — not a deep plunge past the key.
        target_dz = max(0.012, float(self.TOUCH_DZ) - 0.006)
        targets = {}
        for side in sides:
            get_ee = (
                self.env.robot.get_left_ee_pose if side == "left"
                else self.env.robot.get_right_ee_pose
            )
            pose = np.asarray(get_ee(), dtype=np.float64).copy()
            dz_now = self._side_dz(side)
            need = 0.02 if dz_now is None else max(float(dz_now) - target_dz, 0.012)
            q = None
            for scale in (1.0, 1.25, 0.75, 0.5):
                trial = pose.copy()
                trial[2] -= float(need) * scale
                q = self._ik_joints_for_ee(side, trial)
                if q is not None:
                    break
            if q is None:
                return False
            targets[side] = np.asarray(q[:6], dtype=np.float64)

        starts = {side: self._arm_qpos(side) for side in targets}
        n = 48
        viewer = getattr(self.env, "viewer", None)
        side0 = sides[0]
        prev_dz = hover_dz
        for i in range(1, n + 1):
            a = i / float(n)
            for side, goal in targets.items():
                q = starts[side] + (goal - starts[side]) * a
                self._apply_arm_qpos(side, q)
            self.env._update_kinematic_tasks()
            self.env.scene.step()
            if viewer is not None:
                self.env.scene.update_render()
                viewer.render()
            dz = self._side_dz(side0)
            if dz is None:
                continue
            if self._contact_open(mode, hover_dz, dz, prev_dz, min_drop):
                return True
            prev_dz = dz
        dz = self._side_dz(side0)
        if dz is not None and hover_dz - dz >= min_drop and dz <= self.STALL_DZ:
            return self._force_open(mode)
        return False

    def _press_and_open(self, mode) -> bool:
        """Descend; open on the first real fingertip-contact frame."""
        sides = tuple(self.arms_for_mode(mode))
        if not sides:
            return False

        side0 = sides[0]
        hover_dz = self._side_dz(side0)
        if hover_dz is None:
            return False
        # Require a real descent before contact counts (avoid opening at hover).
        min_drop = 0.010
        depth = min(self.max_press_depth, self.single_press_depth)

        paths = {}
        for side in sides:
            path = self._plan_press_path(side, depth)
            if path is None:
                paths = {}
                break
            paths[side] = path

        if paths:
            n = max(len(p) for p in paths.values())
            viewer = getattr(self.env, "viewer", None)
            prev_dz = hover_dz
            best_dz = hover_dz
            for i in range(n):
                for side, path in paths.items():
                    q = path[min(i, len(path) - 1)]
                    vel = np.zeros_like(q) if i + 1 >= len(path) else path[i + 1] - q
                    self.env.robot.set_arm_joints(q, vel, side)
                self.env._update_kinematic_tasks()
                self.env.scene.step()
                if viewer is not None:
                    self.env.scene.update_render()
                    viewer.render()

                dz = self._side_dz(side0)
                if dz is None:
                    continue
                best_dz = min(best_dz, dz)
                if self._contact_open(mode, hover_dz, dz, prev_dz, min_drop):
                    return True
                prev_dz = dz

            if best_dz <= self.STALL_DZ and hover_dz - best_dz >= min_drop:
                return self._force_open(mode)

        # Trajopt stalled above the key (often after a prior same-arm press).
        hover_dz = self._side_dz(side0) or hover_dz
        return self._descend_ik(mode, sides, hover_dz, min_drop)

    def _lift_from_buttons(self, mode):
        """Return above the key, then clear to a higher post-press hover."""
        targets = {
            side: self._hover_qpos[side]
            for side in self.arms_for_mode(mode)
            if side in self._hover_qpos
        }
        if targets:
            self._interpolate_to_qpos(targets)
        self.env.plan_success = True
        self.env._last_plan_fail = None
        lifts = []
        for side in self.arms_for_mode(mode):
            tcp_z = self._tcp_z(side)
            top_z = float(self.get_top_z(self.env, side))
            want_z = top_z + float(self.POST_PRESS_HOVER)
            if tcp_z is None:
                lifts.append((side, float(self.POST_PRESS_HOVER)))
                continue
            extra = want_z - float(tcp_z)
            if extra > 0.001:
                lifts.append((side, extra))
        if lifts:
            self.env.move(*[
                self.env.move_by_displacement(self.arm_tag(side), z=dz)
                for side, dz in lifts
            ])
            if not self.env.plan_success:
                # Fallback: shared clear-lift (uses CLEAR_ABOVE_ACTIVE).
                self.env.plan_success = True
                self._lift_clear_of_keys(mode)
                self._hover_qpos.clear()
                return
        self.clear_latch(self.env)
        for _ in range(12):
            self.env._update_kinematic_tasks()
            self.env.scene.step()
        self._hover_qpos.clear()

    def _move_to_buttons(self, mode):
        self._opened_this_press = False
        sides = tuple(self.arms_for_mode(mode))
        if not sides:
            return
        actions = [
            self.env.grasp_actor(
                self.get_button(self.env, side),
                arm_tag=self.arm_tag(side),
                pre_grasp_dis=self.pre_grasp_dis,
                grasp_dis=self.grasp_dis,
                contact_point_id=0,
                gripper_pos=0.0,
            )
            for side in sides
        ]
        self.env.move(*actions)
        if not self.env.plan_success:
            return
        self._hover_qpos = {side: self._drive_qpos(side) for side in sides}

        # Must start above the touch band; otherwise we'd open before pressing.
        if self._tcp_near_key(mode):
            # Nudge up so the upcoming descent has a clear contact edge.
            self.env.plan_success = True
            self.env.move(*[
                self.env.move_by_displacement(self.arm_tag(side), z=0.03)
                for side in sides
            ])
            self._hover_qpos = {side: self._drive_qpos(side) for side in sides}

        if not self._press_and_open(mode):
            print("Could not reach key; returning to hover (door stays closed).")
            self._interpolate_to_qpos(self._hover_qpos)
            self.clear_latch(self.env)
            return

        viewer = getattr(self.env, "viewer", None)
        for _ in range(self.PRESS_HOLD_STEPS):
            self.set_latch(self.env, mode)
            self.env._update_kinematic_tasks()
            self.env.scene.step()
            if viewer is not None:
                self.env.scene.update_render()
                viewer.render()
        print(f"Robot tapped {mode}.")


class SmoothTrapdoorPressController:
    """Non-blocking, gummy-style vertical key press from the current arm pose.

    Timed on the simulation clock, not wall time: the viewer advances 4 ms of
    physics per rendered frame, so a wall-timed ramp finishes while the arm is
    still on its way down and the tap silently misses the key.
    """

    # Descend at a fixed speed so a press from 5 cm and one from 40 cm both
    # look natural, then clamp the duration for very short / long reaches.
    PRESS_SPEED = 0.70
    RAISE_SPEED = 1.00
    MIN_TRANSITION_SECONDS = 0.10
    MAX_TRANSITION_SECONDS = 0.90
    # Aim a little below the fingertip band and dwell on the key: the drive
    # tracks with a few centimetres of lag on a tall press, and lifting off
    # before it settles is what used to make the first tap do nothing.
    MIN_HOLD_SECONDS = 0.05
    MAX_HOLD_SECONDS = 0.40
    TOUCH_DZ = 0.020
    STALL_DZ = 0.040
    # A descent that stalls this close above the keycap is fingertip contact:
    # the aim point is below the key, so only the key can stop the arm.
    CONTACT_DZ = 0.055
    MIN_DESCENT = 0.010
    MAX_DESCENT = 0.50
    MAX_PRESS_JOINT_TRAVEL = 1.80

    def __init__(self, env, on_press):
        self.env = env
        self.on_press = on_press
        self.phase = "idle"
        self.side = None
        self.idx = None
        self.start_qpos = None
        self.hover_qpos = None
        self.press_qpos = None
        self.started_at = None
        self.holding_from = None
        self.holding_until = None
        self.transition_seconds = self.MIN_TRANSITION_SECONDS
        self.descent = 0.0
        self.commanded_descent = 0.0
        self._last_tcp_z = None
        self._clock = 0.0
        self.opened = False

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

    def _button_top_z(self):
        return (
            float(self.env.buttons[self.idx].get_pose().p[2])
            + float(self.env.button_half[2])
        )

    def _ik_joints(self, ee_pose7):
        """Seeded local IK so the press stays in the arm's current branch."""
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
        desired_tcp_z = self._button_top_z() + self.TOUCH_DZ
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

    def request(self, idx):
        if self.busy:
            return False
        self.idx = int(idx)
        self.side = str(self.env._arm_for_door(self.idx))
        self.hover_qpos = self._drive_qpos(self.side)
        self.press_qpos, descent = self._plan_press_target()
        if self.press_qpos is None:
            print("Could not plan a smooth vertical key press.")
            self._reset()
            return False
        self.descent = descent
        self.env._interactive_teleop_locked = True
        self.opened = False
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

    def _try_open(self):
        if self.opened:
            return
        tcp_z = self._tcp_z()
        previous, self._last_tcp_z = self._last_tcp_z, tcp_z
        # Judge the press by how far it has been *commanded* down, not by how
        # far the fingertip travelled: a tap that starts just above the keycap
        # only has millimetres of travel before the key stops it.
        if self.commanded_descent < min(self.MIN_DESCENT, self.descent):
            return
        above = tcp_z - self._button_top_z()
        stalled = previous is not None and abs(previous - tcp_z) < 0.0004
        if above <= self.STALL_DZ or (stalled and above <= self.CONTACT_DZ):
            self.on_press(self.idx)
            self.opened = True

    def _finish_transition(self, now):
        if self.phase == "pressing":
            self.phase = "holding"
            self.started_at = None
            self.holding_from = now
            self.holding_until = now + self.MAX_HOLD_SECONDS
            self._try_open()
        elif self.phase == "raising":
            if not self.opened:
                print("The fingertip did not reach the key; trapdoor stayed closed.")
            self._reset()

    def update(self):
        if self.phase == "idle":
            return
        # One update per physics step, so the sim clock is the loop's clock.
        self._clock += float(self.env.scene.get_timestep())
        now = self._clock
        if self.phase == "holding":
            self.env.robot.set_arm_joints(
                self.press_qpos,
                np.zeros_like(self.press_qpos),
                self.side,
            )
            self.commanded_descent = self.descent
            self._try_open()
            settled = self.opened or now >= self.holding_until
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
        if self.phase == "pressing":
            self.commanded_descent = self.descent * smooth
            self._try_open()
        if progress >= 1.0:
            self._finish_transition(now)

    def _reset(self):
        self.env._buttons_held.clear()
        self.env._interactive_teleop_locked = False
        self.phase = "idle"
        self.side = None
        self.idx = None
        self.start_qpos = None
        self.hover_qpos = None
        self.press_qpos = None
        self.started_at = None
        self.holding_from = None
        self.holding_until = None
        self.commanded_descent = 0.0
        self._last_tcp_z = None
        self.opened = False

    def release(self):
        if self.busy and self.hover_qpos is not None:
            self.env.robot.set_arm_joints(
                self.hover_qpos,
                np.zeros_like(self.hover_qpos),
                self.side,
            )
        self._reset()


def _print_color_map(env):
    names = list(getattr(env, "button_color_names", []) or [])
    target = int(getattr(env, "target_button_idx", -1))
    mapping = ", ".join(f"{i}:{c}" for i, c in enumerate(names))
    target_name = names[target] if 0 <= target < len(names) else "?"
    left_keys = [f"{i}:{c}" for i, c in enumerate(names) if str(env._arm_for_door(i)) == "left"]
    right_keys = [f"{i}:{c}" for i, c in enumerate(names) if str(env._arm_for_door(i)) == "right"]
    print(f"Buttons L→R: {mapping}")
    print(f"Left-arm keys: {', '.join(left_keys) or '(none)'} | Right-arm keys: {', '.join(right_keys) or '(none)'}")
    print(f"Target marble color: {target_name} (index {target})")


def _make_trapdoor_controller(env):
    """Build the non-blocking smooth press controller."""

    def on_press(idx):
        idx = int(idx)
        color = env.button_color_names[idx]
        env._buttons_held.clear()
        env._buttons_held.add(idx)
        if env._open_door_direct(idx):
            snap = min(55.0, float(env.door_open_angle_deg))
            env._set_door_pose(idx, snap)
            print(f"Opened trapdoor {idx} ({color}) on fingertip contact.")
        else:
            print(f"Could not open trapdoor {idx} ({color}; locked/already open).")
        viewer = getattr(env, "viewer", None)
        if viewer is not None:
            env.scene.update_render()
            viewer.render()

    return SmoothTrapdoorPressController(env, on_press)


def main():
    parser = argparse.ArgumentParser(description="Interactive catch_marbles_trapdoors viewer")
    parser.add_argument("--config", default="demo_dynamic", help="Task config name without .yml")
    parser.add_argument("--seed", type=int, default=0, help="Scene randomization seed")
    add_robot_motion_arg(parser)
    args = parser.parse_args()

    from envs import CONFIGS_PATH
    from envs.catch_marbles_trapdoors import catch_marbles_trapdoors
    from envs.utils.action import ArmTag
    globals()["CONFIGS_PATH"] = CONFIGS_PATH

    print_mode_controls("catch_marbles_trapdoors", args.control, keyboard=CONTROLS_KEYBOARD, robot=CONTROLS_ROBOT)

    env = catch_marbles_trapdoors()
    env.setup_demo(**_configure_task(args.config, args.seed, use_robot=args.control == "robot"))
    env.together_close_gripper(save_freq=None)
    _print_color_map(env)

    keyboard = KeyboardDoorSelect(env.n_buttons) if args.control == "keyboard" else None
    keys_prev = {}
    robot = _make_trapdoor_controller(env) if args.control == "robot" else None

    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    views = make_viewer_view_toggle(env, viewer)

    motion = f", robot-motion={args.robot_motion}" if args.control == "robot" else ""
    if args.control == "robot":
        print(f"Control=robot{motion}. Select an arm (1/2), move over a key, press Space.")
    else:
        print(f"Control=keyboard. Left/Right cycle buttons; Space opens.")

    left_track_since = None
    settle_s = 0.6
    try:
        while not viewer.closed:
            views.update(viewer.window)
            frame_start = time.perf_counter()
            if keyboard is not None:
                keyboard.update(viewer.window, env)
            elif robot is not None:
                if edge_pressed(viewer.window, "space", keys_prev):
                    selected = require_selected_arms(env, exactly_one=True)
                    if not selected:
                        continue
                    side = selected[0]
                    idx = _nearest_button_for_arm(env, side)
                    if idx is None:
                        action_failed(env, (side,), detail="no key under selected arm")
                    elif str(env._arm_for_door(idx)) != side:
                        action_failed(env, (side,), detail="nearest key belongs to other arm")
                    else:
                        color = env.button_color_names[idx]
                        print(f"Robot tapping button {idx} ({color}) with {side} arm...")
                        robot.request(idx)
                robot.update()
            env._update_kinematic_tasks()
            env.scene.step()
            env.scene.update_render()
            viewer.render()
            if viewer.window.key_down("escape"):
                break
            mode = str(getattr(env, "_ball_mode", "track"))
            if mode != "track":
                if left_track_since is None:
                    left_track_since = time.perf_counter()
                elif time.perf_counter() - left_track_since >= settle_s:
                    report_task_result(env, f"ball_mode={mode}")
                    break
            else:
                left_track_since = None
            remaining = float(env.scene.get_timestep()) - (time.perf_counter() - frame_start)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        try:
            if robot is not None:
                robot.release()
        finally:
            env.close_env()


if __name__ == "__main__":
    main()
