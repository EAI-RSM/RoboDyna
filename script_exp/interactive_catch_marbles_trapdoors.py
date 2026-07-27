#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive viewer for ``catch_marbles_trapdoors``.

Run from any directory:

    /path/to/RoboDynaExp/script_exp/interactive_catch_marbles_trapdoors.py --control keyboard
    /path/to/RoboDynaExp/script_exp/interactive_catch_marbles_trapdoors.py --control robot
    /path/to/RoboDynaExp/script_exp/interactive_catch_marbles_trapdoors.py --control robot --robot-motion interpolate

Keyboard mode opens trapdoors directly. Robot mode taps ``buttons[i]`` with
the matching arm (TCP-limited press). Sandbox only.
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
    RobotButtonController,
    add_robot_motion_arg,
    make_button_controller,
    report_task_result,
    print_mode_controls,
)


CONTROLS_KEYBOARD = """
  1 / Q  →  open button/trapdoor index 0
  2 / Up Arrow  →  open button/trapdoor index 1
  3 / E  →  open button/trapdoor index 2
  4 / R  →  open button/trapdoor index 3

  Colors are printed at startup (left→right order may shuffle).
  Press the button whose color matches the moving marble.
  Opens via _open_door_direct (no arm).

  V                 toggle view: top-down ↔ head_camera
  Close the viewer window to quit.
"""

CONTROLS_ROBOT = """
  1 / Q  →  open button/trapdoor index 0
  2 / Up Arrow  →  open button/trapdoor index 1
  3 / E  →  open button/trapdoor index 2
  4 / R  →  open button/trapdoor index 3

  Colors are printed at startup (left→right order may shuffle).
  Press the button whose color matches the moving marble.
  Matching arm taps buttons[i].

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


_KEY_TO_IDX = {
    "1": 0, "q": 0,
    "2": 1, "up": 1,
    "3": 2, "e": 2,
    "4": 3, "r": 3,
}


class EdgeButtons:
    def __init__(self):
        self.prev = {name: False for name in _KEY_TO_IDX}

    def pressed_indices(self, window):
        fired = []
        for name, idx in _KEY_TO_IDX.items():
            down = window.key_down(name)
            if down and not self.prev[name]:
                fired.append(idx)
            self.prev[name] = down
        # De-dupe (1 and Q share index).
        return list(dict.fromkeys(fired))


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


def _print_color_map(env):
    names = list(getattr(env, "button_color_names", []) or [])
    target = int(getattr(env, "target_button_idx", -1))
    mapping = ", ".join(f"{i}:{c}" for i, c in enumerate(names))
    target_name = names[target] if 0 <= target < len(names) else "?"
    print(f"Buttons L→R: {mapping}")
    print(f"Target marble color: {target_name} (index {target})")


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

    edges = EdgeButtons()
    pending = {"idx": None}
    robot = None
    if args.control == "robot":
        def get_button(e, side):
            return e.buttons[int(pending["idx"])]

        def get_top_z(e, side):
            i = int(pending["idx"])
            return float(e.buttons[i].get_pose().p[2]) + float(e.button_half[2])

        def arms_for_mode(m):
            if m is None:
                return ()
            return (str(env._arm_for_door(int(m))),)

        def set_latch(e, m):
            # Hold flag only — does not open the door. Door opens in on_press
            # after the shared controller confirms TCP contact on the key.
            e._buttons_held.clear()
            if m is not None:
                e._buttons_held.add(int(m))

        def clear_latch(e):
            e._buttons_held.clear()

        def on_press(e, m):
            # Same physics frame as fingertip contact (TrapdoorPlannerButtonController).
            idx = int(m)
            color = e.button_color_names[idx]
            if e._open_door_direct(idx):
                # Visible reaction on the contact frame; swing finishes via _advance_doors.
                snap = min(55.0, float(e.door_open_angle_deg))
                e._set_door_pose(idx, snap)
                print(f"Opened trapdoor {idx} ({color}) on fingertip contact.")
            else:
                print(f"Could not open trapdoor {idx} ({color}; locked/already open).")
            viewer = getattr(e, "viewer", None)
            if viewer is not None:
                e.scene.update_render()
                viewer.render()

        def make_ctrl(for_mode):
            pending["idx"] = for_mode
            arm = str(env._arm_for_door(int(for_mode)))
            # Interpolate precomputes per side; rebuild so targets match this button.
            sides = (arm,) if args.robot_motion == "interpolate" else ("left", "right")
            press_depth = float(getattr(env, "button_press_depth", 0.03))
            shared_kwargs = dict(
                get_button=get_button,
                get_top_z=get_top_z,
                set_latch=set_latch,
                clear_latch=clear_latch,
                arms_for_mode=arms_for_mode,
                on_press=on_press,
                hold=False,
                sides=sides,
            )
            if args.robot_motion == "interpolate":
                # Shared interpolate path — no trapdoors planner overrides.
                return make_button_controller(
                    env, ArmTag, "interpolate", **shared_kwargs
                )
            return TrapdoorPlannerButtonController(
                env,
                ArmTag,
                single_press_depth=press_depth,
                **shared_kwargs,
            )

        robot = {"ctrl": None, "make": make_ctrl}

    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    views = make_viewer_view_toggle(env, viewer)

    motion = f", robot-motion={args.robot_motion}" if args.control == "robot" else ""
    print(f"Control={args.control}{motion}. Tap 1–4 / QWER to open a trapdoor.")

    left_track_since = None
    settle_s = 0.6
    try:
        while not viewer.closed:
            views.update(viewer.window)
            frame_start = time.perf_counter()
            for idx in edges.pressed_indices(viewer.window):
                if args.control == "keyboard":
                    if env._open_door_direct(idx):
                        color = env.button_color_names[idx]
                        print(f"Opened trapdoor {idx} ({color}).")
                    else:
                        print(f"Could not open trapdoor {idx} (locked/already open).")
                elif robot is not None:
                    if idx < 0 or idx >= env.n_buttons:
                        continue
                    color = env.button_color_names[idx]
                    arm = env._arm_for_door(idx)
                    print(f"Robot tapping button {idx} ({color}) with {arm} arm...")
                    if args.robot_motion == "interpolate" or robot["ctrl"] is None:
                        robot["ctrl"] = robot["make"](idx)
                    else:
                        pending["idx"] = idx
                    robot["ctrl"].update(idx)
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
            if robot is not None and robot["ctrl"] is not None:
                robot["ctrl"].release()
        finally:
            env.close_env()


if __name__ == "__main__":
    main()
