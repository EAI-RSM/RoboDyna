"""Shared helpers for ``script_exp/interactive_*.py`` sandboxes."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def bootstrap_repo():
    """chdir to repo root and put it on ``sys.path`` (any caller cwd)."""
    os.chdir(REPO_ROOT)
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    bench = str(REPO_ROOT / "script" / "bench_script")
    if bench not in sys.path:
        sys.path.insert(0, bench)


def embodiment_config(robot_file):
    with open(Path(robot_file) / "config.yml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def configure_task(task_name: str, config_name: str, seed: int, use_robot: bool,
                   task_arg_overrides: dict | None = None):
    """Load ``task_config/<config_name>.yml`` and wire embodiment paths."""
    from envs import CONFIGS_PATH

    config_path = REPO_ROOT / "task_config" / f"{config_name}.yml"
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if task_arg_overrides:
        task_args = config.setdefault("task_args", {}).setdefault(task_name, {})
        task_args.update(task_arg_overrides)

    config.update(
        task_name=task_name,
        render_freq=1,
        now_ep_num=0,
        seed=seed,
        need_plan=bool(use_robot),
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
    config["left_embodiment_config"] = embodiment_config(config["left_robot_file"])
    config["right_embodiment_config"] = embodiment_config(config["right_robot_file"])
    return config


def print_banner(title: str, lines: list[str]):
    width = max(len(title), *(len(line) for line in lines), 40)
    bar = "=" * (width + 4)
    print(bar)
    print(f"  {title}")
    print(bar)
    for line in lines:
        print(f"  {line}")
    print(bar)


def edge_pressed(window, key: str, prev: dict) -> bool:
    """True on the frame a key transitions from up → down."""
    if window.key_press(key):
        prev[key] = True
        return True
    down = bool(window.key_down(key))
    was = prev.get(key, False)
    prev[key] = down
    return down and not was


def arrow_nudge_xy(window, step: float = 0.003) -> np.ndarray:
    """Map arrow keys to a world XY delta (viewer top-down friendly)."""
    dx = dy = 0.0
    if window.key_down("left"):
        dx -= step
    if window.key_down("right"):
        dx += step
    if window.key_down("up"):
        dy += step
    if window.key_down("down"):
        dy -= step
    return np.array([dx, dy], dtype=np.float64)


def set_actor_pose(actor, xyz, quat=None):
    import sapien

    if quat is None:
        quat = list(actor.get_pose().q)
    actor.actor.set_pose(sapien.Pose(list(xyz), list(quat)))


def hold_dynamic_at(rigid, actor, xyz, quat=None):
    """Keep a dynamic body pinned at ``xyz`` for keyboard teleop hold."""
    import sapien

    if quat is None:
        quat = list(actor.get_pose().q)
    pose = sapien.Pose(list(xyz), list(quat))
    actor.actor.set_pose(pose)
    if rigid is None:
        return
    try:
        rigid.set_disable_gravity(True)
        rigid.set_linear_velocity([0.0, 0.0, 0.0])
        rigid.set_angular_velocity([0.0, 0.0, 0.0])
        if hasattr(rigid, "set_kinematic_target"):
            try:
                rigid.set_kinematic_target(pose)
            except Exception:
                pass
    except Exception:
        pass


def release_dynamic(rigid):
    if rigid is None:
        return
    try:
        rigid.set_kinematic(False)
    except Exception:
        pass
    try:
        rigid.set_disable_gravity(False)
        rigid.set_linear_velocity([0.0, 0.0, 0.0])
        rigid.set_angular_velocity([0.0, 0.0, 0.0])
    except Exception:
        pass


def report_task_result(env, detail: str | None = None) -> bool:
    """Print ``Task complete: SUCCESS|FAILURE`` from ``check_success``; return success."""
    try:
        ok = bool(env.check_success())
    except Exception as exc:
        print(f"Task complete: FAILURE (check_success error: {exc})")
        return False
    if detail is None and not ok:
        detail = getattr(env, "_last_fail_reason", None) or None
    status = "SUCCESS" if ok else "FAILURE"
    print(f"Task complete: {status}" + (f" ({detail})" if detail else ""))
    return ok


def resolve_head_camera(env=None, viewer=None):
    """Find the sapien ``head_camera`` render camera on ``env`` or ``viewer``."""
    if env is not None:
        cams = getattr(env, "cameras", None)
        if cams is not None:
            names = list(getattr(cams, "static_camera_name", []) or [])
            clist = list(getattr(cams, "static_camera_list", []) or [])
            if "head_camera" in names:
                return clist[names.index("head_camera")]
    if viewer is None and env is not None:
        viewer = getattr(env, "viewer", None)
    if viewer is not None:
        for cam in getattr(viewer, "cameras", []) or []:
            if getattr(cam, "name", None) == "head_camera":
                return cam
    return None


# Nadir viewer: +X camera offset shifts the table left in the frame.
_TOPDOWN_VIEW_X_OFFSET = 0.08


def default_topdown_xyz(env=None) -> tuple[float, float, float]:
    """Overhead pose: table near image center, zoomed to table + dual arms."""
    bias = getattr(env, "table_xy_bias", None) if env is not None else None
    if bias is None:
        bx = by = 0.0
    else:
        bx = float(bias[0])
        by = float(bias[1])
    # Z ≈ 1.68 with ~65° fovy keeps both arm bases (x≈±embodiment_dis/2) in frame.
    return (bx + _TOPDOWN_VIEW_X_OFFSET, by, 1.68)


class ViewerViewToggle:
    """Press V to switch the interactive viewer between top-down and head cam.

    sapien's ``focus_camera`` follow-path is disabled in this build
    (``_handle_focused_camera`` commented out), so we copy the head camera
    pose onto the free-fly viewer each frame instead.
    """

    # Table near center (slight +X so the table sits a bit left of frame center).
    DEFAULT_TOPDOWN_XYZ = (_TOPDOWN_VIEW_X_OFFSET, 0.0, 1.68)
    DEFAULT_TOPDOWN_RPY = (0.0, -np.pi / 2.0, -np.pi / 2.0)
    # Viewer default is 90°; narrow this so the table fills the frame
    # while still showing both arms.
    DEFAULT_TOPDOWN_FOVY = float(np.deg2rad(65.0))
    DEFAULT_HEAD_FOVY = float(np.pi / 2.0)

    def __init__(
        self,
        viewer,
        head_camera=None,
        topdown_xyz=None,
        topdown_rpy=None,
        topdown_fovy=None,
        capture_current_as_topdown: bool = False,
        warn_missing_head: bool = False,
    ):
        self.viewer = viewer
        self._prev_v = False
        self.mode = "topdown"  # always start overhead; V switches to head_camera
        self._head = head_camera
        if self._head is None:
            self._head = resolve_head_camera(viewer=viewer)
        if self._head is None and warn_missing_head:
            print("Warning: head_camera not found; V toggle will stay on top-down.")

        self._topdown_pose = None
        self._topdown_xyz = None
        self._topdown_rpy = None
        self._topdown_fovy = float(
            self.DEFAULT_TOPDOWN_FOVY if topdown_fovy is None else topdown_fovy
        )
        if topdown_xyz is not None:
            self._topdown_xyz = tuple(topdown_xyz)
            self._topdown_rpy = tuple(
                topdown_rpy if topdown_rpy is not None else self.DEFAULT_TOPDOWN_RPY
            )
            self.apply(announce=False)
        elif capture_current_as_topdown:
            try:
                self._topdown_pose = viewer.window.get_camera_pose()
            except Exception:
                self._topdown_xyz = self.DEFAULT_TOPDOWN_XYZ
                self._topdown_rpy = self.DEFAULT_TOPDOWN_RPY
                self.apply(announce=False)
        else:
            self._topdown_xyz = self.DEFAULT_TOPDOWN_XYZ
            self._topdown_rpy = self.DEFAULT_TOPDOWN_RPY
            self.apply(announce=False)

    def _control_window(self):
        for plugin in getattr(self.viewer, "plugins", []) or []:
            if hasattr(plugin, "fps_camera_controller") and hasattr(plugin, "set_camera_xyz"):
                return plugin
        return getattr(self.viewer, "control_window", None)

    def _set_fovy(self, fovy: float):
        window = getattr(self.viewer, "window", None)
        if window is None:
            return
        try:
            near = float(getattr(window, "near", 0.1))
            far = float(getattr(window, "far", 1000.0))
            window.set_camera_parameters(near, far, float(fovy))
        except Exception:
            try:
                cw = self._control_window()
                if cw is not None:
                    cw.fovy = float(fovy)
            except Exception:
                pass

    def _set_viewer_pose(self, pose):
        """Snap free-fly camera to ``pose`` and keep the FPS controller in sync."""
        try:
            self.viewer.focus_camera(None)
        except Exception:
            pass
        self.viewer.set_camera_pose(pose)
        cw = self._control_window()
        if cw is not None and hasattr(cw, "_sync_fps_camera_controller"):
            cw._sync_fps_camera_controller()

    def apply(self, announce=True):
        if self.mode == "head" and self._head is not None:
            self._set_fovy(self.DEFAULT_HEAD_FOVY)
            self._set_viewer_pose(self._head.global_pose)
            if announce:
                print("View: head_camera")
            return
        try:
            self.viewer.focus_camera(None)
        except Exception:
            pass
        self._set_fovy(self._topdown_fovy)
        if self._topdown_pose is not None:
            self._set_viewer_pose(self._topdown_pose)
        else:
            xyz = self._topdown_xyz or self.DEFAULT_TOPDOWN_XYZ
            rpy = self._topdown_rpy or self.DEFAULT_TOPDOWN_RPY
            self.viewer.set_camera_xyz(*xyz)
            self.viewer.set_camera_rpy(*rpy)
        if announce:
            print("View: top-down")

    def _v_pressed(self, window) -> bool:
        down = bool(window.key_down("v"))
        edge = down and not self._prev_v
        self._prev_v = down
        if edge:
            return True
        try:
            return bool(window.key_press("v"))
        except Exception:
            return False

    def update(self, window):
        if self._v_pressed(window):
            if self.mode == "topdown":
                if self._head is None:
                    print("head_camera not available; staying on top-down.")
                else:
                    self.mode = "head"
                    self.apply(announce=True)
            else:
                self.mode = "topdown"
                self.apply(announce=True)
            return
        # Keep the overhead view fixed.  SAPIEN's FPS controller otherwise
        # translates the viewer in response to WASD while the task is running.
        if self.mode == "topdown":
            self.apply(announce=False)
        # Keep head view locked to the moving head_camera.
        if self.mode == "head" and self._head is not None:
            self._set_viewer_pose(self._head.global_pose)


def make_viewer_view_toggle(
    env,
    viewer=None,
    topdown_xyz=None,
    topdown_rpy=None,
    capture_current_as_topdown: bool = False,
    **kwargs,
) -> ViewerViewToggle:
    """Build a V-key top-down ↔ head_camera toggle for an interactive env.

    Always starts on a zoomed, table-centered top-down view.
    Pass ``topdown_xyz`` / ``topdown_rpy`` only to override that framing.
    """
    if viewer is None:
        viewer = getattr(env, "viewer", None)
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    if topdown_xyz is None and not capture_current_as_topdown:
        topdown_xyz = default_topdown_xyz(env)
        if topdown_rpy is None:
            topdown_rpy = ViewerViewToggle.DEFAULT_TOPDOWN_RPY
    return ViewerViewToggle(
        viewer,
        head_camera=resolve_head_camera(env, viewer),
        topdown_xyz=topdown_xyz,
        topdown_rpy=topdown_rpy,
        capture_current_as_topdown=capture_current_as_topdown,
        **kwargs,
    )


def run_viewer_loop(env, on_step, should_stop=None, max_steps: int | None = None,
                    overhead: bool = True, is_done=None):
    """Standard interactive loop: callback → kinematics → step → render.

    ``is_done(step)`` may return ``True`` / ``False``, or ``(done, detail)``.
    When done, prints SUCCESS/FAILURE via ``report_task_result`` and exits.
    ``should_stop`` remains a raw break (no auto print) for backward compatibility.
    Starts top-down; press V to toggle top-down ↔ head_camera.
    """
    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    # Top-down is applied by make_viewer_view_toggle; keep optional pre-set for
    # callers that pass capture_current_as_topdown via a custom loop.
    if overhead:
        viewer.set_camera_xyz(*ViewerViewToggle.DEFAULT_TOPDOWN_XYZ)
        viewer.set_camera_rpy(*ViewerViewToggle.DEFAULT_TOPDOWN_RPY)
    views = make_viewer_view_toggle(env, viewer)
    step = 0
    try:
        while not viewer.closed:
            frame_start = time.perf_counter()
            views.update(viewer.window)
            if on_step is not None:
                on_step(viewer.window, step)
            env._update_kinematic_tasks()
            env.scene.step()
            env.scene.update_render()
            viewer.render()
            step += 1
            if is_done is not None:
                result = is_done(step)
                if isinstance(result, tuple):
                    done = bool(result[0])
                    detail = result[1] if len(result) > 1 else None
                else:
                    done, detail = bool(result), None
                if done:
                    report_task_result(env, detail)
                    break
            if should_stop is not None and should_stop(step):
                break
            if max_steps is not None and step >= max_steps:
                print(f"Reached max_steps={max_steps}; evaluating.")
                report_task_result(env, f"max_steps={max_steps}")
                break
            remaining = float(env.scene.get_timestep()) - (time.perf_counter() - frame_start)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        env.close_env()


# ---------------------------------------------------------------------------
# Shared robot key/button press (same routine as interactive_sort_apples)
# ---------------------------------------------------------------------------


def add_robot_motion_arg(parser, robot_motion_default: str = "planner"):
    """Add ``--control`` + ``--robot-motion`` flags used by every interactive script."""
    parser.add_argument(
        "--control",
        choices=("keyboard", "robot"),
        default="keyboard",
        help="Interaction method (default: keyboard)",
    )
    parser.add_argument(
        "--robot-motion",
        choices=("planner", "interpolate"),
        default=robot_motion_default,
        help=(
            "Robot key-press implementation; interpolate is a faster test mode "
            f"(default: {robot_motion_default})"
        ),
    )
    return parser


def print_mode_controls(task_name: str, mode: str, *, keyboard: str, robot: str) -> None:
    """Print only the help block for the selected ``--control`` mode."""
    body = (robot if mode == "robot" else keyboard).strip("\n")
    bar = "=" * 60
    print(f"{bar}\n {task_name} — {mode} controls\n{bar}\n{body}\n{bar}")


def default_arms_for_mode(mode):
    """``left`` / ``right`` / ``dump`` → arm side names (sort_apples style)."""
    if mode == "dump":
        return ("left", "right")
    return (mode,) if mode else ()


class RobotButtonController:
    """Hold-to-actuate (or tap) key press: grasp → TCP-limited press → latch → clear lift.

    Matches ``interactive_sort_apples.RobotButtonController``. Task scripts supply
    actor / top-z / latch adapters so the motion routine stays shared.
    """

    PRESS_HOLD_STEPS = 8
    CONTACT_CLEARANCE = 0.008
    # Must cover grasp_dis hover (~8 cm). A 4 cm cap left the arm stuck above the key.
    MAX_PRESS_DEPTH = 0.14
    PRESS_STEP = 0.02
    JOINT_RETRACT_STEPS = 25
    CLEAR_ABOVE_ACTIVE = 0.04

    def __init__(
        self,
        env,
        arm_tag,
        get_button,
        get_top_z,
        set_latch=None,
        clear_latch=None,
        arms_for_mode=None,
        on_press=None,
        hold: bool = True,
        active_dz: float | None = None,
        grasp_dis: float = 0.08,
        pre_grasp_dis: float = 0.08,
        max_press_depth: float | None = None,
        **_ignored,
    ):
        """
        Parameters
        ----------
        get_button : callable(env, side) -> actor
        get_top_z : callable(env, side) -> float
        set_latch : callable(env, mode) | None
        clear_latch : callable(env) | None
        arms_for_mode : callable(mode) -> iterable[str]
        on_press : callable(env, mode) | None  — side effect after contact lands
        hold : if False, press then immediately lift (edge tap)
        active_dz : proximity clear height; default env.PRESS_DZ_ACTIVE or 0.12
        max_press_depth : cap on -Z travel; default covers grasp_dis + margin
        """
        self.env = env
        self.arm_tag = arm_tag
        self.get_button = get_button
        self.get_top_z = get_top_z
        self.set_latch = set_latch or (lambda _e, mode: setattr(_e, "_expert_hold", mode))
        self.clear_latch = clear_latch or (lambda _e: setattr(_e, "_expert_hold", None))
        self.arms_for_mode = arms_for_mode or default_arms_for_mode
        self.on_press = on_press
        self.hold = hold
        self.active_dz = active_dz
        self.grasp_dis = grasp_dis
        self.pre_grasp_dis = pre_grasp_dis
        # Allow enough -Z to reach the key from the grasp hover pose.
        self.max_press_depth = float(
            max_press_depth if max_press_depth is not None
            else max(self.MAX_PRESS_DEPTH, float(grasp_dis) + 0.06)
        )
        self.mode = None
        self._hover_qpos = {}

    def _tcp_z(self, side):
        get_tcp = (self.env.robot.get_left_tcp_pose if side == "left"
                   else self.env.robot.get_right_tcp_pose)
        try:
            return float(get_tcp()[2])
        except Exception:
            return None

    def _drive_qpos(self, side):
        joints = self.env.robot.left_arm_joints if side == "left" else self.env.robot.right_arm_joints
        return np.asarray([joint.get_drive_target()[0] for joint in joints], dtype=np.float64)

    def _press_depth_tcp(self, side):
        tcp_z = self._tcp_z(side)
        if tcp_z is None:
            return 0.0
        top_z = float(self.get_top_z(self.env, side))
        need = float(tcp_z) - top_z - self.CONTACT_CLEARANCE
        return float(np.clip(need, 0.0, self.max_press_depth))

    def _active_band(self) -> float:
        """TCP height above key top that still counts as an active press."""
        if self.active_dz is not None:
            return float(self.active_dz)
        return float(getattr(self.env, "PRESS_DZ_ACTIVE", 0.12))

    def _in_key_contact(self, side, slack: float = 0.025) -> bool:
        """True when TCP is at/near the key top — stop descending further."""
        tcp_z = self._tcp_z(side)
        if tcp_z is None:
            return False
        top_z = float(self.get_top_z(self.env, side))
        return float(tcp_z) <= top_z + self.CONTACT_CLEARANCE + float(slack)

    def _in_active_zone(self, side) -> bool:
        """True when TCP is inside the task proximity band (may actuate)."""
        tcp_z = self._tcp_z(side)
        if tcp_z is None:
            return False
        top_z = float(self.get_top_z(self.env, side))
        return float(tcp_z) <= top_z + self._active_band()

    def _mode_in_contact(self, mode) -> bool:
        sides = tuple(self.arms_for_mode(mode))
        return bool(sides) and all(self._in_key_contact(side) for side in sides)

    def _mode_in_active_zone(self, mode) -> bool:
        sides = tuple(self.arms_for_mode(mode))
        return bool(sides) and all(self._in_active_zone(side) for side in sides)

    def _press_until_contact(self, mode) -> bool:
        """Step -Z until TCP touches the key (or budget / plan failure)."""
        sides = tuple(self.arms_for_mode(mode))
        traveled = {side: 0.0 for side in sides}
        max_steps = int(self.max_press_depth / self.PRESS_STEP) + 2
        for _ in range(max_steps):
            if self._mode_in_contact(mode):
                return True
            pending = [
                side for side in sides
                if not self._in_key_contact(side) and traveled[side] < self.max_press_depth
            ]
            if not pending:
                break
            step = min(
                self.PRESS_STEP,
                *(self._press_depth_tcp(side) or self.PRESS_STEP for side in pending),
            )
            step = max(float(step), 0.005)
            self.env.plan_success = True
            self.env.move(*[
                self.env.move_by_displacement(self.arm_tag(side), z=-step)
                for side in pending
            ])
            if not self.env.plan_success:
                print("Press step failed; stopping descent.")
                break
            for side in pending:
                traveled[side] += step
        return self._mode_in_contact(mode)

    def _interpolate_to_qpos(self, targets):
        if not targets:
            return
        starts = {side: self._drive_qpos(side) for side in targets}
        for step in range(1, self.JOINT_RETRACT_STEPS + 1):
            alpha = step / self.JOINT_RETRACT_STEPS
            smooth = alpha * alpha * (3.0 - 2.0 * alpha)
            for side, target in targets.items():
                position = starts[side] + (target - starts[side]) * smooth
                velocity = (target - starts[side]) / max(self.JOINT_RETRACT_STEPS, 1)
                self.env.robot.set_arm_joints(position, velocity, side)
            self.clear_latch(self.env)
            self.env._update_kinematic_tasks()
            self.env.scene.step()
            viewer = getattr(self.env, "viewer", None)
            if viewer is not None:
                self.env.scene.update_render()
                viewer.render()
        for side, target in targets.items():
            self.env.robot.set_arm_joints(target, np.zeros_like(target), side)
        self.env.plan_success = True
        self.env._last_plan_fail = None

    def _activate_press(self, mode) -> bool:
        """Latch + on_press when TCP is in the active press band."""
        if not self._mode_in_active_zone(mode):
            print(f"No key contact for {mode}; not actuating (door/key stays idle).")
            self.clear_latch(self.env)
            return False
        self.set_latch(self.env, mode)
        if self.on_press is not None:
            try:
                self.on_press(self.env, mode)
            except Exception as exc:
                print(f"on_press callback failed: {exc}")
        return True

    def _move_to_buttons(self, mode):
        sides = tuple(self.arms_for_mode(mode))
        if not sides:
            return
        actions = []
        for side in sides:
            button = self.get_button(self.env, side)
            actions.append(self.env.grasp_actor(
                button, arm_tag=self.arm_tag(side),
                pre_grasp_dis=self.pre_grasp_dis, grasp_dis=self.grasp_dis,
                contact_point_id=0, gripper_pos=0.0,
            ))
        self.env.move(*actions)
        if not self.env.plan_success:
            return
        self._hover_qpos = {side: self._drive_qpos(side) for side in sides}
        depths = {side: self._press_depth_tcp(side) for side in sides}
        print(
            "Press depths (TCP-limited): "
            + ", ".join(f"{side}={depths[side] * 100:.1f}cm" for side in depths)
            + f" (max {self.max_press_depth * 100:.1f}cm)"
        )
        # Step down to the keytop when possible. Actuation uses the wider
        # active band (e.g. PRESS_DZ_ACTIVE) so a fingertip touch that never
        # quite satisfies the tight stop-clearance still latches _expert_hold.
        self._press_until_contact(mode)
        if not self._mode_in_active_zone(mode):
            print("Could not reach key active zone; returning to hover (no latch).")
            self._interpolate_to_qpos(self._hover_qpos)
            self.clear_latch(self.env)
            return
        if not self._activate_press(mode):
            return
        for _ in range(self.PRESS_HOLD_STEPS):
            self.set_latch(self.env, mode)
            self.env._update_kinematic_tasks()
            self.env.scene.step()
        print(f"Robot pressed {mode} (hold key to keep actuated)." if self.hold
              else f"Robot tapped {mode}.")

    def _clear_z_target(self, side):
        active = self.active_dz
        if active is None:
            active = float(getattr(self.env, "PRESS_DZ_ACTIVE", 0.12))
        return float(self.get_top_z(self.env, side)) + float(active) + self.CLEAR_ABOVE_ACTIVE

    def _lift_clear_of_keys(self, mode):
        lifts = []
        for side in self.arms_for_mode(mode):
            tcp_z = self._tcp_z(side)
            clear_z = self._clear_z_target(side)
            if tcp_z is None:
                lifts.append((side, 0.10))
                continue
            extra = clear_z - float(tcp_z)
            if extra > 0.001:
                lifts.append((side, extra))
        if not lifts:
            return
        self.env.plan_success = True
        self.env._last_plan_fail = None
        self.env.move(*[
            self.env.move_by_displacement(self.arm_tag(side), z=dz)
            for side, dz in lifts
        ])
        if not self.env.plan_success:
            print("Clear-lift plan failed; applying joint-space +Z retreat.")
            starts = {side: self._drive_qpos(side) for side, _ in lifts}
            self.env.plan_success = True
            for side, dz in lifts:
                pose = np.asarray(
                    self.env.robot.get_left_ee_pose() if side == "left"
                    else self.env.robot.get_right_ee_pose(),
                    dtype=np.float64,
                ).copy()
                pose[2] += dz
                planner = (self.env.robot.left_plan_path if side == "left"
                           else self.env.robot.right_plan_path)
                result = planner(pose, last_qpos=np.asarray(starts[side], dtype=np.float32))
                if result is not None and result.get("status") == "Success":
                    target = np.asarray(result["position"][-1], dtype=np.float64)
                    self._interpolate_to_qpos({side: target})
            self.env.plan_success = True
            self.env._last_plan_fail = None
        self.clear_latch(self.env)
        for _ in range(12):
            self.env._update_kinematic_tasks()
            self.env.scene.step()

    def _lift_from_buttons(self, mode):
        targets = {
            side: self._hover_qpos[side]
            for side in self.arms_for_mode(mode)
            if side in self._hover_qpos
        }
        if targets:
            self._interpolate_to_qpos(targets)
        else:
            actions = [
                self.env.move_by_displacement(self.arm_tag(side), z=0.06)
                for side in self.arms_for_mode(mode)
            ]
            if actions:
                self.env.plan_success = True
                self.env.move(*actions)
        self._lift_clear_of_keys(mode)
        self._hover_qpos.clear()

    def update(self, requested_mode):
        if not self.env.plan_success:
            detail = getattr(self.env, "_last_plan_fail", None)
            print(f"Robot button trajectory failed ({detail or 'unknown'}); controls remain available.")
            self.env.plan_success = True
            self.env._last_plan_fail = None
        if self.hold and requested_mode == self.mode:
            if requested_mode is not None:
                self.set_latch(self.env, requested_mode)
            return
        self.clear_latch(self.env)
        if self.mode is not None:
            self._lift_from_buttons(self.mode)
        if requested_mode is not None:
            self._move_to_buttons(requested_mode)
            if not self.hold:
                # Edge tap: press then release in one update.
                self.clear_latch(self.env)
                self._lift_from_buttons(requested_mode)
                requested_mode = None
        if not self.env.plan_success:
            detail = getattr(self.env, "_last_plan_fail", None)
            print(f"Robot button motion failed ({detail or 'unknown'}); release and try again.")
            if self._hover_qpos:
                self._interpolate_to_qpos(self._hover_qpos)
                self._hover_qpos.clear()
            requested_mode = None
            self.clear_latch(self.env)
        self.mode = requested_mode

    def release(self):
        self.update(None)


class InterpolatedRobotButtonController:
    """Fast joint interpolation between precomputed hover/press poses (test mode)."""

    DURATION = 0.28

    def __init__(
        self,
        env,
        arm_tag,
        get_button,
        get_top_z,
        set_latch=None,
        clear_latch=None,
        arms_for_mode=None,
        on_press=None,
        hold: bool = True,
        active_dz: float | None = None,
        sides=("left", "right"),
        **_ignored,
    ):
        self.env = env
        self.arm_tag = arm_tag
        self.get_button = get_button
        self.get_top_z = get_top_z
        self.set_latch = set_latch or (lambda _e, mode: setattr(_e, "_expert_hold", mode))
        self.clear_latch = clear_latch or (lambda _e: setattr(_e, "_expert_hold", None))
        self.arms_for_mode = arms_for_mode or default_arms_for_mode
        self.on_press = on_press
        self.hold = hold
        self.active_dz = active_dz
        self.mode = None
        self._start = {}
        self._targets = {}
        self._started_at = None
        self._sides = tuple(sides)
        self.hover, self.pressed = self._build_button_targets()
        print("Fast interpolation targets prepared (collision checking is only done during setup).")

    def _plan(self, side, pose, last_qpos=None):
        planner = self.env.robot.left_plan_path if side == "left" else self.env.robot.right_plan_path
        if last_qpos is not None:
            last_qpos = np.asarray(last_qpos, dtype=np.float32)
        result = planner(pose, last_qpos=last_qpos)
        if result is None or result.get("status") != "Success":
            reason = "no result" if result is None else result.get("reason", "unknown reason")
            raise RuntimeError(f"Could not prepare the {side} button interpolation target: {reason}")
        return np.asarray(result["position"][-1], dtype=np.float32)

    def _drive_targets(self, side):
        joints = self.env.robot.left_arm_joints if side == "left" else self.env.robot.right_arm_joints
        return np.asarray([joint.get_drive_target()[0] for joint in joints], dtype=np.float64)

    def _ee_tcp_world_dz(self, side):
        get_ee = (self.env.robot.get_left_ee_pose if side == "left"
                  else self.env.robot.get_right_ee_pose)
        get_tcp = (self.env.robot.get_left_tcp_pose if side == "left"
                   else self.env.robot.get_right_tcp_pose)
        return max(0.0, float(get_ee()[2]) - float(get_tcp()[2]))

    def _build_button_targets(self):
        hover, pressed = {}, {}
        active = self.active_dz
        if active is None:
            active = float(getattr(self.env, "PRESS_DZ_ACTIVE", 0.12))
        clear_dis = float(active) + 0.04
        for side in self._sides:
            button = self.get_button(self.env, side)
            hover_pose = self.env.get_grasp_pose(
                button, self.arm_tag(side), contact_point_id=0, pre_dis=clear_dis,
            )
            hover_qpos = self._plan(side, hover_pose)
            rest_qpos = self._drive_targets(side)
            self.env.robot.set_arm_joints(hover_qpos, np.zeros_like(hover_qpos), side)
            press_pose = np.asarray(hover_pose, dtype=np.float64).copy()
            press_pose[2] = (
                float(self.get_top_z(self.env, side)) + self._ee_tcp_world_dz(side) + 0.008
            )
            # Allow a deep enough press from the clear hover to reach the key.
            press_pose[2] = max(press_pose[2], float(hover_pose[2]) - 0.16)
            self.env.robot.set_arm_joints(rest_qpos, np.zeros_like(rest_qpos), side)
            hover[side] = hover_qpos
            pressed[side] = self._plan(side, press_pose, last_qpos=hover_qpos)
        return hover, pressed

    def _begin(self, requested_mode):
        self.clear_latch(self.env)
        active = set(self.arms_for_mode(requested_mode))
        moving = set(self.arms_for_mode(self.mode)) | active
        # Only sides we precomputed.
        moving = {side for side in moving if side in self.hover}
        self._start = {side: self._drive_targets(side) for side in moving}
        self._targets = {
            side: (self.pressed[side] if side in active else self.hover[side])
            for side in moving
        }
        self._started_at = time.perf_counter()
        self.mode = requested_mode

    def _tcp_z(self, side):
        get_tcp = (self.env.robot.get_left_tcp_pose if side == "left"
                   else self.env.robot.get_right_tcp_pose)
        try:
            return float(get_tcp()[2])
        except Exception:
            return None

    def _active_band(self) -> float:
        if self.active_dz is not None:
            return float(self.active_dz)
        return float(getattr(self.env, "PRESS_DZ_ACTIVE", 0.12))

    def _in_key_contact(self, side, slack: float = 0.02) -> bool:
        tcp_z = self._tcp_z(side)
        if tcp_z is None:
            return False
        top_z = float(self.get_top_z(self.env, side))
        return float(tcp_z) <= top_z + 0.008 + float(slack)

    def _in_active_zone(self, side) -> bool:
        tcp_z = self._tcp_z(side)
        if tcp_z is None:
            return False
        top_z = float(self.get_top_z(self.env, side))
        return float(tcp_z) <= top_z + self._active_band()

    def _mode_in_contact(self, mode) -> bool:
        sides = tuple(self.arms_for_mode(mode))
        return bool(sides) and all(self._in_key_contact(side) for side in sides)

    def _mode_in_active_zone(self, mode) -> bool:
        sides = tuple(self.arms_for_mode(mode))
        return bool(sides) and all(self._in_active_zone(side) for side in sides)

    def _activate_press(self, mode) -> bool:
        if not self._mode_in_active_zone(mode):
            print(f"No key contact for {mode}; not actuating (door/key stays idle).")
            self.clear_latch(self.env)
            return False
        self.set_latch(self.env, mode)
        if self.on_press is not None:
            try:
                self.on_press(self.env, mode)
            except Exception as exc:
                print(f"on_press callback failed: {exc}")
        return True

    def update(self, requested_mode):
        if not self.hold and requested_mode is not None and requested_mode != self.mode:
            # Tap: go to press, latch only on contact, return to hover.
            self._begin(requested_mode)
            while self._started_at is not None:
                self._advance(activate=False)
            activated = self._activate_press(requested_mode)
            if activated:
                print(f"Robot tapped {requested_mode} with fast interpolation.")
            self._begin(None)
            while self._started_at is not None:
                self._advance(activate=False)
            self.clear_latch(self.env)
            self.mode = None
            return
        if requested_mode != self.mode:
            self._begin(requested_mode)
        if self._started_at is None:
            if self.mode is not None:
                if self._mode_in_active_zone(self.mode):
                    self.set_latch(self.env, self.mode)
                else:
                    self.clear_latch(self.env)
            else:
                self.clear_latch(self.env)
            return
        self._advance(activate=True)


    def _advance(self, activate: bool = True):
        progress = min(1.0, (time.perf_counter() - self._started_at) / self.DURATION)
        smooth = progress * progress * (3.0 - 2.0 * progress)
        for side, target in self._targets.items():
            position = self._start[side] + (target - self._start[side]) * smooth
            velocity = (target - self._start[side]) / self.DURATION if progress < 1.0 else np.zeros_like(target)
            self.env.robot.set_arm_joints(position, velocity, side)
        if progress >= 1.0:
            self._started_at = None
            if self.mode is not None:
                if activate:
                    if self._activate_press(self.mode):
                        print(f"Robot pressed {self.mode} with fast interpolation.")
            else:
                self.clear_latch(self.env)

    def release(self):
        self.update(None)


def make_button_controller(env, arm_tag, robot_motion: str = "planner", **kwargs):
    """Factory: planner (TCP press) or interpolate (joint-space) button controller."""
    if robot_motion == "interpolate":
        return InterpolatedRobotButtonController(env, arm_tag, **kwargs)
    return RobotButtonController(env, arm_tag, **kwargs)
