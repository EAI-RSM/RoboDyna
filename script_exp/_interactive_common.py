"""Shared helpers for ``script_exp/interactive_*.py`` sandboxes."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

# When set by interactive_task_gui / household_task_gui, report_task_result
# writes {"ok": bool, "detail": str} here so the launcher can show failure reasons.
TASK_RESULT_ENV = "ROBODYNA_TASK_RESULT_FILE"

# ANSI colors for interactive CLI (TTY only; respect NO_COLOR / FORCE_COLOR).
_ANSI_BLUE = "\033[34m"
_ANSI_GREEN = "\033[32m"
_ANSI_RED = "\033[31m"
_ANSI_RESET = "\033[0m"


def _stdout_supports_color() -> bool:
    """Color when stdout is a TTY, unless ``NO_COLOR`` is set.

    ``FORCE_COLOR`` (any value other than empty / ``0``) forces color even when
    not a TTY — useful for CI demos / piped capture.
    """
    if os.environ.get("NO_COLOR"):
        return False
    force = os.environ.get("FORCE_COLOR")
    if force not in (None, "", "0"):
        return True
    return hasattr(sys.stdout, "isatty") and bool(sys.stdout.isatty())


def colorize(text: str, ansi_code: str) -> str:
    """Wrap ``text`` in ANSI color when stdout is a color-capable TTY."""
    if not text or not _stdout_supports_color():
        return text
    return f"{ansi_code}{text}{_ANSI_RESET}"


def print_instructions(*args, sep: str = " ", end: str = "\n", flush: bool = False) -> None:
    """Print controls / how-to text in blue."""
    msg = sep.join(str(a) for a in args)
    print(colorize(msg, _ANSI_BLUE), end=end, flush=flush)


def print_success(*args, sep: str = " ", end: str = "\n", flush: bool = False) -> None:
    """Print a success / SUCCESS result line in green."""
    msg = sep.join(str(a) for a in args)
    print(colorize(msg, _ANSI_GREEN), end=end, flush=flush)


def print_failure(*args, sep: str = " ", end: str = "\n", flush: bool = False) -> None:
    """Print a failure / FAILURE result line in red."""
    msg = sep.join(str(a) for a in args)
    print(colorize(msg, _ANSI_RED), end=end, flush=flush)


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


_VIEW_HELP_V = "V — cycle view: head_camera ↔ gripper(s)"


def _is_view_help_line(line: str) -> bool:
    low = line.lower()
    s = line.strip()
    return (
        "toggle view" in low
        or "cycle view" in low
        or "gripper view" in low
        or "gripper / wrist" in low
        or s.startswith("V —")
        or s.startswith("V:")
        or s.startswith("V ")
        or "V                 " in line
    )


def _normalize_view_help_lines(lines: list[str]) -> list[str]:
    """Keep a single V head↔gripper help line; drop stale top-down / G-view text."""
    out: list[str] = []
    saw_v = False
    for line in lines:
        if not _is_view_help_line(line):
            out.append(line)
            continue
        # Drop legacy G-only gripper-view lines; V owns the camera cycle now.
        s = line.strip()
        if (
            s.startswith("G ")
            or s.startswith("G:")
            or s.startswith("G —")
            or s.startswith("G\t")
        ) and ("gripper view" in line.lower() or "wrist" in line.lower()):
            continue
        if saw_v:
            continue
        indent = line[: len(line) - len(line.lstrip(" "))]
        # Preserve em-dash style used by some banners.
        sep = "—" if "—" in line else ("-" if " - " in line else None)
        if sep == "—":
            out.append(f"{indent}V — cycle view: head_camera ↔ gripper(s)")
        else:
            out.append(f"{indent}V                 cycle view: head_camera ↔ gripper(s)")
        saw_v = True
    if not saw_v:
        out.append(_VIEW_HELP_V)
    return out


def _ensure_view_help_lines(lines: list[str]) -> list[str]:
    """Normalize / append V camera help (head ↔ grippers; no top-down)."""
    return _normalize_view_help_lines(list(lines))


def print_banner(title: str, lines: list[str]):
    lines = list(lines)
    if any("Mode: robot" in line for line in lines):
        insert_at = 1 if lines else 0
        lines[insert_at:insert_at] = [
            "Arrows — move selected arm(s) in XY | E/Q — move in Z",
            "1 / 2 / 3 — select left / right / both arms",
        ]
    lines = _ensure_view_help_lines(lines)
    width = max(len(title), *(len(line) for line in lines), 40)
    bar = "=" * (width + 4)
    block = "\n".join([bar, f"  {title}", bar, *[f"  {line}" for line in lines], bar])
    print_instructions(block)


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


# Last interactive result for GUI exit codes (household_task_gui convention):
#   0 = SUCCESS, 10 = FAILURE, 2 = closed before a result.
_LAST_TASK_RESULT: bool | None = None
_LAST_TASK_DETAIL: str | None = None


def task_result_exit_code(ok: bool | None = None) -> int:
    """Map SUCCESS / FAILURE / no-result to launcher exit codes."""
    if ok is None:
        ok = _LAST_TASK_RESULT
    if ok is True:
        return 0
    if ok is False:
        return 10
    return 2


def _normalize_result_detail(detail: str | None) -> str | None:
    if detail is None:
        return None
    text = str(detail).strip()
    return text or None


def _persist_task_result(ok: bool | None, detail: str | None) -> None:
    """Write the latest result for a parent GUI launcher, if requested."""
    path = os.environ.get(TASK_RESULT_ENV)
    if not path:
        return
    try:
        Path(path).write_text(
            json.dumps({"ok": ok, "detail": detail or ""}, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        pass


def report_task_result(env, detail: str | None = None) -> bool:
    """Print ``Task complete: SUCCESS|FAILURE`` from ``check_success``; return success.

    Also stores the result for ``task_result_exit_code()`` so ``interactive_task_gui``
    can show SUCCESS/FAILURE like ``household_task_gui``. When ``ROBODYNA_TASK_RESULT_FILE``
    is set, persists ``ok`` + failure/success ``detail`` for the GUI status line.
    """
    global _LAST_TASK_RESULT, _LAST_TASK_DETAIL
    try:
        ok = bool(env.check_success())
    except Exception as exc:
        detail = _normalize_result_detail(f"check_success error: {exc}")
        print_failure(f"Task complete: FAILURE ({detail})")
        _LAST_TASK_RESULT = False
        _LAST_TASK_DETAIL = detail
        _persist_task_result(False, detail)
        return False
    detail = _normalize_result_detail(detail)
    if detail is None and not ok:
        detail = _normalize_result_detail(getattr(env, "_last_fail_reason", None))
    status = "SUCCESS" if ok else "FAILURE"
    msg = f"Task complete: {status}" + (f" ({detail})" if detail else "")
    if ok:
        print_success(msg)
    else:
        print_failure(msg)
    _LAST_TASK_RESULT = bool(ok)
    _LAST_TASK_DETAIL = detail
    _persist_task_result(_LAST_TASK_RESULT, detail)
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


def resolve_wrist_camera_link(env, side: str):
    """Return the robot wrist/camera link for ``left`` or ``right``, if any."""
    robot = getattr(env, "robot", None) if env is not None else None
    if robot is None or side not in ("left", "right"):
        return None
    return getattr(robot, f"{side}_camera", None)


def resolve_wrist_render_camera(env, side: str):
    """Return the sapien wrist render camera when ``collect_wrist_camera`` is on."""
    cams = getattr(env, "cameras", None) if env is not None else None
    if cams is None or not bool(getattr(cams, "collect_wrist_camera", False)):
        return None
    if side not in ("left", "right"):
        return None
    return getattr(cams, f"{side}_camera", None)


def active_gripper_sides(env) -> tuple[str, ...]:
    """Sides whose gripper/wrist views V may cycle through.

    Uses ``env._interactive_selected_arms`` when set (1/2/3 selection, or a
    task's single-arm default). Otherwise both sides that have a wrist camera
    link. Missing links are dropped so single-arm / no-wrist setups degrade.
    """
    available = []
    for side in ("right", "left"):
        if resolve_wrist_camera_link(env, side) is not None:
            available.append(side)
    if not available:
        return ()
    selected = tuple(getattr(env, "_interactive_selected_arms", ()) or ())
    selected = tuple(s for s in selected if s in ("left", "right"))
    if not selected:
        # Prefer right-then-left ordering for dual-arm cycling.
        return tuple(s for s in ("right", "left") if s in available)
    ordered = []
    for side in ("right", "left"):
        if side in selected and side in available:
            ordered.append(side)
    return tuple(ordered)


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


def gripper_width(env, side: str) -> float:
    """Normalized gripper opening in ``[0, 1]`` (1 = open, 0 = closed)."""
    robot = getattr(env, "robot", None)
    if robot is None:
        return 0.0
    if side == "left":
        return float(robot.get_left_gripper_val())
    return float(robot.get_right_gripper_val())


def toggle_selected_grippers(env, *, fallback=("left", "right"), threshold: float = 0.5) -> bool:
    """Edge-action helper: open ↔ close the highlighted gripper(s).

    Uses the current 1/2/3 selection when present; otherwise ``fallback``.
    Per selected arm: width ``> threshold`` → close (0), else open (1).
    Applies via ``robot.set_gripper`` (non-blocking) so teleop stays responsive.
    """
    robot = getattr(env, "robot", None)
    if robot is None or not hasattr(robot, "set_gripper"):
        print("No gripper available to open/close.")
        return False
    selected = tuple(getattr(env, "_interactive_selected_arms", ()) or ())
    if not selected:
        if bool(getattr(env, "_interactive_universal_controls", False)):
            print("Select an arm first: 1 (left), 2 (right), or 3 (both).")
            return False
        selected = tuple(fallback)
    actions = []
    for side in selected:
        if side not in ("left", "right"):
            continue
        width = gripper_width(env, side)
        target = 0.0 if width > float(threshold) else 1.0
        try:
            robot.set_gripper(target, side, gripper_eps=0.0)
        except Exception as exc:
            print(f"Gripper toggle failed ({side}): {exc}")
            continue
        actions.append(f"{side}={'open' if target > 0.5 else 'closed'}")
    if not actions:
        return False
    print("Gripper: " + ", ".join(actions))
    return True


class ViewerViewToggle:
    """V cycles head_camera ↔ gripper/wrist views (no top-down).

    G (edge) opens/closes the selected gripper(s) via ``toggle_selected_grippers``.
    F is kept as an alias for the same action. Camera switching is V-only.

    sapien's ``focus_camera`` follow-path is disabled in this build
    (``_handle_focused_camera`` commented out), so we copy the active camera
    pose onto the free-fly viewer each frame instead.
    """

    # Kept for callers / legacy ``overhead=`` framing helpers; not used by V.
    DEFAULT_TOPDOWN_XYZ = (_TOPDOWN_VIEW_X_OFFSET, 0.0, 1.68)
    DEFAULT_TOPDOWN_RPY = (0.0, -np.pi / 2.0, -np.pi / 2.0)
    DEFAULT_TOPDOWN_FOVY = float(np.deg2rad(65.0))
    # Fallback only: when available, the selected head camera's own fovy is used.
    DEFAULT_HEAD_FOVY = float(np.pi / 2.0)
    # D435-ish wrist fallback when collect_wrist_camera is off.
    DEFAULT_GRIPPER_FOVY = float(np.deg2rad(42.0))
    _GRIPPER_MODES = ("right_gripper", "left_gripper")

    def __init__(
        self,
        viewer,
        env=None,
        head_camera=None,
        topdown_xyz=None,
        topdown_rpy=None,
        topdown_fovy=None,
        capture_current_as_topdown: bool = False,
        warn_missing_head: bool = False,
        robot_controls=None,
    ):
        # topdown_* / capture_current_as_topdown kept for API compat; ignored.
        del topdown_xyz, topdown_rpy, topdown_fovy, capture_current_as_topdown
        self.viewer = viewer
        self.env = env
        self._prev_v = False
        self._prev_g = False
        self._prev_f = False
        self._head = head_camera
        self.robot_controls = robot_controls
        self._warned_missing_gripper = False
        if self._head is None:
            self._head = resolve_head_camera(env, viewer)
        if self._head is None and warn_missing_head:
            print("Warning: head_camera not found; V will cycle gripper views only.")

        self._disable_wasd_camera_move()
        modes = self._view_cycle_modes()
        self.mode = modes[0] if modes else "head"
        self.apply(announce=False)

    def _control_window(self):
        for plugin in getattr(self.viewer, "plugins", []) or []:
            if hasattr(plugin, "fps_camera_controller") and hasattr(plugin, "set_camera_xyz"):
                return plugin
        return getattr(self.viewer, "control_window", None)

    def _disable_wasd_camera_move(self):
        """Stop SAPIEN's FPS controller from translating the view on W/A/S/D.

        Interactive tasks own the camera (head_camera / gripper follow),
        so free-fly WASD only fights the fixed framing.
        """
        cw = self._control_window()
        if cw is not None and hasattr(cw, "move_speed"):
            try:
                cw.move_speed = 0.0
            except Exception:
                pass

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

    def _camera_fovy(self, camera, default: float) -> float:
        try:
            fovy = float(camera.fovy)
            if np.isfinite(fovy) and fovy > 0.0:
                return fovy
        except Exception:
            pass
        return float(default)

    def _head_fovy(self) -> float:
        """Use the render camera's lens so head view keeps its intended framing."""
        return self._camera_fovy(self._head, self.DEFAULT_HEAD_FOVY)

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

    def _view_cycle_modes(self) -> list[str]:
        """V targets: head_camera then each active gripper/wrist view."""
        modes: list[str] = []
        if self._head is not None:
            modes.append("head")
        for side in active_gripper_sides(self.env):
            modes.append(f"{side}_gripper")
        return modes

    def _gripper_side(self) -> str | None:
        if self.mode == "right_gripper":
            return "right"
        if self.mode == "left_gripper":
            return "left"
        return None

    def _wrist_pose(self, side: str):
        """Current wrist camera pose, syncing the render camera when available."""
        link = resolve_wrist_camera_link(self.env, side)
        if link is None:
            return None
        try:
            pose = link.get_pose() if hasattr(link, "get_pose") else link.global_pose
        except Exception:
            return None
        render_cam = resolve_wrist_render_camera(self.env, side)
        if render_cam is not None:
            try:
                render_cam.entity.set_pose(pose)
            except Exception:
                pass
            try:
                return render_cam.global_pose
            except Exception:
                return pose
        return pose

    def _gripper_viewer_pose(self, pose):
        """Keep wrist look direction; roll so screen axes match world teleop.

        Arrow keys are world-fixed (right=+X, up=+Y). Roll the gripper camera
        about its look (+X) so screen-up ≈ world +Y when possible.
        """
        import sapien
        from transforms3d.quaternions import mat2quat, quat2mat

        R = np.asarray(quat2mat(pose.q), dtype=np.float64)
        look = R[:, 0].copy()
        look_n = float(np.linalg.norm(look))
        if look_n < 1e-9:
            return pose
        look /= look_n

        # Prefer world +Y as screen-up (same as arrow-up).
        ref_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        up = ref_up - look * float(np.dot(ref_up, look))
        if float(np.linalg.norm(up)) < 1e-6:
            ref_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            up = ref_up - look * float(np.dot(ref_up, look))
        up /= float(np.linalg.norm(up)) + 1e-12
        # FPS convention: left = cross(up, forward).
        left = np.cross(up, look)
        left /= float(np.linalg.norm(left)) + 1e-12
        up = np.cross(look, left)
        up /= float(np.linalg.norm(up)) + 1e-12
        R_new = np.column_stack((look, left, up))
        return sapien.Pose(p=pose.p, q=mat2quat(R_new))

    def _gripper_fovy(self, side: str) -> float:
        render_cam = resolve_wrist_render_camera(self.env, side)
        if render_cam is not None:
            return self._camera_fovy(render_cam, self.DEFAULT_GRIPPER_FOVY)
        return self.DEFAULT_GRIPPER_FOVY

    def apply(self, announce=True):
        side = self._gripper_side()
        if side is not None:
            pose = self._wrist_pose(side)
            if pose is None:
                if announce:
                    print(f"View: {side} gripper unavailable; staying put.")
                return
            self._set_fovy(self._gripper_fovy(side))
            self._set_viewer_pose(self._gripper_viewer_pose(pose))
            if announce:
                print(f"View: {side} gripper")
            return
        if self.mode != "head" or self._head is None:
            # Recover to the first available mode (head preferred).
            modes = self._view_cycle_modes()
            if not modes:
                if announce:
                    print("No head/gripper camera available.")
                return
            self.mode = modes[0]
            if self.mode != "head":
                return self.apply(announce=announce)
        self._set_fovy(self._head_fovy())
        self._set_viewer_pose(self._head.global_pose)
        if announce:
            print("View: head_camera")

    def _edge_key(self, window, key: str, prev_attr: str) -> bool:
        down = bool(window.key_down(key))
        edge = down and not getattr(self, prev_attr)
        setattr(self, prev_attr, down)
        if edge:
            return True
        try:
            return bool(window.key_press(key))
        except Exception:
            return False

    def _v_pressed(self, window) -> bool:
        return self._edge_key(window, "v", "_prev_v")

    def _g_pressed(self, window) -> bool:
        return self._edge_key(window, "g", "_prev_g")

    def _f_pressed(self, window) -> bool:
        return self._edge_key(window, "f", "_prev_f")

    def _cycle_view(self):
        """Cycle head_camera ↔ active gripper/wrist view(s)."""
        modes = self._view_cycle_modes()
        if not modes:
            print("No head/gripper camera available.")
            return
        if self.mode not in modes:
            self.mode = modes[0]
            self.apply(announce=True)
            return
        if len(modes) == 1:
            label = "head_camera" if modes[0] == "head" else modes[0].replace("_", " ")
            print(f"Only one view available ({label}).")
            return
        idx = modes.index(self.mode)
        self.mode = modes[(idx + 1) % len(modes)]
        self.apply(announce=True)

    def update(self, window):
        if self.robot_controls is not None:
            self.robot_controls.update(window)
        elif self.env is not None:
            # Restore red failure tint when UniversalRobotControls is absent
            # (keyboard mode still uses Space / G action paths).
            gripper_failure_feedback(self.env).update()
        # G (F alias): open/close selected gripper(s). Independent of Space helpers.
        if self.env is not None and (self._g_pressed(window) or self._f_pressed(window)):
            toggle_selected_grippers(self.env)
        if self._v_pressed(window):
            self._cycle_view()
            return
        # Keep head / gripper views locked to the moving cameras.
        if self.mode == "head" and self._head is not None:
            self._set_viewer_pose(self._head.global_pose)
        else:
            side = self._gripper_side()
            if side is not None:
                pose = self._wrist_pose(side)
                if pose is not None:
                    self._set_viewer_pose(self._gripper_viewer_pose(pose))


def make_viewer_view_toggle(
    env,
    viewer=None,
    topdown_xyz=None,
    topdown_rpy=None,
    capture_current_as_topdown: bool = False,
    **kwargs,
) -> ViewerViewToggle:
    """Build V (head ↔ gripper) view switching for an interactive env.

    Always starts on ``head_camera`` when available. Legacy ``topdown_*`` /
    ``capture_current_as_topdown`` kwargs are accepted but ignored.
    """
    if viewer is None:
        viewer = getattr(env, "viewer", None)
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    robot_controls = None
    robot_mode = bool(getattr(env, "_interactive_robot_mode", False))
    control_from_argv = None
    for index, arg in enumerate(sys.argv):
        if arg == "--control" and index + 1 < len(sys.argv):
            control_from_argv = sys.argv[index + 1]
            break
        if arg.startswith("--control="):
            control_from_argv = arg.split("=", 1)[1]
            break
    if control_from_argv is not None:
        robot_mode = control_from_argv == "robot"
    elif not robot_mode:
        # Match add_robot_motion_arg default when --control is omitted.
        robot_mode = True
    if robot_mode:
        robot_controls = UniversalRobotControls(env)
    return ViewerViewToggle(
        viewer,
        env=env,
        head_camera=resolve_head_camera(env, viewer),
        topdown_xyz=topdown_xyz,
        topdown_rpy=topdown_rpy,
        capture_current_as_topdown=capture_current_as_topdown,
        robot_controls=robot_controls,
        **kwargs,
    )


class SeededArmIK:
    """Local seeded IK for one arm via SAPIEN's pinocchio CLIK solver.

    Curobo's ``solve_ik`` is unseeded: for a 2 mm nudge it happily returns an
    elbow/wrist flip several radians away, which is useless for incremental
    teleop or press animations. CLIK starts from the current configuration and
    costs ~0.05 ms, so it can run every physics step.
    """

    def __init__(self, env, side):
        robot = env.robot
        self.env = env
        self.side = side
        self.entity = robot.left_entity if side == "left" else robot.right_entity
        ee_joint = robot.left_ee if side == "left" else robot.right_ee
        arm_names = set(robot.left_arm_joints_name if side == "left"
                        else robot.right_arm_joints_name)
        self.model = self.entity.create_pinocchio_model()
        link_names = [link.get_name() for link in self.entity.get_links()]
        self.link_index = link_names.index(ee_joint.get_child_link().get_name())
        self.mask = np.zeros(self.entity.dof, dtype=np.int32)
        dofs = []
        cursor = 0
        for joint in self.entity.get_active_joints():
            width = joint.get_dof()
            if joint.get_name() in arm_names:
                self.mask[cursor:cursor + width] = 1
                dofs.extend(range(cursor, cursor + width))
            cursor += width
        self.arm_dofs = np.asarray(dofs, dtype=int)
        limits = np.asarray(self.entity.get_qlimits(), dtype=np.float64)
        self.lower = limits[self.arm_dofs, 0]
        self.upper = limits[self.arm_dofs, 1]

    def full_qpos(self):
        return np.asarray(self.entity.get_qpos(), dtype=np.float64)

    def solve(self, gripper_pose, seed=None):
        """Return ``(arm_joints, full_qpos)`` for a world gripper pose, or None."""
        world_target = self.env.robot._trans_from_gripper_to_endlink(
            list(np.asarray(gripper_pose, dtype=np.float64)), arm_tag=self.side)
        target = self.entity.get_root_pose().inv() * world_target
        seed = self.full_qpos() if seed is None else np.asarray(seed, dtype=np.float64)
        qpos, success, _ = self.model.compute_inverse_kinematics(
            self.link_index, target, initial_qpos=seed, active_qmask=self.mask,
            eps=1e-4, max_iterations=60, dt=0.2, damp=1e-3)
        if not success:
            return None
        full = np.asarray(qpos, dtype=np.float64)
        arm = np.clip(full[self.arm_dofs], self.lower, self.upper)
        full[self.arm_dofs] = arm
        return arm, full


def arm_ik(env, side):
    """Per-env cached ``SeededArmIK``; ``None`` when the model cannot be built."""
    cache = getattr(env, "_interactive_arm_ik", None)
    if cache is None:
        cache = {}
        env._interactive_arm_ik = cache
    if side not in cache:
        try:
            cache[side] = SeededArmIK(env, side)
        except Exception as exc:
            print(f"Arm IK unavailable ({exc}); robot motion is disabled.")
            cache[side] = None
    return cache[side]


class UniversalRobotControls:
    """Shared arm selection and Cartesian teleoperation for robot-mode tasks.

    Teleop integrates a commanded gripper pose and tracks it with seeded local
    IK (SAPIEN's pinocchio CLIK, ~0.05 ms per solve). Seeding on the previous
    solution keeps the arm in one kinematic branch; Curobo's ``solve_ik`` is
    unseeded and returns elbow/wrist flips that are unusable for teleop.

    Z / X tip the gripper about world +Y (left / right) for pour-style motions.
    G (open/close selected gripper; F alias) is handled by ``ViewerViewToggle``
    so it also works when teleop is not attached.
    """

    # Interactive teleop rates (m/s). 20% slower than the prior snappy sandbox
    # rates; MAX_LEAD / MAX_JOINT_SPEED scale with them or the caps choke motion.
    XY_SPEED = 0.288
    Z_SPEED = 0.224
    # World-Y tip rate for Z/X (rad/s) — enough to dump a board without feeling twitchy.
    ROLL_SPEED = 1.28
    MAX_DT = 0.05
    # How far the commanded pose may run ahead of the achieved pose, so a
    # blocked or joint-limited arm cannot accumulate an unrecoverable lead.
    MAX_LEAD = 0.044
    # Near a singularity a millimetre of Cartesian travel costs radians of
    # joint travel; slew at this cap instead of whipping the arm.
    MAX_JOINT_SPEED = 4.0

    def __init__(self, env):
        self.env = env
        initial = tuple(getattr(env, "_interactive_selected_arms", ()) or ())
        self.selected = initial or ("left",)
        self._previous = {key: False for key in ("1", "2", "3")}
        self._last_update = None
        self._command = {}
        self._highlight_materials = {}
        env._interactive_selected_arms = self.selected
        env._interactive_universal_controls = True
        env._interactive_robot_controls = self
        # Ensure the shared failure feedback exists for Space/action paths.
        gripper_failure_feedback(env)
        self._highlight_selected()

    def _highlight_selected(self):
        # A new selection cancels any failure tint so the highlight is readable.
        fb = getattr(self.env, "_interactive_gripper_failure", None)
        if isinstance(fb, GripperFailureFeedback) and fb._original:
            fb.restore()
        for material, color in self._highlight_materials.values():
            try:
                material.set_base_color(color)
                material.base_color = color
            except Exception:
                pass
        self._highlight_materials.clear()
        colors = {
            "left": [1.0, 0.85, 0.10, 1.0],
            "right": [0.15, 0.75, 1.0, 1.0],
        }
        for side in self.selected:
            articulation = (self.env.robot.left_entity if side == "left"
                            else self.env.robot.right_entity)
            for link in articulation.get_links():
                if link.get_name() not in GRIPPER_LINK_NAMES:
                    continue
                for component in link.entity.get_components():
                    try:
                        import sapien
                        if not isinstance(component, sapien.render.RenderBodyComponent):
                            continue
                    except Exception:
                        continue
                    for shape in component.render_shapes:
                        material = shape.material
                        if id(material) not in self._highlight_materials:
                            self._highlight_materials[id(material)] = (
                                material, list(material.base_color))
                        try:
                            material.set_base_color_texture(None)
                            material.set_base_color(colors[side])
                            material.base_color = colors[side]
                        except Exception:
                            pass

    def _edge(self, window, key):
        down = bool(window.key_down(key))
        edge = down and not self._previous[key]
        self._previous[key] = down
        return edge

    def _select(self, window):
        selected = None
        if self._edge(window, "1"):
            selected = ("left",)
        elif self._edge(window, "2"):
            selected = ("right",)
        elif self._edge(window, "3"):
            selected = ("left", "right")
        if selected is not None:
            self.selected = selected
            self.env._interactive_selected_arms = selected
            self._highlight_selected()
            print("Selected arm(s): " + " + ".join(selected))

    def _drive_qpos(self, side):
        joints = (self.env.robot.left_arm_joints if side == "left"
                  else self.env.robot.right_arm_joints)
        return np.asarray(
            [joint.get_drive_target()[0] for joint in joints], dtype=np.float64)

    def _ee_pose(self, side):
        getter = (self.env.robot.get_left_ee_pose if side == "left"
                  else self.env.robot.get_right_ee_pose)
        return np.asarray(getter(), dtype=np.float64)

    def _drive(self, side, step, dt, roll: float = 0.0):
        """Advance this arm's commanded pose and track it with seeded IK."""
        from transforms3d.quaternions import axangle2quat, qmult

        solver = arm_ik(self.env, side)
        if solver is None:
            return
        achieved = self._ee_pose(side)
        state = self._command.get(side)
        if state is None:
            # Anchor pose and joints on the measured state (not the drive
            # targets): the two differ by the tracking error, and mixing them
            # makes the first step look like a jump and get rate-limited away.
            measured = solver.full_qpos()
            state = {
                "pose": achieved.copy(),
                "seed": measured,
                "joints": measured[solver.arm_dofs],
            }
            self._command[side] = state

        pose = state["pose"].copy()
        prev_z = float(pose[2])
        pose[:3] += step
        # World-Y tip (Z/X): rotate the commanded gripper orientation in place.
        if abs(float(roll)) > 1e-9:
            dq = axangle2quat([0.0, 1.0, 0.0], float(roll))
            pose[3:7] = np.asarray(qmult(dq, pose[3:7]), dtype=np.float64)
        # Once a reactive key is pressed, block further -Z while over it so the
        # gripper cannot drive through the keycap and ruin the arm pose.
        bank = getattr(self.env, "_reactive_buttons", None)
        if bank is not None and hasattr(bank, "min_ee_z_over_pressed"):
            z_floor = bank.min_ee_z_over_pressed(pose[:2])
            if z_floor is not None and float(pose[2]) < float(z_floor):
                pose[2] = float(z_floor)
        # Billiard / tool-on-surface: if the held cue is already on the felt,
        # reject further -Z on that arm so it stops with the stick.
        if float(step[2]) < -1e-9:
            resting = getattr(self.env, "cue_resting_on_felt", None)
            cue_arm = str(getattr(self.env, "_cue_arm", "") or "")
            if cue_arm == side and callable(resting) and resting():
                pose[2] = prev_z
        floor_fn = getattr(self.env, "interactive_ee_z_floor", None)
        if callable(floor_fn):
            z_floor = floor_fn(side, pose)
            if z_floor is not None and float(pose[2]) < float(z_floor):
                pose[2] = float(z_floor)
        # Keep the command within reach of the achieved pose: a blocked or
        # joint-limited arm must not build up a lead it later snaps through.
        lead = pose[:3] - achieved[:3]
        distance = float(np.linalg.norm(lead))
        if distance > self.MAX_LEAD:
            pose[:3] = achieved[:3] + lead * (self.MAX_LEAD / distance)

        solution = solver.solve(pose, seed=state["seed"])
        if solution is None:
            return
        target, full = solution
        delta = target - state["joints"]
        budget = self.MAX_JOINT_SPEED * dt
        peak = float(np.max(np.abs(delta)))
        if peak > budget:
            delta *= budget / peak
            target = state["joints"] + delta
            full = full.copy()
            full[solver.arm_dofs] = target
        self.env.robot.set_arm_joints(target, delta / max(dt, 1e-3), side)
        state["pose"] = pose
        state["seed"] = full
        state["joints"] = target
        # Expose the commanded EE pose so spring/contact tasks can track the
        # teleop target even when the measured link lags the drive a bit.
        cmd = getattr(self.env, "_interactive_cmd_pose", None)
        if not isinstance(cmd, dict):
            cmd = {}
            self.env._interactive_cmd_pose = cmd
        cmd[side] = pose.copy()

    def _stop(self):
        """Zero the drive velocity targets, else the arms coast after release."""
        for side, state in self._command.items():
            joints = state["joints"]
            self.env.robot.set_arm_joints(joints, np.zeros_like(joints), side)
        self._command.clear()
        if isinstance(getattr(self.env, "_interactive_cmd_pose", None), dict):
            self.env._interactive_cmd_pose.clear()

    def update(self, window):
        self._select(window)
        gripper_failure_feedback(self.env).update()
        now = time.perf_counter()
        dt = 0.0 if self._last_update is None else min(now - self._last_update, self.MAX_DT)
        self._last_update = now
        if bool(getattr(self.env, "_interactive_teleop_locked", False)):
            self._stop()
            return
        x_dir = float(window.key_down("right")) - float(window.key_down("left"))
        y_dir = float(window.key_down("up")) - float(window.key_down("down"))
        z_dir = float(window.key_down("e")) - float(window.key_down("q"))
        # Z tip left / X tip right about world +Y (pour axis for board tasks).
        roll_dir = float(window.key_down("x")) - float(window.key_down("z"))
        if not (x_dir or y_dir or z_dir or roll_dir):
            self._stop()
            return
        if dt <= 0.0:
            return
        step = np.asarray([
            x_dir * self.XY_SPEED * dt,
            y_dir * self.XY_SPEED * dt,
            z_dir * self.Z_SPEED * dt,
        ], dtype=np.float64)
        roll = float(roll_dir) * self.ROLL_SPEED * dt
        for side in self.selected:
            self._drive(side, step, dt, roll=roll)


# Keep stepping/rendering this long after a terminal SUCCESS/FAILURE so the
# result is visible before the viewer closes (wall-clock, not sim time).
TERMINAL_RESULT_HOLD_SECONDS = 2.0


def sleep_to_timestep(env, frame_start: float) -> None:
    """Sleep the remainder of one physics timestep after a frame's work."""
    remaining = float(env.scene.get_timestep()) - (time.perf_counter() - frame_start)
    if remaining > 0:
        time.sleep(remaining)


def terminal_hold_should_close(terminal_started_at: float | None) -> bool:
    """True once the post-result display hold has finished (wall-clock)."""
    if terminal_started_at is None:
        return False
    return time.perf_counter() - terminal_started_at >= TERMINAL_RESULT_HOLD_SECONDS


def run_viewer_loop(env, on_step, should_stop=None, max_steps: int | None = None,
                    overhead: bool = True, is_done=None):
    """Standard interactive loop: callback → kinematics → step → render.

    ``is_done(step)`` may return ``True`` / ``False``, or ``(done, detail)``.
    When done, prints SUCCESS/FAILURE via ``report_task_result``, then continues
    stepping/rendering for ``TERMINAL_RESULT_HOLD_SECONDS`` wall-clock before
    closing. Returns that bool (or ``None`` if the viewer closed without a
    result). ``should_stop`` remains a raw break (no auto print / no hold) for
    backward compatibility.
    Starts on head_camera; press V to cycle head ↔ gripper/wrist view(s).
    ``overhead`` is accepted for API compat but ignored (top-down removed).
    """
    global _LAST_TASK_RESULT, _LAST_TASK_DETAIL, _LAST_EPISODE_CONDITION
    _LAST_TASK_RESULT = None
    _LAST_TASK_DETAIL = None
    _LAST_EPISODE_CONDITION = None
    del overhead  # legacy kwarg; interactive views no longer use top-down
    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    views = make_viewer_view_toggle(env, viewer)
    step = 0
    terminal_started_at = None
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
            # SAPIEN does not close its window on Escape consistently, so make
            # it an explicit launcher-level exit for every shared task loop.
            if viewer.window.key_down("escape"):
                break
            step += 1
            if terminal_started_at is not None:
                if terminal_hold_should_close(terminal_started_at):
                    break
                sleep_to_timestep(env, frame_start)
                continue
            if is_done is not None:
                result = is_done(step)
                if isinstance(result, tuple):
                    done = bool(result[0])
                    detail = result[1] if len(result) > 1 else None
                else:
                    done, detail = bool(result), None
                if done:
                    report_task_result(env, detail)
                    terminal_started_at = time.perf_counter()
                    sleep_to_timestep(env, frame_start)
                    continue
            if should_stop is not None and should_stop(step):
                break
            if max_steps is not None and step >= max_steps:
                print(f"Reached max_steps={max_steps}; evaluating.")
                report_task_result(env, f"max_steps={max_steps}")
                terminal_started_at = time.perf_counter()
                sleep_to_timestep(env, frame_start)
                continue
            sleep_to_timestep(env, frame_start)
    finally:
        env.close_env()
    return _LAST_TASK_RESULT


# ---------------------------------------------------------------------------
# Shared robot key/button press (same routine as interactive_sort_apples_belt)
# ---------------------------------------------------------------------------


def add_robot_motion_arg(parser, robot_motion_default: str = "planner"):
    """Add ``--control`` + ``--robot-motion`` flags used by every interactive script."""
    parser.add_argument(
        "--control",
        choices=("keyboard", "robot"),
        default="robot",
        help="Interaction method (default: robot)",
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


def _line_documents_key(lines: list[str], key: str) -> bool:
    """True when a control banner line already documents ``key`` as a binding."""
    key = key.strip().upper()
    for ln in lines:
        s = ln.strip()
        if s.startswith(f"{key} ") or s.startswith(f"{key}:") or s.startswith(f"{key}\t"):
            return True
        if f"{key}                 " in ln or f"{key}: " in ln:
            return True
        # Compact banners: "V: camera | G: open/close | Escape"
        if f"{key}:" in s or f"| {key}:" in s or f"|{key}:" in s:
            return True
    return False


def print_mode_controls(task_name: str, mode: str, *, keyboard: str, robot: str) -> None:
    """Print only the help block for the selected ``--control`` mode."""
    body = (robot if mode == "robot" else keyboard).strip("\n")
    if mode == "robot":
        # Shared teleop keys; skip G here when the task banner already lists it.
        shared = (
            "  Arrow keys        move selected arm(s) in world XY\n"
            "  E / Q             raise / lower selected arm(s)\n"
            "  Z / X             tip gripper left / right (world Y)\n"
            "  1 / 2 / 3         select left / right / both arms\n"
        )
        if not _line_documents_key(body.splitlines(), "G"):
            shared += "  G                 open / close selected gripper(s)\n"
        body = shared + body
    lines = _normalize_view_help_lines(body.splitlines())
    # Rewrite stale F gripper-toggle help to G; inject G when missing.
    rewritten = []
    for ln in lines:
        s = ln.strip()
        if (
            (s.startswith("F ") or s.startswith("F:") or s.startswith("F\t"))
            and "open" in ln.lower()
            and "close" in ln.lower()
            and "gripper" in ln.lower()
        ):
            indent = ln[: len(ln) - len(ln.lstrip(" "))]
            rewritten.append(f"{indent}G                 open / close selected gripper(s)")
        else:
            rewritten.append(ln)
    lines = rewritten
    if not _line_documents_key(lines, "G"):
        inserted = False
        for i, ln in enumerate(lines):
            if (
                "cycle view" in ln.lower()
                or "toggle view" in ln.lower()
                or ln.strip().startswith("V ")
                or ln.strip().startswith("V:")
                or "V                 " in ln
            ):
                indent = ln[: len(ln) - len(ln.lstrip(" "))]
                lines.insert(
                    i + 1,
                    f"{indent}G                 open / close selected gripper(s)",
                )
                inserted = True
                break
        if not inserted:
            lines.append("  G                 open / close selected gripper(s)")
    body = "\n".join(lines)
    bar = "=" * 60
    print_instructions(f"{bar}\n {task_name} — {mode} controls\n{bar}\n{body}\n{bar}")


def default_arms_for_mode(mode):
    """``left`` / ``right`` / ``dump`` → arm side names (sort_apples style)."""
    if mode == "dump":
        return ("left", "right")
    return (mode,) if mode else ()


def selected_robot_arms(env, fallback=("left",)):
    """Return the arms selected by the universal 1/2/3 robot controls."""

    selected = tuple(getattr(env, "_interactive_selected_arms", ()) or ())
    return selected or tuple(fallback)


GRIPPER_LINK_NAMES = frozenset({
    "wsg_50_base_link", "gripper_left", "gripper_right",
    "finger_left", "finger_right",
})
GRIPPER_FAILURE_RED = [0.92, 0.04, 0.03, 1.0]
GRIPPER_FAILURE_SECONDS = 2.0


def require_selected_arms(env, *, exactly_one: bool = False):
    """Return currently highlighted arms, or ``()`` when the action must not run.

    Unselected grippers never act. When ``exactly_one`` is set, both-arms (3)
    is also rejected so the caller can require a single highlighted gripper;
    both selected grippers flash red as the error cue.
    """
    selected = tuple(getattr(env, "_interactive_selected_arms", ()) or ())
    if not selected:
        print("Select an arm first: 1 (left) or 2 (right)"
              + ("" if exactly_one else " [or 3 for both]") + ".")
        return ()
    if exactly_one and len(selected) != 1:
        action_failed(
            env,
            selected,
            detail="select exactly one arm (1 left / 2 right)",
        )
        return ()
    return selected


def resolve_action_arm(env, arm_tag_cls, *, exactly_one: bool = True):
    """Return an ``ArmTag`` for the highlighted gripper, or ``None`` to abort."""
    selected = require_selected_arms(env, exactly_one=exactly_one)
    if not selected:
        return None
    return arm_tag_cls(selected[0])


class GripperFailureFeedback:
    """Tint failed gripper mesh(es) red for a short, non-acting feedback window."""

    def __init__(self, env):
        self.env = env
        self._until = 0.0
        self._original = {}

    def active(self) -> bool:
        return bool(self._original) and time.perf_counter() < self._until

    def restore(self):
        for material, color in self._original.values():
            try:
                material.set_base_color(color)
                material.base_color = color
            except Exception:
                pass
        self._original.clear()
        self._until = 0.0

    def _tint_side(self, side: str):
        robot = getattr(self.env, "robot", None)
        if robot is None:
            return
        articulation = getattr(robot, f"{side}_entity", None)
        if articulation is None:
            return
        try:
            import sapien
        except Exception:
            return
        for link in articulation.get_links():
            if link.get_name() not in GRIPPER_LINK_NAMES:
                continue
            for component in link.entity.get_components():
                if not isinstance(component, sapien.render.RenderBodyComponent):
                    continue
                for shape in component.render_shapes:
                    material = shape.material
                    key = id(material)
                    if key not in self._original:
                        self._original[key] = (material, list(material.base_color))
                    try:
                        material.set_base_color_texture(None)
                        material.set_base_color(GRIPPER_FAILURE_RED)
                        material.base_color = GRIPPER_FAILURE_RED
                    except Exception:
                        pass

    def flash(self, arms, message: str | None = None):
        """Show red for ``GRIPPER_FAILURE_SECONDS``; do not perform any motion."""
        sides = tuple(str(a) for a in (arms or ()) if str(a) in ("left", "right"))
        if not sides:
            return
        self.restore()
        for side in sides:
            self._tint_side(side)
        self._until = time.perf_counter() + GRIPPER_FAILURE_SECONDS
        if message:
            print(message)

    def update(self):
        if self._original and time.perf_counter() >= self._until:
            self.restore()
            controls = getattr(self.env, "_interactive_robot_controls", None)
            if controls is not None and hasattr(controls, "_highlight_selected"):
                try:
                    controls._highlight_selected()
                except Exception:
                    pass


def gripper_failure_feedback(env) -> GripperFailureFeedback:
    fb = getattr(env, "_interactive_gripper_failure", None)
    if not isinstance(fb, GripperFailureFeedback):
        fb = GripperFailureFeedback(env)
        env._interactive_gripper_failure = fb
    return fb


def flash_gripper_failure(env, arms, message: str | None = None):
    """Shared entry point: red gripper flash when a selected-arm action fails."""
    gripper_failure_feedback(env).flash(arms, message=message)


def action_failed(env, arms, detail: str = "action failed") -> bool:
    """Flash red, clear the plan-failure latch, return False for callers."""
    sides = tuple(str(a) for a in (arms or ()) if str(a))
    label = "+".join(sides) if sides else "arm"
    flash_gripper_failure(env, sides, f"{label} {detail}; no motion applied")
    try:
        env.plan_success = True
        env._last_plan_fail = None
    except Exception:
        pass
    return False


def try_interactive_grasp(env, actor, arm_tag, *, detail: str = "unreachable / grasp failed", **grasp_kwargs) -> bool:
    """Run ``grasp_actor`` + ``move`` without crashing on an unreachable arm.

    Returns True on success. On failure (no pose, plan fail, or legacy
    ``target_pose is None`` assert), flashes the arm red and returns False.
    """
    side = str(arm_tag)
    try:
        env.plan_success = True
        env.move(env.grasp_actor(actor, arm_tag=arm_tag, **grasp_kwargs))
    except AssertionError:
        env.plan_success = False
    if not env.plan_success:
        return action_failed(env, (side,), detail=detail)
    return True


class RobotButtonController:
    """Hold-to-actuate (or tap) key press: grasp → TCP-limited press → latch → clear lift.

    Matches ``interactive_sort_apples_belt.RobotButtonController``. Task scripts supply
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
            sides = tuple(self.arms_for_mode(self.mode or requested_mode))
            action_failed(
                self.env, sides,
                detail=f"button trajectory failed ({detail or 'unknown'})",
            )
        if self.hold and requested_mode == self.mode:
            if requested_mode is not None:
                self.set_latch(self.env, requested_mode)
            return
        self.clear_latch(self.env)
        if self.mode is not None:
            self._lift_from_buttons(self.mode)
        if requested_mode is not None:
            sides = tuple(self.arms_for_mode(requested_mode))
            if not sides:
                # No highlighted arm maps to this action — do nothing.
                requested_mode = None
            else:
                self._move_to_buttons(requested_mode)
                if not self.hold:
                    # Edge tap: press then release in one update.
                    self.clear_latch(self.env)
                    self._lift_from_buttons(requested_mode)
                    requested_mode = None
        if not self.env.plan_success:
            detail = getattr(self.env, "_last_plan_fail", None)
            sides = tuple(self.arms_for_mode(requested_mode or self.mode))
            action_failed(
                self.env, sides,
                detail=f"button motion failed ({detail or 'unknown'})",
            )
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
