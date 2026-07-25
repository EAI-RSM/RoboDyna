#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive viewer for ``stop_valley_ball``.

Run from any directory:

    /path/to/RoboDynaExp/script_exp/interactive_stop_valley_ball.py --control keyboard
    /path/to/RoboDynaExp/script_exp/interactive_stop_valley_ball.py --control robot

Mixed keyboard + mouse: left-click sets the bat XY on the mid-air intercept
plane; Space freezes / arms the bat. Robot mode grasps on Space, then holds
the bat at the click location (gripper stays closed).
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

from _interactive_common import make_viewer_view_toggle, report_task_result  # noqa: E402


CONTROLS = """
============================================================
 stop_valley_ball — interactive controls
============================================================
  Mouse left-click  set bat XY on the mid-air intercept plane
  Space             keyboard: freeze / arm bat at current pose
                    robot: pick up bat (grasp); hold uses click
  Arrow keys        fine nudge XY (optional)
  [ / ]             lower / raise bat height
  T                 snap bat to predicted intercept
  V                 toggle view: top-down ↔ head_camera
  Q / Escape         quit
------------------------------------------------------------
  Flow (robot): Space to pick up → click mid-air to hold
  Flow (keyboard): click to aim → Space to arm / hold
  Success: red ball hits the red circular bat head before
           falling to the table; handle contact does not count.
  --control keyboard  direct bat teleop (default)
  --control robot     arm grasps / holds the bat
  --robot-motion planner|interpolate
============================================================
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
        task_name="stop_valley_ball",
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


def _intercept_xyz(env):
    if getattr(env, "intercept", None) is None:
        env._compute_intercept()
    return np.asarray(env.intercept, dtype=float)


def _hold_plane_z(env):
    """Horizontal plane used for click aiming (predicted intercept height)."""
    return float(_intercept_xyz(env)[2])


def _get_rigid(actor):
    import sapien
    for comp in actor.actor.get_components():
        if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent):
            return comp
    return None


def _bat_face_quat(env):
    if hasattr(env, "_bat_face_quat"):
        return list(env._bat_face_quat())
    return [1.0, 0.0, 0.0, 0.0]


def _clamp_bat_xyz(env, x, y, z):
    """Keep the bat mid-air past the red line and within a reachable band."""
    min_clear = float(getattr(env, "INTERCEPT_MIN_CLEARANCE_DEFAULT", 0.06))
    min_z = float(env.table_top + min_clear + 0.5 * float(env.panel_radius))
    max_z = float(env.table_top + 0.28)
    z = float(np.clip(z, min_z, max_z))
    y = float(np.clip(y, -0.45, 0.30))
    # Stay on the exit side of the red line (same spirit as catch_valley place).
    side = float(getattr(env, "side", 1.0))
    red = float(env.red_line_x)
    if side > 0.0:
        x = float(np.clip(x, red + 0.02, 0.38))
    else:
        x = float(np.clip(x, -0.38, red - 0.02))
    return x, y, z


def _set_bat_xyz(env, x, y, z=None):
    import sapien
    pose = env.panel.get_pose()
    if z is None:
        z = float(pose.p[2])
    x, y, z = _clamp_bat_xyz(env, x, y, z)
    new_pose = sapien.Pose([float(x), float(y), float(z)], _bat_face_quat(env))
    try:
        env.panel.set_pose(new_pose)
    except Exception:
        env.panel.actor.set_pose(new_pose)
    # Keep aliases in sync (weld helpers use self.bowl).
    env.bowl = env.panel
    rigid = _get_rigid(env.panel)
    if rigid is not None:
        try:
            rigid.set_disable_gravity(True)
            rigid.set_kinematic(True)
            rigid.set_linear_velocity(np.zeros(3))
            rigid.set_angular_velocity(np.zeros(3))
            rigid.set_kinematic_target(new_pose)
        except Exception:
            pass
    return x, y, z


def _nudge_from_keys(window, step=0.008):
    dx = dy = dz = 0.0
    if window.key_down("left"):
        dx -= step
    if window.key_down("right"):
        dx += step
    if window.key_down("up"):
        dy += step
    if window.key_down("down"):
        dy -= step
    if window.key_down("["):
        dz -= step
    if window.key_down("]"):
        dz += step
    return dx, dy, dz


def _ray_hit_plane_xy(origin, direction, plane_z):
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        return None
    direction = direction / norm
    if abs(direction[2]) < 1e-12:
        return None
    t = (float(plane_z) - float(origin[2])) / float(direction[2])
    if t < 0.0:
        return None
    hit = origin + t * direction
    return float(hit[0]), float(hit[1])


def _click_to_hold_xy(viewer, pixel_x, pixel_y, plane_z):
    """Map a viewer click to XY on the mid-air hold plane (``z=plane_z``)."""
    window = viewer.window
    px = int(pixel_x)
    py = int(pixel_y)

    try:
        model = np.asarray(window.get_camera_model_matrix(), dtype=np.float64)
    except Exception:
        return None
    origin = model[:3, 3]
    rot = model[:3, :3]

    try:
        pos = np.asarray(window.get_picture_pixel("Position", px, py), dtype=np.float64)
        if pos.shape[0] >= 3 and np.all(np.isfinite(pos[:3])):
            depth_ok = True
            if pos.shape[0] >= 4:
                depth_ok = float(pos[3]) < 0.999
            if depth_ok and float(np.linalg.norm(pos[:3])) > 1e-6:
                world = rot @ pos[:3] + origin
                hit = _ray_hit_plane_xy(origin, world - origin, plane_z)
                if hit is not None:
                    return hit
    except Exception:
        pass

    try:
        tw, th = window.get_picture_size("Color")
    except Exception:
        try:
            tw, th = window.get_picture_size("Segmentation")
        except Exception:
            tw, th = window.size
    if tw <= 0 or th <= 0:
        return None

    try:
        sw, sh = window.get_picture_size("Segmentation")
        if sw > 0 and sh > 0 and (sw != tw or sh != th):
            px = int(px * tw / sw)
            py = int(py * th / sh)
    except Exception:
        pass

    ndc_x = (float(px) + 0.5) / float(tw) * 2.0 - 1.0
    ndc_y = 1.0 - (float(py) + 0.5) / float(th) * 2.0

    if getattr(window, "camera_mode", "perspective") == "orthographic":
        top = float(getattr(window, "ortho_top", 1.0))
        aspect = float(tw) / float(th)
        right = top * aspect
        origin_o = origin + rot @ np.array(
            [ndc_x * right, ndc_y * top, 0.0], dtype=np.float64
        )
        direction = rot @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
        return _ray_hit_plane_xy(origin_o, direction, plane_z)

    fovy = float(window.fovy)
    aspect = float(tw) / float(th)
    tan_y = float(np.tan(0.5 * fovy))
    tan_x = tan_y * aspect
    dir_cam = np.array([ndc_x * tan_x, ndc_y * tan_y, -1.0], dtype=np.float64)
    return _ray_hit_plane_xy(origin, rot @ dir_cam, plane_z)


class EdgeKey:
    def __init__(self):
        self._prev = False

    def poll(self, down):
        edge = bool(down) and not self._prev
        self._prev = bool(down)
        return edge


class KeyboardBatController:
    def __init__(self, env):
        self.env = env
        self.ready = False
        self._space = EdgeKey()
        self._snap = EdgeKey()
        # Lift off the holder to the intercept plane so the first click is mid-air.
        ix, iy, iz = _intercept_xyz(env)
        _set_bat_xyz(env, ix, iy, iz)

    def on_hold_click(self, x, y):
        if self.ready:
            return False
        z = _hold_plane_z(self.env)
        x, y, z = _set_bat_xyz(self.env, x, y, z)
        print(f"Bat aimed at ({x:.3f}, {y:.3f}, {z:.3f}). Press Space to arm.")
        return True

    def update(self, window):
        if self._snap.poll(window.key_down("t")):
            ix, iy, iz = _intercept_xyz(self.env)
            x, y, z = _set_bat_xyz(self.env, ix, iy, iz)
            print(f"Snapped bat to intercept ({x:.3f}, {y:.3f}, {z:.3f}).")
        if not self.ready:
            dx, dy, dz = _nudge_from_keys(window)
            if dx or dy or dz:
                p = np.asarray(self.env.panel.get_pose().p, dtype=float)
                _set_bat_xyz(self.env, p[0] + dx, p[1] + dy, p[2] + dz)
        if self._space.poll(window.key_down("space")):
            p = np.asarray(self.env.panel.get_pose().p, dtype=float)
            x, y, z = _set_bat_xyz(self.env, p[0], p[1], p[2])
            self.env._bowl_ready = True
            self.ready = True
            print(f"Bat armed mid-air at ({x:.3f}, {y:.3f}, {z:.3f}).")


class RobotBatController:
    def __init__(self, env, ArmTag):
        self.env = env
        self.ArmTag = ArmTag
        self.arm = None
        self.holding = False
        self.ready = False
        self.busy = False
        self._pending_xy = None
        self._space = EdgeKey()
        self._snap = EdgeKey()

    def _choose_arm(self):
        return self.ArmTag("left" if self.env.mirrored else "right")

    def grasp(self):
        self.busy = True
        self.arm = self._choose_arm()
        self.env.move(self.env.grasp_actor(self.env.panel, arm_tag=self.arm, pre_grasp_dis=0.10))
        if self.env.plan_success:
            self.env._weld_bowl_to_end_effector(self.arm)
            # Lift clear of the holder before any lateral move.
            panel_now = np.asarray(self.env.panel.get_pose().p, dtype=float)
            lift_z = float(max(0.10, _hold_plane_z(self.env) - panel_now[2]))
            self.env.move(self.env.move_by_displacement(
                self.arm, z=lift_z, move_axis="world",
            ))
            self.holding = True
            print(f"Picked up bat with {self.arm} arm. Left-click to hold mid-air.")
            if self._pending_xy is not None:
                x, y = self._pending_xy
                self._pending_xy = None
                self.busy = False
                self.hold_at(x, y)
                return
        else:
            print("Grasp failed; planner disabled further robot actions.")
        self.busy = False

    def hold_at(self, x, y, z=None):
        if self.ready:
            return
        if not self.holding or self.arm is None:
            self._pending_xy = (float(x), float(y))
            print(
                f"Hold target ({x:.3f}, {y:.3f}) saved — press Space to pick up, "
                "then it will move there."
            )
            return
        self.busy = True
        if z is None:
            z = _hold_plane_z(self.env)
        x, y, z = _clamp_bat_xyz(self.env, x, y, z)
        panel_now = np.asarray(self.env.panel.get_pose().p, dtype=float)
        d = np.array([x, y, z], dtype=float) - panel_now
        self.env.move(self.env.move_by_displacement(
            arm_tag=self.arm,
            x=float(d[0]),
            y=float(d[1]),
            z=float(d[2]),
            move_axis="world",
        ))
        # Keep gripper closed / welded — mid-air stop requires a held bat.
        self.env._bowl_ready = True
        self.ready = True
        print(f"Holding bat mid-air at ({x:.3f}, {y:.3f}, {z:.3f}).")
        self.busy = False

    def on_hold_click(self, x, y):
        if self.busy or self.ready:
            return False
        self.hold_at(x, y)
        return True

    def nudge(self, window):
        if self.busy or not self.holding or self.arm is None or self.ready:
            return
        dx, dy, dz = _nudge_from_keys(window, step=0.02)
        if not (dx or dy or dz):
            return
        self.busy = True
        self.env.move(self.env.move_by_displacement(
            arm_tag=self.arm, x=dx, y=dy, z=dz, move_axis="world",
        ))
        self.busy = False

    def update(self, window):
        if self.busy:
            return
        if self._snap.poll(window.key_down("t")) and self.holding and not self.ready:
            ix, iy, iz = _intercept_xyz(self.env)
            self.hold_at(ix, iy, iz)
            return
        if self._space.poll(window.key_down("space")):
            if not self.holding and not self.ready:
                self.grasp()
            return
        self.nudge(window)


def main():
    parser = argparse.ArgumentParser(description="Interactive stop_valley_ball viewer")
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
    parser.add_argument(
        "--task-arg",
        action="append",
        default=[],
        help="Override task_args.stop_valley_ball entry, e.g. --task-arg wall_bounce_enabled=true",
    )
    args = parser.parse_args()

    from envs import CONFIGS_PATH
    from envs.stop_valley_ball import stop_valley_ball
    from envs.utils.action import ArmTag
    globals()["CONFIGS_PATH"] = CONFIGS_PATH

    print(CONTROLS)
    if args.robot_motion == "interpolate":
        print(
            "Note: --robot-motion interpolate uses planner motions for this teleop task "
            "(key-press sandboxes use joint interpolation)."
        )

    config = _configure_task(args.config, args.seed, use_robot=args.control == "robot")
    # Optional option toggles (same as record_demo --task-arg).
    targs = config.setdefault("task_args", {}).setdefault("stop_valley_ball", {})
    for item in args.task_arg:
        if "=" not in item:
            raise SystemExit(f"--task-arg expects key=value, got: {item}")
        key, raw = item.split("=", 1)
        key = key.strip()
        raw = raw.strip()
        if raw.lower() in ("true", "false"):
            targs[key] = raw.lower() == "true"
        else:
            try:
                targs[key] = int(raw) if raw.isdigit() else float(raw)
            except ValueError:
                targs[key] = raw

    env = stop_valley_ball()
    env.setup_demo(**config)
    # setup_demo already starts ball motion with expert_demo=False.
    ix, iy, iz = _intercept_xyz(env)
    print(
        f"Predicted intercept ≈ ({ix:.3f}, {iy:.3f}, {iz:.3f}); "
        f"red_line_x={env.red_line_x:.3f}; mirrored={env.mirrored}."
    )

    controller = (
        RobotBatController(env, ArmTag) if args.control == "robot" else KeyboardBatController(env)
    )

    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    views = make_viewer_view_toggle(env, viewer)

    def _handle_hold_click(viewer_, pixel_x, pixel_y):
        plane_z = _hold_plane_z(env)
        xy = _click_to_hold_xy(viewer_, pixel_x, pixel_y, plane_z)
        if xy is None:
            print("Click did not hit the mid-air hold plane.")
            return False
        return bool(controller.on_hold_click(xy[0], xy[1]))

    viewer.register_click_handler(_handle_hold_click)
    print("Left-click to set the bat hold location; Space picks up / arms.")

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

            if getattr(env, "_ball_phase", None) == "released":
                if settle_after is None:
                    settle_after = time.perf_counter()
                    print("Ball left the valley exit; waiting to settle…")
                elif time.perf_counter() - settle_after >= 2.5:
                    report_task_result(env)
                    break
            # Early fail if the ball hit the table before a head contact.
            if getattr(env, "_ball_table_before_hit", False) and not getattr(env, "_panel_hit", False):
                report_task_result(env)
                break

            remaining = float(env.scene.get_timestep()) - (time.perf_counter() - frame_start)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        env.close_env()


if __name__ == "__main__":
    main()
