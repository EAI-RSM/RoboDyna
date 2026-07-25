#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive viewer for ``catch_valley_ball``.

Run from any directory:

    /path/to/RoboDynaExp/script_exp/interactive_catch_valley_ball.py --control keyboard
    /path/to/RoboDynaExp/script_exp/interactive_catch_valley_ball.py --control robot

Mixed keyboard + mouse: Space picks up / freezes the bowl; left-click sets the
table XY. Robot mode grasps on Space, then places at the click location.
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
 catch_valley_ball — interactive controls
============================================================
  Mouse left-click  set bowl XY on the table
  Space             keyboard: freeze/place bowl at current XY
                    robot: pick up bowl (grasp); place uses click
  Arrow keys        fine nudge (optional)
  V                 toggle view: top-down ↔ head_camera
  Q / Escape         quit
------------------------------------------------------------
  Flow (robot): Space to pick up → click table to place
  Flow (keyboard): click to aim → Space to freeze/place
  Place snaps past the red line (success requires that).
  Success: red ball in bowl, bowl behind red line
  --control keyboard  direct bowl teleop (default)
  --control robot     arm grasps / places the bowl
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
        task_name="catch_valley_ball",
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


def _target_xy(env):
    landing = np.asarray(env.landing, dtype=float)
    return float(env._catch_target_x(landing[0])), float(landing[1])


def _bowl_place_z(env):
    return float(env.table_top - 0.020)  # release ~2 cm higher than prior place height


def _get_rigid(actor):
    import sapien
    for comp in actor.actor.get_components():
        if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent):
            return comp
    return None


def _set_bowl_xy(env, x, y, z=None):
    import sapien
    pose = env.bowl.get_pose()
    if z is None:
        z = _bowl_place_z(env)
    new_pose = sapien.Pose([float(x), float(y), float(z)], pose.q)
    env.bowl.set_pose(new_pose)
    rigid = _get_rigid(env.bowl)
    if rigid is not None:
        try:
            rigid.set_disable_gravity(True)
            rigid.set_kinematic(True)
            rigid.set_linear_velocity(np.zeros(3))
            rigid.set_angular_velocity(np.zeros(3))
            rigid.set_kinematic_target(new_pose)
        except Exception:
            pass


def _nudge_from_keys(window, step=0.008):
    dx = dy = 0.0
    if window.key_down("left"):
        dx -= step
    if window.key_down("right"):
        dx += step
    if window.key_down("up"):
        dy += step
    if window.key_down("down"):
        dy -= step
    return dx, dy


def _clamp_table_xy(env, x, y):
    """Keep the bowl on a usable patch near the valley exit / red line."""
    # Snap X past the red line (success requires this); keep Y on the table.
    x = float(env._catch_target_x(x))
    y = float(np.clip(y, -0.50, 0.25))
    return x, y


def _ray_hit_table_xy(origin, direction, table_z):
    """Intersect a world-space ray with the horizontal table plane."""
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        return None
    direction = direction / norm
    if abs(direction[2]) < 1e-12:
        return None
    t = (float(table_z) - float(origin[2])) / float(direction[2])
    if t < 0.0:
        return None
    hit = origin + t * direction
    return float(hit[0]), float(hit[1])


def _click_to_table_xy(viewer, pixel_x, pixel_y, table_z):
    """Map a viewer click to XY on the table surface (``z=table_z``)."""
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
                if abs(float(world[2]) - float(table_z)) <= 0.04:
                    return float(world[0]), float(world[1])
                hit = _ray_hit_table_xy(origin, world - origin, table_z)
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
        return _ray_hit_table_xy(origin_o, direction, table_z)

    fovy = float(window.fovy)
    aspect = float(tw) / float(th)
    tan_y = float(np.tan(0.5 * fovy))
    tan_x = tan_y * aspect
    dir_cam = np.array([ndc_x * tan_x, ndc_y * tan_y, -1.0], dtype=np.float64)
    return _ray_hit_table_xy(origin, rot @ dir_cam, table_z)


class EdgeKey:
    def __init__(self):
        self._prev = False

    def poll(self, down):
        edge = bool(down) and not self._prev
        self._prev = bool(down)
        return edge


class KeyboardBowlController:
    def __init__(self, env):
        self.env = env
        self.placed = False
        self._space = EdgeKey()

    def on_table_click(self, x, y):
        if self.placed:
            return False
        x, y = _clamp_table_xy(self.env, x, y)
        _set_bowl_xy(self.env, x, y)
        print(f"Bowl aimed at click ({x:.3f}, {y:.3f}). Press Space to place.")
        return True

    def update(self, window):
        if not self.placed:
            dx, dy = _nudge_from_keys(window)
            if dx or dy:
                p = np.asarray(self.env.bowl.get_pose().p, dtype=float)
                x, y = _clamp_table_xy(self.env, p[0] + dx, p[1] + dy)
                _set_bowl_xy(self.env, x, y)
        if self._space.poll(window.key_down("space")):
            p = np.asarray(self.env.bowl.get_pose().p, dtype=float)
            x, y = _clamp_table_xy(self.env, p[0], p[1])
            _set_bowl_xy(self.env, x, y, _bowl_place_z(self.env))
            self.env._fix_bowl_at_placed_pose()
            self.env._bowl_ready = True
            self.placed = True
            print(f"Bowl placed at ({x:.3f}, {y:.3f}) behind red line.")


class RobotBowlController:
    def __init__(self, env, ArmTag):
        self.env = env
        self.ArmTag = ArmTag
        self.arm = None
        self.holding = False
        self.placed = False
        self.busy = False
        self._pending_xy = None
        self._space = EdgeKey()

    def _choose_arm(self):
        return self.ArmTag("left" if self.env.mirrored else "right")

    def grasp(self):
        self.busy = True
        self.arm = self._choose_arm()
        self.env.move(self.env.grasp_actor(self.env.bowl, arm_tag=self.arm, pre_grasp_dis=0.10))
        if self.env.plan_success:
            self.env._weld_bowl_to_end_effector(self.arm)
            self.env.move(self.env.move_by_displacement(self.arm, z=0.12, move_axis="arm"))
            self.holding = True
            print(f"Picked up bowl with {self.arm} arm. Left-click the table to place.")
            if self._pending_xy is not None:
                x, y = self._pending_xy
                self._pending_xy = None
                self.busy = False
                self.place_at(x, y)
                return
        else:
            print("Grasp failed; planner disabled further robot actions.")
        self.busy = False

    def place_at(self, x, y):
        if self.placed:
            return
        if not self.holding or self.arm is None:
            self._pending_xy = (float(x), float(y))
            print(
                f"Place target ({x:.3f}, {y:.3f}) saved — press Space to pick up, "
                "then it will place."
            )
            return
        self.busy = True
        x, y = _clamp_table_xy(self.env, x, y)
        bowl_now = np.asarray(self.env.bowl.get_pose().p, dtype=float)
        target = np.array([x, y, _bowl_place_z(self.env)], dtype=float)
        d = target - bowl_now
        self.env.move(self.env.move_by_displacement(
            arm_tag=self.arm, x=float(d[0]), y=float(d[1]), z=float(d[2]), move_axis="world",
        ))
        self.env._unweld_bowl()
        self.env.move(self.env.open_gripper(self.arm))
        self.env._fix_bowl_at_placed_pose()
        self.env.move(self.env.move_by_displacement(self.arm, z=0.12, move_axis="arm"))
        self.env._bowl_ready = True
        self.holding = False
        self.placed = True
        print(f"Placed bowl at click ({x:.3f}, {y:.3f}) behind red line.")
        self.busy = False

    def on_table_click(self, x, y):
        if self.busy or self.placed:
            return False
        x, y = _clamp_table_xy(self.env, x, y)
        self.place_at(x, y)
        return True

    def nudge(self, window):
        if self.busy or not self.holding or self.arm is None or self.placed:
            return
        dx, dy = _nudge_from_keys(window, step=0.02)
        if not (dx or dy):
            return
        self.busy = True
        self.env.move(self.env.move_by_displacement(
            arm_tag=self.arm, x=dx, y=dy, move_axis="world",
        ))
        self.busy = False

    def update(self, window):
        if self.busy:
            return
        if self._space.poll(window.key_down("space")):
            if not self.holding and not self.placed:
                self.grasp()
            return
        self.nudge(window)


def main():
    parser = argparse.ArgumentParser(description="Interactive catch_valley_ball viewer")
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
    from envs.catch_valley_ball import catch_valley_ball
    from envs.utils.action import ArmTag
    globals()["CONFIGS_PATH"] = CONFIGS_PATH

    print(CONTROLS)
    if args.robot_motion == "interpolate":
        print(
            "Note: --robot-motion interpolate uses planner motions for this teleop task "
            "(key-press sandboxes use joint interpolation)."
        )

    env = catch_valley_ball()
    env.setup_demo(**_configure_task(args.config, args.seed, use_robot=args.control == "robot"))
    # setup_demo already starts ball motion with expert_demo=False.
    x, y = _target_xy(env)
    print(
        f"Predicted catch target ≈ ({x:.3f}, {y:.3f}); red_line_x={env.red_line_x:.3f}; "
        f"mirrored={env.mirrored}."
    )

    controller = (
        RobotBowlController(env, ArmTag) if args.control == "robot" else KeyboardBowlController(env)
    )

    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    views = make_viewer_view_toggle(env, viewer)

    def _handle_table_click(viewer_, pixel_x, pixel_y):
        table_z = float(env.table_top)
        xy = _click_to_table_xy(viewer_, pixel_x, pixel_y, table_z)
        if xy is None:
            print("Click did not hit the table plane.")
            return False
        return bool(controller.on_table_click(xy[0], xy[1]))

    viewer.register_click_handler(_handle_table_click)
    print("Left-click the table to set the bowl location; Space picks up / places.")

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

            remaining = float(env.scene.get_timestep()) - (time.perf_counter() - frame_start)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        env.close_env()


if __name__ == "__main__":
    main()
