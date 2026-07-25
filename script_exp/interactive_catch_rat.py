#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive viewer for ``catch_rat``.

Run from any directory:

    /path/to/RoboDynaExp/script_exp/interactive_catch_rat.py --control keyboard
    /path/to/RoboDynaExp/script_exp/interactive_catch_rat.py --control robot

Close the gripper while the kinematic rat is rising. Opt1 dual: Q/E select arm.
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

from _interactive_common import make_viewer_view_toggle, report_task_result  # noqa: E402

# ur5-wsg gripper visual links (recolored to show Q/E selection).
_GRIPPER_LINK_NAMES = (
    "wsg_50_base_link",
    "gripper_left",
    "gripper_right",
    "finger_left",
    "finger_right",
)
_ARM_HIGHLIGHT = {
    "left": [1.0, 0.85, 0.10, 1.0],   # yellow
    "right": [0.15, 0.75, 1.0, 1.0],  # cyan
}


CONTROLS = """
============================================================
 catch_rat — interactive controls
============================================================
  Mouse left-click  robot: move selected arm above click XY
  Space             close gripper(s) when the rat is rising
                    robot: first press approaches hole(s) if needed
  Q / E             dual (Opt1): select left / right arm
                    (selected gripper is recolored: yellow=left, cyan=right)
  A                 dual (keyboard): close BOTH grippers
  R                 robot: (re)approach hole(s)
  V                 toggle view: top-down ↔ head_camera
  Escape            quit
------------------------------------------------------------
  Success: rat held in gripper (both if catch_two_mice)
  --control keyboard  direct gripper close (default)
  --control robot     click-to-go / approach hole + Space close
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
        task_name="catch_rat",
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


class ArmGripperHighlight:
    """Recolor gripper meshes so the selected arm is obvious in the viewer."""

    def __init__(self, env):
        self.env = env
        self._orig = {}  # id(material) -> (material, rgba)
        self._entities = {
            "left": self._gripper_entities("left"),
            "right": self._gripper_entities("right"),
        }
        self._selected = None

    def _gripper_entities(self, side):
        robot = self.env.robot
        art = robot.left_entity if side == "left" else robot.right_entity
        out = []
        for link in art.get_links():
            if link.get_name() in _GRIPPER_LINK_NAMES:
                out.append(link.entity)
        return out

    def _iter_materials(self, entity):
        for comp in entity.get_components():
            if not isinstance(comp, sapien.render.RenderBodyComponent):
                continue
            for shape in comp.render_shapes:
                try:
                    yield shape.material
                except Exception:
                    continue

    def _remember(self, mat):
        key = id(mat)
        if key in self._orig:
            return
        try:
            rgba = list(mat.base_color)
        except Exception:
            rgba = [0.75, 0.75, 0.75, 1.0]
        self._orig[key] = (mat, rgba)

    def _apply(self, mat, rgba):
        self._remember(mat)
        try:
            mat.set_base_color_texture(None)
        except Exception:
            pass
        try:
            mat.set_base_color(list(rgba))
            mat.base_color = list(rgba)
        except Exception:
            try:
                mat.set_base_color(list(rgba))
            except Exception:
                pass
        try:
            mat.set_metallic(0.05)
            mat.set_roughness(0.35)
        except Exception:
            pass

    def _restore_all(self):
        for mat, rgba in self._orig.values():
            try:
                mat.set_base_color(list(rgba))
                mat.base_color = list(rgba)
            except Exception:
                try:
                    mat.set_base_color(list(rgba))
                except Exception:
                    pass

    def set_selected(self, side):
        side = "left" if side == "left" else "right"
        if side == self._selected:
            return
        self._restore_all()
        color = _ARM_HIGHLIGHT[side]
        for entity in self._entities.get(side, []):
            for mat in self._iter_materials(entity):
                self._apply(mat, color)
        self._selected = side


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


def _clamp_to_box_top(env, x, y):
    """Clamp XY to the hole-board top footprint."""
    cx = float(env.board_center[0])
    cy = float(env.board_center[1])
    hx = float(env.BOARD_HALF[0])
    hy = float(env.BOARD_HALF[1])
    return (
        float(np.clip(x, cx - hx, cx + hx)),
        float(np.clip(y, cy - hy, cy + hy)),
    )


def _click_to_box_top_xy(viewer, pixel_x, pixel_y, env):
    """Map a viewer click to XY on the box (hole-board) top surface."""
    plane_z = float(env.board_top_z)
    window = viewer.window
    px = int(pixel_x)
    py = int(pixel_y)
    try:
        model = np.asarray(window.get_camera_model_matrix(), dtype=np.float64)
    except Exception:
        return None
    origin = model[:3, 3]
    rot = model[:3, :3]

    # Prefer the rendered surface under the cursor when it is the box top
    # (opaque glass / lattice in the default raster viewer).
    try:
        pos = np.asarray(window.get_picture_pixel("Position", px, py), dtype=np.float64)
        if pos.shape[0] >= 3 and np.all(np.isfinite(pos[:3])):
            depth_ok = True if pos.shape[0] < 4 else float(pos[3]) < 0.999
            if depth_ok and float(np.linalg.norm(pos[:3])) > 1e-6:
                world = rot @ pos[:3] + origin
                if abs(float(world[2]) - plane_z) <= 0.08:
                    return _clamp_to_box_top(env, float(world[0]), float(world[1]))
                # Clicked something else (arm, rat, table): project onto box top.
                hit = _ray_hit_plane_xy(origin, world - origin, plane_z)
                if hit is not None:
                    return _clamp_to_box_top(env, hit[0], hit[1])
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
    fovy = float(window.fovy)
    aspect = float(tw) / float(th)
    tan_y = float(np.tan(0.5 * fovy))
    tan_x = tan_y * aspect
    dir_cam = np.array([ndc_x * tan_x, ndc_y * tan_y, -1.0], dtype=np.float64)
    hit = _ray_hit_plane_xy(origin, rot @ dir_cam, plane_z)
    if hit is None:
        return None
    return _clamp_to_box_top(env, hit[0], hit[1])


def _rat_rising(env, idx=0):
    motion = env._rat_auto_motion[idx] if idx < len(env._rat_auto_motion) else None
    if motion == "rising":
        return True
    # Also accept raised / near crest.
    if idx < len(env.rats):
        top = float(env.rats[idx].get_pose().p[2]) + float(env.rat_half[2])
        return top >= float(env.board_top_z) + 0.005
    return False


def _close_gripper_direct(env, arm_name):
    """Close without planner (keyboard mode)."""
    try:
        env.robot.set_gripper(0.0, arm_name, gripper_eps=0.0)
    except Exception:
        pass


def _try_finish_catch(env, rat_idx, arm):
    caught, offset = env._try_catch(rat_idx, arm)
    if caught or env._rat_in_gripper(env._rat_names[rat_idx]):
        env._release_rat(rat_idx)
        print(f"Caught {env._rat_names[rat_idx]} (offset={offset:.3f} m).")
        return True
    print(f"Missed {env._rat_names[rat_idx]} (rising={_rat_rising(env, rat_idx)}, offset={offset:.3f}).")
    return False


class KeyboardCatchController:
    def __init__(self, env, ArmTag):
        self.env = env
        self.ArmTag = ArmTag
        self.dual = bool(env.dual_catch)
        self.selected = "right"
        if not self.dual:
            hole = env._rat_holes[0]
            self.selected = "right" if env.holes[hole][0] > 0 else "left"
        self._space = EdgeKey()
        self._both = EdgeKey()
        self._q = EdgeKey()
        self._e = EdgeKey()
        self.done = False
        self._pending = None  # (arms, rat_indices, steps_left)

    def _begin_close(self, arms, rat_indices):
        for a in arms:
            _close_gripper_direct(self.env, a)
        self._pending = (list(arms), list(rat_indices), 20)
        print(f"Closing {', '.join(arms)}…")

    def _tick_pending(self):
        if not self._pending:
            return
        arms, indices, left = self._pending
        for a in arms:
            _close_gripper_direct(self.env, a)
        left -= 1
        if left > 0:
            self._pending = (arms, indices, left)
            return
        self._pending = None
        ok_any = False
        for arm_name, idx in zip(arms, indices):
            ok_any = _try_finish_catch(self.env, idx, self.ArmTag(arm_name)) or ok_any
        if self.dual:
            self.done = bool(all(self.env._rats_held()) or self.env.catches >= 2)
        else:
            self.done = bool(ok_any or self.env.check_success())

    def update(self, window):
        if self.done:
            return
        self._tick_pending()
        if self._pending:
            return
        if self.dual:
            if self._q.poll(window.key_down("q")):
                self.selected = "left"
                print("Selected LEFT arm.")
            if self._e.poll(window.key_down("e")):
                self.selected = "right"
                print("Selected RIGHT arm.")
            if self._both.poll(window.key_down("a")):
                self._begin_close(["left", "right"], [0, 1])
                return
        if self._space.poll(window.key_down("space")):
            if self.dual:
                idx = 0 if self.selected == "left" else 1
                self._begin_close([self.selected], [idx])
            else:
                self._begin_close([self.selected], [0])


class RobotCatchController:
    HOVER_ABOVE_BOARD = 0.16

    def __init__(self, env, ArmTag):
        self.env = env
        self.ArmTag = ArmTag
        self.dual = bool(env.dual_catch)
        self.approached = False
        self.busy = False
        self.done = False
        self._space = EdgeKey()
        self._reapproach = EdgeKey()
        self.selected = "right"
        if not self.dual:
            hole = env._rat_holes[0]
            self.selected = "right" if env.holes[hole][0] > 0 else "left"
        self._q = EdgeKey()
        self._e = EdgeKey()
        self._highlight = ArmGripperHighlight(env)
        self._highlight.set_selected(self.selected)

    def _select(self, side):
        side = "left" if side == "left" else "right"
        changed = side != self.selected
        self.selected = side
        self._highlight.set_selected(side)
        if changed:
            print(f"Selected {side.upper()} (gripper highlighted; click moves this arm).")

    def approach(self):
        self.busy = True
        if self.dual:
            self.env.move(
                self.env._approach_rat(0, self.ArmTag("left")),
                self.env._approach_rat(1, self.ArmTag("right")),
            )
        else:
            arm = self.ArmTag(self.selected)
            self.env.move(self.env._approach_rat(0, arm))
        self.approached = bool(self.env.plan_success)
        print("Approached hole(s)." if self.approached else "Approach failed.")
        self.busy = False

    def go_to_xy(self, x, y):
        """Move the selected arm above ``(x, y)`` on the board (world frame)."""
        if self.busy or self.done:
            return False
        # Prefer the arm on that side of the table; Q/E still overrides in dual.
        arm_name = self.selected
        if not self.dual:
            arm_name = "right" if float(x) > 0 else "left"
            self._select(arm_name)
        arm = self.ArmTag(arm_name)
        self.busy = True
        self.env.plan_success = True
        current = np.asarray(self.env.get_arm_pose(arm), dtype=float)
        target_z = float(self.env.board_top_z) + float(self.HOVER_ABOVE_BOARD)
        dx = float(x) - float(current[0])
        dy = float(y) - float(current[1])
        dz = target_z - float(current[2])
        self.env.move(self.env.move_by_displacement(
            arm_tag=arm, x=dx, y=dy, z=dz, move_axis="world",
        ))
        ok = bool(self.env.plan_success)
        if ok:
            self.approached = True
            print(f"Moved {arm_name} arm to ({float(x):.3f}, {float(y):.3f}).")
        else:
            print(f"Move to ({float(x):.3f}, {float(y):.3f}) failed.")
            self.env.plan_success = True
            self.env._last_plan_fail = None
        self.busy = False
        return ok

    def on_board_click(self, x, y):
        return self.go_to_xy(x, y)

    def close_selected(self):
        self.busy = True
        if self.dual:
            # Close both when Space is used in dual after approach.
            self.env.move(
                self.env.close_gripper(self.ArmTag("left")),
                self.env.close_gripper(self.ArmTag("right")),
            )
            self.env._dwell(15)
            ok_l = _try_finish_catch(self.env, 0, self.ArmTag("left"))
            ok_r = _try_finish_catch(self.env, 1, self.ArmTag("right"))
            if ok_l and ok_r:
                self.env.move(
                    self.env.move_by_displacement(self.ArmTag("left"), z=0.12, move_axis="arm"),
                    self.env.move_by_displacement(self.ArmTag("right"), z=0.12, move_axis="arm"),
                )
                self.done = True
        else:
            arm = self.ArmTag(self.selected)
            self.env.move(self.env.close_gripper(arm))
            self.env._dwell(10)
            if _try_finish_catch(self.env, 0, arm):
                self.env.move(self.env.move_by_displacement(arm, z=0.12, move_axis="arm"))
                self.done = True
        self.busy = False

    def update(self, window):
        if self.busy or self.done:
            return
        if self.dual:
            if self._q.poll(window.key_down("q")):
                self._select("left")
            if self._e.poll(window.key_down("e")):
                self._select("right")
        if self._reapproach.poll(window.key_down("r")):
            self.approach()
            return
        if self._space.poll(window.key_down("space")):
            if not self.approached:
                self.approach()
            else:
                self.close_selected()


def main():
    parser = argparse.ArgumentParser(description="Interactive catch_rat viewer")
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
    from envs.catch_rat import catch_rat
    from envs.utils.action import ArmTag
    globals()["CONFIGS_PATH"] = CONFIGS_PATH

    # Ensure default raster viewer (no leftover RT glass shader from prior runs).
    try:
        default_shader = Path(sapien.__file__).resolve().parent / "vulkan_shader" / "default"
        if default_shader.is_dir():
            sapien.render.set_viewer_shader_dir(str(default_shader))
    except Exception:
        pass

    print(CONTROLS)
    if args.robot_motion == "interpolate":
        print(
            "Note: --robot-motion interpolate uses planner motions for this teleop task "
            "(key-press sandboxes use joint interpolation)."
        )

    env = catch_rat()
    env.setup_demo(**_configure_task(args.config, args.seed, use_robot=args.control == "robot"))
    # Start with open grippers so Space has an effect.
    try:
        env.together_open_gripper(save_freq=None)
    except Exception:
        pass
    print(
        f"dual={env.dual_catch}; opaque={env.opaque_surface}; "
        f"holes={env._rat_holes}; speed={env._rat_pop_speed:.3f} m/s."
    )
    if args.control == "robot":
        print("Left-click the board to move the arm; Space closes when the rat rises.")
    else:
        print("Wait for the rat to rise, then press Space.")

    controller = (
        RobotCatchController(env, ArmTag) if args.control == "robot"
        else KeyboardCatchController(env, ArmTag)
    )

    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    views = make_viewer_view_toggle(env, viewer)

    if args.control == "robot":
        def _handle_board_click(viewer_, pixel_x, pixel_y):
            if getattr(controller, "busy", False) or getattr(controller, "done", False):
                return False
            xy = _click_to_box_top_xy(viewer_, pixel_x, pixel_y, env)
            if xy is None:
                print("Click did not hit the box top.")
                return False
            return bool(controller.on_board_click(xy[0], xy[1]))

        viewer.register_click_handler(_handle_board_click)

    settle_after = None
    # Env never increments appearances_done; count completed pop cycles locally.
    num_appearances = int(getattr(env, "num_appearances", 3) or 3)
    cycles_done = 0
    prev_motion = list(getattr(env, "_rat_auto_motion", []) or [])
    try:
        while not viewer.closed:
            frame_start = time.perf_counter()
            views.update(viewer.window)
            controller.update(viewer.window)

            env._update_kinematic_tasks()
            env.scene.step()
            env.scene.update_render()
            viewer.render()

            if viewer.window.key_down("escape"):
                break

            motions = list(getattr(env, "_rat_auto_motion", []) or [])
            for i, motion in enumerate(motions):
                was = prev_motion[i] if i < len(prev_motion) else None
                # A full appearance ends when falling returns to rising (hidden).
                if was == "falling" and motion == "rising":
                    cycles_done += 1
            prev_motion = motions

            appearances_exhausted = (
                cycles_done >= num_appearances and not env.check_success()
            )
            if (getattr(controller, "done", False) or env.check_success()
                    or appearances_exhausted):
                if settle_after is None:
                    settle_after = time.perf_counter()
                elif time.perf_counter() - settle_after >= 1.0:
                    detail = (
                        f"missed after {cycles_done}/{num_appearances} appearances"
                        if appearances_exhausted
                        else f"catches={env.catches}"
                    )
                    report_task_result(env, detail)
                    break

            remaining = float(env.scene.get_timestep()) - (time.perf_counter() - frame_start)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        env.close_env()


if __name__ == "__main__":
    main()
