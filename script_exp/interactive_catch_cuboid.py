#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive viewer for ``catch_cuboid``.

Run from any directory:

    /path/to/RoboDynaExp/script_exp/interactive_catch_cuboid.py --control keyboard
    /path/to/RoboDynaExp/script_exp/interactive_catch_cuboid.py --control robot

Close the gripper (G) while the cuboid is rising to latch it, then lift it out.
Opt1 supports dual arms.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import sapien
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "script" / "bench_script"))
sys.path.insert(0, str(REPO_ROOT / "script_exp"))

from _interactive_common import (  # noqa: E402
    print_instructions,
    action_failed,
    gripper_width,
    make_viewer_view_toggle,
    print_mode_controls,
    report_task_result,
    RealtimePhysicsPacer,
    terminal_hold_should_close,
    print_episode_condition,
)

# ur5-wsg gripper visual links (recolored to show arm selection).
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


CONTROLS_KEYBOARD = """
  G                 open / close gripper (closing latches a rising cuboid; you lift it out)
"""

CONTROLS_ROBOT = """
  G                 open / close gripper (closing latches a rising cuboid; you lift it out)
  Arrow keys        move selected arm in XY
  E / Q             move selected arm in Z
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
        task_name="catch_cuboid",
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


def _cuboid_rising(env, idx=0):
    motion = env._cuboid_auto_motion[idx] if idx < len(env._cuboid_auto_motion) else None
    if motion == "rising":
        return True
    # Also accept raised / near crest.
    if idx < len(env.cuboids):
        top = float(env.cuboids[idx].get_pose().p[2]) + float(env.cuboid_half[2])
        return top >= float(env.board_top_z) + 0.005
    return False


def _close_gripper_direct(env, arm_name):
    """Close without planner (keyboard mode)."""
    try:
        env.robot.set_gripper(0.0, arm_name, gripper_eps=0.0)
    except Exception:
        pass


def _try_latch_catch(env, cuboid_idx, arm):
    """If the closed gripper is pinching the cuboid, release it to dynamics (no lift)."""
    caught, offset = env._try_catch(cuboid_idx, arm)
    held = env._cuboid_in_gripper(env._cuboid_names[cuboid_idx], arm)
    if caught and held:
        env._release_cuboid(cuboid_idx)
        print(f"Latched {env._cuboid_names[cuboid_idx]} (offset={offset:.3f} m); lift it out.")
        return True
    print(
        f"Missed {env._cuboid_names[cuboid_idx]} "
        f"(rising={_cuboid_rising(env, cuboid_idx)}, held={held}, offset={offset:.3f})."
    )
    return False


def _cuboid_idx_for_arm(dual: bool, arm_name: str) -> int:
    if dual:
        return 0 if arm_name == "left" else 1
    return 0


def _selected_arms(env, fallback=("right",)):
    selected = tuple(getattr(env, "_interactive_selected_arms", ()) or ())
    return selected if selected else tuple(fallback)


def _mark_latch_failure(controller, env, arms, detail="insufficient contact"):
    """Closing without a solid pinch ends the episode as failure."""
    action_failed(env, arms, detail=detail)
    controller.done = True
    controller.fail_detail = detail
    print(f"Latch failed ({detail}) — episode FAILURE.")


class KeyboardCatchController:
    """On G close, latch the cuboid; user lifts for success."""

    def __init__(self, env, ArmTag):
        self.env = env
        self.ArmTag = ArmTag
        self.dual = bool(env.dual_catch)
        self.selected = "right"
        if not self.dual:
            hole = env._cuboid_holes[0]
            self.selected = "right" if env.holes[hole][0] > 0 else "left"
        self._g = EdgeKey()
        self._q = EdgeKey()
        self._e = EdgeKey()
        self._pending = None  # (arms, cuboid_indices, steps_left)
        self._prev_width = {"left": 1.0, "right": 1.0}
        self._latched = set()
        self.done = False
        self.fail_detail = None

    def _begin_latch(self, arms):
        indices = [_cuboid_idx_for_arm(self.dual, a) for a in arms]
        self._pending = (list(arms), indices, 20)
        print(f"Gripper closing on {', '.join(arms)}; checking latch…")

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
        attempted = []
        for arm_name, idx in zip(arms, indices):
            if idx in self._latched:
                continue
            attempted.append(idx)
            if _try_latch_catch(self.env, idx, self.ArmTag(arm_name)):
                self._latched.add(idx)
        if attempted and not all(idx in self._latched for idx in attempted):
            _mark_latch_failure(self, self.env, arms)

    def update(self, window):
        if self.done:
            return
        self._tick_pending()
        if self._pending:
            return
        if self.dual:
            if self._q.poll(window.key_down("q")):
                self.selected = "left"
                self.env._interactive_selected_arms = ("left",)
                print("Selected LEFT arm.")
            if self._e.poll(window.key_down("e")):
                self.selected = "right"
                self.env._interactive_selected_arms = ("right",)
                print("Selected RIGHT arm.")

        arms = list(_selected_arms(self.env, (self.selected,)))
        g_close = self._g.poll(window.key_down("g")) and any(
            self._prev_width.get(a, 1.0) > 0.5 for a in arms
        )
        for side in ("left", "right"):
            self._prev_width[side] = gripper_width(self.env, side)
        if g_close:
            self._begin_latch(arms)


class RobotCatchController:
    """On G close, latch the cuboid; user teleops the lift."""

    def __init__(self, env, ArmTag):
        self.env = env
        self.ArmTag = ArmTag
        self.dual = bool(env.dual_catch)
        self.busy = False
        self._g = EdgeKey()
        self.selected = "right"
        if not self.dual:
            hole = env._cuboid_holes[0]
            self.selected = "right" if env.holes[hole][0] > 0 else "left"
        self._highlight = ArmGripperHighlight(env)
        self._highlight.set_selected(self.selected)
        self._prev_width = {"left": 1.0, "right": 1.0}
        self._latched = set()
        self.done = False
        self.fail_detail = None

    def _select(self, side):
        side = "left" if side == "left" else "right"
        changed = side != self.selected
        self.selected = side
        self._highlight.set_selected(side)
        if changed:
            print(f"Selected {side.upper()} (gripper highlighted).")

    def _latch_selected(self, arms):
        self.busy = True
        print(f"Gripper closing on {', '.join(arms)}; checking latch…")
        for side in arms:
            _close_gripper_direct(self.env, side)
        self.env._dwell(15)
        attempted = []
        for side in arms:
            idx = _cuboid_idx_for_arm(self.dual, side)
            if idx in self._latched:
                continue
            attempted.append(idx)
            if _try_latch_catch(self.env, idx, self.ArmTag(side)):
                self._latched.add(idx)
        if attempted and not all(idx in self._latched for idx in attempted):
            _mark_latch_failure(self, self.env, arms)
        self.busy = False

    def update(self, window):
        if self.done or self.busy:
            return
        selected = _selected_arms(self.env, (self.selected,))
        if len(selected) == 1:
            self._select(selected[0])

        arms = list(selected)
        g_close = self._g.poll(window.key_down("g")) and any(
            self._prev_width.get(a, 1.0) > 0.5 for a in arms
        )
        for side in ("left", "right"):
            self._prev_width[side] = gripper_width(self.env, side)
        if g_close:
            self._latch_selected(arms)

def main():
    parser = argparse.ArgumentParser(description="Interactive catch_cuboid viewer")
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
    from envs.catch_cuboid import catch_cuboid
    from envs.utils.action import ArmTag
    globals()["CONFIGS_PATH"] = CONFIGS_PATH

    # Ensure default raster viewer (no leftover RT glass shader from prior runs).
    try:
        default_shader = Path(sapien.__file__).resolve().parent / "vulkan_shader" / "default"
        if default_shader.is_dir():
            sapien.render.set_viewer_shader_dir(str(default_shader))
    except Exception:
        pass

    print_mode_controls("catch_cuboid", args.control, keyboard=CONTROLS_KEYBOARD, robot=CONTROLS_ROBOT)
    if args.robot_motion == "interpolate":
        print(
            "Note: --robot-motion interpolate uses planner motions for this teleop task "
            "(key-press sandboxes use joint interpolation)."
        )

    env = catch_cuboid()
    env.setup_demo(**_configure_task(args.config, args.seed, use_robot=args.control == "robot"))
    print_episode_condition(env)
    if env.dual_catch:
        env._interactive_selected_arms = ("left", "right")
    else:
        hole = env._cuboid_holes[0]
        env._interactive_selected_arms = (
            "right" if env.holes[hole][0] > 0 else "left",
        )
    # Start with open grippers so G close has an effect.
    try:
        env.together_open_gripper(save_freq=None)
    except Exception:
        pass
    print(
        f"dual={env.dual_catch}; opaque={env.opaque_surface}; "
        f"holes={env._cuboid_holes}; speed={env._cuboid_pop_speed:.3f} m/s."
    )
    if args.control == "robot":
        print_instructions(
            "Arrows/E/Q move the arm; close with G to latch, then lift the cuboid out."
        )
    else:
        print_instructions(
            "Wait for the cuboid to rise, close with G to latch, then lift it out."
        )
    controller = (
        RobotCatchController(env, ArmTag) if args.control == "robot"
        else KeyboardCatchController(env, ArmTag)
    )

    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")
    views = make_viewer_view_toggle(env, viewer)

    settle_after = None
    # Env never increments appearances_done; count completed pop cycles locally.
    num_appearances = int(getattr(env, "num_appearances", 5) or 5)
    cycles_done = 0
    prev_motion = list(getattr(env, "_cuboid_auto_motion", []) or [])
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

            motions = list(getattr(env, "_cuboid_auto_motion", []) or [])
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
                    if getattr(controller, "done", False) and not env.check_success():
                        detail = getattr(controller, "fail_detail", None) or "insufficient contact"
                    elif appearances_exhausted:
                        detail = f"missed after {cycles_done}/{num_appearances} appearances"
                    else:
                        detail = f"catches={env.catches}"
                    report_task_result(env, detail)
                    terminal_started_at = time.perf_counter()
    finally:
        env.close_env()


if __name__ == "__main__":
    main()
    # household_task_gui convention: 0=SUCCESS, 10=FAILURE, 2=no result
    from _interactive_common import task_result_exit_code
    raise SystemExit(task_result_exit_code())
