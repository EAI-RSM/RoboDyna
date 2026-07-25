#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive, top-down default-mode viewer for ``sort_apples_belt``.

Run directly from any directory:

    /path/to/RoboDynaExp/script_exp/interactive_sort_apples.py --control keyboard
    /path/to/RoboDynaExp/script_exp/interactive_sort_apples.py --control mouse

Keyboard mode changes the diverter directly. Robot mode makes the corresponding
arm physically press a button while the key is held. Mouse mode: click a red or
green button to toggle its routing direction. This is an interaction sandbox, not a
data-collection or robot-control rollout.
"""

import argparse
from datetime import datetime
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml


# This copy lives at <repo>/script_exp/; resolve the repository independently
# of the caller's working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "script" / "bench_script"))
sys.path.insert(0, str(REPO_ROOT / "script_exp"))

from _interactive_common import (  # noqa: E402
    make_viewer_view_toggle,
    default_arms_for_mode,
    make_button_controller,
    report_task_result,
)


def _embodiment_config(robot_file):
    with open(Path(robot_file) / "config.yml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _configure_default_task(config_name: str, seed: int, use_robot: bool = False):
    config_path = REPO_ROOT / "task_config" / f"{config_name}.yml"
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    # Preserve all task settings from the selected config (including its
    # ``rotten_prob``). The interactive launcher needs rendering enabled even
    # though batch demo collection uses ``render_freq: 0``.
    config.setdefault("task_name", "sort_apples_belt")
    config.setdefault("now_ep_num", 0)
    config.setdefault("need_plan", use_robot)
    config.setdefault("save_data", False)
    config["render_freq"] = 1
    config["seed"] = seed

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


def _update_keyboard_control(env, window):
    """Map arrow keys to left, right, or simultaneous button holds."""

    left_down = window.key_down("left")
    right_down = window.key_down("right")
    down_down = window.key_down("down")
    env._expert_hold = "dump" if down_down else (
        "left" if left_down and not right_down else
        "right" if right_down and not left_down else None
    )


def _requested_button_mode(window):
    """Return the physical button combination requested by the arrow keys."""

    if window.key_down("down"):
        return "dump"
    left_down = window.key_down("left")
    right_down = window.key_down("right")
    if left_down and not right_down:
        return "left"
    if right_down and not left_down:
        return "right"
    return None


def _make_robot_controller(env, arm_tag, robot_motion):
    """Shared TCP-limited / interpolate hold-to-actuate button press."""
    return make_button_controller(
        env,
        arm_tag,
        robot_motion,
        get_button=lambda e, side: e.buttons[side],
        get_top_z=lambda e, _side: e._button_top_z,
        set_latch=lambda e, mode: setattr(e, "_expert_hold", mode),
        clear_latch=lambda e: setattr(e, "_expert_hold", None),
        arms_for_mode=default_arms_for_mode,
        hold=True,
        # Match env press band so fingertip touches latch even when TCP is
        # a bit above the geometric key top (EE-only env check often misses).
        active_dz=float(getattr(env, "PRESS_DZ_ACTIVE", 0.12)),
        sides=("left", "right"),
    )


def _button_click_handler(env):
    """Return a segmentation-based handler that toggles a clicked task button."""

    button_ids = {button.actor.per_scene_id: side for side, button in env.buttons.items()}

    def handle_click(viewer, pixel_x, pixel_y):
        pixel = viewer.window.get_picture_pixel("Segmentation", pixel_x, pixel_y)
        side = button_ids.get(int(pixel[1]))
        if side is None:
            return False

        env._expert_hold = side if getattr(env, "_expert_hold", None) != side else None
        color = "red" if env._side_color[side] == env.COLOR_RED else "green"
        state = f"diverting to {color}" if env._expert_hold else "released (plank at rest)"
        print(f"Button click: {color}; {state}")
        return True

    return handle_click


def _start_recorder(output_path, frame, fps=30):
    """Start an MP4 encoder for RGB frames from the interactive viewer."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frame.shape[:2]
    ffmpeg = Path(sys.executable).with_name("ffmpeg")
    if not ffmpeg.exists():
        raise SystemExit(f"FFmpeg was not found beside the active Python: {ffmpeg}")
    process = subprocess.Popen(
        [
            str(ffmpeg), "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", f"{width}x{height}", "-framerate", str(fps),
            "-i", "-", "-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2,hflip,vflip",
            "-pix_fmt", "yuv420p", "-vcodec", "libx264",
            "-crf", "20", str(output_path),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    print(f"Recording interactive demo to {output_path}")
    return process


def _viewer_rgb_frame(viewer):
    """Read the viewer Color buffer as a top-left-origin uint8 RGB frame."""

    rgba = viewer.window.get_picture("Color")
    return np.flipud((rgba[..., :3] * 255).clip(0, 255).astype(np.uint8))


def _close_recorder(process):
    if process is None:
        return
    process.stdin.close()
    if process.wait() != 0:
        error = process.stderr.read().decode("utf-8", errors="replace").strip()
        print(f"Warning: FFmpeg could not finalize the recording: {error or 'no error output'}")


def _disable_failed_recorder(process):
    """Report an early FFmpeg exit and disable recording without stopping the demo."""

    if process.poll() is None:
        return False
    error = process.stderr.read().decode("utf-8", errors="replace").strip()
    print(f"Recording disabled: FFmpeg exited early: {error or 'no error output'}")
    return True


def _add_second_view(env, primary_viewer):
    """Open a perspective viewer beside the overhead interactive viewer."""

    from sapien.utils.viewer import Viewer

    secondary = Viewer(env.renderer)
    secondary.set_scene(env.scene)
    # The standard simulator perspective, complementary to the overhead view.
    secondary.set_camera_xyz(0.477, 0.253, 1.625)
    secondary.set_camera_rpy(0.0, -0.8, 2.45)

    original_render = primary_viewer.render

    def render_both(*args, **kwargs):
        # Re-apply both poses because the viewer control plugin can otherwise
        # synchronize camera state after a window receives focus.
        primary_viewer.set_camera_xyz(0.0, 0.0, 2.1)
        primary_viewer.set_camera_rpy(0.0, -np.pi / 2.0, -np.pi / 2.0)
        result = original_render(*args, **kwargs)
        if not secondary.closed:
            secondary.set_camera_xyz(0.477, 0.253, 1.625)
            secondary.set_camera_rpy(0.0, -0.8, 2.45)
            secondary.render()
        return result

    # Base-task motions call primary_viewer.render internally; wrapping it keeps
    # the second window live while a planned robot action is executing.
    primary_viewer.render = render_both

    def restore():
        if primary_viewer.render is render_both:
            primary_viewer.render = original_render
        if not secondary.closed:
            secondary.close()

    return secondary, restore


def _camera_pose(sapien, position, look_at, up_hint):
    forward = np.asarray(look_at, dtype=np.float64) - np.asarray(position, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    left = np.cross(np.asarray(up_hint, dtype=np.float64), forward)
    left /= np.linalg.norm(left)
    up = np.cross(forward, left)
    matrix = np.eye(4)
    matrix[:3, :3] = np.stack([forward, left, up], axis=1)
    matrix[:3, 3] = position
    return sapien.Pose(matrix)


class CompositeView:
    """One GUI window containing overhead and robot-head camera renders."""

    WINDOW_NAME = "RoboDyna: Overhead | Head camera"
    DISPLAY_FPS = 20

    def __init__(self, env, sapien):
        self.env = env
        self.window_open = True
        self.action = None
        self.overhead = env.scene.add_camera("interactive_overhead", 640, 480, np.deg2rad(55), 0.1, 10.0)
        self.overhead.entity.set_pose(_camera_pose(
            # Reverse the up axis so the overhead image is rotated 180 degrees:
            # the robot is at the bottom, matching the interactive GUI view.
            sapien, [0.0, 0.0, 2.1], [0.0, 0.0, 0.75], [0.0, -1.0, 0.0],
        ))
        self.head = next(
            (camera for camera, name in zip(env.cameras.static_camera_list, env.cameras.static_camera_name)
             if name == "head_camera"),
            None,
        )
        if self.head is None:
            raise RuntimeError("This task configuration does not provide a head_camera.")
        self._next_refresh = 0.0
        self._last_frame = None
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)

    @staticmethod
    def _rgb(camera):
        camera.take_picture()
        rgba = camera.get_picture("Color")
        return np.flipud((rgba[..., :3] * 255).clip(0, 255).astype(np.uint8))

    def needs_refresh(self):
        return self._last_frame is None or time.perf_counter() >= self._next_refresh

    def render(self, refresh):
        if refresh:
            left = self._rgb(self.overhead)
            right = self._rgb(self.head)
            # Match the overhead panel's orientation without changing the
            # task-owned head camera pose used by observations.
            right = np.rot90(right, 2)
            # The configured head camera is 320x240, while the overhead panel is
            # 640x480. Resize only for display so the two panels align cleanly.
            right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_AREA)
            composite = np.concatenate((left, right), axis=1)
            cv2.putText(composite, "Overhead", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(composite, "Head camera", (658, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            self._last_frame = composite
            self._next_refresh = time.perf_counter() + 1.0 / self.DISPLAY_FPS
            cv2.imshow(self.WINDOW_NAME, cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("x"), ord("X")):
            # Avoid a stuck graphics backend preventing the launcher from exiting.
            os._exit(0)
        if key in (27, ord("q")):
            self.window_open = False
        elif key in (81, ord("a")):
            # The displayed overhead view is rotated 180 degrees, so screen
            # left maps to the task's right-side actuator (and vice versa).
            self.action = None if self.action == "right" else "right"
        elif key in (83, ord("d")):
            self.action = None if self.action == "left" else "left"
        elif key in (84, ord("s")):
            self.action = None if self.action == "dump" else "dump"
        return self._last_frame

    def close(self):
        cv2.destroyWindow(self.WINDOW_NAME)
        self.env.scene.remove_camera(self.overhead)


def _attach_composite_motion_renderer(viewer, composite_view):
    """Keep the composite window responsive during blocking robot motions."""

    original_render = viewer.render

    def render_composite(*_args, **_kwargs):
        # Robot trajectories call ``viewer.render`` from their internal
        # blocking loop.  The stock viewer is hidden in composite mode, so use
        # that callback to refresh the visible OpenCV window instead.
        composite_view.render(composite_view.needs_refresh())

    viewer.render = render_composite

    def restore():
        if viewer.render is render_composite:
            viewer.render = original_render

    return restore


def main():
    parser = argparse.ArgumentParser(description="Interactive top-down sort-apples viewer")
    parser.add_argument("--config", default="demo_dynamic", help="Task config name without .yml")
    parser.add_argument("--seed", type=int, default=0, help="Scene randomization seed")
    parser.add_argument(
        "--record",
        nargs="?",
        const="",
        metavar="PATH",
        help="Record the overhead viewer to an MP4; omit PATH for a timestamped file under demos/interactive_sort_apples.",
    )
    parser.add_argument(
        "--two-views",
        action="store_true",
        help="Open a second live perspective viewer beside the overhead interaction view.",
    )
    parser.add_argument(
        "--composite-view",
        action="store_true",
        help="Show overhead and robot-head cameras side by side in one OpenCV window.",
    )
    parser.add_argument(
        "--control",
        choices=("keyboard", "mouse", "robot"),
        default="keyboard",
        help="Interaction method (default: keyboard)",
    )
    parser.add_argument(
        "--robot-motion",
        choices=("planner", "interpolate"),
        default="interpolate",
        help="Robot key-press implementation; interpolate is a faster test mode (default: interpolate)",
    )
    args = parser.parse_args()
    if args.two_views and args.composite_view:
        raise SystemExit("Choose either --two-views or --composite-view.")
    if args.record == "":
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.record = REPO_ROOT / "demos" / "interactive_sort_apples" / f"sort_apples_{stamp}.mp4"
    elif args.record is not None:
        args.record = Path(args.record).expanduser().resolve()

    # Keep ``--help`` usable on machines without the GPU-only simulator stack.
    from envs import CONFIGS_PATH
    from envs.sort_apples_belt import sort_apples_belt
    from envs.utils.action import ArmTag
    import sapien
    globals()["CONFIGS_PATH"] = CONFIGS_PATH

    env = sort_apples_belt()
    env.setup_demo(**_configure_default_task(
        args.config, args.seed, use_robot=args.control == "robot"
    ))
    # The base task opens both grippers during setup; this interactive launcher
    # starts them closed so button presses use a compact resting posture.
    env.together_close_gripper(save_freq=None)
    belt_clear_since = None
    robot_controller = None
    if args.control == "robot":
        robot_controller = _make_robot_controller(env, ArmTag, args.robot_motion)
    recorder = None
    record_frame_count = 0
    record_every = max(1, round(1.0 / (env.scene.get_timestep() * 30.0)))
    viewer = env.viewer
    if viewer is None:
        raise SystemExit("Viewer was not created; ensure a graphical display is available.")

    # A fixed overhead camera with the robot at the bottom keeps its controls in view.
    viewer.set_camera_xyz(0.0, 0.0, 2.1)
    viewer.set_camera_rpy(0.0, -np.pi / 2.0, -np.pi / 2.0)
    views = make_viewer_view_toggle(env, viewer)
    composite_view = None
    restore_composite_motion_renderer = lambda: None
    if args.composite_view:
        composite_view = CompositeView(env, sapien)
        viewer.window.hide()
        if robot_controller is not None:
            restore_composite_motion_renderer = _attach_composite_motion_renderer(viewer, composite_view)
        print("Composite view ready at 20 FPS. Press A for left, D for right, S for both; press again to release; V toggles view; Q/Escape to quit cleanly; X to force-quit.")
    second_viewer = None
    restore_second_view = lambda: None
    if args.two_views:
        second_viewer, restore_second_view = _add_second_view(env, viewer)
        print("Two-view mode ready: overhead interaction view + perspective view.")
    if args.control == "mouse":
        viewer.register_click_handler(_button_click_handler(env))
        print("Top-down sort-apples sandbox ready. Click red/green to toggle the plank direction.")
    elif args.control == "robot":
        if composite_view is not None:
            print(
                f"Composite robot mode ready ({args.robot_motion}).\n"
                "  A / D / S  — hold to press left / right / both buttons (plank stays diverted while held)\n"
                "  release    — lift and return plank to rest\n"
                "  V          — toggle top-down / head_camera\n"
                "  Q/Esc      — quit; X — force-quit\n"
                "  --robot-motion planner|interpolate"
            )
        else:
            print(
                f"Robot mode ready (hold-to-actuate, {args.robot_motion}).\n"
                "  Left / Right Arrow  — hold to press that button; plank diverts while held\n"
                "  Down Arrow          — hold both buttons (dump hatch when rotten mode is on)\n"
                "  release key         — lift arm(s); plank returns to rest\n"
                "  V                   — toggle top-down / head_camera\n"
                "  Q/Esc               — quit\n"
                "  --robot-motion planner|interpolate"
            )
    else:
        print(
            "Top-down sort-apples sandbox ready (direct diverter control).\n"
            "  Left / Right Arrow  — hold to divert left / right\n"
            "  Down Arrow          — hold both (dump)\n"
            "  V                   — toggle top-down / head_camera\n"
            "  release             — plank returns to rest"
        )

    try:
        while (not viewer.closed and (second_viewer is None or not second_viewer.closed)
               and (composite_view is None or composite_view.window_open)):
            frame_start = time.perf_counter()
            if composite_view is None:
                views.update(viewer.window)
            if composite_view is not None:
                if robot_controller is not None:
                    robot_controller.update(composite_view.action)
                else:
                    env._expert_hold = composite_view.action
            elif args.control == "keyboard":
                _update_keyboard_control(env, viewer.window)
            elif robot_controller is not None:
                # Hold while the key is down. A press-and-immediate-lift "tap"
                # cleared _expert_hold before the plank could stay diverted, and
                # fingertip contact alone often failed the EE-only proximity test.
                robot_controller.update(_requested_button_mode(viewer.window))
            env._update_kinematic_tasks()
            env.scene.step()
            if composite_view is not None:
                refresh_composite = composite_view.needs_refresh()
                if refresh_composite:
                    env.scene.update_render()
                composite_frame = composite_view.render(refresh_composite)
            else:
                env.scene.update_render()
                viewer.render()
                composite_frame = None
            if args.record is not None and record_frame_count % record_every == 0:
                frame = composite_frame if composite_frame is not None else _viewer_rgb_frame(viewer)
                recorder = recorder or _start_recorder(args.record, frame)
                if _disable_failed_recorder(recorder):
                    recorder = None
                    args.record = None
                else:
                    try:
                        recorder.stdin.write(frame.tobytes())
                    except BrokenPipeError:
                        # FFmpeg may fail between the poll and the write.
                        _disable_failed_recorder(recorder)
                        recorder = None
                        args.record = None
            record_frame_count += 1
            timed_out = bool(getattr(env, "_timed_out", False)) or (
                hasattr(env, "_episode_timed_out") and bool(env._episode_timed_out())
            )
            if timed_out:
                report_task_result(env, "timed_out")
                break
            all_spawned = env._spawned >= env.n_apples
            belt_clear = all(not env._apple_on_belt(i) for i in range(env._spawned))
            if all_spawned and belt_clear:
                if belt_clear_since is None:
                    belt_clear_since = time.perf_counter()
                    print("All apples have cleared the belt; stopping in two seconds.")
                elif time.perf_counter() - belt_clear_since >= 2.0:
                    correct = sum(bool(result) for result in env.results)
                    report_task_result(
                        env, f"{correct}/{env.n_apples} apples sorted correctly"
                    )
                    break
            else:
                belt_clear_since = None
            # Keep GUI interaction responsive while preserving the configured physics timestep.
            remaining = float(env.scene.get_timestep()) - (time.perf_counter() - frame_start)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        try:
            if robot_controller is not None:
                robot_controller.release()
            _close_recorder(recorder)
        finally:
            restore_second_view()
            restore_composite_motion_renderer()
            if composite_view is not None:
                composite_view.close()
            env.close_env()


if __name__ == "__main__":
    main()
