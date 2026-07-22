#!/usr/bin/env python3
"""Standard dual-view demo recorder for RoboDyna / DOMINO tasks.

Records ONE successful expert rollout and writes two MP4s:
  - robot head  -> ``head_camera``  (elevated training cam)
  - top-down    -> ``observer_camera`` reframed as a bird's-eye nadir view

Output layout (flat tmp dir, versioned, never overwrites prior demos)::

    ./tmp_<task>/video/vN_head.mp4
    ./tmp_<task>/video/vN_topdown.mp4
    ./tmp_<task>/video/vN_sidebyside.mp4   # convenience montage

Usage (from repo root, with the robodyna / domino env active)::

    python script/bench_script/record_demo.py place_block_belt
    python script/bench_script/record_demo.py place_block_belt --config demo_dynamic
    python script/bench_script/record_demo.py catch_rat --task-arg catch_two_mice=true
    python script/bench_script/record_demo.py catch_rat --task-arg opaque_surface=true
    # legacy shorthand still works: --option 1 (catch_two_mice) / --option 2 (opaque_surface)

Agents MUST use this script for task demo videos (not ad-hoc record_*_demo.py
copies, and not the training ``head_camera``-only collector preview).
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys

sys.path.insert(0, "./")
sys.path.insert(0, "./script/bench_script")

import numpy as np
import sapien
import yaml

from envs import CONFIGS_PATH
from envs.utils.images_to_video import images_to_video
from envs.utils.pkl2hdf5 import load_pkl_file
from script.collect_data import class_decorator, get_embodiment_config, run

# Bird's-eye framing: above table center, looking straight down.
TOPDOWN_POS = np.array([0.0, 0.0, 1.72], dtype=np.float64)
TOPDOWN_LOOK_AT = np.array([0.0, 0.0, 0.78], dtype=np.float64)
TOPDOWN_FOVY_DEG = 55.0
TOPDOWN_WH = (640, 480)  # (w, h)


def next_version(video_dir: str) -> int:
    os.makedirs(video_dir, exist_ok=True)
    nums = []
    for p in glob.glob(os.path.join(video_dir, "v*_head.mp4")):
        m = re.search(r"v(\d+)_", os.path.basename(p))
        if m:
            nums.append(int(m.group(1)))
    # Also accept legacy vN_episode0.mp4 so versioning stays monotonic.
    for p in glob.glob(os.path.join(video_dir, "v*_episode0.mp4")):
        m = re.search(r"v(\d+)_", os.path.basename(p))
        if m:
            nums.append(int(m.group(1)))
    return (max(nums) + 1) if nums else 1


def _encode_mp4(frames: list[np.ndarray], out_mp4: str, fps: float) -> None:
    if not frames:
        raise RuntimeError(f"No frames to encode for {out_mp4}")
    os.makedirs(os.path.dirname(out_mp4) or ".", exist_ok=True)
    stack = np.stack(frames, axis=0)
    if stack.dtype != np.uint8:
        stack = (stack * 255).clip(0, 255).astype(np.uint8)
    n_frames, H, W, _C = stack.shape
    # libx264 yuv420p needs even dims.
    if H % 2 or W % 2:
        import cv2

        H2, W2 = H - (H % 2), W - (W % 2)
        stack = np.stack(
            [cv2.resize(f, (W2, H2), interpolation=cv2.INTER_AREA) for f in stack],
            axis=0,
        )
        n_frames, H, W, _C = stack.shape
    proc = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", f"{W}x{H}", "-framerate", str(fps),
            "-i", "-",
            "-pix_fmt", "yuv420p", "-vcodec", "libx264",
            "-crf", "17", "-preset", "slow",
            out_mp4,
        ],
        stdin=subprocess.PIPE,
    )
    assert proc.stdin is not None
    proc.stdin.write(stack.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        images_to_video(stack, out_path=out_mp4, fps=fps, is_rgb=True)
    else:
        print(f"Wrote {out_mp4} ({n_frames} frames at {W}x{H}, crf=17)")


def _resize_to_height(frame: np.ndarray, height: int) -> np.ndarray:
    import cv2

    h, w = frame.shape[:2]
    if h == height:
        out = frame
    else:
        w2 = int(round(w * (height / h)))
        out = cv2.resize(frame, (w2, height), interpolation=cv2.INTER_AREA)
    if out.shape[1] % 2:
        out = out[:, :-1]
    return out


def merge_dual_view_videos(cache_dir: str, out_head: str, out_topdown: str, out_side: str, fps: float) -> None:
    pkls = sorted(
        glob.glob(os.path.join(cache_dir, "*.pkl")),
        key=lambda p: int(re.search(r"(\d+)", os.path.basename(p)).group(1)),
    )
    head_frames, top_frames, side_frames = [], [], []
    for p in pkls:
        data = load_pkl_file(p)
        obs = data.get("observation", {})
        head = None
        if "head_camera" in obs and "rgb" in obs["head_camera"]:
            head = np.asarray(obs["head_camera"]["rgb"])
        top = data.get("third_view_rgb", None)
        if top is not None:
            top = np.asarray(top)
        if head is None and top is None:
            continue
        if head is not None:
            head_frames.append(head)
        if top is not None:
            top_frames.append(top)
        if head is not None and top is not None:
            H = max(head.shape[0], top.shape[0])
            H += H % 2
            side_frames.append(np.hstack([_resize_to_height(head, H), _resize_to_height(top, H)]))

    if not head_frames:
        raise RuntimeError(f"No head_camera frames in {cache_dir}")
    if not top_frames:
        raise RuntimeError(
            f"No top-down (third_view_rgb) frames in {cache_dir}. "
            "Ensure data_type.third_view=true and the observer was reframed."
        )
    _encode_mp4(head_frames, out_head, fps)
    _encode_mp4(top_frames, out_topdown, fps)
    if side_frames:
        _encode_mp4(side_frames, out_side, fps)


def configure_topdown_camera(task) -> None:
    """Replace observer_camera with a high-res bird's-eye (true top-down) view."""
    cams = getattr(task, "cameras", None)
    if cams is None or getattr(cams, "observer_camera", None) is None:
        return
    scene = getattr(task, "scene", None)
    if scene is None:
        return

    old = cams.observer_camera
    near = float(getattr(old, "near", 0.1))
    far = float(getattr(old, "far", 100.0))
    try:
        scene.remove_camera(old)
    except Exception:
        pass

    w, h = TOPDOWN_WH
    camera = scene.add_camera(
        name="observer_camera",
        width=w,
        height=h,
        fovy=np.deg2rad(TOPDOWN_FOVY_DEG),
        near=near,
        far=far,
    )
    cams.observer_camera = camera

    forward = TOPDOWN_LOOK_AT - TOPDOWN_POS
    forward = forward / np.linalg.norm(forward)
    # Prefer world +y as the image "up" hint so the table's near edge is consistent.
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    left = np.cross(world_up, forward)
    if np.linalg.norm(left) < 1e-6:
        left = np.cross(np.array([1.0, 0.0, 0.0], dtype=np.float64), forward)
    left = left / np.linalg.norm(left)
    up = np.cross(forward, left)
    m = np.eye(4)
    m[:3, :3] = np.stack([forward, left, up], axis=1)
    m[:3, 3] = TOPDOWN_POS
    camera.entity.set_pose(sapien.Pose(m))


def _parse_task_arg_value(raw: str):
    low = raw.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        if any(ch in raw for ch in ".eE"):
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def build_args(
    task_name: str,
    config_name: str,
    save_root: str,
    option: str | None,
    task_arg_overrides: list[str] | None = None,
) -> dict:
    with open(f"./task_config/{config_name}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    args["task_config"] = config_name
    args["episode_num"] = 1
    args["save_path"] = save_root  # flat; call run() directly (not main())
    args["collect_data"] = True
    args["eval_video_log"] = False
    args["save_failed_cases"] = False
    args["use_seed"] = False
    args["check_render_success"] = True
    args["export_lerobot"] = False

    args.setdefault("camera", {})
    args["camera"]["collect_head_camera"] = True
    args["camera"]["collect_wrist_camera"] = False  # demos don't need wrists; faster render
    # Sharper head view for demos (training default is 320x240 D435).
    args["camera"]["head_camera_type"] = "Large_D435"

    args.setdefault("data_type", {})
    args["data_type"]["rgb"] = True
    args["data_type"]["third_view"] = True  # observer -> third_view_rgb (our top-down)

    args.setdefault("task_args", {}).setdefault(task_name, {})
    if option is not None:
        args["task_args"][task_name]["option"] = option
    for item in task_arg_overrides or []:
        if "=" not in item:
            raise ValueError(f"--task-arg expects key=value, got: {item}")
        key, raw = item.split("=", 1)
        args["task_args"][task_name][key.strip()] = _parse_task_arg_value(raw.strip())

    # Failure demos (e.g. force_platform_miss stick) still need a rendered trajectory.
    targs = args["task_args"][task_name]
    if bool(targs.get("force_platform_miss", False)) or bool(targs.get("allow_fail", False)):
        args["save_failed_cases"] = True
        args["check_render_success"] = False

    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
        emb = yaml.load(f.read(), Loader=yaml.FullLoader)

    def emb_file(t):
        return emb[t]["file_path"]

    et = args["embodiment"]
    if len(et) == 1:
        args["left_robot_file"] = emb_file(et[0])
        args["right_robot_file"] = emb_file(et[0])
        args["dual_arm_embodied"] = True
        args["embodiment_name"] = str(et[0])
    else:
        args["left_robot_file"] = emb_file(et[0])
        args["right_robot_file"] = emb_file(et[1])
        args["embodiment_dis"] = et[2]
        args["dual_arm_embodied"] = False
        args["embodiment_name"] = f"{et[0]}+{et[1]}"
    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    return args


def _cleanup_scratch(save_root: str, task_name: str) -> None:
    for junk in (
        "data",
        ".cache",
        "seed.txt",
        "scene_info.json",
        task_name,
        "_traj_data",
        "seed_archive.txt",
        "video/episode0.mp4",
    ):
        p = os.path.join(save_root, junk)
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
        elif os.path.isfile(p):
            os.remove(p)


def record_demo(
    task_name: str,
    config_name: str = "demo_dynamic",
    option: str | None = None,
    task_arg_overrides: list[str] | None = None,
    tag: str | None = None,
) -> dict:
    save_root = os.path.abspath(f"./tmp_{task_name}")
    video_dir = os.path.join(save_root, "video")
    ver = next_version(video_dir)
    # Clear condition tags go in the filename, e.g. v5_opt2_blocker_head.mp4
    stem = f"v{ver}_{tag}" if tag else f"v{ver}"
    out_head = os.path.join(video_dir, f"{stem}_head.mp4")
    out_topdown = os.path.join(video_dir, f"{stem}_topdown.mp4")
    out_side = os.path.join(video_dir, f"{stem}_sidebyside.mp4")

    _cleanup_scratch(save_root, task_name)
    os.makedirs(save_root, exist_ok=True)

    args = build_args(task_name, config_name, save_root, option, task_arg_overrides)
    task = class_decorator(task_name)

    # Force top-down observer framing after every setup_demo (plan + render passes).
    _orig_setup = task.setup_demo

    def _setup_demo(**kwags):
        _orig_setup(**kwags)
        configure_topdown_camera(task)

    task.setup_demo = _setup_demo

    def _merge_dual(self):
        if not self.save_data:
            return
        cache_path = f"{self.save_dir}/.cache/episode{self.ep_num}/"
        fps = 250.0 / float(self.save_freq) if self.save_freq else 15.0
        merge_dual_view_videos(cache_path, out_head, out_topdown, out_side, fps=fps)

    task.merge_pkl_to_hdf5_video = _merge_dual.__get__(task, task.__class__)

    run(task, args)
    _cleanup_scratch(save_root, task_name)

    # Drop leftover empty dirs / conventional episode mp4 if merge didn't write one.
    for junk in (".cache", "data", "seed.txt", "scene_info.json", "_traj_data"):
        p = os.path.join(save_root, junk)
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
        elif os.path.isfile(p):
            os.remove(p)

    print("Demo ready:")
    if tag:
        print("  tag      :", tag)
    print("  head     :", out_head)
    print("  top-down :", out_topdown)
    print("  side-by-side:", out_side)
    return {
        "version": ver,
        "tag": tag,
        "head": out_head,
        "topdown": out_topdown,
        "sidebyside": out_side,
    }


def main():
    parser = argparse.ArgumentParser(description="Record standard head + top-down demo videos.")
    parser.add_argument("task", help="Task name (envs/<task>.py class name), e.g. place_block_belt")
    parser.add_argument(
        "--config",
        default="demo_dynamic",
        help="task_config/<config>.yml to load (default: demo_dynamic)",
    )
    parser.add_argument(
        "--option",
        default=None,
        help="Legacy shorthand for task_args.<task>.option "
             "(catch_rat: 1/catch_two_mice, 2/opaque_surface). Prefer --task-arg.",
    )
    parser.add_argument(
        "--task-arg",
        action="append",
        default=[],
        help="Override task_args.<task> entry, e.g. --task-arg blocker_enabled=true",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Condition label embedded in output names, e.g. opt2_blocker "
             "→ vN_opt2_blocker_head.mp4",
    )
    ns = parser.parse_args()
    # Alias: users sometimes say place_block_task
    task = ns.task
    if task in ("place_block_task", "place_block"):
        print(f"Note: '{task}' -> place_block_belt")
        task = "place_block_belt"
    record_demo(
        task,
        config_name=ns.config,
        option=ns.option,
        task_arg_overrides=ns.task_arg,
        tag=ns.tag,
    )


if __name__ == "__main__":
    main()
