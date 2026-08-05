#!/usr/bin/env python3
"""Record specific seeds for make_soup / catch_cup to dual-view demos."""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "./")
sys.path.insert(0, "./script/bench_script")

from script.bench_script.record_demo import (
    build_args,
    configure_topdown_camera,
    merge_dual_view_videos,
    next_version,
    _cleanup_scratch,
)
from script.collect_data import class_decorator


def record_one(task: str, seed: int, config: str = "demo_dynamic", save_freq: int = 8) -> dict:
    save_root = os.path.abspath(f"./tmp_{task}")
    video_dir = os.path.join(save_root, "video")
    os.makedirs(video_dir, exist_ok=True)
    ver = next_version(video_dir)
    print(f"=== {task} seed={seed} ver={ver} ===", flush=True)
    _cleanup_scratch(save_root, task)
    os.makedirs(save_root, exist_ok=True)

    args = build_args(task, config, save_root, None, [])
    args.update(
        need_plan=True,
        save_data=True,
        collect_data=False,
        check_render_success=False,
        save_failed_cases=True,
        render_freq=0,
        save_freq=int(save_freq),
        use_seed=False,
    )
    # Prefer the lighter D435 so demos still fit when the GPU is busy.
    args.setdefault("camera", {})["head_camera_type"] = "D435"

    env = class_decorator(task)
    _orig = env.setup_demo

    def _setup(**kw):
        _orig(**kw)
        configure_topdown_camera(env)

    env.setup_demo = _setup
    shutil.rmtree(Path(save_root) / ".cache", ignore_errors=True)
    env.setup_demo(now_ep_num=0, seed=seed, **args)
    env.play_once()
    ok = bool(env.plan_success and env.check_success())
    label = "pass" if ok else "fail"
    print(f"RESULT {task} seed={seed} {label}", flush=True)

    stem = f"v{ver}_seed{seed}_{label}"
    outs = {
        k: os.path.join(video_dir, f"{stem}_{k}.mp4")
        for k in ("head", "topdown", "sidebyside")
    }
    cache_path = f"{env.save_dir}/.cache/episode{env.ep_num}/"
    fps = 250.0 / float(env.save_freq)
    merge_dual_view_videos(
        cache_path, outs["head"], outs["topdown"], outs["sidebyside"], fps=fps
    )
    try:
        env.close_env(clear_cache=True)
    except Exception:
        pass
    _cleanup_scratch(save_root, task)
    return {"version": ver, "seed": seed, "ok": ok, **outs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("seeds", nargs="+", type=int)
    ap.add_argument("--config", default="demo_dynamic")
    ns = ap.parse_args()
    for s in ns.seeds:
        record_one(ns.task, s, config=ns.config)


if __name__ == "__main__":
    main()
