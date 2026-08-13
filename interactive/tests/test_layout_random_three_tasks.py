#!/usr/bin/env python3
"""10-seed layout+expert smoke for pour_beer / cook_food / measure_ingredient.

Then records 3 dual-view demos per task from successful seeds for inspection.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import subprocess
import sys
import time
import traceback

import torch

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("./script/bench_script"))

from script.bench_script.record_demo import (
    build_args,
    configure_topdown_camera,
    merge_dual_view_videos,
    next_version,
    _cleanup_scratch,
)
from script.collect_data import class_decorator, run

TASKS = ("pour_beer", "cook_food", "measure_ingredient")
CONFIG = "demo_dynamic"
N_SEEDS = 10
N_DEMOS = 3
MIN_FREE_MIB = 1800


def gpu_free_mib() -> int:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip().split()[0]
        return int(out)
    except Exception:
        return -1


def wait_for_gpu(tag: str = "", min_free: int = MIN_FREE_MIB) -> None:
    for attempt in range(1, 60):
        free = gpu_free_mib()
        print(f"[gpu-wait {tag}] attempt {attempt} free={free}MiB", flush=True)
        if free < 0 or free >= min_free:
            return
        time.sleep(20)


def layout_detail(task: str, env) -> dict:
    if task == "pour_beer":
        return {
            "cup": list(map(float, env.cup_xy)),
            "tap": list(map(float, env.tap_xy)),
            "n_props": len(getattr(env, "_bar_props", []) or []),
        }
    if task == "cook_food":
        return {
            "stove": list(map(float, getattr(env, "range_xy", (0, 0)))),
            "stove_pose": getattr(env, "_stove_pose_choice", "?"),
            "burner": str(getattr(env, "burner_name", "")),
            "handle_yaw": float(getattr(env, "skillet_handle_yaw", 0.0)),
            "board": list(map(float, getattr(env, "board_xy", (0, 0)))),
            "plate": list(map(float, getattr(env, "plate_xy", (0, 0)))),
            "decor": list(map(float, getattr(env, "decor_plate_xy", (0, 0)))),
            "food_type": str(getattr(env, "food_type", "")),
        }
    return {
        "station": list(map(float, env.jar_xy)),
        "dispenser": list(map(float, env.dispenser_xy)),
        "scale": list(map(float, env.scale_xy)),
        "microwave": (
            None
            if getattr(env, "microwave_xy_override", None) is None
            else list(map(float, env.microwave_xy_override))
        ),
        "mw_pose": getattr(env, "_microwave_pose_choice", "?"),
        "arm": str(getattr(env, "arm", "")),
        "decor_on_mw": [
            n
            for n, d in (getattr(env, "_decor_layout", {}) or {}).items()
            if d.get("on_microwave")
        ],
    }


def run_seed(task: str, seed: int) -> dict:
    save_root = os.path.abspath(f"./tmp/tmp_{task}_layout_test")
    os.makedirs(save_root, exist_ok=True)
    args = build_args(task, CONFIG, save_root, None, ["randomize_layout=true"])
    args.update(
        collect_data=False,
        save_data=False,
        eval_video_log=False,
        need_plan=True,
        render_freq=0,
        save_freq=None,
        check_render_success=False,
        episode_num=1,
    )
    # Pin meat for a stable cook_food smoke (onion/sausage add variance).
    if task == "cook_food":
        targs = args.setdefault("task_args", {}).setdefault(task, {})
        targs["food_type"] = "meat"

    env = class_decorator(task)
    ok = False
    detail: dict = {"task": task, "seed": seed}
    try:
        env.setup_demo(now_ep_num=0, seed=int(seed), **args)
        detail.update(layout_detail(task, env))
        print(f"[{task}] seed={seed} layout={detail}", flush=True)
        env.play_once()
        plan = bool(getattr(env, "plan_success", False))
        succ = bool(env.check_success())
        ok = plan and succ
        detail.update({"plan_success": plan, "check_success": succ, "ok": ok})
        print(f"[{task}] seed={seed} RESULT ok={ok} plan={plan} succ={succ}", flush=True)
    except Exception as e:
        detail["error"] = f"{type(e).__name__}: {e}"
        detail["ok"] = False
        traceback.print_exc()
        print(f"[{task}] seed={seed} RESULT ok=False error={detail['error']}", flush=True)
    finally:
        try:
            env.close_env(clear_cache=True)
        except Exception:
            try:
                env.close_env()
            except Exception:
                pass
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    return detail


def record_with_seed(task: str, seed: int, tag: str) -> dict:
    """Record one dual-view demo at a fixed seed via the standard collector."""
    save_root = os.path.abspath(f"./tmp/tmp_{task}")
    video_dir = os.path.join(save_root, "video")
    ver = next_version(video_dir)
    stem = f"v{ver}_{tag}"
    out_head = os.path.join(video_dir, f"{stem}_head.mp4")
    out_topdown = os.path.join(video_dir, f"{stem}_topdown.mp4")
    out_side = os.path.join(video_dir, f"{stem}_sidebyside.mp4")

    _cleanup_scratch(save_root, task)
    os.makedirs(save_root, exist_ok=True)

    args = build_args(task, CONFIG, save_root, None, ["randomize_layout=true"])
    args["episode_num"] = 1
    args["save_failed_cases"] = False
    args["check_render_success"] = True
    args["use_seed"] = False

    task_env = class_decorator(task)
    _orig_setup = task_env.setup_demo
    attempts = {"n": 0}

    def _setup_demo(**kwags):
        # First attempt (plan) and second (render) both use the forced seed.
        # If the collector retries (epid++), keep forcing the same seed so we
        # never silently fall through to a different layout.
        kwags["seed"] = int(seed)
        attempts["n"] += 1
        _orig_setup(**kwags)
        configure_topdown_camera(task_env)

    task_env.setup_demo = _setup_demo

    def _merge_dual(self):
        if not self.save_data:
            return
        cache_path = f"{self.save_dir}/.cache/episode{self.ep_num}/"
        fps = 250.0 / float(self.save_freq) if self.save_freq else 15.0
        merge_dual_view_videos(cache_path, out_head, out_topdown, out_side, fps=fps)

    task_env.merge_pkl_to_hdf5_video = _merge_dual.__get__(task_env, task_env.__class__)

    run(task_env, args)
    _cleanup_scratch(save_root, task)

    if not (os.path.isfile(out_head) and os.path.isfile(out_topdown)):
        raise RuntimeError(f"demo videos missing for {task} seed={seed}")

    print("Demo ready:", out_head, out_topdown, flush=True)
    return {
        "version": ver,
        "tag": tag,
        "head": out_head,
        "topdown": out_topdown,
        "sidebyside": out_side,
        "setup_calls": attempts["n"],
    }


def record_task_demos(task: str, seeds: list[int], n_ok: int = N_DEMOS) -> list[dict]:
    outs = []
    for seed in seeds:
        if sum(1 for o in outs if o.get("ok")) >= n_ok:
            break
        wait_for_gpu(tag=f"{task}-demo-{seed}")
        tag = f"layout_s{seed}"
        print(f"\n======== RECORD {task} seed={seed} tag={tag} ========", flush=True)
        try:
            info = record_with_seed(task, seed, tag)
            outs.append({"task": task, "seed": seed, "ok": True, **info})
        except Exception as e:
            traceback.print_exc()
            outs.append({"task": task, "seed": seed, "ok": False, "error": str(e)})
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    return outs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=list(TASKS))
    parser.add_argument("--n-seeds", type=int, default=N_SEEDS)
    parser.add_argument("--n-demos", type=int, default=N_DEMOS)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--skip-record", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    ns = parser.parse_args()

    all_results = {}
    demo_results = {}

    for task in ns.tasks:
        wait_for_gpu(tag=task)
        seeds = list(ns.seeds) if ns.seeds is not None else list(range(ns.n_seeds))
        if not ns.skip_test:
            print(f"\n########## TEST {task} seeds={seeds} ##########", flush=True)
            results = []
            for seed in seeds:
                wait_for_gpu(tag=f"{task}-s{seed}")
                results.append(run_seed(task, seed))
            n_ok = sum(1 for r in results if r.get("ok"))
            print(f"[{task}] passed {n_ok}/{len(results)}", flush=True)
            all_results[task] = results
        else:
            all_results[task] = [{"seed": s, "ok": True} for s in seeds]

        if not ns.skip_record:
            ok_seeds = [r["seed"] for r in all_results[task] if r.get("ok")]
            pool = ok_seeds if ok_seeds else list(seeds)
            pick = pool[:]
            random.Random(0).shuffle(pick)
            # Keep a longer candidate list so record can skip failures.
            extra = 0
            while len(pick) < max(ns.n_demos * 3, ns.n_demos):
                cand = (max(seeds) if seeds else 0) + 1 + extra
                if cand not in pick:
                    pick.append(cand)
                extra += 1
            demo_results[task] = record_task_demos(task, pick, n_ok=ns.n_demos)

    out_path = os.path.abspath("./tmp/tmp_layout_random_three_tasks_summary.json")
    with open(out_path, "w") as f:
        json.dump({"tests": all_results, "demos": demo_results}, f, indent=2, default=str)
    print(f"\nWrote {out_path}", flush=True)

    print("\n======== FINAL SUMMARY ========", flush=True)
    rc = 0
    for task, results in all_results.items():
        n_ok = sum(1 for r in results if r.get("ok"))
        print(f"{task}: {n_ok}/{len(results)} seeds ok", flush=True)
        for r in results:
            print(
                f"  seed={r.get('seed')} ok={r.get('ok')} "
                f"layout_keys={[k for k in r if k not in ('task','seed','ok','plan_success','check_success','error')]} "
                f"err={r.get('error')}",
                flush=True,
            )
        if n_ok < max(1, len(results) // 2):
            rc = 1
    for task, demos in demo_results.items():
        for d in demos:
            print(
                f"demo {task} seed={d.get('seed')} ok={d.get('ok')} "
                f"head={d.get('head')} top={d.get('topdown')}",
                flush=True,
            )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
