#!/usr/bin/env python3
"""One-off: screenshot sort_apples_belt plasticbox instances.

Forces basket_id via monkeypatch of np.random.choice (no permanent env edits).
Does not modify BASKET_INSTANCE_IDS — can force IDs outside the sampling pool.
"""
from __future__ import annotations

import os
import shutil
import sys

sys.path.insert(0, "./")
sys.path.insert(0, "./script/bench_script")

import cv2
import numpy as np

from script.bench_script.record_demo import (
    build_args,
    configure_topdown_camera,
)
from script.collect_data import class_decorator

TASK = "sort_apples_belt"
CONFIG = "demo_dynamic"
INSTANCE_IDS = (0, 2, 3, 6)

TMP_OUT = "/tmp/robodyna_demos"
REPO_OUT = os.path.abspath("final_task_demos/sort_apples_belt")
SAVE_ROOT = os.path.abspath(f"./tmp_{TASK}_plasticbox_snap")


def _force_choice(forced_id: int):
    """Return forced_id when choice is over the task's BASKET_INSTANCE_IDS pool."""
    orig = np.random.choice

    def choice(a, size=None, replace=True, p=None):
        try:
            arr = np.asarray(a).reshape(-1).tolist()
            # Import late so path hacks above have taken effect.
            from envs.sort_apples_belt import sort_apples_belt

            pool = list(sort_apples_belt.BASKET_INSTANCE_IDS)
            if size is None and arr == pool:
                return int(forced_id)
        except Exception:
            pass
        return orig(a, size=size, replace=replace, p=p)

    np.random.choice = choice
    return orig


def _save_rgb(path: str, rgb: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)
    print(f"Wrote {path}  shape={bgr.shape}", flush=True)


def snap_one(basket_id: int) -> dict:
    os.makedirs(SAVE_ROOT, exist_ok=True)
    args = build_args(TASK, CONFIG, SAVE_ROOT, option=None, task_arg_overrides=None)
    # Default view (no Opt2 dustbin clutter); still collect RGB + third_view.
    args["collect_data"] = False
    args["save_data"] = False
    args["eval_video_log"] = False
    args["need_plan"] = False
    args["render_freq"] = 0
    args["episode_num"] = 1
    args["check_render_success"] = False

    orig_choice = _force_choice(basket_id)
    env = class_decorator(TASK)
    try:
        env.setup_demo(now_ep_num=0, seed=0, **args)
        configure_topdown_camera(env)
        assert int(env.basket_id) == int(basket_id), (
            f"basket_id force failed: got {env.basket_id}, want {basket_id}"
        )

        # Settle a few frames so lighting/actors are stable.
        for _ in range(5):
            env.scene.step()
            env._update_render()

        env.cameras.update_picture()
        head = env.cameras.get_rgb()["head_camera"]["rgb"]
        top = env.cameras.get_observer_rgb()

        paths = {}
        for view, img in (("head", head), ("topdown", top)):
            tmp_p = os.path.join(TMP_OUT, f"{TASK}_plasticbox_id{basket_id}_{view}.png")
            repo_p = os.path.join(REPO_OUT, f"plasticbox_id{basket_id}_{view}.png")
            _save_rgb(tmp_p, img)
            _save_rgb(repo_p, img)
            paths[view] = {"tmp": tmp_p, "repo": repo_p}
        print(f"OK basket_id={basket_id}", flush=True)
        return paths
    finally:
        np.random.choice = orig_choice
        try:
            env.close_env(clear_cache=True)
        except Exception:
            pass


def main():
    os.makedirs(TMP_OUT, exist_ok=True)
    os.makedirs(REPO_OUT, exist_ok=True)
    all_paths = {}
    for bid in INSTANCE_IDS:
        all_paths[bid] = snap_one(bid)

    # Cleanup scratch episode folder.
    if os.path.isdir(SAVE_ROOT):
        shutil.rmtree(SAVE_ROOT, ignore_errors=True)

    print("\n=== Absolute paths ===")
    for bid in INSTANCE_IDS:
        for view in ("topdown", "head"):
            print(all_paths[bid][view]["tmp"])
            print(all_paths[bid][view]["repo"])


if __name__ == "__main__":
    main()
