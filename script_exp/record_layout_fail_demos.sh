#!/usr/bin/env bash
# Record known failing seeds from the layout 10-seed runs (topdown+sidebyside via record_demo pipeline).
set -euo pipefail
cd /home/xuan/Desktop/RoboReal/RoboDynaExp
source /home/xuan/miniconda3/etc/profile.d/conda.sh
conda activate robodyna
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
unset DISPLAY
export PYTHONPATH="./:./script/bench_script:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

python -u - <<'PY'
import gc
import os
import shutil
import sys

import torch

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("./script/bench_script"))

from script.bench_script.record_demo import (
    _cleanup_scratch,
    build_args,
    configure_topdown_camera,
    merge_dual_view_videos,
    next_version,
)
from script.collect_data import class_decorator, run

OUT_ROOT = os.path.abspath("./rand_demos")


def record_fail(task: str, seed: int, tag: str, task_args: list[str]) -> str:
    save_root = os.path.abspath(f"./tmp_{task}")
    video_dir = os.path.join(save_root, "video")
    ver = next_version(video_dir)
    stem = f"v{ver}_{tag}"
    out_head = os.path.join(video_dir, f"{stem}_head.mp4")
    out_topdown = os.path.join(video_dir, f"{stem}_topdown.mp4")
    out_side = os.path.join(video_dir, f"{stem}_sidebyside.mp4")

    _cleanup_scratch(save_root, task)
    os.makedirs(save_root, exist_ok=True)

    overrides = ["randomize_layout=true", "allow_fail=true", *task_args]
    args = build_args(task, "demo_dynamic", save_root, None, overrides)
    args["episode_num"] = 1
    args["save_failed_cases"] = True
    args["check_render_success"] = False
    args["use_seed"] = False
    if task == "cook_food":
        args.setdefault("task_args", {}).setdefault(task, {})["food_type"] = "meat"

    task_env = class_decorator(task)
    _orig_setup = task_env.setup_demo

    def _setup_demo(**kwags):
        kwags["seed"] = int(seed)
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

    print(f"======== FAIL RECORD {task} seed={seed} tag={tag} ========", flush=True)
    run(task_env, args)
    _cleanup_scratch(save_root, task)

    if not (os.path.isfile(out_topdown) and os.path.isfile(out_side)):
        raise RuntimeError(f"fail videos missing for {task} seed={seed}: {out_topdown}")

    dest_dir = os.path.join(OUT_ROOT, task)
    os.makedirs(dest_dir, exist_ok=True)
    label = f"fail_s{seed}"
    for src, name in ((out_topdown, f"{label}_topdown.mp4"), (out_side, f"{label}_sidebyside.mp4")):
        dst = os.path.join(dest_dir, name)
        shutil.copy2(src, dst)
        print(f"copied {dst}", flush=True)
    return stem


# From the original 10-seed layout runs:
# pour_beer seed 0 / 5 overflowed (right-side station at that time)
# cook_food seed 2 completed with check_success=False
record_fail("pour_beer", 0, "layout_fail_s0", ["station_side=right"])
gc.collect(); torch.cuda.empty_cache()
record_fail("pour_beer", 5, "layout_fail_s5", ["station_side=right"])
gc.collect(); torch.cuda.empty_cache()
record_fail("cook_food", 2, "layout_fail_s2", ["stove_pose=center", "burner=left_front"])
print("DONE", flush=True)
PY
