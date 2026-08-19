#!/usr/bin/env python3
"""Generate production collection configs (``task_config/_run_<task>_<op>.yml``) + a manifest.

WHAT THIS GUARANTEES ("use the same config as the basic-success sweep")
----------------------------------------------------------------------
Both sweeps build their per-episode args with ``build_args(task, "demo_dynamic", ...)``
(``script/bench_script/record_demo.py``) and then apply a scenario:
  * conceptual : ``args["task_args"][task].update(SCENARIO_OVERRIDES[task][scenario])``
  * household  : ``args["use_dynamic"] = False`` (no scenario overrides)
Whether a given seed SUCCEEDS depends ONLY on those fields (task_args / use_dynamic /
embodiment / domain-randomization / dynamic params) + the seed -- never on rendering or
what gets saved. So this generator reproduces the sweep's args byte-for-byte and then
inverts ONLY the demo-only "don't save" flags so the collector actually writes data.
Net effect: a task/scenario that passes the sweep collects at that same success rate.

The demo_dynamic.yml ``task_args`` blocks carry HARD demo defaults (e.g.
place_block_belt.bowl_move_enabled: true). It is the SCENARIO_OVERRIDES.update() that
sets the real base/opt1/opt2 condition -- which is exactly why we reproduce the sweep
instead of dumping the raw yml.

INTENDED sweep-vs-collection deltas (none change whether a seed succeeds):
  * saving on: collect_data + export_lerobot, episode_num 50, save_failed_cases off
  * 3 camera views on (head + both wrists); sweep renders nothing
  * head camera back to D435 320x240 (build_args bumps it to Large_D435 for sharp demos)
  * data_type.third_view off (build_args turns it on for the legacy dual-view export)

Usage (repo root, robodyna env):
  python tools/gen_run_configs.py              # write all 81 _run_*.yml + manifest_collect.txt
  python tools/gen_run_configs.py --dry-run    # list the 81 combos, write nothing
  python tools/gen_run_configs.py --only place_block_belt:base   # one config to stdout-path
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
os.environ.pop("DISPLAY", None)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
sys.path[:0] = [
    ROOT,
    os.path.join(ROOT, "script"),
    os.path.join(ROOT, "script/bench_script"),
]

import yaml  # noqa: E402

from script.bench_script.record_demo import build_args  # noqa: E402
# TASKS + SCENARIO_OVERRIDES are the rename-correct, in-code opt spec the sweep validates.
# The sweeps live in tests/ (they were script_exp/ + script_hh_exp/ before aras-dev-final).
from tests._sweep_basic_success import SCENARIO_OVERRIDES, TASKS as CONCEPTUAL_TASKS  # noqa: E402
from tests._sweep_household_success import TASKS as _HH_SWEEP_TASKS  # noqa: E402

CONFIG = "demo_dynamic"  # the shared suite config both sweeps build on

# Household collection = exactly the household sweep's task set, base condition only
# (household tasks have no opt axis defined). serve_dinner was removed upstream.
HOUSEHOLD_TASKS = list(_HH_SWEEP_TASKS)

# build_args injects these; collect_data.main() recomputes every one of them from
# args["embodiment"], so we drop them to keep the emitted YAML small + plain-typed.
_INJECTED_KEYS = (
    "left_robot_file", "right_robot_file",
    "left_embodiment_config", "right_embodiment_config",
    "dual_arm_embodied", "embodiment_name", "embodiment_dis",
    "task_name", "task_config",
)


def make_config(task: str, op: str, family: str) -> dict:
    """Reproduce the sweep's args for (task, op), then flip only the save/camera flags."""
    args = build_args(task, CONFIG, "./data", option=None, task_arg_overrides=[])

    # ---- scenario: applied EXACTLY as the corresponding sweep applies it ----
    if family == "conceptual":
        scenario = "default" if op == "base" else op  # base->default, opt1->opt1, opt2->opt2
        overrides = SCENARIO_OVERRIDES[task][scenario]
        args.setdefault("task_args", {}).setdefault(task, {}).update(overrides)
    else:  # household
        args["use_dynamic"] = False  # matches _sweep_household_success.py

    # ---- trim task_args to just this task's block (others are provably ignored by
    #      collect_data, and this makes the emitted config human-auditable) ----
    task_block = args.get("task_args", {}).get(task, {})
    args["task_args"] = {task: task_block}

    # ---- invert the demo-only "don't save" flags -> real collection ----
    args["episode_num"] = 50
    args["save_freq"] = 15          # demo_dynamic default (~16.7 Hz); the collection standard
    args["collect_data"] = True
    # save_data governs PASS 1 ONLY -- collect_data.py:205 force-sets it True for pass 2.
    # Leave it False (the sweep's value) for three reasons: (a) _base_task._episode_step_count
    # counts saved FRAMES when save_data is on and ACTIONS when it's off, so a True here would
    # give pass 1 a different EPISODE_MAX_STEPS deadline than the sweep and break the
    # sweep-SR == collection-SR guarantee this generator exists to provide; (b) pass 1 would
    # otherwise render + cache every attempt, including the failures (~0.6 GB/episode of
    # transient .cache); (c) the Vulkan readback deadlock only bites pass 1 when it renders.
    args["save_data"] = False
    args["eval_video_log"] = False  # no eval-preview mp4 needed for a training corpus
    args["use_seed"] = False        # random/incrementing seeds -> 100 distinct successes
    args["save_failed_cases"] = False
    args["check_render_success"] = False  # same success gate as the sweep (plan & check_success)
    args["export_lerobot"] = True
    args["lerobot_root"] = f"./data_lerobot/prod_run/{task}__{op}"
    args["save_path"] = "./data"

    # ---- 3 camera views at D435 320x240 (override build_args' demo camera setup) ----
    cam = args.setdefault("camera", {})
    cam["collect_head_camera"] = True
    cam["collect_wrist_camera"] = True   # build_args turns wrists off; we need them
    cam["head_camera_type"] = "D435"     # build_args uses Large_D435 for demos; back to 320x240
    cam["wrist_camera_type"] = "D435"

    dt = args.setdefault("data_type", {})
    dt["rgb"] = True
    dt["third_view"] = False  # build_args turns this on; collection is head+2 wrists only

    for k in _INJECTED_KEYS:
        args.pop(k, None)
    return args


def combos():
    """Yield (task, op, family) for all 81 collection combos."""
    for t in CONCEPTUAL_TASKS:
        for op in ("base", "opt1", "opt2"):
            yield t, op, "conceptual"
    for t in HOUSEHOLD_TASKS:
        yield t, "base", "household"


def _dump(args: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(args, f, sort_keys=False, default_flow_style=False, allow_unicode=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="list combos, write nothing")
    ap.add_argument("--only", default=None,
                    help="generate a single 'task' or 'task:op' config and print its path")
    ap.add_argument("--out-dir", default="task_config", help="where _run_*.yml are written")
    ap.add_argument("--manifest", default="task_config/manifest_collect.txt")
    args_ns = ap.parse_args()

    all_combos = list(combos())

    if args_ns.dry_run:
        for t, op, fam in all_combos:
            print(f"{t} _run_{t}_{op} {fam} {op}")
        print(f"# total: {len(all_combos)} combos "
              f"({len(CONCEPTUAL_TASKS)} conceptual x3 + {len(HOUSEHOLD_TASKS)} household x1)")
        return

    os.makedirs(args_ns.out_dir, exist_ok=True)

    if args_ns.only:
        task = args_ns.only
        op = "base"
        if ":" in args_ns.only:
            task, op = args_ns.only.split(":", 1)
        fam = "conceptual" if task in CONCEPTUAL_TASKS else "household"
        cfg = make_config(task, op, fam)
        path = os.path.join(args_ns.out_dir, f"_run_{task}_{op}.yml")
        _dump(cfg, path)
        print(path)
        return

    manifest_lines = []
    for t, op, fam in all_combos:
        cfg = make_config(t, op, fam)
        cfg_name = f"_run_{t}_{op}"
        _dump(cfg, os.path.join(args_ns.out_dir, f"{cfg_name}.yml"))
        manifest_lines.append(f"{t} {cfg_name} {fam} {op}")

    with open(args_ns.manifest, "w", encoding="utf-8") as f:
        f.write("\n".join(manifest_lines) + "\n")

    print(f"wrote {len(manifest_lines)} configs to {args_ns.out_dir}/_run_*.yml")
    print(f"wrote manifest: {args_ns.manifest} ({len(manifest_lines)} lines)")


if __name__ == "__main__":
    main()
