#!/usr/bin/env python3
"""Capture fresh head-camera scene snapshots for GUI task cards.

Suite (base interactive) tasks write four condition stills::

    final_task_demos/<task>/scene_snapshot_default.png
    final_task_demos/<task>/scene_snapshot_opt1.png
    final_task_demos/<task>/scene_snapshot_opt2.png
    final_task_demos/<task>/scene_snapshot_opt1+2.png

plus ``scene_snapshot.png`` (copy of default) for compatibility.

Household GUI tasks write a single ``scene_snapshot.png``.

Each still is a single ``head_camera`` frame (no top-down montage).

Usage (repo root, robodyna env, headless Vulkan)::

    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json; unset DISPLAY
    python script/bench_script/publish_gui_snapshots.py
    python script/bench_script/publish_gui_snapshots.py cook_meat catch_cuboid
    python script/bench_script/publish_gui_snapshots.py --suite-only
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

sys.path.insert(0, "./")
sys.path.insert(0, "./script/bench_script")
sys.path.insert(0, "./script_exp")
sys.path.insert(0, "./script_hh_exp")

import numpy as np
from PIL import Image

from envs.utils.household_view import configure_standard_head_camera
from script.bench_script.record_demo import build_args
from script.collect_data import class_decorator
from script_exp.interactive_task_gui import (
    SCENARIO_OVERRIDES,
    SCENARIOS,
    TASKS as SUITE_TASKS,
)
from script_hh_exp.household_task_gui import TASKS as HOUSEHOLD_GUI_TASKS

ROOT = os.path.abspath(".")
FINAL = os.path.join(ROOT, "final_task_demos")
# Upscale head stills for crisp GUI cards (native D435 is typically 640×480).
SNAPSHOT_MAX_WIDTH = 1280
SETTLE_STEPS = 8


def _suite_tasks() -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for _label, name in SUITE_TASKS:
        if name not in seen:
            names.append(name)
            seen.add(name)
    return names


def _household_gui_tasks() -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for row in HOUSEHOLD_GUI_TASKS:
        name = row[1]
        if name not in seen:
            names.append(name)
            seen.add(name)
    return names


def _prepare_head_frame(head: np.ndarray) -> np.ndarray:
    """Return an RGB head frame, optionally upscaled to SNAPSHOT_MAX_WIDTH."""
    frame = np.asarray(head)
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError(f"Unexpected head RGB shape: {getattr(frame, 'shape', None)}")
    frame = frame[:, :, :3]
    h, w = int(frame.shape[0]), int(frame.shape[1])
    if w < SNAPSHOT_MAX_WIDTH:
        scale = SNAPSHOT_MAX_WIDTH / float(w)
        new_w = SNAPSHOT_MAX_WIDTH - (SNAPSHOT_MAX_WIDTH % 2)
        new_h = max(1, int(round(h * scale)))
        new_h += new_h % 2
        frame = np.asarray(
            Image.fromarray(frame).resize((new_w, new_h), Image.Resampling.LANCZOS)
        )
    return frame


def _overrides_to_cli(overrides: dict) -> list[str]:
    out: list[str] = []
    for key, value in overrides.items():
        if isinstance(value, bool):
            raw = "true" if value else "false"
        else:
            raw = str(value)
        out.append(f"{key}={raw}")
    return out


def capture_snapshot(
    task_name: str,
    *,
    seed: int = 0,
    scenario: str | None = None,
    task_arg_overrides: list[str] | None = None,
    out_name: str = "scene_snapshot.png",
) -> str:
    save_root = os.path.abspath(f"./tmp/tmp_{task_name}_snapshot")
    os.makedirs(save_root, exist_ok=True)

    args = build_args(
        task_name,
        config_name="demo_dynamic",
        save_root=save_root,
        option=None,
        task_arg_overrides=task_arg_overrides,
    )
    args.update(
        collect_data=False,
        save_data=False,
        eval_video_log=False,
        need_plan=False,
        render_freq=0,
        episode_num=1,
        use_seed=True,
    )
    args.setdefault("data_type", {})
    args["data_type"]["rgb"] = True
    args["data_type"]["third_view"] = False

    task = class_decorator(task_name)
    _orig_setup = task.setup_demo

    def _setup_demo(**kwargs):
        _orig_setup(**kwargs)
        # Same elevated head pose for base suite + household GUI cards.
        configure_standard_head_camera(task)

    task.setup_demo = _setup_demo
    task.setup_demo(now_ep_num=0, seed=int(seed), **args)

    # Let physics settle so floating/resting objects look natural.
    for _ in range(SETTLE_STEPS):
        try:
            task._update_render()
            if getattr(task, "scene", None) is not None:
                task.scene.step()
        except Exception:
            break

    obs = task.get_obs()
    head = _prepare_head_frame(obs["observation"]["head_camera"]["rgb"])

    dest_dir = os.path.join(FINAL, task_name)
    os.makedirs(dest_dir, exist_ok=True)
    out_path = os.path.join(dest_dir, out_name)
    Image.fromarray(head).save(out_path, optimize=True)
    tag = f" [{scenario}]" if scenario else ""
    print(f"  WROTE{tag} {out_path} ({head.shape[1]}x{head.shape[0]})", flush=True)

    try:
        task.close_env(clear_cache=True)
    except Exception:
        try:
            task.close_env()
        except Exception:
            pass
    return out_path


def publish_suite_task(task_name: str, seed: int = 0) -> None:
    if task_name not in SCENARIO_OVERRIDES:
        raise KeyError(f"No SCENARIO_OVERRIDES for suite task {task_name}")
    for scenario in SCENARIOS:
        overrides = SCENARIO_OVERRIDES[task_name][scenario]
        out = capture_snapshot(
            task_name,
            seed=seed,
            scenario=scenario,
            task_arg_overrides=_overrides_to_cli(overrides),
            out_name=f"scene_snapshot_{scenario}.png",
        )
        if scenario == "default":
            # Keep the legacy single-file name in sync with Default.
            legacy = os.path.join(FINAL, task_name, "scene_snapshot.png")
            shutil.copy2(out, legacy)
            print(f"  WROTE [legacy] {legacy}", flush=True)


def publish_household_task(task_name: str, seed: int = 0) -> None:
    capture_snapshot(task_name, seed=seed, out_name="scene_snapshot.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "tasks",
        nargs="*",
        help="Optional task name filter (default: all selected suite/household tasks).",
    )
    parser.add_argument(
        "--suite-only",
        action="store_true",
        help="Only publish base interactive (suite) four-condition snapshots.",
    )
    parser.add_argument(
        "--household-only",
        action="store_true",
        help="Only publish household single snapshots.",
    )
    ns = parser.parse_args()
    only = set(ns.tasks) if ns.tasks else None

    suite = [] if ns.household_only else _suite_tasks()
    household = [] if ns.suite_only else _household_gui_tasks()
    # Avoid double-publishing tasks that appear in both lists.
    household = [t for t in household if t not in set(suite)]

    failed: list[str] = []
    for task in suite:
        if only and task not in only:
            continue
        print(f"\n=== {task} (4 scenarios) ===", flush=True)
        try:
            publish_suite_task(task, seed=0)
        except Exception as exc:
            print(f"FAILED {task}: {exc}", flush=True)
            failed.append(task)

    for task in household:
        if only and task not in only:
            continue
        print(f"\n=== {task} ===", flush=True)
        try:
            publish_household_task(task, seed=0)
        except Exception as exc:
            print(f"FAILED {task}: {exc}", flush=True)
            failed.append(task)

    if failed:
        print("\nFailed tasks:", ", ".join(failed), flush=True)
        return 1
    print("\nAll GUI snapshots published.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
