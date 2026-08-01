#!/usr/bin/env python3
"""Seed sweep for catch_cup.

Also reports how far every dynamic décor prop drifted from its spawn pose, which
is how we tell real cup/prop contacts apart from spawn-overlap explosions.
"""
from __future__ import annotations

import os
import sys
import traceback

import numpy as np

sys.path.insert(0, "./")
sys.path.insert(0, "./script/bench_script")

from script.bench_script.record_demo import build_args
from script.collect_data import class_decorator

TASK = "catch_cup"
CONFIG = "demo_dynamic"


def _spawn_poses(env):
    out = []
    for a in env.decor:
        try:
            out.append(np.array(a.get_pose().p, dtype=np.float64))
        except Exception:
            out.append(None)
    return out


def run_seed(seed: int) -> dict:
    save_root = os.path.abspath(f"./tmp_{TASK}_test")
    os.makedirs(save_root, exist_ok=True)
    args = build_args(TASK, CONFIG, save_root, option=None, task_arg_overrides=[])
    args["collect_data"] = False
    args["save_data"] = False
    args["eval_video_log"] = False
    args["need_plan"] = True
    args["render_freq"] = 0
    args["episode_num"] = 1
    args["check_render_success"] = False
    args["use_dynamic"] = False

    env = class_decorator(TASK)
    row = {"seed": seed, "plan": False, "check": False, "err": None}
    try:
        env.setup_demo(now_ep_num=0, seed=seed, **args)
        start = _spawn_poses(env)
        env.play_once()
        row["plan"] = bool(env.plan_success)
        row["check"] = bool(env.check_success())
        row["state"] = env._cup_state
        row["pillow_moved"] = round(
            float(np.linalg.norm(env._pillow_xy() - env.pillow_start[:2])), 4
        )
        drift = []
        for a, p0 in zip(env.decor, start):
            if p0 is None:
                continue
            p1 = np.array(a.get_pose().p, dtype=np.float64)
            drift.append((a.get_name(), round(float(np.linalg.norm(p1 - p0)), 4)))
        row["decor_drift"] = drift
        row["decor_max_drift"] = max((d for _, d in drift), default=0.0)
    except Exception as e:  # noqa: BLE001
        row["err"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        try:
            env.close_env()
        except Exception:
            pass
    return row


def main() -> None:
    seeds = [int(s) for s in (sys.argv[1:] or range(6))]
    rows = []
    for s in seeds:
        r = run_seed(s)
        rows.append(r)
        print(f"  -> {r}", flush=True)
    n_ok = sum(1 for r in rows if r["plan"] and r["check"])
    print(f"\n{TASK}: {n_ok}/{len(rows)} successes", flush=True)


if __name__ == "__main__":
    main()
