#!/usr/bin/env python3
"""10-seed layout-randomization smoke for make_soup + catch_cup."""
from __future__ import annotations

import os
import sys
import traceback

import numpy as np

sys.path.insert(0, "./")
sys.path.insert(0, "./script/bench_script")

from script.bench_script.record_demo import build_args
from script.collect_data import class_decorator


def run_seed(task: str, seed: int, config: str = "demo_dynamic") -> dict:
    save_root = os.path.abspath(f"./tmp_{task}_layout_test")
    os.makedirs(save_root, exist_ok=True)
    args = build_args(task, config, save_root, option=None, task_arg_overrides=[])
    args["collect_data"] = False
    args["save_data"] = False
    args["eval_video_log"] = False
    args["need_plan"] = True
    args["render_freq"] = 0
    args["episode_num"] = 1
    args["check_render_success"] = False
    args["use_dynamic"] = False

    env = class_decorator(task)
    row: dict = {"task": task, "seed": seed, "plan": False, "check": False, "err": None}
    try:
        env.setup_demo(now_ep_num=0, seed=seed, **args)
        if task == "make_soup":
            row["slot"] = getattr(env, "_range_slot", None)
            row["burner"] = getattr(env, "burner_name", None)
            row["arm"] = str(getattr(env, "arm", ""))
            row["board"] = list(np.round(np.asarray(env.board_xy, dtype=float), 3))
            row["range"] = list(np.round(np.asarray(env.range_xy, dtype=float), 3))
            row["pot"] = list(np.round(np.asarray(env.pot_xy, dtype=float), 3))
        else:
            row["arm"] = str(getattr(env, "arm_side", ""))
            row["cup_x"] = round(float(env.cup_start[0]), 3)
            row["drop_x"] = round(float(env.drop_x), 3)
            row["roll_speed"] = round(float(env.roll_speed), 4)
            row["n_shelf_cups"] = int(len(getattr(env, "shelf_cups", [])) + 1)
            row["n_decor"] = int(len(getattr(env, "decor", [])))
        env.play_once()
        row["plan"] = bool(env.plan_success)
        row["check"] = bool(env.check_success())
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
    tasks = sys.argv[1:] or ["make_soup", "catch_cup"]
    seeds = list(range(10))
    for task in tasks:
        print(f"\n===== {task}: seeds {seeds[0]}..{seeds[-1]} =====")
        rows = []
        for s in seeds:
            row = run_seed(task, s)
            rows.append(row)
            ok = "OK" if row["check"] else ("PLAN" if row["plan"] else "FAIL")
            extra = {k: v for k, v in row.items() if k not in ("task", "seed", "plan", "check", "err")}
            print(f"  seed={s:2d} [{ok}] {extra}" + (f" err={row['err']}" if row["err"] else ""))
        n_ok = sum(1 for r in rows if r["check"])
        n_plan = sum(1 for r in rows if r["plan"])
        print(f"  summary: check={n_ok}/10 plan={n_plan}/10")


if __name__ == "__main__":
    main()
