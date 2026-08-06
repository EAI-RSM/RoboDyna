#!/usr/bin/env python3
"""Seed sweep for make_soup: layout randomization + success (all veggies in pot)."""
from __future__ import annotations

import os
import sys
import traceback

import numpy as np

sys.path.insert(0, "./")
sys.path.insert(0, "./script/bench_script")

from script.bench_script.record_demo import build_args
from script.collect_data import class_decorator

TASK = "make_soup"
CONFIG = "demo_kitchens"


def run_seed(seed: int) -> dict:
    save_root = os.path.abspath(f"./tmp/tmp_{TASK}_test")
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
    row = {
        "seed": seed,
        "plan": False,
        "check": False,
        "in_pot": 0,
        "fallen": False,
        "stove_on": False,
        "err": None,
    }
    try:
        env.setup_demo(now_ep_num=0, seed=seed, **args)
        mw = getattr(env, "microwave", None)
        row["microwave"] = mw is not None
        row["range_xy"] = [float(v) for v in np.asarray(env.range_xy, dtype=float)]
        row["board_xy"] = [float(v) for v in np.asarray(env.board_xy, dtype=float)]
        row["pot_xy"] = [float(v) for v in np.asarray(env.pot_xy, dtype=float)]
        # Pot must stay on the same burner relative to the stove.
        burner = np.asarray(env.burner_xy, dtype=float)
        pot = np.asarray(env.pot_xy, dtype=float)
        row["pot_on_burner"] = bool(np.linalg.norm(burner - pot) < 1e-3)
        print(
            f"[seed {seed}] range={np.round(row['range_xy'], 3)} "
            f"board={np.round(row['board_xy'], 3)} pot={np.round(row['pot_xy'], 3)} "
            f"microwave={row['microwave']} pot_on_burner={row['pot_on_burner']}",
            flush=True,
        )
        env.play_once()
        row["plan"] = bool(env.plan_success)
        row["check"] = bool(env.check_success())
        row["in_pot"] = int(sum(env._veg_in_pot(v) for v in env.veggies))
        row["fallen"] = bool(env._veg_fallen)
        row["stove_on"] = bool(getattr(env, "stove_on", False))
        pot_xy = env.pot_xy
        for v in env.veggies:
            p = v.get_pose().p
            print(
                f"     {v.get_name():12s} d_pot="
                f"{((p[0]-pot_xy[0])**2 + (p[1]-pot_xy[1])**2) ** 0.5:.3f} "
                f"z={p[2]:.3f} in={env._veg_in_pot(v)}"
            )
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
    seeds = [int(s) for s in (sys.argv[1:] or range(5))]
    rows = []
    for s in seeds:
        r = run_seed(s)
        rows.append(r)
        print(f"  -> {r}", flush=True)
    n_ok = sum(1 for r in rows if r["plan"] and r["check"] and not r.get("err"))
    n_mw = sum(1 for r in rows if r.get("microwave"))
    n_pot = sum(1 for r in rows if r.get("pot_on_burner"))
    xs = [r["range_xy"][0] for r in rows if r.get("range_xy")]
    print(
        f"\nmake_soup: {n_ok}/{len(rows)} successes | "
        f"microwave_present={n_mw} | pot_on_burner={n_pot}/{len(rows)} | "
        f"range_x={np.round(xs, 3) if xs else []}",
        flush=True,
    )


if __name__ == "__main__":
    main()
