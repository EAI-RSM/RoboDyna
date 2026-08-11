#!/usr/bin/env python3
"""Five-episode stop_ball sweep with forced roll headings + one miss check.

Success criterion: gripper grasps the moving ball. Without the arm, the ball
must fall off the table.
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

TASK = "stop_ball"
CONFIG = "demo_dynamic"

# Diverse seeds + forced headings (rad from −Y): front / L / R / steep L / steep R.
CASES = [
    {"seed": 0, "angle": 0.00, "label": "front"},
    {"seed": 3, "angle": -0.35, "label": "left-mild"},
    {"seed": 8, "angle": 0.35, "label": "right-mild"},
    {"seed": 12, "angle": -0.60, "label": "left-steep"},
    {"seed": 16, "angle": 0.55, "label": "right-steep"},
]


def _make_args(save_root: str):
    args = build_args(TASK, CONFIG, save_root, option=None, task_arg_overrides=[])
    args["collect_data"] = False
    args["save_data"] = False
    args["eval_video_log"] = False
    args["need_plan"] = True
    args["render_freq"] = 0
    args["episode_num"] = 1
    args["check_render_success"] = False
    args["use_dynamic"] = False
    return args


def run_case(case: dict, miss: bool = False) -> dict:
    save_root = os.path.abspath(f"./tmp/tmp_{TASK}_test")
    os.makedirs(save_root, exist_ok=True)
    args = _make_args(save_root)
    seed = int(case["seed"])
    angle = float(case["angle"])
    # Keep spawn / arm on the same half as the forced heading so feasibility
    # does not silently flip a right roll onto the left arm.
    if abs(angle) < 1e-6:
        side = "left" if int(seed) % 2 == 0 else "right"
    else:
        side = "right" if angle > 0.0 else "left"
    args.setdefault("task_args", {}).setdefault(TASK, {})["arm_side"] = side

    env = class_decorator(TASK)
    row = {
        "seed": seed,
        "label": case["label"],
        "forced_angle": angle,
        "miss": miss,
        "plan": False,
        "check": False,
        "err": None,
    }
    try:
        orig_setup = env.setup_demo

        def setup_demo(**kwags):
            orig_load = env.load_actors

            def load_actors():
                orig_build = env._build_trajectory
                orig_feasible = env._feasible_angle

                def build():
                    env._sample_roll_heading = lambda: (
                        float(angle),
                        np.array([np.sin(angle), -np.cos(angle)], dtype=np.float64),
                    )
                    # Keep the forced heading (still clamp only if unreachable).
                    env._feasible_angle = lambda ang, *a, **k: float(ang)
                    try:
                        return orig_build()
                    finally:
                        env._feasible_angle = orig_feasible

                env._build_trajectory = build
                return orig_load()

            env.load_actors = load_actors
            return orig_setup(**kwags)

        env.setup_demo = setup_demo
        env.setup_demo(now_ep_num=0, seed=seed, **args)

        if miss:
            # Skip the arm: release the ball and let the scripted path run out.
            env._release_ball()
            while env._traj_step < len(env._traj) and env._ball_state == "rolling":
                env._dwell(1)
                if env._fell_off or env._ball_state == "fallen":
                    break
            for _ in range(int(getattr(env, "max_live_steps", 520))):
                if env._ball_state not in ("rolling", "live"):
                    break
                env._dwell(1)
        else:
            env.play_once()

        row["plan"] = bool(env.plan_success)
        row["check"] = bool(env.check_success())
        row["ball_state"] = str(env._ball_state)
        row["fell_off"] = bool(env._fell_off)
        row["grasped"] = bool(getattr(env, "_grasped", False))
        row["welded"] = bool(getattr(env, "_welded", False))
        row["arm"] = str(env.arm_side)
        row["roll_angle"] = round(float(env._roll_angle), 3)
        row["exit_edge"] = str(env._exit_edge)
        c = env._ball_centre()
        row["ball_xyz"] = [round(float(x), 3) for x in c]
        row["ball_speed"] = round(float(env._ball_speed()), 4)
        row["success_crit"] = {
            "not_fallen": not (row["fell_off"] or row["ball_state"] == "fallen"),
            "grasped": row["grasped"],
            "welded": row["welded"],
            "check_success": row["check"],
        }
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
    print(f"=== {TASK}: 5 diverse layout / roll-direction grasp trials ===", flush=True)
    rows = []
    for case in CASES:
        r = run_case(case, miss=False)
        rows.append(r)
        ok = r["check"] and r.get("grasped") and not r["fell_off"]
        mark = "PASS" if ok else "FAIL"
        print(
            f"[{mark}] seed={r['seed']} {r['label']:12s} "
            f"angle={r.get('roll_angle')} edge={r.get('exit_edge')} "
            f"arm={r.get('arm')} state={r.get('ball_state')} "
            f"grasped={r.get('grasped')} check={r['check']} "
            f"fell={r.get('fell_off')} xyz={r.get('ball_xyz')} err={r.get('err')}",
            flush=True,
        )

    n_ok = sum(
        1 for r in rows
        if r["check"] and r.get("grasped") and not r["fell_off"] and not r["err"]
    )
    print(f"\nExpert grasp success: {n_ok}/{len(rows)}", flush=True)

    print("\n=== Miss control (no arm grasp) — ball must fall ===", flush=True)
    miss = run_case(CASES[0], miss=True)
    miss_ok = (not miss["check"]) and miss["fell_off"] and not miss.get("grasped")
    print(
        f"[{'PASS' if miss_ok else 'FAIL'}] miss seed={miss['seed']} "
        f"state={miss.get('ball_state')} check={miss['check']} "
        f"grasped={miss.get('grasped')} fell={miss.get('fell_off')} "
        f"xyz={miss.get('ball_xyz')} err={miss.get('err')}",
        flush=True,
    )
    print(
        f"\nSummary: expert {n_ok}/{len(rows)} grasped; "
        f"miss control {'ok' if miss_ok else 'BROKEN'}",
        flush=True,
    )
    if n_ok < 3 or not miss_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
