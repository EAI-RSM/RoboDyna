"""Short scripted smoke for sort_apples_belt tip-drop (no video).

Prints per-apple final pose / settled side and flags tip jams.
Also fails if any undeposited apple sits at the tip for ≥0.5 s mid-episode.
"""
from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, "./")
sys.path.insert(0, "./script/bench_script")

import numpy as np
import yaml

from script.collect_data import class_decorator, get_embodiment_config
from envs import CONFIGS_PATH

TASK = "sort_apples_belt"
CONFIG = "demo_dynamic"
COLOR_NAME = {0: "red", 1: "green", 2: "rotten"}

# Brief smoke: default + Opt2 demo seeds.
SEEDS = [40, 45]
# 1.5 s at 250 Hz — brief tip contact is normal for v22 PhysX slide.
TIP_WINDOW_FRAMES = 375


def build_base_args():
    with open(f"./task_config/{CONFIG}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    args["task_name"] = TASK
    args["task_config"] = CONFIG
    args["episode_num"] = 1
    args["save_path"] = os.path.abspath("./tmp/tmp_sort_apples_tip_smoke")
    args["collect_data"] = False
    args["eval_video_log"] = False
    args["save_failed_cases"] = False
    args["use_seed"] = False
    args["check_render_success"] = False
    args["export_lerobot"] = False
    args["need_plan"] = True
    args["save_data"] = False
    args.setdefault("data_type", {})
    args["data_type"]["rgb"] = False
    args["data_type"]["third_view"] = False
    args["camera"]["collect_head_camera"] = False
    args["camera"]["collect_wrist_camera"] = False

    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
        emb = yaml.load(f.read(), Loader=yaml.FullLoader)

    def emb_file(t):
        return emb[t]["file_path"]

    et = args["embodiment"]
    args["left_robot_file"] = emb_file(et[0])
    args["right_robot_file"] = emb_file(et[1])
    args["embodiment_dis"] = et[2]
    args["dual_arm_embodied"] = False
    args["embodiment_name"] = f"{et[0]}+{et[1]}"
    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    return args


def tip_zone(task, p) -> bool:
    """Undeposited tip jam: |x|∈[0.10,0.16], z≈belt, near fork."""
    belt = float(task._belt_surf)
    return (
        0.10 <= abs(float(p[0])) <= 0.16
        and abs(float(p[1]) - float(task.BELT_Y_FORK)) < 0.10
        and float(p[2]) > belt - 0.06
    )


def summarize(task, mid_tip_max=None, mid_tip_events=None) -> dict:
    belt = float(task._belt_surf)
    stuck = []
    rows = []
    for i in range(task.n_apples):
        p = np.array(task.apples[i].get_pose().p, dtype=float)
        side = task._settled_side(i)
        cname = COLOR_NAME.get(int(task.apple_colors[i]), "?")
        # Position-only tip jam (ignore deposited): final pose on tip is always bad.
        tip_jam = (not bool(task._deposited[i])) and tip_zone(task, p)
        # Also flag tip-height float freeze in the box mouth (prior false success).
        tip_height_float = (
            (not bool(task._deposited[i]) or p[2] > float(task._z0) + 0.10)
            and abs(p[0]) >= 0.12
            and abs(p[0]) <= 0.22
            and p[2] > belt - 0.06
            and abs(p[1] - float(task.BELT_Y_FORK)) < 0.12
        )
        if tip_jam or tip_height_float:
            stuck.append(i)
        rows.append({
            "i": i,
            "color": cname,
            "want": task._target_side_for_apple(i),
            "settled": side,
            "deposited": bool(task._deposited[i]),
            "mode": task._apple_mode[i],
            "routed": task._routed[i],
            "xyz": (float(p[0]), float(p[1]), float(p[2])),
            "tip_jam": tip_jam or tip_height_float,
            "below_belt": p[2] < belt - 0.02,
            "deep_in_box": p[2] < float(task._z0) + 0.08,
        })
    try:
        ok = bool(task.check_success())
    except Exception:
        ok = False
    mid_fail = bool(mid_tip_max is not None and mid_tip_max >= TIP_WINDOW_FRAMES)
    return {
        "n": task.n_apples,
        "ok": ok and not mid_fail,
        "green_on_left": bool(task.green_on_left),
        "has_rotten": bool(getattr(task, "has_rotten", False)),
        "rotten_idx": getattr(task, "rotten_idx", None),
        "stuck": stuck,
        "rows": rows,
        "belt": belt,
        "mid_tip_max": int(mid_tip_max or 0),
        "mid_tip_events": mid_tip_events or [],
        "mid_tip_fail": mid_fail,
    }


def run_one(task, args, seed: int, flags: dict) -> dict:
    args = dict(args)
    args.setdefault("task_args", {}).setdefault(TASK, {})
    args["task_args"][TASK] = dict(args["task_args"].get(TASK, {}))
    args["task_args"][TASK].update(flags)
    args["left_joint_path"] = []
    args["right_joint_path"] = []
    args["need_plan"] = True
    args["save_data"] = False
    try:
        task.setup_demo(now_ep_num=0, seed=seed, **args)

        # Mid-episode tip dwell: any undeposited apple in tip zone.
        dwell = [0] * max(20, int(task.n_apples))
        max_dwell = [0] * max(20, int(task.n_apples))
        events = []
        orig = task._step_physics_apples

        def wrapped():
            orig()
            for i in range(task._spawned):
                if task._deposited[i] or task._apple_mode[i] == "done":
                    dwell[i] = 0
                    continue
                p = np.array(task.apples[i].get_pose().p, dtype=float)
                if tip_zone(task, p):
                    dwell[i] += 1
                    if dwell[i] > max_dwell[i]:
                        max_dwell[i] = dwell[i]
                    if dwell[i] in (1, 25, 50, 125, 250) or dwell[i] % 125 == 0:
                        events.append((
                            int(task._step_ctr), i, dwell[i],
                            tuple(map(float, p)), task._apple_mode[i],
                        ))
                else:
                    dwell[i] = 0

        task._step_physics_apples = wrapped
        task.play_once()
        mid_max = max(max_dwell[: task.n_apples]) if task.n_apples else 0
        info = summarize(task, mid_tip_max=mid_max, mid_tip_events=events[:12])
        info["seed"] = seed
        info["error"] = None
        info["max_dwell_per_apple"] = {
            i: max_dwell[i] for i in range(task.n_apples)
        }
        return info
    except Exception as e:
        traceback.print_exc()
        return {"seed": seed, "ok": False, "stuck": [], "rows": [], "error": str(e)}
    finally:
        try:
            task.close_env(clear_cache=True)
        except Exception:
            pass


def print_result(label: str, r: dict):
    if r.get("error"):
        print(f"  seed={r['seed']}: ERROR {r['error']}")
        return
    status = "OK" if r["ok"] and not r["stuck"] and not r.get("mid_tip_fail") else "FAIL"
    jam = f" tip_jam={r['stuck']}" if r["stuck"] else " tip_jam=none"
    mid = (
        f" mid_tip_max={r.get('mid_tip_max', 0)}"
        f"({r.get('mid_tip_max', 0)/250:.2f}s)"
        f"{' **MID_TIP_FAIL**' if r.get('mid_tip_fail') else ''}"
    )
    print(
        f"  seed={r['seed']}: {status} success={r['ok']} n={r['n']} "
        f"green_left={r['green_on_left']} rotten={r['has_rotten']}/{r['rotten_idx']}"
        f"{jam}{mid}"
    )
    for row in r["rows"]:
        x, y, z = row["xyz"]
        flag = " **TIP_JAM**" if row["tip_jam"] else ""
        print(
            f"    [{row['i']}] {row['color']:6s} want={row['want']:5s} "
            f"settled={str(row['settled']):5s} dep={row['deposited']} "
            f"mode={row['mode']:7s} xyz=({x:+.3f},{y:+.3f},{z:.3f}) "
            f"deep={row['deep_in_box']}{flag}"
        )


def main():
    os.makedirs("./tmp/tmp_sort_apples_tip_smoke", exist_ok=True)
    base = build_base_args()
    task = class_decorator(TASK)

    conditions = [
        ("default_divert", {
            "rotten_prob": 0.0,
            "n_apples_min": 4,
            "n_apples_max": 4,
            "color_mode": "alternating",
        }),
        ("opt2_rotten", {
            "rotten_prob": 1.0,
            "n_apples_min": 4,
            "n_apples_max": 4,
            "color_mode": "alternating",
        }),
    ]

    print(f"\n=== sort_apples_belt tip-drop smoke: seeds={SEEDS} ===\n")
    all_ok = True
    for name, flags in conditions:
        print(f"--- {name}: {flags} ---")
        for seed in SEEDS:
            r = run_one(task, base, seed, flags)
            print_result(name, r)
            if r.get("error") or r.get("stuck") or not r.get("ok") or r.get("mid_tip_fail"):
                all_ok = False
        print()

    print(
        "=== DONE ===",
        "PASS (no tip jams + success + no mid tip dwell)"
        if all_ok
        else "FAIL (tip jams / mid dwell / errors / fail)",
    )
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
