#!/usr/bin/env python3
"""Run expert play_once for every script_exp basic task; report success counts.

Uses the default (non-option) scenario from ``demo_dynamic.yml`` — the normal
scripted expert, not the interactive sandboxes.
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
os.environ.pop("DISPLAY", None)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
sys.path[:0] = [ROOT, os.path.join(ROOT, "script"), os.path.join(ROOT, "script/bench_script")]

from script.bench_script.record_demo import build_args  # noqa: E402
from script.collect_data import class_decorator  # noqa: E402

# Same suite as script_exp/interactive_task_gui.py TASKS (CLI names).
TASKS = (
    "catch_marbles_trapdoors",
    "catch_ramp_ball",
    "catch_rat",
    "catch_shelf_marble",
    "catch_valley_ball",
    "catch_valley_ball_v1",
    "stop_valley_ball",
    "cook_meat",
    "put_cup_belt",
    "dispense_gummy",
    "punch_dual_holes",
    "save_goal",
    "hit_target",
    "load_train",
    "marble_shelf_maze",
    "pack_fruits",
    "pick_ripe_apple",
    "place_block_belt",
    "play_billiard",
    "control_quality",
    "drop_ball_hole",
    "sort_apples_belt",
    "whack_moles",
)

# Explicit default-scenario overrides (mirrors interactive_task_gui SCENARIO_OVERRIDES["default"]).
DEFAULT_OVERRIDES = {
    "catch_marbles_trapdoors": {"door_open_once": False, "enable_distractor": False},
    "catch_ramp_ball": {"wall_bounce_enabled": False, "enable_distractor": False},
    "catch_rat": {"catch_two_mice": False, "opaque_surface": False},
    "catch_shelf_marble": {"reactive_marble": False, "oscillating_shelf_enabled": False},
    "catch_valley_ball": {"wall_bounce_enabled": False, "enable_distractor": False},
    "catch_valley_ball_v1": {"wall_bounce_enabled": False, "enable_distractor": False},
    "stop_valley_ball": {"wall_bounce_enabled": False, "enable_distractor": False},
    "cook_meat": {"cook_button_enabled": False, "dual_setup_enabled": False},
    "put_cup_belt": {"blue_curtains_enabled": False, "blue_curtain_dynamic_enabled": False},
    "dispense_gummy": {"layout_mode": "alternating", "belt_continuous_motion": False},
    "punch_dual_holes": {"missing_tile_mode": False, "belt_continous_motion": False},
    "save_goal": {"players_enabled": False, "cover_enabled": False},
    "hit_target": {"blocker_enabled": False, "blocker_dynamic": False},
    "load_train": {"target_wagon_mode": False, "tunnel_enabled": False},
    "marble_shelf_maze": {"continuous_ball_motion": False, "oscillating_bowl_enabled": False},
    "pack_fruits": {
        "spawn_mode": "parallel",
        "pair_stagger_enabled": False,
        "single_wave_any_belt": False,
        "distractor_enabled": False,
    },
    "pick_ripe_apple": {"two_apples_enabled": False, "basket_move_enabled": False},
    "place_block_belt": {"bowl_move_enabled": False, "blocker_enabled": False},
    "play_billiard": {"specific_hole": False, "enable_distractors": False},
    "control_quality": {"color_mode": "alternating", "black_frac_max": 0.0},
    "drop_ball_hole": {"stick_to_surface": False, "add_dummy_hole": False},
    "sort_apples_belt": {"color_mode": "alternating", "rotten_prob": 0.0},
    "whack_moles": {"distractor_enabled": False, "relocating_moles": False},
}

CONFIG = "demo_dynamic"


def run_seed(task_name: str, seed: int) -> dict:
    save_root = os.path.abspath(f"./tmp/tmp_{task_name}_basic_sweep")
    os.makedirs(save_root, exist_ok=True)
    args = build_args(task_name, CONFIG, save_root, option=None, task_arg_overrides=[])
    args["collect_data"] = False
    args["save_data"] = False
    args["eval_video_log"] = False
    args["need_plan"] = True
    args["render_freq"] = 0
    args["episode_num"] = 1
    args["check_render_success"] = False
    args["export_lerobot"] = False
    # Keep suite default dynamics; force default (non-opt) task args.
    task_args = args.setdefault("task_args", {}).setdefault(task_name, {})
    task_args.update(DEFAULT_OVERRIDES.get(task_name, {}))

    env = class_decorator(task_name)
    row = {
        "task": task_name,
        "seed": seed,
        "plan": False,
        "check": False,
        "ok": False,
        "err": None,
    }
    try:
        env.setup_demo(now_ep_num=0, seed=seed, **args)
        env.play_once()
        plan = bool(getattr(env, "plan_success", True))
        check = bool(env.check_success())
        row["plan"] = plan
        row["check"] = check
        row["ok"] = bool(plan and check)
    except Exception as exc:  # noqa: BLE001
        row["err"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        try:
            env.close_env()
        except Exception:
            pass
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10, help="seeds per task (0..n-1)")
    p.add_argument("--tasks", nargs="*", default=list(TASKS))
    p.add_argument("--start-seed", type=int, default=0)
    args = p.parse_args()
    seeds = list(range(int(args.start_seed), int(args.start_seed) + int(args.n)))

    summary = []
    for task in args.tasks:
        print(f"\n===== {task} ({len(seeds)} seeds) =====", flush=True)
        ok_seeds = []
        fail_seeds = []
        for seed in seeds:
            row = run_seed(task, seed)
            tag = "OK" if row["ok"] else "FAIL"
            if row["err"]:
                extra = f" err={row['err']}"
            else:
                extra = f" plan={row['plan']} check={row['check']}"
            print(f"  seed={seed} {tag}{extra}", flush=True)
            if row["ok"]:
                ok_seeds.append(seed)
            else:
                fail_seeds.append(seed)
        summary.append((task, len(ok_seeds), len(seeds), ok_seeds, fail_seeds))

    print("\n========== SUMMARY ==========", flush=True)
    print(f"{'task':26s}  success  rate   ok_seeds", flush=True)
    for task, n_ok, n_tot, ok_seeds, fail_seeds in summary:
        print(
            f"{task:26s}  {n_ok:2d}/{n_tot:<2d}     {100.0 * n_ok / n_tot:5.1f}%  {ok_seeds}",
            flush=True,
        )


if __name__ == "__main__":
    main()
