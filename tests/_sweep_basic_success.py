#!/usr/bin/env python3
"""Run expert play_once for every base-suite task; report success counts.

Uses scenario overrides matching ``interactive.base_task_gui.SCENARIO_OVERRIDES``
(default / opt1 / opt2 / opt1+2). The scripted expert path — not interactive.
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

# Same suite as the GUI 23-task sweep.
TASKS = (
    "catch_marbles_trapdoors",
    "catch_ramp_ball",
    "catch_cuboid",
    "catch_shelf_marble",
    "catch_valley_ball",
    "stop_valley_ball",
    "cook_meat",
    "cook_meat_timer",
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

# Mirrors base_task_gui.SCENARIO_OVERRIDES.
SCENARIO_OVERRIDES = {
    "catch_marbles_trapdoors": {
        "default": {"door_open_once": False, "enable_distractor": False},
        "opt1": {"door_open_once": True, "enable_distractor": False},
        "opt2": {"door_open_once": False, "enable_distractor": True},
        "opt1+2": {"door_open_once": True, "enable_distractor": True},
    },
    "catch_ramp_ball": {
        "default": {"wall_bounce_enabled": False, "enable_distractor": False},
        "opt1": {"wall_bounce_enabled": True, "enable_distractor": False},
        "opt2": {"wall_bounce_enabled": False, "enable_distractor": True},
        "opt1+2": {"wall_bounce_enabled": True, "enable_distractor": True},
    },
    "catch_cuboid": {
        "default": {"catch_two_cuboids": False, "opaque_surface": False},
        "opt1": {"catch_two_cuboids": True, "opaque_surface": False},
        "opt2": {"catch_two_cuboids": False, "opaque_surface": True},
        "opt1+2": {"catch_two_cuboids": True, "opaque_surface": True},
    },
    "catch_shelf_marble": {
        "default": {"reactive_marble": False, "oscillating_shelf_enabled": False},
        "opt1": {"reactive_marble": True, "oscillating_shelf_enabled": False},
        "opt2": {"reactive_marble": False, "oscillating_shelf_enabled": True},
        "opt1+2": {"reactive_marble": True, "oscillating_shelf_enabled": True},
    },
    "catch_valley_ball": {
        "default": {"wall_bounce_enabled": False, "enable_distractor": False},
        "opt1": {"wall_bounce_enabled": True, "enable_distractor": False},
        "opt2": {"wall_bounce_enabled": False, "enable_distractor": True},
        "opt1+2": {"wall_bounce_enabled": True, "enable_distractor": True},
    },
    "stop_valley_ball": {
        "default": {"wall_bounce_enabled": False, "enable_distractor": False},
        "opt1": {"wall_bounce_enabled": True, "enable_distractor": False},
        "opt2": {"wall_bounce_enabled": False, "enable_distractor": True},
        "opt1+2": {"wall_bounce_enabled": True, "enable_distractor": True},
    },
    "cook_meat": {
        "default": {"cook_button_enabled": False, "dual_setup_enabled": False},
        "opt1": {"cook_button_enabled": True, "dual_setup_enabled": False},
        "opt2": {"cook_button_enabled": False, "dual_setup_enabled": True},
        "opt1+2": {"cook_button_enabled": True, "dual_setup_enabled": True},
    },
    "cook_meat_timer": {
        "default": {"cook_button_enabled": False, "dual_setup_enabled": False},
        "opt1": {"cook_button_enabled": True, "dual_setup_enabled": False},
        "opt2": {"cook_button_enabled": False, "dual_setup_enabled": True},
        "opt1+2": {"cook_button_enabled": True, "dual_setup_enabled": True},
    },
    "put_cup_belt": {
        "default": {"blue_curtains_enabled": False, "blue_curtain_dynamic_enabled": False},
        "opt1": {"blue_curtains_enabled": True, "blue_curtain_dynamic_enabled": False},
        "opt2": {"blue_curtains_enabled": False, "blue_curtain_dynamic_enabled": True},
        "opt1+2": {"blue_curtains_enabled": True, "blue_curtain_dynamic_enabled": True},
    },
    "dispense_gummy": {
        "default": {"layout_mode": "alternating", "belt_continuous_motion": False},
        "opt1": {"layout_mode": "random", "belt_continuous_motion": False},
        "opt2": {"layout_mode": "alternating", "belt_continuous_motion": True},
        "opt1+2": {"layout_mode": "random", "belt_continuous_motion": True},
    },
    "punch_dual_holes": {
        "default": {"missing_tile_mode": False, "belt_continous_motion": False},
        "opt1": {"missing_tile_mode": True, "belt_continous_motion": False},
        "opt2": {"missing_tile_mode": False, "belt_continous_motion": True},
        "opt1+2": {"missing_tile_mode": True, "belt_continous_motion": True},
    },
    "save_goal": {
        "default": {"players_enabled": False, "cover_enabled": False},
        "opt1": {"players_enabled": True, "cover_enabled": False},
        "opt2": {"players_enabled": False, "cover_enabled": True},
        "opt1+2": {"players_enabled": True, "cover_enabled": True},
    },
    "hit_target": {
        "default": {"blocker_enabled": False, "blocker_dynamic": False},
        "opt1": {"blocker_enabled": True, "blocker_dynamic": False},
        "opt2": {"blocker_enabled": False, "blocker_dynamic": True},
        "opt1+2": {"blocker_enabled": True, "blocker_dynamic": True},
    },
    "load_train": {
        "default": {"target_wagon_mode": False, "tunnel_enabled": False},
        "opt1": {"target_wagon_mode": True, "tunnel_enabled": False},
        "opt2": {"target_wagon_mode": False, "tunnel_enabled": True},
        "opt1+2": {"target_wagon_mode": True, "tunnel_enabled": True},
    },
    "marble_shelf_maze": {
        "default": {"continuous_ball_motion": False, "oscillating_bowl_enabled": False},
        "opt1": {"continuous_ball_motion": True, "oscillating_bowl_enabled": False},
        "opt2": {"continuous_ball_motion": False, "oscillating_bowl_enabled": True},
        "opt1+2": {"continuous_ball_motion": True, "oscillating_bowl_enabled": True},
    },
    "pack_fruits": {
        "default": {"two_colors_enabled": False, "distractor_enabled": False},
        "opt1": {"two_colors_enabled": True, "distractor_enabled": False},
        "opt2": {"two_colors_enabled": False, "distractor_enabled": True},
        "opt1+2": {"two_colors_enabled": True, "distractor_enabled": True},
    },
    "pick_ripe_apple": {
        "default": {"two_apples_enabled": False, "basket_move_enabled": False},
        "opt1": {"two_apples_enabled": True, "basket_move_enabled": False},
        "opt2": {"two_apples_enabled": False, "basket_move_enabled": True},
        "opt1+2": {"two_apples_enabled": True, "basket_move_enabled": True},
    },
    "place_block_belt": {
        "default": {"bowl_move_enabled": False, "blocker_enabled": False},
        "opt1": {"bowl_move_enabled": True, "blocker_enabled": False},
        "opt2": {"bowl_move_enabled": False, "blocker_enabled": True},
        "opt1+2": {"bowl_move_enabled": True, "blocker_enabled": True},
    },
    "play_billiard": {
        "default": {"specific_hole": False, "enable_distractors": False},
        "opt1": {"specific_hole": True, "enable_distractors": False},
        "opt2": {"specific_hole": False, "enable_distractors": True},
        "opt1+2": {"specific_hole": True, "enable_distractors": True},
    },
    "control_quality": {
        "default": {"color_mode": "alternating", "black_frac_max": 0.0},
        "opt1": {"color_mode": "random", "black_frac_max": 0.0},
        "opt2": {"color_mode": "alternating", "black_frac_max": 0.5},
        "opt1+2": {"color_mode": "random", "black_frac_max": 0.5},
    },
    "drop_ball_hole": {
        "default": {"stick_to_surface": False, "add_dummy_hole": False},
        "opt1": {"stick_to_surface": True, "add_dummy_hole": False},
        "opt2": {"stick_to_surface": False, "add_dummy_hole": True},
        "opt1+2": {"stick_to_surface": True, "add_dummy_hole": True},
    },
    "sort_apples_belt": {
        "default": {"color_mode": "alternating", "rotten_prob": 0.0},
        "opt1": {"color_mode": "random", "rotten_prob": 0.0},
        "opt2": {"color_mode": "alternating", "rotten_prob": 0.3},
        "opt1+2": {"color_mode": "random", "rotten_prob": 0.3},
    },
    "whack_moles": {
        "default": {"distractor_enabled": False, "relocating_moles": False},
        "opt1": {"distractor_enabled": True, "relocating_moles": False},
        "opt2": {"distractor_enabled": False, "relocating_moles": True},
        "opt1+2": {"distractor_enabled": True, "relocating_moles": True},
    },
}

# Back-compat alias used by older call sites.
DEFAULT_OVERRIDES = {t: SCENARIO_OVERRIDES[t]["default"] for t in TASKS}

CONFIG = "demo_dynamic"
SCENARIOS = ("default", "opt1", "opt2", "opt1+2")


def run_seed(task_name: str, seed: int, scenario: str = "default") -> dict:
    sweep_root = os.environ.get("ROBODYNA_SWEEP_ROOT", "./tmp")
    safe_scenario = scenario.replace("+", "plus")
    save_root = os.path.abspath(os.path.join(sweep_root, f"tmp_{task_name}_{safe_scenario}_basic_sweep"))
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
    task_args = args.setdefault("task_args", {}).setdefault(task_name, {})
    overrides = SCENARIO_OVERRIDES.get(task_name, {}).get(scenario)
    if overrides is None:
        raise KeyError(f"No overrides for task={task_name!r} scenario={scenario!r}")
    task_args.update(overrides)

    env = class_decorator(task_name)
    row = {
        "task": task_name,
        "scenario": scenario,
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
    p.add_argument("--n", type=int, default=10, help="seeds per task×scenario (0..n-1)")
    p.add_argument("--tasks", nargs="*", default=list(TASKS))
    p.add_argument("--start-seed", type=int, default=0)
    p.add_argument(
        "--scenarios",
        nargs="*",
        default=["default"],
        choices=list(SCENARIOS),
        help="Scenarios to sweep (default: default only)",
    )
    args = p.parse_args()
    seeds = list(range(int(args.start_seed), int(args.start_seed) + int(args.n)))
    scenarios = list(args.scenarios)

    summary = []
    for scenario in scenarios:
        for task in args.tasks:
            print(
                f"\n===== {task} [{scenario}] ({len(seeds)} seeds) =====",
                flush=True,
            )
            ok_seeds = []
            fail_seeds = []
            for seed in seeds:
                row = run_seed(task, seed, scenario=scenario)
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
            summary.append(
                (task, scenario, len(ok_seeds), len(seeds), ok_seeds, fail_seeds)
            )

    print("\n========== SUMMARY ==========", flush=True)
    print(f"{'task':26s}  {'scen':7s}  success  rate   ok_seeds", flush=True)
    for task, scenario, n_ok, n_tot, ok_seeds, fail_seeds in summary:
        print(
            f"{task:26s}  {scenario:7s}  {n_ok:2d}/{n_tot:<2d}     "
            f"{100.0 * n_ok / n_tot:5.1f}%  {ok_seeds}",
            flush=True,
        )


if __name__ == "__main__":
    main()
