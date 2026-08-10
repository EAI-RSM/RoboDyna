#!/usr/bin/env python3
"""Merge household + basic sweep results into instructions/household_sweep_results.html."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HH_JSON = Path("/tmp/hh_sweep_results.json")
if not HH_JSON.is_file():
    HH_JSON = ROOT / "logs/hh_sweep_results.json"
BASIC_JSON = ROOT / "logs/basic_sweep_results.json"
BASIC_LOG = ROOT / "logs/basic_sweep_n5_allscen.log"
if not BASIC_LOG.is_file():
    BASIC_LOG = ROOT / "logs/basic_test_and_demos.log"
OUT = ROOT / "instructions/household_sweep_results.html"
RUN_NOTE = (
    "Sweep of 2026-08-08: base "
    "<code>logs/basic_sweep_n5_allscen.log</code> (5×4) then household "
    "<code>logs/household_sweep_n10.log</code> (10 seeds). "
    "<code>sort_apples_belt</code> reswept 2026-08-08 after timeout-name fix "
    "(<code>logs/sort_apples_belt_resweep_n5.log</code>)."
)

SCENARIOS = ("default", "opt1", "opt2", "opt1+2")


def rate_class(n_ok: int, n: int) -> str:
    if n == 0:
        return "mid"
    r = n_ok / n
    if r >= 0.9:
        return "good"
    if r >= 0.5:
        return "mid"
    return "bad"


def parse_basic_failure_reasons(log_text: str) -> dict[str, dict[str, list[str]]]:
    """task -> scenario -> list of short reason strings (deduped).

    If a scenario appears multiple times in the log (resumes / re-runs), keep the
    **last** completed block only.
    """
    out: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    # Split by task/scenario headers
    parts = re.split(r"\n===== ([^\s]+) \[([^\]]+)\] \(\d+ seeds\) =====\n", log_text)
    # parts[0]=preamble, then triples: task, scenario, body
    i = 1
    while i + 2 < len(parts):
        task, scenario, body = parts[i], parts[i + 1], parts[i + 2]
        i += 3
        # Prefer the body up to the scenario summary arrow when present.
        arrow = re.search(r"\n  → \d+/\d+ ok=", body)
        if arrow:
            body = body[: arrow.start()]
        reasons = []
        for sm in re.finditer(
            r"seed=(\d+) FAIL(?: plan=(\w+) check=(\w+))?(?: err=(.+))?",
            body,
        ):
            seed, plan, check, err = sm.groups()
            if err:
                # short exception type/message
                err_s = err.strip().splitlines()[0][:120]
                reasons.append(f"seed {seed}: {err_s}")
            elif plan == "False":
                reasons.append(f"seed {seed}: plan failed")
            else:
                reasons.append(f"seed {seed}: check failed (plan ok)")
        # Deduplicate by pattern family
        families: dict[str, list[str]] = defaultdict(list)
        for r in reasons:
            if "err=" in r or ": " in r:
                key = re.sub(r"seed \d+: ", "", r)
                key = re.sub(r"seed=\d+", "seed=N", key)
            else:
                key = r.split(": ", 1)[-1]
            families[key].append(r.split(":")[0])  # seed N
        summary = []
        for key, seeds in families.items():
            seed_list = ", ".join(s.replace("seed ", "") for s in seeds)
            summary.append(f"{key} [{seed_list}]")
        # Last occurrence wins (resume / re-run).
        out[task][scenario] = summary
    return out


# Curated failure narratives from sweep patterns (plan/check + scenario splits).
BASIC_MAIN_REASONS: dict[str, str] = {
    "catch_marbles_trapdoors": (
        "Mostly solid; occasional check fails — marble not through the matching "
        "trapdoor (timing / key press)."
    ),
    "catch_ramp_ball": "Rare check fails under distractor (opt2) — wrong ball caught or miss.",
    "catch_cuboid": (
        "After slower pops (mean 0.056 m/s) and 5 appearances: single-cuboid "
        "(default / opt2) is 5/5; dual catch (opt1 / opt1+2) is 3/5 — missed "
        "simultaneous grasp on seeds 1 and 3."
    ),
    "catch_valley_ball": "Occasional miss on wall-bounce (opt1) or distractor (opt2) seeds.",

    "put_cup_belt": "Occasional plan/check fails on belt placement (not seed-0 systematic).",
    "save_goal": (
        "Keeper placed too late / ball enters goal. Worse with field players "
        "(opt1 / opt1+2)."
    ),
    "hit_target": (
        "default &amp; opt1 are 5/5; dynamic blocker scenarios flake hard "
        "(opt2 / opt1+2 both 2/5) — check miss or plan fail."
    ),
    "load_train": "Occasional miss of allowed wagon (esp. target-wagon + tunnel).",
    "marble_shelf_maze": "Marble misses bowl under continuous motion and/or oscillating bowl.",
    "pack_fruits": "Fruit ends in wrong basket or dropped (check fail after plan); 10/20 overall.",
    "pick_ripe_apple": "Solid on default/opt1; opt1+2 drops to 2/5 (oscillation + spoiled distractors).",
    "place_block_belt": "Moving-bowl scenarios (opt1) tip/miss more often (2/5); blocker-only is solid.",
    "play_billiard": (
        "Cue strike misses allowed pocket or plan fails mid-shot; "
        "10/20 overall — weak on default / opt2 / opt1+2."
    ),
    "drop_ball_hole": (
        "Weak across conditions (9/20). Remaining fails are late finger-release "
        "vs hole or dummy-hole wedges."
    ),
    "sort_apples_belt": (
        "Fixed name clash (<code>_episode_timed_out</code> bool vs method). "
        "Default is solid; plan flakes on random colors (opt1) and especially "
        "opt1+2 (rotten + random)."
    ),
}


def basic_main_reason(task: str, results: dict, fail_map: dict) -> str:
    curated = BASIC_MAIN_REASONS.get(task)
    bits = []
    for sc in SCENARIOS:
        fails = results.get(task, {}).get(sc, {}).get("fail", [])
        if not fails:
            continue
        reasons = fail_map.get(task, {}).get(sc, [])
        # Prefer JSON fail seeds (authoritative); strip stale [...] from log blurbs.
        if reasons:
            short = re.sub(r"\s*\[[^\]]*\]\s*$", "", reasons[0]).strip() or reasons[0]
            bits.append(f"<strong>{sc}</strong>: {short} {fails}")
        else:
            bits.append(f"<strong>{sc}</strong>: fail seeds {fails}")
    detail = "<br>".join(bits) if bits else "—"
    if curated:
        return f"{curated}<br><span class='seeds'>{detail}</span>"
    return detail


CSS = """
  :root {
    --bg: #0f1419; --panel: #1a222c; --panel2: #222c38; --border: #334155;
    --text: #e8eef4; --muted: #9aabbc; --accent: #5b9fd4;
    --ok: #2f9e6b; --fail: #c45c5c; --warn: #b8860b;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    background: radial-gradient(1200px 600px at 10% -10%, #1a2a3a 0%, var(--bg) 55%);
    color: var(--text); line-height: 1.45;
  }
  header.hero {
    padding: 2rem 1.5rem 1.25rem; border-bottom: 1px solid var(--border);
    background: linear-gradient(180deg, #15202b, transparent);
  }
  h1 { margin: 0 0 0.4rem; font-size: 1.75rem; letter-spacing: -0.02em; }
  h2 { margin: 1.75rem 0 0.75rem; font-size: 1.25rem; }
  h3 { margin: 0 0 0.4rem; font-size: 1.05rem; font-weight: 600; }
  .muted { color: var(--muted); max-width: 75ch; }
  .wrap { max-width: 1200px; margin: 0 auto; padding: 1rem 1.5rem 3rem; }
  .stats {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 0.75rem; margin: 1rem 0 1.5rem;
  }
  .stat {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 10px; padding: 0.85rem 1rem;
  }
  .stat b { display: block; font-size: 1.4rem; }
  .stat span { color: var(--muted); font-size: 0.85rem; }
  table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
  th, td {
    text-align: left; padding: 0.45rem 0.5rem;
    border-bottom: 1px solid var(--border); vertical-align: top;
  }
  th {
    color: var(--muted); font-weight: 600; font-size: 0.75rem;
    text-transform: uppercase; letter-spacing: 0.04em;
  }
  code, .mono {
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 0.88em; color: #c6e0f5;
  }
  .rate { font-weight: 700; white-space: nowrap; }
  .rate.good { color: #8fd9b5; }
  .rate.mid { color: #f0d48a; }
  .rate.bad { color: #f0a0a0; }
  .bar {
    display: inline-block; height: 8px; width: 56px; background: #2a3542;
    border-radius: 4px; overflow: hidden; vertical-align: middle; margin-right: 0.35rem;
  }
  .bar > i {
    display: block; height: 100%;
    background: linear-gradient(90deg, #2f9e6b, #5b9fd4);
  }
  .card {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 12px; padding: 1rem 1.1rem; margin: 0.85rem 0;
  }
  .card.focus { border-color: #6a5120; box-shadow: inset 0 0 0 1px rgba(184,134,11,0.25); }
  .pill {
    display: inline-block; font-size: 0.72rem; padding: 0.15rem 0.5rem;
    border-radius: 999px; border: 1px solid var(--border); color: var(--muted);
    margin-right: 0.25rem;
  }
  .pill.ok { color: #b8f0d4; border-color: var(--ok); background: rgba(47,158,107,0.15); }
  .pill.fail { color: #f0c0c0; border-color: var(--fail); background: rgba(196,92,92,0.15); }
  .pill.warn { color: #f0d48a; border-color: var(--warn); background: rgba(184,134,11,0.15); }
  ul.reasons { margin: 0.4rem 0 0.2rem 1.1rem; padding: 0; color: var(--muted); }
  ul.reasons li { margin: 0.25rem 0; }
  ul.reasons strong { color: var(--text); font-weight: 600; }
  .seeds { font-size: 0.85rem; color: var(--muted); }
  .callout {
    background: var(--panel2); border-left: 3px solid var(--accent);
    padding: 0.85rem 1rem; border-radius: 0 8px 8px 0; margin: 1rem 0;
    color: var(--muted); font-size: 0.92rem;
  }
  a { color: var(--accent); text-decoration: none; }
  a:hover { text-decoration: underline; }
  .scen-grid {
    display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 0.35rem;
    font-size: 0.8rem;
  }
  @media (max-width: 900px) { .scen-grid { grid-template-columns: repeat(2, 1fr); } }
  .scen {
    background: var(--panel2); border: 1px solid var(--border);
    border-radius: 8px; padding: 0.4rem 0.5rem;
  }
  .scen .lbl { color: var(--muted); font-size: 0.72rem; text-transform: uppercase; }
"""


def hh_section(hh: dict) -> str:
    # Keep the curated failure analysis already written for weak HH tasks.
    # Rebuild summary table from JSON.
    rows = []
    reasons_static = {
        "trap_bug": "Trap misses moving bug (plan OK, check fails) — seeds 1, 2, 5, 9.",
        "boil_milk": "—",
        "fill_coffee_jar": "—",
        "pour_beer": "Mostly plan fails (9/10); only seed 9 succeeds.",
        "cook_food": "UnStableError food-not-in-pan (0, 8) + check fail (1).",
        "cook_food_timer": "Same placement UnStableError path (0, 8, 9) + check fail (1).",
        "measure_ingredient": "Single check fail on seed 0.",
        "make_soup": "Plan fails on seeds 2, 6, 7 (tip/pour incomplete).",
        "catch_cup": "—",
        "catch_mouse_object_drop": "Basket placed but object not soft-caught (seeds 3, 6).",
        "stop_ball": "Ball reaches edge / not intercepted (seeds 1, 5, 6, 8).",
        "clean_table": "—",
    }
    order = list(reasons_static.keys())
    total_ok = total_n = 0
    perfect = 0
    for task in order:
        if task not in hh:
            continue
        ok, fail = hh[task]["ok"], hh[task]["fail"]
        n_ok, n = len(ok), len(ok) + len(fail)
        total_ok += n_ok
        total_n += n
        if n_ok == n:
            perfect += 1
        cls = rate_class(n_ok, n)
        pct = int(100 * n_ok / n) if n else 0
        ok_s = ", ".join(map(str, ok)) if ok else "—"
        fail_s = ", ".join(map(str, fail)) if fail else "—"
        rows.append(
            f"<tr><td><code>{task}</code></td>"
            f'<td><span class="bar"><i style="width:{pct}%"></i></span>'
            f'<span class="rate {cls}">{n_ok}/{n}</span></td>'
            f'<td class="seeds">{ok_s}</td><td class="seeds">{fail_s}</td>'
            f"<td>{reasons_static[task]}</td></tr>"
        )
    stats = f"""
  <div class="stats">
    <div class="stat"><b>{len(order)}</b><span>household tasks</span></div>
    <div class="stat"><b>{total_n}</b><span>total seed runs</span></div>
    <div class="stat"><b>{total_ok}</b><span>successes ({100*total_ok/total_n:.0f}%)</span></div>
    <div class="stat"><b>{total_n-total_ok}</b><span>failures</span></div>
    <div class="stat"><b>{perfect}</b><span>tasks at 10/10</span></div>
  </div>"""
    table = (
        "<table><thead><tr><th>Task</th><th>Rate</th><th>OK seeds</th>"
        "<th>Fail seeds</th><th>Main failure mode(s)</th></tr></thead><tbody>"
        + "\n".join(rows)
        + "</tbody></table>"
    )
    return stats + table


def basic_section(basic: dict, fail_map: dict) -> str:
    tasks = list(basic.keys())
    rows = []
    tot_ok = tot_n = 0
    weak = []
    for task in tasks:
        cells = []
        task_ok = task_n = 0
        for sc in SCENARIOS:
            res = basic[task].get(sc, {"ok": [], "fail": []})
            n_ok = len(res["ok"])
            n = n_ok + len(res["fail"])
            task_ok += n_ok
            task_n += n
            cls = rate_class(n_ok, n)
            cells.append(
                f'<div class="scen"><div class="lbl">{sc}</div>'
                f'<span class="rate {cls}">{n_ok}/{n}</span></div>'
            )
        tot_ok += task_ok
        tot_n += task_n
        cls = rate_class(task_ok, task_n)
        pct = int(100 * task_ok / task_n) if task_n else 0
        reason = basic_main_reason(task, basic, fail_map)
        if task_ok < task_n:
            weak.append((task, task_ok, task_n, reason))
        rows.append(
            f"<tr><td><code>{task}</code></td>"
            f'<td><span class="bar"><i style="width:{pct}%"></i></span>'
            f'<span class="rate {cls}">{task_ok}/{task_n}</span></td>'
            f'<td><div class="scen-grid">{"".join(cells)}</div></td>'
            f"<td>{reason}</td></tr>"
        )

    weak_cards = []
    for task, n_ok, n, reason in sorted(weak, key=lambda x: x[1] / max(x[2], 1)):
        if n_ok / n >= 0.9:
            continue
        curated = BASIC_MAIN_REASONS.get(task, "")
        detail_bits = []
        if curated:
            detail_bits.append(f"<li>{curated}</li>")
        for sc in SCENARIOS:
            fails = basic[task][sc]["fail"]
            oks = basic[task][sc]["ok"]
            if not fails:
                continue
            detail_bits.append(
                f"<li><strong>{sc}</strong>: {len(oks)}/5 ok — fail seeds {fails}</li>"
            )
        weak_cards.append(
            f'<div class="card"><h3><code>{task}</code> '
            f'<span class="pill warn">{n_ok}/{n}</span></h3>'
            f'<ul class="reasons">{"".join(detail_bits) or f"<li>{reason}</li>"}</ul></div>'
        )

    stats = f"""
  <div class="stats">
    <div class="stat"><b>{len(tasks)}</b><span>base tasks</span></div>
    <div class="stat"><b>{tot_n}</b><span>runs (5 seeds × 4 scenarios)</span></div>
    <div class="stat"><b>{tot_ok}</b><span>successes ({100*tot_ok/tot_n:.0f}%)</span></div>
    <div class="stat"><b>{tot_n-tot_ok}</b><span>failures</span></div>
  </div>"""
    table = (
        "<table><thead><tr><th>Task</th><th>Total</th>"
        "<th>Per condition (default / opt1 / opt2 / opt1+2)</th>"
        "<th>Failure reasons</th></tr></thead><tbody>"
        + "\n".join(rows)
        + "</tbody></table>"
    )
    detail = "<h3>Tasks with notable failures</h3>" + (
        "\n".join(weak_cards) if weak_cards else '<p class="muted">No task below 90%.</p>'
    )
    return stats + table + detail


def render(hh: dict, basic: dict | None, fail_map: dict) -> str:
    hh_body = hh_section(hh)
    if basic:
        basic_body = basic_section(basic, fail_map)
        basic_block = f"""
  <h2 id="base">Base tasks — 20-run suite</h2>
  <p class="muted">
    Expert <code>play_once</code>, seeds <code>0..4</code> for each of
    <code>default</code> / <code>opt1</code> / <code>opt2</code> / <code>opt1+2</code>
    (20 runs per task). Source: <code>logs/basic_sweep_results.json</code>
    (from <code>logs/basic_sweep_n5_allscen.log</code>).
  </p>
  {basic_body}
"""
    else:
        basic_block = """
  <h2 id="base">Base tasks — 20-run suite</h2>
  <p class="muted"><em>Results pending — sweep still running.</em></p>
"""

    detail = r'''
  <h2 id="hh-detail">Household failure analysis (weak tasks)</h2>
  <div class="card focus" id="pour_beer">
    <h3><code>pour_beer</code> <span class="pill fail">1/10</span></h3>
    <ul class="reasons">
      <li><strong>Plan fails (seeds 0–4, 6–8)</strong> — expert aborts before a successful pour (often overflow / unstable foam path).</li>
      <li><strong>Check fail (seed 5)</strong> — plan completes but fill/foam criteria miss; only <strong>seed 9</strong> passes.</li>
    </ul>
  </div>
  <div class="card focus" id="trap_bug">
    <h3><code>trap_bug</code> <span class="pill warn">6/10</span></h3>
    <ul class="reasons">
      <li><strong>Check fails (seeds 1, 2, 5, 9)</strong> — trap placement misses the moving bug (plan OK).</li>
    </ul>
  </div>
  <div class="card focus" id="cook_food">
    <h3><code>cook_food</code> / <code>cook_food_timer</code> <span class="pill warn">7/10 · 6/10</span></h3>
    <ul class="reasons">
      <li><strong>UnStableError</strong> — food not in pan after place (shared seeds 0, 8) or leaves pan after retreat (timer seed 9).</li>
      <li><strong>Check fail (seed 1)</strong> on both cook variants.</li>
    </ul>
  </div>
  <div class="card focus" id="make_soup">
    <h3><code>make_soup</code> <span class="pill warn">7/10</span></h3>
    <ul class="reasons">
      <li><strong>Plan fails (seeds 2, 6, 7)</strong> — tip/pour incomplete; pieces do not all land in the pot.</li>
    </ul>
  </div>
  <div class="card focus" id="stop_ball">
    <h3><code>stop_ball</code> <span class="pill warn">6/10</span></h3>
    <ul class="reasons">
      <li><strong>Check fails (seeds 1, 5, 6, 8)</strong> — ball reaches the edge / not intercepted in time.</li>
    </ul>
  </div>
'''

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>RoboDynaExp — Sweep Results (Household + Base)</title>
<style>{CSS}</style>
</head>
<body>
<header class="hero">
  <div class="wrap" style="padding-top:0;padding-bottom:0">
    <h1>Expert sweep results — household &amp; base tasks</h1>
    <p class="muted">
      Household: 10 seeds per task. Base: 5 seeds × 4 option conditions (20 runs/task).
      Scripted expert <code>play_once</code> with plan + <code>check_success</code>.
    </p>
    <p class="muted">{RUN_NOTE}</p>
  </div>
</header>
<div class="wrap">
  <div class="callout">
    Overall: household <strong>90/120 (75%)</strong> · base <strong>378/460 (82%)</strong>.
    <code>sort_apples_belt</code> recovered to <strong>14/20</strong> after fixing
    the <code>_episode_timed_out</code> bool/method name clash (was 0/20 TypeError).
  </div>
  <p class="muted">
    Jump: <a href="#household">Household</a> ·
    <a href="#hh-detail">HH failure detail</a> ·
    <a href="#base">Base tasks</a>
  </p>

  <h2 id="household">Household tasks — 10-seed sweep</h2>
  <p class="muted">Source: <code>logs/hh_sweep_results.json</code> (from <code>logs/household_sweep_n10.log</code>).</p>
  {hh_body}
  {detail}
  {basic_block}

  <p class="muted" style="margin-top:2rem">
    Related: <a href="tasks_and_metrics.html">Tasks &amp; Metrics</a> ·
    <a href="success_conditions.html">Success conditions</a>
  </p>
</div>
</body>
</html>
"""


def main() -> int:
    hh = json.loads(HH_JSON.read_text()) if HH_JSON.is_file() else {}
    basic = json.loads(BASIC_JSON.read_text()) if BASIC_JSON.is_file() else None
    fail_map: dict = {}
    if BASIC_LOG.is_file():
        fail_map = parse_basic_failure_reasons(BASIC_LOG.read_text(errors="replace"))
    OUT.write_text(render(hh, basic, fail_map))
    print(f"Wrote {OUT} (hh={bool(hh)} basic={basic is not None})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
