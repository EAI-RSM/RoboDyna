#!/usr/bin/env python3
"""Build instructions/tasks_final_seed_test.html from base 5×4 seed-test results.

Reads JSON (preferred) or falls back to parsing the latest
``logs/basic_test_and_demos_*.log``. Does not modify tasks or solvers.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_HTML = ROOT / "instructions" / "tasks_final_seed_test.html"
OUT_JSON = ROOT / "logs" / "tasks_final_seed_test.json"
SCENARIOS = ("default", "opt1", "opt2", "opt1+2")
N_SEEDS = 5


def rate_class(n_ok: int, n: int) -> str:
    if n <= 0:
        return "mid"
    r = n_ok / n
    if r >= 0.8:
        return "good"
    if r >= 0.4:
        return "mid"
    return "bad"


def fmt_seeds(xs: list[int]) -> str:
    return ", ".join(str(x) for x in xs) if xs else "—"


def parse_log(log_text: str) -> dict:
    """Parse completed scenario blocks from a run log."""
    results: dict[str, dict[str, dict]] = {}
    pat = re.compile(
        r"===== (\S+) \[([^\]]+)\] \(\d+ seeds\) =====\n"
        r"(.*?)\n  → (\d+)/(\d+) ok=(\[[^\]]*\]) fail=(\[[^\]]*\])",
        re.S,
    )
    for m in pat.finditer(log_text):
        task, scenario, _body, okn, n, ok_s, fail_s = m.groups()
        ok = ast.literal_eval(ok_s)
        fail = ast.literal_eval(fail_s)
        results.setdefault(task, {})[scenario] = {
            "ok": list(ok),
            "fail": list(fail),
            "n": int(n),
        }
        assert int(okn) == len(ok)
    return results


def load_results(json_path: Path | None, log_path: Path | None) -> tuple[dict, str]:
    if json_path and json_path.is_file():
        data = json.loads(json_path.read_text())
        # Normalize: ensure ok/fail lists
        out = {}
        for task, scens in data.items():
            out[task] = {}
            for scen, row in scens.items():
                out[task][scen] = {
                    "ok": list(row.get("ok", [])),
                    "fail": list(row.get("fail", [])),
                    "n": int(row.get("n", N_SEEDS)),
                }
        return out, f"JSON {json_path.relative_to(ROOT)}"

    if log_path and log_path.is_file():
        return parse_log(log_path.read_text(errors="replace")), (
            f"log {log_path.relative_to(ROOT)}"
        )

    raise SystemExit("Need --json or --log with existing results")


def latest_log() -> Path | None:
    logs = sorted(
        (ROOT / "logs").glob("basic_test_and_demos_*.log"),
        key=lambda p: p.stat().st_mtime,
    )
    return logs[-1] if logs else None


def cell_rate(n_ok: int, n: int) -> str:
    cls = rate_class(n_ok, n)
    return f"<td class='rate {cls}'>{n_ok}/{n}</td>"


def cell_overall(n_ok: int, n: int) -> str:
    cls = rate_class(n_ok, n)
    pct = int(round(100.0 * n_ok / n)) if n else 0
    return f"<td class='rate {cls}'>{n_ok}/{n} ({pct}%)</td>"


def detail_row(task: str, scenario: str, row: dict) -> str:
    ok = row["ok"]
    fail = row["fail"]
    n = int(row.get("n", N_SEEDS))
    n_ok = len(ok)
    pct = int(round(100.0 * n_ok / n)) if n else 0
    cls = rate_class(n_ok, n)
    return (
        f"<tr><td class='mono'>{task}</td><td>{scenario}</td>"
        f"<td><span class='bar'><i style='width:{pct}%'></i></span>"
        f"<span class='rate {cls}'>{n_ok}/{n} ({pct}%)</span></td>"
        f"<td class='seeds'>ok: {fmt_seeds(ok)}<br>fail: {fmt_seeds(fail)}</td></tr>"
    )


def render(results: dict, source: str, incomplete: bool) -> str:
    now = datetime.now(timezone.utc).isoformat()
    task_order = list(results.keys())

    total_ok = total_n = 0
    summary_rows = []
    detail_rows = []
    complete_tasks = 0

    for task in task_order:
        scens = results[task]
        cells = []
        t_ok = t_n = 0
        for scen in SCENARIOS:
            if scen not in scens:
                cells.append("<td class='rate mid'>—</td>")
                continue
            row = scens[scen]
            n = int(row.get("n", N_SEEDS))
            n_ok = len(row["ok"])
            cells.append(cell_rate(n_ok, n))
            t_ok += n_ok
            t_n += n
            detail_rows.append(detail_row(task, scen, row))
        if all(s in scens for s in SCENARIOS):
            complete_tasks += 1
        total_ok += t_ok
        total_n += t_n
        if t_n:
            summary_rows.append(
                f"<tr><td class='mono'>{task}</td>"
                + "".join(cells)
                + cell_overall(t_ok, t_n)
                + "</tr>"
            )

    status = (
        f"IN PROGRESS — {complete_tasks}/{len(task_order)} tasks fully scored "
        f"({total_n} runs so far)."
        if incomplete
        else f"Complete — {complete_tasks} tasks × 4 conditions × {N_SEEDS} seeds."
    )
    overall_cls = rate_class(total_ok, total_n) if total_n else "mid"
    overall_pct = int(round(100.0 * total_ok / total_n)) if total_n else 0

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Base tasks — final seed test (5×4)</title>
<style>
  :root {{
    --bg: #0f1419; --panel: #1a222c; --border: #334155;
    --text: #e8eef4; --muted: #9aabbc;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    background: radial-gradient(1200px 600px at 10% -10%, #1a2a3a 0%, var(--bg) 55%);
    color: var(--text); line-height: 1.45;
  }}
  header.hero {{
    padding: 2rem 1.5rem 1.25rem; border-bottom: 1px solid var(--border);
    background: linear-gradient(180deg, #15202b, transparent);
  }}
  h1 {{ margin: 0 0 0.4rem; font-size: 1.75rem; letter-spacing: -0.02em; }}
  h2 {{ margin: 1.75rem 0 0.75rem; font-size: 1.15rem; }}
  .muted {{ color: var(--muted); max-width: 85ch; }}
  .wrap {{ max-width: 1200px; margin: 0 auto; padding: 1rem 1.5rem 3rem; }}
  .stats {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 0.75rem; margin: 1rem 0 1.5rem;
  }}
  .stat {{
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 10px; padding: 0.85rem 1rem;
  }}
  .stat b {{ display: block; font-size: 1.4rem; }}
  .stat span {{ color: var(--muted); font-size: 0.85rem; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
  th, td {{
    text-align: left; padding: 0.45rem 0.5rem;
    border-bottom: 1px solid var(--border); vertical-align: top;
  }}
  th {{
    color: var(--muted); font-weight: 600; font-size: 0.75rem;
    text-transform: uppercase; letter-spacing: 0.04em;
  }}
  .mono {{ font-family: "IBM Plex Mono", ui-monospace, monospace; color: #c6e0f5; }}
  .rate {{ font-weight: 700; white-space: nowrap; }}
  .rate.good {{ color: #8fd9b5; }}
  .rate.mid {{ color: #f0d48a; }}
  .rate.bad {{ color: #f0a0a0; }}
  .bar {{
    display: inline-block; height: 8px; width: 56px; background: #2a3542;
    border-radius: 4px; overflow: hidden; vertical-align: middle; margin-right: 0.35rem;
  }}
  .bar > i {{
    display: block; height: 100%;
    background: linear-gradient(90deg, #2f9e6b, #5b9fd4);
  }}
  .seeds {{ font-size: 0.85rem; color: var(--muted); }}
  .card {{
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 12px; padding: 1rem 1.1rem; margin: 0.85rem 0;
  }}
  .callout {{
    background: rgba(91,159,212,0.08); border: 1px solid rgba(91,159,212,0.35);
    border-radius: 10px; padding: 0.75rem 1rem; margin: 0.75rem 0 1.25rem;
    color: var(--muted); font-size: 0.9rem;
  }}
  .pill {{
    display: inline-block; font-size: 0.68rem; padding: 0.1rem 0.4rem;
    border-radius: 999px; border: 1px solid #6a5120; color: #f0d48a;
    background: rgba(184,134,11,0.15); margin-left: 0.35rem; vertical-align: middle;
  }}
</style>
</head>
<body>
<header class="hero">
  <div class="wrap">
    <h1>Base tasks — final seed test</h1>
    <p class="muted">5 seeds × 4 conditions (default / opt1 / opt2 / opt1+2) per task.
    Scripted expert only — no task/solver edits.
    Generated {now}.</p>
  </div>
</header>
<main class="wrap">
  <div class="stats">
    <div class="stat"><b class="rate {overall_cls}">{total_ok}/{total_n}</b><span>overall ({overall_pct}%)</span></div>
    <div class="stat"><b>{complete_tasks}</b><span>tasks complete (of 23)</span></div>
    <div class="stat"><b>{len(detail_rows)}</b><span>scenario blocks scored</span></div>
    <div class="stat"><b>{N_SEEDS}×4</b><span>seeds × conditions</span></div>
  </div>

  <div class="callout">
    {status}
    Source: <span class="mono">{source}</span>.
    {"<span class='pill'>partial</span>" if incomplete else ""}
  </div>

  <h2>Per-task summary</h2>
  <div class="card">
    <table>
      <thead><tr><th>Task</th><th>default</th><th>opt1</th><th>opt2</th><th>opt1+2</th><th>Overall</th></tr></thead>
      <tbody>
        {"".join(summary_rows)}
      </tbody>
    </table>
  </div>

  <h2>Per-condition detail</h2>
  <div class="card">
    <table>
      <thead><tr><th>Task</th><th>Condition</th><th>Success</th><th>Seeds</th></tr></thead>
      <tbody>
        {"".join(detail_rows)}
      </tbody>
    </table>
  </div>
  <p class="muted">JSON: <span class="mono">logs/tasks_final_seed_test.json</span>
  · also: <span class="mono">logs/basic_sweep_results.json</span> (written when sweep finishes)</p>
</main>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--log", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=OUT_HTML)
    ap.add_argument(
        "--prefer-json",
        action="store_true",
        help="Use basic_sweep_results.json if present and complete (23×4)",
    )
    ns = ap.parse_args()

    json_path = ns.json
    log_path = ns.log or latest_log()

    if ns.prefer_json or json_path:
        cand = json_path or (ROOT / "logs" / "basic_sweep_results.json")
        if cand.is_file():
            data = json.loads(cand.read_text())
            n_scen = sum(len(v) for v in data.values())
            # Only trust JSON when it looks like a finished 23×4 run matching the live log age,
            # or when explicitly passed via --json.
            if json_path is not None or n_scen >= 23 * 4:
                results, source = load_results(cand, None)
            else:
                results, source = load_results(None, log_path)
        else:
            results, source = load_results(None, log_path)
    else:
        # Prefer live log while sweep is running (JSON may be stale from a prior run).
        results, source = load_results(None, log_path)

    incomplete = any(
        not all(s in results.get(t, {}) for s in SCENARIOS) for t in results
    ) or len(results) < 23

    # Persist snapshot JSON for this HTML.
    snap = {
        t: {
            s: {"ok": results[t][s]["ok"], "fail": results[t][s]["fail"]}
            for s in results[t]
        }
        for t in results
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(snap, indent=2) + "\n")

    html = render(results, source, incomplete=incomplete)
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    ns.out.write_text(html)
    print(
        f"Wrote {ns.out.relative_to(ROOT)} "
        f"({sum(len(v) for v in results.values())} scenarios, "
        f"{'partial' if incomplete else 'complete'}) from {source}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
