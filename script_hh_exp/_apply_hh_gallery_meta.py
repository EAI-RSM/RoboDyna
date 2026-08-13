#!/usr/bin/env python3
"""Apply /tmp/hh_gallery_meta.json into task_gallery/index.html + README tables."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_GALLERY_ROOT = Path(
    os.environ.get(
        "TASK_GALLERY_ROOT",
        "/home/aras/Desktop/workspace/task_gallery/final_task_demos",
    )
)
GALLERY_HTML = _GALLERY_ROOT.parent / "index.html"
README = ROOT / "README.md"
HH_README = ROOT / "interactive/household/README.md"
META = Path("/tmp/hh_gallery_meta.json")

# Prefer README order (matches prior README household section).
README_ORDER = [
    "trap_bug",
    "catch_cup",
    "catch_mouse_object_drop",
    "stop_ball",
    "clean_table",
    "fill_coffee_jar",
    "pour_beer",
    "boil_milk",
    "cook_food",
    "cook_food_timer",
    "make_soup",
    "measure_ingredient",
]


def js_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'")


def build_household_js(meta: list[dict]) -> str:
    by_task = {m["task"]: m for m in meta}
    lines = ["    const householdTasks = ["]
    for task in README_ORDER:
        m = by_task.get(task)
        if not m:
            continue
        files = m["files"]
        labels = m["labels"]
        captions = m["captions"]
        blurb = m.get("blurb", task)
        files_js = ",".join(f"'{f}'" for f in files)
        labels_js = ",".join(f"'{js_escape(l)}'" for l in labels)
        caps_js = ",".join(f"'{js_escape(c)}'" for c in captions)
        lines.append(
            f"      ['{task}', '{js_escape(blurb)}', [{files_js}], [{labels_js}], [{caps_js}]],"
        )
    lines.append("    ];")
    return "\n".join(lines)


def update_gallery_html(meta: list[dict]) -> None:
    text = GALLERY_HTML.read_text()
    block = build_household_js(meta)
    new_text, n = re.subn(
        r"    const householdTasks = \[[\s\S]*?\n    \];",
        block,
        text,
        count=1,
    )
    if n != 1:
        raise SystemExit(f"Failed to replace householdTasks block (n={n})")
    GALLERY_HTML.write_text(new_text)
    print(f"Updated {GALLERY_HTML}")


def readme_cell(task: str, m: dict) -> str:
    """Two-GIF cell for README household table (head-camera demos)."""
    files = m["files"]
    captions = m["captions"]
    parts = []
    for f, cap in zip(files, captions):
        parts.append(
            f'<div align="center"><img src="final_task_demos/{task}/{f}" '
            f'width="160" height="120"/><br><sub>{cap}</sub></div>'
        )
    return " ".join(parts) if parts else "—"


def update_main_readme(meta: list[dict]) -> None:
    by_task = {m["task"]: m for m in meta}
    text = README.read_text()
    # Replace household table body between header and next ## heading.
    header = (
        "| Task | Demo |\n|---|---|\n"
        if "| Task | Demo |" in text[text.find("## Household Tasks") :]
        else None
    )
    # Prefer a 2-column Demo layout: Success / Failure (or Success / Success)
    start = text.find("## Household Tasks")
    if start < 0:
        raise SystemExit("## Household Tasks not found in README")
    # Find table start
    t0 = text.find("| Task |", start)
    if t0 < 0:
        raise SystemExit("Household table not found")
    # End at blank line after table or next ##
    t1 = text.find("\n## ", t0)
    if t1 < 0:
        t1 = len(text)
    # Also stop before "\n\n## " style
    after_header = text.find("\n|---|", t0)
    row_start = text.find("\n", after_header) + 1

    rows = [
        "| Task | Success / Failure demos |",
        "|---|---|",
    ]
    for task in README_ORDER:
        m = by_task.get(task)
        if not m:
            continue
        blurb = m.get("blurb", task)
        cell = readme_cell(task, m)
        rows.append(f"| **`{task}`**<br><sub>{blurb}</sub> | {cell} |")
    new_table = "\n".join(rows) + "\n"
    new_text = text[:t0] + new_table + text[t1:]
    # Normalize lead-in to head-camera wording (any prior side-by-side phrasing).
    for old in (
        "Each row shows the latest side-by-side expert demo (head camera + top-down).",
        "Each row shows success and failure (or two successes) side-by-side expert demos (head camera + top-down).",
        "Each row shows success and failure (or two successes) head-camera expert demos.",
    ):
        new_text = new_text.replace(
            old,
            "Each row shows success and failure (or two successes) head-camera expert demos.",
        )
    README.write_text(new_text)
    print(f"Updated {README}")


def update_hh_readme(meta: list[dict]) -> None:
    by_task = {m["task"]: m for m in meta}
    text = HH_README.read_text()
    start = text.find("| Task | Demo |")
    if start < 0:
        raise SystemExit("script_hh_exp README table not found")
    end = text.find("\nRefresh GUI", start)
    if end < 0:
        end = text.find("\n```", start)
    rows = ["| Task | Demo |", "|---|---|"]
    for task in README_ORDER:
        m = by_task.get(task)
        if not m:
            continue
        imgs = " ".join(
            f'<img src="../final_task_demos/{task}/{f}" width="200"/>'
            for f in m["files"]
        )
        rows.append(f"| **`{task}`** | {imgs} |")
    new_table = "\n".join(rows) + "\n\n"
    HH_README.write_text(text[:start] + new_table + text[end:])
    print(f"Updated {HH_README}")


def main() -> int:
    if not META.is_file():
        raise SystemExit(f"Missing {META}")
    meta = json.loads(META.read_text())
    update_gallery_html(meta)
    update_main_readme(meta)
    update_hh_readme(meta)
    print(f"Applied {len(meta)} household tasks from {META}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
