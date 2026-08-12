#!/usr/bin/env python3
"""Sync final_task_demos → task_gallery and refresh index.html from sweep results."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "final_task_demos"
_GALLERY_ROOT = Path(
    os.environ.get(
        "TASK_GALLERY_ROOT",
        "/home/aras/Desktop/workspace/task_gallery/final_task_demos",
    )
)
GALLERY = _GALLERY_ROOT
GALLERY_HTML = _GALLERY_ROOT.parent / "index.html"
HH_JSON = Path("/tmp/hh_sweep_results.json")
if not HH_JSON.is_file():
    HH_JSON = ROOT / "logs/hh_sweep_results.json"

FFMPEG = "ffmpeg"

HH_ORDER = [
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

HH_BLURBS = {
    "trap_bug": "Trap a scurrying cockroach, spider, or ant under the office bookshelf with a transparent glass box.",
    "boil_milk": "Turn the stove on so milk rises in the pot, then shut it off before the milk overflows.",
    "fill_coffee_jar": "Press the coffee dispenser's lid to fill a marked glass jar to the target fill line.",
    "pour_beer": "Hold the draft-tap button to pour beer into a mug; foam ramps while held; click the finish bell to score. Overflow fails.",
    "cook_food": "Drop meat, sausage, or onion into a pre-lit pan, then shut the stove off at the target doneness (food stays in the pan).",
    "cook_food_timer": "Same as cook_food, with a pie timer that advances while the stove is on and freezes when it is off.",
    "measure_ingredient": "Push a marked jar under an oil nozzle and fill it to the target ring; oil that misses the jar fails.",
    "make_soup": "Tip 2–4 distinct chopping-board vegetables into a pot of water on an already-lit stove without dropping any pieces.",
    "catch_cup": "Push a pillow under a tipping mug so it lands softly instead of hitting the table.",
    "catch_mouse_object_drop": "Place a pillow-lined basket under a shelf object knocked by a scurrying mouse so it does not hit the table.",
    "stop_ball": "Block a table-tennis ball that falls from the shelf and rolls toward the near table edge.",
    "clean_table": "Wipe a spreading coffee spill with a sponge before it reaches a laptop on the opposite side.",
}

# Base tasks that gallery index should list (and stems for demos).
MAIN_EXTRA = {
    "catch_cuboid": {
        "blurb": "Grasp the cuboid or cuboids during their timed pop-up windows.",
        "files": [
            "default_transparent_1cuboid_sidebyside.gif",
            "opt1_catch_two_cuboids_sidebyside.gif",
            "opt2_opaque_1cuboid_sidebyside.gif",
            "opt1+2_catch_two_cuboids_opaque_sidebyside.gif",
        ],
        "conds": [
            "One cuboid; transparent board.",
            "Two simultaneous cuboids; transparent board.",
            "One cuboid; opaque board.",
            "Two simultaneous cuboids; opaque board.",
        ],
    },
    "cook_meat_timer": {
        "blurb": "Cook steak with a pie timer (green→yellow→red) that tracks doneness, then return it to the board.",
        "files": [
            "default_sidebyside.gif",
            "opt1_sidebyside.gif",
            "opt2_sidebyside.gif",
            "opt1+2_sidebyside.gif",
        ],
        "conds": [
            "One station; timer advances on pan contact.",
            "One station; timer runs while the cook key is held.",
            "Two stations; contact cook with a pie timer each.",
            "Two stations; each key holds its own pie timer.",
        ],
    },
}


def gif_from_mp4(mp4: Path, gif: Path, scale: int = 720, fps: int = 8) -> None:
    gif.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            FFMPEG, "-y", "-loglevel", "error", "-i", str(mp4),
            "-vf", f"fps={fps},scale={scale}:-1:flags=lanczos",
            "-loop", "0", str(gif),
        ],
        check=True,
    )


def publish_pair(task: str, mp4: Path, stem: str) -> Path:
    """Copy mp4 + make gif into repo + gallery under stem_sidebyside.*"""
    for root in (REPO / task, GALLERY / task):
        root.mkdir(parents=True, exist_ok=True)
        dst_mp4 = root / f"{stem}.mp4"
        dst_gif = root / f"{stem}.gif"
        shutil.copy2(mp4, dst_mp4)
        gif_from_mp4(mp4, dst_gif)
        print(f"  PUBLISH {dst_gif}")
    return GALLERY / task / f"{stem}.gif"


def sync_globs(task: str, patterns: list[str]) -> None:
    src_dir = REPO / task
    dst_dir = GALLERY / task
    dst_dir.mkdir(parents=True, exist_ok=True)
    for pat in patterns:
        for src in src_dir.glob(pat):
            dst = dst_dir / src.name
            shutil.copy2(src, dst)
            print(f"  SYNC {dst}")


def pick_hh_files(task: str, sweep: dict) -> tuple[list[str], list[str], list[str]]:
    """Choose display files/labels/captions from available gifs + sweep."""
    d = GALLERY / task
    ok = sweep.get("ok", [])
    fail = sweep.get("fail", [])

    def has(name: str) -> bool:
        return (d / name).is_file()

    if fail and ok:
        # Prefer success + failure
        if has("success_sidebyside.gif") and has("failure_sidebyside.gif"):
            return (
                ["success_sidebyside.gif", "failure_sidebyside.gif"],
                ["Success", "Failure case"],
                [f"Success (seed {ok[0]}).", f"Failure case (seed {fail[0]})."],
            )
        if has("success1_sidebyside.gif") and has("failure_sidebyside.gif"):
            return (
                ["success1_sidebyside.gif", "failure_sidebyside.gif"],
                ["Success", "Failure case"],
                [f"Success (seed {ok[0]}).", f"Failure case (seed {fail[0]})."],
            )
    # Two successes
    if has("success1_sidebyside.gif") and has("success2_sidebyside.gif"):
        s1 = ok[0] if ok else 0
        s2 = ok[1] if len(ok) > 1 else s1
        return (
            ["success1_sidebyside.gif", "success2_sidebyside.gif"],
            ["Success", "Success"],
            [f"Success rollout (seed {s1}).", f"Success rollout (seed {s2})."],
        )
    if has("success_sidebyside.gif") and has("failure_sidebyside.gif"):
        return (
            ["success_sidebyside.gif", "failure_sidebyside.gif"],
            ["Success", "Failure case"],
            ["Success.", "Failure case."],
        )
    if has("success_sidebyside.gif") and has("failure2_sidebyside.gif"):
        return (
            ["success_sidebyside.gif", "failure_sidebyside.gif" if has("failure_sidebyside.gif") else "failure2_sidebyside.gif"],
            ["Success", "Failure case"],
            ["Success.", "Failure case."],
        )
    # Fallback: default
    if has("default_sidebyside.gif"):
        return (
            ["default_sidebyside.gif"],
            ["Demo"],
            ["Expert rollout."],
        )
    return [], [], []


def js_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'")


def update_index(hh_meta: list[dict]) -> None:
    text = GALLERY_HTML.read_text()

    # Household block
    lines = ["    const householdTasks = ["]
    for m in hh_meta:
        files_js = ",".join(f"'{f}'" for f in m["files"])
        labels_js = ",".join(f"'{js_escape(l)}'" for l in m["labels"])
        caps_js = ",".join(f"'{js_escape(c)}'" for c in m["captions"])
        lines.append(
            f"      ['{m['task']}', '{js_escape(m['blurb'])}', [{files_js}], [{labels_js}], [{caps_js}]],"
        )
    lines.append("    ];")
    hh_block = "\n".join(lines)
    text, n = __import__("re").subn(
        r"    const householdTasks = \[[\s\S]*?\n    \];",
        hh_block,
        text,
        count=1,
    )
    if n != 1:
        raise SystemExit(f"householdTasks replace failed n={n}")

    # Ensure catch_cuboid + cook_meat_timer appear in mainTasks (replace catch_rat if present)
    import re

    # Replace catch_rat entry with catch_cuboid if catch_rat exists
    if "['catch_rat'" in text and "['catch_cuboid'" not in text:
        cub = MAIN_EXTRA["catch_cuboid"]
        files_js = ",".join(f"'{f}'" for f in cub["files"])
        conds_js = ",".join(f"'{js_escape(c)}'" for c in cub["conds"])
        replacement = (
            f"      ['catch_cuboid', '{js_escape(cub['blurb'])}', "
            f"[{files_js}], [{conds_js}]],"
        )
        text, n = re.subn(
            r"      \['catch_rat',[\s\S]*?\],\n",
            replacement + "\n",
            text,
            count=1,
        )
        print(f"  replaced catch_rat → catch_cuboid (n={n})")
    elif "['catch_cuboid'" not in text:
        # Insert after catch_ramp_ball
        cub = MAIN_EXTRA["catch_cuboid"]
        files_js = ",".join(f"'{f}'" for f in cub["files"])
        conds_js = ",".join(f"'{js_escape(c)}'" for c in cub["conds"])
        entry = (
            f"      ['catch_cuboid', '{js_escape(cub['blurb'])}', "
            f"[{files_js}], [{conds_js}]],\n"
        )
        text = text.replace(
            "['catch_ramp_ball',",
            entry + "      ['catch_ramp_ball',",
            1,
        )
        print("  inserted catch_cuboid")

    if "['cook_meat_timer'" not in text:
        cmt = MAIN_EXTRA["cook_meat_timer"]
        files_js = ",".join(f"'{f}'" for f in cmt["files"])
        conds_js = ",".join(f"'{js_escape(c)}'" for c in cmt["conds"])
        entry = (
            f"      ['cook_meat_timer', '{js_escape(cmt['blurb'])}', "
            f"[{files_js}], [{conds_js}]],\n"
        )
        # after cook_meat line
        text, n = re.subn(
            r"(      \['cook_meat',.*?\],\n)",
            r"\1" + entry,
            text,
            count=1,
        )
        print(f"  inserted cook_meat_timer (n={n})")

    GALLERY_HTML.write_text(text)
    print(f"Updated {GALLERY_HTML}")


def main() -> int:
    # 1) Publish pour_beer success from household_demos if missing in repo
    pb_success = ROOT / "household_demos/pour_beer/success_s1_sidebyside.mp4"
    if pb_success.is_file() and not (REPO / "pour_beer/success_sidebyside.gif").is_file():
        print("=== pour_beer success ===")
        publish_pair("pour_beer", pb_success, "success_sidebyside")

    # 2) Sync base demos missing from gallery
    print("=== sync catch_cuboid ===")
    sync_globs("catch_cuboid", ["*sidebyside.gif", "*sidebyside.mp4"])
    print("=== sync cook_meat_timer ===")
    sync_globs("cook_meat_timer", ["*sidebyside.gif", "*sidebyside.mp4"])

    # 3) Sync all household success/failure gifs from repo → gallery
    print("=== sync household clips ===")
    for task in HH_ORDER:
        sync_globs(task, ["success*_sidebyside.gif", "success*_sidebyside.mp4",
                          "failure*_sidebyside.gif", "failure*_sidebyside.mp4"])

    # 4) For catch_mouse with failures: ensure success_sidebyside exists
    #    (copy success1 → success if needed)
    cm = GALLERY / "catch_mouse_object_drop"
    if (cm / "success1_sidebyside.gif").is_file() and not (cm / "success_sidebyside.gif").is_file():
        for root in (REPO / "catch_mouse_object_drop", cm):
            shutil.copy2(root / "success1_sidebyside.gif", root / "success_sidebyside.gif")
            mp4 = root / "success1_sidebyside.mp4"
            if mp4.is_file():
                shutil.copy2(mp4, root / "success_sidebyside.mp4")
        print("  aliased catch_mouse success1 → success")

    # cook_food: if failure missing but we have only successes, keep success1+2
    # cook_food_timer: ensure success1+2 synced (already)

    # 5) Build household meta from sweep + files on disk
    sweep = json.loads(HH_JSON.read_text()) if HH_JSON.is_file() else {}
    meta = []
    for task in HH_ORDER:
        files, labels, captions = pick_hh_files(task, sweep.get(task, {}))
        if not files:
            print(f"  WARN no display files for {task}")
            continue
        meta.append({
            "task": task,
            "blurb": HH_BLURBS.get(task, task),
            "files": files,
            "labels": labels,
            "captions": captions,
        })
        print(f"  {task}: {files}")

    Path("/tmp/hh_gallery_meta.json").write_text(json.dumps(meta, indent=2))
    update_index(meta)
    print(f"Done — {len(meta)} household tasks in gallery index.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
