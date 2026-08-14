"""Per-user human-experiment logs under ``data/exp_logs/<user>/``.

Robot and keyboard sessions are separate files in that folder:
``user_robot.json`` and ``user_keyboard.json``.

Used by ``experiment_gui.py`` and by the base / household task GUIs when they
run in experiment mode (``ROBODYNA_EXPERIMENT=1``).
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_LOGS_DIR = REPO_ROOT / "data" / "exp_logs"

EXPERIMENT_ENV = "ROBODYNA_EXPERIMENT"
EXPERIMENT_USER_ENV = "ROBODYNA_EXPERIMENT_USER"
EXPERIMENT_LOG_ENV = "ROBODYNA_EXPERIMENT_LOG"

LOG_CONTROLLER_TAGS = ("robot", "keyboard")
_KEYBOARD_ALIASES = {
    "keyboard",
    "keyboard+mouse",
    "keyboardmouse",
    "key+mouse",
    "keymouse",
    "km",
}

EXPERIENCE_QUESTIONS = (
    ("video_games", "Do you have experience playing video games?"),
    ("keyboard_video_games", "Do you have experience playing video games using a keyboard?"),
    ("robotic_simulators", "Do you have previous experience with robotic simulators?"),
)

# Base suite: 23 tasks × 4 scenarios. Household: 12 tasks.
# Tutorial parts are unlimited practice and are never written to these logs.
BASE_SCENARIO_SLOTS = 23 * 4
HOUSEHOLD_TASK_SLOTS = 12

_TRUE = {"1", "true", "yes", "on"}


def experiment_mode() -> bool:
    return os.environ.get(EXPERIMENT_ENV, "").strip().lower() in _TRUE


def slugify_user_name(name: str) -> str:
    """Stable folder tag for a display name (case-insensitive)."""
    text = str(name or "").strip().lower()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"[-\s]+", "_", text).strip("_")
    return text or "user"


def user_dir(name_or_slug: str) -> Path:
    return EXP_LOGS_DIR / slugify_user_name(name_or_slug)


def log_controller_tag(value) -> str:
    """Folder filename tag: ``robot`` or ``keyboard`` (covers keyboard+mouse)."""
    text = str(value or "robot").strip().lower().replace("_", "+").replace(" ", "")
    if text in _KEYBOARD_ALIASES:
        return "keyboard"
    return "robot"


def session_control_mode() -> str:
    """Canonical play mode for the GUI / ``ROBODYNA_CONTROL``: ``robot`` or ``keyboard+mouse``."""
    return "keyboard+mouse" if log_controller_tag(os.environ.get("ROBODYNA_CONTROL")) == "keyboard" else "robot"


def current_controller() -> str:
    return log_controller_tag(os.environ.get("ROBODYNA_CONTROL", "robot"))


def user_log_filename(controller: str | None = None) -> str:
    return f"user_{log_controller_tag(controller if controller is not None else current_controller())}.json"


def user_log_path(name_or_slug: str, controller: str | None = None) -> Path:
    """``data/exp_logs/<user>/user_robot.json`` or ``user_keyboard.json``."""
    return user_dir(name_or_slug) / user_log_filename(controller)


def _iter_user_log_files(folder: Path):
    """Yield ``(tag, path)`` for tagged logs, then legacy ``user.json``."""
    if not folder.is_dir():
        return
    for tag in LOG_CONTROLLER_TAGS:
        path = folder / f"user_{tag}.json"
        if path.is_file():
            yield tag, path
    legacy = folder / "user.json"
    if legacy.is_file():
        yield "legacy", legacy


def iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_user_log(path: Path | None = None) -> dict[str, Any] | None:
    if path is None:
        raw = os.environ.get(EXPERIMENT_LOG_ENV, "").strip()
        path = Path(raw) if raw else None
    if path is None or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def save_user_log(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(data)
    data["updated_at"] = iso_now()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def find_user(name: str) -> dict[str, Any] | None:
    """Return an existing user log if this name (or its slug) already exists."""
    slug = slugify_user_name(name)
    folder = user_dir(slug)
    for _tag, path in _iter_user_log_files(folder):
        data = load_user_log(path)
        if data:
            return data
    if not EXP_LOGS_DIR.is_dir():
        return None
    needle = str(name or "").strip().casefold()
    for folder in EXP_LOGS_DIR.iterdir():
        if not folder.is_dir():
            continue
        for _tag, candidate in _iter_user_log_files(folder):
            other = load_user_log(candidate)
            if not other:
                continue
            if str(other.get("user_name", "")).strip().casefold() == needle:
                return other
            if str(other.get("user_id", "")).strip() == slug:
                return other
    return None


def _empty_user_log(
    *,
    slug: str,
    display: str,
    controller: str,
    path: Path,
    experience: dict[str, Any] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    now = iso_now()
    return {
        "user_id": slug,
        "user_name": display,
        "controller": controller,
        "created_at": created_at or now,
        "updated_at": now,
        "experience": dict(experience or {}),
        "completed_keys": [],
        "play_counts": {},
        "success_counts": {},
        "plays": [],
        "log_path": str(path),
    }


def ensure_controller_log(
    name_or_slug: str,
    *,
    display: str | None = None,
    experience: dict[str, Any] | None = None,
    controller: str | None = None,
    template: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load or create the robot/keyboard log inside this user's folder.

    Progress is per controller: ``user_robot.json`` and ``user_keyboard.json``
    do not share plays. Experience is copied from any existing file for the user.
    A legacy ``user.json`` is migrated into ``user_robot.json`` when that tagged
    file does not exist yet.
    """
    tag = log_controller_tag(controller if controller is not None else current_controller())
    slug = slugify_user_name(name_or_slug)
    folder = user_dir(slug)
    path = folder / f"user_{tag}.json"
    existing = load_user_log(path)
    if existing:
        existing["controller"] = tag
        existing["log_path"] = str(path)
        return existing

    legacy = folder / "user.json"
    if legacy.is_file():
        data = load_user_log(legacy)
        if data:
            file_tag = log_controller_tag(data.get("controller") or "robot")
            if file_tag == tag:
                data["controller"] = tag
                data["log_path"] = str(path)
                save_user_log(path, data)
                return data

    source = template
    if source is None:
        for _other_tag, other_path in _iter_user_log_files(folder):
            source = load_user_log(other_path)
            if source:
                break
    display_name = (
        display
        or (source or {}).get("user_name")
        or str(name_or_slug or "").strip()
        or slug
    )
    exp = experience if experience is not None else (source or {}).get("experience") or {}
    data = _empty_user_log(
        slug=slug,
        display=str(display_name),
        controller=tag,
        path=path,
        experience=exp,
        created_at=(source or {}).get("created_at"),
    )
    save_user_log(path, data)
    return data


def create_user(
    name: str,
    experience: dict[str, str],
    controller: str | None = None,
) -> dict[str, Any]:
    display = str(name or "").strip()
    answers = {
        key: str(experience.get(key, "")).strip().lower()
        for key, _label in EXPERIENCE_QUESTIONS
    }
    return ensure_controller_log(
        display,
        display=display,
        experience=answers,
        controller=controller,
    )


def play_key(suite: str, task: str, scenario: str | None = None) -> str:
    suite = str(suite or "").strip()
    task = str(task or "").strip()
    if suite == "household" or not scenario:
        return f"{suite}:{task}"
    return f"{suite}:{task}:{scenario}"


def completed_keys(log: dict[str, Any] | None = None) -> set[str]:
    data = log if log is not None else load_user_log()
    if not data:
        return set()
    keys = data.get("completed_keys") or []
    return {str(k) for k in keys if k}


def is_completed(suite: str, task: str, scenario: str | None = None) -> bool:
    """True after at least one SUCCESS/FAILURE. Prefer ``is_slot_locked`` for limits."""
    return play_key(suite, task, scenario) in completed_keys()


def _play_matches_slot(play: dict[str, Any], suite: str, task: str, scenario: str | None) -> bool:
    if str(play.get("suite") or "") != suite:
        return False
    if str(play.get("task") or "") != task:
        return False
    if suite != "household":
        if str(play.get("scenario") or "") != str(scenario or ""):
            return False
    return True


def terminal_play_count(
    suite: str,
    task: str,
    scenario: str | None = None,
    log: dict[str, Any] | None = None,
) -> int:
    """How many SUCCESS/FAILURE plays this slot has (Stop / close do not count)."""
    data = log if log is not None else load_user_log()
    if not data:
        return 0
    key = play_key(suite, task, scenario)
    counts = data.get("play_counts") or {}
    if key in counts:
        try:
            return max(0, int(counts[key]))
        except (TypeError, ValueError):
            pass
    n = 0
    for play in data.get("plays") or []:
        if not isinstance(play, dict):
            continue
        if not _play_matches_slot(play, suite, task, scenario):
            continue
        if counts_as_completed(str(play.get("result") or "")):
            n += 1
    if n == 0 and key in completed_keys(data):
        return 1
    return n


def terminal_success_count(
    suite: str,
    task: str,
    scenario: str | None = None,
    log: dict[str, Any] | None = None,
) -> int:
    """How many of this slot's terminal plays were SUCCESS."""
    data = log if log is not None else load_user_log()
    if not data:
        return 0
    key = play_key(suite, task, scenario)
    counts = data.get("success_counts") or {}
    if key in counts:
        try:
            return max(0, int(counts[key]))
        except (TypeError, ValueError):
            pass
    n = 0
    for play in data.get("plays") or []:
        if not isinstance(play, dict):
            continue
        if not _play_matches_slot(play, suite, task, scenario):
            continue
        if str(play.get("result") or "") == "SUCCESS":
            n += 1
    return n


def slot_success_label(
    suite: str,
    task: str,
    scenario: str | None = None,
    log: dict[str, Any] | None = None,
) -> str:
    """``m/n`` successes over terminal plays, for a grayed-out slot."""
    total = terminal_play_count(suite, task, scenario, log=log)
    ok = terminal_success_count(suite, task, scenario, log=log)
    if total <= 0:
        return "0/0"
    ok = min(ok, total)
    return f"{ok}/{total}"


def is_slot_locked(
    suite: str,
    task: str,
    scenario: str | None = None,
    *,
    max_plays: int | None,
    log: dict[str, Any] | None = None,
) -> bool:
    """True when this slot has used up its SUCCESS/FAILURE play budget."""
    if max_plays is None:
        return False
    try:
        limit = int(max_plays)
    except (TypeError, ValueError):
        return False
    if limit <= 0:
        return False
    return terminal_play_count(suite, task, scenario, log=log) >= limit


def result_from_exit_code(exit_code: int | None, *, stopped: bool = False) -> str:
    if stopped:
        return "stopped"
    if exit_code == 0:
        return "SUCCESS"
    if exit_code == 10:
        return "FAILURE"
    if exit_code == 2:
        return "closed"
    if exit_code is None:
        return "unknown"
    return "error"


def counts_as_completed(result: str) -> bool:
    return result in ("SUCCESS", "FAILURE")


def progress_counts(
    log: dict[str, Any] | None = None,
    *,
    base_task_names: list[str] | None = None,
    household_task_names: list[str] | None = None,
    n_scenarios: int = 4,
    cfg=None,
) -> dict[str, Any]:
    """Slots that have reached their play limit vs slots that have a limit.

    Unlimited slots (``plays_per_scenario: null``) are omitted from the totals.
    """
    try:
        from experiment_config import (
            BASE_TASK_TABLE,
            HOUSEHOLD_TASK_TABLE,
            load_experiment_config,
        )
    except ImportError:
        from interactive.experiment_config import (
            BASE_TASK_TABLE,
            HOUSEHOLD_TASK_TABLE,
            load_experiment_config,
        )

    if cfg is None:
        cfg = load_experiment_config()
    scenarios = ("default", "opt1", "opt2", "opt1+2")[: max(1, int(n_scenarios))]
    if base_task_names is None:
        base_names = [key for _n, key, _label in BASE_TASK_TABLE]
    else:
        base_names = list(base_task_names)
    if household_task_names is None:
        household_names = [key for _n, key, _label in HOUSEHOLD_TASK_TABLE]
    else:
        household_names = list(household_task_names)

    scenario_done = {name: 0 for name in scenarios}
    scenario_total = {name: 0 for name in scenarios}
    base_done = 0
    base_total = 0
    for task in base_names:
        for scenario in scenarios:
            limit = cfg.max_plays("base", task, scenario)
            if limit is None:
                continue
            base_total += 1
            scenario_total[scenario] += 1
            if terminal_play_count("base", task, scenario, log=log) >= int(limit):
                base_done += 1
                scenario_done[scenario] += 1
    household_done = 0
    household_total = 0
    for task in household_names:
        limit = cfg.max_plays("household", task)
        if limit is None:
            continue
        household_total += 1
        if terminal_play_count("household", task, log=log) >= int(limit):
            household_done += 1
    return {
        "base_done": base_done,
        "base_total": base_total,
        "household_done": household_done,
        "household_total": household_total,
        "scenario_done": scenario_done,
        "scenario_total": scenario_total,
    }


def append_play(
    *,
    suite: str,
    task: str,
    task_label: str | None = None,
    scenario: str | None = None,
    controller: str | None = None,
    seed: int | None = None,
    exit_code: int | None = None,
    payload: dict[str, Any] | None = None,
    stopped: bool = False,
    wall_fallback_s: float | None = None,
    write_entry: bool = True,
) -> dict[str, Any] | None:
    """Record one play. Always bumps ``play_counts`` on SUCCESS/FAILURE.

    When ``write_entry`` is false, skip appending the detailed ``plays`` row
    (the slot is still counted so play limits work).
    """
    path_raw = os.environ.get(EXPERIMENT_LOG_ENV, "").strip()
    path = Path(path_raw) if path_raw else None
    if path is None:
        user = os.environ.get(EXPERIMENT_USER_ENV, "").strip()
        if user:
            path = user_log_path(user, controller or current_controller())
    if path is None:
        return None
    payload = dict(payload or {})
    data = load_user_log(path) or {
        "user_id": slugify_user_name(os.environ.get(EXPERIMENT_USER_ENV, "user")),
        "user_name": os.environ.get(EXPERIMENT_USER_ENV, "user"),
        "created_at": iso_now(),
        "experience": {},
        "completed_keys": [],
        "play_counts": {},
        "success_counts": {},
        "plays": [],
        "log_path": str(path),
    }
    tag = log_controller_tag(
        controller or payload.get("controller") or data.get("controller") or current_controller()
    )
    data["controller"] = tag
    data["log_path"] = str(path)
    result = result_from_exit_code(exit_code, stopped=stopped)
    if result in ("unknown", "error") and payload.get("ok") is True:
        result = "SUCCESS"
    elif result in ("unknown", "error") and payload.get("ok") is False:
        result = "FAILURE"

    time_block = payload.get("time") if isinstance(payload.get("time"), dict) else {}
    wall = time_block.get("wall_clock_s")
    if wall is None and wall_fallback_s is not None:
        wall = round(float(wall_fallback_s), 4)
        time_block = dict(time_block)
        time_block["wall_clock_s"] = wall

    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    if "success" not in metrics and payload.get("ok") is not None:
        metrics = dict(metrics)
        metrics["success"] = bool(payload.get("ok"))

    entry = {
        "played_at": iso_now(),
        "suite": suite,
        "task": task,
        "task_label": task_label or task,
        "scenario": scenario,
        "controller": controller or payload.get("controller") or "",
        "seed": seed if seed is not None else payload.get("seed"),
        "result": result,
        "detail": payload.get("detail") or "",
        "condition": payload.get("condition") or "",
        "exit_code": exit_code,
        "metrics": metrics,
        "time": {
            "wall_clock_s": time_block.get("wall_clock_s"),
            "simulation_s": time_block.get("simulation_s"),
            "simulation_steps": time_block.get("simulation_steps"),
        },
    }
    counted = False
    if counts_as_completed(result):
        key = play_key(suite, task, scenario)
        counts = dict(data.get("play_counts") or {})
        try:
            counts[key] = int(counts.get(key, 0) or 0) + 1
        except (TypeError, ValueError):
            counts[key] = 1
        data["play_counts"] = counts
        if result == "SUCCESS":
            wins = dict(data.get("success_counts") or {})
            try:
                wins[key] = int(wins.get(key, 0) or 0) + 1
            except (TypeError, ValueError):
                wins[key] = 1
            data["success_counts"] = wins
        counted = True
        done = list(data.get("completed_keys") or [])
        if key not in done:
            done.append(key)
        data["completed_keys"] = done
    if write_entry:
        plays = list(data.get("plays") or [])
        plays.append(entry)
        data["plays"] = plays
    elif not counted:
        return None
    save_user_log(path, data)
    return entry


def stamp_child_env(
    env: dict[str, str],
    *,
    controller: str,
    seed: int,
    scenario: str | None = None,
) -> dict[str, str]:
    env["ROBODYNA_CONTROL"] = str(controller)
    env["ROBODYNA_SEED"] = str(seed)
    if scenario:
        env["ROBODYNA_SCENARIO"] = str(scenario)
    return env


def child_experiment_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Env for a spawned task GUI, preserving the logged-in user."""
    env = dict(base if base is not None else os.environ)
    env[EXPERIMENT_ENV] = "1"
    for key in (EXPERIMENT_USER_ENV, EXPERIMENT_LOG_ENV):
        value = os.environ.get(key, "").strip()
        if value:
            env[key] = value
    return env


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)
