"""Per-user human-experiment logs under ``exp_logs/<user>/``.

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
EXP_LOGS_DIR = REPO_ROOT / "exp_logs"

EXPERIMENT_ENV = "ROBODYNA_EXPERIMENT"
EXPERIMENT_USER_ENV = "ROBODYNA_EXPERIMENT_USER"
EXPERIMENT_LOG_ENV = "ROBODYNA_EXPERIMENT_LOG"

EXPERIENCE_QUESTIONS = (
    ("video_games", "Do you have experience playing video games?"),
    ("keyboard_video_games", "Do you have experience playing video games using a keyboard?"),
    ("robotic_simulators", "Do you have previous experience with robotic simulators?"),
)

# Base suite: 23 tasks × 4 scenarios. Household: 12 tasks. Tutorial is practice.
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


def user_log_path(name_or_slug: str) -> Path:
    return user_dir(name_or_slug) / "user.json"


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
    path = user_log_path(slug)
    data = load_user_log(path)
    if data:
        return data
    # Also match display-name equality against other folders (legacy / renamed).
    if not EXP_LOGS_DIR.is_dir():
        return None
    needle = str(name or "").strip().casefold()
    for folder in EXP_LOGS_DIR.iterdir():
        candidate = folder / "user.json"
        if not candidate.is_file():
            continue
        other = load_user_log(candidate)
        if not other:
            continue
        if str(other.get("user_name", "")).strip().casefold() == needle:
            return other
        if str(other.get("user_id", "")).strip() == slug:
            return other
    return None


def create_user(name: str, experience: dict[str, str]) -> dict[str, Any]:
    display = str(name or "").strip()
    slug = slugify_user_name(display)
    path = user_log_path(slug)
    now = iso_now()
    data: dict[str, Any] = {
        "user_id": slug,
        "user_name": display,
        "created_at": now,
        "updated_at": now,
        "experience": {
            key: str(experience.get(key, "")).strip().lower()
            for key, _label in EXPERIENCE_QUESTIONS
        },
        "completed_keys": [],
        "plays": [],
        "log_path": str(path),
    }
    save_user_log(path, data)
    return data


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
    return play_key(suite, task, scenario) in completed_keys()


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


def progress_counts(log: dict[str, Any] | None = None) -> dict[str, int]:
    keys = completed_keys(log)
    base_done = sum(1 for k in keys if k.startswith("base:"))
    household_done = sum(1 for k in keys if k.startswith("household:"))
    return {
        "base_done": base_done,
        "base_total": BASE_SCENARIO_SLOTS,
        "household_done": household_done,
        "household_total": HOUSEHOLD_TASK_SLOTS,
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
) -> dict[str, Any] | None:
    """Append one play to the current experiment user log and save it."""
    path_raw = os.environ.get(EXPERIMENT_LOG_ENV, "").strip()
    path = Path(path_raw) if path_raw else None
    if path is None:
        user = os.environ.get(EXPERIMENT_USER_ENV, "").strip()
        if user:
            path = user_log_path(user)
    if path is None:
        return None
    data = load_user_log(path) or {
        "user_id": slugify_user_name(os.environ.get(EXPERIMENT_USER_ENV, "user")),
        "user_name": os.environ.get(EXPERIMENT_USER_ENV, "user"),
        "created_at": iso_now(),
        "experience": {},
        "completed_keys": [],
        "plays": [],
        "log_path": str(path),
    }
    payload = dict(payload or {})
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
    plays = list(data.get("plays") or [])
    plays.append(entry)
    data["plays"] = plays
    if counts_as_completed(result):
        key = play_key(suite, task, scenario)
        done = list(data.get("completed_keys") or [])
        if key not in done:
            done.append(key)
        data["completed_keys"] = done
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
