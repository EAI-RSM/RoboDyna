"""Load ``interactive/experiment.yml`` for the human-experiment GUI."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_PATH = Path(__file__).resolve().parent / "experiment.yml"
CONFIG_ENV = "ROBODYNA_EXPERIMENT_CONFIG"

BASE_SCENARIOS = ("default", "opt1", "opt2", "opt1+2")

# 1-based card numbers, matching base_task_gui.TASKS / household_task_gui.TASKS.
# Tutorial is 00 and is not part of these lists.
BASE_TASK_TABLE: tuple[tuple[int, str, str], ...] = (
    (1, "catch_marbles_trapdoors", "Catch Marbles Trapdoors"),
    (2, "catch_ramp_ball", "Catch Ramp Ball"),
    (3, "catch_cuboid", "Catch Cuboid"),
    (4, "catch_shelf_marble", "Catch Shelf Marble"),
    (5, "catch_valley_ball", "Catch Valley Ball"),
    (6, "stop_valley_ball", "Stop Valley Ball"),
    (7, "cook_meat", "Cook Meat"),
    (8, "cook_meat_timer", "Cook Meat Timer"),
    (9, "put_cup_belt", "Put Cup Belt"),
    (10, "dispense_gummy", "Dispense Gummy"),
    (11, "punch_dual_holes", "Punch Dual Holes"),
    (12, "save_goal", "Save Goal"),
    (13, "hit_target", "Hit Target"),
    (14, "load_train", "Load Train"),
    (15, "marble_shelf_maze", "Marble Shelf Maze"),
    (16, "pack_fruits", "Pack Fruits"),
    (17, "pick_ripe_apple", "Pick Ripe Apple"),
    (18, "place_block_belt", "Place Block Belt"),
    (19, "play_billiard", "Play Billiard"),
    (20, "control_quality", "Control Quality"),
    (21, "drop_ball_hole", "Drop Ball Hole"),
    (22, "sort_apples_belt", "Sort Apples Belt"),
    (23, "whack_moles", "Whack Moles"),
)
HOUSEHOLD_TASK_TABLE: tuple[tuple[int, str, str], ...] = (
    (1, "trap_bug", "Trap Bug"),
    (2, "boil_milk", "Boil Milk"),
    (3, "fill_coffee_jar", "Fill Coffee Jar"),
    (4, "pour_beer", "Pour Beer"),
    (5, "cook_food", "Cook Food"),
    (6, "cook_food_timer", "Cook Food Timer"),
    (7, "measure_ingredient", "Measure Ingredient"),
    (8, "make_soup", "Make Soup"),
    (9, "catch_cup", "Catch Cup"),
    (10, "catch_mouse_object_drop", "Catch Mouse Object Drop"),
    (11, "stop_ball", "Stop Ball"),
    (12, "clean_table", "Clean Table"),
)

# Default experiment set: all base except cuboid / cook_meat / punch_dual_holes,
# all household except fill_coffee_jar / cook_food.
DEFAULT_BASE_TASKS = [n for n, key, _ in BASE_TASK_TABLE if key not in (
    "catch_cuboid",
    "cook_meat",
    "punch_dual_holes",
)]
DEFAULT_HOUSEHOLD_TASKS = [n for n, key, _ in HOUSEHOLD_TASK_TABLE if key not in (
    "fill_coffee_jar",
    "cook_food",
)]

_SCENARIO_ALIASES = {
    "default": "default",
    "opt1": "opt1",
    "opt 1": "opt1",
    "option1": "opt1",
    "opt2": "opt2",
    "opt 2": "opt2",
    "option2": "opt2",
    "opt1+2": "opt1+2",
    "opt1+opt2": "opt1+2",
    "opt 1+2": "opt1+2",
    "opt12": "opt1+2",
}


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on", "all"):
        return True
    if text in ("0", "false", "no", "off", "none"):
        return False
    return default


def _as_int_list(value, default: list[int]) -> list[int]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        value = [part.strip() for part in value.split(",") if part.strip()]
    if not isinstance(value, (list, tuple)):
        return list(default)
    out: list[int] = []
    seen: set[int] = set()
    for item in value:
        try:
            number = int(item)
        except (TypeError, ValueError):
            continue
        if number in seen:
            continue
        seen.add(number)
        out.append(number)
    return out


def _as_seed(value) -> int | None:
    if value is None or value == "":
        return None
    try:
        seed = int(value)
    except (TypeError, ValueError):
        return None
    if seed < 0:
        return None
    return min(seed, 500)


def _as_controller(value) -> str:
    text = str(value or "robot").strip().lower()
    if text in ("keyboard", "robot"):
        return text
    return "robot"


def _is_unlimited(value) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return False
    if isinstance(value, str) and value.strip().lower() in (
        "null",
        "none",
        "unlimited",
        "inf",
        "infinite",
    ):
        return True
    return False


def _as_play_limit(value) -> int | None:
    """Positive int, or None for unlimited."""
    if _is_unlimited(value):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 1
    if number <= 0:
        return None
    return number


def _normalize_scenario(value) -> str | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    return _SCENARIO_ALIASES.get(text, text if text in BASE_SCENARIOS else None)


def _task_lookups(suite: str) -> tuple[dict[int, str], dict[str, str]]:
    table = BASE_TASK_TABLE if suite == "base" else HOUSEHOLD_TASK_TABLE
    by_number = {number: key for number, key, _label in table}
    by_name = {key: key for _number, key, _label in table}
    return by_number, by_name


def resolve_task_name(suite: str, raw) -> str | None:
    """Map a 1-based card number or task key to the canonical task name."""
    by_number, by_name = _task_lookups(suite)
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return by_number.get(int(raw))
    text = str(raw or "").strip()
    if not text:
        return None
    if text.isdigit():
        return by_number.get(int(text))
    key = text.replace("-", "_")
    return by_name.get(key) or by_name.get(text)


def _as_seeds(value) -> list[int]:
    """``null`` / empty → random each play. One int or a list of ints (0–500)."""
    if value is None or value == "" or value is False:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        value = [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        out: list[int] = []
        for item in value:
            seed = _as_seed(item)
            if seed is not None:
                out.append(seed)
        return out
    seed = _as_seed(value)
    return [seed] if seed is not None else []


def _is_policy_map(raw) -> bool:
    if not isinstance(raw, dict):
        return False
    for key in raw:
        text = str(key or "").strip().lower()
        if text in ("base", "household", "default", "all"):
            return True
        if _normalize_scenario(key) in BASE_SCENARIOS:
            return True
    return False


def _top_level_has_opt_keys(raw: dict) -> bool:
    for key in raw:
        scenario = _normalize_scenario(key)
        if scenario in ("opt1", "opt2", "opt1+2"):
            return True
    return False


@dataclass
class SlotPolicy:
    """Global default plus optional per-suite / per-task / per-scenario overrides.

    Resolution order:
      exact task+scenario → task → suite+scenario kind → global scenario kind
      → suite → default.

    Scenario kind (``opt1+2``, ``opt1``, …) applies to every task in that suite.
    For boolean whitelist maps, listing specific scenarios of a task leaves the
    other scenarios of that task at ``missing_slot`` (False) rather than default.
    """

    default: Any
    suite: dict[str, Any] = field(default_factory=dict)
    task: dict[tuple[str, str], Any] = field(default_factory=dict)
    slot: dict[tuple[str, str, str], Any] = field(default_factory=dict)
    scenario_kind: dict[str, Any] = field(default_factory=dict)
    suite_scenario: dict[tuple[str, str], Any] = field(default_factory=dict)
    slot_explicit_tasks: set[tuple[str, str]] = field(default_factory=set)
    missing_slot: Any = False
    bool_mode: bool = False

    def resolve(self, suite: str, task: str, scenario: str | None = None):
        suite = str(suite or "").strip()
        task = str(task or "").strip()
        if suite == "household":
            scenario = None
        if scenario:
            key = (suite, task, str(scenario))
            if key in self.slot:
                return self.slot[key]
            if self.bool_mode and (suite, task) in self.slot_explicit_tasks:
                return self.missing_slot
        task_key = (suite, task)
        if task_key in self.task:
            return self.task[task_key]
        if scenario:
            suite_sc = (suite, str(scenario))
            if suite_sc in self.suite_scenario:
                return self.suite_scenario[suite_sc]
            if str(scenario) in self.scenario_kind:
                return self.scenario_kind[str(scenario)]
        if suite in self.suite:
            return self.suite[suite]
        return self.default

    def any_true(self) -> bool:
        if self.default:
            return True
        if any(self.suite.values()):
            return True
        if any(self.task.values()):
            return True
        if any(self.slot.values()):
            return True
        if any(self.scenario_kind.values()):
            return True
        if any(self.suite_scenario.values()):
            return True
        return False


def _ingest_bool_task(policy: SlotPolicy, suite: str, task: str, spec) -> None:
    if isinstance(spec, (list, tuple)):
        policy.slot_explicit_tasks.add((suite, task))
        for item in spec:
            scenario = _normalize_scenario(item)
            if scenario:
                policy.slot[(suite, task, scenario)] = True
        return
    if isinstance(spec, dict):
        policy.slot_explicit_tasks.add((suite, task))
        for raw_scenario, raw_value in spec.items():
            scenario = _normalize_scenario(raw_scenario)
            if scenario:
                policy.slot[(suite, task, scenario)] = _as_bool(raw_value, False)
        return
    policy.task[(suite, task)] = _as_bool(spec, False)


def _ingest_bool_suite(policy: SlotPolicy, suite: str, spec) -> None:
    if isinstance(spec, (list, tuple)):
        for item in spec:
            task = resolve_task_name(suite, item)
            if task:
                policy.task[(suite, task)] = True
        return
    if isinstance(spec, dict):
        for raw_key, raw_value in spec.items():
            scenario = _normalize_scenario(raw_key)
            task = resolve_task_name(suite, raw_key)
            if scenario and not task:
                policy.suite_scenario[(suite, scenario)] = _as_bool(raw_value, False)
                continue
            if task:
                _ingest_bool_task(policy, suite, task, raw_value)
        return
    policy.suite[suite] = _as_bool(spec, False)


def parse_bool_policy(raw, default: bool = False) -> SlotPolicy:
    """Parse ``true`` / ``false`` or a suite/task/scenario map (whitelist unless ``default`` is set)."""
    if _is_policy_map(raw):
        has_opts = _top_level_has_opt_keys(raw)
        if "all" in raw:
            fallback = _as_bool(raw.get("all"), default)
        elif "default" in raw and not has_opts:
            fallback = _as_bool(raw.get("default"), default)
        elif "base" in raw or "household" in raw:
            fallback = False
        else:
            fallback = False
        policy = SlotPolicy(
            default=fallback,
            bool_mode=True,
            missing_slot=False,
        )
        for key, value in raw.items():
            text = str(key or "").strip().lower()
            if text in ("all",):
                continue
            if text in ("base", "household"):
                _ingest_bool_suite(policy, text, value)
                continue
            scenario = _normalize_scenario(key)
            if scenario is None:
                continue
            if text == "default" and not has_opts and "all" not in raw:
                continue
            policy.scenario_kind[scenario] = _as_bool(value, False)
        return policy
    return SlotPolicy(default=_as_bool(raw, default), bool_mode=True, missing_slot=False)


def _ingest_int_task(policy: SlotPolicy, suite: str, task: str, spec) -> None:
    if isinstance(spec, dict):
        for raw_scenario, raw_value in spec.items():
            scenario = _normalize_scenario(raw_scenario)
            if scenario:
                policy.slot[(suite, task, scenario)] = _as_play_limit(raw_value)
        return
    policy.task[(suite, task)] = _as_play_limit(spec)


def _ingest_int_suite(policy: SlotPolicy, suite: str, spec) -> None:
    if isinstance(spec, dict):
        for raw_key, raw_value in spec.items():
            scenario = _normalize_scenario(raw_key)
            task = resolve_task_name(suite, raw_key)
            if scenario and not task:
                if isinstance(raw_value, dict):
                    continue
                policy.suite_scenario[(suite, scenario)] = _as_play_limit(raw_value)
                continue
            if task:
                _ingest_int_task(policy, suite, task, raw_value)
        return
    policy.suite[suite] = _as_play_limit(spec)


def parse_int_policy(raw, default: int | None = 1) -> SlotPolicy:
    """Parse a play-limit integer, ``null`` (unlimited), or a per-slot map.

    An explicit ``None`` / ``null`` means unlimited. Omit the yaml key to keep
    the constructor default (1).

    Map form (scenario kind applies to every task)::

        default: 1
        opt1: 2
        opt2: 2
        opt1+2: 5
        household: 1
    """
    if _is_policy_map(raw):
        has_opts = _top_level_has_opt_keys(raw)
        if "all" in raw:
            fallback = _as_play_limit(raw.get("all"))
        elif "default" in raw and not has_opts:
            fallback = _as_play_limit(raw.get("default"))
        else:
            fallback = default
        policy = SlotPolicy(default=fallback, bool_mode=False)
        for key, value in raw.items():
            text = str(key or "").strip().lower()
            if text in ("all",):
                continue
            if text in ("base", "household"):
                _ingest_int_suite(policy, text, value)
                continue
            scenario = _normalize_scenario(key)
            if scenario is None:
                continue
            if text == "default" and not has_opts and "all" not in raw:
                continue
            policy.scenario_kind[scenario] = _as_play_limit(value)
        return policy
    if raw is None:
        return SlotPolicy(default=None, bool_mode=False)
    return SlotPolicy(default=_as_play_limit(raw), bool_mode=False)


@dataclass
class ExperimentConfig:
    record_data_policy: SlotPolicy = field(default_factory=lambda: parse_bool_policy(False))
    save_video_policy: SlotPolicy = field(default_factory=lambda: parse_bool_policy(False))
    log_plays_policy: SlotPolicy = field(default_factory=lambda: parse_bool_policy(True))
    plays_policy: SlotPolicy = field(default_factory=lambda: parse_int_policy(1))
    controller: str = "robot"
    seeds: list[int] = field(default_factory=list)
    base_tasks: list[int] = field(default_factory=lambda: list(DEFAULT_BASE_TASKS))
    household_tasks: list[int] = field(default_factory=lambda: list(DEFAULT_HOUSEHOLD_TASKS))
    path: Path = CONFIG_PATH

    @property
    def record_data(self) -> bool:
        """True if any slot records HDF5 (used by the locked GUI checkbox)."""
        return self.record_data_policy.any_true()

    @property
    def save_video(self) -> bool:
        return self.save_video_policy.any_true()

    @property
    def seed(self) -> int | None:
        """Single fixed seed, or None when random / a list is used."""
        if len(self.seeds) == 1:
            return self.seeds[0]
        return None

    def pick_seed(self, play_index: int = 0) -> int | None:
        """Seed for this play, or None to randomize.

        A list is consumed in order and wraps: play 0 → seeds[0], play 1 → seeds[1], …
        """
        if not self.seeds:
            return None
        index = max(0, int(play_index))
        return self.seeds[index % len(self.seeds)]

    def seed_display(self) -> str:
        if not self.seeds:
            return ""
        return ",".join(str(seed) for seed in self.seeds)

    def visible_indices(self, suite: str, n_tasks: int) -> list[int]:
        """0-based TASKS indices that should appear in the given suite GUI."""
        ones = self.base_tasks if suite == "base" else self.household_tasks
        out: list[int] = []
        for number in ones:
            index = int(number) - 1
            if 0 <= index < n_tasks:
                out.append(index)
        return out

    def task_names(self, suite: str) -> list[str]:
        table = BASE_TASK_TABLE if suite == "base" else HOUSEHOLD_TASK_TABLE
        lookup = {number: key for number, key, _label in table}
        ones = self.base_tasks if suite == "base" else self.household_tasks
        return [lookup[n] for n in ones if n in lookup]

    def max_plays(self, suite: str, task: str, scenario: str | None = None) -> int | None:
        """Allowed SUCCESS/FAILURE plays for this slot. ``None`` = unlimited."""
        return self.plays_policy.resolve(suite, task, scenario)

    def should_log(self, suite: str, task: str, scenario: str | None = None) -> bool:
        return bool(self.log_plays_policy.resolve(suite, task, scenario))

    def should_record_data(self, suite: str, task: str, scenario: str | None = None) -> bool:
        return bool(self.record_data_policy.resolve(suite, task, scenario))

    def should_save_video(self, suite: str, task: str, scenario: str | None = None) -> bool:
        return bool(self.save_video_policy.resolve(suite, task, scenario))


def config_path() -> Path:
    raw = os.environ.get(CONFIG_ENV, "").strip()
    if raw:
        return Path(raw)
    return CONFIG_PATH


def default_experiment_config() -> ExperimentConfig:
    return ExperimentConfig()


def load_experiment_config(path: Path | None = None) -> ExperimentConfig:
    path = path or config_path()
    cfg = default_experiment_config()
    cfg.path = path
    if not path.is_file():
        return cfg
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return cfg
    if not isinstance(raw, dict):
        return cfg
    cfg.record_data_policy = parse_bool_policy(raw.get("record_data"), False)
    cfg.save_video_policy = parse_bool_policy(raw.get("save_video"), False)
    log_raw = raw.get("log_plays", True)
    cfg.log_plays_policy = parse_bool_policy(log_raw, True)
    if "plays_per_scenario" in raw:
        cfg.plays_policy = parse_int_policy(raw.get("plays_per_scenario"), 1)
    cfg.controller = _as_controller(raw.get("controller", "robot"))
    cfg.seeds = _as_seeds(raw.get("seed"))
    if "base_tasks" in raw:
        cfg.base_tasks = _as_int_list(raw.get("base_tasks"), DEFAULT_BASE_TASKS)
    if "household_tasks" in raw:
        cfg.household_tasks = _as_int_list(raw.get("household_tasks"), DEFAULT_HOUSEHOLD_TASKS)
    return cfg
