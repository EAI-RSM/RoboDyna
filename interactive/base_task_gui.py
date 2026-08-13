#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Scrollable graphical launcher for the dynamic interactive tasks."""
from __future__ import annotations

import json
import os
import re
import secrets
import signal
import subprocess
import sys
import tempfile
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import yaml
from PIL import Image, ImageTk

_INTERACTIVE_DIR = Path(__file__).resolve().parent
if str(_INTERACTIVE_DIR) not in sys.path:
    sys.path.insert(0, str(_INTERACTIVE_DIR))
from _task_briefing import (  # noqa: E402
    GUI_INK,
    GUI_MUTED,
    GUI_PAGE_BG,
    GUI_WM_CLASS,
    apply_gui_logo,
    build_briefing_text,
    setup_gui_app_icon,
    show_task_briefing,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent / "base"
TUTORIAL_DIR = Path(__file__).resolve().parent / "Tutorial"
CONFIG_DIR = ROOT / "task_config"
DEMO_DIR = ROOT / "final_task_demos"
README_PATH = ROOT / "README.md"
# Must match interactive._interactive_common.TASK_RESULT_ENV
TASK_RESULT_ENV = "ROBODYNA_TASK_RESULT_FILE"


def record_status_note(payload) -> str:
    """Append-only note when the child wrote collect_data-format files."""
    if not isinstance(payload, dict):
        return ""
    hdf5 = str(payload.get("record_hdf5") or "").strip()
    if hdf5:
        return f" Recorded {hdf5}."
    path = str(payload.get("record_path") or "").strip()
    ep = payload.get("record_episode")
    if path and ep is not None:
        return f" Recorded {path}/data/episode{ep}.hdf5."
    return ""

TASKS = (
    ("Catch Marbles Trapdoors", "catch_marbles_trapdoors"),
    ("Catch Ramp Ball", "catch_ramp_ball"),
    ("Catch Cuboid", "catch_cuboid"),
    ("Catch Shelf Marble", "catch_shelf_marble"),
    ("Catch Valley Ball", "catch_valley_ball"),
    ("Stop Valley Ball", "stop_valley_ball"),
    ("Cook Meat", "cook_meat"),
    ("Cook Meat Timer", "cook_meat_timer"),
    ("Put Cup Belt", "put_cup_belt"),
    ("Dispense Gummy", "dispense_gummy"),
    ("Punch Dual Holes", "punch_dual_holes"),
    ("Save Goal", "save_goal"),
    ("Hit Target", "hit_target"),
    ("Load Train", "load_train"),
    ("Marble Shelf Maze", "marble_shelf_maze"),
    ("Pack Fruits", "pack_fruits"),
    ("Pick Ripe Apple", "pick_ripe_apple"),
    ("Place Block Belt", "place_block_belt"),
    ("Play Billiard", "play_billiard"),
    ("Control Quality", "control_quality"),
    ("Drop Ball Hole", "drop_ball_hole"),
    ("Sort Apples Belt", "sort_apples_belt"),
    ("Whack Moles", "whack_moles"),
)

TUTORIAL_PARTS = (
    ("Part 1", "tutorial_part1", "Select arms (1, 2, 3) then switch camera (V)."),
    ("Part 2", "tutorial_part2", "Move with arrows, E/Q, R/T, F/G, then Space."),
    ("Part 3", "tutorial_part3", "Grasp, hold-button, switch, then push a box."),
    ("Part 4", "tutorial_part4", "Ball, stove knob, mallet, then force key."),
)

SCENARIOS = ("default", "opt1", "opt2", "opt1+2")
SCENARIO_LABELS = {
    "default": "Default",
    "opt1": "Opt 1",
    "opt2": "Opt 2",
    "opt1+2": "Opt 1+2",
}

# Explicit values make Default independent of any feature flags currently set
# in demo_dynamic.yml. Some options need more than one task argument.
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

PLAY_BLUE = "#3182bd"
PLAY_BLUE_ACTIVE = "#4295d0"
PAGE_BG = GUI_PAGE_BG
HEADER_BG = GUI_PAGE_BG
HEADER_FG = GUI_INK
HEADER_MUTED = GUI_MUTED
CARD_BG = "#202c38"
CARD_BORDER = "#405367"
TEXT_PRIMARY = "#f4f8fb"
TEXT_SECONDARY = "#aebdca"
RANDOM_SEED_MAX = 500

# Tasks whose README row lives under a different name (or is shared).
README_TASK_ALIASES: dict[str, str] = {}


def load_condition_descriptions(readme_path: Path = README_PATH) -> dict[str, dict[str, str]]:
    """Parse README task-table ``<sub>…</sub>`` blurbs into per-condition text.

    Each task row is:
      task summary, Default, Opt 1, Opt 2, Opt 1+2
    """
    if not readme_path.exists():
        return {}
    text = readme_path.read_text(encoding="utf-8")
    row_re = re.compile(r"^\|\s*\*\*`([a-z0-9_]+)`\*\*.*$", re.MULTILINE)
    sub_re = re.compile(r"<sub>(.*?)</sub>", re.DOTALL)
    out: dict[str, dict[str, str]] = {}
    for match in row_re.finditer(text):
        task = match.group(1)
        line_end = text.find("\n", match.start())
        line = text[match.start() : line_end if line_end >= 0 else None]
        subs = [re.sub(r"\s+", " ", s).strip() for s in sub_re.findall(line)]
        if len(subs) < 5:
            continue
        out[task] = {
            "summary": subs[0],
            "default": subs[1],
            "opt1": subs[2],
            "opt2": subs[3],
            "opt1+2": subs[4],
        }
    return out


CONDITION_DESCRIPTIONS = load_condition_descriptions()


def condition_description(task: str, scenario: str) -> str:
    """README blurb for ``task`` / ``scenario``, or empty if unavailable."""
    key = README_TASK_ALIASES.get(task, task)
    return str(CONDITION_DESCRIPTIONS.get(key, {}).get(scenario, "") or "")


def task_summary(task: str) -> str:
    """README task-summary blurb, or empty if unavailable."""
    key = README_TASK_ALIASES.get(task, task)
    return str(CONDITION_DESCRIPTIONS.get(key, {}).get("summary", "") or "")


def resolve_seed(value: str) -> int:
    """Return a fixed entered seed, or a fresh random seed for a blank value."""
    value = value.strip()
    if not value:
        return secrets.randbelow(RANDOM_SEED_MAX + 1)
    try:
        seed = int(value)
    except ValueError as exc:
        raise ValueError("Seed must be a whole number or left blank for a random seed.") from exc
    if not 0 <= seed <= RANDOM_SEED_MAX:
        raise ValueError(f"Seed must be between 0 and {RANDOM_SEED_MAX}.")
    return seed


def build_scenario_config(task: str, scenario: str) -> dict:
    """Load the standard interactive config and apply one scenario's flags."""
    config_path = CONFIG_DIR / "demo_dynamic.yml"
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    task_args = config.setdefault("task_args", {}).setdefault(task, {})
    overrides = dict(SCENARIO_OVERRIDES.get(task, {}).get(scenario, {}) or {})
    task_args.update(overrides)
    # Top-level marker so the task env can recover the scenario even if a
    # nested key is missing / stale.
    config["interactive_scenario"] = scenario
    config["interactive_task"] = task
    return config


class RoundedButton(tk.Canvas):
    """Canvas-backed button with rounded corners."""

    def __init__(
        self,
        parent,
        *,
        text,
        command,
        bg,
        activebackground,
        font,
        width,
        height,
        radius=24,
        on_enter=None,
        on_leave=None,
    ):
        super().__init__(
            parent,
            width=width,
            height=height,
            bg=parent.cget("bg"),
            highlightthickness=0,
            bd=0,
            cursor="hand2",
        )
        self.command = command
        self.normal_color = bg
        self.active_color = activebackground
        self.button_state = "normal"
        self.radius = radius
        self._text = text
        self._font = font
        self._shape = None
        self._on_enter_cb = on_enter
        self._on_leave_cb = on_leave
        self.bind("<Configure>", self._redraw)
        self.bind("<Enter>", self._enter)
        self.bind("<Leave>", self._leave)
        self.bind("<ButtonRelease-1>", self._click)

    def _rounded_polygon(self, width, height):
        radius = min(self.radius, width // 2, height // 2)
        return (
            radius, 1, width - radius, 1, width - 1, 1, width - 1, radius,
            width - 1, height - radius, width - 1, height - 1,
            width - radius, height - 1, radius, height - 1, 1, height - 1,
            1, height - radius, 1, radius, 1, 1,
        )

    def _redraw(self, _event=None):
        self.delete("all")
        width = max(2, self.winfo_width())
        height = max(2, self.winfo_height())
        color = self.normal_color if self.button_state == "normal" else "#59616b"
        self._shape = self.create_polygon(
            self._rounded_polygon(width, height),
            smooth=True,
            splinesteps=24,
            fill=color,
            outline="",
        )
        self.create_text(
            width / 2,
            height / 2,
            text=self._text,
            fill="white" if self.button_state == "normal" else "#b4bac2",
            font=self._font,
        )

    def _enter(self, _event):
        if self.button_state == "normal" and self._shape is not None:
            self.itemconfigure(self._shape, fill=self.active_color)
        if self._on_enter_cb is not None:
            self._on_enter_cb()

    def _leave(self, _event):
        if self.button_state == "normal" and self._shape is not None:
            self.itemconfigure(self._shape, fill=self.normal_color)
        if self._on_leave_cb is not None:
            self._on_leave_cb()

    def _click(self, _event):
        if self.button_state == "normal":
            self.command()

    def configure(self, cnf=None, **kwargs):
        options = dict(cnf or {})
        options.update(kwargs)
        if "text" in options:
            self._text = options.pop("text")
        if "bg" in options:
            self.normal_color = options.pop("bg")
        if "activebackground" in options:
            self.active_color = options.pop("activebackground")
        if "state" in options:
            self.button_state = options.pop("state")
        if "font" in options:
            self._font = options.pop("font")
        if "radius" in options:
            self.radius = int(options.pop("radius"))
        if options:
            super().configure(**options)
        self._redraw()

    config = configure


class InteractiveTaskLauncher(tk.Tk):
    # Head-camera stills are ~4:3. Width seeds the first layout; height follows
    # each image's native aspect so previews are not letterboxed.
    IMAGE_SIZE = (1400, 1050)
    CARD_PAD = 8
    PREVIEW_SIDE_PAD = 28
    DESIGN_SIZE = (1600, 1000)
    UI_SCALE_MIN = 0.55
    UI_SCALE_MAX = 1.15

    def __init__(self):
        # className becomes StartupWMClass for the Ubuntu dock .desktop entry.
        super().__init__(className=GUI_WM_CLASS["interactive"])
        self.title("Base Tasks")
        self.geometry("1600x1000")
        self.minsize(900, 640)
        self.configure(bg=PAGE_BG)
        setup_gui_app_icon(
            self,
            suite="interactive",
            script_path=Path(__file__),
        )
        self.protocol("WM_DELETE_WINDOW", self.exit_app)

        self.child: subprocess.Popen | None = None
        self.active_selection: tuple | None = None
        self.temporary_config: Path | None = None
        self.result_file: Path | None = None
        # Per task: one source/photo/label per scenario (Default / Opt1 / Opt2 / Opt1+2).
        self.preview_sources: list[list[Image.Image | None]] = []
        self.preview_photos: list[list[ImageTk.PhotoImage | None]] = []
        self.preview_labels: list[list[tk.Label]] = []
        self.task_buttons: list[list[RoundedButton]] = []
        self.card_index_labels: list[tk.Label] = []
        self.card_title_labels: list[tk.Label] = []
        self.card_badge_labels: list[tk.Label] = []
        self._preview_resize_job: str | None = None
        self._ui_scale_job: str | None = None
        self._preview_width = self.IMAGE_SIZE[0]
        self._ui_scale = 1.0
        self._header_layout_key: tuple | None = None
        self.tutorial_sources: list[Image.Image | None] = []
        self.tutorial_photos: list[ImageTk.PhotoImage | None] = []
        self.tutorial_preview_labels: list[tk.Label] = []
        self.tutorial_buttons: list[RoundedButton] = []
        self._idle_status = (
            f"Tutorial (4 parts)  ·  {len(TASKS)} tasks  |  "
            "Hover a scenario key for its README description."
        )
        self._idle_status_fg = HEADER_MUTED

        self._build_ui()
        self.bind("<Configure>", self._on_root_configure)
        self.after(0, self._apply_ui_scale)
        self.after(250, self._poll_child)

    def _build_ui(self):
        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass
        self.style.configure(
            "Task.TCombobox",
            padding=(8, 6),
            arrowsize=16,
            fieldbackground="#f7fafc",
            foreground="#182633",
            selectbackground="#f7fafc",
            selectforeground="#182633",
        )

        self.logo_bar = tk.Frame(self, bg=PAGE_BG)
        self.logo_bar.pack(fill="x", padx=24, pady=(16, 0))
        self.logo_label = tk.Label(self.logo_bar, bg=PAGE_BG, bd=0, highlightthickness=0)
        self.logo_label.pack(anchor="center")
        apply_gui_logo(self.logo_label, height=160)

        self.header = tk.Frame(
            self,
            bg=HEADER_BG,
            highlightbackground="#d4d5db",
            highlightthickness=1,
        )
        self.header.pack(fill="x", padx=24, pady=(12, 12))

        # Title + Exit live in the top row so Exit is packed first and never clipped.
        self.header_top = tk.Frame(self.header, bg=HEADER_BG)
        self.header_top.pack(fill="x")
        self.exit_button = RoundedButton(
            self.header_top,
            text="Exit",
            command=self.exit_app,
            bg="#e34a33",
            activebackground="#eb6854",
            font=("Sans", 17, "bold"),
            width=150,
            height=76,
            radius=28,
        )
        self.exit_button.pack(side="right", padx=(8, 22), pady=16)

        self.heading = tk.Frame(self.header_top, bg=HEADER_BG)
        self.heading.pack(side="left", padx=24, pady=18)
        self.title_label = tk.Label(
            self.heading,
            text="Base Tasks",
            bg=HEADER_BG,
            fg=HEADER_FG,
            anchor="w",
            justify="left",
            font=("Sans", 34, "bold"),
        )
        self.title_label.pack(anchor="w")
        self.subtitle_label = tk.Label(
            self.heading,
            text="Choose a tutorial part or a task and one of its four scenario variants.",
            bg=HEADER_BG,
            fg=HEADER_MUTED,
            anchor="w",
            justify="left",
            wraplength=720,
            font=("Sans", 14),
        )
        self.subtitle_label.pack(anchor="w", pady=(3, 0))

        # Master is header so this can sit on the title row or wrap onto a full-width row.
        self.controls = tk.Frame(self.header, bg=HEADER_BG)
        self.controls.pack(in_=self.header_top, side="right", padx=22, pady=16)

        self.brief_group = tk.Frame(self.controls, bg=HEADER_BG)
        self.brief_group.pack(side="left", padx=(0, 16))
        self.brief_caption = tk.Label(
            self.brief_group,
            text="Briefing",
            bg=HEADER_BG,
            fg=HEADER_MUTED,
            font=("Sans", 13, "bold"),
        )
        self.brief_caption.pack(anchor="w")
        self.show_briefing = tk.BooleanVar(value=True)
        self.briefing_check = tk.Checkbutton(
            self.brief_group,
            text="Show before start",
            variable=self.show_briefing,
            onvalue=True,
            offvalue=False,
            bg=HEADER_BG,
            fg=HEADER_FG,
            activebackground=HEADER_BG,
            activeforeground=HEADER_FG,
            selectcolor="#ffffff",
            highlightthickness=0,
            font=("Sans", 14, "bold"),
            cursor="hand2",
        )
        self.briefing_check.pack(anchor="w", pady=(6, 0))

        self.record_group = tk.Frame(self.controls, bg=HEADER_BG)
        self.record_group.pack(side="left", padx=(0, 16))
        self.record_caption = tk.Label(
            self.record_group,
            text="Record data",
            bg=HEADER_BG,
            fg=HEADER_MUTED,
            font=("Sans", 13, "bold"),
        )
        self.record_caption.pack(anchor="w")
        self.record_data = tk.BooleanVar(value=False)
        self.record_check = tk.Checkbutton(
            self.record_group,
            text="Save this episode",
            variable=self.record_data,
            onvalue=True,
            offvalue=False,
            bg=HEADER_BG,
            fg=HEADER_FG,
            activebackground=HEADER_BG,
            activeforeground=HEADER_FG,
            selectcolor="#ffffff",
            highlightthickness=0,
            font=("Sans", 14, "bold"),
            cursor="hand2",
        )
        self.record_check.pack(anchor="w", pady=(6, 0))

        self.seed_group = tk.Frame(self.controls, bg=HEADER_BG)
        self.seed_group.pack(side="left", padx=(0, 18))
        self.seed_caption = tk.Label(
            self.seed_group,
            text="Seed (blank = random)",
            bg=HEADER_BG,
            fg=HEADER_MUTED,
            font=("Sans", 13, "bold"),
        )
        self.seed_caption.pack(anchor="w")
        self.seed_entry = tk.Entry(
            self.seed_group,
            width=16,
            font=("Sans", 13, "bold"),
            bg="#f7fafc",
            fg="#182633",
            insertbackground="#182633",
            relief="flat",
        )
        self.seed_entry.pack(ipady=5, pady=(4, 0))

        self.control_group = tk.Frame(self.controls, bg=HEADER_BG)
        self.control_group.pack(side="left", padx=(0, 14))
        self.control_caption = tk.Label(
            self.control_group,
            text="Control",
            bg=HEADER_BG,
            fg=HEADER_MUTED,
            font=("Sans", 13, "bold"),
        )
        self.control_caption.pack(anchor="w")
        self.control = ttk.Combobox(
            self.control_group,
            values=("keyboard", "robot"),
            state="readonly",
            width=8,
            font=("Sans", 13, "bold"),
            style="Task.TCombobox",
        )
        self.control.set("robot")
        self.control.pack(pady=(4, 0))
        self._style_control_menu(("Sans", 13, "bold"))
        self._control_groups = (self.brief_group, self.seed_group, self.control_group)

        self.status = tk.Label(
            self,
            text=self._idle_status,
            bg=PAGE_BG,
            fg=self._idle_status_fg,
            anchor="w",
            justify="left",
            wraplength=1500,
            font=("Sans", 19),
        )
        self.status.pack(fill="x", padx=34, pady=(2, 12))

        outer = tk.Frame(self, bg=PAGE_BG)
        outer.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.canvas = tk.Canvas(outer, bg=PAGE_BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.page = tk.Frame(self.canvas, bg=PAGE_BG)
        self.page_window = self.canvas.create_window((0, 0), window=self.page, anchor="nw")
        self.page.bind("<Configure>", self._update_scroll_region)
        self.canvas.bind("<Configure>", self._resize_page)
        self.canvas.bind_all("<MouseWheel>", self._cuboidwheel)
        self.canvas.bind_all("<Button-4>", lambda _event: self.canvas.yview_scroll(-3, "units"))
        self.canvas.bind_all("<Button-5>", lambda _event: self.canvas.yview_scroll(3, "units"))

        self._add_tutorial_section()
        for index, (label, task) in enumerate(TASKS):
            self._add_task_card(index, label, task)

    @staticmethod
    def _scaled_font(size: float, weight: str = "", scale: float = 1.0) -> tuple:
        px = max(8, int(round(size * scale)))
        return ("Sans", px, weight) if weight else ("Sans", px)

    def _style_control_menu(self, font) -> None:
        """Enlarge the Control combobox dropdown list independently of the field."""
        try:
            popdown = self.tk.call("ttk::combobox::PopdownWindow", self.control)
            self.tk.call(f"{popdown}.f.l", "configure", "-font", font)
        except tk.TclError:
            self.option_add("*TCombobox*Listbox.font", font)

    def _compute_ui_scale(self, width: int | None = None, height: int | None = None) -> float:
        design_w, design_h = self.DESIGN_SIZE
        width = int(width if width is not None else max(self.winfo_width(), 1))
        height = int(height if height is not None else max(self.winfo_height(), 1))
        scale = min(width / design_w, height / design_h)
        return max(self.UI_SCALE_MIN, min(self.UI_SCALE_MAX, scale))

    def _on_root_configure(self, event):
        if event.widget is not self:
            return
        if self._ui_scale_job is not None:
            self.after_cancel(self._ui_scale_job)
        self._ui_scale_job = self.after(80, self._apply_ui_scale)

    def _apply_ui_scale(self):
        self._ui_scale_job = None
        if getattr(self, "title_label", None) is None:
            return
        scale = self._compute_ui_scale()
        if abs(scale - self._ui_scale) < 0.02:
            # Width-only change near the same scale: refresh wrap + header stack.
            self.status.configure(
                wraplength=max(360, self.winfo_width() - self._px(68, scale))
            )
            self._relayout_header(scale)
            return
        self._ui_scale = scale
        s = scale

        self.logo_bar.pack_configure(padx=self._px(24, s), pady=(self._px(16, s), 0))
        apply_gui_logo(self.logo_label, height=self._px(160, s))
        self.header.pack_configure(padx=self._px(24, s), pady=(self._px(12, s), self._px(12, s)))
        self.title_label.configure(font=self._scaled_font(34, "bold", s))
        self.subtitle_label.configure(font=self._scaled_font(14, scale=s))
        self.brief_caption.configure(font=self._scaled_font(13, "bold", s))
        self.briefing_check.configure(font=self._scaled_font(14, "bold", s))
        self.seed_caption.configure(font=self._scaled_font(13, "bold", s))
        self.control_caption.configure(font=self._scaled_font(13, "bold", s))
        self.seed_entry.configure(font=self._scaled_font(13, "bold", s))
        self.seed_entry.pack_configure(ipady=self._px(5, s), pady=(self._px(4, s), 0))
        self.control.configure(font=self._scaled_font(13, "bold", s))
        self._style_control_menu(self._scaled_font(13, "bold", s))
        self.style.configure(
            "Task.TCombobox",
            padding=(self._px(8, s), self._px(6, s)),
            arrowsize=max(10, self._px(16, s)),
            fieldbackground="#f7fafc",
            foreground="#182633",
            selectbackground="#f7fafc",
            selectforeground="#182633",
        )
        self.exit_button.configure(
            font=self._scaled_font(17, "bold", s),
            width=self._px(150, s),
            height=self._px(76, s),
            radius=self._px(28, s),
        )
        self.status.configure(
            font=self._scaled_font(19, scale=s),
            wraplength=max(360, self.winfo_width() - self._px(68, s)),
        )
        self.status.pack_configure(padx=self._px(34, s), pady=(self._px(2, s), self._px(12, s)))

        idx_padx = self._px(13, s)
        idx_pady = self._px(6, s)
        for label in self.card_index_labels:
            label.configure(
                font=self._scaled_font(17, "bold", s),
                padx=idx_padx,
                pady=idx_pady,
            )
        for label in self.card_title_labels:
            label.configure(
                font=self._scaled_font(27, "bold", s),
                wraplength=max(160, self.winfo_width() - self._px(280, s)),
            )
        for label in self.card_badge_labels:
            label.configure(font=self._scaled_font(12, "bold", s))

        btn_h = self._px(70, s)
        btn_radius = self._px(26, s)
        btn_font = self._scaled_font(16, "bold", s)
        for row in self.task_buttons:
            for button in row:
                button.configure(font=btn_font, height=btn_h, radius=btn_radius)
        for button in self.tutorial_buttons:
            button.configure(font=btn_font, height=btn_h, radius=btn_radius)

        self._relayout_header(s)

    @staticmethod
    def _px(value: float, scale: float) -> int:
        return max(1, int(round(value * scale)))

    def _relayout_header(self, scale: float):
        """Keep Exit visible and reflow title/controls as the window width changes."""
        if getattr(self, "_relayouting", False):
            return
        self._relayouting = True
        try:
            self._relayout_header_body(scale)
        finally:
            self._relayouting = False

    def _relayout_header_body(self, scale: float):
        header_w = int(self.header.winfo_width())
        if header_w <= 1:
            header_w = max(1, int(self.winfo_width()) - 2 * self._px(24, scale))
        pad_exit = self._px(16, scale)
        pad_head = self._px(24, scale)
        pad_ctrl = self._px(22, scale)
        gap = self._px(16, scale)
        self.update_idletasks()
        exit_w = self.exit_button.winfo_reqwidth() + pad_exit + self._px(8, scale)
        avail = max(self._px(160, scale), header_w - exit_w)
        controls_w = self._controls_natural_width(scale)
        min_heading = self._px(220, scale)
        stacked = controls_w + gap + min_heading > avail
        wrap_groups = stacked and controls_w > avail - self._px(12, scale)
        if stacked:
            wrap = max(self._px(160, scale), avail - 2 * pad_head)
        else:
            wrap = max(self._px(160, scale), avail - controls_w - gap - 2 * pad_head)
        wrap -= wrap % 8
        key = (stacked, wrap_groups, wrap, round(scale, 2))
        if key == self._header_layout_key:
            return
        self._header_layout_key = key

        self.title_label.configure(wraplength=wrap)
        self.subtitle_label.configure(wraplength=wrap)
        self.heading.pack_forget()
        self.controls.pack_forget()
        self.exit_button.pack_forget()

        self.exit_button.pack(
            side="right",
            padx=(self._px(8, scale), pad_exit),
            pady=self._px(16, scale),
        )
        self._pack_control_groups(wrap=wrap_groups, scale=scale)
        if stacked:
            self.heading.pack(
                side="left",
                fill="x",
                expand=True,
                padx=pad_head,
                pady=(self._px(14, scale), self._px(10, scale)),
            )
            self.controls.pack(
                in_=self.header,
                side="top",
                anchor="w",
                fill="x",
                padx=pad_ctrl,
                pady=(0, self._px(14, scale)),
            )
        else:
            self.heading.pack(side="left", padx=pad_head, pady=self._px(18, scale))
            self.controls.pack(
                in_=self.header_top,
                side="right",
                padx=pad_ctrl,
                pady=self._px(16, scale),
            )

    def _controls_natural_width(self, scale: float) -> int:
        pads = (self._px(16, scale), self._px(18, scale), self._px(14, scale))
        return sum(
            group.winfo_reqwidth() + pad
            for group, pad in zip(self._control_groups, pads)
        )

    def _pack_control_groups(self, *, wrap: bool, scale: float) -> None:
        pads = (self._px(16, scale), self._px(18, scale), self._px(14, scale))
        for group in self._control_groups:
            group.pack_forget()
        if wrap:
            for group in self._control_groups:
                group.pack(side="top", anchor="w", pady=(0, self._px(6, scale)))
        else:
            for group, pad in zip(self._control_groups, pads):
                group.pack(side="left", padx=(0, pad))

    def _update_scroll_region(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _resize_page(self, event):
        page_width = max(event.width, 720)
        self.canvas.itemconfigure(self.page_window, width=page_width)
        preview_width = max(
            480, page_width - 2 * (self.CARD_PAD + self.PREVIEW_SIDE_PAD)
        )
        if abs(preview_width - self._preview_width) < 8:
            return
        if self._preview_resize_job is not None:
            self.after_cancel(self._preview_resize_job)
        self._preview_resize_job = self.after(
            80, lambda width=preview_width: self._apply_preview_width(width)
        )

    def _cell_preview_width(self, total_width: int) -> int:
        """Width of one of the four scenario preview columns."""
        gap = 6 * 2 * len(SCENARIOS)  # padx=6 on each side of each column
        return max(160, (total_width - gap) // len(SCENARIOS))

    def _apply_preview_width(self, width: int):
        self._preview_resize_job = None
        self._preview_width = width
        cell_width = self._cell_preview_width(width)
        for task_index, sources in enumerate(self.preview_sources):
            photos: list[ImageTk.PhotoImage | None] = []
            for scen_index, source in enumerate(sources):
                photo = self._render_preview(source, cell_width)
                photos.append(photo)
                label = self.preview_labels[task_index][scen_index]
                if photo is None:
                    label.configure(
                        image="",
                        text="No preview",
                        width=max(8, cell_width // 10),
                        height=max(4, (cell_width * 3) // (8 * 16)),
                    )
                else:
                    label.configure(image=photo, text="", width=0, height=0)
            self.preview_photos[task_index] = photos
        self._refresh_tutorial_previews(cell_width)

    def _cuboidwheel(self, event):
        if event.delta:
            self.canvas.yview_scroll(-int(event.delta / 120), "units")

    @staticmethod
    def _preview_path(task: str, scenario: str = "default") -> Path | None:
        directory = DEMO_DIR / task
        names = (
            f"scene_snapshot_{scenario}.png",
            "scene_snapshot.png" if scenario == "default" else None,
            "default_sidebyside.gif" if scenario == "default" else None,
            f"{scenario}_sidebyside.gif",
            f"*{scenario}*sidebyside.gif",
        )
        for name in names:
            if not name:
                continue
            if "*" in name:
                match = next(directory.glob(name), None) if directory.exists() else None
                if match is not None:
                    return match
                continue
            preferred = directory / name
            if preferred.exists():
                return preferred
        if scenario == "default" and directory.exists():
            patterns = ("default*sidebyside.gif", "*.gif", "*.png", "*.jpg", "*.jpeg")
            for pattern in patterns:
                match = next(directory.glob(pattern), None)
                if match is not None:
                    return match
        return None

    def _load_preview_source(self, task: str, scenario: str = "default") -> Image.Image | None:
        path = self._preview_path(task, scenario)
        if path is None and scenario != "default":
            # Fall back to the default still so the grid stays populated.
            path = self._preview_path(task, "default")
        if path is None:
            return None
        try:
            with Image.open(path) as source:
                source.seek(0)
                return source.convert("RGB")
        except Exception:
            return None

    def _render_preview(self, source: Image.Image | None, width: int) -> ImageTk.PhotoImage | None:
        if source is None:
            return None
        # Fill the column width at the image's native aspect ratio (no letterbox).
        aspect = source.width / max(source.height, 1)
        height = max(1, int(round(width / aspect)))
        image = source.resize((width, height), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(image)

    def _tutorial_snapshot_path(self, part_index: int | None = None) -> Path | None:
        names: list[str] = []
        if part_index is not None:
            names.append(f"scene_snapshot_part{part_index + 1}.png")
        names.append("scene_snapshot.png")
        directories = (TUTORIAL_DIR, DEMO_DIR / "tutorial_empty")
        for directory in directories:
            for name in names:
                path = directory / name
                if path.exists():
                    return path
        return None

    def _load_tutorial_source(self, part_index: int | None = None) -> Image.Image | None:
        path = self._tutorial_snapshot_path(part_index)
        if path is None:
            return None
        try:
            with Image.open(path) as source:
                source.seek(0)
                return source.convert("RGB")
        except Exception:
            return None

    def _refresh_tutorial_previews(self, width: int | None = None) -> None:
        cell_width = int(width if width is not None else self._cell_preview_width(self._preview_width))
        photos: list[ImageTk.PhotoImage | None] = []
        for index, source in enumerate(self.tutorial_sources):
            photo = self._render_preview(source, cell_width)
            photos.append(photo)
            if index >= len(self.tutorial_preview_labels):
                continue
            label = self.tutorial_preview_labels[index]
            if photo is None:
                label.configure(
                    image="",
                    text="No preview",
                    width=max(8, cell_width // 10),
                    height=max(4, (cell_width * 3) // (8 * 16)),
                )
            else:
                label.configure(image=photo, text="", width=0, height=0)
        self.tutorial_photos = photos

    def _add_tutorial_section(self):
        """Same four-column card as other base tasks; index 00."""
        card = tk.Frame(
            self.page,
            bg=CARD_BG,
            highlightbackground=CARD_BORDER,
            highlightthickness=2,
        )
        card.pack(fill="x", padx=self.CARD_PAD, pady=12)

        card_header = tk.Frame(card, bg=CARD_BG)
        card_header.pack(fill="x", padx=self.PREVIEW_SIDE_PAD, pady=(14, 8))
        index_label = tk.Label(
            card_header,
            text="00",
            bg=PLAY_BLUE,
            fg="white",
            font=("Sans", 17, "bold"),
            padx=13,
            pady=6,
        )
        index_label.pack(side="left", padx=(0, 14))
        title_label = tk.Label(
            card_header,
            text="Tutorial",
            bg=CARD_BG,
            fg=TEXT_PRIMARY,
            anchor="w",
            font=("Sans", 27, "bold"),
        )
        title_label.pack(side="left", fill="x", expand=True)
        badge_label = tk.Label(
            card_header,
            text="4 PARTS",
            bg=CARD_BG,
            fg="#7fb6dc",
            font=("Sans", 12, "bold"),
        )
        badge_label.pack(side="right")
        self.card_index_labels.append(index_label)
        self.card_title_labels.append(title_label)
        self.card_badge_labels.append(badge_label)

        grid = tk.Frame(card, bg=CARD_BG)
        grid.pack(fill="x", padx=self.PREVIEW_SIDE_PAD, pady=(0, 16))
        self.tutorial_sources = [
            self._load_tutorial_source(index) for index in range(len(TUTORIAL_PARTS))
        ]
        cell_width = self._cell_preview_width(self._preview_width)
        self.tutorial_photos = []

        for index, (label, _script, hint) in enumerate(TUTORIAL_PARTS):
            col = tk.Frame(grid, bg=CARD_BG)
            col.pack(side="left", expand=True, fill="both", padx=6)

            photo = self._render_preview(self.tutorial_sources[index], cell_width)
            self.tutorial_photos.append(photo)
            preview_label = tk.Label(
                col,
                image=photo,
                text="No preview" if photo is None else "",
                bg=CARD_BG,
                fg=TEXT_SECONDARY,
                font=("Sans", 12),
                bd=0,
                highlightthickness=0,
            )
            if photo is None:
                preview_label.configure(
                    width=max(8, cell_width // 10),
                    height=max(4, (cell_width * 3) // (8 * 16)),
                )
            preview_label.pack(fill="x", pady=(0, 8))
            self.tutorial_preview_labels.append(preview_label)

            button = RoundedButton(
                col,
                text=label,
                command=lambda i=index: self.play_or_stop_tutorial(i),
                bg=PLAY_BLUE,
                activebackground=PLAY_BLUE_ACTIVE,
                font=("Sans", 16, "bold"),
                width=80,
                height=70,
                radius=26,
                on_enter=lambda h=hint, l=label: self._show_tutorial_hint(l, h),
                on_leave=self._clear_condition_hint,
            )
            button.pack(fill="x")
            self.tutorial_buttons.append(button)

    def _show_tutorial_hint(self, label: str, hint: str):
        if self.child is not None:
            return
        self.status.configure(text=f"Tutorial · {label}: {hint}", fg="#7fb6dc")

    def _add_task_card(self, index: int, label: str, task: str):
        card = tk.Frame(
            self.page,
            bg=CARD_BG,
            highlightbackground=CARD_BORDER,
            highlightthickness=2,
        )
        card.pack(fill="x", padx=self.CARD_PAD, pady=12)

        card_header = tk.Frame(card, bg=CARD_BG)
        card_header.pack(fill="x", padx=self.PREVIEW_SIDE_PAD, pady=(14, 8))
        index_label = tk.Label(
            card_header,
            text=f"{index + 1:02d}",
            bg=PLAY_BLUE,
            fg="white",
            font=("Sans", 17, "bold"),
            padx=13,
            pady=6,
        )
        index_label.pack(side="left", padx=(0, 14))
        title_label = tk.Label(
            card_header,
            text=label,
            bg=CARD_BG,
            fg=TEXT_PRIMARY,
            anchor="w",
            font=("Sans", 27, "bold"),
        )
        title_label.pack(side="left", fill="x", expand=True)
        badge_label = tk.Label(
            card_header,
            text="4 SCENARIOS",
            bg=CARD_BG,
            fg="#7fb6dc",
            font=("Sans", 12, "bold"),
        )
        badge_label.pack(side="right")
        self.card_index_labels.append(index_label)
        self.card_title_labels.append(title_label)
        self.card_badge_labels.append(badge_label)

        # Four columns: screenshot on top, matching scenario key underneath.
        grid = tk.Frame(card, bg=CARD_BG)
        grid.pack(fill="x", padx=self.PREVIEW_SIDE_PAD, pady=(0, 16))
        cell_width = self._cell_preview_width(self._preview_width)
        sources: list[Image.Image | None] = []
        photos: list[ImageTk.PhotoImage | None] = []
        labels: list[tk.Label] = []
        buttons: list[RoundedButton] = []
        for scenario in SCENARIOS:
            col = tk.Frame(grid, bg=CARD_BG)
            col.pack(side="left", expand=True, fill="both", padx=6)

            source = self._load_preview_source(task, scenario)
            photo = self._render_preview(source, cell_width)
            sources.append(source)
            photos.append(photo)
            preview_label = tk.Label(
                col,
                image=photo,
                text="No preview" if photo is None else "",
                bg=CARD_BG,
                fg=TEXT_SECONDARY,
                font=("Sans", 12),
                bd=0,
                highlightthickness=0,
            )
            if photo is None:
                preview_label.configure(
                    width=max(8, cell_width // 10),
                    height=max(4, (cell_width * 3) // (8 * 16)),
                )
            preview_label.pack(fill="x", pady=(0, 8))
            labels.append(preview_label)

            button = RoundedButton(
                col,
                text=SCENARIO_LABELS[scenario],
                command=lambda i=index, s=scenario: self.play_or_stop(i, s),
                bg=PLAY_BLUE,
                activebackground=PLAY_BLUE_ACTIVE,
                font=("Sans", 16, "bold"),
                width=80,
                height=70,
                radius=26,
                on_enter=lambda t=task, s=scenario, l=label: self._show_condition_hint(t, s, l),
                on_leave=self._clear_condition_hint,
            )
            button.pack(fill="x")
            buttons.append(button)

        self.preview_sources.append(sources)
        self.preview_photos.append(photos)
        self.preview_labels.append(labels)
        self.task_buttons.append(buttons)

    def _set_status(self, text: str, fg: str, *, sticky: bool = False):
        """Update the status bar; ``sticky`` also becomes the post-hover restore text."""
        self.status.configure(text=text, fg=fg)
        if sticky:
            self._idle_status = text
            self._idle_status_fg = fg

    def _show_condition_hint(self, task: str, scenario: str, label: str):
        """Show the README condition blurb while the pointer is over a scenario key."""
        if self.child is not None:
            return
        desc = condition_description(task, scenario)
        scenario_label = SCENARIO_LABELS[scenario]
        if desc:
            text = f"{label} · {scenario_label}: {desc}"
        else:
            text = f"{label} · {scenario_label}"
        self.status.configure(text=text, fg="#7fb6dc")

    def _clear_condition_hint(self):
        if self.child is not None:
            return
        self.status.configure(text=self._idle_status, fg=self._idle_status_fg)

    def play_or_stop(self, index: int, scenario: str):
        if self.child is not None:
            if self.active_selection == (index, scenario):
                self._stop_task("Task stopped. Select another scenario when ready.")
            else:
                self._set_status(
                    "Stop the running task before starting another.",
                    "#e6a15c",
                    sticky=True,
                )
            return
        self._start_task(index, scenario)

    def play_or_stop_tutorial(self, index: int):
        if self.child is not None:
            if self.active_selection == ("tutorial", index):
                self._stop_task("Tutorial stopped. Select a part or task when ready.")
            else:
                self._set_status(
                    "Stop the running session before starting another.",
                    "#e6a15c",
                    sticky=True,
                )
            return
        self._start_tutorial(index)

    def _mark_running_buttons(self):
        """Disable every Play control except the active Stop button."""
        for task_index, row in enumerate(self.task_buttons):
            for button_scenario, button in zip(SCENARIOS, row):
                is_active = self.active_selection == (task_index, button_scenario)
                button.configure(state="normal" if is_active else "disabled")
        for part_index, button in enumerate(self.tutorial_buttons):
            is_active = self.active_selection == ("tutorial", part_index)
            button.configure(state="normal" if is_active else "disabled")

    def _start_tutorial(self, index: int):
        label, script_stem, hint = TUTORIAL_PARTS[index]
        script = TUTORIAL_DIR / f"{script_stem}.py"
        if not script.exists():
            messagebox.showerror("Tutorial unavailable", f"Missing launcher:\n{script}")
            return
        try:
            seed = resolve_seed(self.seed_entry.get())
        except ValueError as exc:
            messagebox.showerror("Invalid seed", str(exc))
            self.seed_entry.focus_set()
            return

        control_mode = str(self.control.get() or "robot")
        if bool(self.show_briefing.get()):
            if index == 0:
                summary = (
                    "Empty table with both arms. Test arm selection (1 / 2 / 3), "
                    "then press V to switch camera views."
                )
                instruction = (
                    "Key figures appear at the top right of the viewer. "
                    "Press 1, 2, and 3 to select left / right / both arms. "
                    "When those are tested, the overlay switches to V — press V to cycle "
                    "head and gripper views."
                )
            elif index == 1:
                summary = (
                    "Empty table with both arms. Practice the base teleop keys "
                    "on the selected (green) arm."
                )
                instruction = (
                    "The left arm starts selected. Key figures at the top right "
                    "walk through: arrow keys (move), E/Q (height — Z min/max is capped), "
                    "R/T (rotate), F/G (tilt), then Space twice (open and close). "
                    "Each lesson key turns green. After that a smaller overlay stays up "
                    "to practice — keys flash while you press them. Esc quits."
                )
            elif index == 2:
                summary = (
                    "Four basic actions on the left side of the table, one at a time: "
                    "pick up a cube, hold a spring button, toggle an on/off switch, "
                    "then push a box to a green line."
                )
                instruction = (
                    "The left arm starts selected. Key figures at the top right show "
                    "which keys to use. (1) Grasp the orange cube — Space to close, "
                    "E to lift. (2) Hold Q on the green button until it goes red, then "
                    "lift off with E. (3) Press the switch ON (stays down, red) then "
                    "press again to turn it OFF. (4) Close the gripper and push the "
                    "blue box onto the green line."
                )
            else:
                summary = (
                    "Four advanced actions on the left side of the table, one at a time: "
                    "catch a rolling ball, turn the stove knob on and off, pick up a mallet, "
                    "then press a multi-stage key to a target force."
                )
                instruction = (
                    "The left arm starts selected. The overlay window is larger so the "
                    "instructions stay on screen. (1) Catch the red ball as it rolls toward "
                    "you — Space to close, E to lift. If it falls or you drop it, it respawns. "
                    "(2) Grasp the stove knob, yaw left until fire appears, yaw back to turn "
                    "it off. (3) Grasp the mallet handle and lift it. (4) Press Q on the blue "
                    "key: the bar shows force. Hit the yellow target band, then release. "
                    "Too light or too hard — try again."
                )
            briefing = build_briefing_text(
                label=f"Tutorial · {label}",
                task="tutorial_empty",
                scenario_label=label,
                scenario_desc=hint,
                summary=summary,
                control_mode=control_mode,
                script_path=TUTORIAL_DIR / "_run.py",
            )
            briefing["instruction"] = instruction
            if not show_task_briefing(self, briefing):
                self._set_status(
                    "Briefing cancelled. Select a tutorial part when ready.",
                    TEXT_SECONDARY,
                    sticky=True,
                )
                return

        try:
            child_env = os.environ.copy()
            child_env.setdefault(
                "PYTHONWARNINGS",
                "ignore::UserWarning,ignore::FutureWarning,ignore::DeprecationWarning",
            )
            self._prepare_result_file()
            child_env[TASK_RESULT_ENV] = str(self.result_file)
            command = [
                sys.executable,
                str(script),
                "--config",
                "demo_dynamic",
                "--seed",
                str(seed),
                "--control",
                control_mode,
                "--part",
                str(index + 1),
            ]
            self.child = subprocess.Popen(
                command, cwd=ROOT, start_new_session=True, env=child_env
            )
        except Exception as exc:
            self._remove_result_file()
            messagebox.showerror("Could not start tutorial", str(exc))
            self.child = None
            return

        self.active_selection = ("tutorial", index)
        self._mark_running_buttons()
        self.control.configure(state="disabled")
        self.seed_entry.configure(state="disabled")
        self.briefing_check.configure(state="disabled")
        self.tutorial_buttons[index].configure(
            text="Stop", bg="#b06a20", activebackground="#d0842b"
        )
        run_text = (
            f"Running Tutorial / {label} with seed {seed}. "
            "Close its viewer or press Stop."
        )
        self._run_status_base = run_text
        self._shown_episode_condition = None
        self._set_status(run_text, "#70d6a2", sticky=True)

    def _write_temporary_config(self, task: str, scenario: str) -> str:
        config = build_scenario_config(task, scenario)
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=".interactive_gui_",
            suffix=".yml",
            dir=CONFIG_DIR,
            delete=False,
        )
        try:
            yaml.safe_dump(config, handle, sort_keys=False)
        finally:
            handle.close()
        self.temporary_config = Path(handle.name)
        return self.temporary_config.stem

    def _start_task(self, index: int, scenario: str):
        label, task = TASKS[index]
        script = SCRIPT_DIR / f"interactive_{task}.py"
        if not script.exists():
            messagebox.showerror("Task unavailable", f"Missing launcher:\n{script}")
            return
        try:
            seed = resolve_seed(self.seed_entry.get())
        except ValueError as exc:
            messagebox.showerror("Invalid seed", str(exc))
            self.seed_entry.focus_set()
            return

        control_mode = str(self.control.get() or "robot")
        if bool(self.show_briefing.get()):
            briefing = build_briefing_text(
                label=label,
                task=task,
                scenario_label=SCENARIO_LABELS.get(scenario, scenario),
                scenario_desc=condition_description(task, scenario),
                summary=task_summary(task),
                control_mode=control_mode,
                script_path=script,
            )
            if not show_task_briefing(self, briefing):
                self._set_status(
                    "Briefing cancelled. Select a scenario when ready.",
                    TEXT_SECONDARY,
                    sticky=True,
                )
                return

        try:
            config_name = self._write_temporary_config(task, scenario)
            child_env = os.environ.copy()
            child_env.setdefault(
                "PYTHONWARNINGS",
                "ignore::UserWarning,ignore::FutureWarning,ignore::DeprecationWarning",
            )
            # Authoritative scenario for tasks that read ROBODYNA_SCENARIO
            # (pack_fruits applies Opt1/Opt2 flags from this even if a stale
            # temp yml is missing the new keys).
            child_env["ROBODYNA_SCENARIO"] = scenario
            self._prepare_result_file()
            child_env[TASK_RESULT_ENV] = str(self.result_file)
            command = [
                sys.executable,
                str(script),
                "--config",
                config_name,
                "--seed",
                str(seed),
                "--control",
                control_mode,
            ]
            # Prefer an explicit CLI scenario when the interactive script
            # understands it (pack_fruits); unknown flags are avoided below.
            if task == "pack_fruits":
                command.extend(["--scenario", scenario])
            self.child = subprocess.Popen(
                command, cwd=ROOT, start_new_session=True, env=child_env
            )
        except Exception as exc:
            self._remove_temporary_config()
            self._remove_result_file()
            messagebox.showerror("Could not start task", str(exc))
            self.child = None
            return

        self.active_selection = (index, scenario)
        self._mark_running_buttons()
        self.control.configure(state="disabled")
        self.seed_entry.configure(state="disabled")
        self.briefing_check.configure(state="disabled")
        active_button = self.task_buttons[index][SCENARIOS.index(scenario)]
        active_button.configure(text="Stop", bg="#b06a20", activebackground="#d0842b")
        desc = condition_description(task, scenario)
        run_text = (
            f"Running {label} / {SCENARIO_LABELS[scenario]} with seed {seed}. "
            "Close its viewer or press Stop."
        )
        if desc:
            run_text = f"{run_text}  ({desc})"
        self._run_status_base = run_text
        self._shown_episode_condition = None
        self._set_status(run_text, "#70d6a2", sticky=True)

    def _poll_child(self):
        if self.child is not None:
            code = self.child.poll()
            if code is not None:
                self.child = None
                self.active_selection = None
                payload = self._read_result_payload()
                reason = None
                if isinstance(payload, dict):
                    detail = payload.get("detail")
                    if isinstance(detail, str) and detail.strip():
                        reason = detail.strip()
                self._run_status_base = None
                self._shown_episode_condition = None
                self._remove_temporary_config()
                self._remove_result_file()
                self._reset_task_buttons()
                # Match household_task_gui: 0=SUCCESS, 10=FAILURE, 2=closed early.
                if code == 0:
                    self._set_status(
                        "Task result: SUCCESS. Select another scenario below.",
                        "#70d6a2",
                        sticky=True,
                    )
                elif code == 10:
                    msg = "Task result: FAILURE"
                    if reason:
                        msg = f"{msg} ({reason})"
                    self._set_status(
                        f"{msg}. Select another scenario below.",
                        "#e6a15c",
                        sticky=True,
                    )
                elif code == 2:
                    self._set_status(
                        "Task closed before a result was reached.",
                        TEXT_SECONDARY,
                        sticky=True,
                    )
                else:
                    self._set_status(
                        f"Task result: ERROR (exit status {code}). Check the terminal.",
                        "#e6a15c",
                        sticky=True,
                    )
            else:
                # While running, surface the episode-specific condition once available.
                cond = self._read_result_condition()
                if cond and cond != getattr(self, "_shown_episode_condition", None):
                    self._shown_episode_condition = cond
                    base = getattr(self, "_run_status_base", None) or "Running task."
                    self._set_status(f"{base}  Condition: {cond}", "#70d6a2", sticky=True)
        self.after(250, self._poll_child)

    def _reset_task_buttons(self):
        self.control.configure(state="readonly")
        self.seed_entry.configure(state="normal")
        self.briefing_check.configure(state="normal")
        for row in self.task_buttons:
            for scenario, button in zip(SCENARIOS, row):
                button.configure(
                    state="normal",
                    text=SCENARIO_LABELS[scenario],
                    bg=PLAY_BLUE,
                    activebackground=PLAY_BLUE_ACTIVE,
                )
        for index, button in enumerate(self.tutorial_buttons):
            button.configure(
                state="normal",
                text=TUTORIAL_PARTS[index][0],
                bg=PLAY_BLUE,
                activebackground=PLAY_BLUE_ACTIVE,
            )

    def _remove_temporary_config(self):
        path = self.temporary_config
        self.temporary_config = None
        if path is not None:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    def _prepare_result_file(self):
        self._remove_result_file()
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix="robodyna_interactive_result_",
            delete=False,
        )
        handle.close()
        self.result_file = Path(handle.name)

    def _read_result_payload(self) -> dict | None:
        path = self.result_file
        if path is None or not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            return None
        return data if isinstance(data, dict) else None

    def _read_result_detail(self) -> str | None:
        data = self._read_result_payload()
        if not data:
            return None
        detail = data.get("detail")
        if isinstance(detail, str):
            detail = detail.strip()
            return detail or None
        return None

    def _read_result_condition(self) -> str | None:
        data = self._read_result_payload()
        if not data:
            return None
        cond = data.get("condition")
        if isinstance(cond, str):
            cond = cond.strip()
            return cond or None
        return None

    def _remove_result_file(self):
        path = self.result_file
        self.result_file = None
        if path is not None:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    def _stop_task(self, status=None):
        child = self.child
        if child is None:
            self._remove_temporary_config()
            self._remove_result_file()
            return
        try:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=2)
        except ProcessLookupError:
            pass
        finally:
            self.child = None
            self.active_selection = None
            self._remove_temporary_config()
            self._remove_result_file()
            self._reset_task_buttons()
        if status:
            self._set_status(status, "#e6a15c", sticky=True)

    def exit_app(self):
        self._stop_task()
        self.destroy()


def main():
    InteractiveTaskLauncher().mainloop()


if __name__ == "__main__":
    main()
