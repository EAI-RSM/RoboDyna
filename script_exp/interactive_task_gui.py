#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Scrollable graphical launcher for the dynamic interactive tasks."""
from __future__ import annotations

import os
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


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "task_config"
DEMO_DIR = ROOT / "final_task_demos"

TASKS = (
    ("Catch Marbles Trapdoors", "catch_marbles_trapdoors"),
    ("Catch Ramp Ball", "catch_ramp_ball"),
    ("Catch Rat", "catch_rat"),
    ("Catch Shelf Marble", "catch_shelf_marble"),
    ("Catch Valley Ball", "catch_valley_ball"),
    ("Catch Valley Ball V1", "catch_valley_ball_v1"),
    ("Stop Valley Ball", "stop_valley_ball"),
    ("Cook Meat", "cook_meat"),
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
    "catch_rat": {
        "default": {"catch_two_mice": False, "opaque_surface": False},
        "opt1": {"catch_two_mice": True, "opaque_surface": False},
        "opt2": {"catch_two_mice": False, "opaque_surface": True},
        "opt1+2": {"catch_two_mice": True, "opaque_surface": True},
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
    "catch_valley_ball_v1": {
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
        "default": {"spawn_mode": "parallel", "pair_stagger_enabled": False, "single_wave_any_belt": False, "distractor_enabled": False},
        "opt1": {"spawn_mode": "random", "pair_stagger_enabled": True, "single_wave_any_belt": True, "distractor_enabled": False},
        "opt2": {"spawn_mode": "parallel", "pair_stagger_enabled": False, "single_wave_any_belt": False, "distractor_enabled": True},
        "opt1+2": {"spawn_mode": "random", "pair_stagger_enabled": True, "single_wave_any_belt": True, "distractor_enabled": True},
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
PAGE_BG = "#111820"
HEADER_BG = "#1b2733"
CARD_BG = "#202c38"
CARD_BORDER = "#405367"
TEXT_PRIMARY = "#f4f8fb"
TEXT_SECONDARY = "#aebdca"
RANDOM_SEED_MAX = 500


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
    task_args.update(SCENARIO_OVERRIDES[task][scenario])
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

    def _leave(self, _event):
        if self.button_state == "normal" and self._shape is not None:
            self.itemconfigure(self._shape, fill=self.normal_color)

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
        if options:
            super().configure(**options)
        self._redraw()

    config = configure


class InteractiveTaskLauncher(tk.Tk):
    # Side-by-side demos are 8:3 (e.g. 960x360). Size previews to fill a
    # typical window width while keeping that aspect — avoids letterbox bars.
    IMAGE_SIZE = (1400, 525)
    CARD_PAD = 8
    PREVIEW_SIDE_PAD = 28

    def __init__(self):
        super().__init__()
        self.title("Interactive Tasks")
        self.geometry("1600x1000")
        self.minsize(1100, 760)
        self.configure(bg=PAGE_BG)
        self.protocol("WM_DELETE_WINDOW", self.exit_app)

        self.child: subprocess.Popen | None = None
        self.active_selection: tuple[int, str] | None = None
        self.temporary_config: Path | None = None
        self.preview_sources: list[Image.Image | None] = []
        self.preview_photos: list[ImageTk.PhotoImage | None] = []
        self.preview_labels: list[tk.Label] = []
        self.task_buttons: list[list[RoundedButton]] = []
        self._preview_resize_job: str | None = None
        self._preview_width = self.IMAGE_SIZE[0]

        self._build_ui()
        self.after(250, self._poll_child)

    def _build_ui(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure(
            "Task.TCombobox",
            padding=(14, 10),
            arrowsize=26,
            fieldbackground="#f7fafc",
            foreground="#182633",
            selectbackground="#f7fafc",
            selectforeground="#182633",
        )

        header = tk.Frame(
            self,
            bg=HEADER_BG,
            highlightbackground="#34495d",
            highlightthickness=1,
        )
        header.pack(fill="x", padx=24, pady=(18, 12))

        heading = tk.Frame(header, bg=HEADER_BG)
        heading.pack(side="left", padx=24, pady=18)
        tk.Label(
            heading,
            text="Interactive Tasks",
            bg=HEADER_BG,
            fg=TEXT_PRIMARY,
            font=("Sans", 34, "bold"),
        ).pack(anchor="w")
        tk.Label(
            heading,
            text="Choose a task and one of its four scenario variants.",
            bg=HEADER_BG,
            fg=TEXT_SECONDARY,
            font=("Sans", 14),
        ).pack(anchor="w", pady=(3, 0))

        controls = tk.Frame(header, bg=HEADER_BG)
        controls.pack(side="right", padx=22, pady=16)

        seed_group = tk.Frame(controls, bg=HEADER_BG)
        seed_group.pack(side="left", padx=(0, 18))
        tk.Label(
            seed_group,
            text="Seed (blank = random)",
            bg=HEADER_BG,
            fg=TEXT_SECONDARY,
            font=("Sans", 13, "bold"),
        ).pack(anchor="w")
        self.seed_entry = tk.Entry(
            seed_group,
            width=14,
            font=("Sans", 22, "bold"),
            bg="#f7fafc",
            fg="#182633",
            insertbackground="#182633",
            relief="flat",
        )
        self.seed_entry.pack(ipady=8, pady=(4, 0))

        control_group = tk.Frame(controls, bg=HEADER_BG)
        control_group.pack(side="left", padx=(0, 14))
        tk.Label(
            control_group,
            text="Control",
            bg=HEADER_BG,
            fg=TEXT_SECONDARY,
            font=("Sans", 13, "bold"),
        ).pack(anchor="w")
        self.control = ttk.Combobox(
            control_group,
            values=("keyboard", "robot"),
            state="readonly",
            width=10,
            font=("Sans", 22, "bold"),
            style="Task.TCombobox",
        )
        self.control.set("robot")
        self.control.pack(pady=(4, 0))

        self.exit_button = RoundedButton(
            controls,
            text="Exit",
            command=self.exit_app,
            bg="#e34a33",
            activebackground="#eb6854",
            font=("Sans", 17, "bold"),
            width=150,
            height=76,
            radius=28,
        )
        self.exit_button.pack(side="left", pady=(18, 0))

        self.status = tk.Label(
            self,
            text=f"{len(TASKS)} tasks available  |  Select a scenario below.",
            bg=PAGE_BG,
            fg=TEXT_SECONDARY,
            anchor="w",
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
        self.canvas.bind_all("<MouseWheel>", self._mousewheel)
        self.canvas.bind_all("<Button-4>", lambda _event: self.canvas.yview_scroll(-3, "units"))
        self.canvas.bind_all("<Button-5>", lambda _event: self.canvas.yview_scroll(3, "units"))

        for index, (label, task) in enumerate(TASKS):
            self._add_task_card(index, label, task)

    def _update_scroll_region(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _resize_page(self, event):
        page_width = max(event.width, 900)
        self.canvas.itemconfigure(self.page_window, width=page_width)
        preview_width = max(
            720, page_width - 2 * (self.CARD_PAD + self.PREVIEW_SIDE_PAD)
        )
        if abs(preview_width - self._preview_width) < 8:
            return
        if self._preview_resize_job is not None:
            self.after_cancel(self._preview_resize_job)
        self._preview_resize_job = self.after(
            80, lambda width=preview_width: self._apply_preview_width(width)
        )

    def _apply_preview_width(self, width: int):
        self._preview_resize_job = None
        self._preview_width = width
        for index, source in enumerate(self.preview_sources):
            photo = self._render_preview(source, width)
            self.preview_photos[index] = photo
            label = self.preview_labels[index]
            if photo is None:
                label.configure(
                    image="",
                    text="No preview available",
                    width=width // 12,
                    height=max(6, (width * 3) // (8 * 18)),
                )
            else:
                label.configure(image=photo, text="", width=0, height=0)

    def _mousewheel(self, event):
        if event.delta:
            self.canvas.yview_scroll(-int(event.delta / 120), "units")

    @staticmethod
    def _preview_path(task: str) -> Path | None:
        directory = DEMO_DIR / task
        preferred = directory / "default_sidebyside.gif"
        if preferred.exists():
            return preferred
        if directory.exists():
            patterns = ("default*sidebyside.gif", "*.gif", "*.png", "*.jpg", "*.jpeg")
            for pattern in patterns:
                match = next(directory.glob(pattern), None)
                if match is not None:
                    return match
        return None

    def _load_preview_source(self, task: str) -> Image.Image | None:
        path = self._preview_path(task)
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
        # Fill the card width at the image's native aspect ratio (no letterbox).
        aspect = source.width / max(source.height, 1)
        height = max(1, int(round(width / aspect)))
        image = source.resize((width, height), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(image)

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
        tk.Label(
            card_header,
            text=f"{index + 1:02d}",
            bg=PLAY_BLUE,
            fg="white",
            font=("Sans", 17, "bold"),
            padx=13,
            pady=6,
        ).pack(side="left", padx=(0, 14))
        tk.Label(
            card_header,
            text=label,
            bg=CARD_BG,
            fg=TEXT_PRIMARY,
            anchor="w",
            font=("Sans", 27, "bold"),
        ).pack(side="left", fill="x", expand=True)
        tk.Label(
            card_header,
            text="4 SCENARIOS",
            bg=CARD_BG,
            fg="#7fb6dc",
            font=("Sans", 12, "bold"),
        ).pack(side="right")

        preview_holder = tk.Frame(card, bg=CARD_BG)
        preview_holder.pack(fill="x", padx=self.PREVIEW_SIDE_PAD)
        source = self._load_preview_source(task)
        photo = self._render_preview(source, self._preview_width)
        self.preview_sources.append(source)
        self.preview_photos.append(photo)
        preview_label = tk.Label(
            preview_holder,
            image=photo,
            text="No preview available" if photo is None else "",
            bg=CARD_BG,
            fg=TEXT_SECONDARY,
            font=("Sans", 16),
            bd=0,
            highlightthickness=0,
        )
        if photo is None:
            preview_label.configure(
                width=self._preview_width // 12,
                height=max(6, (self._preview_width * 3) // (8 * 18)),
            )
        preview_label.pack(fill="x")
        self.preview_labels.append(preview_label)

        action_row = tk.Frame(card, bg=CARD_BG)
        action_row.pack(fill="x", padx=self.PREVIEW_SIDE_PAD, pady=(12, 16))
        buttons = []
        for scenario in SCENARIOS:
            button = RoundedButton(
                action_row,
                text=SCENARIO_LABELS[scenario],
                command=lambda i=index, s=scenario: self.play_or_stop(i, s),
                bg=PLAY_BLUE,
                activebackground=PLAY_BLUE_ACTIVE,
                font=("Sans", 18, "bold"),
                width=300,
                height=78,
                radius=30,
            )
            button.pack(side="left", expand=True, fill="x", padx=6)
            buttons.append(button)
        self.task_buttons.append(buttons)

    def play_or_stop(self, index: int, scenario: str):
        if self.child is not None:
            if self.active_selection == (index, scenario):
                self._stop_task("Task stopped. Select another scenario when ready.")
            else:
                self.status.configure(text="Stop the running task before starting another.", fg="#e6a15c")
            return
        self._start_task(index, scenario)

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

        try:
            config_name = self._write_temporary_config(task, scenario)
            command = [
                sys.executable,
                str(script),
                "--config",
                config_name,
                "--seed",
                str(seed),
                "--control",
                self.control.get(),
            ]
            self.child = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
        except Exception as exc:
            self._remove_temporary_config()
            messagebox.showerror("Could not start task", str(exc))
            self.child = None
            return

        self.active_selection = (index, scenario)
        for task_index, row in enumerate(self.task_buttons):
            for button_scenario, button in zip(SCENARIOS, row):
                button.configure(
                    state="normal" if (task_index, button_scenario) == self.active_selection else "disabled"
                )
        self.control.configure(state="disabled")
        self.seed_entry.configure(state="disabled")
        active_button = self.task_buttons[index][SCENARIOS.index(scenario)]
        active_button.configure(text="Stop", bg="#b06a20", activebackground="#d0842b")
        self.status.configure(
            text=f"Running {label} / {SCENARIO_LABELS[scenario]} with seed {seed}. Close its viewer or press Stop.",
            fg="#70d6a2",
        )

    def _poll_child(self):
        if self.child is not None:
            code = self.child.poll()
            if code is not None:
                self.child = None
                self.active_selection = None
                self._remove_temporary_config()
                self._reset_task_buttons()
                # Match household_task_gui: 0=SUCCESS, 10=FAILURE, 2=closed early.
                if code == 0:
                    self.status.configure(
                        text="Task result: SUCCESS. Select another scenario below.",
                        fg="#70d6a2",
                    )
                elif code == 10:
                    self.status.configure(
                        text="Task result: FAILURE. Select another scenario below.",
                        fg="#e6a15c",
                    )
                elif code == 2:
                    self.status.configure(
                        text="Task closed before a result was reached.",
                        fg=TEXT_SECONDARY,
                    )
                else:
                    self.status.configure(
                        text=f"Task result: ERROR (exit status {code}). Check the terminal.",
                        fg="#e6a15c",
                    )
        self.after(250, self._poll_child)

    def _reset_task_buttons(self):
        self.control.configure(state="readonly")
        self.seed_entry.configure(state="normal")
        for row in self.task_buttons:
            for scenario, button in zip(SCENARIOS, row):
                button.configure(
                    state="normal",
                    text=SCENARIO_LABELS[scenario],
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

    def _stop_task(self, status=None):
        child = self.child
        if child is None:
            self._remove_temporary_config()
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
            self._reset_task_buttons()
        if status:
            self.status.configure(text=status, fg="#e6a15c")

    def exit_app(self):
        self._stop_task()
        self.destroy()


def main():
    InteractiveTaskLauncher().mainloop()


if __name__ == "__main__":
    main()
