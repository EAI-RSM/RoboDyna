#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Scrollable graphical launcher for the household interactive tasks."""
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

from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent / "household"
DEMO_DIR = ROOT / "final_task_demos"
README_PATH = ROOT / "README.md"
# Must match interactive._interactive_common.TASK_RESULT_ENV
TASK_RESULT_ENV = "ROBODYNA_TASK_RESULT_FILE"

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

TASKS = (
    ("Trap Bug", "trap_bug", "interactive_trap_bug.py"),
    ("Boil Milk", "boil_milk", "interactive_boil_milk.py"),
    ("Fill Coffee Jar", "fill_coffee_jar", "interactive_fill_coffee_jar.py"),
    ("Pour Beer", "pour_beer", "interactive_pour_beer.py"),
    ("Cook Food", "cook_food", "interactive_cook_food.py"),
    ("Cook Food Timer", "cook_food_timer", "interactive_cook_food_timer.py"),
    ("Measure Ingredient", "measure_ingredient", "interactive_measure_ingredient.py"),
    ("Make Soup", "make_soup", "interactive_make_soup.py"),
    ("Catch Cup", "catch_cup", "interactive_catch_cup.py"),
    ("Catch Mouse Object Drop", "catch_mouse_object_drop", "interactive_catch_mouse_object_drop.py"),
    ("Stop Ball", "stop_ball", "interactive_stop_ball.py"),
    ("Clean Table", "clean_table", "interactive_clean_table.py"),
)
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
HINT_FG = "#7fb6dc"
RANDOM_SEED_MAX = 500

def load_task_descriptions(readme_path: Path = README_PATH) -> dict[str, str]:
    """Parse README household-task ``<sub>…</sub>`` blurbs into task → description."""
    if not readme_path.exists():
        return {}
    text = readme_path.read_text(encoding="utf-8")
    # Prefer the Household Tasks section when present.
    section = text
    marker = "## Household Tasks"
    if marker in text:
        section = text.split(marker, 1)[1]
    row_re = re.compile(r"^\|\s*\*\*`([a-z0-9_]+)`\*\*.*$", re.MULTILINE)
    sub_re = re.compile(r"<sub>(.*?)</sub>", re.DOTALL)
    out: dict[str, str] = {}
    for match in row_re.finditer(section):
        task = match.group(1)
        line_end = section.find("\n", match.start())
        line = section[match.start() : line_end if line_end >= 0 else None]
        subs = [re.sub(r"\s+", " ", s).strip() for s in sub_re.findall(line)]
        if not subs:
            continue
        out[task] = subs[0]
    return out


TASK_DESCRIPTIONS = load_task_descriptions()


def task_description(task: str) -> str:
    """README blurb for ``task``, or empty if unavailable."""
    return str(TASK_DESCRIPTIONS.get(task, "") or "")


def resolve_seed(value: str) -> int:
    """Return the entered seed, or generate a fresh one when left blank."""
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


class RoundedButton(tk.Canvas):
    """Canvas-backed button with genuinely rounded corners."""

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
        radius=28,
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
        self._shape = None
        self._label = None
        self._text = text
        self._font = font
        self._on_enter_cb = on_enter
        self._on_leave_cb = on_leave
        self.bind("<Configure>", self._redraw)
        self.bind("<Enter>", self._enter)
        self.bind("<Leave>", self._leave)
        self.bind("<ButtonRelease-1>", self._click)

    def _rounded_polygon(self, width, height):
        r = min(self.radius, width // 2, height // 2)
        return (
            r, 1, width - r, 1, width - 1, 1, width - 1, r,
            width - 1, height - r, width - 1, height - 1,
            width - r, height - 1, r, height - 1, 1, height - 1,
            1, height - r, 1, r, 1, 1,
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
        self._label = self.create_text(
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


class HouseholdTaskLauncher(tk.Tk):
    # Head-camera stills are ~4:3. Width seeds the first layout; height follows
    # each image's native aspect so previews are not letterboxed.
    IMAGE_SIZE = (1400, 1050)
    COLUMNS = 4
    CARD_PAD = 8
    PREVIEW_SIDE_PAD = 28
    DESIGN_SIZE = (1560, 980)
    UI_SCALE_MIN = 0.55
    UI_SCALE_MAX = 1.15

    def __init__(self):
        # className becomes StartupWMClass for the Ubuntu dock .desktop entry.
        super().__init__(className=GUI_WM_CLASS["household"])
        self.title("Household Tasks")
        self.geometry("1560x980")
        self.minsize(900, 640)
        self.configure(bg=PAGE_BG)
        setup_gui_app_icon(
            self,
            suite="household",
            script_path=Path(__file__),
        )
        self.protocol("WM_DELETE_WINDOW", self.exit_app)

        self.child: subprocess.Popen | None = None
        self.active_index: int | None = None
        self.result_file: Path | None = None
        self.preview_sources: list[Image.Image | None] = []
        self.preview_photos: list[ImageTk.PhotoImage | None] = []
        self.preview_labels: list[tk.Label] = []
        self.task_buttons: list[RoundedButton] = []
        self.card_index_labels: list[tk.Label] = []
        self.card_title_labels: list[tk.Label] = []
        self._preview_resize_job: str | None = None
        self._ui_scale_job: str | None = None
        self._preview_width = self.IMAGE_SIZE[0]
        self._ui_scale = 1.0
        self._header_narrow: bool | None = None
        self._idle_status = (
            f"{len(TASKS)} tasks available  |  Hover a task for its README description."
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

        self.heading = tk.Frame(self.header, bg=HEADER_BG)
        self.heading.pack(side="left", padx=24, pady=18)
        self.title_label = tk.Label(
            self.heading,
            text="Household Tasks",
            bg=HEADER_BG,
            fg=HEADER_FG,
            font=("Sans", 34, "bold"),
        )
        self.title_label.pack(anchor="w")
        self.subtitle_label = tk.Label(
            self.heading,
            text="Select one of the household tasks",
            bg=HEADER_BG,
            fg=HEADER_MUTED,
            font=("Sans", 14),
        )
        self.subtitle_label.pack(anchor="w", pady=(3, 0))

        self.controls = tk.Frame(self.header, bg=HEADER_BG)
        self.controls.pack(side="right", padx=22, pady=16)

        # Pack Exit first on the right so it stays visible when the header is tight.
        self.exit_button = RoundedButton(
            self.controls,
            text="Exit",
            command=self.exit_app,
            bg="#e34a33",
            activebackground="#eb6854",
            font=("Sans", 17, "bold"),
            width=150,
            height=76,
            radius=28,
        )
        self.exit_button.pack(side="right", pady=(18, 0))

        brief_group = tk.Frame(self.controls, bg=HEADER_BG)
        brief_group.pack(side="left", padx=(0, 16))
        self.brief_caption = tk.Label(
            brief_group,
            text="Briefing",
            bg=HEADER_BG,
            fg=HEADER_MUTED,
            font=("Sans", 13, "bold"),
        )
        self.brief_caption.pack(anchor="w")
        self.show_briefing = tk.BooleanVar(value=True)
        self.briefing_check = tk.Checkbutton(
            brief_group,
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

        seed_group = tk.Frame(self.controls, bg=HEADER_BG)
        seed_group.pack(side="left", padx=(0, 18))
        self.seed_caption = tk.Label(
            seed_group,
            text="Seed (blank = random)",
            bg=HEADER_BG,
            fg=HEADER_MUTED,
            font=("Sans", 13, "bold"),
        )
        self.seed_caption.pack(anchor="w")
        self.seed_entry = tk.Entry(
            seed_group,
            width=16,
            font=("Sans", 13, "bold"),
            bg="#f7fafc",
            fg="#182633",
            insertbackground="#182633",
            relief="flat",
        )
        self.seed_entry.pack(ipady=5, pady=(4, 0))

        control_group = tk.Frame(self.controls, bg=HEADER_BG)
        control_group.pack(side="left", padx=(0, 14))
        self.control_caption = tk.Label(
            control_group,
            text="Control",
            bg=HEADER_BG,
            fg=HEADER_MUTED,
            font=("Sans", 13, "bold"),
        )
        self.control_caption.pack(anchor="w")
        self.control = ttk.Combobox(
            control_group,
            values=("keyboard", "robot"),
            state="readonly",
            width=8,
            font=("Sans", 13, "bold"),
            style="Task.TCombobox",
        )
        self.control.set("robot")
        self.control.pack(pady=(4, 0))
        self._style_control_menu(("Sans", 13, "bold"))

        self.status = tk.Label(
            self,
            text=self._idle_status,
            bg=PAGE_BG,
            fg=self._idle_status_fg,
            anchor="w",
            justify="left",
            wraplength=1460,
            font=("Sans", 19),
        )
        self.status.pack(fill="x", padx=34, pady=(2, 12))

        # Scrollable 4-column task grid, matching the base suite layout.
        self.outer = tk.Frame(self, bg=PAGE_BG)
        self.outer.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.canvas = tk.Canvas(self.outer, bg=PAGE_BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.outer, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.page = tk.Frame(self.canvas, bg=PAGE_BG)
        self.page_window = self.canvas.create_window((0, 0), window=self.page, anchor="nw")
        self.page.bind("<Configure>", self._update_scroll_region)
        self.canvas.bind("<Configure>", self._resize_page)
        self.canvas.bind_all("<MouseWheel>", self._mousewheel)
        self.canvas.bind_all("<Button-4>", lambda event: self.canvas.yview_scroll(-3, "units"))
        self.canvas.bind_all("<Button-5>", lambda event: self.canvas.yview_scroll(3, "units"))

        for row_start in range(0, len(TASKS), self.COLUMNS):
            self._add_task_row(row_start)

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

    @staticmethod
    def _px(value: float, scale: float) -> int:
        return max(1, int(round(value * scale)))

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
        # Longer title than the base suite GUI — stack earlier so Exit stays on-screen.
        narrow = self.winfo_width() < 1320 or scale < 0.82
        if abs(scale - self._ui_scale) < 0.02:
            self.status.configure(
                wraplength=max(360, self.winfo_width() - self._px(68, scale))
            )
            if narrow != getattr(self, "_header_narrow", None):
                self._header_narrow = narrow
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
        self.control.pack_configure(pady=(self._px(4, s), 0))
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
        self.exit_button.pack_configure(pady=(self._px(18, s), 0))
        self.status.configure(
            font=self._scaled_font(19, scale=s),
            wraplength=max(360, self.winfo_width() - self._px(68, s)),
        )
        self.status.pack_configure(padx=self._px(34, s), pady=(self._px(2, s), self._px(12, s)))
        self.outer.pack_configure(padx=self._px(12, s), pady=(0, self._px(12, s)))

        idx_padx = self._px(13, s)
        idx_pady = self._px(6, s)
        for label in self.card_index_labels:
            label.configure(
                font=self._scaled_font(17, "bold", s),
                padx=idx_padx,
                pady=idx_pady,
            )
        for label in self.card_title_labels:
            label.configure(font=self._scaled_font(15, "bold", s))

        btn_h = self._px(70, s)
        btn_radius = self._px(26, s)
        btn_font = self._scaled_font(16, "bold", s)
        for button in self.task_buttons:
            button.configure(font=btn_font, height=btn_h, radius=btn_radius)

        self._header_narrow = narrow
        self._relayout_header(s)

    def _relayout_header(self, scale: float):
        """Keep header controls on-screen by stacking when the window is narrow."""
        narrow = self.winfo_width() < 1320 or scale < 0.82
        self.heading.pack_forget()
        self.controls.pack_forget()
        if narrow:
            self.heading.pack(
                side="top",
                anchor="w",
                padx=self._px(24, scale),
                pady=(self._px(14, scale), 0),
            )
            self.controls.pack(
                side="top",
                anchor="w",
                padx=self._px(22, scale),
                pady=(self._px(8, scale), self._px(14, scale)),
            )
        else:
            self.heading.pack(side="left", padx=self._px(24, scale), pady=self._px(18, scale))
            self.controls.pack(side="right", padx=self._px(22, scale), pady=self._px(16, scale))

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
        """Width of one of the four task preview columns."""
        gap = 6 * 2 * self.COLUMNS  # padx=6 on each side of each column
        return max(160, (total_width - gap) // self.COLUMNS)

    def _apply_preview_width(self, width: int):
        self._preview_resize_job = None
        self._preview_width = width
        cell_width = self._cell_preview_width(width)
        photos: list[ImageTk.PhotoImage | None] = []
        for index, source in enumerate(self.preview_sources):
            photo = self._render_preview(source, cell_width)
            photos.append(photo)
            label = self.preview_labels[index]
            if photo is None:
                label.configure(
                    image="",
                    text="No preview",
                    width=max(8, cell_width // 10),
                    height=max(4, (cell_width * 3) // (8 * 16)),
                )
            else:
                label.configure(image=photo, text="", width=0, height=0)
            if index < len(self.card_title_labels):
                self.card_title_labels[index].configure(
                    wraplength=max(80, cell_width - self._px(56, self._ui_scale))
                )
        self.preview_photos = photos

    def _mousewheel(self, event):
        if event.delta:
            self.canvas.yview_scroll(-int(event.delta / 120), "units")

    @staticmethod
    def _preview_path(task):
        directory = DEMO_DIR / task
        for name in ("scene_snapshot.png", "default_sidebyside.gif"):
            preferred = directory / name
            if preferred.exists():
                return preferred
        if directory.exists():
            for pattern in ("*.png", "*.jpg", "*.jpeg", "*.gif"):
                match = next(directory.glob(pattern), None)
                if match is not None:
                    return match
        return None

    def _load_preview_source(self, task) -> Image.Image | None:
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
        aspect = source.width / max(source.height, 1)
        height = max(1, int(round(width / aspect)))
        image = source.resize((width, height), Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(image)

    def _add_task_row(self, row_start: int):
        """One card holding up to four household tasks, matching the base GUI grid."""
        card = tk.Frame(
            self.page,
            bg=CARD_BG,
            highlightbackground=CARD_BORDER,
            highlightthickness=2,
        )
        card.pack(fill="x", padx=self.CARD_PAD, pady=12)
        grid = tk.Frame(card, bg=CARD_BG)
        grid.pack(fill="x", padx=self.PREVIEW_SIDE_PAD, pady=(14, 16))
        for index in range(row_start, min(row_start + self.COLUMNS, len(TASKS))):
            label, task, _script_name = TASKS[index]
            self._add_task_cell(grid, index, label, task)

    def _add_task_cell(self, grid, index, label, task):
        col = tk.Frame(grid, bg=CARD_BG)
        col.pack(side="left", expand=True, fill="both", padx=6)

        cell_header = tk.Frame(col, bg=CARD_BG)
        cell_header.pack(fill="x", pady=(0, 8))
        index_label = tk.Label(
            cell_header,
            text=f"{index + 1:02d}",
            bg=PLAY_BLUE,
            fg="white",
            font=("Sans", 17, "bold"),
            padx=13,
            pady=6,
        )
        index_label.pack(side="left", padx=(0, 8))
        title = tk.Label(
            cell_header,
            text=label,
            bg=CARD_BG,
            fg=TEXT_PRIMARY,
            anchor="w",
            justify="left",
            font=("Sans", 15, "bold"),
        )
        title.pack(side="left", fill="x", expand=True)
        for widget in (index_label, title):
            widget.bind("<Enter>", lambda _e, t=task, l=label: self._show_task_hint(t, l))
            widget.bind("<Leave>", lambda _e: self._clear_task_hint())
        self.card_index_labels.append(index_label)
        self.card_title_labels.append(title)

        cell_width = self._cell_preview_width(self._preview_width)
        source = self._load_preview_source(task)
        photo = self._render_preview(source, cell_width)
        self.preview_sources.append(source)
        self.preview_photos.append(photo)
        preview_label = tk.Label(
            col,
            image=photo,
            text="No preview" if photo is None else "",
            bg=CARD_BG,
            fg=TEXT_SECONDARY,
            font=("Sans", 12),
            bd=0,
            highlightthickness=0,
            cursor="hand2",
        )
        if photo is None:
            preview_label.configure(
                width=max(8, cell_width // 10),
                height=max(4, (cell_width * 3) // (8 * 16)),
            )
        preview_label.pack(fill="x", pady=(0, 8))
        preview_label.bind(
            "<Enter>",
            lambda _e, t=task, l=label: self._show_task_hint(t, l),
        )
        preview_label.bind("<Leave>", lambda _e: self._clear_task_hint())
        self.preview_labels.append(preview_label)

        button = RoundedButton(
            col,
            text="Play",
            command=lambda i=index: self.play_or_stop(i),
            bg=PLAY_BLUE,
            activebackground=PLAY_BLUE_ACTIVE,
            font=("Sans", 16, "bold"),
            width=220,
            height=70,
            radius=26,
            on_enter=lambda t=task, l=label: self._show_task_hint(t, l),
            on_leave=self._clear_task_hint,
        )
        button.pack(fill="x")
        self.task_buttons.append(button)

    def _set_status(self, text: str, fg: str, *, sticky: bool = False):
        """Update the status bar; ``sticky`` also becomes the post-hover restore text."""
        self.status.configure(text=text, fg=fg)
        if sticky:
            self._idle_status = text
            self._idle_status_fg = fg

    def _show_task_hint(self, task: str, label: str):
        """Show the README task blurb while the pointer is over preview / Play."""
        if self.child is not None:
            return
        desc = task_description(task)
        text = f"{label}: {desc}" if desc else label
        self.status.configure(text=text, fg=HINT_FG)

    def _clear_task_hint(self):
        if self.child is not None:
            return
        self.status.configure(text=self._idle_status, fg=self._idle_status_fg)

    def play_or_stop(self, index):
        if self.child is not None:
            if self.active_index == index:
                self._stop_task("Task stopped. Select another task when ready.")
            else:
                self._set_status(
                    "Stop the running task before starting another.",
                    "#e6a15c",
                    sticky=True,
                )
            return
        self._start_task(index)

    def _start_task(self, index):
        label, task, script_name = TASKS[index]
        script = SCRIPT_DIR / script_name
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
                scenario_label=None,
                scenario_desc=None,
                summary=task_description(task),
                control_mode=control_mode,
                script_path=script,
            )
            if not show_task_briefing(self, briefing):
                self._set_status(
                    "Briefing cancelled. Select a task when ready.",
                    TEXT_SECONDARY,
                    sticky=True,
                )
                return
        command = [
            sys.executable,
            str(script),
            "--seed",
            str(seed),
            "--control",
            control_mode,
        ]
        try:
            self._prepare_result_file()
            child_env = os.environ.copy()
            child_env[TASK_RESULT_ENV] = str(self.result_file)
            self.child = subprocess.Popen(
                command, cwd=ROOT, start_new_session=True, env=child_env
            )
        except Exception as exc:
            self._remove_result_file()
            messagebox.showerror("Could not start task", str(exc))
            self.child = None
            return
        self.active_index = index
        for i, button in enumerate(self.task_buttons):
            button.configure(state="normal" if i == index else "disabled")
        self.control.configure(state="disabled")
        self.seed_entry.configure(state="disabled")
        self.briefing_check.configure(state="disabled")
        self.task_buttons[index].configure(text="Stop", bg="#b06a20", activebackground="#d0842b")
        run_text = (
            f"Running {label} with seed {seed}. Close its viewer or press Stop to return."
        )
        self._run_status_base = run_text
        self._shown_episode_condition = None
        self._set_status(run_text, "#70d6a2", sticky=True)

    def _poll_child(self):
        if self.child is not None:
            code = self.child.poll()
            if code is not None:
                self.child = None
                self.active_index = None
                payload = self._read_result_payload()
                reason = None
                if isinstance(payload, dict):
                    detail = payload.get("detail")
                    if isinstance(detail, str) and detail.strip():
                        reason = detail.strip()
                self._run_status_base = None
                self._shown_episode_condition = None
                self._remove_result_file()
                self._reset_task_buttons()
                if code == 0:
                    self._set_status(
                        "Task result: SUCCESS. Select another task below.",
                        "#70d6a2",
                        sticky=True,
                    )
                elif code == 10:
                    msg = "Task result: FAILURE"
                    if reason:
                        msg = f"{msg} ({reason})"
                    self._set_status(
                        f"{msg}. Select another task below.",
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
        for button in self.task_buttons:
            button.configure(state="normal", text="Play", bg=PLAY_BLUE, activebackground=PLAY_BLUE_ACTIVE)

    def _prepare_result_file(self):
        self._remove_result_file()
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix="robodyna_household_result_",
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
            self.active_index = None
            self._remove_result_file()
            self._reset_task_buttons()
        if status:
            self._set_status(status, "#e6a15c", sticky=True)

    def exit_app(self):
        self._stop_task()
        self.destroy()


def main():
    HouseholdTaskLauncher().mainloop()


if __name__ == "__main__":
    main()
