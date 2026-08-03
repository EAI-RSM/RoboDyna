#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Scrollable graphical launcher for the household interactive tasks."""
from __future__ import annotations

import os
import secrets
import signal
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from PIL import Image, ImageTk


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
DEMO_DIR = ROOT / "final_task_demos"

TASKS = (
    ("Trap Bug", "trap_bug", "interactive_trap_bug.py"),
    ("Boil Milk", "boil_milk", "interactive_boil_milk.py"),
    ("Fill Coffee Jar", "fill_coffee_jar", "interactive_fill_coffee_jar.py"),
    ("Pour Beer", "pour_beer", "interactive_pour_beer.py"),
    ("Cook Food", "cook_food", "interactive_cook_food.py"),
    ("Measure Ingredient", "measure_ingredient", "interactive_measure_ingredient.py"),
    ("Make Soup", "make_soup", "interactive_make_soup.py"),
    ("Catch Cup", "catch_cup", "interactive_catch_cup.py"),
    ("Mouse Object Drop", "mouse_object_drop", "interactive_mouse_obj_drop.py"),
    ("Stop Ball", "stop_ball", "interactive_stop_ball.py"),
    ("Clean Table", "clean_table", "interactive_clean_table.py"),
)
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
        if self.button_state == "normal":
            self.itemconfigure(self._shape, fill=self.active_color)

    def _leave(self, _event):
        if self.button_state == "normal":
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


class HouseholdTaskLauncher(tk.Tk):
    # Three times the old 480×270 preview size.
    IMAGE_SIZE = (1440, 810)

    def __init__(self):
        super().__init__()
        self.title("Household Interactive Tasks")
        self.geometry("1560x980")
        self.minsize(980, 700)
        self.configure(bg=PAGE_BG)
        self.protocol("WM_DELETE_WINDOW", self.exit_app)

        self.child: subprocess.Popen | None = None
        self.active_index: int | None = None
        self.preview_photos: list[ImageTk.PhotoImage | None] = []
        self.task_buttons: list[RoundedButton] = []

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
            padding=(28, 18),
            arrowsize=42,
            fieldbackground="#f7fafc",
            foreground="#182633",
            selectbackground="#f7fafc",
            selectforeground="#182633",
        )
        # Popup was 156 pt / 276 px; reduce both by 30 percent.
        self.option_add("*TCombobox*Listbox.font", ("Sans", 109, "bold"))
        self.option_add("*TCombobox*Listbox.rowHeight", 193)

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
            text="Household Interactive Tasks",
            bg=HEADER_BG,
            fg=TEXT_PRIMARY,
            font=("Sans", 36, "bold"),
        ).pack(anchor="w")
        tk.Label(
            heading,
            text="Choose a scenario, deploy the robot, and return here when it finishes.",
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
        control_group.pack(side="left", padx=(0, 12))
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
            width=11,
            font=("Sans", 22, "bold"),
            style="Task.TCombobox",
        )
        self.control.set("robot")
        self.control.pack(ipady=4, pady=(4, 0))
        self.exit_button = RoundedButton(
            controls,
            text="Exit",
            command=self.exit_app,
            bg="#e34a33",
            activebackground="#eb6854",
            font=("Sans", 18, "bold"),
            width=190,
            height=88,
            radius=30,
        )
        self.exit_button.pack(side="left")

        self.status = tk.Label(
            self,
            text=f"{len(TASKS)} scenarios available  |  Select a task and press Play.",
            bg=PAGE_BG,
            fg=TEXT_SECONDARY,
            anchor="w",
            font=("Sans", 21),
        )
        self.status.pack(fill="x", padx=34, pady=(2, 12))

        # Scrollable table of task cards. The window stays fixed while the
        # vertically extended page contains every large task preview.
        outer = tk.Frame(self, bg=PAGE_BG)
        outer.pack(fill="both", expand=True, padx=22, pady=(0, 22))
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
        self.canvas.bind_all("<Button-4>", lambda event: self.canvas.yview_scroll(-3, "units"))
        self.canvas.bind_all("<Button-5>", lambda event: self.canvas.yview_scroll(3, "units"))

        for index, (label, task, script_name) in enumerate(TASKS):
            self._add_task_card(index, label, task, script_name)

    def _update_scroll_region(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _resize_page(self, event):
        self.canvas.itemconfigure(self.page_window, width=max(event.width, 900))

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

    def _load_preview(self, task):
        path = self._preview_path(task)
        if path is None:
            return None
        try:
            with Image.open(path) as source:
                source.seek(0)
                image = source.convert("RGB")
            image = image.resize(self.IMAGE_SIZE, Image.Resampling.LANCZOS)
            return ImageTk.PhotoImage(image)
        except Exception:
            return None

    def _add_task_card(self, index, label, task, script_name):
        card = tk.Frame(
            self.page,
            bg=CARD_BG,
            highlightbackground=CARD_BORDER,
            highlightthickness=2,
        )
        card.pack(fill="x", padx=14, pady=16)

        card_header = tk.Frame(card, bg=CARD_BG)
        card_header.pack(fill="x", padx=20, pady=(16, 10))
        tk.Label(
            card_header,
            text=f"{index + 1:02d}",
            bg=PLAY_BLUE,
            fg="white",
            font=("Sans", 17, "bold"),
            padx=13,
            pady=6,
        ).pack(side="left", padx=(0, 14))
        title = tk.Label(
            card_header,
            text=label,
            bg=CARD_BG,
            fg=TEXT_PRIMARY,
            anchor="w",
            font=("Sans", 27, "bold"),
        )
        title.pack(side="left", fill="x", expand=True)
        tk.Label(
            card_header,
            text="ROBOT SCENARIO",
            bg=CARD_BG,
            fg="#7fb6dc",
            font=("Sans", 12, "bold"),
        ).pack(side="right")

        preview_holder = tk.Frame(card, bg="#080a0d")
        preview_holder.pack(padx=20, pady=(0, 20))
        photo = self._load_preview(task)
        self.preview_photos.append(photo)
        image_label = tk.Label(
            preview_holder,
            image=photo,
            text="No preview available" if photo is None else "",
            bg="#080a0d",
            fg="#aab2bd",
            width=self.IMAGE_SIZE[0],
            height=self.IMAGE_SIZE[1],
        )
        image_label.pack()

        # The action is deliberately over the lower edge of the screenshot,
        # so every task has its own obvious launch control in front of it.
        button = RoundedButton(
            preview_holder,
            text="Play",
            command=lambda i=index: self.play_or_stop(i),
            bg=PLAY_BLUE,
            activebackground=PLAY_BLUE_ACTIVE,
            font=("Sans", 20, "bold"),
            width=270,
            height=106,
            radius=38,
        )
        button.place(relx=0.5, rely=0.98, anchor="s")
        self.task_buttons.append(button)

    def play_or_stop(self, index):
        if self.child is not None:
            if self.active_index == index:
                self._stop_task("Task stopped. Select another task when ready.")
            else:
                self.status.configure(text="Stop the running task before starting another.", fg="#e6a15c")
            return
        self._start_task(index)

    def _start_task(self, index):
        label, _task, script_name = TASKS[index]
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
        command = [
            sys.executable,
            str(script),
            "--seed",
            str(seed),
            "--control",
            self.control.get(),
        ]
        try:
            self.child = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
        except Exception as exc:
            messagebox.showerror("Could not start task", str(exc))
            self.child = None
            return
        self.active_index = index
        for i, button in enumerate(self.task_buttons):
            button.configure(state="normal" if i == index else "disabled")
        self.control.configure(state="disabled")
        self.seed_entry.configure(state="disabled")
        self.task_buttons[index].configure(text="Stop", bg="#b06a20", activebackground="#d0842b")
        self.status.configure(
            text=f"Running {label} with seed {seed}. Close its viewer or press Stop to return.",
            fg="#70d6a2",
        )

    def _poll_child(self):
        if self.child is not None:
            code = self.child.poll()
            if code is not None:
                self.child = None
                index = self.active_index
                self.active_index = None
                self._reset_task_buttons()
                if code == 0:
                    self.status.configure(text="Task result: SUCCESS. Select another task below.", fg="#70d6a2")
                elif code == 10:
                    self.status.configure(text="Task result: FAILURE. Select another task below.", fg="#e6a15c")
                elif code == 2:
                    self.status.configure(text="Task closed before a result was reached.", fg=TEXT_SECONDARY)
                else:
                    self.status.configure(text=f"Task result: ERROR (exit status {code}). Check the terminal.", fg="#e6a15c")
        self.after(250, self._poll_child)

    def _reset_task_buttons(self):
        self.control.configure(state="readonly")
        self.seed_entry.configure(state="normal")
        for button in self.task_buttons:
            button.configure(state="normal", text="Play", bg=PLAY_BLUE, activebackground=PLAY_BLUE_ACTIVE)

    def _stop_task(self, status=None):
        child = self.child
        if child is None:
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
            self._reset_task_buttons()
        if status:
            self.status.configure(text=status, fg="#e6a15c")

    def exit_app(self):
        self._stop_task()
        self.destroy()


def main():
    HouseholdTaskLauncher().mainloop()


if __name__ == "__main__":
    main()
