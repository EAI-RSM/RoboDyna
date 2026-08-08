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
SCRIPT_DIR = Path(__file__).resolve().parent
DEMO_DIR = ROOT / "final_task_demos"
README_PATH = ROOT / "README.md"
# Must match script_exp._interactive_common.TASK_RESULT_ENV
TASK_RESULT_ENV = "ROBODYNA_TASK_RESULT_FILE"

# Shared briefing dialog lives next to the dynamic interactive GUI helpers.
_SCRIPT_EXP = ROOT / "script_exp"
if str(_SCRIPT_EXP) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_EXP))
from _task_briefing import build_briefing_text, show_task_briefing  # noqa: E402

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
PAGE_BG = "#111820"
HEADER_BG = "#1b2733"
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
        if options:
            super().configure(**options)
        self._redraw()

    config = configure


class HouseholdTaskLauncher(tk.Tk):
    # Head-camera stills are ~4:3; width is the card max, height follows aspect.
    IMAGE_SIZE = (1280, 960)

    def __init__(self):
        super().__init__()
        self.title("Household Interactive Tasks")
        self.geometry("1560x980")
        self.minsize(980, 700)
        self.configure(bg=PAGE_BG)
        self.protocol("WM_DELETE_WINDOW", self.exit_app)

        self.child: subprocess.Popen | None = None
        self.active_index: int | None = None
        self.result_file: Path | None = None
        self.preview_photos: list[ImageTk.PhotoImage | None] = []
        self.task_buttons: list[RoundedButton] = []
        self._idle_status = f"{len(TASKS)} scenarios available  |  Select a task and press Play."
        self._idle_status_fg = TEXT_SECONDARY

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

        brief_group = tk.Frame(controls, bg=HEADER_BG)
        brief_group.pack(side="left", padx=(0, 16))
        tk.Label(
            brief_group,
            text="Briefing",
            bg=HEADER_BG,
            fg=TEXT_SECONDARY,
            font=("Sans", 13, "bold"),
        ).pack(anchor="w")
        self.show_briefing = tk.BooleanVar(value=True)
        self.briefing_check = tk.Checkbutton(
            brief_group,
            text="Show before start",
            variable=self.show_briefing,
            onvalue=True,
            offvalue=False,
            bg=HEADER_BG,
            fg=TEXT_PRIMARY,
            activebackground=HEADER_BG,
            activeforeground=TEXT_PRIMARY,
            selectcolor="#182633",
            highlightthickness=0,
            font=("Sans", 14, "bold"),
            cursor="hand2",
        )
        self.briefing_check.pack(anchor="w", pady=(6, 0))

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
            text=self._idle_status,
            bg=PAGE_BG,
            fg=self._idle_status_fg,
            anchor="w",
            justify="left",
            wraplength=1460,
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
            max_w, max_h = self.IMAGE_SIZE
            aspect = image.width / max(image.height, 1)
            width = max_w
            height = max(1, int(round(width / aspect)))
            if height > max_h:
                height = max_h
                width = max(1, int(round(height * aspect)))
            image = image.resize((width, height), Image.Resampling.LANCZOS)
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
            cursor="hand2",
        )
        if photo is None:
            image_label.configure(width=self.IMAGE_SIZE[0], height=self.IMAGE_SIZE[1])
        image_label.pack()
        image_label.bind(
            "<Enter>",
            lambda _e, t=task, l=label: self._show_task_hint(t, l),
        )
        image_label.bind("<Leave>", lambda _e: self._clear_task_hint())

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
            on_enter=lambda t=task, l=label: self._show_task_hint(t, l),
            on_leave=self._clear_task_hint,
        )
        button.place(relx=0.5, rely=0.98, anchor="s")
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
