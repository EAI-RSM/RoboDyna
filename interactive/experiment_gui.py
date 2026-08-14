#!/usr/bin/env python3
"""Human-experiment launcher: login, experience survey, then base / household GUIs."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

_INTERACTIVE_DIR = Path(__file__).resolve().parent
if str(_INTERACTIVE_DIR) not in sys.path:
    sys.path.insert(0, str(_INTERACTIVE_DIR))

from _task_briefing import (  # noqa: E402
    GUI_INK,
    GUI_MUTED,
    GUI_PAGE_BG,
    apply_gui_logo,
    setup_gui_app_icon,
)
from base_task_gui import (  # noqa: E402
    PLAY_BLUE,
    PLAY_BLUE_ACTIVE,
    RoundedButton,
    SCENARIOS,
    TASKS as BASE_TASKS,
)
from experiment_config import CONFIG_ENV, load_experiment_config  # noqa: E402
from experiment_logs import (  # noqa: E402
    EXPERIENCE_QUESTIONS,
    EXPERIMENT_ENV,
    EXPERIMENT_LOG_ENV,
    EXPERIMENT_USER_ENV,
    child_experiment_env,
    create_user,
    ensure_controller_log,
    find_user,
    load_user_log,
    log_controller_tag,
    progress_counts,
    slugify_user_name,
)
from household_task_gui import TASKS as HOUSEHOLD_TASKS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PAGE_BG = GUI_PAGE_BG
HEADER_BG = GUI_PAGE_BG
CARD_BG = "#202c38"
CARD_BORDER = "#405367"
TEXT_PRIMARY = "#f4f8fb"
TEXT_SECONDARY = "#aebdca"

CONTROLLER_DISPLAY = ("Robot", "Keyboard + mouse")
CONTROLLER_MODE = {
    "Robot": "robot",
    "Keyboard + mouse": "keyboard+mouse",
}
MODE_DISPLAY = {mode: label for label, mode in CONTROLLER_MODE.items()}
SCENARIO_REMAINING_LABELS = (
    ("default", "Default"),
    ("opt1", "Opt 1"),
    ("opt2", "Opt 2"),
    ("opt1+2", "Opt 1+2"),
)


class ExperimentLauncher(tk.Tk):
    DESIGN_SIZE = (920, 780)
    UI_SCALE_MIN = 0.7
    UI_SCALE_MAX = 1.15

    def __init__(self):
        super().__init__(className="Robodynaexperiment")
        self.title("RoboDyna Human Experiment")
        self.geometry("920x780")
        self.minsize(720, 620)
        self.configure(bg=PAGE_BG)
        setup_gui_app_icon(
            self,
            suite="experiment",
            script_path=Path(__file__),
        )
        self.protocol("WM_DELETE_WINDOW", self.exit_app)

        self.child: subprocess.Popen | None = None
        self.user_data: dict | None = None
        self._ui_scale = 1.0
        self._ui_scale_job: str | None = None
        self._screen = "name"

        self.logo_bar = tk.Frame(self, bg=PAGE_BG)
        self.logo_bar.pack(fill="x", padx=24, pady=(16, 0))
        self.logo_label = tk.Label(self.logo_bar, bg=PAGE_BG, bd=0, highlightthickness=0)
        self.logo_label.pack(anchor="center")
        apply_gui_logo(self.logo_label, height=140)

        self.body = tk.Frame(self, bg=PAGE_BG)
        self.body.pack(fill="both", expand=True, padx=28, pady=(8, 20))

        self._build_name_screen()
        self._build_experience_screen()
        self._build_suite_screen()
        self._show_screen("name")

        self.bind("<Configure>", self._on_root_configure)
        self.after(0, self._apply_ui_scale)
        self.after(250, self._poll_child)

    def _yes_no_choice(self, parent, var: tk.StringVar, value: str, text: str) -> tk.Frame:
        """Clickable Yes/No control with a large custom radio dot (Tk's is tiny on Linux)."""
        wrap = tk.Frame(parent, bg=CARD_BG, cursor="hand2")
        size = 28
        canvas = tk.Canvas(
            wrap,
            width=size,
            height=size,
            bg=CARD_BG,
            highlightthickness=0,
            bd=0,
            cursor="hand2",
        )
        canvas.pack(side="left")
        label = tk.Label(
            wrap,
            text=text,
            bg=CARD_BG,
            fg=TEXT_PRIMARY,
            activebackground=CARD_BG,
            activeforeground=TEXT_PRIMARY,
            font=("Sans", 16, "bold"),
            cursor="hand2",
        )
        label.pack(side="left", padx=(10, 0))

        def paint(*_args):
            canvas.delete("all")
            outer = int(wrap._dot_size)
            canvas.configure(width=outer, height=outer)
            pad = max(2, outer // 12)
            ring = max(2, outer // 12)
            canvas.create_oval(
                pad,
                pad,
                outer - pad,
                outer - pad,
                outline="#d7e4ef",
                width=ring,
            )
            if var.get() == value:
                inset = max(pad + ring + 1, outer // 4)
                canvas.create_oval(
                    inset,
                    inset,
                    outer - inset,
                    outer - inset,
                    fill=PLAY_BLUE,
                    outline=PLAY_BLUE,
                )

        def select(_event=None):
            var.set(value)

        wrap._dot_size = size
        wrap._dot_canvas = canvas
        wrap._dot_label = label
        wrap._dot_paint = paint
        for widget in (wrap, canvas, label):
            widget.bind("<Button-1>", select)
        var.trace_add("write", paint)
        paint()
        return wrap

    def _card(self, parent) -> tk.Frame:
        card = tk.Frame(
            parent,
            bg=CARD_BG,
            highlightbackground=CARD_BORDER,
            highlightthickness=2,
        )
        return card

    def _build_name_screen(self):
        self.name_screen = tk.Frame(self.body, bg=PAGE_BG)
        header = tk.Frame(self.name_screen, bg=HEADER_BG, highlightbackground="#d4d5db", highlightthickness=1)
        header.pack(fill="x", pady=(0, 16))
        inner = tk.Frame(header, bg=HEADER_BG)
        inner.pack(fill="x", padx=24, pady=18)
        self.name_title = tk.Label(
            inner,
            text="Human Experiment",
            bg=HEADER_BG,
            fg=GUI_INK,
            anchor="w",
            font=("Sans", 30, "bold"),
        )
        self.name_title.pack(anchor="w")
        self.name_subtitle = tk.Label(
            inner,
            text="Enter your name to start or continue. Returning participants skip the survey.",
            bg=HEADER_BG,
            fg=GUI_MUTED,
            anchor="w",
            wraplength=780,
            justify="left",
            font=("Sans", 14),
        )
        self.name_subtitle.pack(anchor="w", pady=(4, 0))

        card = self._card(self.name_screen)
        card.pack(fill="x")
        pad = tk.Frame(card, bg=CARD_BG)
        pad.pack(fill="x", padx=28, pady=28)
        self.name_caption = tk.Label(
            pad,
            text="Your name",
            bg=CARD_BG,
            fg=TEXT_SECONDARY,
            anchor="w",
            font=("Sans", 13, "bold"),
        )
        self.name_caption.pack(anchor="w")
        self.name_entry = tk.Entry(
            pad,
            font=("Sans", 18, "bold"),
            bg="#f7fafc",
            fg="#182633",
            insertbackground="#182633",
            relief="flat",
        )
        self.name_entry.pack(fill="x", ipady=10, pady=(8, 18))
        self.name_entry.bind("<Return>", lambda _e: self._submit_name())
        self.name_continue = RoundedButton(
            pad,
            text="Continue",
            command=self._submit_name,
            bg=PLAY_BLUE,
            activebackground=PLAY_BLUE_ACTIVE,
            font=("Sans", 16, "bold"),
            width=220,
            height=64,
            radius=26,
        )
        self.name_continue.pack(anchor="w")
        self.name_status = tk.Label(
            pad,
            text="",
            bg=CARD_BG,
            fg="#7fb6dc",
            anchor="w",
            justify="left",
            wraplength=760,
            font=("Sans", 13),
        )
        self.name_status.pack(anchor="w", pady=(16, 0))

    def _build_experience_screen(self):
        self.exp_screen = tk.Frame(self.body, bg=PAGE_BG)
        header = tk.Frame(self.exp_screen, bg=HEADER_BG, highlightbackground="#d4d5db", highlightthickness=1)
        header.pack(fill="x", pady=(0, 16))
        inner = tk.Frame(header, bg=HEADER_BG)
        inner.pack(fill="x", padx=24, pady=18)
        self.exp_title = tk.Label(
            inner,
            text="A few questions",
            bg=HEADER_BG,
            fg=GUI_INK,
            anchor="w",
            font=("Sans", 30, "bold"),
        )
        self.exp_title.pack(anchor="w")
        self.exp_subtitle = tk.Label(
            inner,
            text="Asked once for new participants. Answers are stored in your experiment log.",
            bg=HEADER_BG,
            fg=GUI_MUTED,
            anchor="w",
            wraplength=780,
            justify="left",
            font=("Sans", 14),
        )
        self.exp_subtitle.pack(anchor="w", pady=(4, 0))

        card = self._card(self.exp_screen)
        card.pack(fill="both", expand=True)
        pad = tk.Frame(card, bg=CARD_BG)
        pad.pack(fill="both", expand=True, padx=28, pady=24)
        self.exp_vars: dict[str, tk.StringVar] = {}
        self.exp_question_labels: list[tk.Label] = []
        self.exp_choice_rows: list[tk.Frame] = []
        for key, question in EXPERIENCE_QUESTIONS:
            var = tk.StringVar(value="")
            self.exp_vars[key] = var
            q_label = tk.Label(
                pad,
                text=question,
                bg=CARD_BG,
                fg=TEXT_PRIMARY,
                anchor="w",
                justify="left",
                wraplength=760,
                font=("Sans", 15, "bold"),
            )
            q_label.pack(anchor="w", pady=(0, 8))
            self.exp_question_labels.append(q_label)
            row = tk.Frame(pad, bg=CARD_BG)
            row.pack(anchor="w", pady=(0, 18))
            for value, text in (("yes", "Yes"), ("no", "No")):
                choice = self._yes_no_choice(row, var, value, text)
                choice.pack(side="left", padx=(0, 22))
                self.exp_choice_rows.append(choice)
        self.exp_continue = RoundedButton(
            pad,
            text="Continue",
            command=self._submit_experience,
            bg=PLAY_BLUE,
            activebackground=PLAY_BLUE_ACTIVE,
            font=("Sans", 16, "bold"),
            width=220,
            height=64,
            radius=26,
        )
        self.exp_continue.pack(anchor="w", pady=(8, 0))

    def _build_suite_screen(self):
        self.suite_screen = tk.Frame(self.body, bg=PAGE_BG)
        header = tk.Frame(self.suite_screen, bg=HEADER_BG, highlightbackground="#d4d5db", highlightthickness=1)
        header.pack(fill="x", pady=(0, 16))
        top = tk.Frame(header, bg=HEADER_BG)
        top.pack(fill="x")
        self.suite_exit = RoundedButton(
            top,
            text="Exit",
            command=self.exit_app,
            bg="#e34a33",
            activebackground="#eb6854",
            font=("Sans", 16, "bold"),
            width=140,
            height=64,
            radius=26,
        )
        self.suite_exit.pack(side="right", padx=(8, 18), pady=14)
        inner = tk.Frame(top, bg=HEADER_BG)
        inner.pack(side="left", fill="x", expand=True, padx=24, pady=18)
        self.suite_title = tk.Label(
            inner,
            text="Choose a suite",
            bg=HEADER_BG,
            fg=GUI_INK,
            anchor="w",
            font=("Sans", 30, "bold"),
        )
        self.suite_title.pack(anchor="w")
        self.suite_subtitle = tk.Label(
            inner,
            text="",
            bg=HEADER_BG,
            fg=GUI_MUTED,
            anchor="w",
            wraplength=620,
            justify="left",
            font=("Sans", 14),
        )
        self.suite_subtitle.pack(anchor="w", pady=(4, 0))
        control_row = tk.Frame(inner, bg=HEADER_BG)
        control_row.pack(anchor="w", fill="x", pady=(12, 0))
        self.controller_caption = tk.Label(
            control_row,
            text="Controller",
            bg=HEADER_BG,
            fg=GUI_MUTED,
            anchor="w",
            font=("Sans", 13, "bold"),
        )
        self.controller_caption.pack(side="left")
        self.controller_combo = ttk.Combobox(
            control_row,
            values=CONTROLLER_DISPLAY,
            state="readonly",
            width=22,
            font=("Sans", 13, "bold"),
        )
        self.controller_combo.set("Robot")
        self.controller_combo.pack(side="left", padx=(10, 0))
        self.controller_combo.bind("<<ComboboxSelected>>", self._on_controller_change)

        self.remaining_stats = tk.Label(
            self.suite_screen,
            text="",
            bg=PAGE_BG,
            fg=GUI_INK,
            anchor="w",
            justify="left",
            wraplength=820,
            font=("Sans", 13, "bold"),
        )
        self.remaining_stats.pack(fill="x", pady=(0, 12))

        grid = tk.Frame(self.suite_screen, bg=PAGE_BG)
        grid.pack(fill="both", expand=True)
        self.base_card = self._suite_card(
            grid,
            title="Base Tasks",
            blurb="Dynamic table tasks with Default / Opt 1 / Opt 2 / Opt 1+2.",
            command=lambda: self._launch_suite("base"),
        )
        self.base_card.pack(fill="x", pady=(0, 12))
        self.household_card = self._suite_card(
            grid,
            title="Household Tasks",
            blurb="Kitchen / office tasks with per-episode randomization.",
            command=lambda: self._launch_suite("household"),
        )
        self.household_card.pack(fill="x")
        self.switch_user = tk.Button(
            self.suite_screen,
            text="Switch user",
            command=self._switch_user,
            bg=PAGE_BG,
            fg=GUI_MUTED,
            activebackground=PAGE_BG,
            activeforeground=GUI_INK,
            bd=0,
            highlightthickness=0,
            font=("Sans", 12, "underline"),
            cursor="hand2",
        )
        self.switch_user.pack(anchor="w", pady=(14, 0))

    def _suite_card(self, parent, *, title: str, blurb: str, command) -> tk.Frame:
        card = self._card(parent)
        pad = tk.Frame(card, bg=CARD_BG)
        pad.pack(fill="x", padx=24, pady=20)
        title_label = tk.Label(
            pad,
            text=title,
            bg=CARD_BG,
            fg=TEXT_PRIMARY,
            anchor="w",
            font=("Sans", 22, "bold"),
        )
        title_label.pack(anchor="w")
        blurb_label = tk.Label(
            pad,
            text=blurb,
            bg=CARD_BG,
            fg=TEXT_SECONDARY,
            anchor="w",
            wraplength=780,
            justify="left",
            font=("Sans", 13),
        )
        blurb_label.pack(anchor="w", pady=(4, 10))
        progress = tk.Label(
            pad,
            text="",
            bg=CARD_BG,
            fg="#7fb6dc",
            anchor="w",
            font=("Sans", 13, "bold"),
        )
        progress.pack(anchor="w", pady=(0, 12))
        button = RoundedButton(
            pad,
            text="Open",
            command=command,
            bg=PLAY_BLUE,
            activebackground=PLAY_BLUE_ACTIVE,
            font=("Sans", 16, "bold"),
            width=180,
            height=60,
            radius=26,
        )
        button.pack(anchor="w")
        card._title = title_label
        card._blurb = blurb_label
        card._progress = progress
        card._button = button
        return card

    def _show_screen(self, name: str):
        self._screen = name
        for screen in (self.name_screen, self.exp_screen, self.suite_screen):
            screen.pack_forget()
        if name == "name":
            self.name_screen.pack(fill="both", expand=True)
            self.after(50, lambda: self.name_entry.focus_set())
        elif name == "experience":
            self.exp_screen.pack(fill="both", expand=True)
        else:
            self._refresh_suite_progress()
            self.suite_screen.pack(fill="both", expand=True)

    def _submit_name(self):
        name = self.name_entry.get().strip()
        if not name:
            self.name_status.configure(text="Please enter your name.", fg="#e6a15c")
            self.name_entry.focus_set()
            return
        if not slugify_user_name(name):
            self.name_status.configure(text="Please use letters or numbers in your name.", fg="#e6a15c")
            return
        existing = find_user(name)
        if existing:
            self._activate_user(existing)
            self._show_screen("suite")
            return
        self._pending_name = name
        self.name_status.configure(text="")
        for var in self.exp_vars.values():
            var.set("")
        self._show_screen("experience")

    def _submit_experience(self):
        answers = {key: var.get().strip() for key, var in self.exp_vars.items()}
        missing = [label for key, label in EXPERIENCE_QUESTIONS if answers.get(key) not in ("yes", "no")]
        if missing:
            messagebox.showinfo(
                "Please answer every question",
                "Select Yes or No for each experience question before continuing.",
            )
            return
        name = getattr(self, "_pending_name", "").strip()
        if not name:
            self._show_screen("name")
            return
        data = create_user(name, answers)
        self._activate_user(data)
        self._show_screen("suite")

    def _selected_control_mode(self) -> str:
        label = str(self.controller_combo.get() or "Robot").strip()
        return CONTROLLER_MODE.get(label, "robot")

    def _session_controller(self) -> str:
        return log_controller_tag(self._selected_control_mode())

    def _set_controller_combo(self, mode_or_tag: str | None):
        tag = log_controller_tag(mode_or_tag or "robot")
        mode = "keyboard+mouse" if tag == "keyboard" else "robot"
        self.controller_combo.set(MODE_DISPLAY.get(mode, "Robot"))

    def _on_controller_change(self, _event=None):
        if self.child is not None:
            messagebox.showinfo(
                "Task still running",
                "Close the task window before switching controllers.",
            )
            tag = str((self.user_data or {}).get("controller") or "robot")
            self._set_controller_combo(tag)
            return
        if self.user_data is None:
            return
        self._bind_session_log(self.user_data)
        self._refresh_suite_progress()

    def _bind_session_log(self, data: dict | None = None) -> dict:
        """Point this session at ``user_robot.json`` or ``user_keyboard.json``."""
        source = data if data is not None else self.user_data or {}
        display = str(source.get("user_name") or source.get("user_id") or "user")
        slug = str(source.get("user_id") or slugify_user_name(display))
        mode = self._selected_control_mode()
        tag = log_controller_tag(mode)
        log = ensure_controller_log(
            slug,
            display=display,
            experience=source.get("experience"),
            controller=tag,
            template=source,
        )
        os.environ[EXPERIMENT_ENV] = "1"
        os.environ[EXPERIMENT_USER_ENV] = display
        os.environ["ROBODYNA_CONTROL"] = mode
        os.environ[EXPERIMENT_LOG_ENV] = str(log.get("log_path") or "")
        self.user_data = log
        return log

    def _activate_user(self, data: dict):
        self._set_controller_combo(str((data or {}).get("controller") or "robot"))
        log = self._bind_session_log(data)
        display = str(log.get("user_name") or log.get("user_id") or "user")
        tag = str(log.get("controller") or self._session_controller())
        control_label = MODE_DISPLAY.get(
            "keyboard+mouse" if tag == "keyboard" else "robot",
            "Robot",
        )
        self.suite_title.configure(text=f"Welcome back, {display}" if log.get("plays") else f"Hello, {display}")
        self.suite_subtitle.configure(
            text=(
                f"{control_label} session — progress is stored in user_{tag}.json. "
                "Finished items stay gray after you close this window."
            )
        )

    def _protocol_task_names(self) -> tuple[list[str], list[str]]:
        cfg = load_experiment_config()
        base_idx = cfg.visible_indices("base", len(BASE_TASKS))
        household_idx = cfg.visible_indices("household", len(HOUSEHOLD_TASKS))
        return (
            [BASE_TASKS[i][1] for i in base_idx],
            [HOUSEHOLD_TASKS[i][1] for i in household_idx],
        )

    def _remaining_summary(self, data: dict | None = None, counts: dict | None = None) -> str:
        log = data if data is not None else load_user_log()
        cfg = load_experiment_config()
        if counts is None:
            base_names, household_names = self._protocol_task_names()
            counts = progress_counts(
                log,
                base_task_names=base_names,
                household_task_names=household_names,
                n_scenarios=len(SCENARIOS),
                cfg=cfg,
            )
        parts: list[str] = []
        all_done = True
        any_slot = False
        scenario_done = counts.get("scenario_done") or {}
        scenario_total = counts.get("scenario_total") or {}
        for key, label in SCENARIO_REMAINING_LABELS:
            total = int(scenario_total.get(key, 0) or 0)
            done = int(scenario_done.get(key, 0) or 0)
            if total <= 0:
                continue
            any_slot = True
            left = max(0, total - done)
            if left:
                all_done = False
            parts.append(f"{label} {left} left ({done}/{total})")
        household_total = int(counts.get("household_total") or 0)
        household_done = int(counts.get("household_done") or 0)
        if household_total > 0:
            any_slot = True
            left = max(0, household_total - household_done)
            if left:
                all_done = False
            parts.append(f"Household {left} left ({household_done}/{household_total})")
        if not any_slot:
            return "No limited scenarios in this protocol."
        if all_done:
            return "All protocol scenarios are complete for this controller."
        return "  ·  ".join(parts)

    def _refresh_suite_progress(self):
        if self.user_data:
            self._bind_session_log(self.user_data)
        log = load_user_log()
        if log:
            self.user_data = log
        cfg = load_experiment_config()
        base_names, household_names = self._protocol_task_names()
        counts = progress_counts(
            log,
            base_task_names=base_names,
            household_task_names=household_names,
            n_scenarios=len(SCENARIOS),
            cfg=cfg,
        )
        base_total = counts["base_total"]
        household_total = counts["household_total"]
        self.base_card._blurb.configure(
            text=(
                f"{len(base_names)} dynamic table tasks, each with "
                "Default / Opt 1 / Opt 2 / Opt 1+2."
            )
        )
        self.household_card._blurb.configure(
            text=f"{len(household_names)} kitchen / office tasks with per-episode randomization."
        )
        base_left = max(0, int(base_total) - int(counts["base_done"]))
        household_left = max(0, int(household_total) - int(counts["household_done"]))
        self.base_card._progress.configure(
            text=(
                f"{counts['base_done']} / {base_total} scenarios completed"
                + (f"  ·  {base_left} left" if base_total else "")
            )
        )
        self.household_card._progress.configure(
            text=(
                f"{counts['household_done']} / {household_total} tasks completed"
                + (f"  ·  {household_left} left" if household_total else "")
            )
        )
        remaining = self._remaining_summary(log, counts=counts)
        self.remaining_stats.configure(text=remaining)
        display = ""
        if self.user_data:
            display = str(self.user_data.get("user_name") or "")
        if display:
            plays = bool((log or {}).get("plays"))
            tag = str((log or {}).get("controller") or self._session_controller())
            control_label = MODE_DISPLAY.get(
                "keyboard+mouse" if tag == "keyboard" else "robot",
                "Robot",
            )
            self.suite_title.configure(text=f"Welcome back, {display}" if plays else f"Hello, {display}")
            self.suite_subtitle.configure(
                text=(
                    f"{control_label} session — progress is stored in user_{tag}.json. "
                    "Finished items stay gray after you close this window."
                )
            )
        if counts["base_done"] >= base_total > 0:
            self.base_card._button.configure(text="Review")
        else:
            self.base_card._button.configure(text="Open")
        if counts["household_done"] >= household_total > 0:
            self.household_card._button.configure(text="Review")
        else:
            self.household_card._button.configure(text="Open")

    def _launch_suite(self, suite: str):
        if self.child is not None:
            messagebox.showinfo("Already open", "Close the task window before opening another suite.")
            return
        if self.user_data is None:
            self._show_screen("name")
            return
        self._bind_session_log(self.user_data)
        script = ROOT / "interactive" / (
            "base_task_gui.py" if suite == "base" else "household_task_gui.py"
        )
        if not script.exists():
            messagebox.showerror("Unavailable", f"Missing launcher:\n{script}")
            return
        child_env = child_experiment_env()
        cfg_path = os.environ.get(CONFIG_ENV, "").strip()
        if cfg_path:
            child_env[CONFIG_ENV] = cfg_path
        try:
            self.child = subprocess.Popen(
                [sys.executable, str(script)],
                cwd=ROOT,
                start_new_session=True,
                env=child_env,
            )
        except Exception as exc:
            messagebox.showerror("Could not start suite", str(exc))
            self.child = None
            return
        self.withdraw()

    def _poll_child(self):
        if self.child is not None:
            code = self.child.poll()
            if code is not None:
                self.child = None
                try:
                    self.deiconify()
                    self.lift()
                    self.focus_force()
                except tk.TclError:
                    pass
                self._refresh_suite_progress()
        self.after(250, self._poll_child)

    def _switch_user(self):
        if self.child is not None:
            messagebox.showinfo("Task still running", "Close the task window before switching users.")
            return
        self.user_data = None
        os.environ.pop(EXPERIMENT_USER_ENV, None)
        os.environ.pop(EXPERIMENT_LOG_ENV, None)
        os.environ.pop("ROBODYNA_CONTROL", None)
        self.name_entry.delete(0, "end")
        self.name_status.configure(text="")
        self._show_screen("name")

    def _on_root_configure(self, event):
        if event.widget is not self:
            return
        if self._ui_scale_job is not None:
            self.after_cancel(self._ui_scale_job)
        self._ui_scale_job = self.after(80, self._apply_ui_scale)

    def _apply_ui_scale(self):
        self._ui_scale_job = None
        width = max(self.winfo_width(), 1)
        height = max(self.winfo_height(), 1)
        scale = min(width / self.DESIGN_SIZE[0], height / self.DESIGN_SIZE[1])
        scale = max(self.UI_SCALE_MIN, min(self.UI_SCALE_MAX, scale))
        if abs(scale - self._ui_scale) < 0.02:
            wrap = max(360, width - 80)
            self.name_subtitle.configure(wraplength=wrap)
            self.exp_subtitle.configure(wraplength=wrap)
            self.suite_subtitle.configure(wraplength=max(280, wrap - 160))
            self.remaining_stats.configure(wraplength=max(360, width - 80))
            return
        self._ui_scale = scale

        def font(size, weight=""):
            px = max(8, int(round(size * scale)))
            return ("Sans", px, weight) if weight else ("Sans", px)

        def px(value):
            return max(1, int(round(value * scale)))

        apply_gui_logo(self.logo_label, height=px(140))
        self.name_title.configure(font=font(30, "bold"))
        self.name_subtitle.configure(font=font(14), wraplength=max(360, width - 80))
        self.name_caption.configure(font=font(13, "bold"))
        self.name_entry.configure(font=font(18, "bold"))
        self.name_continue.configure(font=font(16, "bold"), width=px(220), height=px(64), radius=px(26))
        self.name_status.configure(font=font(13), wraplength=max(360, width - 100))
        self.exp_title.configure(font=font(30, "bold"))
        self.exp_subtitle.configure(font=font(14), wraplength=max(360, width - 80))
        for label in self.exp_question_labels:
            label.configure(font=font(15, "bold"), wraplength=max(360, width - 100))
        for choice in self.exp_choice_rows:
            choice._dot_size = px(28)
            choice._dot_label.configure(font=font(16, "bold"))
            choice._dot_paint()
        self.exp_continue.configure(font=font(16, "bold"), width=px(220), height=px(64), radius=px(26))
        self.suite_title.configure(font=font(30, "bold"))
        self.suite_subtitle.configure(font=font(14), wraplength=max(280, width - 240))
        self.controller_caption.configure(font=font(13, "bold"))
        self.controller_combo.configure(font=font(13, "bold"))
        self.remaining_stats.configure(font=font(13, "bold"), wraplength=max(360, width - 80))
        self.suite_exit.configure(font=font(16, "bold"), width=px(140), height=px(64), radius=px(26))
        for card in (self.base_card, self.household_card):
            card._title.configure(font=font(22, "bold"))
            card._blurb.configure(font=font(13), wraplength=max(360, width - 100))
            card._progress.configure(font=font(13, "bold"))
            card._button.configure(font=font(16, "bold"), width=px(180), height=px(60), radius=px(26))
        self.switch_user.configure(font=font(12, "underline"))

    def _stop_child(self):
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

    def exit_app(self):
        self._stop_child()
        self.destroy()


def main():
    ExperimentLauncher().mainloop()


if __name__ == "__main__":
    main()
