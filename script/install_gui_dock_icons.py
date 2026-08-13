#!/usr/bin/env python3
"""Install RoboDyna GUI icons for the Ubuntu dock (GNOME .desktop + hicolor).

Tk ``iconphoto`` alone does not change the Ubuntu taskbar icon. GNOME matches a
running window's WM_CLASS to a ``.desktop`` ``StartupWMClass`` and uses that
entry's ``Icon=``.

Run once (or rely on the GUIs, which install on launch)::

    python script/install_gui_dock_icons.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "interactive"))

from _task_briefing import GUI_WM_CLASS, install_ubuntu_dock_icon  # noqa: E402


def main() -> int:
    py = sys.executable
    paths = [
        install_ubuntu_dock_icon(
            desktop_id="robodyna-interactive-tasks",
            name="RoboDyna Base Tasks",
            comment="Dynamic interactive task launcher",
            script_path=ROOT / "interactive" / "base_task_gui.py",
            wm_class=GUI_WM_CLASS["interactive"],
            python_exe=py,
        ),
        install_ubuntu_dock_icon(
            desktop_id="robodyna-household-tasks",
            name="RoboDyna Household Tasks",
            comment="Household interactive task launcher",
            script_path=ROOT / "interactive" / "household_task_gui.py",
            wm_class=GUI_WM_CLASS["household"],
            python_exe=py,
        ),
        install_ubuntu_dock_icon(
            desktop_id="robodyna-experiment",
            name="RoboDyna Human Experiment",
            comment="Human-experiment task launcher",
            script_path=ROOT / "interactive" / "experiment_gui.py",
            wm_class=GUI_WM_CLASS["experiment"],
            python_exe=py,
        ),
    ]
    for p in paths:
        print(f"installed {p}")
    print("Restart the GUI (or log out/in if the dock still shows a generic icon).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
