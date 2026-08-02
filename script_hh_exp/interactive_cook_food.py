#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive household task: cook_food."""
try:
    from ._interactive_common import make_parser, run_task
except ImportError:
    from _interactive_common import make_parser, run_task

KEYBOARD = """\n  Space: hold/release food | arrows/Q/E: move it\n  F: toggle burner knob | V: top-down/head camera | Escape: quit\n"""
ROBOT = """\n  Space: grasp/release food | F: toggle burner knob\n  V: top-down/head camera | Escape: quit\n"""

if __name__ == "__main__":
    a = make_parser("cook_food", __doc__)
    result = run_task("cook_food", a.parse_args(), KEYBOARD, ROBOT)
    raise SystemExit(10 if result is False else 0 if result is True else 2)
