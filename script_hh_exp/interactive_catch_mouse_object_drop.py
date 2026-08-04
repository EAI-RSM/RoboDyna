#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive household task: mouse_object_drop."""
try:
    from ._interactive_common import make_parser, run_task
except ImportError:
    from _interactive_common import make_parser, run_task

KEYBOARD = """\n  Mouse/object motion starts as soon as the viewer is ready\n  Space: hold/release catcher basket | arrows/Q/E: move it\n  V: top-down/head camera | Escape: quit\n"""
ROBOT = """\n  Mouse/object motion starts as soon as the viewer is ready\n  Space: grasp/release basket\n  V: top-down/head camera | Escape: quit\n"""

if __name__ == "__main__":
    a = make_parser("mouse_object_drop", __doc__)
    result = run_task("mouse_object_drop", a.parse_args(), KEYBOARD, ROBOT)
    raise SystemExit(10 if result is False else 0 if result is True else 2)
