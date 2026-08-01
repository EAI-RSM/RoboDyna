#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive household task: mouse_object_drop."""
try:
    from ._interactive_common import make_parser, run_task
except ImportError:
    from _interactive_common import make_parser, run_task

KEYBOARD = """\n  Space: hold/release catcher basket | arrows/Q/E: move it\n  F: release/activate the moving mouse | V: top-down/head camera\n  Escape: quit\n"""
ROBOT = """\n  1/2/3: select arm(s) | arrows/Q/E: Cartesian arm teleoperation\n  Space: grasp/release basket | F: activate mouse\n  V: top-down/head camera | Escape: quit\n"""

if __name__ == "__main__":
    a = make_parser("mouse_object_drop", __doc__)
    result = run_task("mouse_object_drop", a.parse_args(), KEYBOARD, ROBOT)
    raise SystemExit(10 if result is False else 0 if result is True else 2)
