#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive household task: measure_ingredient."""
try:
    from ._interactive_common import make_parser, run_task
except ImportError:
    from _interactive_common import make_parser, run_task

KEYBOARD = """\n  Space: hold/release jar | arrows/Q/E: move it\n  F: toggle dispenser switch | V: top-down/head camera | Escape: quit\n"""
ROBOT = """\n  1/2/3: select arm(s) | arrows/Q/E: Cartesian arm teleoperation\n  Space: grasp/release jar | F: toggle dispenser switch\n  V: top-down/head camera | Escape: quit\n"""

if __name__ == "__main__":
    a = make_parser("measure_ingredient", __doc__)
    result = run_task("measure_ingredient", a.parse_args(), KEYBOARD, ROBOT)
    raise SystemExit(10 if result is False else 0 if result is True else 2)
