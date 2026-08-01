#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive household task: fill_coffee_jar."""
try:
    from ._interactive_common import make_parser, run_task
except ImportError:
    from _interactive_common import make_parser, run_task

KEYBOARD = """\n  Space/F: press the coffee dispenser once\n  Repeat until the beans reach the requested red fill line\n  V: top-down/head camera | Escape: quit\n"""
ROBOT = """\n  1/2/3: select arm(s) | arrows/Q/E: Cartesian arm teleoperation\n  F: press the dispenser | V: top-down/head camera | Escape: quit\n"""

if __name__ == "__main__":
    a = make_parser("fill_coffee_jar", __doc__)
    result = run_task("fill_coffee_jar", a.parse_args(), KEYBOARD, ROBOT)
    raise SystemExit(10 if result is False else 0 if result is True else 2)
