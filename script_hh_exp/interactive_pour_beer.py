#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive household task: pour_beer."""
try:
    from ._interactive_common import make_parser, run_task
except ImportError:
    from _interactive_common import make_parser, run_task

KEYBOARD = """\n  Space/F: toggle the beer tap lever open/closed\n  Watch the liquid and foam levels; close before overflow\n  V: top-down/head camera | Escape: quit\n"""
ROBOT = """\n  1/2/3: select arm(s) | arrows/Q/E: Cartesian arm teleoperation\n  Space/F: grasp, open, then close/release the tap lever\n  V: top-down/head camera | Escape: quit\n"""

if __name__ == "__main__":
    a = make_parser("pour_beer", __doc__)
    result = run_task("pour_beer", a.parse_args(), KEYBOARD, ROBOT)
    raise SystemExit(10 if result is False else 0 if result is True else 2)
