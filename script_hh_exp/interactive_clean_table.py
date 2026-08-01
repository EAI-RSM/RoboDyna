#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive household task: clean_table."""
try:
    from ._interactive_common import make_parser, run_task
except ImportError:
    from _interactive_common import make_parser, run_task

KEYBOARD = """\n  F once: tip mug and start spill\n  Space: hold/release sponge | arrows/Q/E: move it\n  F while over a spot: dab/clean it (repeat for contact dwell)\n  V: top-down/head camera | Escape: quit\n"""
ROBOT = """\n  1/2/3: select arm(s) | arrows/Q/E: Cartesian arm teleoperation\n  Space: grasp/release sponge | F: dab the next dirty spot\n  V: top-down/head camera | Escape: quit\n"""

if __name__ == "__main__":
    a = make_parser("clean_table", __doc__)
    result = run_task("clean_table", a.parse_args(), KEYBOARD, ROBOT)
    raise SystemExit(10 if result is False else 0 if result is True else 2)
