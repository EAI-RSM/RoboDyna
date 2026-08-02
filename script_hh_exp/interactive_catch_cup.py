#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive household task: catch_cup."""
try:
    from ._interactive_common import make_parser, run_task
except ImportError:
    from _interactive_common import make_parser, run_task

KEYBOARD = """\n  Cup starts moving as soon as the viewer is ready\n  Space: hold/release landing pillow | arrows/Q/E: move it\n  V: top-down/head camera | Escape: quit\n"""
ROBOT = """\n  Cup starts moving as soon as the viewer is ready\n  Space: grasp/release pillow\n  V: top-down/head camera | Escape: quit\n"""

if __name__ == "__main__":
    a = make_parser("catch_cup", __doc__)
    result = run_task("catch_cup", a.parse_args(), KEYBOARD, ROBOT)
    raise SystemExit(10 if result is False else 0 if result is True else 2)
