#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive household task: clean_table."""
try:
    from ._interactive_common import make_parser, run_task
except ImportError:
    from _interactive_common import make_parser, run_task

KEYBOARD = """\n  Mug tips and spill starts as soon as the viewer is ready\n  Space: hold/release sponge | arrows/Q/E: move it\n  Press the yellow pad onto a stain to wipe it (must touch the table)\n  F: force-clear stains currently under the sponge\n  V: top-down/head camera | Escape: quit\n"""
ROBOT = """\n  Mug tips and spill starts as soon as the viewer is ready\n  2 (or 1): select the sponge-side arm first\n  Space: grasp/release the small handle on the sponge\n  Lower the yellow pad onto a stain to wipe it (must touch the table)\n  F: dab the next dirty spot\n  V: top-down/head camera | Escape: quit\n"""

if __name__ == "__main__":
    a = make_parser("clean_table", __doc__)
    result = run_task("clean_table", a.parse_args(), KEYBOARD, ROBOT)
    raise SystemExit(10 if result is False else 0 if result is True else 2)
