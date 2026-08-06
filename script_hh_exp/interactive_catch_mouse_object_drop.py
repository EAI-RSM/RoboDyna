#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive household task: catch_mouse_object_drop."""
try:
    from ._interactive_common import make_parser, run_task
except ImportError:
    from _interactive_common import make_parser, run_task

KEYBOARD = """
  Space             hold/release catcher basket
  Arrow keys        move held basket in XY
  E / Q             move held basket in Z

  Mouse/object motion starts as soon as the viewer is ready.
"""
ROBOT = """
  Space             grasp/release basket

  Mouse/object motion starts as soon as the viewer is ready.
"""

if __name__ == "__main__":
    a = make_parser("catch_mouse_object_drop", __doc__)
    result = run_task("catch_mouse_object_drop", a.parse_args(), KEYBOARD, ROBOT)
    raise SystemExit(10 if result is False else 0 if result is True else 2)
