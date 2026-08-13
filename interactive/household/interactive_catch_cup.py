#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive household task: catch_cup."""
try:
    from ._interactive_common import make_parser, run_task
except ImportError:
    from _interactive_common import make_parser, run_task

KEYBOARD = """
  Prefer --control robot: Space only opens/closes the gripper.
  Cup starts moving as soon as the viewer is ready.
  Pillow is PhysX-dynamic (pushable with the closed gripper).
"""
ROBOT = """
  Space             open / close selected gripper(s) only

  Cup starts moving as soon as the viewer is ready.
  Pillow is PhysX-dynamic — shove it with the closed gripper (no teleport).
"""

if __name__ == "__main__":
    a = make_parser("catch_cup", __doc__)
    result = run_task("catch_cup", a.parse_args(), KEYBOARD, ROBOT)
    raise SystemExit(10 if result is False else 0 if result is True else 2)
