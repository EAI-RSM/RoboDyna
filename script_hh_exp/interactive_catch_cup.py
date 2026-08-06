#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive household task: catch_cup."""
try:
    from ._interactive_common import make_parser, run_task
except ImportError:
    from _interactive_common import make_parser, run_task

KEYBOARD = """
  Space             optional god-mode hold of the pillow
  Arrow keys        teleport held pillow in XY
  E / Q             teleport held pillow in Z

  Cup starts moving as soon as the viewer is ready.
  Pillow is PhysX-dynamic (pushable).
"""
ROBOT = """
  Space             close/open gripper for pushing

  Cup starts moving as soon as the viewer is ready.
  Pillow is PhysX-dynamic — shove it with the closed gripper (no teleport).
"""

if __name__ == "__main__":
    a = make_parser("catch_cup", __doc__)
    result = run_task("catch_cup", a.parse_args(), KEYBOARD, ROBOT)
    raise SystemExit(10 if result is False else 0 if result is True else 2)
