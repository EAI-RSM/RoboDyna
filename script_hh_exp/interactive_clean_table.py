#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive household task: clean_table."""
try:
    from ._interactive_common import make_parser, run_task
except ImportError:
    from _interactive_common import make_parser, run_task

KEYBOARD = """
  Space             pick up / release sponge
  Arrow keys        move held sponge in XY
  E / Q             move held sponge in Z

  Mug tips and spill starts as soon as the viewer is ready.
  Press the yellow pad onto a stain to wipe it (must touch the table).
"""
ROBOT = """
  Space             pick up / release sponge

  Mug tips and spill starts as soon as the viewer is ready.
  Grippers start open — select the sponge-side arm (2 if mug is right).
  Approach the small top cube open, then pinch; lower the yellow pad onto a stain to wipe.
"""

if __name__ == "__main__":
    a = make_parser("clean_table", __doc__)
    result = run_task("clean_table", a.parse_args(), KEYBOARD, ROBOT)
    raise SystemExit(10 if result is False else 0 if result is True else 2)
