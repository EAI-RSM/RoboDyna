#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive household task: cook_food_timer."""
try:
    from ._interactive_common import make_parser, run_task
except ImportError:
    from _interactive_common import make_parser, run_task

KEYBOARD = """
  Space             hold/release food
  Arrow keys        move held food in XY
  E / Q             move held food in Z

  Turn the stove by closing on the cooktop knob and twisting; the pie timer follows the stove.
"""
ROBOT = """
  Space             grasp/release food
  C                 gripper grasp-and-twist cooktop knob

  Timer runs while the stove is on; or close on the knob and twist with teleop.
"""

if __name__ == "__main__":
    a = make_parser("cook_food_timer", __doc__)
    result = run_task("cook_food_timer", a.parse_args(), KEYBOARD, ROBOT)
    raise SystemExit(10 if result is False else 0 if result is True else 2)
