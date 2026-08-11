#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive household task: cook_food_timer."""
try:
    from ._interactive_common import make_parser, run_task
except ImportError:
    from _interactive_common import make_parser, run_task

KEYBOARD = """
  Prefer --control robot: Space only opens/closes the gripper.
  Grasp food and shut the stove off by twisting the knob; the pie timer follows the stove.
"""
ROBOT = """
  Space opens/closes the gripper only — no automatic food grasp.
  Grasp food with teleop, then twist the cooktop knob to shut off; timer runs while cooking.
"""

if __name__ == "__main__":
    a = make_parser("cook_food_timer", __doc__)
    result = run_task("cook_food_timer", a.parse_args(), KEYBOARD, ROBOT)
    raise SystemExit(10 if result is False else 0 if result is True else 2)
