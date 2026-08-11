#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive household task: boil_milk."""
try:
    from ._interactive_common import make_parser, run_task
except ImportError:
    from _interactive_common import make_parser, run_task

KEYBOARD = """
  Prefer --control robot: Space only opens/closes the gripper.
  Close on the cooktop knob and twist with teleop to control the stove.
"""
ROBOT = """
  Space opens/closes the gripper only — no automatic knob twist.
  Close on the cooktop knob and twist with teleop to turn the stove on/off.
"""

if __name__ == "__main__":
    a = make_parser("boil_milk", __doc__)
    result = run_task("boil_milk", a.parse_args(), KEYBOARD, ROBOT)
    raise SystemExit(10 if result is False else 0 if result is True else 2)
