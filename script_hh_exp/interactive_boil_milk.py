#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive household task: boil_milk."""
try:
    from ._interactive_common import make_parser, run_task
except ImportError:
    from _interactive_common import make_parser, run_task

KEYBOARD = """
  Space             toggle the cooktop knob
"""
ROBOT = """
  Space             selected arm turns the knob
"""

if __name__ == "__main__":
    a = make_parser("boil_milk", __doc__)
    result = run_task("boil_milk", a.parse_args(), KEYBOARD, ROBOT)
    raise SystemExit(10 if result is False else 0 if result is True else 2)
