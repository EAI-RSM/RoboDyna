#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive household task: make_soup."""
try:
    from ._interactive_common import make_parser, run_task
except ImportError:
    from _interactive_common import make_parser, run_task

KEYBOARD = """
  Space             hold/release vegetable board
  Arrow keys        move held board in XY
  E / Q             move held board in Z
  C                 turn on burner; while holding, move board over pot
  Z / X             tip board left / right

  Hold Z/X until vegetables release into the pot.
"""
ROBOT = """
  Space             grasp/release board
  C                 turn burner on (then carry board over pot)
  Z / X             tip gripper left / right to pour
"""

if __name__ == "__main__":
    a = make_parser("make_soup", __doc__)
    result = run_task("make_soup", a.parse_args(), KEYBOARD, ROBOT)
    raise SystemExit(10 if result is False else 0 if result is True else 2)
