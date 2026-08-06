#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive household task: stop_ball."""
try:
    from ._interactive_common import make_parser, run_task
except ImportError:
    from _interactive_common import make_parser, run_task

KEYBOARD = """
  Ball starts rolling as soon as the viewer is ready. Block it with the gripper before it falls off the table.
"""
ROBOT = """
  Ball starts rolling as soon as the viewer is ready. Block it with the gripper before it falls off the table.
"""

if __name__ == "__main__":
    a = make_parser("stop_ball", __doc__)
    result = run_task("stop_ball", a.parse_args(), KEYBOARD, ROBOT)
    raise SystemExit(10 if result is False else 0 if result is True else 2)
