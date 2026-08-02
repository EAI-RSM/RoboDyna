#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive household task: fill_coffee_jar."""
try:
    from ._interactive_common import make_parser, run_task
except ImportError:
    from _interactive_common import make_parser, run_task

KEYBOARD = """\n  Use robot control for this task: select an arm, move above the blue key, then lower in Z to press\n  V: top-down/head camera | Escape: quit\n"""
ROBOT = """\n  Select one arm, move it above the blue key, then lower with Q to press and raise with E to release\n  Coffee dispenses only from measured key pressure; Space is unused for this task\n  V: top-down/head camera | Escape: quit\n"""

if __name__ == "__main__":
    a = make_parser("fill_coffee_jar", __doc__)
    args = a.parse_args()
    # Interactive SAPIEN viewer: plain alpha glass (no transmission/IOR).
    if not any(str(x).startswith("plain_glass=") for x in (args.task_arg or [])):
        args.task_arg.append("plain_glass=true")
    result = run_task("fill_coffee_jar", args, KEYBOARD, ROBOT)
    raise SystemExit(10 if result is False else 0 if result is True else 2)
