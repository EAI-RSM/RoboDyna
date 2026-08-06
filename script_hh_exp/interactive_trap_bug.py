#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive household task: trap_bug."""
try:
    from ._interactive_common import make_parser, run_task
except ImportError:  # direct ``python script_hh_exp/...py`` execution
    from _interactive_common import make_parser, run_task

KEYBOARD = """
  Space             hold/release the transparent trap
  Arrow keys        move held trap in XY
  E / Q             move held trap in Z

  Bug starts running as soon as the viewer is ready.
"""
ROBOT = """
  Space             grasp/release trap

  Bug starts running as soon as the viewer is ready.
"""

if __name__ == "__main__":
    a = make_parser("trap_bug", __doc__)
    args = a.parse_args()
    # Interactive SAPIEN viewer: simple alpha box (no glass transmission/IOR).
    if not any(str(x).startswith("plain_trap=") for x in (args.task_arg or [])):
        args.task_arg.append("plain_trap=true")
    result = run_task("trap_bug", args, KEYBOARD, ROBOT)
    raise SystemExit(10 if result is False else 0 if result is True else 2)
