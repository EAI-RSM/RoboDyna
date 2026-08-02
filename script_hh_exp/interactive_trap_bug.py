#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive household task: trap_bug."""
try:
    from ._interactive_common import make_parser, run_task
except ImportError:  # direct ``python script_hh_exp/...py`` execution
    from _interactive_common import make_parser, run_task

KEYBOARD = """\n  Bug starts running as soon as the viewer is ready\n  Space: hold/release the transparent trap\n  Arrows: move held trap in XY | Q/E: move it in Z\n  V: top-down/head camera | Escape: quit\n"""
ROBOT = """\n  Bug starts running as soon as the viewer is ready\n  Space: grasp/release trap\n  V: top-down/head camera | Escape: quit\n"""

if __name__ == "__main__":
    a = make_parser("trap_bug", __doc__)
    args = a.parse_args()
    # Interactive SAPIEN viewer: simple alpha box (no glass transmission/IOR).
    if not any(str(x).startswith("plain_trap=") for x in (args.task_arg or [])):
        args.task_arg.append("plain_trap=true")
    result = run_task("trap_bug", args, KEYBOARD, ROBOT)
    raise SystemExit(10 if result is False else 0 if result is True else 2)
