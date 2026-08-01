#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive household task: trap_bug."""
try:
    from ._interactive_common import make_parser, run_task
except ImportError:  # direct ``python script_hh_exp/...py`` execution
    from _interactive_common import make_parser, run_task

KEYBOARD = """\n  Space: hold/release the glass trap\n  Arrows: move held trap in XY | Q/E: move it in Z\n  F: start the running bug | V: top-down/head camera | Escape: quit\n"""
ROBOT = """\n  1/2/3: select arm(s) | arrows/Q/E: Cartesian arm teleoperation\n  Space: grasp/release trap | F: start the running bug\n  V: top-down/head camera | Escape: quit\n"""

if __name__ == "__main__":
    a = make_parser("trap_bug", __doc__)
    args = a.parse_args()
    # Interactive SAPIEN viewer: simple alpha box (no glass transmission/IOR).
    if not any(str(x).startswith("plain_trap=") for x in (args.task_arg or [])):
        args.task_arg.append("plain_trap=true")
    result = run_task("trap_bug", args, KEYBOARD, ROBOT)
    raise SystemExit(10 if result is False else 0 if result is True else 2)
