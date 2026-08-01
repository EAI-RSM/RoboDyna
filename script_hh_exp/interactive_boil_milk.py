#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive household task: boil_milk."""
try:
    from ._interactive_common import make_parser, run_task
except ImportError:
    from _interactive_common import make_parser, run_task

KEYBOARD = """\n  F: toggle the cooktop knob (milk rises while on)\n  V: top-down/head camera | Escape: quit\n"""
ROBOT = """\n  1/2/3: select arm(s) | arrows/Q/E: Cartesian arm teleoperation\n  F: toggle the cooktop knob | V: top-down/head camera | Escape: quit\n"""

if __name__ == "__main__":
    a = make_parser("boil_milk", __doc__)
    result = run_task("boil_milk", a.parse_args(), KEYBOARD, ROBOT)
    raise SystemExit(10 if result is False else 0 if result is True else 2)
