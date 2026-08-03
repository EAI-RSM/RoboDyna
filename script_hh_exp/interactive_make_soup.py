#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive household task: make_soup."""
try:
    from ._interactive_common import make_parser, run_task
except ImportError:
    from _interactive_common import make_parser, run_task

KEYBOARD = """\n  Space: hold/release vegetable board | arrows/Q/E: move it\n  F once: turn on burner | F while holding: move board over pot\n  Hold Z: tip left | Hold X: tip right (releases vegetables when steep)\n  V: top-down/head camera | Escape: quit\n"""
ROBOT = """\n  Space: grasp/release board | F: turn burner on (then carry board over pot)\n  Hold Z / X: tip gripper left / right to pour vegetables into the pot\n  V: top-down/head camera | Escape: quit\n"""

if __name__ == "__main__":
    a = make_parser("make_soup", __doc__)
    result = run_task("make_soup", a.parse_args(), KEYBOARD, ROBOT)
    raise SystemExit(10 if result is False else 0 if result is True else 2)
