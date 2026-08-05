#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive household task: pour_beer."""
try:
    from ._interactive_common import make_parser, print_instructions, run_task
except ImportError:
    from _interactive_common import make_parser, print_instructions, run_task

KEYBOARD = """\n  Open the tap with robot arm teleop (--control robot); no keyboard lever shortcut\n  Stream thickness and fill rate follow how far the handle is turned\n  V: top-down/head camera | G: gripper view | F: open/close gripper | Escape: quit\n"""
ROBOT = """\n  Push the tap lever with the gripper (spring returns when released)\n  Hold the handle open — fill rate + stream thickness scale with how far it turns\n  Arm teleop: arrows/Q/E; use arm 2 (right) for the tap\n  V: top-down/head camera | G: gripper view | F: open/close gripper | Escape: quit\n"""


def _post_setup(env):
    # Viewer treats a solid transmission cylinder as a filled volume — swap to
    # a hollow glass shell so the rising beer level is visible from outside
    # (same path as interactive_measure_ingredient → use_viewer_hollow_jar).
    env.use_viewer_hollow_mug()
    # Expert flow is tuned for fast sim idle-steps; interactive runs one step
    # per viewer frame, so bump rate so partial teleop pushes fill promptly.
    base = float(getattr(env, "FLOW_RATE_SCALE", 1.55))
    env.flow_rate_scale = max(float(getattr(env, "flow_rate_scale", base)), base * 1.35)
    print_instructions(
        f"[pour_beer] interactive flow_rate_scale={env.flow_rate_scale:.3g} "
        f"(hollow mug shell; open lever with arm teleop)"
    )


if __name__ == "__main__":
    a = make_parser("pour_beer", __doc__)
    args = a.parse_args()
    # Keep real transmission glass on the mug by default (do NOT force plain_glass).
    # Caller may still pass plain_glass=true for other materials if needed.
    result = run_task(
        "pour_beer", args, KEYBOARD, ROBOT, post_setup=_post_setup
    )
    raise SystemExit(10 if result is False else 0 if result is True else 2)
