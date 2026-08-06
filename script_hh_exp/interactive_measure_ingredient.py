#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive household task: measure_ingredient."""
try:
    from ._interactive_common import make_parser, print_instructions, run_task
except ImportError:
    from _interactive_common import make_parser, print_instructions, run_task

KEYBOARD = """
  Space             hold/release jar
  Arrow keys        move held jar in XY
  E / Q             move held jar in Z

  Lower the gripper onto the red key to latch oil ON/OFF.
"""
ROBOT = """
  Space             side-grasp / release jar

  Lower onto the red key with Q to latch oil ON/OFF (press again to turn off).
  Releasing the jar ends the episode; success requires correct fill on the scale.
"""


def _post_setup(env):
    # Allow physical key presses for the whole interactive session.
    env._ignore_tab = False
    env._interactive_universal_controls = True
    # Viewer treats a solid transmission cylinder as a filled volume — swap to
    # a hollow glass shell so the rising oil level is visible from outside.
    env.use_viewer_hollow_jar()
    # Expert pour_rate is tuned for fast sim idle-steps; interactive runs one
    # step per viewer frame, so bump rate so the jar visibly fills (~8–12s).
    env.pour_rate = max(float(getattr(env, "pour_rate", 0.0)), 0.00085)
    print_instructions(
        f"[measure_ingredient] interactive pour_rate={env.pour_rate:.6g} "
        f"(Z-press red key; Space grasps jar)"
    )


if __name__ == "__main__":
    a = make_parser("measure_ingredient", __doc__)
    args = a.parse_args()
    # Keep real transmission glass on the jar (do NOT force plain_glass).
    # Caller may still pass plain_glass=true for other materials if needed.
    result = run_task(
        "measure_ingredient", args, KEYBOARD, ROBOT, post_setup=_post_setup
    )
    raise SystemExit(10 if result is False else 0 if result is True else 2)
