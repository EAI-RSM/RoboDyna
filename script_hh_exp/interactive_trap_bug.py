#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Interactive household task: trap_bug."""
try:
    from ._interactive_common import make_parser, run_task
except ImportError:  # direct ``python script_hh_exp/...py`` execution
    from _interactive_common import make_parser, run_task

KEYBOARD = """
  Bug starts running as soon as the viewer is ready.
  Prefer --control robot: Space only opens/closes the gripper (no auto grasp).
"""
ROBOT = """
  Bug starts running as soon as the viewer is ready.
  Select the trap-side arm (1/2), put the fingers over the lid, Space to close —
  trap latches and lifts with the arm. Space again to open and drop.
"""

if __name__ == "__main__":
    a = make_parser("trap_bug", __doc__)
    args = a.parse_args()
    # Interactive SAPIEN viewer: simple alpha box (no glass transmission/IOR).
    if not any(str(x).startswith("plain_trap=") for x in (args.task_arg or [])):
        args.task_arg.append("plain_trap=true")

    def _preselect_trap_arm(env):
        # Trap always spawns on arm_side; highlight that gripper so Space works.
        side = str(getattr(env, "arm_side", "right") or "right")
        if side in ("left", "right"):
            env._interactive_selected_arms = (side,)
            print(f"[trap_bug] selected {side} arm (trap side) — Space closes/opens gripper")

    result = run_task("trap_bug", args, KEYBOARD, ROBOT, post_setup=_preselect_trap_arm)
    raise SystemExit(10 if result is False else 0 if result is True else 2)
