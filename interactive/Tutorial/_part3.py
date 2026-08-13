"""Part 3 coach: grasp a cube, hold a button, toggle a switch, push a box."""
from __future__ import annotations

from _interactive_common import print_instructions, run_viewer_loop, task_result_exit_code
from _key_hud import TutorialKeyHud, build_part3_key_hud
from _keycaps import draw_action_stage

# (stage, window keys to highlight, prompt)
_STAGES: tuple[tuple[str, tuple[tuple[str, str], ...], str], ...] = (
    (
        "grasp",
        (("space", "Space"), ("e", "E")),
        "Pick up the orange cube: move onto it, close the gripper (Space), then lift (E).",
    ),
    (
        "hold",
        (("q", "Q"), ("e", "E")),
        "Hold-to-press: lower onto the green button with Q, hold, then lift off with E.",
    ),
    (
        "switch",
        (("q", "Q"),),
        "On/off switch: press Q to turn it ON (red, stays down), then press again to turn it OFF.",
    ),
    (
        "push",
        (
            ("space", "Space"),
            ("left", "left"),
            ("right", "right"),
            ("up", "up"),
            ("down", "down"),
        ),
        "Close the gripper (Space) and push the blue box onto the green line.",
    ),
)


class Part3Coach:
    def __init__(self, env, hud: TutorialKeyHud):
        self.env = env
        self.hud = hud
        self.stage_index = 0
        self.pressed: set[str] = set()
        self._prev: dict[str, bool] = {}
        self.done = False
        self._armed = False
        for _stage, keys, _prompt in _STAGES:
            for window_key, _label in keys:
                self._prev[window_key] = False

    def _edge(self, window, key: str) -> bool:
        down = bool(window.key_down(key))
        edge = down and not self._prev.get(key, False)
        self._prev[key] = down
        return edge

    def _ensure_left_arm(self) -> None:
        if self._armed:
            return
        controls = getattr(self.env, "_interactive_robot_controls", None)
        if controls is None:
            return
        try:
            if not controls.selected:
                controls.selected = ("left",)
                self.env._interactive_selected_arms = ("left",)
                controls._highlight_selected()
            self._armed = True
        except Exception:
            pass

    def start(self) -> None:
        stage, _keys, prompt = _STAGES[0]
        self.env.tutorial_set_stage(stage)
        self.hud.set_stage(stage)
        print_instructions(prompt)

    def _advance(self) -> None:
        finished = _STAGES[self.stage_index][0]
        print_instructions(f"{finished} — done.")
        self.stage_index += 1
        self.pressed = set()
        if self.stage_index >= len(_STAGES):
            self.done = True
            self.env.tutorial_set_stage(None)
            self.env._tutorial_complete = True
            print_instructions("Basic actions — part 3 complete.")
            return
        stage, _keys, prompt = _STAGES[self.stage_index]
        self.env.tutorial_set_stage(stage)
        self.hud.set_stage(stage)
        print_instructions(prompt)

    def update(self, window) -> None:
        if self.done or window is None:
            return
        self._ensure_left_arm()
        if self.env.tutorial_stage_complete():
            self._advance()
            return
        stage, keys, _prompt = _STAGES[self.stage_index]
        changed = False
        for window_key, label in keys:
            if not self._edge(window, window_key):
                continue
            if label in self.pressed:
                continue
            self.pressed.add(label)
            changed = True
        if changed:
            self.hud.update_texture(stage, draw_action_stage(stage, self.pressed))

    def is_done(self, _step: int):
        if self.done:
            return True, "basic actions"
        return False


def run_part3(env) -> int:
    env._tutorial_complete = False
    hud = build_part3_key_hud(env.scene)
    coach = Part3Coach(env, hud)
    coach.start()
    hud.on_frame = coach.update
    run_viewer_loop(
        env,
        on_step=lambda window, _step: coach.update(window),
        is_done=coach.is_done,
        extra_plugins=[hud],
    )
    return task_result_exit_code()
