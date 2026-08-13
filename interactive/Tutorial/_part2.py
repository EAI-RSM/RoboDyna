"""Part 2 coach: arrows, E/Q height, R/T yaw, F/G tilt, Space, then free practice."""
from __future__ import annotations

from _interactive_common import print_instructions, run_viewer_loop, task_result_exit_code
from _key_hud import TutorialKeyHud, build_part2_key_hud
from _keycaps import draw_control_stage

# (stage, window_key, overlay label) — finish every key in a stage before advancing.
_STAGES: tuple[tuple[str, tuple[tuple[str, str], ...], str], ...] = (
    (
        "arrows",
        (("left", "left"), ("right", "right"), ("up", "up"), ("down", "down")),
        "Arrow keys move the arm on the table. Try all four directions.",
    ),
    (
        "height",
        (("e", "E"), ("q", "Q")),
        "Now E raises the arm and Q lowers it. Z min and max is capped.",
    ),
    (
        "yaw",
        (("r", "R"), ("t", "T")),
        "R / T rotate the gripper left or right.",
    ),
    (
        "tilt",
        (("f", "F"), ("g", "G")),
        "F / G tilt the gripper left or right.",
    ),
    (
        "space",
        (("space", "Space"),),
        "Space opens and closes the gripper. Press it twice.",
    ),
)

_PLAY_KEYS = (
    ("left", "left"),
    ("right", "right"),
    ("up", "up"),
    ("down", "down"),
    ("e", "E"),
    ("q", "Q"),
    ("r", "R"),
    ("t", "T"),
    ("f", "F"),
    ("g", "G"),
    ("space", "Space"),
    ("escape", "Esc"),
)


class Part2Coach:
    def __init__(self, env, hud: TutorialKeyHud):
        self.env = env
        self.hud = hud
        self.stage_index = 0
        self.pressed: set[str] = set()
        self._prev: dict[str, bool] = {}
        self.space_toggles = 0
        self.play = False
        self._armed = False
        for _stage, keys, _prompt in _STAGES:
            for window_key, _label in keys:
                self._prev[window_key] = False
        for window_key, _label in _PLAY_KEYS:
            self._prev.setdefault(window_key, False)

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

    def _enter_play(self) -> None:
        self.play = True
        self.hud.set_stage("play")
        print_instructions(
            "Keep experimenting with the controls. Keys flash while you press them. "
            "Esc quits."
        )

    def _advance(self) -> None:
        self.stage_index += 1
        self.pressed = set()
        if self.stage_index >= len(_STAGES):
            self._enter_play()
            return
        stage, _keys, prompt = _STAGES[self.stage_index]
        self.hud.set_stage(stage)
        print_instructions(prompt)

    def _update_play(self, window) -> None:
        held = set()
        for sapien_key, label in _PLAY_KEYS:
            if window.key_down(sapien_key):
                held.add(label)
            if self._edge(window, sapien_key):
                self.hud.flash(label)
        self.hud.set_held(held)

    def update(self, window) -> None:
        if window is None:
            return
        self._ensure_left_arm()
        if self.play:
            self._update_play(window)
            return
        stage, keys, _prompt = _STAGES[self.stage_index]
        for window_key, label in keys:
            if not self._edge(window, window_key):
                continue
            if stage == "space":
                self.space_toggles += 1
                if self.space_toggles < 2:
                    print_instructions("Gripper toggled — press Space once more.")
                    continue
            elif label in self.pressed:
                continue
            self.pressed.add(label)
            remaining = [lbl for _wk, lbl in keys if lbl not in self.pressed]
            if remaining:
                print_instructions(
                    f"{label} — still need " + ", ".join(remaining)
                )
            self.hud.update_texture(stage, draw_control_stage(stage, self.pressed))
            if not remaining:
                self._advance()


def run_part2(env) -> int:
    env._tutorial_complete = False
    hud = build_part2_key_hud(env.scene)
    coach = Part2Coach(env, hud)
    hud.on_frame = coach.update
    print_instructions(_STAGES[0][2])
    run_viewer_loop(
        env,
        on_step=lambda window, _step: coach.update(window),
        is_done=None,
        extra_plugins=[hud],
    )
    return task_result_exit_code()
