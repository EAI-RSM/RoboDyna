"""Part 2 coach: arrows, E/Q height, R/T yaw, F/G tilt, then Space gripper."""
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
        "Now E raises the arm and Q lowers it.",
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


class Part2Coach:
    def __init__(self, env, hud: TutorialKeyHud):
        self.env = env
        self.hud = hud
        self.stage_index = 0
        self.pressed: set[str] = set()
        self._prev: dict[str, bool] = {}
        self.space_toggles = 0
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

    def _advance(self) -> None:
        self.stage_index += 1
        self.pressed = set()
        if self.stage_index >= len(_STAGES):
            self.done = True
            self.env._tutorial_complete = True
            print_instructions("Base controls — part 2 complete.")
            return
        stage, _keys, prompt = _STAGES[self.stage_index]
        self.hud.set_stage(stage)
        print_instructions(prompt)

    def update(self, window) -> None:
        if self.done or window is None:
            return
        self._ensure_left_arm()
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

    def is_done(self, _step: int):
        if self.done:
            return True, "base teleop controls"
        return False


def run_part2(env) -> int:
    env._tutorial_complete = False
    hud = build_part2_key_hud(env.scene)
    coach = Part2Coach(env, hud)
    hud.on_frame = coach.update
    print_instructions(_STAGES[0][2])
    run_viewer_loop(
        env,
        on_step=lambda window, _step: coach.update(window),
        is_done=coach.is_done,
        extra_plugins=[hud],
    )
    return task_result_exit_code()
