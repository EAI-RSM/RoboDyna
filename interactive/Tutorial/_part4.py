"""Part 4 coach: catch a rolling ball, stove knob, mallet, then a force key."""
from __future__ import annotations

from _interactive_common import print_instructions, run_viewer_loop, task_result_exit_code
from _key_hud import TutorialKeyHud, build_part4_key_hud

_MOVE_KEYS = (
    ("q", "Q"),
    ("e", "E"),
    ("left", "left"),
    ("right", "right"),
    ("up", "up"),
    ("down", "down"),
)

# (stage, window keys to highlight, prompt)
_STAGES: tuple[tuple[str, tuple[tuple[str, str], ...], str], ...] = (
    (
        "ball",
        (
            ("space", "Space"),
            *_MOVE_KEYS,
        ),
        "Catch the rolling ball: move onto it, close (Space), lift (E). "
        "Touching it is fine — it only respawns if it falls off the table.",
    ),
    (
        "stove",
        (("space", "Space"), ("r", "R"), ("t", "T")),
        "Stove knob: close on the knob (Space), yaw left to light the fire, "
        "yaw back to turn it off.",
    ),
    (
        "mallet",
        (("space", "Space"), ("e", "E")),
        "Pick up the mallet: close on the handle (Space), then lift (E).",
    ),
    (
        "force_key",
        (("q", "Q"), ("e", "E")),
        "Multi-stage key: press Q. The bar is live force. Hit the yellow "
        "target band, then release with E. Too light or too hard — try again.",
    ),
)


class Part4Coach:
    def __init__(self, env, hud: TutorialKeyHud):
        self.env = env
        self.hud = hud
        self.stage_index = 0
        self._prev: dict[str, bool] = {}
        self.done = False
        self._armed = False
        self._last_feedback = ""
        # After ball success, wait for release before spawning the stove.
        self._pending_advance = False
        self._release_prompted = False
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
        self._last_feedback = ""
        self.hud.flash_kwargs = {}
        self._pending_advance = False
        self._release_prompted = False
        if self.stage_index >= len(_STAGES):
            self.done = True
            self.env._tutorial_complete = True
            print_instructions("Advanced actions — part 4 complete.")
            return
        stage, _keys, prompt = _STAGES[self.stage_index]
        self.env.tutorial_set_stage(stage)
        self.hud.set_stage(stage)
        print_instructions(prompt)

    def _try_finish_pending(self) -> None:
        """Park the held ball, then spawn the next stage once contact is gone."""
        if not self._pending_advance:
            return
        released = True
        try:
            released = bool(self.env.tutorial_ball_released())
        except Exception:
            released = True
        if not released:
            # Keep asking the jaws to open every tick until contact clears.
            try:
                self.env.tutorial_force_open_grippers()
            except Exception:
                pass
            if not self._release_prompted:
                self._release_prompted = True
                print_instructions(
                    "Ball caught — open the gripper (Space) so the next object can appear."
                )
            return
        self._advance()

    def _sync_force_kwargs(self) -> None:
        if _STAGES[self.stage_index][0] != "force_key":
            self.hud.flash_kwargs = {}
            return
        force = float(getattr(self.env, "_key_force_n", 0.0) or 0.0)
        peak = float(getattr(self.env, "_press_peak_force", 0.0) or 0.0)
        target = int(getattr(self.env, "_force_target_level", 2) or 2)
        feedback = str(getattr(self.env, "_force_feedback", "") or "")
        if feedback and feedback != self._last_feedback:
            self._last_feedback = feedback
            print_instructions(feedback)
        self.hud.flash_kwargs = {
            "force_n": force,
            "peak_n": peak,
            "target_level": target,
            "feedback": feedback,
        }
        # Force a redraw so the bar moves even if no key edge fired.
        self.hud._play_drawn = None

    def update_keys(self, window) -> None:
        """HUD flashes only — never spawn/despawn (runs during render)."""
        if self.done or window is None:
            return
        self._ensure_left_arm()
        _stage, keys, _prompt = _STAGES[self.stage_index]
        held: set[str] = set()
        for window_key, label in keys:
            if window.key_down(window_key):
                held.add(label)
            if self._edge(window, window_key):
                self.hud.flash(label)
        self.hud.set_held(held)
        self._sync_force_kwargs()

    def update_stage(self, window, _step: int) -> None:
        """Advance props on the physics tick, not during viewer.render()."""
        if self.done:
            return
        self._ensure_left_arm()
        if self._pending_advance:
            self._try_finish_pending()
            return
        if not self.env.tutorial_stage_complete():
            return
        finished = _STAGES[self.stage_index][0]
        # Ball → stove: wait for release so we never teleport a held body or
        # unhide the cooktop into an active grasp contact.
        if finished == "ball":
            self._pending_advance = True
            self._try_finish_pending()
            return
        self._advance()

    def is_done(self, _step: int):
        if self.done:
            return True, "advanced actions"
        return False


def run_part4(env) -> int:
    env._tutorial_complete = False
    hud = build_part4_key_hud(env.scene)
    coach = Part4Coach(env, hud)
    coach.start()
    hud.on_frame = coach.update_keys
    run_viewer_loop(
        env,
        on_step=coach.update_stage,
        is_done=coach.is_done,
        extra_plugins=[hud],
    )
    return task_result_exit_code()
