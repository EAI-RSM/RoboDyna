"""Per-task briefing copy for the interactive instruction dialog."""

from __future__ import annotations

# keyboard: list of (key_label, description). Use "mouse click" for click rows.
# notes: list of {"text": str, "red": list[str]}  — ``red`` phrases render in red.
# tips: extra advice (keyboard+mouse only unless a task sets robot_tips).

TASK_BRIEFING: dict[str, dict] = {
    "catch_marbles_trapdoors": {
        "instruction": (
            "Press the matching-color button to trap the target marble into the box."
        ),
        "success": "The target marble is in the box through the matching-color door.",
        "keyboard": [
            ("1 / 2 / 3 / 4", "open the corresponding trapdoor"),
            ("mouse click", "on the corresponding key"),
        ],
    },
    "catch_ramp_ball": {
        "instruction": (
            "Place the blue cup in front of the ramp to catch the red ball before hitting the table."
        ),
        "success": "The red ball is in the cup; a distractor (if present) is not.",
        "keyboard": [
            ("mouse click", "moves the cup to the corresponding location"),
        ],
        "notes": [
            {"text": "The cup can be placed only once.", "red": ["once"]},
        ],
    },
    "catch_cuboid": {
        "instruction": "Grasp and pull out cuboid(s) once it(they) emerge from the box.",
        "success": "Every cuboid is pulled out during its pop-up window.",
        "keyboard": [
            ("mouse click", "on a cuboid (once per cuboid) to pull it out"),
        ],
        "notes": [
            {
                "text": "If the cuboid is clicked while inside, the task fails.",
                "red": ["while inside"],
            },
        ],
    },
    "catch_shelf_marble": {
        "instruction": (
            "Move the bowl to left and right by pressing the corresponding keys to catch the red ball."
        ),
        "success": "The red ball is caught in the bowl.",
        "keyboard": [
            ("Arrows", "hold right/left arrows to move the bowl"),
            ("mouse click", "on buttons and hold to move the bowl right/left"),
        ],
    },
    "catch_valley_ball": {
        "instruction": (
            "Push the blue box in front of the ramp behind the red line to catch the red ball."
        ),
        "success": (
            "The red ball is in the box, the box is behind the red line, "
            "and a distractor (if present) is not in the box."
        ),
        "keyboard": [
            ("mouse click", "moves the box to the corresponding location"),
        ],
        "notes": [
            {"text": "The box can be placed only once.", "red": ["once"]},
        ],
    },
    "stop_valley_ball": {
        "instruction": (
            "Hold the ping-pong bat so the red ball hits its circular head before the table."
        ),
        "success": "The red ball contacts the bat head before hitting the table.",
        "keyboard": [
            ("mouse click", "moves the bat to the corresponding location"),
        ],
    },
    "cook_meat": {
        "instruction": (
            "Cook the steak on the pan to the target doneness and return it to the board."
        ),
        "success": "Each steak is cooked to the target doneness and placed back on the board.",
        "keyboard": [
            ("Space", "single-station cook key (press on / press again off; hold in Opt 1)"),
            ("Left / Right", "dual-station cook keys"),
        ],
        "tips": ["Meat starts in the pan."],
    },
    "cook_meat_timer": {
        "instruction": (
            "Cook the steak using the pie timer (green→yellow→red) and return it to the board."
        ),
        "success": (
            "Each steak is cooked to the target doneness (timer in the success band) "
            "and placed back on the board."
        ),
        "keyboard": [
            ("Space", "single-station cook key (press on / press again off; hold in Opt 1)"),
            ("Left / Right", "dual-station cook keys"),
        ],
        "tips": ["Meat starts in the pan. The pie timer tracks doneness."],
    },
    "put_cup_belt": {
        "instruction": "Place the cup in the slot between the yellow tools on the belt.",
        "success": "The cup sits between the tools without hitting a curtain (if present).",
        "keyboard": [
            ("mouse click", "on the belt to place the cup"),
        ],
    },
    "dispense_gummy": {
        "instruction": (
            "Dispense gummies and move the bowl so only the target-colored gummies are collected."
        ),
        "success": "The bowl holds target-colored gummies only.",
        "keyboard": [
            ("Space", "dispense"),
            ("Left / Right", "move the bowl left / right"),
            ("mouse click", "on the keys (hold to keep pressed)"),
        ],
    },
    "punch_dual_holes": {
        "instruction": "Punch every present tile on both belts.",
        "success": "Every present tile has been punched.",
        "keyboard": [
            ("Left / Right", "fire the left / right punch"),
            ("mouse click", "on a punch button to fire that side"),
        ],
    },
    "save_goal": {
        "instruction": (
            "Place the goalkeeper to block the ball from entering the goal."
        ),
        "success": "The keeper blocks the ball before it crosses the goal line.",
        "keyboard": [
            ("mouse click", "places the goalkeeper at that location"),
        ],
        "notes": [
            {"text": "The goalkeeper can be placed only once.", "red": ["once"]},
        ],
        "tips": ["Place it in the green zone before the ball crosses the red line."],
    },
    "hit_target": {
        "instruction": (
            "Hit the moving target's yellow center while avoiding blockers."
        ),
        "success": "The dart sticks in the yellow center and did not hit a blocker.",
        "keyboard": [
            ("mouse click", "on the target to attach the dart"),
        ],
        "notes": [
            {
                "text": "If you click a blocker, the dart hits the blocker and the task fails.",
                "red": ["fails"],
            },
        ],
    },
    "load_train": {
        "instruction": "Drop the marble into an allowed wagon of the circling train.",
        "success": "The marble lands in an allowed wagon.",
        "keyboard": [
            ("Space", "release the marble (it hovers over the near-rail drop)"),
        ],
    },
    "marble_shelf_maze": {
        "instruction": "Tilt the shelves to route the marble through the maze into the bowl.",
        "success": "The marble ends in the bowl.",
        "keyboard": [
            ("Arrows", "hold right/left arrows to tilt the active shelf"),
            ("mouse click", "on buttons and hold to tilt the shelf"),
        ],
    },
    "pack_fruits": {
        "instruction": (
            "Pack red and green apples from the belts into their matching baskets."
        ),
        "success": "Matching apples are in matching baskets; distractors (if present) are not packed.",
        "keyboard": [
            ("mouse click", "on a fruit, then on a basket"),
        ],
        "tips": ["The chosen fruit highlights lighter."],
    },
    "pick_ripe_apple": {
        "instruction": (
            "Pick the good apple and place it in the basket while ignoring spoiled fruit."
        ),
        "success": "The ripe apple is in the basket; spoiled fruit is not.",
        "keyboard": [
            ("Left / Right", "pick the apple on that side (moves over the basket)"),
            ("Space", "release the apple"),
            ("mouse click", "on an apple to pick, click again to release"),
        ],
    },
    "place_block_belt": {
        "instruction": (
            "Place the block onto the belt so it rides upright into the exit bowl."
        ),
        "success": "The block stays upright and enters the exit bowl.",
        "keyboard": [
            ("mouse click", "on the belt to place the cube"),
        ],
    },
    "play_billiard": {
        "instruction": "Strike the red ball into an allowed pocket without robot contact.",
        "success": "The red ball enters an allowed pocket; the robot does not touch the ball.",
        "keyboard": [
            ("mouse click", "places the cue tip at that location"),
            ("Left / Right", "rotate the stick counterclockwise / clockwise"),
            ("Space", "hit"),
        ],
    },
    "control_quality": {
        "instruction": "Stamp red and green tiles while skipping black outliers.",
        "success": "Red and green tiles are stamped; black tiles are skipped.",
        "keyboard": [
            ("Left / Right", "hold to stamp red / green"),
            ("mouse click", "on buttons and hold to stamp"),
        ],
        "tips": ["Do not press while a black tile is under the stamp."],
    },
    "drop_ball_hole": {
        "instruction": (
            "Guide the ball through the target hole of the rotating sorter into the container."
        ),
        "success": "The ball falls through the target hole into the container.",
        "keyboard": [
            ("mouse click", "on the rotating platform to drop the ball at that location"),
        ],
    },
    "sort_apples_belt": {
        "instruction": (
            "Sort red and green apples into matching bins; send rotten apples to the dump."
        ),
        "success": "Each apple is in the correct bin, and rotten apples go to the dump.",
        "keyboard": [
            ("Left / Right", "rotate the gate"),
            ("Up", "open the gate for a spoiled apple"),
        ],
    },
    "whack_moles": {
        "instruction": "Whack both moles from above without touching a rabbit.",
        "success": "Both moles are hit; a rabbit (if present) is not touched.",
        "keyboard": [
            ("mouse click", "on a mole to hit it"),
        ],
        "notes": [
            {
                "text": "If a mole is clicked while inside, the hit misses. Clicking a rabbit fails.",
                "red": ["while inside", "fails"],
            },
        ],
    },
    # Household
    "trap_bug": {
        "instruction": "Trap the scurrying bug under the glass box.",
        "success": "The bug is contained under the trap.",
        "keyboard": [
            ("mouse click", "on the table — the trap drops from 4 cm above that spot"),
        ],
    },
    "boil_milk": {
        "instruction": (
            "Turn the stove on so the milk rises, then shut it off before the milk overflows."
        ),
        "success": "The milk has risen and the stove is off before overflow.",
        "keyboard": [
            ("Space", "turn the stove on / off"),
            ("mouse click", "on the knob to turn the stove on / off"),
        ],
    },
    "fill_coffee_jar": {
        "instruction": "Fill the marked glass jar to the <amount> line.",
        "success": "The coffee fill is at <amount> (±5%).",
        "keyboard": [
            ("1 / 2 / 3 / 4", "fill force levels 1–4"),
        ],
    },
    "pour_beer": {
        "instruction": (
            "Pour beer into the mug, then click the finish bell. Overflow fails."
        ),
        "success": "The mug is filled without overflow and the finish bell is clicked.",
        "keyboard": [
            ("Space", "hold to pour"),
            ("mouse click", "and hold on the button to pour; click the finish bell to score"),
        ],
    },
    "cook_food": {
        "instruction": (
            "Shut the stove off at the target doneness (food starts in the pan)."
        ),
        "success": "The food is at the target doneness and the stove is off.",
        "keyboard": [
            ("Space", "turn the stove on / off"),
            ("mouse click", "on the knob to turn the stove on / off"),
        ],
        "tips": ["Food starts inside the pan."],
    },
    "cook_food_timer": {
        "instruction": (
            "Shut the stove off at the target doneness using the pie timer (food starts in the pan)."
        ),
        "success": "The food is at the target doneness and the stove is off.",
        "keyboard": [
            ("Space", "turn the stove on / off"),
            ("mouse click", "on the knob to turn the stove on / off"),
        ],
        "tips": ["Food starts inside the pan. The pie timer follows the stove."],
    },
    "measure_ingredient": {
        "instruction": "Fill the marked jar to the <amount> ring from the oil nozzle.",
        "success": "The fill is at <amount>; oil that misses the jar fails.",
        "keyboard": [
            ("Space", "turn the nozzle on / off"),
            ("mouse click", "on the button to turn the nozzle on / off"),
        ],
    },
    "make_soup": {
        "instruction": (
            "Tip the chopping-board vegetables into the pot without dropping any pieces."
        ),
        "success": "All pieces go into the pot.",
        "keyboard": [
            ("mouse click", "places the board (top center at the click, 2 cm above the pot)"),
            ("Left / Right", "tilt the board left / right"),
        ],
    },
    "catch_cup": {
        "instruction": (
            "Place the pillow under the tipping mug so it lands softly instead of hitting the table."
        ),
        "success": "The mug lands on the pillow, not the table.",
        "keyboard": [
            ("mouse click", "on the table to place the pillow"),
        ],
        "notes": [
            {"text": "The pillow can be placed only once.", "red": ["once"]},
        ],
    },
    "catch_mouse_object_drop": {
        "instruction": (
            "Place the pillow-lined basket under the falling object so it does not hit the table."
        ),
        "success": "The object lands in the basket, not on the table.",
        "keyboard": [
            ("mouse click", "on the table to place the basket"),
        ],
        "notes": [
            {"text": "The basket can be placed only once.", "red": ["once"]},
        ],
    },
    "stop_ball": {
        "instruction": "Stop the rolling ball before it falls off the table.",
        "success": "The ball is stopped on the table.",
        "keyboard": [
            ("mouse click", "on the ball while it is rolling on the table"),
        ],
        "notes": [
            {
                "text": "An earlier click will result in failure.",
                "red": ["failure"],
            },
        ],
    },
    "clean_table": {
        "instruction": (
            "Wipe the spreading coffee spill with the sponge before it reaches the laptop."
        ),
        "success": "The stain is cleaned before it reaches the laptop.",
        "keyboard": [
            ("mouse click", "on the table — the sponge makes contact at that spot"),
        ],
        "tips": ["The sponge hovers 5 cm over the table until you click."],
    },
}

# Match env sampling (no SAPIEN import). fill_coffee: RandomState(seed+202).
# measure_ingredient: RandomState(seed+202) after the env change to _layout_rng(202).
_COFFEE_FILL_LEVELS = tuple(round(0.15 + 0.05 * i, 2) for i in range(14))
_OIL_FILL_LEVELS = (0.25, 0.50, 0.75, 1.0)


def episode_amount(task: str, seed: int | None) -> str:
    """Human-readable fill target for this task/seed, or empty if unused."""
    if seed is None:
        return ""
    try:
        seed_i = int(seed)
    except (TypeError, ValueError):
        return ""
    if task == "fill_coffee_jar":
        frac = float(__import__("numpy").random.RandomState(seed_i + 202).choice(_COFFEE_FILL_LEVELS))
        return f"{int(round(frac * 100))}%"
    if task == "measure_ingredient":
        frac = float(__import__("numpy").random.RandomState(seed_i + 202).choice(_OIL_FILL_LEVELS))
        return f"{int(round(frac * 100))}%"
    return ""


def fill_amount_placeholders(text: str, amount: str) -> str:
    if not text:
        return text
    if amount:
        return text.replace("<amount>", amount)
    return (
        text.replace("the <amount> ", "the target ")
        .replace("to <amount>", "to the target")
        .replace("at <amount>", "at the target")
        .replace("<amount>", "the target")
    )


ROBOT_BASIC_CONTROLS = """\
Space — open / close selected gripper(s)
1 / 2 / 3 — select left / right / both arms
V — cycle view: head_camera ↔ gripper(s)
"""
