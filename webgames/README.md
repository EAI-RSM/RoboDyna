# RoboDyna Arcade — conceptual branch

A lightweight, browser-native reinterpretation of the RoboDyna dynamic-manipulation
tasks as arcade mini-games. This is a **concept / outreach track**, not the SAPIEN
physics sim — each game keeps the *signature reflex* of its source task, stripped to a
playable Canvas core (no robot, no curobo, no physics engine).

Everything lives in a single self-contained HTML file (`robodyna_arcade.html`): inline
CSS/JS, no external assets, no build step. Open it in any browser, or view the hosted
version below.

**Hosted artifact:** https://claude.ai/code/artifact/232778b1-b8ab-4685-8faa-caeac3cbc079

## Games (each maps to a real task)

| Game | Archetype | Source task | Signature rule kept |
|------|-----------|-------------|---------------------|
| `whack_a_mole` | whack | `envs/whack_a_mole` | Hit moles, **never the rabbit decoy** |
| `goalkeeper` | block | `envs/goalkeeper` | Move keeper to block a **telegraphed** shot in time |
| `sort_apples_belt` | sort | `envs/sort_apples_belt` | red→RED, green→GREEN, **rotten→GARBAGE** |
| `play_billiard` | aim | `envs/play_billiard` | Slingshot red ball into a pocket; blue = **foul** |
| `pour_beer` | meter | `envs/pour_beer` | Burst-pour; **foam settles**; don't overflow the rim |
| `trap_bug` | trap | `envs/trap_bug` | Slam the box on the roach; **spare the ladybug** |
| `catch_rolling_cup` | catch | `envs/catch_rolling_cup` | Catch tumbling cups; **upright** = bonus |
| `dispense_gummy` | count | `envs/dispense_gummy` | Dispense the **exact** count (tap = 1, hold = stream) |
| `catch_ramp_ball` | predict | `envs/catch_ramp_ball` | Read the launch arc, place the cup at the landing |
| `clean_table` | contain | README household task | Wipe the spreading spill before it reaches the laptop |

## Design

Single deliberate "telemetry cabinet" visual world — blue-charcoal ground, one
signal-amber accent, red/green reserved as semantic object colors, monospace HUD with
tabular numerals. Hub grid of cartridges → full-frame Canvas game with its own HUD and a
back-to-hub control. Keyboard and pointer/touch controls; `prefers-reduced-motion`
respected; CSP-clean (no CDN).

## Extending

Add a game by writing a `makeX()` factory (methods: `intro`, `update`, `render`, and any
of `down`/`move`/`up`/`key`/`keyup`), an emblem SVG in `EMB`, and a `GAMES[]` entry. The
shared harness handles the loop, DPR/resize, input dispatch, HUD, and win/lose overlays.
Good next candidates from the task suite: `hit_target`, `marble_shelf_maze`,
`rotating_shape_sorter`, `quality_control`, `place_block_belt`, `cup_curtain_slot`.
