# RoboDyna Arcade — conceptual branch

A lightweight, browser-native reinterpretation of the RoboDyna dynamic-manipulation
tasks as arcade mini-games. This is a **concept / outreach track**, not the SAPIEN
physics sim — each game keeps the *signature reflex* of its source task, stripped to a
playable Canvas core (no robot, no curobo, no physics engine).

Everything lives in a single self-contained HTML file (`robodyna_arcade.html`): inline
CSS/JS, no external assets, no build step. Open it in any browser, or view the hosted
version below.

**Hosted artifact:** https://claude.ai/code/artifact/232778b1-b8ab-4685-8faa-caeac3cbc079

**26 cartridges** — all **22** dynamic-manipulation tasks (the `_pilot_*` suite) plus
**4** household bonus rounds.

## The 22 dynamic tasks

| Game | Archetype | Signature rule kept |
|------|-----------|---------------------|
| `whack_a_mole` | whack | Bop moles, **never the rabbit decoy** |
| `goalkeeper` | block | Read a **telegraphed** shot, slide the keeper in time |
| `sort_apples_belt` | sort | red→RED, green→GREEN, **rotten→GARBAGE** |
| `play_billiard` | aim | Slingshot the red ball into a pocket; blue = **foul** |
| `dispense_gummy` | count | Dispense the **exact** count (tap = 1, hold = stream) |
| `catch_ramp_ball` | predict | Read the launch arc, park the cup at the landing |
| `catch_rat` | grab | Grab rats the instant they surface, before they retract |
| `catch_shelf_marble` | slide-catch | Track a marble down zig-zag shelves; catch it in the bowl |
| `catch_valley_ball` | place | Place the cup past the line for the **red** ball; let the **black decoy** fall |
| `stop_valley_ball` | block | Hold the bat in the **red** ball's flight; keep clear of the decoy |
| `catch_marbles_trapdoors` | react | Open the trapdoor whose **colour matches** the rolling marble |
| `quality_control` | react | Stamp reds & greens in the zone; **never stamp a black reject** |
| `cup_curtain_slot` | timing | Push the cup up through a **swaying gap** into the slot |
| `hit_target` | aim | Fire darts into a **drifting** bullseye; inner rings score more |
| `load_train` | drop | Drop the ball into a **red** wagon circling past, not the grey ones |
| `marble_shelf_maze` | tilt | Tilt the board to roll the marble down ledges into the bowl |
| `rotating_shape_sorter` | rotate-match | Drop each block when its **matching hole** spins to the marker |
| `place_block_belt` | balance | Nudge a top-heavy block so it rides **upright** to the bowl |
| `pick_ripe_apple` | timing | Pick the apple at **ripe red** — not green (early), not yellow (spoiled) |
| `packing` | sort-drag | Drag apples/oranges into their **matching** basket before they ride off |
| `cook_meat` | doneness meter | Sear the steak to the **green band**, then plate — not raw, not burnt |
| `dual_hole_punch` | dual-timing | Punch pages on **both belts** as each lines up under its head |

## Household bonus rounds

| Game | Archetype | Signature rule kept |
|------|-----------|---------------------|
| `pour_beer` | meter | Burst-pour; **foam settles**; don't overflow the rim |
| `trap_bug` | trap | Slam the box on the roach; **spare the ladybug** |
| `catch_rolling_cup` | catch | Catch tumbling cups; **upright** = bonus |
| `clean_table` | contain | Wipe the spreading spill before it reaches the laptop |

## Design

Single deliberate "telemetry cabinet" visual world — blue-charcoal ground, one
signal-amber accent, red/green reserved as semantic object colors, monospace HUD with
tabular numerals. Hub grid of cartridges → full-frame Canvas game with its own HUD and a
back-to-hub control. Keyboard and pointer/touch controls; `prefers-reduced-motion`
respected; CSP-clean (no CDN, no external assets), so it runs as a hosted artifact.

## Extending

Add a game by writing a `makeX()` factory (methods: `intro`, `update`, `render`, and any
of `down`/`move`/`up`/`key`/`keyup`), an emblem SVG in `EMB`, and a `GAMES[]` entry. The
shared harness handles the loop, DPR/resize, virtual coords (810×540), input dispatch,
HUD (`setHUD`), and win/lose overlays (`showOverlay`/`hideOverlay`). Keys arrive
lowercased, so match `'arrowleft'`, `' '`, etc.
