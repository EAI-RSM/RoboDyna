# Worked example: the `cook_meat` task

A full task built start-to-finish, including a novel **time-evolving state** (the meat browns from
raw → cooked → burnt while on the pan). The live code is `envs/cook_meat.py`. This walks through the
decisions and the fixes that made it work — read it alongside the task file.

## Goal (final form)

One arm picks a raw steak off the **table**, places it in a pan, the steak's color changes gradually
(vivid red → dark brown, the full gradient runs red→brown→black) over an internal cooking timer, then
once a target doneness is reached the arm lifts it off and sets it back **on the table**. The
"dynamic" element is the **color**, not object motion. Pan + steak spawn on the **same side**
(arm-reach limit); positions are otherwise random.

Earlier iterations tried a plate as source/destination (plate→pan→plate); it was dropped because two
large surfaces (0.30 m pan + 0.23 m plate) don't fit on one arm's side and the extra two
manipulations tanked the success rate. The lesson — not the plate — is what's worth keeping.

## What the asset situation forced

The object library had **no raw meat** (`006_hamburg` is a wrapped burger; `005_french-fries` is a
fries carton). So a new asset was sourced: a CC0 low-poly steak from Poly Pizza
(`static.poly.pizza/<uuid>.glb`), integrated as `assets/dyna_assets/200_steak/`. Two integration lessons:
- The GLB had a **node-scale transform**; exporting `scene.geometry[0]` dropped it and SAPIEN loaded a
  ~1 mm mesh. Fix: bake transforms (`scene.to_geometry()` / `dump(concatenate=True)`) before export.
- The steak's **texture overrode `base_color`**, so recoloring did nothing. Fix: strip the texture
  (bake a plain PBR material) — then `set_base_color` fully controls the appearance, which is exactly
  what a cooking color needs.
Both are handled by `scripts/integrate_object.py --strip-texture`. The pan reused `106_skillet`
(already annotated; skillet asset `106_skillet`).

## The task structure (`envs/cook_meat.py`)

- `setup_demo` captures `task_args.cook_meat` (cook_steps, target_doneness) before `_init_task_env_`.
- `load_actors` spawns the pan (static, central on a chosen side) and the steak (same side, outer),
  caches the steak's `RenderBodyComponent.render_shapes`, and inits `doneness=0`.
- The cooking machinery:
  - `_set_meat_color(d)` — piecewise-lerp a 3-stop palette (light red @0 → dark brown @0.5 →
    near-black @1) and set `base_color` on every render shape.
  - `_update_kinematic_tasks(self)` — `super()` first, then if cooking is active **and** the steak is
    in contact with the pan, advance `doneness += 1/cook_steps` and recolor. Runs every physics step.
  - `_cook_idle()` — loop `_update_kinematic_tasks(); scene.step()` and `_take_picture()` every
    `save_freq` steps, until `doneness >= target_doneness`. (Records the browning into the video;
    `delay()` would not, because it disables capture.)
- `play_once` → `_play_once_static` (and an optional `_play_once_dynamic` intercept path):
  grasp steak → lift → place on pan → `_cook_idle` → grasp off → lift.
- `check_success` — reached target doneness on the pan AND steak lifted clear of the pan.
- `get_obs` override records `cooking/doneness` per frame into the HDF5.

## The fixes that turned a 0%-yield task into a working one

1. **Opposite-side layout → unplannable place.** First version spawned steak and pan on opposite
   sides; the single arm couldn't reach across to place. Fix: spawn both on the **same side**,
   choose the arm by the steak's x-sign.
2. **`place_actor(functional_point_id=0)` → unreachable pose.** Passing `functional_point_id` aligned
   the steak's authored bottom-frame orientation to the pan, producing a bad gripper pose. Fix: drop
   `functional_point_id`, use `constrain="free"` and the pan's `get_functional_point(0)` as
   `target_pose` (the pattern in `pack_fruits.py` / `place_block_belt.py`).
3. **Gripper descending into the pan → rim collision.** Fix: release slightly above (`pre_dis=0.08`,
   `dis=0.03`) so the steak drops the last bit instead of the gripper diving in.
4. **Color change too subtle / too short.** Widened the palette to full light-red→black, raised
   `cook_steps` (longer, more frames) and `target_doneness` to 1.0 (cook through the whole range).
5. **Silent infinite loop.** A failing policy made the collector retry seeds forever. Always
   smoke-test with `python -u` + a `timeout`, and gate `plan_success` debug prints behind an env var
   to localize which `self.move` breaks.
6. **Raising the success rate (a separate hunt from "does it work").** Instrumenting each stage showed
   planning was ~100% — the SR drains were downstream: (a) the steak missing the pan bowl → shrink the
   steak (10→7.4 cm made landing reliable); (b) the set-down using absolute `move_to_pose` failing and
   dropping the steak over the pan → switch to relative `move_by_displacement` (shift outboard, lower,
   open); (c) `check_success` too strict → require only "cooked AND on the table AND horizontally
   clear of the pan", no return-to-start clause; (d) `check_render_success: true` makes every episode
   succeed twice (plan + render), roughly squaring the failure rate — disable it or accept the cost.

After fixes: episodes collect and render, the video shows the steak picked off the table, browning in
the pan, then set back on the table; the HDF5 `cooking/doneness` ramps to the target then freezes.
Diagnosing SR means measuring *which stage* fails, not guessing — and on a slow planner each episode
is tens of seconds, so use a short `cook_steps` to get more samples per minute while measuring.

## Reusable shape of a "time-evolving state" task

1. Cache the actor's render shapes (or any state handle) in `load_actors`.
2. Advance the state in an overridden `_update_kinematic_tasks` (per physics step), gated on a
   condition (here: contact with the pan). Drive it by **step count** for two-pass determinism.
3. Use a frame-recording idle loop (not `delay()`) to let time pass on camera.
4. Record the state into `get_obs()` so it lands in the trajectory.
5. Trigger the next action when the state crosses a threshold.
