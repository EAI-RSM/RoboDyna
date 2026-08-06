---
name: SAPIEN-task-creator
description: >-
  End-to-end process for designing a NEW manipulation task in a RoboTwin 2.0 / DOMINO / SAPIEN
  benchmark (the deployment at /shared_work/markhsp/DOMINO): scaffolding the task class, sourcing or
  authoring object assets, wiring the scripted expert policy + success check, adding the config and
  language instructions, and validating by collecting + watching a rollout. Use this skill whenever
  the user wants to add, design, author, scaffold, or extend a task or object in RoboTwin/DOMINO/
  SAPIEN — including phrasings like "make a new task", "add an object", "create a pick-and-place /
  cooking / stacking task", "introduce a new state or property", or naming a task idea (e.g.
  "cook_meat", "pour_water"). Reach for it even when the user only describes the behavior they want
  rather than saying "task" — authoring one touches assets, planner primitives, configs, and a
  two-pass collector whose pitfalls are non-obvious, so route through this skill rather than guessing.
---

# Designing a new SAPIEN / RoboTwin / DOMINO task

This is the worked process for adding a task to the benchmark at `/shared_work/markhsp/DOMINO`
(RoboTwin 2.0 + SAPIEN sim + curobo planner; conda env `/shared_work/markhsp/envs/domino`). It was
distilled from building the `cook_meat` task end-to-end, including the failures along the way.

A task = a Python class that **spawns objects** and runs a **scripted expert policy** that solves
them. The collector runs that policy over many random seeds; successful rollouts become trajectories
(HDF5 + video). You are writing a deterministic-ish solver + a success check, not labeling data.

Always work in the env with the headless-render vars set:
```bash
source /shared_work/markhsp/miniforge3/etc/profile.d/conda.sh
conda activate /shared_work/markhsp/envs/domino
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json; unset DISPLAY
```

## The process at a glance

1. **Scope the task** — objects involved, the goal, the success condition, whether anything has a
   time-evolving state. Pick the closest existing task in `envs/` as a template.
2. **Get the objects** — reuse an annotated asset, or add a new one (see step 2 below).
3. **Write `envs/<task>.py`** — `load_actors`, `play_once`, `check_success` (+ optional dynamic path
   and custom per-step state).
4. **Add a general config** + a `task_args.<task>` block; add a `description/task_instruction/<task>.json`.
5. **Validate** — offscreen-render the assets, smoke-collect 2 episodes, watch the video, iterate.

No registration step exists — the collector finds a task by filename:
`importlib.import_module("envs.<task>")` then `getattr(module, "<task>")`. So the **file name, the
class name, and the CLI task arg must all match**.

## Step 1 — Scope and pick a template

Read 1–2 existing tasks closest to your goal and mirror their structure:
- pick-and-place onto a target → `envs/place_object_scale.py`, `envs/place_bread_skillet.py`
- press/click/contact → `envs/click_bell.py`, `envs/press_stapler.py`
- bimanual / handover → `envs/handover_block.py`
- moving-object intercept (DOMINO "dynamic") → `envs/adjust_bottle.py`

For the API surface (the four methods, the motion primitives, the per-step hooks, contact detection,
and how the collector drives an episode) read `references/task-anatomy-and-api.md`.

## Step 2 — Get the objects

Objects live in `assets/objects/<NNN_name>/` with `visual/base<id>.glb`, `collision/base<id>.glb`,
a per-variant `model_data<id>.json` (grasp/placement points + scale), and `points_info.json`.

- **Reuse first.** If an annotated object fits, just `create_actor(self, pose, modelname=..., model_id=..., convex=True)`. Check it has the `contact_points_pose` (graspable) and `functional_matrix`
  (placeable) you need.
- **No suitable asset?** Add one. The library has surprising gaps (e.g. it has no raw meat —
  `006_hamburg` is a wrapped burger). Source a **CC0** mesh (Poly Pizza `static.poly.pizza/<id>.glb`
  is scriptable and login-free; Sketchfab CC0 for higher fidelity), then integrate it with
  `scripts/integrate_object.py` (bakes scene-graph transforms, optionally strips the texture, scales
  to a real-world size, and writes `model_data0.json` + `points_info.json` + a `NOTICE`). User
  preference on this deployment: **new object ids start at 200**.
- For the full asset format, the `model_data` schema, and how to set physical properties
  (mass/friction/damping/static), read `references/objects-and-properties.md`.

**Then validate the asset before writing task code** with `scripts/validate_asset.py` — it renders
the object at its authored scale (catch a wrong scale early), and can test a `base_color` recolor.

## Step 3 — Write `envs/<task>.py`

Skeleton (mirrors the real tasks):
```python
from ._base_task import Base_Task
from .utils import *
import sapien
import numpy as np

class my_task(Base_Task):
    def setup_demo(self, **kwags):
        # capture any task-scoped config BEFORE init (kwags isn't stored on self otherwise)
        self._cfg = kwags.get("task_args", {}).get("my_task", {})
        super()._init_task_env_(**kwags)

    def load_actors(self):
        # spawn objects at seed-randomized poses; set masses; reserve space
        self.obj = rand_create_actor(self, modelname="...", model_id=0, convex=True,
                                     xlim=..., ylim=..., zlim=[0.74 + self.table_z_bias], qpos=...)
        self.add_prohibit_area(self.obj, padding=0.05)

    def play_once(self):
        arm = ArmTag("right" if self.obj.get_pose().p[0] > 0 else "left")
        self.move(self.grasp_actor(self.obj, arm_tag=arm, pre_grasp_dis=0.1))
        self.move(self.move_by_displacement(arm, z=0.1, move_axis="arm"))
        self.move(self.place_actor(self.obj, target_pose=<target>, arm_tag=arm, constrain="free"))
        self.info["info"] = {"{A}": "modelname/base0", "{a}": str(arm)}
        return self.info

    def check_success(self):
        p = self.obj.get_pose().p
        return bool(<geometric / contact condition>)
```

### Hard-won pitfalls (these cost real debugging time — apply them up front)

- **Each arm only reaches its own half of the table — this is a hard reachability limit, not a
  tuning knob.** The **left arm cannot reach the right side of the table, and the right arm cannot
  reach the left side** (table x roughly spans `[-0.35, +0.35]`; the divide is the centerline `x=0`).
  Consequences for any single-arm task: every object that one arm must touch — the source, the
  target, and any intermediate (pan, basket, plate) — must spawn on the **same side** (same sign of
  x). Pick the arm by the object's x-sign (`ArmTag("right" if pose.p[0] > 0 else "left")`). You may
  keep spawn positions fully random *within* a side, but never split a single-arm task's objects
  across the centerline. If the task inherently needs both sides, it must be bimanual (one object per
  arm, brought together near the middle — see `handover_*` / `place_bread_skillet`).
- **Big objects don't both fit on one half.** A side is only ~0.35 m wide, so two large surfaces
  (e.g. a 0.30 m pan + a 0.23 m plate) can't coexist there without overlap. Shrink one (author a
  smaller-scale variant) or drop it; don't fight the geometry with tighter spawn ranges.
- **`place_actor` onto a functional point: use `constrain="free"` and do NOT pass
  `functional_point_id`.** Passing it forces the held object's authored functional-frame orientation
  onto the target, which usually yields an unreachable gripper pose. The working pattern is
  `place_actor(obj, target_pose=target.get_functional_point(0), arm_tag=arm, constrain="free", pre_dis=0.08, dis=0.03)`.
- **Release slightly above the target** (`dis≈0.02–0.03`, `pre_dis≈0.08`). Descending the gripper to
  near-contact inside a concave target (pan, basket) collides with the rim and the plan fails.
- **To set an object down on open table, prefer relative `move_by_displacement` over an absolute
  `move_to_pose`/`place_actor`.** Shifting the held object outboard then lowering (keeping the current,
  already-reachable gripper orientation) plans far more reliably than solving IK to an absolute far
  pose — the latter silently fails and the object gets dropped wherever the gripper happened to be
  (often back over the source). Use `place_actor` only where alignment matters (seating into a bowl).
- **Smaller manipulands land in concave targets (and grasp) far more reliably.** If an object keeps
  missing a bowl/plate or is too big to grasp, shrinking it is often a bigger SR win than tuning the
  place — a 10 cm steak missed a pan ~40% of the time; 7 cm landed every time. **How to shrink
  depends on whether the asset is shared:** for a *dedicated* asset, lower `scale` in its
  `model_data` (rendered size = `extents * scale`). For a *shared/stock* asset (don't resize it for
  every other task), pass **`scale_mult`** to `create_actor` (e.g. `scale_mult=0.5`) — a per-spawn,
  load-time, in-memory multiplier on the authored scale that also scales the contact/functional
  points; the asset file is untouched and other tasks still get stock size. Plain `scale=` is ignored
  (`model_data["scale"]` overrides it) — use `scale_mult`. See `references/objects-and-properties.md`.
- **Make `check_success` permissive about exact final pose.** Require the task's *intent* (e.g.
  "cooked AND resting on the table AND clear of the pan"), not an exact location. A back-to-origin or
  within-N-cm-of-start clause rejects perfectly good rollouts. Verify on the table with a z-window +
  a horizontal-distance-from-the-other-object check rather than strict contact flags.
- **Grasp/lift succeeding but place failing is the common failure mode** — instrument each
  `self.move(...)` by printing `self.plan_success` (gated behind an env var) to see exactly which
  step breaks. `self.move` is a no-op once `plan_success` is False, so the first false is the culprit.
- **`_init_task_env_` calls `_update_kinematic_tasks()` during setup — BEFORE `load_actors` runs.**
  This is the #1 crash when overriding `_update_kinematic_tasks`: it fires inside `load_camera`
  (part of `_init_task_env_`) while your per-step state (`self.proj`, `self.attached`, counters,
  cached actors) doesn't exist yet → `AttributeError ... has no attribute '...'` on EVERY episode
  (so the job error-loops forever). **Fix:** guard the top of your override and bail until state
  exists — `super()._update_kinematic_tasks(); if not getattr(self, "<state>", None): return` —
  and/or initialize all per-step state in `setup_demo` before `super()._init_task_env_(**kwags)`.
- **A task that never succeeds loops the collector forever** (it retries seeds until `episode_num`
  successes). Two defenses: (a) fix the policy so it actually succeeds; (b) for a sweep where you
  just need terminating jobs + trajectories, set `save_failed_cases: true` in the task config — the
  collector then saves every attempt (with its success flag) and stops after `episode_num`, so the
  job can't hang. Don't submit an unbounded job you haven't seen succeed at least once.
- **The collector retries seeds forever until it hits `episode_num` successes.** A buggy policy →
  infinite loop. Always smoke-test with a short `timeout` and unbuffered output (`python -u`,
  `PYTHONUNBUFFERED=1`) so you actually see the per-episode prints / tracebacks.

### Adding a custom time-evolving state (e.g. a cooking timer / color change)

This is the most powerful extension and the least obvious. The hooks:
- Override `_update_kinematic_tasks(self)` — call `super()` first, then run your per-step logic. This
  method runs on **every physics step** inside `take_dense_action`, so it's where a timer advances
  and where you mutate appearance.
- **Recolor**: reach the render body like tasks reach physics components —
  `for c in actor.actor.get_components(): if isinstance(c, sapien.render.RenderBodyComponent): for s in c.render_shapes: s.material.set_base_color([r,g,b,1])`.
  A textured GLB **ignores `base_color`** — strip the texture when integrating the asset so a flat
  material can be tinted (`scripts/integrate_object.py --strip-texture`). Harsh directional lighting
  also hides the color (saturates to white/black); the real collector's lighting renders it fine.
- **Make time pass while the arm is idle, with frames recorded**: don't use `delay()` (it zeroes
  rendering and skips frame capture). Loop `_update_kinematic_tasks(); self.scene.step()` and call
  `self._take_picture()` every `self.save_freq` steps.
- **Determinism across the two passes**: the collector runs once to plan (no cameras) and again to
  render. Drive any timer by **step count**, never by the rendered pixels, so a timed action fires
  identically in both passes.
- **Record the state into the trajectory**: override `get_obs()`, call `super()`, and add your fields
  (e.g. `obs["cooking"] = {"doneness": float(self.doneness)}`) — they get serialized per frame into
  the HDF5.

A complete annotated example of all of the above is `references/worked-example-cook_meat.md`
(and the live task at `envs/cook_meat.py`).

## Step 4 — Config and instructions

- **Config**: `task_config/*.yml` are **general, reusable** collection settings — do NOT make a
  task-named config. Reuse one (`demo_clean`, `demo_clean_dynamic`) or add a general one. The whole
  yaml is passed as `**args` to the task, so put task-specific params in a namespaced block and read
  them defensively:
  ```yaml
  task_args:
    my_task: { param_a: 1.0 }
  ```
  `cfg = kwags.get("task_args", {}).get("my_task", {}); self.param_a = cfg.get("param_a", DEFAULT)`.
- **Instructions**: add `description/task_instruction/<task>.json` with `full_description`, `schema`,
  and `seen`/`unseen` paraphrase lists using `{A}`/`{B}`/`{a}` placeholders that `play_once` fills via
  `self.info["info"]`. Mirror an existing file.

## Step 5 — Validate

```bash
# smoke: 2 episodes, unbuffered, time-capped so a broken policy can't loop forever
sed -i 's/^episode_num: .*/episode_num: 2/' task_config/<config>.yml   # remember to restore
timeout 300 python -u script/collect_data.py <task> <config> 2>&1 | grep -vE "OIDN Error|svulkan2"
```
- Outputs land in `data/<task>/<config>/`: HDF5 in `data/`, mp4 in `video/`.
- **Watch the video**: extract frames with the env's ffmpeg and view them
  (`ffmpeg -i video/episode0.mp4 -vf fps=3 frames/f_%03d.png`), then Read the PNGs. Confirm the
  manipulation looks right and any custom state renders.
- Check `check_success` actually fires (episode saved as SUCCESS) and the HDF5 carries your fields.
- Iterate spawn ranges / place distances / params; then restore `episode_num` and scale up (Slurm:
  `sbatch collect_demos.sbatch <task> <config>` — see the DOMINO-benchmark skill).

### Always record and show a sample demo — this is a required deliverable, not optional polish

**Every time you finish building or meaningfully changing a task, produce a short visual demo and
embed it in your reply — don't just report a success rate.** The user should be able to see the
rollout, not just read numbers. Do this even for quick iterations, not only the final handoff.

**Use the shared dual-view recorder** (see the `record-video-demo` skill). Do **not** invent a
per-task `record_*_demo.py` or rely on the training collector's preview mp4:

```bash
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json; unset DISPLAY
python script/bench_script/record_demo.py <task>
```

Outputs (auto-versioned `vN`, never overwrites prior demos):

- `./tmp/tmp_<task>/video/vN_head.mp4` — robot `head_camera`
- `./tmp/tmp_<task>/video/vN_topdown.mp4` — bird's-eye top-down (`observer_camera` reframed)
- `./tmp/tmp_<task>/video/vN_sidebyside.mp4` — convenience montage

For chat embeds (mp4 won't inline), convert **both** views to GIF:

```bash
mkdir -p /tmp/robodyna_demos
ffmpeg -y -i tmp/tmp_<task>/video/v<N>_head.mp4 \
  -vf "fps=10,scale=480:-1:flags=lanczos" -loop 0 /tmp/robodyna_demos/<task>_v<N>_head.gif
ffmpeg -y -i tmp/tmp_<task>/video/v<N>_topdown.mp4 \
  -vf "fps=10,scale=480:-1:flags=lanczos" -loop 0 /tmp/robodyna_demos/<task>_v<N>_topdown.gif
```

Embed both: `![<task> vN head](/tmp/robodyna_demos/<task>_vN_head.gif)` and
`![<task> vN top-down](/tmp/robodyna_demos/<task>_vN_topdown.gif)`.

If yield is low, it's almost always the planner failing on reach/placement — revisit the pitfalls
above before adding complexity.

## Files you end up adding (checklist)

- `assets/objects/<NNN_name>/` — only if adding an object (visual+collision glb, `model_data0.json`,
  `points_info.json`, `NOTICE`).
- `envs/<task>.py` — the task class (name == filename == CLI arg).
- `description/task_instruction/<task>.json` — language templates.
- `task_config/<general-config>.yml` — reuse or add a general config; task params go in `task_args`.

## Bundled resources

- `scripts/integrate_object.py` — turn a CC0 GLB into a benchmark object (bake transforms, optional
  texture strip, scale to real size, author `model_data0.json` + `points_info.json` + `NOTICE`).
- `scripts/validate_asset.py` — offscreen-render an object at its authored scale and optionally test a
  `base_color` recolor, before writing task code.
- `references/task-anatomy-and-api.md` — the task lifecycle, motion primitives, per-step hooks,
  contact detection, and the collector's two-pass loop.
- `references/objects-and-properties.md` — asset layout, `model_data` schema, CC0 sourcing, physical
  properties (mass/friction/damping/static), recolor mechanics.
- `references/worked-example-cook_meat.md` — annotated walkthrough of a full task with a novel
  time-evolving (cooking color) state, including every fix that made it work.
