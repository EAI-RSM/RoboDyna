# Designing a new DOMINO / RoboTwin task

How task authoring actually works in this codebase, derived from reading `envs/_base_task.py`,
example tasks (`envs/punch_dual_holes.py`, `envs/catch_ramp_ball.py`), `script/collect_data.py`, the
object-asset format, and the `code_gen/` LLM pipeline.

## Mental model

A "task" is a Python class that (a) spawns objects into a tabletop scene and (b) provides a
**scripted expert policy** that solves it. Data collection runs that policy under many random
seeds; successful rollouts become training trajectories. You are not labelling data — you are
writing a deterministic-ish solver and a success check, and randomization + rendering do the rest.

## Task discovery (no registry)

`script/collect_data.py::class_decorator` does:
```python
envs_module = importlib.import_module(f"envs.{task_name}")
env_class   = getattr(envs_module, task_name)   # class name MUST equal file/module name
```
So a new task = a new file `envs/<task_name>.py` containing `class <task_name>(Base_Task)`. The
filename, the class name, and the `task_name` CLI arg must all match. Nothing else to register.

Task configs (`task_config/*.yml`) are **generic** — `demo_clean`, `demo_clean_dynamic`,
`demo_randomized`, etc. apply to any task. A new task does **not** need its own config; reuse an
existing one. Configs only carry collection settings (cameras, episode_num, the dynamic knobs).

## The four methods you implement

```python
from ._base_task import Base_Task
from .utils import *
import sapien, numpy as np

class my_task(Base_Task):
    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags)        # boilerplate: builds scene/robot/cameras, then
                                                # calls load_actors() for you

    def load_actors(self):
        # spawn objects with randomized poses (driven by the episode seed)
        self.obj = rand_create_actor(self, modelname="001_bottle", model_id=13, convex=True,
                                     xlim=[-0.12,-0.08], ylim=[-0.13,-0.08], zlim=[0.752],
                                     rotate_rand=True, qpos=[0.707,0,0,-0.707])
        self.add_prohibit_area(self.obj, padding=0.1)   # keeps clutter/other objects away
        self.target_pose = [0.25, -0.12, 0.95, 0, 1, 0, 0]

    def play_once(self):
        # the scripted expert: build primitive actions and execute with self.move(...)
        arm = ArmTag("right")
        self.move(self.grasp_actor(self.obj, arm_tag=arm, pre_grasp_dis=0.1))
        self.move(self.move_by_displacement(arm, z=0.1, move_axis="arm"))
        self.move(self.place_actor(self.obj, target_pose=self.target_pose, arm_tag=arm,
                                   functional_point_id=0, is_open=False))
        self.info["info"] = {"{A}": f"001_bottle/base{13}", "{a}": str(arm)}  # fills instruction template
        return self.info

    def check_success(self):
        p = self.obj.get_functional_point(0)      # world pose of a semantic point on the object
        return p[2] > 0.9 and p[0] > 0.15
```

The episode loop (`collect_data.py`) calls, per seed:
`setup_demo(seed=…) → play_once() → success = plan_success and check_success()`. On success the
trajectory is saved; it keeps trying increasing seeds until `episode_num` successes accumulate.
A second pass replays the saved seeds with cameras on to render RGB + HDF5 + mp4.

## Motion-primitive API (the vocabulary of `play_once`)

Each primitive **returns** `(arm_tag, [Action,...])`; `self.move(...)` executes it through the
planner (CuroboPlanner). You can pass two primitives to `self.move(a, b)` to drive both arms in
parallel. Key primitives (signatures from `_base_task.py`):

- `grasp_actor(actor, arm_tag, pre_grasp_dis=0.1, grasp_dis=0, gripper_pos=0., contact_point_id=None)`
  — approach + close on a **contact point** of the object.
- `place_actor(actor, arm_tag, target_pose, functional_point_id=None, pre_dis=0.1, dis=0.02, is_open=True)`
  — move the held object so its functional point reaches `target_pose`, then (optionally) release.
- `move_by_displacement(arm_tag, x=0,y=0,z=0, quat=None, move_axis="world"|"arm")` — relative EE move
  (use `move_axis="arm"` + `z` to lift along the gripper axis).
- `move_to_pose(arm_tag, target_pose)`, `open_gripper`/`close_gripper`, `back_to_origin(arm_tag)`.
- Gripper state checks: `is_left_gripper_close()`, `is_right_gripper_open()`, etc.
- Contact queries for success: `get_gripper_actor_contact_position(modelname)`,
  `check_actors_contact(...)`.
- `self.move(...)` honors `self.plan_success`; if planning fails the episode is discarded — that's
  why expert scripts can be "optimistic" and rely on the success/plan gating.

`ArmTag("left"|"right")` is the arm selector. Convention in existing tasks: pick the arm by object
x-sign (`"right" if pose.p[0] > 0 else "left"`).

## Objects and their annotation points (the crux)

Objects live in `assets/objects/<NNN_name>/` with multiple `model_data<id>.json` variants (the
`model_id`/`base{id}` you choose). Each json carries, in object-local coordinates:
- `scale`, `center`, `extents`, `transform_matrix`
- `contact_points_pose` — list of 4×4 grasp frames → `actor.get_contact_point(id)`
- `functional_point` / functional points → `actor.get_functional_point(id)` (semantic spots:
  bottle mouth, mug handle, the place-here target)
- `target_pose`, `orientation_point`

`get_contact_point(id)` / `get_functional_point(id)` return these transformed to **world** coords
(matrix/list/`sapien.Pose`). Grasping targets a contact point; placing aligns a functional point to
`target_pose`. **A new task can only manipulate objects that have the contact/functional points it
needs** — so either reuse annotated objects from `assets/objects/` or annotate new meshes
(`script/create_object_data.py` helps build these). `points_info.json` in each object dir catalogs
the available points.

`create_actor(...)` / `rand_create_actor(...)` (from `envs/utils/`) spawn an object by `modelname`
+ `model_id` at a (randomized) pose; `rand_pose(xlim, ylim, qpos, ...)` generates the seed-driven
pose. `add_prohibit_area(actor, padding)` reserves space so randomized clutter won't overlap it.

## Natural-language instructions

`description/task_instruction/<task_name>.json` holds templated instructions with `{A}` (object)
and `{a}` (arm) placeholders plus a `seen`/`unseen` list of phrasings. `play_once` returns
`self.info["info"] = {"{A}": ..., "{a}": ...}`, which the pipeline substitutes into a sampled
template to produce the per-episode language label. Generate these with the helpers in
`description/` (`gen_task_instruction_templates.sh`, the `_generate_task_prompt*.txt` prompts) or
hand-write the json.

## Adding the dynamic (DOMINO) dimension

DOMINO's novelty over RoboTwin is moving objects. A dynamic-aware task adds:
- a `use_dynamic` branch in `load_actors` (lighter mass, damping, non-static body),
- `_play_once_dynamic()` that wraps the grasp in a `robot_action_sequence` callback and calls
  `self.execute_dynamic_workflow(target_actor=…, end_position=…, robot_action_sequence=…, table_bounds=…)`,
- `get_dynamic_motion_config()` returning `{target_actor, end_position, table_bounds, ...}`,
- a `check_success` that (in dynamic mode) also checks displacement/proximity, and often a custom
  `check_stable`.
See `envs/punch_dual_holes.py` and `envs/catch_ramp_ball.py` for the two canonical patterns. The motion
itself is parameterized by the config knobs `dynamic_level` (1–3) and `dynamic_coefficient`.

## Two authoring paths

1. **Manual** (recommended to start): copy the closest existing task (`pack_fruits` / `place_block_belt` for pick-place,
   `punch_dual_holes` / `dispense_gummy` for contact, `cook_meat` / `make_soup` for bimanual), swap the object + target + success
   check, run a 2–3 episode smoke collection, iterate. Fastest way to learn the primitives.

2. **LLM-assisted** (`code_gen/`): RoboTwin's generator (`code_gen/task_generation.py`) writes a
   `gpt_<task>` subclass implementing `play_once` from a `task_description` + `actor_list`, using a
   documented primitive API (`prompt.py`'s `AVAILABLE_ENV_FUNCTION`/`FUNCTION_EXAMPLE`) and
   auto-retries against simulator errors (`test_gen_code.py`). Good for scaling out many tasks once
   the manual workflow is understood. Needs an OpenAI/Azure key (see `code_gen/gpt_agent.py`).

## Concrete checklist for a new task

1. Pick/annotate objects in `assets/objects/` that have the contact + functional points you need.
2. Write `envs/<task_name>.py` with `class <task_name>(Base_Task)` and the four methods.
3. (Optional) add `_play_once_dynamic` + `get_dynamic_motion_config` for the dynamic variant.
4. Write/generate `description/task_instruction/<task_name>.json`.
5. Smoke test: `episode_num: 2` config, `bash collect_data.sh <task_name> demo_smoke 0`, watch the
   mp4 and confirm `check_success` fires.
6. Scale up with `demo_clean` / `demo_clean_dynamic` and full `episode_num`.
