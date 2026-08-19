# Introducing new objects and properties

Two separate things: adding a new **object asset** (mesh + annotation) vs. tuning **physical
properties** (mass, friction, damping, scale, static/dynamic) of objects already in a task.
Derived from `envs/utils/create_actor.py`, `envs/utils/actor_utils.py`, `envs/_base_task.py`,
`script/create_object_data.py`, and the `assets/objects/` + `assets/dyna_assets/` layout.

## 1. Adding a new object asset

### Where it lives
```
assets/dyna_assets/<NNN_name>/        custom RoboDyna meshes (ids ≥ 200), e.g. 200_steak
assets/objects/<NNN_name>/            stock RoboTwin library, e.g. 001_bottle, 050_bell
├── visual/      base<id>.glb         # render mesh(es)
├── collision/   base<id>.glb         # collision mesh(es); convex decomposition or nonconvex
├── model_data<id>.json               # ONE per instance variant (the model_id you pass)
└── points_info.json                  # human-readable catalog of the points (doc, not used at runtime)
```
`create_actor(...)` resolves `assets/dyna_assets/<modelname>/` first, then falls back to
`assets/objects/<modelname>/`. If `collision/` or `visual/` subdirs are absent it falls back to a mesh in
the model dir. So to add a custom object:

1. Create `assets/dyna_assets/<NNN_yourname>/` with `visual/base0.glb` and `collision/base0.glb`
   (a watertight/convex-decomposed collision mesh; use `convex=True` at spawn for multi-convex).
2. Write `model_data0.json` (see schema below).
3. Reference it from a task: `create_actor(self, pose, modelname="<NNN_yourname>", model_id=0, convex=True)`.

Multiple `model_data<id>.json` + `base<id>.glb` in one folder = instance variants; tasks pick one
with `model_id=np.random.choice([...])` for visual/shape randomization.

### Sizing an asset — `scale` (in the asset) vs `scale_mult` (per-spawn)
Rendered size = `extents * scale`. There are two ways to set it; pick by whether the asset is shared:
- **Dedicated asset you authored** (e.g. `200_steak`, only `cook_meat` uses it): bake the size into
  `model_data["scale"]` (steak is `0.07`). The size lives in the asset. `integrate_object.py --scale`
  sets this at authoring time.
- **Shared / stock asset you must NOT resize for everyone** (e.g. `035_apple`, used by several tasks):
  pass **`scale_mult`** at spawn — `create_actor(self, pose, modelname="035_apple", model_id=0,
  scale_mult=0.5)`. It multiplies the asset's authored `scale` **at load time, in memory only** (it
  copies `model_data`; the on-disk file is untouched), scaling the mesh AND the contact/functional
  points consistently. Other tasks importing the same asset with no `scale_mult` get the stock size
  (default `1.0` = no-op). This is the right tool to shrink a too-big shared object for grasping
  without forking it into a new asset. `pick_ripe_apple` uses it on `035_apple`.
  (Plain `scale=` does NOT work for this — `model_data["scale"]` overrides it; only `scale_mult`
  composes on top.)

### model_data<id>.json schema (rigid object)
All transforms are 4×4 matrices in **object-local** coords; `get_*_point` multiplies the point's
local translation by `scale` and composes with the actor's world pose.
```json
{
  "scale": [0.13, 0.13, 0.13],          // applied to mesh AND point offsets; overrides create_actor's `scale` arg
  "center": [..],                        // bbox center (from trimesh)
  "extents": [..],                       // raw mesh bbox -> RENDERED SIZE = extents * scale (* scale_mult)
  "transform_matrix": [[..]],            // canonicalizing transform
  "contact_points_pose": [ [[4x4]], .. ],// GRASP frames -> get_contact_point(id); grasp_actor targets these
  "functional_point": [ [[4x4]], .. ],   // SEMANTIC spots (mouth, handle, place-here) -> get_functional_point(id)
  "target_pose": [ [[4x4]] ],            // canonical upright/target pose
  "orientation_point": [[4x4]]           // optional, get_orientation_point()
}
```
The point lists are what make an object usable: `grasp_actor(..., contact_point_id=k)` needs a
`contact_points_pose[k]`; `place_actor(..., functional_point_id=k)` needs a functional point.
**No points = the object can be spawned as scenery but not grasped/placed by the primitives.**

### Annotating the points
`script/create_object_data.py` is an **interactive SAPIEN viewer** tool to click/define contact &
functional points on a mesh and dump the json. Caveat: it opens a GUI window
(`scene.create_viewer()`), so on this headless GB10 box it needs X forwarding / a virtual display
(or run it on a workstation). Alternatively, author `model_data*.json` by hand — the matrices are
just position+rotation frames in local coordinates, and you can compute them from the mesh in a
short script with `trimesh` + `transforms3d`.

### Articulated objects (drawers, cabinets, microwaves)
Use a URDF instead: dir contains `mobility.urdf` + `model_data.json`, loaded via
`create_urdf_obj` / `create_sapien_urdf_obj`, wrapped as `ArticulationActor`. Its points carry a
`{"matrix":..., "base": "<link_name>"}` so they track a specific link. Joints driven with
`set_qpos`/`set_qvel`; see `_kitchens_base_task.py`.

### Objaverse pool
`assets/objects/objaverse/list.json` catalogs the large object pool (used by clutter generation,
`rand_create_cluttered_actor`). `script/create_object_data.py` / `create_object_data` workflows
import from there.

## 2. Setting physical properties

### Scene-global (friction, restitution, timestep)
Set once in `_base_task._init_task_env_` from kwargs, applied via `scene.default_physical_material`
which every collision shape uses:
```python
self.scene.set_timestep(kwargs.get("timestep", 1/250))
self.scene.default_physical_material = self.scene.create_physical_material(
    kwargs.get("static_friction", 0.5),
    kwargs.get("dynamic_friction", 0.5),
    kwargs.get("restitution", 0))
```
Those kwargs flow from the **task_config yml** (merged into `args` and passed through
`setup_demo(**args)`), so to change global friction/restitution for a benchmark, add
`static_friction: 0.8` etc. to your `task_config/<config>.yml` — no code change. For per-object
friction you'd build a dedicated `create_physical_material` and pass it to that actor's collision
shape (the helper uses the default; extend `create_actor` if you need per-object materials).

### Per-actor (the common case — done in `load_actors`)
Rigid `Actor`:
- `actor.set_mass(m)` — used everywhere to tune dynamics (lighter = easier to push/topple).
- Via physx component (what tasks do for dynamic scenes):
  ```python
  for c in actor.actor.get_components():
      if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
          c.set_linear_damping(10.0); c.set_angular_damping(10.0)
          c.set_linear_velocity(v);   c.set_angular_velocity(w)
  ```
- Spawn flags: `is_static=True` (immovable), `convex=True/False` (collision type), `scale`,
  `model_id` (which variant). Note `scale` from `model_data` overrides the `create_actor` arg.

Articulated `ArticulationActor`:
- `set_mass(mass, links_name=[...])`, `set_properties(damping, stiffness, friction, force_limit)`,
  `set_qpos`, `set_qvel`, `get_qlimits`.

### The static-vs-dynamic pattern
Tasks branch on `self.use_dynamic`: in static mode objects are spawned `is_static=True` (rock
solid, deterministic grasp); in dynamic mode they're `is_static=False` with low mass + high damping
so DOMINO's motion controller can push them and the arm must intercept. This is the single most
important "property" lever for the benchmark's difficulty. See `punch_dual_holes.py` vs
`catch_ramp_ball.py` for static fixture vs intercept-style mass/damping choices.

## Quick recipes

- **New graspable object:** make `assets/dyna_assets/<NNN_name>/{visual,collision}/base0.glb` +
  `model_data0.json` with at least one `contact_points_pose` (and a `functional_point` if it must be
  placed) + `points_info.json`; then `create_actor(self, pose, modelname="<NNN_name>", model_id=0, convex=True)`.
- **Make an existing object heavier/slippery in a task:** in `load_actors`, `actor.set_mass(...)`;
  for friction, add `static_friction`/`dynamic_friction` to the task_config (global) or attach a
  custom physical material.
- **New difficulty knob:** read a custom key from the config in `load_actors`
  (`self.my_param = kwargs.get(...)` via the task env) and use it to scale mass/damping/spawn range.

## Runtime recoloring (time-evolving appearance, e.g. cooking)

To change an object's color during an episode (rendered into RGB observations):
```python
import sapien.render
for c in actor.actor.get_components():
    if isinstance(c, sapien.render.RenderBodyComponent):
        for s in c.render_shapes:
            s.material.set_base_color([r, g, b, 1.0])
# get_obs() calls scene.update_render() before each capture, so the new color is picked up
```
Two gotchas learned the hard way:
- **A textured GLB ignores `base_color`** — `baseColorTexture` overrides the factor, so the recolor
  is invisible. Strip the texture at integration time (bake a plain PBR material) so the flat base
  color is fully controllable — `scripts/integrate_object.py --strip-texture`.
- **Harsh directional lighting** saturates the surface to white/black and hides the albedo in a quick
  standalone render (raw and cooked look identical). Verify with flat/ambient lighting
  (`scripts/validate_asset.py` does); the collector's own balanced lighting renders it correctly.

## Sourcing a CC0 mesh from the web

The library has gaps (e.g. no raw meat). Download a **CC0** mesh:
- **Poly Pizza** — low-poly, login-free, scriptable. A page `poly.pizza/m/<id>` embeds the GLB at
  `https://static.poly.pizza/<uuid>.glb` (curl it). Mixed CC0/CC-BY — record attribution in `NOTICE`.
- **Sketchfab CC0** — higher fidelity (PBR textures); download usually needs login/token.

`scripts/integrate_object.py` places it under `assets/dyna_assets/<id_name>/`, **bakes the scene-graph
transform into the vertices** (trimesh `bounds` includes node transforms but `geometry[0]` does not —
exporting bare geometry yields a microscopic mesh in SAPIEN), scales to a real-world size, and writes
`model_data0.json` (grasp at top-center, functional point at bottom-center by default) +
`points_info.json` + `NOTICE`. New-object ids start at **200** on this deployment.
