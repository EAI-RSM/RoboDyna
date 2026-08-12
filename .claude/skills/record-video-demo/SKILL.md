---
name: record-video-demo
description: >-
  Record standard head-view task demo videos (robot head_camera) with the
  shared recorder. Use when the user asks for a demo video, sample rollout
  video, task GIF, visual demo, or to record/show a task; also after finishing
  or meaningfully changing a SAPIEN/RoboTwin/ DOMINO task. Prefer this over
  ad-hoc record_*_demo.py scripts or training-collector preview mp4s.
---

# Record video demos (standard)

**Always use the shared script** — do not copy older per-task `record_*_demo.py`
helpers or invent a new recorder.

```bash
# from repo root, robodyna/domino env active, headless Vulkan set
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json; unset DISPLAY
python script/bench_script/record_demo.py <task>
# optional:
python script/bench_script/record_demo.py <task> --config demo_dynamic
python script/bench_script/record_demo.py <task> --task-arg key=value
python script/bench_script/record_demo.py <task> --fps 25   # default; use 20–30
# catch_cuboid examples:
#   --task-arg catch_two_cuboids=true     # option 1
#   --task-arg opaque_surface=true     # option 2
# legacy: --option 1 / --option 2
```

Demo videos default to **25 Hz** (`save_freq=10`, via `fps ≈ 250/save_freq`). Do not wrap long household cooks in a short shell `timeout`.

Local conda env on this machine: `conda activate robodyna`.

## What it records

One successful expert episode → one required video:

| File | Camera | Meaning |
|------|--------|---------|
| `tmp/tmp_<task>/video/vN_head.mp4` | `head_camera` | Robot head / elevated training view |

- Version tag `vN` auto-increments; never overwrite prior demos.
- Do **not** also write top-down or side-by-side unless the user explicitly asks.
- Pass `save_path` via `run()` directly (script already does) — not `collect_data.py` `main()`, which nests `<task>/<config>`.

## After recording — show the user

Chat cannot inline mp4. Convert the head view to GIF and embed:

```bash
mkdir -p /tmp/robodyna_demos
N=<version>
TASK=<task>
ffmpeg -y -i tmp/tmp_${TASK}/video/v${N}_head.mp4 \
  -vf "fps=10,scale=480:-1:flags=lanczos" -loop 0 \
  /tmp/robodyna_demos/${TASK}_v${N}_head.gif
```

Embed in the reply:

```markdown
![<task> vN head](/tmp/robodyna_demos/<task>_vN_head.gif)
```

## Rules

- Required deliverable after building/changing a task — not optional polish.
- Standard demos are **head view only** (no top-down / side-by-side).
- Do **not** point users only at `data/<task>/<config>/video/episode0.mp4` (training preview).
- Task CLI name must match `envs/<task>.py` class name. Alias: `place_block_task` → `place_block_belt`.
- Smoke-test with `timeout` if the expert may hang (`timeout 600 python -u script/bench_script/record_demo.py <task>`).
