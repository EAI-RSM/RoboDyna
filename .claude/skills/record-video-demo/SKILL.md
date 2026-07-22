---
name: record-video-demo
description: >-
  Record standard dual-view task demo videos (robot head_camera + top-down
  bird's-eye) with the shared recorder. Use when the user asks for a demo
  video, sample rollout video, task GIF, visual demo, or to record/show a
  task; also after finishing or meaningfully changing a SAPIEN/RoboTwin/
  DOMINO task. Prefer this over ad-hoc record_*_demo.py scripts or
  training-collector preview mp4s.
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
# catch_rat examples:
#   --task-arg catch_two_mice=true     # option 1
#   --task-arg opaque_surface=true     # option 2
# legacy: --option 1 / --option 2
```

Local conda env on this machine: `conda activate robodyna`.

## What it records

One successful expert episode → two required videos (+ a montage):

| File | Camera | Meaning |
|------|--------|---------|
| `tmp_<task>/video/vN_head.mp4` | `head_camera` | Robot head / elevated training view |
| `tmp_<task>/video/vN_topdown.mp4` | `observer_camera` (reframed nadir) | Top-down bird's-eye |
| `tmp_<task>/video/vN_sidebyside.mp4` | both | Convenience montage (head \| top-down) |

- Version tag `vN` auto-increments; never overwrite prior demos.
- Pass `save_path` via `run()` directly (script already does) — not `collect_data.py` `main()`, which nests `<task>/<config>`.

## After recording — show the user

Chat cannot inline mp4. Convert both views to GIF and embed:

```bash
mkdir -p /tmp/robodyna_demos
N=<version>
TASK=<task>
ffmpeg -y -i tmp_${TASK}/video/v${N}_head.mp4 \
  -vf "fps=10,scale=480:-1:flags=lanczos" -loop 0 \
  /tmp/robodyna_demos/${TASK}_v${N}_head.gif
ffmpeg -y -i tmp_${TASK}/video/v${N}_topdown.mp4 \
  -vf "fps=10,scale=480:-1:flags=lanczos" -loop 0 \
  /tmp/robodyna_demos/${TASK}_v${N}_topdown.gif
```

Embed in the reply:

```markdown
![<task> vN head](/tmp/robodyna_demos/<task>_vN_head.gif)
![<task> vN top-down](/tmp/robodyna_demos/<task>_vN_topdown.gif)
```

## Rules

- Required deliverable after building/changing a task — not optional polish.
- Do **not** use third-person-only `observer` corner views as the standard demo anymore; head + top-down is the suite convention.
- Do **not** point users only at `data/<task>/<config>/video/episode0.mp4` (training preview).
- Task CLI name must match `envs/<task>.py` class name. Alias: `place_block_task` → `place_block_belt`.
- Smoke-test with `timeout` if the expert may hang (`timeout 600 python -u script/bench_script/record_demo.py <task>`).
