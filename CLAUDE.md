# CLAUDE.md

Author: Rui Heng Yang

This file provides guidance to Claude Code (claude.ai/code) when working with the RoboDyna project in this repository.

## Project Identity / Purpose

RoboDyna is a dynamic dual-arm robotic-manipulation benchmark (simulation environment + data-collection pipeline + policy training/eval harness), forked and extended from DOMINO (https://github.com/h-embodvis/DOMINO — RoboTwin 2.0 + SAPIEN + curobo). (Source: git log 91f3d6c commit message: "DOMINO/RoboTwin 2.0 fork"; External: upstream repo)

All tasks run on the dual-UR5 (`ur5-wsg`) embodiment, standardized on SAPIEN 3.0.3, exporting to HDF5 + LeRobot v2.1. (Source: README.md:5,11)

Upstream DOMINO is a large-scale dynamic-manipulation benchmark (35 tasks, 110K+ expert trajectories) with a companion VLA called PUMA, paper arXiv:2603.15620 (HUST/Huawei). (External: upstream DOMINO README — the local README was intentionally rewritten RoboDyna-only on 2026-07-06 and no longer carries upstream docs, see README.md:7)

### Task Status

- 4 "done" (production) tasks: `cook_meat`, `catch_ramp_ball`, `sort_apples_belt`, `pick_ripe_apple`. (Source: git log 91f3d6c: "4 complete tasks"; README.md:56-64 "Featured Tasks" table)
- 1 "in progress" task: `stab_moving_target`. (Source: README.md:64; git log 91f3d6c)
- Additional prototype tasks exist as `envs/*.py` files beyond the featured five; the LeRobot exporter has stable task-index entries for 16 tasks. (Source: README.md:66)

### Git State (as of this clone)

- Single local commit `91f3d6c`, author markhsp <mark@preludeos.com>, dated Mon Jun 29 03:55:44 2026 +0000, message "RoboDyna: pick_ripe_apple promoted to done (4 complete tasks). DOMINO/RoboTwin 2.0 fork, SAPIEN 3.0.3, dual-UR5." (Source: git log)
- Remote: `origin git@github.com:EAI-RSM/RoboDyna.git`. Working tree clean, 1758 tracked files. (Source: git remote / git status)

## Structure

Top-level layout: (Source: directory listing)
- `assets/` (33M, object meshes, mostly gitignored)
- `build_domino_aarch64.sh`
- `.claude/` (bundled Claude Code skill, see below)
- `code_gen/` (156K)
- `collect_data.sh`
- `collect_demos.sbatch`
- `data/` (8.0K, essentially empty — only `data/process_stuck.py`)
- `description/` (3.1M — 120 object description folders + 68 task-instruction JSON files)
- `envs/` (1.2M — one `<task>.py` per task e.g. `envs/cook_meat.py`, plus `_base_task.py`, `_GLOBAL_CONFIGS.py`, `camera/`, `robot/{ik.py,planner.py,robot.py}`, `utils/`)
- `index.html`
- `LICENSE`
- `policy/` (11M — ACT, DexVLA, DP, DP3, GO1, LLaVA-VLA, openvla-oft, pi0, pi05, PUMA, RDT, TinyVLA, Your_Policy)
- `README.md`
- `repro_one.py`
- `script/` (304K)
- `scripts/`
- `task_config/` (116K — `demo_dynamic.yml`, `demo_clean.yml`, `debug_dynamic.yml`, `_camera_config.yml`, `_embodiment_config.yml`, `_config_template.yml`, plus `_archive/` of retired per-task configs)

### Task Authoring (codeless registry)

Task authoring is a codeless registry: the collector imports `envs.<task_name>` and looks up a class with the same name — file name == class name == CLI arg, no registry. (Source: README.md:70; script/collect_data.py:24-28; .claude/skills/SAPIEN-task-creator/SKILL.md:43-45)

A bundled Claude Code skill at `.claude/skills/SAPIEN-task-creator/` (SKILL.md + `scripts/integrate_object.py`, `scripts/validate_asset.py` + 3 reference docs) automates task authoring. (Source: .claude/skills/SAPIEN-task-creator/ directory listing) This skill is invoked automatically by Claude Code when working under this directory tree; refer to it rather than duplicating its task-authoring instructions here. The README's "Add Or Modify A Task" section documents the task-class shape (`setup_demo`/`load_actors`/`play_once`/`check_success`, `_update_kinematic_tasks` hook, `task_args.<task_name>` config blocks). (Source: README.md:68-78)

## Environment & Dependencies

There is no single top-level requirements/environment file — dependencies are split by component. (Source: directory listing)

### Core DOMINO sim pipeline

`script/requirements.txt` pins `sapien==3.0.3` (fixed 2026-07-03; upstream wrongly pinned 3.0.0b1), plus torch==2.4.1, torchvision, transforms3d==0.4.2, scipy==1.10.1, mplib==0.2.1, gymnasium==0.29.1, trimesh==4.4.3, open3d==0.18.0, imageio==2.34.2, pydantic, zarr, openai, huggingface_hub==0.25.0, h5py, azure==4.0.0, azure-ai-inference, pyglet<2, wandb, moviepy, termcolor, av, matplotlib, ffmpeg, toppra (added), pyarrow (added). (Source: script/requirements.txt, full contents)

The install script `script/_install.sh` runs `pip install -r script/requirements.txt`, builds pytorch3d from GitHub with `--no-build-isolation` in a retry loop, patches `sapien/wrapper/urdf_loader.py` and `mplib/planner.py` via sed, clones curobo into `envs/curobo` if absent, checks out `v0.7.8`, installs it editable with `--no-build-isolation` in a retry loop, then re-pins `warp-lang==1.4.2 scipy==1.10.1`. Validated end-to-end from scratch — see "Working Setup" below. (Source: script/_install.sh; README.md:18)

### Per-policy environments

Each `policy/<X>/` subproject carries its own env files: (Source: directory listing)
- `policy/ACT/conda_env.yaml`, `policy/DexVLA/conda_env.yaml`, `policy/TinyVLA/conda_env.yaml`
- `policy/DexVLA/setup.py` + `requirements.txt`
- `policy/DP/pyproject.toml`, `policy/openvla-oft/pyproject.toml`
- `policy/pi0/pyproject.toml` + `.python-version` + `uv.lock` (uses uv, not conda); `policy/pi05/` same
- `policy/PUMA/pyproject.toml` + `requirements.txt` (own README: conda env `puma`, flash-attn==2.7.4.post1, GroundingDINO + SAM2 sub-steps)
- `policy/RDT/requirements.txt`, `policy/GO1/requirements.txt`

### System requirements

Linux, NVIDIA GPU, Python 3.10 (working envs use 3.10.20), Vulkan required for SAPIEN rendering. Headless collection convention: `export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json` and `unset DISPLAY`. (Source: README.md:22-27; working env `robodyna`)

## Entry Points

- `collect_data.sh` wraps `python script/collect_data.py $task_name $task_config`, sets `CUDA_VISIBLE_DEVICES`, deletes `.cache` after. Usage: `bash collect_data.sh <task> <task_config> <gpu_id>`. (Source: README.md:34,37; collect_data.sh:3-13)
- `script/collect_data.py` — the actual collector; `save_path` logic builds the effective path as `os.path.join(args["save_path"], task_name, task_config)`. (Source: script/collect_data.py:107-108; README.md:48)
- Evaluation: `script/eval_policy.py`, `script/eval_policy_client.py`, `script/eval_metrics.py`, `script/policy_model_server.py`. (Source: script/ directory listing; not yet exercised on this machine)
- Setup scripts: `script/_install.sh`, `script/_download_assets.sh`, `script/update_embodiment_config_path.py`. (Source: README.md:14-15,18-20)
- Per-policy entry points: each `policy/<X>/` has its own `train.sh`/`train_vla.py`/`imitate_episodes.py`, `eval.sh`, `deploy_policy.py`, `deploy_policy.yml`. (Source: directory listing)

## Datasets / Checkpoints

Locally collected (gitignored, on ruiheng-4080): `data/cook_meat/debug_dynamic/` (8 episodes, 80M, fixed debug params) and `data/cook_meat/demo_dynamic/` (50 episodes, 716M, randomized production params), plus the LeRobot v2.1 root `data_lerobot/domino_suite/` (58 episodes total). Assets (16G) downloaded under `assets/`. (Source: collect_smoke_log4.txt; collect_demo_dynamic_log.txt; du output 2026-07-06) `.gitignore` excludes `data/*`, `converted_data/*`, `data_lerobot/*`, and most of `assets/*` except a few tracked reference objects. (Source: .gitignore:4-6,29-31; README.md:82)

Upstream sources (not local): DOMINO dataset at HuggingFace `h-embodvis/DOMINO` / ModelScope `H-EmbodVis/DOMINO`; PUMA checkpoint at HuggingFace `H-EmbodVis/PUMA`; expected PUMA checkpoint layout under `policy/PUMA/playground/Pretrained_models/`. (External: upstream DOMINO/PUMA docs — removed from the local README in the 2026-07-06 RoboDyna-only rewrite; consult policy/PUMA/README.md or the upstream repo if needed)

## Working Setup on ruiheng-4080 (verified 2026-07-03; install script validated from scratch 2026-07-06)

The core sim pipeline is INSTALLED AND VERIFIED on ruiheng-4080: conda env `robodyna` at `/home/ruiheng/miniconda3/envs/robodyna` (Python 3.10.20), smoke test passed — `bash collect_data.sh cook_meat debug_dynamic 1` collected 8/8 episodes with HDF5+mp4 under `data/cook_meat/debug_dynamic/` and a LeRobot v2.1 export under `data_lerobot/domino_suite/`. (Source: collect_smoke_log4.txt, final lines)

**`script/_install.sh` is now SELF-SUFFICIENT and validated from scratch (2026-07-06):** a fresh env (`robodyna-test`, Python 3.10) provisioned by a single `bash script/_install.sh` run passed the full import matrix, renderer test, and a 2-episode collection. The script now encodes everything below: sapien/toppra/pyarrow via requirements.txt, pytorch3d `--no-build-isolation`, curobo `git checkout v0.7.8`, warp-lang/scipy re-pins, and retry loops (see next paragraph). (Source: install_test_log2.txt; script/_install.sh)

**Flaky `torch.utils.cpp_extension` crash on this machine (IMPORTANT):** importing `torch.utils.cpp_extension` (done at pip's metadata stage by both the pytorch3d and curobo builds) crashes nondeterministically (~40-60% of attempts) with sre/heap-corruption symptoms (`ValueError: not enough values to unpack (expected 92, got 2)`, `malloc(): invalid next size`, segfault) — torch 2.4.1's giant hipify regex triggering a CPython 3.10.20 bug. The crash is fail-stop at the metadata stage, so `_install.sh` wraps both builds in up-to-10x retry loops with post-install import checks. Runtime is unaffected (compiled `.so` files don't re-import `cpp_extension`). Measured: attempt-1 run had BOTH builds fail; attempt-2 run needed 1 curobo retry, 0 pytorch3d retries. (Source: script/_install.sh:11-15,67-71; install_test_log.txt:738-818; install_test_log2.txt:440)

**Pin set that works (deviations from upstream, all required):**
- `sapien==3.0.3` — fixed in requirements.txt line 4 (upstream wrongly pinned 3.0.0b1; README mandates 3.0.3). (Source: script/requirements.txt:4; README.md:7-12)
- `toppra` — added to requirements.txt line 27; imported by `envs/_base_task.py:9`, `envs/robot/robot.py:6`, `envs/robot/planner.py:14`, `script/test_render.py:18` but missing upstream. (Source: script/requirements.txt:27)
- `pyarrow` — added to requirements.txt line 28; required by `envs/utils/lerobot_v21.py:21-22`; without it the collector SILENTLY skips LeRobot export ("LeRobot export init FAILED, skipping"). (Source: script/requirements.txt:28; envs/utils/lerobot_v21.py:21-22)
- **curobo v0.7.8** (commit `d64c4b0`) — curobo HEAD is the restructured 0.8.x API (`curobo/_src/`), incompatible with RoboDyna's classic-API imports (`envs/robot/planner.py:21-33`: `curobo.types.math`, `curobo.types.robot`, `curobo.wrap.reacher.motion_gen`, `curobo.util.logger`). v0.7.8 is the newest tag with the classic layout. NOW AUTOMATED: `_install.sh` pins the checkout itself. (Source: envs/robot/planner.py:21-33; script/_install.sh:62-65)
- **warp-lang 1.4.2** — curobo 0.7.8 calls `wp.torch.device_from_torch()`, which breaks on warp-lang 1.14 (`module 'warp' has no attribute 'torch'`). Symptom cascade: seed 0 fails in `CuroboPlanner.__init__`, then every later seed fails with `'Robot' object has no attribute 'left_planner'` and the collector seed-searches forever with 0 saves. NOW AUTOMATED: `_install.sh` re-pins after the curobo step. (Source: envs/curobo/src/curobo/geom/sdf/world_mesh.py:67; collect_smoke_log2.txt; script/_install.sh:74-78)
- `scipy==1.10.1` — curobo's editable install silently upgrades scipy past the project pin. NOW AUTOMATED: re-pinned by `_install.sh` alongside warp-lang. (Source: script/requirements.txt:5; script/_install.sh:74-78)
- `--no-build-isolation` added to the pytorch3d line in `script/_install.sh:8` (otherwise its build env can't see torch and fails). (Source: script/_install.sh:8)
- torch resolves to `2.4.1+cu121` (bare pin); extensions compiled with nvcc 12.4 — works. GPU 1 used for all runs.

**Operational gotchas discovered:**
- `script/_install.sh` has NO `set -e` — it exits 0 even when internal steps fail. Never trust its exit code; verify each import individually. (Source: script/_install.sh)
- The collector never aborts on systematic failure — it seed-searches indefinitely. If early seeds all fail with the same error, kill it and read the FIRST seed's error (later seeds show the misleading `left_planner` symptom). (Source: script/collect_data.py, collect_smoke_log2.txt)
- Assets (16G) come from the public HF dataset `TianxingChen/RoboTwin2.0` via `script/_download_assets.sh` (no auth); zips are deleted after unzip. (Source: assets/_download.py)
- Non-fatal warning during collection: `description/objects_description/200_steak/base0.json` missing (episode instructions still generated). (Source: collect_smoke_log4.txt)

## Known Issues / TODO

**IMPORTANT for future work on this machine — read before running anything.**

1. **SAPIEN version conflict — RESOLVED 2026-07-03.** requirements.txt now pins `sapien==3.0.3` directly (see Working Setup above). (Source: script/requirements.txt:4)

2. **Hardcoded `/shared_work/markhsp/DOMINO` paths.** Several files hardcode a DIFFERENT machine's path, `/shared_work/markhsp/DOMINO` (NOT `/shared_work/physical_intelligence/benchmarks/RoboDyna`, the actual location on THIS machine). These will NOT resolve here and must be fixed/parameterized before use:
   - `repro_one.py` — `os.chdir("/shared_work/markhsp/DOMINO")` + matching `sys.path` inserts. (Source: repro_one.py:4-6)
   - `build_domino_aarch64.sh` — `CONDA=/shared_work/markhsp/miniforge3`, `ENV=/shared_work/markhsp/envs/domino`, `REPO=/shared_work/markhsp/DOMINO`, `PIP_CACHE_DIR`, `HF_HOME`, wheel paths under `/shared_work/markhsp/wheels` and `/shared_work/jack/wheels`. (Source: build_domino_aarch64.sh:9-14,27-29)
   - `collect_demos.sbatch` — Slurm log path `/shared_work/markhsp/DOMINO/logs/collect-%j.out` and `cd /shared_work/markhsp/DOMINO`. (Source: collect_demos.sbatch:5,20)
   - `envs/catch_ramp_ball.py` — debug image dump hardcoded to `/shared_work/markhsp/DOMINO/preview_out/crb_{n}.png`. (Source: envs/catch_ramp_ball.py:261)
   - `.claude/skills/SAPIEN-task-creator/SKILL.md` + `scripts/validate_asset.py` + `scripts/integrate_object.py` — all default to `/shared_work/markhsp/DOMINO` repo root and conda env `/shared_work/markhsp/envs/domino`. (Source: .claude/skills/SAPIEN-task-creator/SKILL.md:5,18-19,28-29; scripts/validate_asset.py:27; scripts/integrate_object.py:43)
   - By contrast, "normal" pipeline paths are relative: `task_config/_config_template.yml` sets `save_path: ./data`. (Source: task_config/_config_template.yml:32)

3. **Uninitialized pi0/pi05 submodules.** `third_party/aloha` and `third_party/libero` are declared in `.gitmodules` but `git submodule status` is empty. (Source: .gitmodules; git submodule status)

4. **`assets/` mostly gitignored.** A clean clone needs a separate asset drop to be fully runnable. (Source: README.md:120-121)

5. **No verified conda env yet.** This is a brand-new project (initial setup only). No conda env has been created or tested on this machine for RoboDyna's core sim pipeline — do NOT assume any env name exists. (Source: project status / this clone)
