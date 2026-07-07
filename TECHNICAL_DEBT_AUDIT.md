# RoboDyna Technical Debt Report

Author: Rui Heng Yang

Synthesized from 6 independent read-only audit passes (S2-S7), one targeted conflict-resolution check, and the 2026-07-07 Codex follow-up additions. No code fixes were applied during the audit; this document is a progress tracker for the findings below.

---

## How to Use This Doc

Checkboxes are literal progress trackers: check one off only when that specific item has actually been fixed, not because a related script changed elsewhere. A working end-to-end run is useful evidence that the basic pipeline works, but it does not close latent robustness findings unless their underlying failure mode was fixed. File:line citations are the source of truth for each item.

## Table of Contents

Counts are current checklist rows counted from this file. The vendored-code section is table-only and has no checklist rows.

| Section | Checklist Count |
|---|---:|
| Install & Build Tooling | 28 |
| README.md | 7 |
| Hardcoded Paths, Dead Code & Architecture | 22 |
| Testing, CI, Security & Git Hygiene | 11 |
| Vendored Third-Party Code | 0 |
| Docs & Config Consistency | 11 |
| Codex Follow-Up Additions | 5 |
| **Total checklist rows** | **84** |

---

## Executive Summary

RoboDyna's core simulation and collection path has been shown to work, but the repo still carries substantial reproducibility, install, documentation, and maintenance debt. The highest-risk open items are fragile shell/install scripts, missing or incorrect dependency declarations, hardcoded machine paths, an undocumented evaluation/policy stack, unvalidated LeRobot resume metadata behavior, and localized security/DoS hazards. The checked boxes are limited to superseded, refuted, or informational items; the install-robustness findings remain open. (Source: scripts/merge_lerobot_meta.py:1; Source: data/process_stuck.py:1; Source: envs/utils/lerobot_v21.py:114; Source: envs/utils/lerobot_v21.py:217)

### Prioritized Findings Table (Critical / High, sorted by severity)

| ID | Severity | Category | File:Line | Description |
|---|---|---|---|---|
| INST-1 | High | Install/Build | `script/_install.sh` (whole file) | No shebang, no `set -e`/`set -euo pipefail` — script always exits 0 regardless of internal failures |
| INST-2 | High | Install/Build | `script/_install.sh:2` | `pip install -r requirements.txt` result never checked |
| INST-9 | High | Install/Build | `requirements.txt` (no `cv2`/`opencv-python` entry) | `cv2` imported by 3 first-party files but not declared as a dependency anywhere; not a standard transitive dep |
| INST-10 | High | Install/Build | `requirements.txt:26` | `ffmpeg` PyPI package (defunct, ~2015, no CLI binary) pinned but 3 first-party files `subprocess.Popen(["ffmpeg",...])`; project already knows the correct fix (`imageio-ffmpeg` + symlink) in `build_domino_aarch64.sh:65-67` but doesn't apply it on the primary x86_64 path |
| DEAD-1 | High | Dead Code / Install | `collect_data.sh:7` | Documented primary workflow calls `./script/.update_path.sh`, which does not exist anywhere in repo history; output redirected to `/dev/null`, no `set -e` — every invocation silently no-ops with zero trace |
| PATH-1 | High | Hardcoded Paths | `repro_one.py:4-6` | `os.chdir("/shared_work/markhsp/DOMINO")` + 2 `sys.path.insert` calls hardcode a foreign machine path; script unusable from any other checkout, including current one |
| PATH-2 | High | Hardcoded Paths | `build_domino_aarch64.sh:9-14,27-29` | Hardcodes conda/env/repo/pip-cache/hf-cache/wheel-cache paths under `/shared_work/markhsp/...` plus a second foreign user path `/shared_work/jack/wheels/...`; none resolve on current host |
| PATH-3 | High | Hardcoded Paths | `collect_demos.sbatch:6,14,15,20,24` | Slurm log path, conda source/activate, and `cd` all hardcode `/shared_work/markhsp/DOMINO` |
| README-1 | **SUPERSEDED** (was High) | README | `README.md:16-20` (current) | **STALE finding, downgraded.** Original audit read a version of README.md that lacked install instructions. The current on-disk README.md (uncommitted local edit) already has an **Install:** subsection at lines 16-20 running `bash script/_install.sh` and `bash script/_download_assets.sh`. See corrected write-up below. |
| README-2 | High | README | `README.md` (whole file, 147 lines total) | Evaluation and the entire 13-subproject policy stack are undocumented — reconfirmed against the current file: no "eval" or "policy" section/heading anywhere in README.md |
| ARCH-4 | High | Architecture | `policy/Your_Policy/eval_double_env.sh:38-48` | Server-launch command is missing a trailing `&`, so the foreground `listen()`/`while True:` server call blocks forever; `SERVER_PID=$!` (line 50) and entire client-launch/cleanup section (lines 57-72) are unreachable in practice |
| LERO-1 | High | LeRobot Export | `envs/utils/lerobot_v21.py:114-125,217-237`; `script/collect_data.py:218-237` | Resumed LeRobot export can silently desynchronize data files from metadata: existing parquet/video files remain on disk, but the per-task `_parts` metadata slice is rebuilt from only the current writer's in-memory episode buffers |
| SEC-1 | High | Security | `code_gen/task_generation.py:224`, `task_generation_mm.py:346-347`, `task_generation_simple.py:94`, `run_code.py:102-103` | `exec(f'now_task = {task_name}')` on unsanitized CLI argument — code injection |
| SEC-2 | Medium-High | Security | `script/policy_model_server.py:154-163,75` | Unauthenticated TCP RPC server: `getattr(self.model, cmd, None)` dispatches any attacker-supplied method name with client-supplied `obs`; no allowlist/auth/TLS. Mitigated today by hardcoded `host='localhost'`, but any co-located process on shared HPC nodes can connect |
| INST-11 | Medium-High | Install/Build | `script/_download_assets.sh` (whole file) | No shebang, no `set -e`; unzip/rm/download steps unchecked; partial/failed download proceeds into config-patching against an incomplete `assets/` tree with no final verification |

---

## Fix Progress (2026-07-07)

The only code change actually made to `script/_install.sh` so far is a new Python-3.10 conda-bootstrap block that auto-creates/activates a `robodyna-test` env when the active interpreter is not Python 3.10. None of the audited install-robustness findings were touched by that change: no shebang/no `set -e` (INST-1), unchecked `pip install` (INST-2), unchecked `sed` patches (INST-3), unchecked `git clone`/`cd` (INST-4), missing `cv2` (INST-9), defunct pip `ffmpeg` (INST-10), curobo tag mismatch (INST-7), duplicate `imageio` (INST-16), or the dead `.update_path.sh` reference (DEAD-1). User-confirmed on 2026-07-07: running the install script, creating a new conda env, and collecting data all work end-to-end in practice today. That is valuable evidence that the basic pipeline works on this environment, but it does not resolve the 13 individual install-robustness findings; they remain latent risks that happened not to trigger here, so their checkboxes stay unchecked.

## Verified vs Refuted Prior Assumptions

| Rumored lead | Outcome | Detail |
|---|---|---|
| `script/_install.sh` swallows failures / always "succeeds" | **CONFIRMED** | No shebang, no `set -e`; multiple unchecked steps (INST-1, INST-2, and several sub-steps below) |
| curobo pinned to a specific tag to avoid API breakage in HEAD | **CONFIRMED** | `_install.sh:63-65` pins `v0.7.8` with explanatory comment; verified against the actually-cloned `envs/curobo` — `git rev-parse v0.7.8` = `d64c4b005459db10c5dd867d8b30a87d5bda9bdb`, matches the tag exactly (also cross-checked in S6) |
| `_download_assets.sh` doesn't verify asset integrity | **CONFIRMED** | No shebang, no `set -e`; partial download can proceed silently into downstream config-baking |
| `collect_data.sh` documented entry point calls a broken/missing script | **CONFIRMED** | `collect_data.sh:7` → `./script/.update_path.sh` does not exist anywhere in repo history (confirmed via `ls`, repo-wide grep, and `git log --all`) — independently hit by 3 agents (S2, S3, S4) |
| Hardcoded `/shared_work/markhsp/DOMINO` paths present from prior deployment | **CONFIRMED** | `repro_one.py:4-6`, `build_domino_aarch64.sh:9-14,27-29`, `collect_demos.sbatch:6,14,15,20,24`, `envs/catch_ramp_ball.py:261`, `validate_asset.py:27`, `integrate_object.py:43`, `SKILL.md:18,28-30` — this is a wide, systemic pattern, not a single isolated file |
| No packaging metadata (no `setup.py`/`pyproject.toml`/lockfile at root) | **CONFIRMED** | Repo root has none; `requirements.txt` is the only first-party manifest; repo is not pip-installable |
| `requirements.txt` pins `sapien==3.0.0b1` (wrong/broken version) | **REFUTED** | True historically (original commit `91f3d6c`) but fixed same-day in commit `6e52857`; current `requirements.txt:4` correctly pins `sapien==3.0.3` |
| `toppra`/`pyarrow` are missing dependencies | **REFUTED** | Both present in `requirements.txt:27-28` and actively imported by first-party code (`_base_task.py:9`, `robot.py:6`, `planner.py:14`, `test_render.py:18,39` for toppra; `lerobot_v21.py:21-22` for pyarrow) |
| `collect_demos.sbatch` hardcodes a foreign path | **CHANGED** | Confirmed, but with more locations than the prior lead described and shifted line numbers (5→6, 20→24, plus two additional lines 14-15 not previously noted) |
| CLAUDE.md git-state description | **CHANGED** | Doc says "single local commit 91f3d6c, 1758 tracked files"; actual repo now has 3 commits (HEAD `608b73e`, same-day README rewrite) and 1759 tracked files |
| README/CLAUDE.md cross-references (section titles, line numbers) | **CHANGED** | README was rewritten 2026-07-06; several CLAUDE.md citations to README section titles/line numbers no longer match (see DOC-3/DOC-4 below) — underlying facts are still correct, only the citations drifted |
| README's setup story is only `pip install sapien==3.0.3`, no installer/downloader mentioned (README-1) | **SUPERSEDED** (cross-model review, re-verified against current file 2026-07-06) | `git status --short` shows `M README.md` (uncommitted, actively being edited); `git diff HEAD -- README.md` shows a new **Install:** block was added at lines 16-20 referencing `bash script/_install.sh` and `bash script/_download_assets.sh`. True when originally audited, false now — this report's own synthesis lagged a concurrent user edit. |

---

## Install & Build Tooling (28 findings)

Scope: `script/_install.sh`, `script/_download_assets.sh`, `script/requirements.txt`, `collect_data.sh`, `build_domino_aarch64.sh`, `collect_demos.sbatch`, `repro_one.py`. (Source: S2)

### `script/_install.sh`

- [x] **INST-1** `HIGH`: No shebang and no `set -e`/`set -euo pipefail`; the script can exit 0 after internal failures.
  - `script/_install.sh:1-2`; added `#!/usr/bin/env bash` + `set -e`. Verified the existing `pip install ... && break` retry loops (pytorch3d/curobo, needed for a flaky `cpp_extension` crash) are unaffected by bash's documented `set -e` exemption for commands before the final `&&`/`||` in a list. Cross-model review (Claude+Codex) caught a regression this introduced in the conda-not-found bootstrap path; fixed by switching to a `command -v conda` check before the `CONDA_BASE=$(...)` assignment. Also hardened the pytorch3d/curobo post-retry import checks (`|| echo` → `|| { echo; exit 1; }`) since those were the two steps most likely to silently "succeed" under the old exit-0 behavior. **FIXED 2026-07-07.**

- [x] **INST-2** `HIGH`: `pip install -r script/requirements.txt` result is never checked.
  - `script/_install.sh:26`; fixed as a direct consequence of INST-1's `set -e` (now a bare simple command, aborts on failure). **FIXED 2026-07-07.**

- [x] **INST-3** `MEDIUM`: `pip show` locations and `sed` patches are unguarded.
  - `script/_install.sh`; `SAPIEN_LOCATION`/`MPLIB_LOCATION` now checked non-empty after `pip show`, target files checked to exist before `sed -i`, and both patches verified post-hoc against their specific patched-form string (not a generic substring) so a silent no-op can't pass. Cross-model review (Claude+Codex) both independently flagged the sapien post-check as too broad (`grep -q 'encoding="utf-8"'` could false-pass on an unrelated occurrence) vs. the mplib check being appropriately specific; tightened to match the mplib pattern. Verified idempotent (safe to re-run on an already-patched checkout) on synthetic fixtures, both fresh and second-run. **FIXED 2026-07-07.**

- [x] **INST-4** `MEDIUM`: `git clone` result is unchecked, followed by bare `cd curobo`.
  - `script/_install.sh`; a failing `git clone` was already fail-fast as a side effect of INST-1's `set -e` (verified empirically: a bad-URL clone aborts before `cd curobo` runs). The residual gap — a pre-existing but corrupted/partial `envs/curobo` directory (e.g. an interrupted prior clone) silently skipping re-clone and proceeding into a broken checkout — is now caught by an explicit `git -C curobo rev-parse --is-inside-work-tree` validity check before the clone step. Cross-model review: Claude critic found the first version of this guard (`[ ! -d curobo/.git ]`) too narrow — `git clone` creates `.git/` almost immediately, so a genuinely interrupted clone would usually still have a (partial) `.git/` dir and slip past a directory-presence check; both critics also flagged that a `.git` FILE (submodule/worktree layout) would false-positive on a directory-presence check. Replaced with the `rev-parse` check, which correctly fails on partial/corrupt `.git/` dirs and correctly accepts `.git`-as-file layouts. Verified against synthetic fixtures (empty dir, stray non-git dir, dir with an empty `.git/` subdir) and the real repo/`envs/curobo` (both pass as valid). **FIXED 2026-07-07.**

- [x] **INST-5** `LOW`: pytorch3d installs from unpinned GitHub HEAD inside a retry loop.
  - `script/_install.sh:11-15`; retry/import check only prints on failure and does not stop the script. **CONFIRMED-NEW informational. VERIFIED UNCHANGED 2026-07-07.**

- [x] **INST-6** `LOW`: curobo is pinned to `v0.7.8` with an explanatory API-risk comment.
  - `script/_install.sh:63-65`; `envs/curobo` resolves `v0.7.8` to `d64c4b005459db10c5dd867d8b30a87d5bda9bdb`. **CONFIRMED-matches-prior-lead. VERIFIED UNCHANGED 2026-07-07.**

- [ ] **INST-7** `MEDIUM`: `_install.sh` and `build_domino_aarch64.sh` pin different curobo tags without explanation.
  - `script/_install.sh:65` uses `v0.7.8`; `build_domino_aarch64.sh:48,53` uses `v0.7.7` (`build_domino_aarch64.sh:53`). **CONFIRMED-NEW. STILL-OPEN 2026-07-07.**

- [x] **INST-8** `LOW`: `warp-lang==1.4.2` and `scipy==1.10.1` are re-pinned after curobo install.
  - `script/_install.sh:74-78` targets the same `scipy==1.10.1` as `requirements.txt:5`. **CONFIRMED-matches-prior-lead. VERIFIED UNCHANGED 2026-07-07.**

### `script/_download_assets.sh`

- [x] **INST-11** `MEDIUM-HIGH`: No shebang or `set -e`; download/unzip/rm steps are unchecked.
  - `script/_download_assets.sh:1-2`; added `#!/usr/bin/env bash` + `set -e`. Verified safe by reading `assets/_download.py`, which unconditionally fetches all three zips (`background_texture`, `embodiments`, `objects`) via `snapshot_download(allow_patterns=[...])` — no conditional/optional-skip case exists, so `unzip` failing on any of them always indicates a genuine failure. Cross-model review (Claude+Codex) independently converged on one real gap: `unzip` had no `-o`, so a re-run on already-extracted assets would hit an interactive overwrite prompt that, in a non-interactive context (sbatch/nohup) and now under `set -e`, hard-aborts instead of the old silent no-op; added `-o` to all three `unzip` calls (no effect on a fresh/first run, verified functionally against real `unzip` with a synthetic overwrite test). **FIXED 2026-07-07.**

- [ ] **INST-12** `LOW-MEDIUM`: Asset config patching bakes an absolute path and can hang/fail non-interactively.
  - `script/_download_assets.sh:18`; `script/update_embodiment_config_path.py:23-76`, with `assets_path=os.getcwd()` at line 24 and `input()` fallback at lines 15-19. **CONFIRMED-NEW. STILL-OPEN 2026-07-07.**

- [ ] **INST-13** `LOW`: Asset downloader has no shebang and is not executable.
  - `script/_download_assets.sh` permissions `-rw-rw-r--`; README must invoke it as `bash script/_download_assets.sh`. **CONFIRMED-NEW. STILL-OPEN 2026-07-07.**

### `script/requirements.txt`

- [x] **INST-14** `N/A (REFUTED)`: `sapien==3.0.3` is currently correct.
  - `script/requirements.txt:4`; historical `sapien==3.0.0b1` was fixed in commit `6e52857`. **REFUTED-prior-lead-false.**

- [x] **INST-15** `N/A (REFUTED)`: `toppra` and `pyarrow` are present and used.
  - `requirements.txt:27-28`; toppra imports at `envs/_base_task.py:9`, `envs/robot/robot.py:6`, `envs/robot/planner.py:14`, `script/test_render.py:18,39`; pyarrow imports at `envs/utils/lerobot_v21.py:21-22`. **REFUTED-prior-lead-false.**

- [ ] **INST-16** `MEDIUM`: `imageio` appears twice, once pinned and once unpinned.
  - `script/requirements.txt:10,22`; both entries remain. **CONFIRMED-NEW. STILL-OPEN 2026-07-07.**

- [ ] **INST-9** `HIGH`: `cv2`/`opencv-python` is absent despite first-party imports.
  - Imports at `envs/camera/camera.py:8`, `envs/utils/images_to_video.py:1`, `envs/utils/pkl2hdf5.py:4,80`; no `cv2`/`opencv-python` entry in `script/requirements.txt`. **CONFIRMED-NEW. STILL-OPEN 2026-07-07.**

- [ ] **INST-10** `HIGH`: `ffmpeg` is pinned as a defunct pip package, while code shells out to an ffmpeg CLI.
  - `script/requirements.txt:26`; CLI calls at `envs/utils/images_to_video.py:18-20`, `script/eval_policy.py:340-342`, `script/eval_policy_client.py:476-478`; correct `imageio-ffmpeg`+symlink pattern exists only in `build_domino_aarch64.sh:65-67`. **CONFIRMED-NEW. STILL-OPEN 2026-07-07.**

- [ ] **INST-17** `MEDIUM`: Legacy `azure==4.0.0` is pinned beside modern `azure-ai-inference`.
  - `script/requirements.txt:17-18`; `description/utils/agent.py:5-7` imports `azure.ai.inference`. **CONFIRMED-NEW.**

- [ ] **INST-18** `MEDIUM`: Corrected dependency count: 27 non-comment lines, 15 unpinned.
  - Unpinned list from current `script/requirements.txt`: `torchvision`, `pydantic`, `zarr`, `openai`, `h5py`, `azure-ai-inference`, `wandb`, `moviepy`, duplicate `imageio`, `termcolor`, `av`, `matplotlib`, `ffmpeg`, `toppra`, `pyarrow`. **CONFIRMED-NEW, count corrected. STILL-OPEN 2026-07-07.**

- [ ] **INST-19** `INFORMATIONAL`: `scipy==1.10.1` and `mplib==0.2.1` align with install expectations.
  - `script/requirements.txt:5-6`; `_install.sh` re-pins scipy and patches against the `mplib==0.2.1` planner layout. **CONFIRMED.**

- [ ] **INST-20** `LOW`: No explicit `numpy` pin in the primary requirements.
  - `script/requirements.txt`; `build_domino_aarch64.sh:24` treats `numpy<2` as load-bearing while primary requirements rely on scipy's transitive constraint. **CONFIRMED-NEW.**

- [ ] **INST-21** `LOW`: PIL/Pillow is imported without an explicit pin.
  - Imports at `envs/catch_ramp_ball.py:258`, `script/create_messy_data.py:19`; likely satisfied transitively. **CONFIRMED-NEW minor.**

### `collect_data.sh` / `build_domino_aarch64.sh` / `collect_demos.sbatch` / `repro_one.py`

- [ ] **DEAD-1** `HIGH`: `collect_data.sh:7` references missing `./script/.update_path.sh`.
  - `collect_data.sh:7`; direct `ls`, repo grep, and `git log --all -- script/.update_path.sh` found no file/history; output is redirected to `/dev/null 2>&1` and no `set -e` exists. **CONFIRMED-matches-prior-lead. STILL-OPEN 2026-07-07.**

- [ ] **INST-22** `MEDIUM`: `collect_data.sh` has no `set -e` and deletes cache unconditionally.
  - `collect_data.sh:13`; no validation of `$1`/`$2`/`$3`, and `rm -rf data/${task_name}/${task_config}/.cache` runs regardless of collection outcome. **CONFIRMED-NEW. STILL-OPEN 2026-07-07.**

- [ ] **PATH-2** `HIGH`: `build_domino_aarch64.sh` hardcodes foreign absolute paths.
  - `build_domino_aarch64.sh:9-14,27-29`; paths under `/shared_work/markhsp/...` and `/shared_work/jack/wheels` do not resolve here; script does have `set -eo pipefail` at line 7. **CONFIRMED-matches-prior-lead.**

- [ ] **PATH-3** `HIGH`: `collect_demos.sbatch` hardcodes the old DOMINO layout.
  - `collect_demos.sbatch:6,14,15,20,24`; Slurm log path, conda source/activate, and `cd` point to `/shared_work/markhsp/DOMINO`; line-number details changed from earlier pass. **CONFIRMED-matches-prior-lead.**

- [ ] **PATH-1** `HIGH`: `repro_one.py` changes into a foreign checkout and inserts that path into `sys.path`.
  - `repro_one.py:4-6`; unusable as-is from this checkout. **CONFIRMED-matches-prior-lead.**

- [ ] **INST-23** `INFORMATIONAL`: Root packaging metadata is absent.
  - Repo root has no `setup.py`, `pyproject.toml`, `environment.yml`, or `Dockerfile`; `script/requirements.txt` is the only first-party dependency manifest. **CONFIRMED-matches-prior-lead.**

- [ ] **INST-24** `MEDIUM`: The repo is not pip-installable.
  - No package metadata means no `pip install -e .`, console entry points, or lockfile despite sed-based patches, retry builds, and many unpinned deps. **CONFIRMED-NEW.**

---

## README.md (7 findings)

Source: S3, cross-referenced against S7 where overlapping.

- [x] **README-1** `SUPERSEDED`: The original "no install instructions" finding is stale.
  - Current `README.md:16-20` has an Install block with `bash script/_install.sh` and `bash script/_download_assets.sh`; `git status --short` shows README is locally modified. **SUPERSEDED; downgraded from High.**

- [ ] **README-2** `HIGH`: Evaluation and the 13-subproject policy stack are undocumented.
  - Current README has no eval/policy section; grep only finds SAPIEN policy-evaluable wording at `README.md:11-13` and a task adjective, not workflow docs for `policy/*` or `script/eval_policy*.py`/`policy_model_server.py`. **CONFIRMED.**

- [ ] **README-3** `MEDIUM`: Stale asset example `202_bread_toast` does not exist.
  - `README.md:125-127`, with `202_bread_toast` at line 126; `assets/objects/` has `075_bread` and `076_breadbasket`, not `202_bread_toast`. **CONFIRMED; line citation corrected.**

- [ ] **README-4** `LOW`: Documented `collect_data.sh` entry point invokes a missing helper.
  - `README.md:35` and `README.md:101` document the flow; `collect_data.sh:7` -> `./script/.update_path.sh` missing; non-fatal only because output is redirected and no `set -e` exists. **CONFIRMED; see DEAD-1.**

- [ ] **README-5** `LOW`: README output-path wording is imprecise.
  - `README.md:37`; actual HDF5 path is `data/<task>/<config>/data/episodeN.hdf5`, from `script/collect_data.py:108,219` plus `envs/_base_task.py:930`, not just under `data/<task>/<config>/`. **CONFIRMED.**

- [ ] **README-6** `LOW`: README lacks a prerequisites/system-requirements and troubleshooting section.
  - Current README mentions `VK_ICD_FILENAMES` inline but does not state Python 3.10/GPU/Vulkan requirements or setup failure troubleshooting. **CONFIRMED.**

- [ ] **README-7** `LOW`: CLAUDE.md references stale README headings and line numbers.
  - CLAUDE.md cites `README.md:56-64` and `README.md:64`; current README task table is `README.md:42-48`, `stab_moving_target` is `README.md:48`, and heading is `README.md:40` "Tasks". **CONFIRMED; see DOC-3/DOC-4.**

**Verified-correct README claims (no issue):** aarch64 wheel URL matches `build_domino_aarch64.sh:30`; codeless registry pattern matches `collect_data.py:24-28`; shared configs `demo_dynamic.yml`/`debug_dynamic.yml` both have a `task_args` block; `kwargs.get` pattern matches `cook_meat.py:26`; gitignore claim matches `.gitignore:4,6,22`; skill layout matches `.claude/skills/SAPIEN-task-creator/`; all 17 demo GIFs exist; `sapien==3.0.3` matches `requirements.txt:4`.

---

## Hardcoded Paths, Dead Code & Architecture (22 checklist rows / 24 findings)

Source: S4, with overlaps from S2 and S5 merged and noted.

### Hardcoded / Machine-Specific Paths

- [ ] **PATH-1, PATH-2, PATH-3** `HIGH`: See Install & Build Tooling section above.
  - Merged cross-reference row; canonical evidence is `repro_one.py:4-6`, `build_domino_aarch64.sh:9-14,27-29`, and `collect_demos.sbatch:6,14,15,20,24`. **CONFIRMED.**

- [ ] **PATH-4** `LOW`: Debug-image dump path is hardcoded to the old DOMINO tree.
  - `envs/catch_ramp_ball.py:261`; gated behind `CRB_RENDER`. **CONFIRMED-matches-prior-lead.**

- [ ] **PATH-5** `MEDIUM`: Argparse defaults hardcode a separate `hfang` dev-machine path.
  - `script/extract_dynamic_gt.py:421,435`; applies if CLI flags are omitted. **NEW.**

- [ ] **PATH-6** `MEDIUM`: Skill helper defaults also point to `/shared_work/markhsp/DOMINO`.
  - `validate_asset.py:27`, `integrate_object.py:43`, consistent with CLAUDE.md:122-126. **NEW.**

### Dead / Broken Code

- [ ] **DEAD-1** `HIGH`: See Install & Build Tooling section above.
  - Merged cross-reference row; canonical evidence is `collect_data.sh:7` plus missing `script/.update_path.sh`. **CONFIRMED.**

- [ ] **DEAD-2** `MEDIUM`: `envs/robot/ik.py` is an orphaned stub.
  - `envs/robot/ik.py` whole file is one `# TODO` line with no first-party references. **NEW.**

- [ ] **DEAD-3** `LOW`: Invalid regex escape sequence may become a future SyntaxError.
  - `script/create_object_data.py:779` contains non-raw string `'_(\d+)<(.*?)>'`. **NEW.**

- [ ] **DEAD-4** `MEDIUM`: `script/create_messy_data.py` contains multiple large commented-out code blocks.
  - Blocks at `script/create_messy_data.py:564-590,881-900,116-124,131-141,374-392,444-453,538-546,610-616,838-855,1004-1067`. **NEW.**

- [ ] **DEAD-5** `LOW`: Sensor-camera construction block is commented out inside live setup loop.
  - `envs/camera/camera.py:194-204`. **NEW.**

- [ ] **DEAD-6** `LOW`: LLM prompt contains stale/superseded API-doc dict entries.
  - `code_gen/prompt.py:47-63`; dead `move_to_pose` and stale `move_by_displacement` signature risk drift from `Base_Task` API. **NEW.**

- [ ] **DEAD-7** `MEDIUM`: `description/objects_description/200_steak/` is missing.
  - `envs/cook_meat.py:79,119,180,233` uses object `200_steak`; runtime resolves against `assets/objects/200_steak/`, so no current functional break, but the description-tree gap is real. **CONFIRMED.**

### Error Handling & Architecture Smells

- [ ] **ARCH-1** `MEDIUM`: Bare `except:` around task lookup masks real instantiation failures.
  - `script/collect_data.py:29`, `script/eval_policy_client.py:80`, `script/eval_policy.py:35`. **NEW.**

- [ ] **ARCH-2** `MEDIUM`: `eval()` on CLI overrides is covered by SEC-3.
  - Merged into Security section for full per-site nuance at `policy_model_server.py:262`, `eval_policy_client.py:616`, and `eval_policy.py:477`. **NEW.**

- [ ] **ARCH-3** `LOW-MEDIUM`: First-party scope has 30 bare `except:` clauses.
  - Examples include `eval_policy_client.py:80,194,617`, `test_render.py:52`, `place_phone_stand.py:188`, `policy_model_server.py:103`, `create_messy_data.py:368,780`, `add_annotation.py:251`, `eval_policy.py:35,478`, `collect_data.py:29,304,315,327,400`, `envs/robot/robot.py:342`, `envs/_base_task.py:162,977,979`, `envs/utils/create_actor.py:417,463,543,591`, `code_gen/task_generation.py:225`, `envs/utils/actor_utils.py:41`, `code_gen/task_generation_mm.py:103`, `envs/camera/camera.py:35`, `code_gen/test_gen_code.py:96,115`, `policy/Your_Policy/deploy_policy_double_env.py:5`. **NEW.**

- [ ] **ARCH-4** `HIGH`: `eval_double_env.sh` launches the server in the foreground forever.
  - `policy/Your_Policy/eval_double_env.sh:38-48`; `policy_model_server.py:89,111` listens forever, so `SERVER_PID=$!` at line 50 and client/cleanup lines 57-72 are unreachable; earlier line citation was corrected. **NEW, line citation corrected.**

- [ ] **ARCH-5** `LOW`: Placeholder policy deploy script has a bare `except:`/`pass` template.
  - `policy/Your_Policy/deploy_policy_double_env.py:2-5`. **NEW.**

- [ ] **ARCH-6** `MEDIUM`: Multiple first-party shell scripts are missing `set -e`.
  - `collect_data.sh`, `task_config/create_task_config.sh`, `description/gen_task_instruction_templates.sh`, `description/gen_episode_instructions.sh`, `description/gen_object_descriptions.sh`, `script/_download_assets.sh`, and `script/_install.sh`; only `build_domino_aarch64.sh` and `collect_demos.sbatch` have it. **NEW except INST-1/11 overlap.**

- [ ] **ARCH-7** `MEDIUM`: Two description-generation scripts lack shebangs and cwd setup.
  - `description/gen_episode_instructions.sh`, `description/gen_task_instruction_templates.sh`; they call `python utils/generate_*.py` relative to `description/`, so repo-root invocation fails. **NEW.**

- [ ] **ARCH-8** `MEDIUM`: Repo has 26 TODO/FIXME/HACK-style markers, clustered in the base task file.
  - Notable markers in `envs/_base_task.py` around `~1317`, `~1321`, `~3781`, `~3796`, `~3818`, including commented-out returns and gripper-control uncertainty. **NEW.**

- [ ] **ARCH-9** `LOW`: `envs/_base_task.py` is a 4081-line god-file.
  - Next-largest in-scope files are `script/create_object_data.py` at 1092 lines and `envs/robot/robot.py` at 729 lines. **NEW.**

- [ ] **ARCH-10** `MEDIUM`: `script/policy_model_server.py` has resource-exhaustion/DoS exposure distinct from SEC-2.
  - Unbounded daemon thread per connection at `script/policy_model_server.py:109-125,116-118`; unbounded client length prefix at `:127-180,137,140-147`; listen-socket timeout at `:88` does not apply to accepted `client_socket.recv()` at `:133`. **NEW.**

- [ ] **ARCH-11** `MEDIUM`: `script/collect_data.py` has unbounded retry/seed-search loops.
  - Main loop at `script/collect_data.py:135` catches broad exception at `:172` and never compares attempts to a cap; regeneration counter at `:335,413` is never read; known CLAUDE.md gotcha now traced to source. **NEW.**

---

## Testing, CI, Security & Git Hygiene (11 findings)

Source: S5. Confirmed absent: CI config files, real root tests, linter config, pipe-to-shell installs, non-HTTPS asset/model URLs, `subprocess(shell=True)`, hardcoded real secrets/API keys (`code_gen/gpt_agent.py:3-5` and `script/add_annotation.py:206` are placeholder strings; `description/utils/agent.py:12-15` reads `AZURE_API_KEY` from env properly), AWS-style key patterns, and `.env`/credentials files.

### Testing / CI / Lint

- [ ] **TEST-1** `MEDIUM`: No real automated tests exist in first-party scope.
  - `script/test_render.py` is a GPU/Vulkan renderer smoke class with prints/exit; `code_gen/test_gen_code.py` is a runtime helper/evaluation harness with no assertions. **NEW.**

- [ ] **TEST-2** `MEDIUM`: No CI pipeline exists.
  - No `.github/`, `.gitlab-ci.yml`, `.travis.yml`, `.circleci/`, `azure-pipelines*`, or `Jenkinsfile`. **CONFIRMED absent. NEW.**

- [ ] **TEST-3** `LOW`: No first-party linter/formatter config exists.
  - No `.flake8`, `.pylintrc`, ruff/black `pyproject.toml`, `.pre-commit-config.yaml`, `setup.cfg`, or `tox.ini` under root, `script/`, `envs/` excluding curobo, `description/`, `code_gen/`, `task_config/`, or `policy/Your_Policy/`; vendored policy pyprojects are out of scope. **NEW.**

### Security

- [ ] **SEC-1** `HIGH`: Dev-tool scripts use `exec(f'now_task = {task_name}')` on unsanitized CLI args.
  - `code_gen/task_generation.py:224`, `task_generation_mm.py:346-347`, `task_generation_simple.py:94`, `run_code.py:102-103`; pattern from `run_code.py:100-104`; caveat: local CLI self-injection today, not a remote surface. **NEW.**

- [ ] **SEC-2** `MEDIUM-HIGH`: Unauthenticated TCP RPC server dispatches arbitrary model methods by client-selected name.
  - `script/policy_model_server.py:154-163` uses `getattr(self.model, cmd, None)` with client `obs`; `host='localhost'` at `:75` mitigates remote reachability, but co-located local processes can connect; `json_to_numpy` at `:63-70` also trusts dtype/shape. **NEW.**

- [ ] **SEC-3** `MEDIUM`: CLI override `eval()` exists at three sites, with one materially safer gated site.
  - `policy_model_server.py:262` gates `eval(val)` behind `val.isnumeric()`; `eval_policy_client.py:616` and `eval_policy.py:477` evaluate override strings verbatim inside bare try/except. **NEW; merges S4 ARCH-2.**

- [ ] **SEC-4** `LOW-MEDIUM`: `pickle.load` is used on pipeline-written files without schema validation.
  - `extract_dynamic_gt.py:208,589`, `envs/_base_task.py:916`, `envs/utils/pkl2hdf5.py:50`, `envs/utils/lerobot_export.py:111`; normally trusted provenance, unsafe if tampered on shared filesystem. **NEW.**

### Git Hygiene

- [ ] **GIT-1** `LOW-MEDIUM`: `.gitignore` has practical coverage gaps.
  - Present coverage includes `__pycache__/`, data/assets/models/logs patterns, root `/*.txt`, etc.; missing `.venv`/`venv/`, `.env`, generic `.cache`, `*.pyc`, and model-weight extension patterns like `*.pth`, `*.ckpt`, `*.safetensors`. **NEW.**

- [x] **GIT-2** `N/A (INFORMATIONAL)`: Root `*_log.txt` files are ignored, not tracked.
  - Verified by `git ls-files`, `git status --short`, `git log --diff-filter=A -- '*.txt'`, and `.gitignore:41`/`git check-ignore -v`. **INFORMATIONAL, not a hygiene violation.**

- [ ] **GIT-3** `LOW`: `assets/__MACOSX/` extraction artifact exists but is ignored/untracked.
  - `git status --short --ignored=matching` shows `!! assets/__MACOSX/`; `git ls-files` shows nothing. **NEW.**

- [x] **GIT-4** `N/A (INFORMATIONAL)`: Single-commit history note is superseded.
  - Earlier CLAUDE.md state said one commit `91f3d6c`; S7 later found 3 commits after subsequent work, see DOC-1. **SUPERSEDED informational.**

---

## Vendored Third-Party Code — Shallow Pass (12 subprojects)

Scope: `envs/curobo/` and `policy/*`. Per user decision, this was a shallow scan only: license/README/version-pin presence, not a deep code audit. (Source: S6)

| Directory | LICENSE | README | Version-pin flags | Severity | Size |
|---|---|---|---|---|---|
| `envs/curobo/` | Yes (NVIDIA proprietary "NVIDIA License") | Yes + CHANGELOG.md | None vs RoboDyna stack directly; own `pyproject.toml`/`setup.py` not inspected | Low | 326M / 763 files |
| `policy/ACT/` | Yes, MIT (Tony Z. Zhao 2023) | None found | `conda_env.yaml` pins py3.9/torch2.0.0/pytorch-cuda11.8 — differs from RoboDyna base, but ACT gets its own conda env (`aloha`), not a real conflict | Low | 276K / 34 files |
| `policy/DexVLA/` | Yes, MIT (Tony Z Zhao 2023 — reused header, likely copy-paste from ACT lineage, not DexVLA's actual authors) | Yes | Two inconsistent dep manifests: `conda_env.yaml` (py3.9/torch2.0.0) vs `Eval_Tiny_DexVLA_requirements.txt`/`environment.yml` (py3.10.13/torch2.4.1) — internally inconsistent, not just vs RoboDyna | Low-Medium | 716K / 67 files |
| `policy/DP/` | **No** | None | `pyproject.toml` requires-python≥3.8, broad, no conflict | Low | 564K / 73 files |
| `policy/DP3/` | **No** (checked `3D-Diffusion-Policy/` subdir too) | None | Nested `setup.py` bare stub, no version pins | Low | 352K / 45 files |
| `policy/GO1/` | **No** | Yes | `requirements.txt` has no torch/python pin at all — too sparse to conflict | Low | 60K / 10 files |
| `policy/LLaVA-VLA/` | **No** | **No** (no original_README, no requirements/setup/pyproject anywhere) | Can't assess — no manifest exists | Low-Medium | 436K / 39 files (weakest vendoring hygiene) |
| `policy/openvla-oft/` | Yes, MIT (Kim, Finn, Liang 2025) | Yes (original_README.md, original_ALOHA.md, original_SETUP.md — renamed to avoid clobbering RoboDyna's docs) | `pyproject.toml` requires-python≥3.8 (up to 3.10 classifiers), no obvious conflict | Low | 1.5M / 128 files (best-documented) |
| `policy/pi0/` | Yes, Apache 2.0 | None found (has `docs/` but no top-level README) | `pyproject.toml` requires-python≥3.11, torch≥2.5.1 — conflicts with RoboDyna base (py3.10, torch==2.4.1); has own `uv.lock`, isolated env by design; dangling `.gitmodules` referencing `third_party/aloha` and `third_party/libero`, neither dir exists on disk, no own `.git` | Low | 1.6M / 122 files |
| `policy/pi05/` | Yes, Apache 2.0 | None | Same as pi0 — requires-python≥3.11/torch≥2.5.1 vs base py3.10/torch2.4.1; same dangling `.gitmodules` issue | Low | 1.9M / 136 files |
| `policy/PUMA/` | **No** | Yes | `requirements.txt` pins torchvision==0.21.0; `pyproject.toml` requires-python≥3.10; no direct torch pin visible | Low | 1.8M / 163 files |
| `policy/RDT/` | **No** | None | `requirements.txt`: transformers==4.41.0, diffusers==0.27.2, deepspeed==0.14.2, numpy<2.0 — no explicit torch/python pin | Low | 1.2M / 60 files (missing LICENSE + README both flagged) |
| `policy/TinyVLA/` | Yes, MIT (Tony Z Zhao 2023, same reused header as ACT/DexVLA) | Yes | Same dual/inconsistent-manifest pattern as DexVLA; contains files literally named `Eval_Tiny_DexVLA_*` inside `TinyVLA/` (cross-contamination naming artifact confirming shared lineage/copy-paste) | Low-Medium | 312K / 37 files |

**Curobo commit vs install-script cross-check:** no mismatch. Vendored `envs/curobo/.git` is detached at `d64c4b005459db10c5dd867d8b30a87d5bda9bdb`, and `git describe --tags` resolves to `v0.7.8`; `script/_install.sh:57-64` explicitly clones NVlabs/curobo and checks out `v0.7.8`.

**pi0/pi05 dangling submodule references:** confirmed vendoring-hygiene noise, low severity, inherited from upstream. Both `pi0/.gitmodules` and `pi05/.gitmodules` declare `third_party/aloha` and `third_party/libero`, neither exists on disk, and neither subproject has its own `.git`.

**Cross-cutting observations:** Missing LICENSE in 6 of 12 policy dirs (DP, DP3, GO1, LLaVA-VLA, PUMA, RDT). Missing README in 7 of 12 (ACT, DP, DP3, LLaVA-VLA, pi0, pi05, RDT). LLaVA-VLA has no dependency manifest. pi0/pi05 intentionally conflict with RoboDyna's base env because they use their own `uv.lock` envs. No Critical/High findings in the shallow vendored pass.

---

## Docs & Config Consistency (11 findings)

Source: S7. Spot-verified: `sapien==3.0.3` at `requirements.txt:4`; `toppra` at `requirements.txt:27` and imports in `_base_task.py:9`, `robot.py:6`, `planner.py:14`, `test_render.py:18`; `pyarrow` at `requirements.txt:28` and `lerobot_v21.py:21-22`; curobo classic imports at `planner.py:21-33`; curobo `v0.7.8` at `_install.sh:65`; warp/scipy re-pin at `_install.sh:78`; collector import/save path at `collect_data.py:25,27,108`; hardcoded path examples at `repro_one.py:4-6`, `catch_ramp_ball.py:261`, `validate_asset.py:27`, `integrate_object.py:43`.

### CLAUDE.md Accuracy

- [ ] **DOC-1** `MEDIUM`: CLAUDE.md git-state section describes an old single-commit state.
  - CLAUDE.md says one commit `91f3d6c` and 1758 tracked files; actual repo has 3 commits, HEAD `608b73e`, and 1759 tracked files. **STALE.**

- [ ] **DOC-2** `MEDIUM`: CLAUDE.md cites a root `.gitmodules` that does not exist.
  - Root `ls .gitmodules` fails; only `policy/pi0/.gitmodules` and `policy/pi05/.gitmodules` exist, and root `git submodule status` is empty. **WRONG CITATION.**

- [ ] **DOC-3** `MEDIUM`: The "16 tasks" claim cites the wrong README row and misses the real source.
  - CLAUDE.md cites `README.md:66`, now a prototype row; real map is `envs/utils/lerobot_export.py:17-24`, whose comment at line 16 says 15-task suite despite 16 literal entries. **WRONG CITATION.**

- [ ] **DOC-4** `MEDIUM`: README task/status line pointers in CLAUDE.md are stale.
  - CLAUDE.md cites `README.md:56-64`, `:64`, and `:70`; current done table is `README.md:44-47`, `stab_moving_target` is `README.md:48`, registry text is `README.md:78-79`, and task-class shape is `README.md:81-84`. **WRONG CITATION.**

- [ ] **DOC-5** `LOW`: CLAUDE.md description-directory counts are off.
  - Actual counts: `description/objects_description` has 118 folders, not 120; `description/task_instruction` has 67 JSON files, not 68. **WRONG.**

### Config Consistency

- [ ] **DOC-6** `MEDIUM`: "All tasks run on ur5-wsg" is contradicted by active configs.
  - `demo_dynamic.yml:6` and `debug_dynamic.yml:5` use dual `ur5-wsg`; `demo_clean.yml:7`, `demo_clean_dynamic.yml:5`, `demo_randomized.yml:7`, and `demo_random_dynamic.yml:5` use `aloha-agilex`. **INCONSISTENT.**

- [ ] **DOC-7** `MEDIUM`: README says "done" means 100 episodes, but production config is 50.
  - `task_config/demo_dynamic.yml:2` has `episode_num:50`; CLAUDE.md:87 also reports 50 episodes, contradicting README's 100-episode definition. **INCONSISTENT.**

- [ ] **DOC-8** `LOW`: README lists absent object `202_bread_toast`.
  - Duplicate of README-3; `README.md:126` lists it, but `assets/objects/` lacks it. **CONFIRMED.**

### SKILL.md Accuracy

- [ ] **DOC-9** `LOW`: SKILL.md is written for old DOMINO paths, not RoboDyna.
  - `SKILL.md:4-5,18-19,28-30` names `/shared_work/markhsp/DOMINO` and `/shared_work/markhsp/envs/domino`; same issue appears in `validate_asset.py:27` and `integrate_object.py:43`. **STALE.**

- [ ] **DOC-10** `LOW`: SKILL.md references a missing DOMINO-benchmark skill.
  - `SKILL.md:220`; `.claude/skills/` contains only `SAPIEN-task-creator/`. **WRONG.**

### Licensing Hygiene

- [ ] **DOC-11** `LOW (INFORMATIONAL)`: Root Apache-2.0 LICENSE is coherent, with vendored licenses kept separately.
  - Root LICENSE is Apache 2.0; bundled subprojects with licenses include pi0, pi05, DexVLA, ACT, openvla-oft, TinyVLA, and `envs/curobo/`; several policy dirs lack top-level LICENSE, which is out of scope here. **OK, with note.**

---

## Codex Follow-Up Additions — 2026-07-07 (5 findings)

Scope: targeted follow-up over the previously identified first-party coverage gap (`scripts/merge_lerobot_meta.py`, `data/process_stuck.py`) plus cheap repo-wide pattern checks for debug residue and expanded-scope policy-script hazards. (Source: scripts/merge_lerobot_meta.py:1; Source: data/process_stuck.py:1)

- [ ] **LERO-1** `HIGH`: LeRobot writer resume metadata can silently desynchronize existing parquet/video files from rebuilt per-task metadata.
  - `LeRobotV21Writer.__init__` resumes from existing parquet row counts but starts `_ep_lines`, `_stats_lines`, and `_task_strings` empty at `envs/utils/lerobot_v21.py:114,123`; `close()` overwrites per-task metadata at `envs/utils/lerobot_v21.py:215,217,220,223,236`; collector resume/export path is `script/collect_data.py:218,222,233,235`; later merge propagation is `scripts/merge_lerobot_meta.py:25,37,52`. **NEW.**

- [ ] **LERO-2** `MEDIUM`: Fallback LeRobot `task_index` is not deterministic and can collide for tasks outside `SUITE_TASK_INDEX`.
  - Fallback context is `envs/utils/lerobot_export.py:75,76,77`, with `self.task_index = len(SUITE_TASK_INDEX) + abs(hash(self.task_name)) % 1000` exactly at `envs/utils/lerobot_export.py:77`; Python `hash()` is process-salted and the modulo bucket can collide; the same file says "15-task suite" while the literal map has 16 names at `envs/utils/lerobot_export.py:16,17,22`. **NEW.**

- [ ] **LERO-3** `LOW-MEDIUM`: `scripts/merge_lerobot_meta.py` lacks metadata consistency validation.
  - Reads and merges task slices while silently overwriting duplicate `task_strings` and not rejecting duplicate `episode_index` values at `scripts/merge_lerobot_meta.py:26,34,37,38,39`; adopts first non-null global fields without equality checks at `scripts/merge_lerobot_meta.py:44,45,46,47,52,63,71,72,77`. **NEW.**

- [ ] **DATA-1** `MEDIUM`: `data/process_stuck.py` is stale and dangerous for the current output layout.
  - Mutates `seed.txt` in place without bounds check/dry run/backup at `data/process_stuck.py:10,15,20,22,25`; deletes/renames old-layout `data/episodeN.pkl` at `data/process_stuck.py:12,28,32` while current outputs are `_traj_data/episodeN.pkl`, `data/episodeN.hdf5`, and `video/episodeN.mp4` at `envs/_base_task.py:893,914,930,931`; current collector repair path is `script/collect_data.py:432,443,448,464,466,470`. **NEW.**

- [ ] **DEAD-8** `LOW`: Debug breakpoint residue in `Base_Task.get_scene_contact()` can hang headless collection.
  - Contact iteration enters `pdb.set_trace()` before printing details at `envs/_base_task.py:1480,1481,1482,1483,1484,1485`. **NEW.**

**Expanded-scope follow-up:** `policy/*` remained shallow-scan scope in the original audit. One concrete policy-script hazard found during the follow-up: `policy/LLaVA-VLA/eval.sh` (Source: policy/LLaVA-VLA/eval.sh:9) still calls `/yourpath/RoboTwin/script/eval_policy.py` rather than the current RoboDyna checkout's relative `script/eval_policy.py`.

---

## Scope Note

First-party RoboDyna code (`script/`, `envs/` excluding `envs/curobo/`, `description/`, `code_gen/`, `task_config/`, root-level scripts and docs) was deep-audited across the original 6 passes. Vendored `policy/*` and `envs/curobo/` were shallow-scanned only per user decision. No code was modified during the original audit, synthesis, or conflict-resolution pass.

The 2026-07-06 coverage gap was `scripts/` (plural, distinct from audited `script/`) and `data/process_stuck.py`. A targeted 2026-07-07 Codex follow-up covered those files and added LERO-1, LERO-2, LERO-3, DATA-1, and DEAD-8 above. (Source: scripts/merge_lerobot_meta.py:1; Source: data/process_stuck.py:1)
