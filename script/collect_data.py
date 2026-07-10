"""Multiprocess data collector for RoboDyna (DOMINO / RoboTwin 2.0 fork).

Author: Rui Heng Yang

Rewrite of the original single-process collector: the seed-search, render, and
seed-regeneration phases now run across N parallel workers (``--num-workers``).

Seed partitioning guarantees workers NEVER attempt overlapping seeds: worker
``w`` of ``N`` only ever tries seeds ``s`` with ``s % N == w`` (e.g. with 2
workers, worker 0 tries even seeds and worker 1 odd seeds). This holds in both
the initial seed search and the regeneration phase.

Workers are separate *processes* (spawn start method), not Python threads:
SAPIEN/Vulkan renderer contexts, curobo/CUDA planners, and the GIL make
in-process threading unusable for this simulator.

Concurrency-safety map (why the pipeline splits the way it does):
- Per-episode outputs (``.cache/episode{i}/``, ``data/episode{i}.hdf5``,
  ``video/episode{i}.mp4``, ``_traj_data/episode{i}.pkl``) are keyed by a
  globally unique episode index, so workers write them directly.
- Shared files (``seed.txt``, ``scene_info.json``) and the combined LeRobot
  v2.1 dataset (one stateful writer) are written ONLY by the main process,
  fed by a message queue. A worker keeps an episode's ``.cache`` directory
  alive until the main process has exported it; main then deletes it.

Behavioral notes vs. the original collector:
- Episode index == trajectory index == HDF5 index (the original compacted
  indices on the fly; here failed render slots are re-filled in place by the
  regeneration phase, so numbering stays dense without a final reorder pass).
- LeRobot episodes are appended in episode *completion* order, which under
  parallelism is not necessarily HDF5 index order.
- Like the original, the collector never aborts on systematic failure — it
  seed-searches indefinitely. If early seeds all fail identically, kill it and
  read the first seed's error.
"""

import sys

sys.path.append("./")

import os
import json
import time
import shutil
import yaml
import importlib
from copy import deepcopy
from argparse import ArgumentParser
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch.multiprocessing as mp

from envs import *  # noqa: F401,F403 -- provides CONFIGS_PATH and UnStableError

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


# ======================================================================
# Small helpers
# ======================================================================


def class_decorator(task_name: str):
    """Instantiate the task class ``envs.<task_name>.<task_name>`` (codeless registry)."""
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except AttributeError:
        raise SystemExit("No such task")
    return env_instance


def get_embodiment_config(robot_file: str) -> Dict[str, Any]:
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def _first_seed(base: int, worker_id: int, num_workers: int) -> int:
    """Smallest seed >= base with seed % num_workers == worker_id."""
    return base + ((worker_id - base) % num_workers)


def _hdf5_path(save_path: str, idx: int) -> str:
    return os.path.join(save_path, "data", f"episode{idx}.hdf5")


def _cache_dir(save_path: str, idx: int) -> str:
    return os.path.join(save_path, ".cache", f"episode{idx}")


def _seed_file(save_path: str) -> str:
    return os.path.join(save_path, "seed.txt")


def _write_seed_file(save_path: str, seed_by_idx: Dict[int, int]) -> None:
    """Write the contiguous-from-0 prefix of seed_by_idx to seed.txt.

    Only the gap-free prefix is persisted so that a crash mid-collection never
    leaves a seed file whose flat ordering is silently shifted on resume.
    """
    seeds = []
    idx = 0
    while idx in seed_by_idx:
        seeds.append(seed_by_idx[idx])
        idx += 1
    with open(_seed_file(save_path), "w") as f:
        for seed in seeds:
            f.write("%s " % seed)


def _read_seed_file(save_path: str) -> List[int]:
    with open(_seed_file(save_path), "r") as f:
        return [int(s) for s in f.read().split()]


def _wprint(worker_id: int, msg: str) -> None:
    print(f"\033[90m[W{worker_id}]\033[0m {msg}", flush=True)


def _safe_close(task, clear_cache: bool = False) -> None:
    try:
        task.close_env(clear_cache=clear_cache)
    except Exception as e:
        print(f"close_env failed (ignored): {e}", flush=True)


# ======================================================================
# Worker processes
# ======================================================================


def seed_search_worker(
    worker_id: int,
    num_workers: int,
    task_name: str,
    args: Dict[str, Any],
    start_seed: int,
    target: int,
    counter,
    lock,
    queue,
) -> None:
    """Phase 1: search seeds ``start.., step num_workers`` in this worker's
    residue class until the shared success counter reaches ``target``.

    On each planning success the worker atomically claims the next global
    episode index from ``counter``, saves the trajectory pickle at that index,
    and reports ('seed_found', idx, seed, plan_ok) to the main process.
    """
    task = class_decorator(task_name)
    search_args = deepcopy(args)
    search_args["need_plan"] = True
    save_failed_cases = args.get("save_failed_cases", False)

    seed = _first_seed(start_seed, worker_id, num_workers)
    tries, fails = 0, 0

    while True:
        with lock:
            if counter.value >= target:
                break
        tries += 1
        try:
            task.setup_demo(now_ep_num=seed, seed=seed, **deepcopy(search_args))
            task.play_once()

            plan_ok = bool(task.plan_success and task.check_success())

            claimed_idx: Optional[int] = None
            if plan_ok or save_failed_cases:
                with lock:
                    if counter.value < target:
                        claimed_idx = counter.value
                        counter.value += 1
                if claimed_idx is not None:
                    # save while the env is still alive (reads planner outputs),
                    # and report the claim IMMEDIATELY so an exception in the
                    # cleanup below can never orphan an allocated index
                    task.save_traj_data(claimed_idx)
                    queue.put(("seed_found", claimed_idx, seed, plan_ok))

            if plan_ok:
                if claimed_idx is not None:
                    _wprint(worker_id, f"seed search episode {claimed_idx} success! (seed = {seed})")
            else:
                fails += 1
                if claimed_idx is not None:
                    _wprint(worker_id, f"seed search episode {claimed_idx} \033[91mFAIL\033[0m! "
                                       f"(seed = {seed}) - \033[93mSAVING ANYWAY\033[0m")
                else:
                    _wprint(worker_id, f"seed {seed} fail!")

            _safe_close(task)
            if args["render_freq"]:
                task.viewer.close()

            if claimed_idx is None and (plan_ok or save_failed_cases):
                break  # success, but target already reached -> stop searching
        except UnStableError as e:  # noqa: F405
            fails += 1
            _wprint(worker_id, f"seed {seed} fail! (UnStableError: {e})")
            _safe_close(task)
            if args["render_freq"]:
                task.viewer.close()
            time.sleep(0.3)
        except Exception as e:
            fails += 1
            _wprint(worker_id, f"seed {seed} fail! (Error: {e})")
            _safe_close(task)
            if args["render_freq"]:
                task.viewer.close()
            time.sleep(1)

        seed += num_workers

    queue.put(("worker_done", worker_id, {"tries": tries, "fails": fails}))


def _render_episode(
    task,
    template: Dict[str, Any],
    idx: int,
    seed: int,
    queue,
    worker_id: int,
    clear_cache_freq: int,
    check_render_success: bool,
    keep_cache_for_export: bool,
) -> bool:
    """Render one episode (index ``idx``, seed ``seed``) from its saved
    trajectory pickle, merge it to HDF5+video, and report the result.

    When ``keep_cache_for_export`` is set the .cache/episode{idx}/ frames are
    left on disk; the main process exports them to LeRobot and deletes them.
    Returns True on a successful render.
    """
    try:
        render_args = deepcopy(template)
        task.setup_demo(now_ep_num=idx, seed=seed, **render_args)

        traj_data = task.load_tran_data(idx)
        render_args["left_joint_path"] = traj_data["left_joint_path"]
        render_args["right_joint_path"] = traj_data["right_joint_path"]
        task.set_path_lst(render_args)

        info = task.play_once()

        # compute the success verdict ONCE (before close_env tears down the
        # scene) so it can both gate the render and be persisted in the export.
        try:
            ep_success = bool(task.check_success())
        except Exception:
            ep_success = False
        render_success = ep_success if check_render_success else True

        if render_success:
            _safe_close(task, clear_cache=((idx + 1) % clear_cache_freq == 0))
            task.merge_pkl_to_hdf5_video()
            marks = [dict(m) for m in getattr(task, "_phase_marks", [])]
            if not keep_cache_for_export:
                task.remove_data_cache()
            _wprint(worker_id, f"\033[92mEpisode {idx} render "
                               f"{'SUCCESS' if check_render_success else 'COMPLETE'} (seed={seed})\033[0m")
            queue.put(("rendered", idx, seed, ep_success, info, marks, keep_cache_for_export))
            return True

        _wprint(worker_id, f"\033[91mEpisode {idx} render FAILED for seed {seed} - skipping\033[0m")
        _safe_close(task, clear_cache=True)
        try:
            task.remove_data_cache()
        except Exception:
            pass
        queue.put(("render_failed", idx, seed))
        time.sleep(0.5)
        return False
    except Exception as e:
        _wprint(worker_id, f"\033[91mException during render for seed {seed} (episode {idx}): {e}\033[0m")
        _safe_close(task, clear_cache=True)
        # only touch the cache if it belongs to THIS episode -- a stale ep_num
        # here could otherwise delete a finished episode still pending export
        if getattr(task, "ep_num", None) == idx:
            try:
                task.remove_data_cache()
            except Exception:
                pass
        queue.put(("render_failed", idx, seed))
        time.sleep(0.5)
        return False


def render_worker(
    worker_id: int,
    num_workers: int,
    task_name: str,
    collect_args: Dict[str, Any],
    seed_list: List[int],
    n_target: int,
    keep_cache_for_export: bool,
    queue,
) -> None:
    """Phase 2: render episode indices ``worker_id, worker_id+N, ...`` < n_target.

    Indices whose HDF5 already exists (resume) are reported as 'skipped'.
    """
    task = class_decorator(task_name)
    clear_cache_freq = collect_args["clear_cache_freq"]
    check_render_success = collect_args.get("check_render_success", False)

    for idx in range(worker_id, n_target, num_workers):
        if seed_list[idx] is None:  # phase-1 gap -> hand straight to regen
            queue.put(("render_failed", idx, None))
            continue
        if os.path.exists(_hdf5_path(collect_args["save_path"], idx)):
            _wprint(worker_id, f"episode{idx}.hdf5 exists - skipping")
            queue.put(("skipped", idx, seed_list[idx]))
            continue
        _wprint(worker_id, f"\033[34mTask name: {collect_args['task_name']} | "
                           f"Episode: {idx} | Seed: {seed_list[idx]}\033[0m")
        _render_episode(task, collect_args, idx, seed_list[idx], queue, worker_id,
                        clear_cache_freq, check_render_success, keep_cache_for_export)

    queue.put(("worker_done", worker_id, {}))


def regen_worker(
    worker_id: int,
    num_workers: int,
    task_name: str,
    base_collect_args: Dict[str, Any],
    start_seed: int,
    holes,
    lock,
    keep_cache_for_export: bool,
    queue,
) -> None:
    """Phase 3: refill failed episode indices ('holes') with NEW seeds.

    Seeds stay in this worker's residue class (seed % num_workers == worker_id),
    starting at the first such seed >= start_seed. For each planning success the
    worker pops a hole index, overwrites its trajectory pickle, and renders it in
    place; a failed render puts the hole back for another attempt.
    """
    task = class_decorator(task_name)
    clear_cache_freq = base_collect_args["clear_cache_freq"]
    check_render_success = base_collect_args.get("check_render_success", False)

    plan_template = deepcopy(base_collect_args)
    plan_template["need_plan"] = True
    plan_template["save_data"] = False
    plan_template["left_joint_path"] = []
    plan_template["right_joint_path"] = []

    seed = _first_seed(start_seed, worker_id, num_workers)

    while True:
        with lock:
            if len(holes) == 0:
                break
        try:
            task.setup_demo(now_ep_num=seed, seed=seed, **deepcopy(plan_template))
            task.play_once()
            plan_ok = bool(task.plan_success and task.check_success())

            if plan_ok:
                with lock:
                    idx = holes.pop(0) if len(holes) else None
                if idx is None:
                    _safe_close(task, clear_cache=True)
                    break  # every hole got filled while we were planning
                _wprint(worker_id, f"\033[92mNew seed {seed} planning succeeded -> episode {idx}\033[0m")
                task.save_traj_data(idx)
                _safe_close(task, clear_cache=True)
                time.sleep(0.3)

                ok = _render_episode(task, base_collect_args, idx, seed, queue, worker_id,
                                     clear_cache_freq, check_render_success, keep_cache_for_export)
                if not ok:
                    with lock:
                        holes.append(idx)
            else:
                _wprint(worker_id, f"New seed {seed} planning failed")
                _safe_close(task, clear_cache=True)
                time.sleep(0.3)
        except Exception as e:
            _wprint(worker_id, f"Error with seed {seed}: {e}")
            _safe_close(task, clear_cache=True)
            time.sleep(0.5)

        seed += num_workers

    queue.put(("worker_done", worker_id, {}))


# ======================================================================
# Main-process orchestration
# ======================================================================


def _spawn_and_consume(
    ctx,
    worker_fn: Callable,
    per_worker_args: List[Tuple],
    queue,
    on_message: Callable[[Tuple], None],
) -> List[Dict[str, Any]]:
    """Spawn one process per per_worker_args entry, consume queue messages until
    every worker reported 'worker_done', then join. Returns the workers' stats.

    Raises RuntimeError if a worker dies without reporting (avoids hanging on a
    queue that will never produce the missing 'worker_done').
    """
    procs = [ctx.Process(target=worker_fn, args=wargs, daemon=False) for wargs in per_worker_args]
    for p in procs:
        p.start()

    stats: List[Dict[str, Any]] = []
    pending = len(procs)
    while pending > 0:
        try:
            msg = queue.get(timeout=30)
        except Exception:
            dead = [p for p in procs if not p.is_alive()]
            if len(dead) + pending > len(procs):
                for p in procs:
                    p.terminate()
                raise RuntimeError(
                    f"{len(dead) - (len(procs) - pending)} worker process(es) died "
                    f"without reporting; aborting to avoid a hang.")
            continue
        if msg[0] == "worker_done":
            pending -= 1
            stats.append(msg[2])
        else:
            on_message(msg)

    for p in procs:
        p.join()
        if p.exitcode not in (0, None):
            print(f"\033[93mWarning: worker process {p.pid} exited with code {p.exitcode}\033[0m")
    return stats


def run(task_name: str, args: Dict[str, Any], num_workers: int) -> None:
    ctx = mp.get_context("spawn")
    save_path = args["save_path"]
    episode_num = int(args["episode_num"])

    print(f"Task Name: \033[34m{args['task_name']}\033[0m | Workers: {num_workers}")
    os.makedirs(save_path, exist_ok=True)

    seed_by_idx: Dict[int, int] = {}

    # =========== Phase 1: Collect Seeds (parallel, strided) ===========
    if not args["use_seed"]:
        print("\033[93m" + "[Start Seed and Pre Motion Data Collection]" + "\033[0m")

        if os.path.exists(_seed_file(save_path)):
            existing = _read_seed_file(save_path)
            seed_by_idx = {i: s for i, s in enumerate(existing)}
            if existing:
                print(f"Exist seed file, Start from: {max(existing) + 1} / {len(existing)}")

        if len(seed_by_idx) < episode_num:
            start_seed = (max(seed_by_idx.values()) + 1) if seed_by_idx else 0
            counter = ctx.Value("i", len(seed_by_idx))
            lock = ctx.Lock()
            queue = ctx.Queue()

            def on_seed_msg(msg: Tuple) -> None:
                kind, idx, seed, plan_ok = msg
                assert kind == "seed_found", f"unexpected message in seed phase: {kind}"
                seed_by_idx[idx] = seed
                _write_seed_file(save_path, seed_by_idx)

            stats = _spawn_and_consume(
                ctx, seed_search_worker,
                [(w, num_workers, task_name, args, start_seed, episode_num, counter, lock, queue)
                 for w in range(num_workers)],
                queue, on_seed_msg)

            total_tries = sum(s.get("tries", 0) for s in stats)
            total_fails = sum(s.get("fails", 0) for s in stats)
            _write_seed_file(save_path, seed_by_idx)
            print(f"\nComplete seed collection, failed \033[91m{total_fails}\033[0m times "
                  f"/ {total_tries} tries \n")
    else:
        print("\033[93m" + "Use Saved Seeds List".center(30, "-") + "\033[0m")
        seed_by_idx = {i: s for i, s in enumerate(_read_seed_file(save_path))}

    # Gap-tolerant view: an index claimed by a worker that then crashed before
    # saving/reporting shows up as None; the render phase routes such indices
    # straight to the regeneration phase instead of crashing on a KeyError.
    n_have = (max(seed_by_idx) + 1) if seed_by_idx else 0
    seed_list: List[Optional[int]] = [seed_by_idx.get(i) for i in range(n_have)]
    gap_indices = [i for i, s in enumerate(seed_list) if s is None]
    if gap_indices:
        print(f"\033[93mWarning: seed indices {gap_indices} have no recorded seed "
              f"(worker crash during phase 1?); they will be regenerated\033[0m")

    # =========== Phase 2: Collect Data (parallel render) ===========
    if args["collect_data"]:
        print("\033[93m" + "[Start Data Collection]" + "\033[0m")

        args["need_plan"] = False
        args["render_freq"] = 0
        args["save_data"] = True
        base_collect_args = deepcopy(args)

        check_render_success = args.get("check_render_success", False)
        print(f"\033[93m[Render Success Check: {'ENABLED' if check_render_success else 'DISABLED'}]\033[0m")

        n_target = min(episode_num, len(seed_list))
        if n_target < episode_num:
            print(f"\033[93mWarning: seed list has only {len(seed_list)} seeds "
                  f"(< episode_num={episode_num}); rendering {n_target}\033[0m")

        # Optional: export each finished episode into a combined LeRobot v2.1
        # dataset. The writer is stateful and single-process -> it lives HERE in
        # the main process; workers hand finished episodes over via the queue.
        lr_exporter = None
        if args.get("export_lerobot"):
            try:
                from envs.utils.lerobot_export import LeRobotEpisodeExporter
                lr_exporter = LeRobotEpisodeExporter(args)
                print(f"\033[93m[LeRobot v2.1 export: ENABLED -> {lr_exporter.root} "
                      f"(task_index={lr_exporter.task_index})]\033[0m")
            except Exception as e:
                print(f"\033[91m[LeRobot export init FAILED, skipping: {e}]\033[0m")
                lr_exporter = None
        keep_cache_for_export = lr_exporter is not None

        # scene_info.json is a shared read-modify-write JSON -> main process only
        info_file_path = os.path.join(save_path, "scene_info.json")
        info_db: Dict[str, Any] = {}
        if os.path.exists(info_file_path):
            with open(info_file_path, "r", encoding="utf-8") as f:
                info_db = json.load(f)

        class _EpisodeHandle:
            """Duck-typed stand-in for the live task env in LeRobotEpisodeExporter.export():
            it only reads .save_dir, .ep_num and ._phase_marks."""

            def __init__(self, ep_num: int, phase_marks: List[Dict[str, Any]]):
                self.save_dir = save_path
                self.ep_num = ep_num
                self._phase_marks = phase_marks

        original_seed_list = seed_list.copy()
        rendered_count = 0
        failed_indices: List[int] = []
        in_regen = False

        def on_render_msg(msg: Tuple) -> None:
            nonlocal rendered_count
            kind = msg[0]
            if kind == "rendered":
                _, idx, seed, ep_success, info, marks, cache_kept = msg
                seed_by_idx[idx] = seed
                info_db[f"episode_{idx}"] = info
                with open(info_file_path, "w", encoding="utf-8") as f:
                    json.dump(info_db, f, ensure_ascii=False, indent=4)
                if cache_kept:
                    if lr_exporter is not None:
                        try:
                            lr_exporter.export(_EpisodeHandle(idx, marks), ep_success)
                        except Exception as e:
                            print(f"\033[91m[LeRobot export failed ep {idx}: {e}]\033[0m")
                    shutil.rmtree(_cache_dir(save_path, idx), ignore_errors=True)
                rendered_count += 1
            elif kind == "render_failed":
                _, idx, seed = msg
                # during regen the shared 'holes' list is the source of truth;
                # re-appending here would only inflate the final failure count
                if not in_regen and idx not in failed_indices:
                    failed_indices.append(idx)
            elif kind == "skipped":
                _, idx, seed = msg
                rendered_count += 1
            else:
                raise AssertionError(f"unexpected message in render phase: {kind}")

        if n_target > 0:
            queue = ctx.Queue()
            _spawn_and_consume(
                ctx, render_worker,
                [(w, num_workers, task_name, base_collect_args, seed_list, n_target,
                  keep_cache_for_export, queue)
                 for w in range(num_workers)],
                queue, on_render_msg)

        # ===== Phase 3: refill failed episode slots with new seeds (parallel, strided) =====
        holes = sorted(set(failed_indices))
        if holes:
            print(f"\033[93m[Need {len(holes)} more episodes, regenerating seeds...]\033[0m")
            failed_indices.clear()
            in_regen = True
            manager = ctx.Manager()
            shared_holes = manager.list(holes)
            lock = ctx.Lock()
            queue = ctx.Queue()
            regen_start = max(seed_by_idx.values()) + 1 if seed_by_idx else 0

            _spawn_and_consume(
                ctx, regen_worker,
                [(w, num_workers, task_name, base_collect_args, regen_start,
                  shared_holes, lock, keep_cache_for_export, queue)
                 for w in range(num_workers)],
                queue, on_render_msg)
            manager.shutdown()

        if lr_exporter is not None:
            try:
                lr_exporter.close()
                print(f"\033[92m[LeRobot v2.1 export: wrote "
                      f"{lr_exporter.writer._local if lr_exporter.writer else 0} "
                      f"episodes to {lr_exporter.root}]\033[0m")
            except Exception as e:
                print(f"\033[91m[LeRobot export close failed: {e}]\033[0m")

        # persist the final index->seed mapping; archive it if regen changed it
        final_seed_list = [seed_by_idx[i] for i in sorted(seed_by_idx)]
        if original_seed_list != final_seed_list:
            with open(os.path.join(save_path, "seed_archive.txt"), "a") as f:
                f.write("original: " + " ".join(str(s) for s in original_seed_list) + "\n")
                f.write("adjusted: " + " ".join(str(s) for s in final_seed_list) + "\n\n")
        _write_seed_file(save_path, seed_by_idx)

        print(f"\n\033[92mData collection complete: {rendered_count} episodes\033[0m")
        if failed_indices:
            print(f"\033[93mNote: {len(failed_indices)} episodes failed during render\033[0m")

        command = (f"cd description && bash gen_episode_instructions.sh "
                   f"{args['task_name']} {args['task_config']} {args['language_num']}")
        os.system(command)


def main(task_name: str, task_config: str, num_workers: Optional[int] = None) -> None:
    # fail fast if the task module/class doesn't exist (workers import it themselves)
    envs_module = importlib.import_module(f"envs.{task_name}")
    if not hasattr(envs_module, task_name):
        raise SystemExit("No such task")

    config_path = f"./task_config/{task_config}.yml"
    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name

    if num_workers is None:
        num_workers = int(args.get("num_workers", 1))
    if num_workers < 1:
        raise SystemExit("num_workers must be >= 1")

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")  # noqa: F405

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise ValueError("missing embodiment files")
        return robot_file

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("number of embodiment config parameters should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    # show config
    print("============= Config =============\n")
    print("\033[96mUse Dynamic:\033[0m " + str(args.get("use_dynamic", False)))
    if args.get("use_dynamic", False):
        print(" - Dynamic Level: " + str(args.get("dynamic_level", "N/A")))
        print(" - Dynamic Coefficient: " + str(args.get("dynamic_coefficient", "N/A")))
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\033[94mWorkers:\033[0m " + str(num_workers) +
          (f" (worker w takes seeds with seed % {num_workers} == w)" if num_workers > 1 else ""))
    print("\n==================================")

    args["embodiment_name"] = embodiment_name
    args["task_config"] = task_config
    args["save_path"] = os.path.join(args["save_path"], str(args["task_name"]), args["task_config"])
    run(task_name, args, num_workers)


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    mp.set_start_method("spawn", force=True)

    parser = ArgumentParser()
    parser.add_argument("task_name", type=str)
    parser.add_argument("task_config", type=str)
    parser.add_argument("--num-workers", "-n", type=int, default=None,
                        help="parallel worker processes; worker w only tries seeds with "
                             "seed %% N == w (default: config key 'num_workers', else 1)")
    cli = parser.parse_args()

    main(task_name=cli.task_name, task_config=cli.task_config, num_workers=cli.num_workers)
