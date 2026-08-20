#!/usr/bin/env bash
# Local (4080) equivalent of scripts/slurm/collect.sbatch: run a task/scenario manifest one at a time on the
# single RTX 4080S, with the same attempt/timeout/zero-progress guards, then finalize LeRobot.
# Usage: collect_4080.sh <manifest>
set -u
REPO=/shared_work/markhsp/RoboDyna
MANIFEST="${1:?manifest required}"
TARGET="${TARGET:-50}"
PER_TRY_TIMEOUT="${PER_TRY_TIMEOUT:-9000}"
MAX_TRIES="${MAX_TRIES:-6}"
MAX_WALL="${MAX_WALL:-43200}"
ZERO_STREAK_SKIP="${ZERO_STREAK_SKIP:-3}"
# Some tasks wedge inside SAPIEN with the main thread parked in poll() and ~0% CPU (cook_food did
# this at frame 81 of episode 0). PER_TRY_TIMEOUT alone would burn hours on that, so watch the log
# and kill the attempt as soon as it stops producing output.
STALL_SECS="${STALL_SECS:-900}"
IDLE_TICKS="${IDLE_TICKS:-40}"      # <0.4 cpu-seconds per 20s window counts as "not computing"
IDLE_WINDOWS="${IDLE_WINDOWS:-3}"   # kill after this many consecutive idle windows (3 = 60s)

cd "$REPO" || exit 1
source /home/user/miniforge3/etc/profile.d/conda.sh
conda activate robodyna
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
unset DISPLAY
export PYTHONHASHSEED=0
export PYTHONWARNINGS=ignore::UserWarning
export PYTHONPATH="$REPO:$REPO/script:$REPO/script/bench_script:${PYTHONPATH:-}"
mkdir -p "$REPO/run_results" "$REPO/logs"

while read -r TASK SCENARIO FAM; do
  [ -z "${TASK:-}" ] && continue
  CFG=demo_dynamic
  DATADIR="$REPO/data/$TASK/$SCENARIO/data"
  SEEDFILE="$REPO/data/$TASK/$SCENARIO/seed.txt"
  LOG="$REPO/logs/local4080-${TASK}__${SCENARIO}.log"
  count_hdf5() { find "$DATADIR" -name 'episode*.hdf5' 2>/dev/null | wc -l | tr -d ' '; }
  count_seeds() { [ -f "$SEEDFILE" ] && wc -w < "$SEEDFILE" 2>/dev/null | tr -d ' ' || echo 0; }
  count_progress() { echo $(( $(count_seeds) + $(count_hdf5) )); }
  # Move every trajectory already fed to pass 2 into _traj_used/ so no later attempt replays it --
  # see the call site below for why a plain retry otherwise writes duplicate episodes.
  retire_consumed() {
    local t n=0 cdir="$REPO/data/$TASK/$SCENARIO"
    [ -s "$LOG" ] || return 0
    mkdir -p "$cdir/_traj_used"
    for t in $(tr '\r' '\n' < "$LOG" 2>/dev/null | grep -ao 'traj: episode[0-9]*' | grep -o '[0-9]*' | sort -un); do
      if [ -f "$cdir/_traj_data/episode${t}.pkl" ]; then
        mv "$cdir/_traj_data/episode${t}.pkl" "$cdir/_traj_used/" && n=$((n + 1))
      fi
    done
    [ "$n" -gt 0 ] && echo "[local4080] retired $n consumed trajectory file(s) -> _traj_used/ (anti-duplicate)" | tee -a "$LOG"
    return 0
  }

  mkdir -p "$DATADIR" "$REPO/data_lerobot/prod_run/${TASK}__${SCENARIO}"
  echo "[local4080] $(date -Is) task=$TASK scenario=$SCENARIO config=$CFG target=$TARGET" | tee -a "$LOG"
  t0=$SECONDS; have=$(count_hdf5); prog=$(count_progress); SKIP_REASON=""; zero_streak=0
  for try in $(seq 1 "$MAX_TRIES"); do
    [ "$have" -ge "$TARGET" ] && break
    if [ $((SECONDS - t0)) -ge "$MAX_WALL" ]; then SKIP_REASON="wall>${MAX_WALL}s"; break; fi
    before=$prog
    echo "[local4080] attempt $try/$MAX_TRIES (hdf5 $have/$TARGET, progress=$before) $(date -Is)" | tee -a "$LOG"
    timeout "$PER_TRY_TIMEOUT" python -u -X faulthandler script/collect_data.py "$TASK" "$CFG" --scenario "$SCENARIO" --production >> "$LOG" 2>&1 &
    run_pid=$!
    stalled=0
    # CPU-idle wedge detector. The Vulkan readback deadlock (camera.get_picture -> do_poll on a
    # GPU fence that never signals) leaves the process burning ~0.5% CPU, while any legitimate
    # phase -- startup, render, hdf5 write -- saturates a core. That makes CPU time a far sharper
    # signal than log mtime: it fires in ~60s instead of STALL_SECS, and the log-age check below
    # stays as the backstop.
    idle_windows=0; prev_ticks=-1; py_pid=""
    while kill -0 "$run_pid" 2>/dev/null; do
      sleep 20
      kill -0 "$run_pid" 2>/dev/null || break
      [ -z "$py_pid" ] && py_pid=$(pgrep -P "$run_pid" | head -1)
      if [ -n "$py_pid" ] && [ -r "/proc/$py_pid/stat" ]; then
        ticks=$(awk '{print $14 + $15}' "/proc/$py_pid/stat" 2>/dev/null || echo "")
        if [ -n "$ticks" ] && [ "$prev_ticks" -ge 0 ] 2>/dev/null; then
          if [ $(( ticks - prev_ticks )) -lt "$IDLE_TICKS" ]; then
            idle_windows=$(( idle_windows + 1 ))
          else
            idle_windows=0
          fi
        fi
        prev_ticks=$ticks
      fi
      if [ "$idle_windows" -ge "$IDLE_WINDOWS" ]; then
        echo "[local4080] WEDGE: cpu idle for $(( idle_windows * 20 ))s -> aborting attempt" | tee -a "$LOG"
        pkill -ABRT -P "$run_pid" 2>/dev/null; kill -ABRT "$run_pid" 2>/dev/null
        sleep 10
        pkill -9 -P "$run_pid" 2>/dev/null; kill -9 "$run_pid" 2>/dev/null
        stalled=1
        break
      fi
      age=$(( $(date +%s) - $(stat -c %Y "$LOG") ))
      if [ "$age" -ge "$STALL_SECS" ]; then
        echo "[local4080] STALL: no log output for ${age}s -> aborting attempt" | tee -a "$LOG"
        # SIGABRT first so faulthandler prints where it wedged, then make sure it dies.
        pkill -ABRT -P "$run_pid" 2>/dev/null; kill -ABRT "$run_pid" 2>/dev/null
        sleep 15
        pkill -9 -P "$run_pid" 2>/dev/null; kill -9 "$run_pid" 2>/dev/null
        stalled=1
        break
      fi
    done
    wait "$run_pid" 2>/dev/null
    rc=$?
    [ "$stalled" = 1 ] && rc=STALL
    have=$(count_hdf5); prog=$(count_progress)
    echo "[local4080] after attempt $try: hdf5=$have progress=$prog rc=$rc dt=$((SECONDS - t0))s" | tee -a "$LOG"
    # Pass 2 restarts BOTH its hdf5 index and its trajectory cursor at st_idx = #hdf5, but inside an
    # attempt the two drift (a skipped trajectory bumps the cursor without writing an hdf5). So a
    # plain retry replays already-rendered trajectories -> byte-identical duplicate episodes.
    # Retiring them makes the next attempt fail fast on those indices and land on a fresh one.
    retire_consumed
    if [ "$prog" -le "$before" ]; then
      zero_streak=$((zero_streak + 1))
      echo "[local4080] zero-progress attempt (streak=$zero_streak/$ZERO_STREAK_SKIP)" | tee -a "$LOG"
      # A repeated wedge is always the SAME trajectory: pass 2 resumes at st_idx = #hdf5, so that
      # index is the poison one. Moving its pkl aside makes load_tran_data raise inside the try
      # block -> the collector logs the failure, skips to the next index, and tops the bank back up
      # with fresh seeds in its phase-2 block. Touches no task code.
      if [ "$rc" = STALL ] && [ "$zero_streak" -ge 2 ]; then
        poison=$(count_hdf5)
        ptraj="$REPO/data/$TASK/$SCENARIO/_traj_data/episode${poison}.pkl"
        if [ -f "$ptraj" ]; then
          mkdir -p "$REPO/data/$TASK/$SCENARIO/_traj_quarantine"
          mv "$ptraj" "$REPO/data/$TASK/$SCENARIO/_traj_quarantine/" && \
            echo "[local4080] QUARANTINE traj episode${poison}.pkl (wedged x${zero_streak}) -> skip + regenerate" | tee -a "$LOG"
          rm -rf "$REPO/data/$TASK/$SCENARIO/.cache/episode${poison}" 2>/dev/null
          zero_streak=0
        fi
      fi
      [ "$zero_streak" -ge "$ZERO_STREAK_SKIP" ] && { SKIP_REASON="0-progress-x${zero_streak}"; break; }
    else
      zero_streak=0
    fi
  done
  rm -rf "$REPO/data/$TASK/$SCENARIO/.cache" 2>/dev/null

  have=$(count_hdf5); STATUS=OK
  [ "$have" -ge "$TARGET" ] || STATUS=PARTIAL
  [ "$have" -eq 0 ] && STATUS=EMPTY
  echo "LOCAL4080 task=$TASK scenario=$SCENARIO config=$CFG status=$STATUS hdf5=$have/$TARGET secs=$((SECONDS-t0)) skip=${SKIP_REASON:-none}" \
    | tee "$REPO/run_results/local4080_${TASK}__${SCENARIO}.txt" | tee -a "$LOG"

  LRROOT="$REPO/data_lerobot/prod_run/${TASK}__${SCENARIO}"
  if ls "$LRROOT"/data/task-*/*.parquet >/dev/null 2>&1 || ls "$LRROOT"/data/chunk-*/*.parquet >/dev/null 2>&1; then
    python "$REPO/scripts/rebuild_lerobot_parts.py" --root "$LRROOT" >> "$LOG" 2>&1
    python "$REPO/scripts/merge_lerobot_meta.py" --root "$LRROOT" >> "$LOG" 2>&1 \
      || echo "[local4080] lerobot merge FAILED for ${TASK}__${SCENARIO}" | tee -a "$LOG"
  fi
  echo "[local4080] DONE $(date -Is) $TASK/$SCENARIO status=$STATUS hdf5=$have/$TARGET" | tee -a "$LOG"
done < "$MANIFEST"
echo "[local4080] MANIFEST COMPLETE $(date -Is)"
