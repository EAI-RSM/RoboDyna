#!/bin/bash
#SBATCH -J rd-collect
#SBATCH -p all
#SBATCH --gres=gpu:rtx5080:1
#SBATCH -c 6
#SBATCH --mem=28G
#SBATCH -N1
#SBATCH -t 08:00:00
#SBATCH -w TRT-EAI-OLDLAB-1,aaron-workstation,trt-eai-oldlab-4
#SBATCH --exclude=trt-eai-oldlab-3,trt-eai-oldlab-5
#SBATCH -o /shared_work/markhsp/RoboDyna/logs/collect-%A_%a.out
# Production collection array. One manifest line -> one (task,scenario) combo -> 50 HDF5 + LeRobot.
# Self-contained: sets PYTHONPATH itself (trt has no .update_path.sh) and calls collect_data.py
# directly. Resumes from partials (seed.txt). EXTENDED policy (user: "collect longer until 100"):
# keep collecting up to a 7h wall so slow-but-progressing combos can actually reach 100; the only
# early stop is the hopeless-guard = TWO consecutive zero-progress hours (catches true 0% tasks like
# pick_ripe_apple without hogging a GPU forever). Then finalizes the LeRobot meta from disk.
set -uo pipefail
REPO=/shared_work/markhsp/RoboDyna
cd "$REPO" || exit 1
mkdir -p logs run_results

MANIFEST="${MANIFEST:-$REPO/task_config/manifest_collect.txt}"
TARGET="${TARGET:-50}"
PER_TRY_TIMEOUT="${PER_TRY_TIMEOUT:-3600}"   # seconds per collect attempt (1h) = hopeless-guard checkpoint
MAX_TRIES="${MAX_TRIES:-10}"                  # resume attempts; 7h wall usually binds first
MAX_WALL="${MAX_WALL:-25200}"                 # 7h per combo (extended: give slow combos time to reach 100)
ZERO_STREAK_SKIP="${ZERO_STREAK_SKIP:-2}"     # skip after N consecutive zero-progress hours (true 0% guard)

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
TASK=$(echo "$LINE" | awk '{print $1}')
SCENARIO=$(echo "$LINE" | awk '{print $2}')
FAM=$(echo "$LINE" | awk '{print $3}')
CFG=demo_dynamic
[ -z "$TASK" ] && { echo "empty manifest line $SLURM_ARRAY_TASK_ID"; exit 1; }

export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
unset DISPLAY
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$REPO/.venv/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONHASHSEED=0
export PYTHONWARNINGS=ignore::UserWarning
# trt has no .update_path.sh -> set the import path ourselves (scene_utils lives in bench_script)
export PYTHONPATH="$REPO:$REPO/script:$REPO/script/bench_script:${PYTHONPATH:-}"

DATADIR="$REPO/data/$TASK/$SCENARIO/data"
SEEDFILE="$REPO/data/$TASK/$SCENARIO/seed.txt"
count_hdf5() { find "$DATADIR" -name 'episode*.hdf5' 2>/dev/null | wc -l | tr -d ' '; }
count_seeds() { [ -f "$SEEDFILE" ] && wc -w < "$SEEDFILE" 2>/dev/null | tr -d ' ' || echo 0; }
# collect_data.py is TWO-PASS: pass 1 finds all TARGET success seeds (seed.txt grows, hdf5 still 0),
# pass 2 renders each seed -> hdf5. So hdf5 alone is a FALSE stuck-signal during pass 1. The real
# forward-progress metric is seeds_found + hdf5_written (0->TARGET in pass1, ->2*TARGET after pass2).
count_progress() { echo $(( $(count_seeds) + $(count_hdf5) )); }

RES="$REPO/run_results/${TASK}__${SCENARIO}.txt"
echo "[collect] $(date -Is) task=$TASK scenario=$SCENARIO config=$CFG family=$FAM target=$TARGET node=$(hostname)"
t0=$SECONDS
have=$(count_hdf5)
prog=$(count_progress)
SKIP_REASON=""
zero_streak=0
for try in $(seq 1 "$MAX_TRIES"); do
  [ "$have" -ge "$TARGET" ] && break
  if [ $((SECONDS - t0)) -ge "$MAX_WALL" ]; then SKIP_REASON="wall>${MAX_WALL}s"; break; fi
  before=$prog
  echo "[collect] attempt $try/$MAX_TRIES (hdf5 $have/$TARGET, progress=$before) $(date -Is)"
  timeout "$PER_TRY_TIMEOUT" python -u script/collect_data.py "$TASK" "$CFG" --scenario "$SCENARIO" --production
  rc=$?
  have=$(count_hdf5)
  prog=$(count_progress)
  echo "[collect] after attempt $try: hdf5=$have progress=$prog rc=$rc dt=$((SECONDS - t0))s"
  # hopeless-guard: only skip after ZERO_STREAK_SKIP consecutive attempts with NO new seeds AND no new
  # hdf5. One unlucky empty hour on a low-SR task no longer kills it; a genuine 0% task still skips.
  if [ "$prog" -le "$before" ]; then
    zero_streak=$((zero_streak + 1))
    echo "[collect] zero-progress hour (streak=$zero_streak/$ZERO_STREAK_SKIP)"
    if [ "$zero_streak" -ge "$ZERO_STREAK_SKIP" ]; then SKIP_REASON="0-progress-x${zero_streak}(attempt $try)"; break; fi
  else
    zero_streak=0
  fi
done
rm -rf "$REPO/data/$TASK/$SCENARIO/.cache" 2>/dev/null

dt=$((SECONDS - t0))
have=$(count_hdf5)
STATUS=OK
[ "$have" -ge "$TARGET" ] || STATUS=PARTIAL
[ "$have" -eq 0 ] && STATUS=EMPTY
echo "COLLECT task=$TASK scenario=$SCENARIO config=$CFG family=$FAM status=$STATUS hdf5=$have/$TARGET secs=$dt skip=${SKIP_REASON:-none} node=$(hostname)" | tee "$RES"

# --- finalize LeRobot: rebuild _parts FROM DISK (survives a timeout/scancel that skips the
#     exporter's close()) then merge -> loadable top-level v2.1 meta. ---
LRROOT="$REPO/data_lerobot/prod_run/${TASK}__${SCENARIO}"
if ls "$LRROOT"/data/task-*/*.parquet >/dev/null 2>&1 || ls "$LRROOT"/data/chunk-*/*.parquet >/dev/null 2>&1; then
  echo "[collect] finalize lerobot for ${TASK}__${SCENARIO} (rebuild _parts from disk + merge)"
  python "$REPO/scripts/rebuild_lerobot_parts.py" --root "$LRROOT" 2>&1 | tail -3
  python "$REPO/scripts/merge_lerobot_meta.py" --root "$LRROOT" 2>&1 | tail -3 || echo "[collect] lerobot merge FAILED for ${TASK}__${SCENARIO}"
else
  echo "[collect] no parquet under $LRROOT -- skipping lerobot finalize (status=$STATUS)"
fi
echo "[collect] DONE $(date -Is) task=$TASK scenario=$SCENARIO status=$STATUS hdf5=$have/$TARGET"
