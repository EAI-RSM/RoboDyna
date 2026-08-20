#!/bin/bash
#SBATCH -J rd-salvage
#SBATCH -p all
#SBATCH --gres=gpu:rtx5080:1
#SBATCH -c 6
#SBATCH --mem=28G
#SBATCH -N1
#SBATCH -t 01:40:00
#SBATCH -o /shared_work/markhsp/RoboDyna/logs/salvage-%A_%a.out
# Salvage array: render the already-collected success seeds of combos that hit the 2h wall in pass-1
# (so pass-2 never ran -> 0/partial hdf5). Sets use_seed=True + episode_num=<Nseeds> on the SAME config
# (same save_path) so collect_data.py skips the search and just replays the saved _traj_data pkls into
# hdf5 + LeRobot. Idempotent + resumes (pass-2 skips existing episodeN.hdf5). Only run AFTER the main
# collection element for that combo has finished (driver waits for the whole array).
set -uo pipefail
REPO=/shared_work/markhsp/RoboDyna
cd "$REPO" || exit 1
mkdir -p logs run_results

MANIFEST="${MANIFEST:-$REPO/task_config/manifest_salvage.txt}"
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
TASK=$(echo "$LINE" | awk '{print $1}')
CFG=$(echo "$LINE" | awk '{print $2}')
OP=$(echo "$LINE" | awk '{print $3}')
NSEED=$(echo "$LINE" | awk '{print $4}')
[ -z "$TASK" ] && { echo "empty manifest line $SLURM_ARRAY_TASK_ID"; exit 1; }

export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
unset DISPLAY
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$REPO/.venv/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONHASHSEED=0
export PYTHONWARNINGS=ignore::UserWarning
export PYTHONPATH="$REPO:$REPO/script:$REPO/script/bench_script:$REPO/script_exp:$REPO/script_hh_exp:${PYTHONPATH:-}"

CFGYML="$REPO/task_config/${CFG}.yml"
# render-only: use the already-saved seeds, cap episode_num at what we actually collected.
python - "$CFGYML" "$NSEED" <<'PY'
import sys, yaml
p, n = sys.argv[1], int(sys.argv[2])
d = yaml.safe_load(open(p))
d["use_seed"] = True
d["episode_num"] = n
with open(p, "w") as f:
    yaml.safe_dump(d, f, sort_keys=False)
print(f"[salvage] patched {p}: use_seed=True episode_num={n}")
PY

DATADIR="$REPO/data/$TASK/$CFG/data"
count_hdf5() { find "$DATADIR" -name 'episode*.hdf5' 2>/dev/null | wc -l | tr -d ' '; }
b4=$(count_hdf5)
echo "[salvage] $(date -Is) task=$TASK cfg=$CFG op=$OP nseed=$NSEED before_hdf5=$b4 node=$(hostname)"
timeout 5400 python -u script/collect_data.py "$TASK" "$CFG"
rc=$?
have=$(count_hdf5)
STATUS=OK
[ "$have" -ge "$NSEED" ] || STATUS=PARTIAL
[ "$have" -eq 0 ] && STATUS=EMPTY
echo "SALVAGE task=$TASK cfg=$CFG op=$OP status=$STATUS hdf5=$have/$NSEED before=$b4 rc=$rc node=$(hostname)" | tee "$REPO/run_results/salv_${CFG}.txt"

# finalize LeRobot from disk (same as scripts/slurm/collect.sbatch tail)
LRROOT="$REPO/data_lerobot/prod_run/${TASK}__${OP}"
if ls "$LRROOT"/data/task-*/*.parquet >/dev/null 2>&1 || ls "$LRROOT"/data/chunk-*/*.parquet >/dev/null 2>&1; then
  echo "[salvage] finalize lerobot ${TASK}__${OP}"
  python "$REPO/scripts/rebuild_lerobot_parts.py" --root "$LRROOT" 2>&1 | tail -3
  python "$REPO/scripts/merge_lerobot_meta.py" --root "$LRROOT" 2>&1 | tail -3 || echo "[salvage] merge FAILED ${TASK}__${OP}"
fi
echo "[salvage] DONE $(date -Is) task=$TASK status=$STATUS hdf5=$have/$NSEED"
