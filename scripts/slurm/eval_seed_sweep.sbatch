#!/bin/bash
#SBATCH -J rd-seeds
#SBATCH -p all
#SBATCH --gres=gpu:rtx5080:1
#SBATCH -c 4
#SBATCH --mem=20G
#SBATCH -N1
#SBATCH -t 08:00:00
#SBATCH --array=0-103
#SBATCH -o /shared_work/markhsp/RoboDyna/logs/eval-seeds-%A_%a.out

# Evaluation seed search: 23 conceptual tasks x 4 scenarios + 12 household tasks = 104 jobs.
# Each job searches from seed 10000 until it has 10 passing seeds. It is resumable and saves no
# HDF5, video, or LeRobot artifacts.
set -euo pipefail
REPO=/shared_work/markhsp/RoboDyna
cd "$REPO"

export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
unset DISPLAY
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$REPO/.venv/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO:$REPO/script:$REPO/script/bench_script:${PYTHONPATH:-}"
export PYTHONHASHSEED=0
export PYTHONWARNINGS=ignore::UserWarning
export ROBODYNA_SWEEP_ROOT="$REPO/run_results/eval_seed_sweep"
mkdir -p "$ROBODYNA_SWEEP_ROOT" "$REPO/logs"

conceptual=(
  catch_marbles_trapdoors catch_ramp_ball catch_cuboid catch_shelf_marble
  catch_valley_ball stop_valley_ball cook_meat cook_meat_timer put_cup_belt
  dispense_gummy punch_dual_holes save_goal hit_target load_train marble_shelf_maze
  pack_fruits pick_ripe_apple place_block_belt play_billiard control_quality
  drop_ball_hole sort_apples_belt whack_moles
)
scenarios=(default opt1 opt2 'opt1+2')
household=(
  trap_bug boil_milk fill_coffee_jar pour_beer cook_food cook_food_timer
  measure_ingredient make_soup catch_cup catch_mouse_object_drop stop_ball clean_table
)

id=$SLURM_ARRAY_TASK_ID
if (( id < 92 )); then
  task=${conceptual[$((id / 4))]}
  scenario=${scenarios[$((id % 4))]}
  echo "[seed-search] task=$task scenario=$scenario start=10000 target=10 node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset}"
  python -u script/bench_script/collect_eval_seeds.py --task "$task" --family conceptual --scenario "$scenario" \
    --start-seed 10000 --target 10 --output-dir "$ROBODYNA_SWEEP_ROOT"
else
  task=${household[$((id - 92))]}
  echo "[seed-search] task=$task family=household start=10000 target=10 node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset}"
  python -u script/bench_script/collect_eval_seeds.py --task "$task" --family household --scenario base \
    --start-seed 10000 --target 10 --output-dir "$ROBODYNA_SWEEP_ROOT"
fi
echo "[seed-sweep] DONE task=$task array_id=$id $(date -Is)"
