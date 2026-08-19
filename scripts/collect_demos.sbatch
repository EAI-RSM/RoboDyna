#!/bin/bash
#SBATCH -J domino-collect
#SBATCH -p gb10
#SBATCH --gres=gpu:1
#SBATCH -N1
#SBATCH -o /shared_work/markhsp/DOMINO/logs/collect-%j.out
# Usage: sbatch scripts/collect_demos.sbatch <task_name> [task_config]
#   task_config defaults to the shared suite config demo_dynamic (production: episode_num=50).
#   Use debug_dynamic for tuning (episode_num=8, save_failed_cases so jobs always terminate).
set -eo pipefail
TASK="${1:-adjust_bottle}"
CFG="${2:-demo_dynamic}"

source /shared_work/markhsp/miniforge3/etc/profile.d/conda.sh
conda activate /shared_work/markhsp/envs/domino

# Required runtime env (NVIDIA Vulkan ICD; headless offscreen render)
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
unset DISPLAY
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

cd /shared_work/markhsp/DOMINO
echo "[domino] host=$(hostname) task=$TASK cfg=$CFG gpu=$CUDA_VISIBLE_DEVICES start=$(date)"
# collect_data.sh uses CUDA_VISIBLE_DEVICES; Slurm pins us to one GPU -> local index 0
bash scripts/collect_data.sh "$TASK" "$CFG" 0
echo "[domino] EXIT rc=$? videos -> data/$TASK/$CFG/video/  $(date)"
