#!/bin/bash
set -euo pipefail
cd /home/xuan/Desktop/RoboReal/RoboDyna
source /home/xuan/miniconda3/etc/profile.d/conda.sh
conda activate robodyna
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
unset DISPLAY
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p docs/final_task_demos/load_train
exec python -u script/bench_script/test_load_train.py \
  > docs/final_task_demos/load_train/run.log 2>&1
