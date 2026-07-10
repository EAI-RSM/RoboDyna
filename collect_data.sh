#!/bin/bash
# Author: Rui Heng Yang
# Usage: bash collect_data.sh <task_name> <task_config> <gpu_id> [num_workers]
# num_workers (default 1): parallel collector processes; worker w only tries
# seeds with seed % num_workers == w, so seed sets never overlap.

task_name=${1}
task_config=${2}
gpu_id=${3}
num_workers=${4:-1}

./script/.update_path.sh > /dev/null 2>&1

export CUDA_VISIBLE_DEVICES=${gpu_id}

PYTHONWARNINGS=ignore::UserWarning \
python script/collect_data.py $task_name $task_config --num-workers ${num_workers}
rm -rf data/${task_name}/${task_config}/.cache
