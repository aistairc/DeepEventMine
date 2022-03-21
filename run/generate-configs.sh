#!/bin/bash

TASK=$1 # cg
EXPERIMENT_NAME=$2 # basic

EXPERIMENT_DIR="experiments/"
mkdir -p $EXPERIMENT_DIR

python scripts/generate_configs.py \
  --experiment_dir $EXPERIMENT_DIR \
  --task_name $TASK \
  --experiment_name $EXPERIMENT_NAME \
