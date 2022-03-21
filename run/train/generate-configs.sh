#!/bin/bash

experiment_dir="experiments/"
mkdir -p $experiment_dir

task=$1
setting=$2

python scripts/generate_configs.py $experiment_dir $task $setting
