#!/bin/bash

experiment_dir="experiments/"
tasks="cg-ge11-ge13-epi-id-pc-mlee"
gpu=$1
mkdir -p $experiment_dir

python scripts/generate_configs.py $experiment_dir $tasks $gpu
