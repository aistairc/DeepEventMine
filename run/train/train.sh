#!/bin/bash

ROOT=$PWD

export PYTHONPATH="${PYTHONPATH}:$ROOT"
export PYTHONPATH="${PYTHONPATH}:$ROOT/model"

CONFIG_PATH=$1

CONFIG_NAME=$(basename "$CONFIG_PATH")
CONFIG_NAME=(${CONFIG_NAME//./ })

CONFIG_DIR=$(dirname "$CONFIG_PATH")
EXP_DIR=$(dirname "$CONFIG_DIR")
LOG_DIR=$EXP_DIR/logs
mkdir -p $LOG_DIR

nohup python -u train.py --yaml $CONFIG_PATH >> $LOG_DIR/$CONFIG_NAME.log &

echo "The training log can be viewed at: $LOG_DIR/$CONFIG_NAME.log"
echo "Please wait several seconds to see it here."

sleep 5
tail -f $LOG_DIR/$CONFIG_NAME.log