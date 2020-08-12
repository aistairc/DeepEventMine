#!/bin/bash

CORPUS_NAME=$1 # cg, ge11, pc, etc
TASK=$2 # predict, eval
GOLD_E2E=$3 # gold, e2e
DEV_TEST=$4 # predict for dev, test sets

experiment_dir="experiments/$CORPUS_NAME"
log_dir="$experiment_dir/logs"
mkdir -p $log_dir

# predict
if [ "$TASK" = "predict" ]; then
    echo "Predict: "

    # predict
    python predict.py --yaml $experiment_dir/configs/$TASK-$GOLD_E2E-$DEV_TEST.yaml

# evaluate
elif [ "$TASK" = "eval" ]; then

    echo "Evaluate: "

    # paths
    REFDIR="data/corpora/$CORPUS_NAME/$DEV_TEST/" # reference gold data
    PREDDIR="$experiment_dir/predict-$GOLD_E2E-$DEV_TEST/ev-last/ev-ann/"
    ZIPDIR="$experiment_dir/predict-$GOLD_E2E-$DEV_TEST/ev-last/" # retrieve the original offsets

    # raw text
    if [ "$GOLD_E2E" = "raw" ]; then
        REFDIR="data/processed-raw-text/$CORPUS_NAME/" # reference gold data
    fi

    # retrieve the original offsets and create zip format for online evaluation
    python scripts/postprocess.py --corpusdir $REFDIR --indir $PREDDIR --outdir $ZIPDIR --corpus_name $CORPUS_NAME --dev_test $DEV_TEST

fi

echo "Done!"