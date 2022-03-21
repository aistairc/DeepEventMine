#!/bin/bash

TASK=$1 # generate configs, predict, retrieve offsets, evaluate
CORPUS_NAME=$2 # cg, ge11, pc, etc

TASK_DIR="experiments/$CORPUS_NAME"
LOG_DIR="$TASK_DIR/logs"
mkdir -p $LOG_DIR

# configs
if [ "$TASK" = "config" ]; then

    GPU=$3
    
    python scripts/generate_configs_bio.py $TASK_DIR $CORPUS_NAME $GPU

# predict
elif [ "$TASK" = "predict" ]; then
    echo "Predict: "
    
    GOLD_E2E=$3 # gold, e2e
    DEV_TEST=$4 # predict for dev, test sets

    # predict
    python scripts/predict_bio.py --yaml $TASK_DIR/deepem-bionlp/configs/$TASK-$GOLD_E2E-$DEV_TEST.yaml

# retrieve offset
elif [ "$TASK" = "offset" ]; then

    echo "Retrieve original offsets: "
    
    GOLD_E2E=$3 # gold, e2e
    DEV_TEST=$4 # predict for dev, test sets

    # paths
    REFDIR="data/corpora/$CORPUS_NAME/$DEV_TEST/" # reference gold data
    PREDDIR="$TASK_DIR/predict-$GOLD_E2E-$DEV_TEST/ev-last/ev-tok-a2/"
    OUTDIR="$TASK_DIR/predict-$GOLD_E2E-$DEV_TEST/ev-last/" # retrieve the original offsets

    # retrieve the original offsets and create zip format for online evaluation
    python scripts/postprocess.py $REFDIR $PREDDIR $OUTDIR $CORPUS_NAME $DEV_TEST

# evaluate
elif [ "$TASK" = "eval" ]; then

    echo "Evaluate:"

    GOLD_E2E=$3 # gold, e2e
    DEV_TEST=$4 # predict for dev, test sets
    EVAL_OPTION=$5 # s: softboundary; p: partialrecursive

    # paths
    REFDIR="data/original_corpora/$CORPUS_NAME/$DEV_TEST/" # reference gold data
    PREDDIR="$TASK_DIR/predict-$GOLD_E2E-$DEV_TEST/ev-last/ev-orig-a2/"



    python eval/scripts/eval-ev-$CORPUS_NAME.py -r $REFDIR -d $PREDDIR -$EVAL_OPTION

# prepare data for brat
elif [ "$TASK" = "brat" ]; then

    echo "Prepare data for brat"

    GOLD_E2E=$3 # gold, e2e
    DEV_TEST=$4 # predict for dev, test sets

    PRED_DIR="$TASK_DIR/predict-$GOLD_E2E-$DEV_TEST/ev-last/$CORPUS_NAME-brat/"
    BRAT_DIR="brat/brat-v1.3_Crunchy_Frog/data/"

    # annotation file
    CONFIG="configs/brat/$CORPUS_NAME"
    if [ -d $CONFIG ]; then
        cp $CONFIG/* $PRED_DIR
    fi

    # brat
    cp -r $PRED_DIR $BRAT_DIR

fi

echo "Done!"