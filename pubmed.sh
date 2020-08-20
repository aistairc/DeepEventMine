#!/bin/bash

TASK=$1

# Get Text from PubMed ID & PMC ID
if [ "$TASK" = "pmids" ]; then
    echo "Get text from PubMed ID and PMC ID list"

    PMIDS="data/my-pubmed/pmid.txt"
    TEXT_DIR="data/my-pubmed/text/"
    python pubmed/pubmed2text.py $TASK $PMIDS $TEXT_DIR

# Get Text from PubMed ID
elif [ "$TASK" = "pmid" ]; then
    echo "Get text from PubMed ID"

    PMID=$2

    TEXT_DIR="data/my-pubmed/text/"
    python pubmed/pubmed2text.py $TASK $PMID $TEXT_DIR

# Get Text from PMC ID
elif [ "$TASK" = "pmcid" ]; then
    echo "Get text from PMC ID"

    PMCID=$2

    TEXT_DIR="data/my-pubmed/text/"
    python pubmed/pubmed2text.py $TASK $PMCID $TEXT_DIR

# Preprocess
elif [ "$TASK" = "preprocess" ]; then

    echo "Preprocess pubmed text"
    MY_DATA=$2

    IN_DIR="data/$MY_DATA"
    OUT_DIR="data/tmp/"

    python scripts/preprocess.py --indir $IN_DIR --outdir $OUT_DIR

    mv "$OUT_DIR" "$IN_DIR/processed-text"

# Generate config
elif [ "$TASK" = "config" ]; then
    echo "Generate pubmed config"

    MY_DATA=$2
    MODEL_NAME=$3
    GPU=$4
    EXP_DIR="experiments/"

    python scripts/generate_configs.py $EXP_DIR $MY_DATA $MODEL_NAME $GPU

# predict
elif [ "$TASK" = "predict" ]; then
    echo "Predict: "

    MY_DATA=$2
    EXP_DIR="experiments/$MY_DATA"

    # predict
    python predict.py --yaml $EXP_DIR/configs/$TASK-$MY_DATA.yaml

# retrieve offset
elif [ "$TASK" = "offset" ]; then

    echo "Retrieve original offsets: "

    MY_DATA=$2

    # paths
    REFDIR="data/$MY_DATA/processed-text/text" # reference gold data
    PREDDIR="experiments/$MY_DATA/results/ev-last/ev-tok-ann/"
    OUTDIR="experiments/$MY_DATA/results/ev-last/" # retrieve the original offsets

    python scripts/postprocess.py $REFDIR $PREDDIR $OUTDIR $MY_DATA

# prepare data for brat
elif [ "$TASK" = "brat" ]; then

    echo "Prepare data for brat"

    MY_DATA=$2
    MODEL_NAME=$3

    PRED_DIR="experiments/$MY_DATA/results/ev-last/$MY_DATA-brat/"
    BRAT_DIR="brat/brat-v1.3_Crunchy_Frog/data/"

    # annotation file
    CONFIG="configs/brat/$MODEL_NAME"
    if [ -d $CONFIG ]; then
        cp $CONFIG/* $PRED_DIR
    fi

    # brat
    cp -r $PRED_DIR $BRAT_DIR

elif [ "$TASK" = "e2e" ]; then

    echo "End-to-end event extraction"
    echo "--------------------------------"
    echo "1. Prepare input raw text"

    PM_TYPE=$2
    MY_DATA=$3 # a single PMID (e.g 1370299) or data name (e.g my-pubmed)

    # Get Text from a single PubMed ID
    if [ "$PM_TYPE" = "pmid" || "$PM_TYPE" = "pmcid" ] ; then
        echo "Get text from PubMed ID"

        PMID=$MY_DATA
        TEXT_DIR="data/$PMID/$PMID-text/"
        python pubmed/pubmed2text.py $PM_TYPE $PMID $TEXT_DIR

    # list of pmid
    elif [ "$PM_TYPE" = "pmids" ]; then
        echo "Get text from PubMed ID and PMC ID list"

        PMIDS="data/$MY_DATA/pmid.txt"
        TEXT_DIR="data/$MY_DATA/text/"
        python pubmed/pubmed2text.py $PM_TYPE $PMIDS $TEXT_DIR

    # already have raw text
    elif [ "$PM_TYPE" = "rawtext" ]; then
        TEXT_DIR="data/$MY_DATA/text/"
        echo "Process the raw text:" $TEXT_DIR

    fi

    echo "--------------------------------"
    echo "2. Preprocess pubmed text"

    IN_DIR="data/$MY_DATA"
    OUT_DIR="data/tmp/"

    python scripts/preprocess.py --indir $IN_DIR --outdir $OUT_DIR

    mv "$OUT_DIR" "$IN_DIR/processed-text"

    echo "--------------------------------"
    echo "3. Generate pubmed config"

    MODEL_NAME=$4
    GPU=$5
    EXP_DIR="experiments/"

    python scripts/generate_configs.py $EXP_DIR $MY_DATA $MODEL_NAME $GPU

    echo "--------------------------------"
    echo "4. Predict: "

    EXP_DIR="experiments/$MY_DATA"

    # predict
    python predict.py --yaml $EXP_DIR/configs/predict-$MY_DATA.yaml

    echo "--------------------------------"
    echo "5. Retrieve original offsets: "

    # paths
    REFDIR="data/$MY_DATA/processed-text/text" # reference gold data
    PREDDIR="experiments/$MY_DATA/results/ev-last/ev-tok-ann/"
    OUTDIR="experiments/$MY_DATA/results/ev-last/" # retrieve the original offsets

    python scripts/postprocess.py $REFDIR $PREDDIR $OUTDIR $MY_DATA

    echo "--------------------------------"
    echo "6. Prepare data for brat"

    PRED_DIR="experiments/$MY_DATA/results/ev-last/$MY_DATA-brat/"
    BRAT_DIR="brat/brat-v1.3_Crunchy_Frog/data/"

    # annotation file
    CONFIG="configs/brat/$MODEL_NAME"
    if [ -d $CONFIG ]; then
        cp $CONFIG/* $PRED_DIR
    fi

    # brat
    cp -r $PRED_DIR $BRAT_DIR

    echo "Done end-to-end prediction for: " $PMID
    echo "Check the prediction at: " $PRED_DIR
    echo "Data for brat visualization at: " "$BRAT_DIR$MY_DATA-brat"

fi


echo "Done!"

