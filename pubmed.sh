#!/bin/bash

TASK=$1

# Get Text from PubMed ID & PMC ID
if [ "$TASK" = "pmids" ]; then
    echo "Get text from PubMed ID and PMC ID list"

    PMIDS="data/my-pubmed/my-pmid.txt"
    TEXT_DIR="data/my-pubmed/my-pubmed-text/"
    python pubmed/pubmed2text.py $TASK $PMIDS $TEXT_DIR

# Get Text from PubMed ID
elif [ "$TASK" = "pmid" ]; then
    echo "Get text from PubMed ID"

    PMID=$2

    TEXT_DIR="data/my-pubmed/my-pubmed-text/"
    python pubmed/pubmed2text.py $TASK $PMID $TEXT_DIR

# Get Text from PMC ID
elif [ "$TASK" = "pmcid" ]; then
    echo "Get text from PMC ID"

    PMCID=$2

    TEXT_DIR="data/my-pubmed/my-pubmed-text/"
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

fi


echo "Done!"

