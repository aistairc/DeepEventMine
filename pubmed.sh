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

    MY_DATA_DIR=$2

    IN_DIR="data/$MY_DATA_DIR"
    OUT_DIR="data/tmp/"

    python scripts/preprocess.py --indir $IN_DIR --outdir $OUT_DIR

    mv "$OUT_DIR" "$IN_DIR/processed-text"

fi

echo "Done!"

