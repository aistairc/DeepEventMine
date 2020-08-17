#!/bin/bash

TASK=$1

# Get Text from PubMed ID & PMC ID
if [ "$TASK" = "pmids" ]; then
    echo "Get text from PubMed ID and PMC ID list"

    PMIDS="data/my-pubmed/my-pmid.txt"
    TEXT_DIR="data/my-pubmed/original_text/"
    python pubmed/pubmed2text.py $TASK $PMIDS $TEXT_DIR

# Get Text from PubMed ID
elif [ "$TASK" = "pmid" ]; then
    echo "Get text from PubMed ID"

    PMID=$2

    TEXT_DIR="data/my-pubmed/original_text/"
    python pubmed/pubmed2text.py $TASK $PMID $TEXT_DIR

fi

# Get Text from PMC ID
elif [ "$TASK" = "pmcid" ]; then
    echo "Get text from PMC ID"

    PMCID=$2

    TEXT_DIR="data/my-pubmed/original_text/"
    python pubmed/pubmed2text.py $TASK $PMCID $TEXT_DIR

fi

echo "Done!"

