#!/bin/bash

TASK=$1

# pubmed_parser
if [ "$TASK" = "pmids" ]; then
    echo "Get text from PubMed ID and PMC ID list"

    PMIDS="data/my-pubmed/my-pmid.txt"
    TEXT_DIR="data/my-pubmed/original_text"
    python pubmed/pubmed2text.py $PMIDS $TEXT_DIR
fi

echo "Done!"

