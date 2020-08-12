#!/bin/bash

SETTING=$1

# bionlp data
if [ "$SETTING" = "bionlp" ]; then

    input_dir=data/original_corpora/
    output_dir=data/corpora/

    python scripts/preprocess.py --indir $input_dir --outdir $output_dir

# raw text
elif [ "$SETTING" = "raw" ]; then

    input_dir=data/raw-text/
    output_dir=data/processed-raw-text/

    python scripts/preprocess.py --indir $input_dir --outdir $output_dir

fi

echo "Done!"



