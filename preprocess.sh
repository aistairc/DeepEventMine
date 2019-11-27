#!/bin/bash

# requirement: python3, regex, bs4, cchardet

input_dir=data/original_corpora/
output_dir=data/corpora/

python scripts/preprocess.py --indir $input_dir --outdir $output_dir
