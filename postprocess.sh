#!/bin/bash

corpusdir=data/corpora/CG13/test/
preddir=results/cg/cg-predict-test/ev-ann/
outdir=results/cg/cg-predict-test/

python scripts/postprocess.py --corpusdir $corpusdir --indir $preddir --outdir $outdir