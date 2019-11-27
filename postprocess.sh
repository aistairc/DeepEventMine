#!/bin/bash

corpusdir=data/CG13/test/
preddir=results/cg-predict-test/ev-ann/
outdir=results/cg-predict-test/

python scripts/postprocess.py --corpusdir $corpusdir --indir $preddir --outdir $outdir
