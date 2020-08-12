#!/bin/bash

refdir=data/corpora/cg/dev/
preddir=results/cg-predict-dev/ev-ann/
#scorepath=results/cg-predict-dev/ev-score.txt

python eval/scripts/eval-ev-cg.py -r $refdir -d $preddir -sp