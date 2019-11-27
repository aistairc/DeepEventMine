#!/bin/bash

refdir=data/corpora/CG13/dev/
preddir=results/cg-predict-dev/ev-ann/
#scorepath=results/cg-predict-dev/ev-score.txt

python scripts/eval-EV-CG13.py -r $refdir -d $preddir -sp
#python scripts/eval-EV-CG13.py -r $refdir -d $preddir -sp > $scorepath
