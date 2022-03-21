#!/bin/bash

ROOT=$PWD
BRAT_DIR="$ROOT/brat"

git clone https://github.com/nlplab/brat.git

cd $BRAT_DIR
./install.sh -u

