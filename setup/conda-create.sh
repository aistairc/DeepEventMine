#!/bin/sh

ROOT=$PWD
source $ROOT/miniconda3/bin/activate
conda create -n deepeventmine-conda-env python=3.6

echo "Created conda environment: deepeventmine-conda-env"