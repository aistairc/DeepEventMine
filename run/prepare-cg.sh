#!/bin/bash

# to root
ROOT=$PWD

# 1. DOWNLOAD DATA
echo "-----------------------------"
echo "1. Download CG data"

CG_DIR="data/original_corpora/cg"

mkdir -p $CG_DIR

cd $CG_DIR

wget http://2013.bionlp-st.org/tasks/BioNLP-ST_2013_CG_training_data.tar.gz
wget http://2013.bionlp-st.org/tasks/BioNLP-ST_2013_CG_development_data.tar.gz
wget http://2013.bionlp-st.org/tasks/BioNLP-ST_2013_CG_test_data.tar.gz

tar -xzvf BioNLP-ST_2013_CG_training_data.tar.gz
tar -xzvf BioNLP-ST_2013_CG_development_data.tar.gz
tar -xzvf BioNLP-ST_2013_CG_test_data.tar.gz

mv BioNLP-ST_2013_CG_training_data train
mv BioNLP-ST_2013_CG_development_data dev
mv BioNLP-ST_2013_CG_test_data test

rm *.tar.gz

cd $ROOT

# 2. TOKENIZATION
echo "-----------------------------"
echo "2. Tokenization"

input_dir=data/original_corpora/
output_dir=data/corpora/
python scripts/preprocess.py \
  --indir $input_dir \
  --outdir $output_dir

cd $ROOT

# 3. EVENT STRUCTURES
echo "-----------------------------"
echo "3. Download processed event structures"

# our processed event structures data
FILEID=1qzjY38RdnCvomVAD3XMk_J4pi1gIraA5

CG_EVENT_STRUCTURES="data/event-structures"
mkdir -p $CG_EVENT_STRUCTURES
cd $CG_EVENT_STRUCTURES

gdown https://drive.google.com/uc?id=$FILEID
tar -xzvf cg.tar.gz
rm *.tar.gz

cd $ROOT

# DONE
echo "-----------------------------"
echo "Done!"