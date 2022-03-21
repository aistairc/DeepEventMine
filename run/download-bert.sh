#!/bin/bash

root_dir=$PWD

# download BERT model
BERT_DIR="data/bert"
mkdir -p $BERT_DIR
cd $BERT_DIR

wget https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models/scibert_scivocab_cased.tar
tar -xvf scibert_scivocab_cased.tar
rm *.tar

cd scibert_scivocab_cased
tar -xzvf weights.tar.gz
rm *.tar.gz

cd $root_dir
