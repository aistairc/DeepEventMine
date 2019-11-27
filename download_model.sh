#!/bin/bash

root_dir=$PWD


# download BERT model
bert_dir="data/bert"

mkdir -p $bert_dir
cd $bert_dir
wget https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models/scibert_scivocab_cased.tar
tar -xvf scibert_scivocab_cased.tar
rm *.tar
cd scibert_scivocab_cased
tar -xzvf weights.tar.gz
rm *.tar.gz
cd $root_dir

# download pretrained model on CG13
# shared file id
FILEID=1-jSBQSdJP6brC1QIOj_N6RHEzPBR9AYT
model_dir="data/models"

mkdir -p $model_dir
cd $model_dir
gdown https://drive.google.com/uc?id=$FILEID
tar -xzvf cg.tar.gz
rm *.tar.gz

cd $root_dir