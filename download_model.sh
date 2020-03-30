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

# download pretrained models
model_dir="data/models"

mkdir -p $model_dir
cd $model_dir

# cg
wget https://b2share.eudat.eu/api/files/3cf6c1f4-5eed-4ee3-99c5-d99f5f011be3/cg.tar.gz
tar -xzvf cg.tar.gz

# ge11
wget https://b2share.eudat.eu/api/files/3cf6c1f4-5eed-4ee3-99c5-d99f5f011be3/ge11.tar.gz
tar -xzvf ge11.tar.gz

# ge13
wget https://b2share.eudat.eu/api/files/3cf6c1f4-5eed-4ee3-99c5-d99f5f011be3/ge13.tar.gz
tar -xzvf ge13.tar.gz

# id
wget https://b2share.eudat.eu/api/files/3cf6c1f4-5eed-4ee3-99c5-d99f5f011be3/id.tar.gz
tar -xzvf id.tar.gz

# mlee
wget https://b2share.eudat.eu/api/files/3cf6c1f4-5eed-4ee3-99c5-d99f5f011be3/mlee.tar.gz
tar -xzvf mlee.tar.gz

# epi
wget https://b2share.eudat.eu/api/files/3cf6c1f4-5eed-4ee3-99c5-d99f5f011be3/epi.tar.gz
tar -xzvf epi.tar.gz

# pc
wget https://b2share.eudat.eu/api/files/3cf6c1f4-5eed-4ee3-99c5-d99f5f011be3/pc.tar.gz
tar -xzvf pc.tar.gz

rm *.tar.gz

cd $root_dir