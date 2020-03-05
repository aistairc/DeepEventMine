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
wget https://b2share.eudat.eu/api/files/6103b075-ef3f-4bbc-949c-93bec7fbda78/cg.tar.gz
tar -xzvf cg.tar.gz

# ge11
wget https://b2share.eudat.eu/api/files/6103b075-ef3f-4bbc-949c-93bec7fbda78/ge11.tar.gz
tar -xzvf ge11.tar.gz

# ge13
wget https://b2share.eudat.eu/api/files/6103b075-ef3f-4bbc-949c-93bec7fbda78/ge13.tar.gz
tar -xzvf ge13.tar.gz

# id
wget https://b2share.eudat.eu/api/files/6103b075-ef3f-4bbc-949c-93bec7fbda78/id.tar.gz
tar -xzvf id.tar.gz

# mlee
wget https://b2share.eudat.eu/api/files/6103b075-ef3f-4bbc-949c-93bec7fbda78/mlee.tar.gz
tar -xzvf mlee.tar.gz

rm *.tar.gz

cd $root_dir