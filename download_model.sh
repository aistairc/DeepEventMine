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
wget https://b2share.eudat.eu/api/files/46da6e84-580e-4a1b-a54c-f24df280c19f/cg.tar.gz
tar -xzvf cg.tar.gz

# ge11
wget https://b2share.eudat.eu/api/files/46da6e84-580e-4a1b-a54c-f24df280c19f/ge11.tar.gz
tar -xzvf ge11.tar.gz

# ge13
wget https://b2share.eudat.eu/api/files/46da6e84-580e-4a1b-a54c-f24df280c19f/ge13.tar.gz
tar -xzvf ge13.tar.gz

# id
wget https://b2share.eudat.eu/api/files/46da6e84-580e-4a1b-a54c-f24df280c19f/id.tar.gz
tar -xzvf id.tar.gz

# mlee
wget https://b2share.eudat.eu/api/files/46da6e84-580e-4a1b-a54c-f24df280c19f/mlee.tar.gz
tar -xzvf mlee.tar.gz

# epi
wget https://b2share.eudat.eu/api/files/46da6e84-580e-4a1b-a54c-f24df280c19f/epi.tar.gz
tar -xzvf epi.tar.gz

# pc
wget https://b2share.eudat.eu/api/files/46da6e84-580e-4a1b-a54c-f24df280c19f/pc.tar.gz
tar -xzvf pc.tar.gz

rm *.tar.gz

cd $root_dir