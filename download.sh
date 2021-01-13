#!/bin/bash

DATA_TYPE=$1

ROOT_DIR=$PWD

# BIONLP SHARED TASK DATA
if [ "$DATA_TYPE" = "bionlp" ]; then
    echo "Download the BioNLP shared task data sets: "

    CORPUS_NAME=$2


    if [ "$CORPUS_NAME" = "cg" ]; then
        # CG13 -----------------------------
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

        cd $ROOT_DIR

    elif [ "$CORPUS_NAME" = "ge13" ]; then

        # GE13 -----------------------------
        GE13_DIR="data/original_corpora/ge13"

        mkdir -p $GE13_DIR

        cd $GE13_DIR

        wget http://2013.bionlp-st.org/tasks/BioNLP-ST-2013_GE_train_data_rev3.tar.gz
        wget http://2013.bionlp-st.org/tasks/BioNLP-ST-2013_GE_devel_data_rev3.tar.gz
        wget http://2013.bionlp-st.org/tasks/BioNLP-ST-2013_GE_test_data_rev1.tar.gz

        tar -xzvf BioNLP-ST-2013_GE_train_data_rev3.tar.gz
        tar -xzvf BioNLP-ST-2013_GE_devel_data_rev3.tar.gz
        tar -xzvf BioNLP-ST-2013_GE_test_data_rev1.tar.gz

        mv BioNLP-ST-2013_GE_train_data_rev3 train
        mv BioNLP-ST-2013_GE_devel_data_rev3 dev
        mv BioNLP-ST-2013_GE_test_data_rev1 test

        rm *.tar.gz

        cd $ROOT_DIR

    elif [ "$CORPUS_NAME" = "ge11" ]; then

        # GE11 -----------------------------
        GE11_DIR="data/original_corpora/ge11"

        mkdir -p $GE11_DIR

        cd $GE11_DIR

        wget http://weaver.nlplab.org/~bionlp-st/BioNLP-ST/downloads/files/BioNLP-ST_2011_genia_train_data_rev1.tar.gz
        wget http://weaver.nlplab.org/~bionlp-st/BioNLP-ST/downloads/files/BioNLP-ST_2011_genia_devel_data_rev1.tar.gz
        wget http://weaver.nlplab.org/~bionlp-st/BioNLP-ST/downloads/files/BioNLP-ST_2011_genia_test_data.tar.gz

        tar -xzvf BioNLP-ST_2011_genia_train_data_rev1.tar.gz
        tar -xzvf BioNLP-ST_2011_genia_devel_data_rev1.tar.gz
        tar -xzvf BioNLP-ST_2011_genia_test_data.tar.gz

        mv BioNLP-ST_2011_genia_train_data_rev1 train
        mv BioNLP-ST_2011_genia_devel_data_rev1 dev
        mv BioNLP-ST_2011_genia_test_data test

        rm *.tar.gz

        cd $ROOT_DIR

    elif [ "$CORPUS_NAME" = "id" ]; then

        # ID11 -----------------------------
        ID_DIR="data/original_corpora/id"

        mkdir -p $ID_DIR

        cd $ID_DIR

        wget http://weaver.nlplab.org/~bionlp-st/BioNLP-ST/downloads/files/BioNLP-ST_2011_Infectious_Diseases_training_data_rev1.tar.gz
        wget http://weaver.nlplab.org/~bionlp-st/BioNLP-ST/downloads/files/BioNLP-ST_2011_Infectious_Diseases_development_data_rev1.tar.gz
        wget http://weaver.nlplab.org/~bionlp-st/BioNLP-ST/downloads/files/BioNLP-ST_2011_Infectious_Diseases_test_data.tar.gz

        tar -xzvf BioNLP-ST_2011_Infectious_Diseases_training_data_rev1.tar.gz
        tar -xzvf BioNLP-ST_2011_Infectious_Diseases_development_data_rev1.tar.gz
        tar -xzvf BioNLP-ST_2011_Infectious_Diseases_test_data.tar.gz

        mv BioNLP-ST_2011_Infectious_Diseases_training_data_rev1 train
        mv BioNLP-ST_2011_Infectious_Diseases_development_data_rev1 dev
        mv BioNLP-ST_2011_Infectious_Diseases_test_data test

        rm *.tar.gz

        cd $ROOT_DIR

    elif [ "$CORPUS_NAME" = "epi" ]; then

        # EPI -----------------------------
        EPI_DIR="data/original_corpora/epi"

        mkdir -p $EPI_DIR

        cd $EPI_DIR

        wget http://weaver.nlplab.org/~bionlp-st/BioNLP-ST/downloads/files/BioNLP-ST_2011_Epi_and_PTM_training_data_rev1.tar.gz
        wget http://weaver.nlplab.org/~bionlp-st/BioNLP-ST/downloads/files/BioNLP-ST_2011_Epi_and_PTM_development_data_rev1.tar.gz
        wget http://weaver.nlplab.org/~bionlp-st/BioNLP-ST/downloads/files/BioNLP-ST_2011_Epi_and_PTM_test_data.tar.gz

        tar -xzvf BioNLP-ST_2011_Epi_and_PTM_training_data_rev1.tar.gz
        tar -xzvf BioNLP-ST_2011_Epi_and_PTM_development_data_rev1.tar.gz
        tar -xzvf BioNLP-ST_2011_Epi_and_PTM_test_data.tar.gz

        mv BioNLP-ST_2011_Epi_and_PTM_training_data_rev1 train
        mv BioNLP-ST_2011_Epi_and_PTM_development_data_rev1 dev
        mv BioNLP-ST_2011_Epi_and_PTM_test_data test

        rm *.tar.gz

        cd $ROOT_DIR

    elif [ "$CORPUS_NAME" = "pc" ]; then

        # PC -----------------------------
        PC_DIR="data/original_corpora/pc"

        mkdir -p $PC_DIR

        cd $PC_DIR

        wget http://2013.bionlp-st.org/tasks/BioNLP-ST_2013_PC_training_data.tar.gz
        wget http://2013.bionlp-st.org/tasks/BioNLP-ST_2013_PC_development_data.tar.gz
        wget http://2013.bionlp-st.org/tasks/BioNLP-ST_2013_PC_test_data.tar.gz

        tar -xzvf BioNLP-ST_2013_PC_training_data.tar.gz
        tar -xzvf BioNLP-ST_2013_PC_development_data.tar.gz
        tar -xzvf BioNLP-ST_2013_PC_test_data.tar.gz

        mv BioNLP-ST_2013_PC_training_data train
        mv BioNLP-ST_2013_PC_development_data dev
        mv BioNLP-ST_2013_PC_test_data test

        rm *.tar.gz

        cd $ROOT_DIR
    fi

# SCI-BERT
elif [ "$DATA_TYPE" = "bert" ]; then

    echo "Download pre-trained BERT"

    BERT_DIR="data/bert"

    mkdir -p $BERT_DIR
    cd $BERT_DIR
    wget https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models/scibert_scivocab_cased.tar
    tar -xvf scibert_scivocab_cased.tar
    rm *.tar
    cd scibert_scivocab_cased
    tar -xzvf weights.tar.gz
    rm *.tar.gz
    cd $ROOT_DIR

# PRE-TRAINED DeepEventMine MODELS
elif [ "$DATA_TYPE" = "deepeventmine" ]; then

    echo "Download pre-trained DeepEventMine models"

    CORPUS_NAME=$2
    MODEL_DIR="data/models"

    mkdir -p $MODEL_DIR
    cd $MODEL_DIR

    if [ "$CORPUS_NAME" = "cg" ]; then

        wget https://b2share.eudat.eu/api/files/3cf6c1f4-5eed-4ee3-99c5-d99f5f011be3/cg.tar.gz
        tar -xzvf cg.tar.gz
        rm *.tar.gz
        cd $ROOT_DIR

    elif [ "$CORPUS_NAME" = "pc" ]; then

        wget https://b2share.eudat.eu/api/files/3cf6c1f4-5eed-4ee3-99c5-d99f5f011be3/pc.tar.gz
        tar -xzvf pc.tar.gz
        rm *.tar.gz
        cd $ROOT_DIR

    elif [ "$CORPUS_NAME" = "ge13" ]; then

        wget https://b2share.eudat.eu/api/files/3cf6c1f4-5eed-4ee3-99c5-d99f5f011be3/ge13.tar.gz
        tar -xzvf ge13.tar.gz
        rm *.tar.gz
        cd $ROOT_DIR

    elif [ "$CORPUS_NAME" = "ge11" ]; then

        wget https://b2share.eudat.eu/api/files/3cf6c1f4-5eed-4ee3-99c5-d99f5f011be3/ge11.tar.gz
        tar -xzvf ge11.tar.gz
        rm *.tar.gz
        cd $ROOT_DIR

    elif [ "$CORPUS_NAME" = "id" ]; then

        wget https://b2share.eudat.eu/api/files/3cf6c1f4-5eed-4ee3-99c5-d99f5f011be3/id.tar.gz
        tar -xzvf id.tar.gz
        rm *.tar.gz
        cd $ROOT_DIR

    elif [ "$CORPUS_NAME" = "epi" ]; then

        wget https://b2share.eudat.eu/api/files/3cf6c1f4-5eed-4ee3-99c5-d99f5f011be3/epi.tar.gz
        tar -xzvf epi.tar.gz
        rm *.tar.gz
        cd $ROOT_DIR

    elif [ "$CORPUS_NAME" = "mlee" ]; then

        wget https://b2share.eudat.eu/api/files/3cf6c1f4-5eed-4ee3-99c5-d99f5f011be3/mlee.tar.gz
        tar -xzvf mlee.tar.gz
        rm *.tar.gz
        cd $ROOT_DIR

    fi

# BRAT FOR VISUALIZATION
elif [ "$DATA_TYPE" = "brat" ]; then
    echo "Download brat"

    wget http://weaver.nlplab.org/~brat/releases/brat-v1.3_Crunchy_Frog.tar.gz
    tar -xzvf brat-v1.3_Crunchy_Frog.tar.gz
    rm *.tar.gz
    mv brat-v1.3_Crunchy_Frog brat

    cd $ROOT_DIR
        
fi

echo "Done!"

