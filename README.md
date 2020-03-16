# DeepEventMine
DeepEventMine: End-to-end Neural Nested Event Extraction from Biomedical Texts

A model to predict nested events from biomedical texts using our pretrained models.

## Requirements
- Python 3.7.0
- PyTorch
- Python packages: regex, bs4, cchardet, gdown

## How to run

### Prepare data
1. Download corpora
- To download the original data sets from BioNLP shared tasks.
```bash
sh download_corpora.sh
```

2. Preprocess data
- Tokenize texts and prepare data for prediction
```bash
sh preprocess.sh
```

3. Download models
- Download SciBERT model from PyTorch AllenNLP
- Download our trained models to predict
```bash
sh download_model.sh
```


### Predict

- On development and test sets. For instance: CG task

```bash
python predict.py --yaml configs/cg-predict-dev.yaml
python predict.py --yaml configs/cg-predict-test.yaml
```

- Similarly, for other tasks.

### Postprocess output and evaluate

1. Postprocess
- Retrieve the original offsets
- Create a zipped file as the required format
```bash
sh postprocess.sh
```

2. Evaluate online

Submit the zipped file to the shared task evaluation sites:

- CG: [Test set evaluation](http://weaver.nlplab.org/~bionlp-st/BioNLP-ST-2013/CG/submission/)
- GE11: [Test set evaluation](http://bionlp-st.dbcls.jp/GE/2011/eval-test/), [Development set evaluation](http://bionlp-st.dbcls.jp/GE/2011/eval-development/)
- GE13: [Test set evaluation](http://bionlp-st.dbcls.jp/GE/2013/eval-test/), [Development set evaluation](http://bionlp-st.dbcls.jp/GE/2013/eval-development/)
- ID11: [Test set evaluation](http://weaver.nlplab.org/~bionlp-st/BioNLP-ST/ID/test-eval.html), [Development set evaluation](http://weaver.nlplab.org/~bionlp-st/BioNLP-ST/ID/devel-eval.htm)
- EPI: [Test set evaluation](http://weaver.nlplab.org/~bionlp-st/BioNLP-ST/EPI/test-eval.html), [Development set evaluation](http://weaver.nlplab.org/~bionlp-st/BioNLP-ST/EPI/devel-eval.htm)
- PC: [Test set evaluation](http://weaver.nlplab.org/~bionlp-st/BioNLP-ST-2013/PC/submission/)

3. Evaluate using the shared task script

```bash
sh evaluate_event.sh
```

4. Supplemenary data

- Trained models: [link](https://b2share.eudat.eu/records/a207fc06b1d04180a526fd85332e0fb2)
- Scores: [link](https://b2share.eudat.eu/api/files/46da6e84-580e-4a1b-a54c-f24df280c19f/scores.tar.gz)

## Acknowledgements
This work is based on results obtained from a project commissioned by the New Energy and Industrial Technology Development Organization (NEDO).
This work is also supported by PRISM (Public/Private R&D Investment Strategic Expansion PrograM).