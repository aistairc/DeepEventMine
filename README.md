# DeepEventMine
DeepEventMine: End-to-end Neural Nested Event Extraction from Biomedical Texts

A model to predict nested events from biomedical texts using our pretrained models.

## Requirements
- Python 3.7.0
- PyTorch
- For preprocessing data: regex, bs4, cchardet

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
- Donwload our trained models to predict
```bash
sh download_model.sh
```

4. Config
- Correct the data path in the config files 'config/cg-predict.yaml'

### Predict

```bash
python predict.py --yaml configs/cg-predict.yaml
```

### Postprocess output and evaluate

1. Postprocess
- Retrieve the original offsets
- Create a zipped file as the required format

2. Evaluate
- Submit the zipped file to the shared task evaluation sites:

- CG: [Test](http://weaver.nlplab.org/~bionlp-st/BioNLP-ST-2013/CG/submission/)
- GE11: [Test](http://bionlp-st.dbcls.jp/GE/2011/eval-test/), [Development](http://bionlp-st.dbcls.jp/GE/2011/eval-development/)
- GE13: [Test](http://bionlp-st.dbcls.jp/GE/2013/eval-test/), [Development](http://bionlp-st.dbcls.jp/GE/2013/eval-development/)
- ID11: [Test](http://weaver.nlplab.org/~bionlp-st/BioNLP-ST/ID/test-eval.html), [Development](http://weaver.nlplab.org/~bionlp-st/BioNLP-ST/ID/devel-eval.htm)

3. Using the shared task script
```bash
sh evaluate_event.sh
```