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

- On development and test sets
```bash
python predict.py --yaml configs/cg-predict-dev.yaml
python predict.py --yaml configs/cg-predict-test.yaml
```

### Postprocess output and evaluate

1. Postprocess
- Retrieve the original offsets
- Create a zipped file as the required format
```bash
sh postprocess.sh
```

2. Evaluate online

Submit the zipped file to the shared task evaluation sites:

- CG: [Test](http://weaver.nlplab.org/~bionlp-st/BioNLP-ST-2013/CG/submission/) ([Our CG13-test-single score](https://drive.google.com/file/d/1RaSyfNgpfOVHb7iDnK2PT9K8SEydQDQe/view?usp=sharing),
[Our CG13-test-ensemble score](https://drive.google.com/file/d/1p7GqdvckZ33xFRDomkYoHSVZ3Mpzyp4d/view?usp=sharing))
- GE11: [Test](http://bionlp-st.dbcls.jp/GE/2011/eval-test/), [Development](http://bionlp-st.dbcls.jp/GE/2011/eval-development/)
- GE13: [Test](http://bionlp-st.dbcls.jp/GE/2013/eval-test/), [Development](http://bionlp-st.dbcls.jp/GE/2013/eval-development/)
- ID11: [Test](http://weaver.nlplab.org/~bionlp-st/BioNLP-ST/ID/test-eval.html), [Development](http://weaver.nlplab.org/~bionlp-st/BioNLP-ST/ID/devel-eval.htm)

3. Evaluate using the shared task script

```bash
sh evaluate_event.sh
```
- [Our CG13-dev score](https://drive.google.com/file/d/1u1zKcGavDVEAnAwZq6cI_cxQcqERvoAZ/view?usp=sharing)