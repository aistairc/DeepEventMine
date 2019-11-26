# DeepEventMine
DeepEventMine: End-to-end Neural Nested Event Extraction from Biomedical Texts

A model to predict nested events from biomedical texts using our pretrained models.

## Requirements
- Python 3.7.0
- PyTorch

## Data Links

### BERT: [PyTorch AllenNLP Models](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models/scibert_scivocab_cased.tar)
### Test data sets:
- CG: [Cancer Genetics task](http://2013.bionlp-st.org/tasks/cancer-genetics)
- GE11: [GENIA Event Extraction](http://2011.bionlp-st.org/home/genia-event-extraction-genia)
- GE13: [Genia event extraction task](http://bionlp.dbcls.jp/projects/bionlp-st-ge-2013/wiki)
- ID11: [Infectious Diseases Task](http://2011.bionlp-st.org/home/infectious-diseases)

## Prepare data

1. BERT: download BERT model into 'data/bert/

2. Test sets:
- Download the original test sets from BioNLP 2011 and BioNLP 2013 shared tasks, then tokenize texts and preprocess data.
- Or, download [our preprocessed data sets](https://drive.google.com/file/d/1Gze6LQr3XC9636eX2MudA_5SW6uoB2VV/view?usp=sharing)
- Put data into, e.g: 'data/CG13/test/'

3. Pretrained models:
- Download our pretrained models
- Put data into, e.g: 'data/models/cg/cg.param' and 'data/models/cg/model/***.pt'

4. Set correct path in the config file, e.g: 'configs/cg-predict.yaml'

## Predict

```bash
python predict.py --yaml configs/cg-predict.yaml
```

## Evaluate

### Evaluate using the shared task scripts, e.g: CG
- [Evaluation script link](http://weaver.nlplab.org/~bionlp-st/BioNLP-ST-2013/CG/tools/evaluation-CG.py)

```bash
python evaluation-CG.py -r data/CG13/dev/ -d results/cg-predict-dev/ev-ann/ -sp
```

### Evaluate online
Submit the zipped file to the shared task evaluation sites:

- CG: [Test set online evaluation](http://weaver.nlplab.org/~bionlp-st/BioNLP-ST-2013/CG/submission/)
- GE11: [Test set online evaluation](http://bionlp-st.dbcls.jp/GE/2011/eval-test/), [Development set online evaluation](http://bionlp-st.dbcls.jp/GE/2011/eval-development/)
- GE13: [Test set online evaluation](http://bionlp-st.dbcls.jp/GE/2013/eval-test/), [Development set online evaluation](http://bionlp-st.dbcls.jp/GE/2013/eval-development/)
- ID11: [Test set online evaluation](http://weaver.nlplab.org/~bionlp-st/BioNLP-ST/ID/test-eval.html), [Development set online evaluation](http://weaver.nlplab.org/~bionlp-st/BioNLP-ST/ID/devel-eval.htm)

