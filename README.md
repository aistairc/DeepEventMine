# DeepEventMine
DeepEventMine: End-to-end Neural Nested Event Extraction from Biomedical Texts

## Requirements
- Python 3.7.0
- PyTorch

## Prepare data

1. **BERT: [PyTorch AllenNLP Models](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models/scibert_scivocab_cased.tar)
2. **Test data sets:
- CG: [Cancer Genetics task](http://2013.bionlp-st.org/tasks/cancer-genetics)
- GE11: [GENIA Event Extraction](http://2011.bionlp-st.org/home/genia-event-extraction-genia)
- GE13: [Genia event extraction task](http://bionlp.dbcls.jp/projects/bionlp-st-ge-2013/wiki)
- ID11: [Infectious Diseases Task](http://2011.bionlp-st.org/home/infectious-diseases)

## Predict

```bash
python predict.py --yaml configs/cg-predict.yaml
```