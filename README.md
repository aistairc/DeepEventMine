# DeepEventMine
A model to predict nested events from biomedical texts using our pretrained models.

- The model and results are reported in our paper: [DeepEventMine: End-to-end Neural Nested Event Extraction from Biomedical Texts](https://doi.org/10.1093/bioinformatics/btaa540)
- Bioinformatics, 2020.

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

4. Generate configs
```bash
sh generate-config.sh
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
- Create a zipped file as the required format. For instance: CG task, test
```bash
python scripts/postprocess.py --corpusdir data/corpora/cg/test/
--indir results/cg/cg-predict-test/ev-ann/
--outdir results/cg/cg-predict-test/
```

- Similarly for other tasks by changing the corresponding paths.

2. Evaluate online

Submit the zipped file to the shared task evaluation sites:

- [CG Test](http://weaver.nlplab.org/~bionlp-st/BioNLP-ST-2013/CG/submission/)
- [GE11 Test](http://bionlp-st.dbcls.jp/GE/2011/eval-test/), [GE11 Devel](http://bionlp-st.dbcls.jp/GE/2011/eval-development/)
- [GE13 Test](http://bionlp-st.dbcls.jp/GE/2013/eval-test/), [GE13 Devel](http://bionlp-st.dbcls.jp/GE/2013/eval-development/)
- [ID Test](http://weaver.nlplab.org/~bionlp-st/BioNLP-ST/ID/test-eval.html), [ID Devel](http://weaver.nlplab.org/~bionlp-st/BioNLP-ST/ID/devel-eval.htm)
- [EPI Test](http://weaver.nlplab.org/~bionlp-st/BioNLP-ST/EPI/test-eval.html), [EPI Devel](http://weaver.nlplab.org/~bionlp-st/BioNLP-ST/EPI/devel-eval.htm)
- [PC Test](http://weaver.nlplab.org/~bionlp-st/BioNLP-ST-2013/PC/submission/)

3. Evaluate using the shared task script

```bash
sh evaluate_event.sh
```

4. Supplemenary data

- Trained models: [link](https://b2share.eudat.eu/records/80d2de0c57d64419b722dc1afa375f28)
- Scores: [link](https://b2share.eudat.eu/api/files/3cf6c1f4-5eed-4ee3-99c5-d99f5f011be3/scores.tar.gz)

## Acknowledgements
This work is based on results obtained from a project commissioned by the New Energy and Industrial Technology Development Organization (NEDO).
This work is also supported by PRISM (Public/Private R&D Investment Strategic Expansion PrograM).

## Citation
```bash
@article{10.1093/bioinformatics/btaa540,
    author = {Trieu, Hai-Long and Tran, Thy Thy and Duong, Khoa N A and Nguyen, Anh and Miwa, Makoto and Ananiadou, Sophia},
    title = "{DeepEventMine: End-to-end Neural Nested Event Extraction from Biomedical Texts}",
    journal = {Bioinformatics},
    year = {2020},
    month = {06},
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btaa540},
    url = {https://doi.org/10.1093/bioinformatics/btaa540},
    note = {btaa540},
    eprint = {https://academic.oup.com/bioinformatics/article-pdf/doi/10.1093/bioinformatics/btaa540/33399046/btaa540.pdf},
}
```