# DeepEventMine
A model to predict nested events from biomedical texts using our pretrained models.

- The model and results are reported in our paper: [DeepEventMine: End-to-end Neural Nested Event Extraction from Biomedical Texts](https://doi.org/10.1093/bioinformatics/btaa540)
- Bioinformatics, 2020.

## Requirements
- Python 3.6.5
- PyTorch (torch==1.1.0 torchvision==0.3.0, cuda92)
- Install Python packages

```bash
pip install -r requirements.txt
```

## Tasks
1. cg: [Cancer Genetics (CG), 2013](http://2013.bionlp-st.org/tasks/cancer-genetics)
2. ge11: [GENIA Event Extraction (GENIA), 2011](http://2011.bionlp-st.org/home/genia-event-extraction-genia)
3. ge13: [GENIA Event Extraction (GENIA), 2013](http://bionlp.dbcls.jp/projects/bionlp-st-ge-2013/wiki/Overview)
4. id: [Infectious Diseases (ID), 2011](http://2011.bionlp-st.org/home/infectious-diseases)
5. epi: [Epigenetics and Post-translational Modifications (EPI), 2011](http://2011.bionlp-st.org/home/epigenetics-and-post-translational-modifications)
6. pc: [Pathway Curation (PC), 2013](http://2013.bionlp-st.org/tasks/pathway-curation)
7. mlee: [Multi-Level Event Extraction (MLEE)](http://nactem.ac.uk/MLEE/)

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
sh preprocess.sh bionlp
```

3. Download models
- Download SciBERT model from PyTorch AllenNLP
- Download our trained models to predict
```bash
sh download_model.sh
```

4. Generate configs
- If using GPU: [gpu] = 0, otherwise: [gpu] = -1
```bash
sh generate-config.sh [gpu]
```

### Predict (with the BioNLP shared task data sets)

1. For development and test sets (given gold entities)
- CG task: [task] = cg
- PC task: [task] = pc
- Similarly for: ge11, ge13, epi, id, mlee

```bash
sh run.sh [task] predict gold dev
sh run.sh [task] predict gold test
```
- Check the output in the path
```bash
experiments/[task]/predict-gold-dev/
experiments/[task]/predict-gold-test/
```

### Evaluate

1. Retrieve the original offsets and create zip format
```bash
sh run.sh [task] offset gold dev
sh run.sh [task] offset gold test
```

2. Submit the zipped file to the shared task evaluation sites:

- [CG Test](http://weaver.nlplab.org/~bionlp-st/BioNLP-ST-2013/CG/submission/)
- [GE11 Test](http://bionlp-st.dbcls.jp/GE/2011/eval-test/), [GE11 Devel](http://bionlp-st.dbcls.jp/GE/2011/eval-development/)
- [GE13 Test](http://bionlp-st.dbcls.jp/GE/2013/eval-test/), [GE13 Devel](http://bionlp-st.dbcls.jp/GE/2013/eval-development/)
- [ID Test](http://weaver.nlplab.org/~bionlp-st/BioNLP-ST/ID/test-eval.html), [ID Devel](http://weaver.nlplab.org/~bionlp-st/BioNLP-ST/ID/devel-eval.htm)
- [EPI Test](http://weaver.nlplab.org/~bionlp-st/BioNLP-ST/EPI/test-eval.html), [EPI Devel](http://weaver.nlplab.org/~bionlp-st/BioNLP-ST/EPI/devel-eval.htm)
- [PC Test](http://weaver.nlplab.org/~bionlp-st/BioNLP-ST-2013/PC/submission/)

3. Evaluate events

- Evaluate event prediction for PC and CG tasks on the development sets using the shared task scripts.
- Evaluation options: s (softboundary), p(partialrecursive)

```bash
sh run.sh [task] eval gold dev sp
```

4. Supplemenary data

- [Our trained models](https://b2share.eudat.eu/records/80d2de0c57d64419b722dc1afa375f28)
- [Our scores](https://b2share.eudat.eu/api/files/3cf6c1f4-5eed-4ee3-99c5-d99f5f011be3/scores.tar.gz)

### Predict (with raw text)

1. Prepare raw text input

- If you want to predict for your raw text using our trained model for a task ([task] = cg, pc, ge11, etc), put your raw text as the following path

```bash
data/raw-text/[task]/PMID-*.txt
```

2. Preprocess raw text

- Tokenize raw text and prepare data for prediction
```bash
sh preprocess.sh raw
```

3. Predict
```bash
sh run.sh [task] predict raw text
```

- Check the output in
```bash
experiments/[task]/predict-raw-text/
```

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