# 1. DeepEventMine
A deep leanring model to predict named entities, triggers, and nested events from biomedical texts.

- The model and results are reported in our paper:

[DeepEventMine: End-to-end Neural Nested Event Extraction from Biomedical Texts](https://doi.org/10.1093/bioinformatics/btaa540), Bioinformatics, 2020.

## 1.1. Features
- Based on [pre-trained BERT](https://github.com/allenai/scibert)
- Predict nested entities and nested events
- Provide our trained models on the seven biomedical tasks
- Reproduce the results reported in our [Bioinformatics](https://doi.org/10.1093/bioinformatics/btaa540) paper
- Predict for new data given raw text input or PubMed ID
- Visualize the predicted entities and events on the [brat](http://brat.nlplab.org)

## 1.2. Tasks

- DeepEventMine has been trained and evaluated on the following tasks (six BioNLP shared tasks and MLEE).

1. cg: [Cancer Genetics (CG), 2013](http://2013.bionlp-st.org/tasks/cancer-genetics)
2. ge11: [GENIA Event Extraction (GENIA), 2011](http://2011.bionlp-st.org/home/genia-event-extraction-genia)
3. ge13: [GENIA Event Extraction (GENIA), 2013](http://bionlp.dbcls.jp/projects/bionlp-st-ge-2013/wiki/Overview)
4. id: [Infectious Diseases (ID), 2011](http://2011.bionlp-st.org/home/infectious-diseases)
5. epi: [Epigenetics and Post-translational Modifications (EPI), 2011](http://2011.bionlp-st.org/home/epigenetics-and-post-translational-modifications)
6. pc: [Pathway Curation (PC), 2013](http://2013.bionlp-st.org/tasks/pathway-curation)
7. mlee: [Multi-Level Event Extraction (MLEE)](http://nactem.ac.uk/MLEE/)

## 1.3. Our trained models and scores

- [Our trained models](https://b2share.eudat.eu/records/80d2de0c57d64419b722dc1afa375f28)
- [Our scores](https://b2share.eudat.eu/api/files/3cf6c1f4-5eed-4ee3-99c5-d99f5f011be3/scores.tar.gz)

# 2. Preparation
## 2.1. Requirements
- Python 3.6.5
- PyTorch (torch==1.1.0 torchvision==0.3.0, cuda92)
- Install Python packages

```bash
sh install.sh
```

## 2.2. BERT
- Download SciBERT BERT model from PyTorch AllenNLP

```bash
sh download.sh bert
```

## 2.3. DeepEventMine
- Download  pre-trained DeepEventMine model on a given task

```bash
sh download.sh deepeventmine [task]
```

# 3. Predict (BioNLP tasks)

## 3.1. Prepare data
1. Download corpora
- To download the original data sets from BioNLP shared tasks.
- [task] = cg, pc, ge11, etc

```bash
sh download.sh bionlp [task]
```

2. Preprocess data
- Tokenize texts and prepare data for prediction
```bash
sh preprocess.sh bionlp
```

3. Generate configs
- If using GPU: [gpu] = 0, otherwise: [gpu] = -1
- [task] = cg, pc, etc
```bash
sh run.sh config [task] [gpu]
```

## 3.2. Predict

1. For development and test sets (given gold entities)
- CG task: [task] = cg
- PC task: [task] = pc
- Similarly for: ge11, ge13, epi, id, mlee

```bash
sh run.sh predict [task] gold dev
sh run.sh predict [task] gold test
```
- Check the output in the path
```bash
experiments/[task]/predict-gold-dev/
experiments/[task]/predict-gold-test/
```

## 3.3. Evaluate

1. Retrieve the original offsets and create zip format
```bash
sh run.sh offset [task] gold dev
sh run.sh offset [task] gold test
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
sh run.sh eval [task] gold dev sp
```

# 4. End-to-end

- Abstract
```bash
sh pubmed.sh e2e pmid 1370299 cg 0
```

- Full text
```bash
sh pubmed.sh e2e pmcid PMC4353630 cg 0
```

- Input: [PMID: 1370299](https://pubmed.ncbi.nlm.nih.gov/1370299/),  [PMCID: PMC4353630](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4353630/) (a single PubMed ID to get raw text)
- Model to predict: DeepEventMine trained on [cg (Cancer Genetics 2013)](http://2013.bionlp-st.org/tasks/cancer-genetics), (other options: pc, ge11, etc)
- GPU: 0 (if CPU: -1)
- Output: in brat format and [brat visualization](http://brat.nlplab.org)

```bash
T9	Gene_or_gene_product 1184 1188	u-PA
T23	Positive_regulation 1157 1166	increases
E9	Positive_regulation:T23 Theme:T9

T10	Gene_or_gene_product 1320 1325	c-src
T24	Gene_expression 1326 1336	expression
E10	Gene_expression:T24 Theme:T10

T32	Positive_regulation 1310 1319	increased
E24	Positive_regulation:T32 Theme:E10
```

<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/aistairc/DeepEventMine/master/img/PMID-1370299.png" width="900"/>
    <br>
<p>



# 5. Predict for new data

- Input: your own raw text or PubMed ID
- Output: predicted entities and events in brat format

## 5.1. Raw text

- You can prepare your own raw text in the following path, E.g [your_data_name] = my-pubmed

```bash
data/[your_data_name]/text/PMID-*.txt
data/[your_data_name]/text/PMC-*.txt
```

## 5.2. PubMed ID

- Or, you can automatically get raw text given PubMed ID or PMC ID

### Get raw text

1. PubMed ID list
- In order to get full text given PMC ID, the text should be available in ePub (for our current version).
- Prepare your list of PubMed ID and PMC ID in the path, E.g [your_data_name] = my-pubmed

```bash
data/[your_data_name]/my-pmid.txt
```

- Get text from the PubMed ID
```bash
sh pubmed.sh pmids
```

2. PubMed ID
- You can also get text by directly input a PubMed or PMC ID
```bash
sh pubmed.sh pmid 1370299
sh pubmed.sh pmcid PMC4353630
```

### Preprocess

- E.g: [your_data_name] = my-pubmed

```bash
sh pubmed.sh preprocess [your_data_name]
```

## 5.3. Predict

1. Generate config
- Generate config for prediction
- The data to predict: [your_data_name] = my-pubmed
- The trained model used for predict: [model_name] = cg (or pc, ge11, etc)
- If you use gpu [gpu]=0, otherwise [gpu]=-1

```bash
sh pubmed.sh config [your_data_name] [model_name] [gpu]
```

2. Predict

- [your_data_name] = my-pubmed

```bash
sh pubmed.sh predict [your_data_name]
```

3. Retrieve the original offsets

```bash
sh pubmed.sh offset [your_data_name]
```

- Check the output in
```bash
experiments/[your_data_name]/results/ev-last/[your_data_name]-brat
```

# 6. Visualization

- Visualize the output using the [brat](http://brat.nlplab.org)

## 6.1. Install brat

- Download [brat v1.3](http://brat.nlplab.org)

```bash
sh download.sh brat
```

- Install brat based on the [brat instructions](http://brat.nlplab.org/installation.html)
```bash
cd brat/brat-v1.3_Crunchy_Frog/
./install.sh -u
python2 standalone.py
```

## 6.2. Prepare data

- Copy the predicted data into the brat folder to visualize
- For the raw text prediction: [your_data_name] = my-pubmed, [model_name]=cg
```bash
sh pubmed.sh brat [your_data_name] [model_name]
```

- Or for the shared task
```bash
sh run.sh brat [task] gold dev
sh run.sh brat [task] gold test
```

## 6.3. Visualize

- The data to visualize is located in

```bash
brat/brat-v1.3_Crunchy_Frog/data/[your_data_name]-brat
brat/brat-v1.3_Crunchy_Frog/data/[task]-brat
```

# 7. Acknowledgements
This work is based on results obtained from a project commissioned by the New Energy and Industrial Technology Development Organization (NEDO).
This work is also supported by PRISM (Public/Private R&D Investment Strategic Expansion PrograM).

# 8. Citation
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