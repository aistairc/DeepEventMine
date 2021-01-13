#!/bin/bash

HOME=$PWD

# dependencies
pip install --upgrade pip
pip install -r requirements.txt

# we used an older sklearn version (temporarily install the following; we may update an alternative way in the future)
pip install scikit-learn==0.23.2

# pubmed_parser
echo "Install pubmed_parser and dependencies"
pip install git+https://github.com/titipata/pubmed_parser.git
pip install ebooklib beautifulsoup4 lxml

# brat (for visualization)
echo "Install BRAT for visualization"
git clone https://github.com/nlplab/brat.git
cd brat
./install.sh -u

cd $HOME

echo "Done!"