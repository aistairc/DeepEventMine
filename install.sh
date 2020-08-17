#!/bin/bash

# dependencies
pip install -r requirements.txt

# pubmed_parser
echo "Install pubmed_parser and dependencies"
pip install git+https://github.com/titipata/pubmed_parser.git
pip install ebooklib beautifulsoup4 lxml

echo "Done!"