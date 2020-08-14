#!/bin/bash

TASK=$1

# pubmed_parser
if [ "$TASK" = "pubmed" ]; then
    echo "Install pubmed_parser and dependencies"
    pip install git+https://github.com/titipata/pubmed_parser.git
    pip install ebooklib beautifulsoup4 lxml
fi

echo "Done!"

