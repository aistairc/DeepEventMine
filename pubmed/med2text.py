"""Get text from pubmed id"""

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import pubmed_parser as pp
import requests
import tempfile


def medline2text(mlid):
    outputs = pp.parse_xml_web(mlid, save_xml=False)
    return outputs["title"], outputs["abstract"]


def pubmed2text(pmid):
    headers = {
        "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/84.0.4147.105 Safari/537.36"}
    response = requests.get(url=f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmid}/epub/", headers=headers)
    response.raise_for_status()

    with tempfile.NamedTemporaryFile() as epub_file:
        epub_file.write(response.content)
        epub_file.flush()

        article = epub.read_epub(epub_file.name)

        for item in article.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            if pmid[3:] in item.get_name():
                markup = BeautifulSoup(item.get_content(), "lxml-xml")

                title_tag = markup.find("span", class_="article-title")
                title = title_tag.string.strip() if title_tag else "N/A"

                abstract_tag = markup.find("div", class_="abstract")
                abstract = abstract_tag.get_text().strip() if abstract_tag else "N/A"

                body_tag = markup.find("div", class_="body")
                sec_tags = body_tag and body_tag.find_all("div", class_="sec")
                content = "\n".join(sec_tag.get_text().strip() for sec_tag in sec_tags) if sec_tags else "N/A"

                return title, abstract, content


def main():

    # medline
    title, abstract = medline2text("18483370")
    print("Title: \n", title)
    print("Abstract: \n", abstract)

    # pubmed
    title, abstract, content = pubmed2text("PMC441591")
    print("Title: \n", title)
    print("Abstract: \n", abstract)
    print("Content: \n", content)

    return


if __name__ == '__main__':
    main()
