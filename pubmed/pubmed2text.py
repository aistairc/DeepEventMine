"""Get text from pubmed id"""
import os
import urllib3
import requests
import sys

from pubmed import med2text


def pmid2text(pmid_path, textdir):
    # read pmid list
    pmid_list = []
    pmcid_list = []

    with open(pmid_path, 'r') as fi:
        for line in fi:

            # pmid
            if line.startswith('PMID:'):
                pmid = line.split('PMID:')[1].strip()
                pmid_list.append(pmid)

            # pmcid
            elif line.startswith('PMCID:'):
                pmcid = line.split('PMCID:')[1].strip()
                pmcid_list.append(pmcid)

    # text dir
    if not os.path.exists(textdir):
        os.makedirs(textdir)
    else:
        os.system('rm ' + textdir + '*.txt')

    # get text given each PMID, write to file
    for pmid in pmid_list:
        title, abstract = med2text.pmid2text(pmid)

        if len(title) > 0:
            with open(os.path.join(textdir, ''.join(['PMID-', pmid, '.txt'])), 'w') as fo:
                fo.write(title)
                fo.write('\n\n')
                fo.write(abstract)
            print('Done', pmid)

    # get text given PMCID
    for pmcid in pmcid_list:
        try:
            title, abstract, content = med2text.pmc2text(pmcid)
            if len(title) > 0:
                with open(os.path.join(textdir, ''.join(['PMC-', pmcid.replace('PMC', ''), '.txt'])), 'w') as fo:
                    fo.write(title)
                    fo.write('\n\n')
                    fo.write(abstract)
                    fo.write('\n\n')
                    fo.write(content)
                print('Done', pmcid)

        except urllib3.exceptions.ProtocolError as error:
            print('Protocol Error', pmcid)
        except ConnectionError as error:
            print('Connection Error', pmcid)
        except requests.exceptions.HTTPError as error:
            print('HTTP Error', pmcid)

    return


if __name__ == '__main__':
    # pmid2text('../data/my-pubmed/pmid.txt', '../data/my-pubmed/original_text/')
    pmid2text(sys.argv[1], sys.argv[2])