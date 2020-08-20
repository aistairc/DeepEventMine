"""Get text from pubmed id"""
import os
import urllib3
import requests
import sys

sys.path.insert(0, '.')
from pubmed import med2text


def process_pmc(pmcid, textdir):
    print(pmcid)
    title, abstract, sections_ = med2text.pmc2text(pmcid)

    if len(title) > 0:
        ftitle = ''.join(['PMC-', pmcid.replace('PMC', ''), '-0', str(0), '-', 'TIAB', '.txt'])
        with open(os.path.join(textdir, ftitle), 'w') as fo:
            fo.write(title)
            fo.write('\n\n')
            fo.write(abstract)

    if len(sections_) > 0:

        for sec_id, sec_data in enumerate(sections_):
            sec_title = sec_data[0].strip()

            # shorten the section title
            sec_title_words = sec_title.split(' ')
            if len(sec_title_words) > 5:
                sec_title = '_'.join(sec_title_words[:5])
            else:
                sec_title = sec_title.replace(' ', '_')

            sec_text = sec_data[1]
            if sec_id < 9:
                ftitle = ''.join(
                    ['PMC-', pmcid.replace('PMC', ''), '-0', str(sec_id + 1), '-', sec_title, '.txt'])
            else:
                ftitle = ''.join(
                    ['PMC-', pmcid.replace('PMC', ''), '-', str(sec_id + 1), '-', sec_title, '.txt'])
            with open(os.path.join(textdir, ftitle), 'w') as fo:
                fo.write(sec_text)

    print('Done', pmcid)


def pmids2text(pmid_path, textdir, nloop=20):
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

        print(pmid)
        title, abstract = med2text.pmid2text(pmid)

        if len(title) > 0:
            with open(os.path.join(textdir, ''.join(['PMID-', pmid, '.txt'])), 'w') as fo:
                fo.write(title)
                fo.write('\n\n')
                fo.write(abstract)
            print('Done', pmid)

    # get text given PMCID
    # store list of error pmid to try again
    err_pmids_ = []

    for pmcid in pmcid_list:
        is_err = True
        try:

            process_pmc(pmcid, textdir)
            is_err = False

        except urllib3.exceptions.ProtocolError as error:
            print('Protocol Error', pmcid)
        except requests.exceptions.ConnectionError as error:
            print('Connection Error', pmcid)
        except requests.exceptions.HTTPError as error:
            print('HTTP Error', pmcid)

        # add error id
        if is_err:
            err_pmids_.append(pmcid)

    # loop for error id with maximum nloop (default 20 times)
    for iloop in range(0, nloop):

        print('Error PMC IDs: ', err_pmids_)
        print('Try to collect again, time: ', iloop + 1)

        if len(err_pmids_) == 0:
            break

        tmp_err_pmids_ = []

        for pmcid in err_pmids_:

            is_err = True
            try:

                process_pmc(pmcid, textdir)
                is_err = False

            except urllib3.exceptions.ProtocolError as error:
                print('Protocol Error', pmcid)
            except requests.exceptions.ConnectionError as error:
                print('Connection Error', pmcid)
            except requests.exceptions.HTTPError as error:
                print('HTTP Error', pmcid)

            # add error id
            if is_err:
                tmp_err_pmids_.append(pmcid)

        if len(tmp_err_pmids_) == 0:
            break
        else:
            err_pmids_ = tmp_err_pmids_

    return


def pmid2text(pmid, textdir):
    if not os.path.exists(textdir):
        os.makedirs(textdir)

    if os.path.exists(textdir):
        os.system('rm ' + textdir + 'PMID-' + pmid + '*')

    # get text given each PMID, write to file
    print(pmid)
    title, abstract = med2text.pmid2text(pmid)

    if len(title) > 0:
        with open(os.path.join(textdir, ''.join(['PMID-', pmid, '.txt'])), 'w') as fo:
            fo.write(title)
            fo.write('\n\n')
            fo.write(abstract)
        print('Done', pmid)

    return


def pmcid2text(pmcid, textdir, nloop=20):
    if not os.path.exists(textdir):
        os.makedirs(textdir)

    is_err = True
    try:

        if os.path.exists(textdir):
            os.system('rm ' + textdir + pmcid.replace('PMC', 'PMC-') + '*')

        process_pmc(pmcid, textdir)
        is_err = False

    except urllib3.exceptions.ProtocolError as error:
        print('Protocol Error', pmcid)
    except requests.exceptions.ConnectionError as error:
        print('Connection Error', pmcid)
    except requests.exceptions.HTTPError as error:
        print('HTTP Error', pmcid)

    # try again if error
    if is_err:
        iloop = 0

        # try until successful or exceed a maximum time (default=20 times)
        while is_err and iloop < nloop:
            print('Error PMC ID: ', pmcid)
            print('Try to collect again, time: ', iloop + 1)

            is_err = True
            iloop += 1
            try:

                if os.path.exists(textdir):
                    os.system('rm ' + textdir + pmcid.replace('PMC', 'PMC-') + '*')

                process_pmc(pmcid, textdir)
                is_err = False

            except urllib3.exceptions.ProtocolError as error:
                print('Protocol Error', pmcid)
            except requests.exceptions.ConnectionError as error:
                print('Connection Error', pmcid)
            except requests.exceptions.HTTPError as error:
                print('HTTP Error', pmcid)


if __name__ == '__main__':
    # pmid2text('../data/my-pubmed/pmid.txt', '../data/my-pubmed/original_text/')
    # pmcid2text('PMC4353630', '../data/my-pubmed/original_text/')

    option = sys.argv[1]

    # pubmed id list
    if option == 'pmids':
        pmids2text(sys.argv[2], sys.argv[3])

    elif option == 'pmid':
        pmid2text(sys.argv[2], sys.argv[3])

    elif option == 'pmcid':
        pmcid2text(sys.argv[2], sys.argv[3])
