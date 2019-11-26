"""Process batch data for each sentence."""

import collections
from collections import OrderedDict
import numpy as np


def calculate_offset(sentences, i):
    """
    Get the offset for each word in
    the ith sentence.
    """
    offsets = []
    sentence = sentences[i]
    words = sentence.split(' ')
    if i == 0:
        start_pos = 0
    else:
        start_pos = len('\n'.join(sentences[0:i])) + 1

    for j, word in enumerate(words):
        if j == 0:
            start = start_pos
        else:
            start = start_pos + len(' '.join(words[0:j])) + 1

        offsets.append([start, start + len(word)])

    return offsets, words


def prep_sentence_offsets(sentences0):
    sentences_ = []
    sent_words = []
    words_ = []
    sentences1 = OrderedDict()
    sent_lens = []
    for pmid in sentences0:
        sentences = sentences0[pmid]
        sentences_.extend(sentences)

        doc_data = []
        for xx, sentence in enumerate(sentences):
            offsets, words = calculate_offset(sentences, xx)

            sent_lens.append(len(words))
            sent_words.append(words)
            words_.extend(words)

            doc_data.append({
                'sentence': sentence,
                'words': words,
                'offsets': offsets
            })

        sentences1[pmid] = doc_data

    max_sent_len = np.max(sent_lens)

    sentences2 = OrderedDict()
    sentences2['doc_data'] = sentences1
    sentences2['sentences'] = sentences_
    sentences2['sent_words'] = sent_words
    sentences2['words'] = words_
    sentences2['max_sent_len'] = max_sent_len

    return sentences2


def process_input(input0):
    for pmid in input0:
        sentences_data = input0[pmid]

        for sid, sentence in enumerate(sentences_data):
            eids = sentence['eids']
            readable_ents = sentence['readable_ents']

            readable_entsA = OrderedDict()
            read_temp = OrderedDict()

            for ee1 in eids:
                if ee1.startswith('TR'):
                    readable_entsA[ee1] = readable_ents[ee1]
                else:
                    read_temp[ee1] = readable_ents[ee1]
            readable_entsB = OrderedDict()
            readable_entsB.update(read_temp)
            readable_entsB.update(readable_entsA)

            r_idxs = OrderedDict()

            sentence['idx'] = r_idxs

            sent_evs = OrderedDict()

            sentence['readable_ev'] = sent_evs

            trigger_ev = collections.defaultdict(list)

            sentence['trigger_ev'] = trigger_ev

    input1 = OrderedDict()
    for pmid in input0:
        for sid, sentence in enumerate(input0[pmid]):
            sent_id = pmid + ':' + 'sent' + str(sid)
            input1[sent_id] = sentence

    return input1
