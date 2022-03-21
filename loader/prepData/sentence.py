"""Process batch data for each sentence."""

import collections
from collections import OrderedDict
import numpy as np

from loader.prepData.relation import process_relations


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
    # chars_ = []
    sentences1 = OrderedDict()
    sent_lens = []
    for pmid in sentences0:
        sentences = sentences0[pmid]
        sentences_.extend(sentences)

        doc_data = []
        for xx, sentence in enumerate(sentences):
            offsets, words = calculate_offset(sentences, xx)
            # chars = ["".join([w for w in words])]
            # chars2 = [[c for c in w] for w in words]

            sent_lens.append(len(words))
            sent_words.append(words)
            words_.extend(words)
            # chars_.extend(chars)

            doc_data.append({
                'sentence': sentence,
                'words': words,
                # 'chars': chars2,
                'offsets': offsets
            })

        sentences1[pmid] = doc_data

    max_sent_len = np.max(sent_lens)

    sentences2 = OrderedDict()
    sentences2['doc_data'] = sentences1
    sentences2['sentences'] = sentences_
    sentences2['sent_words'] = sent_words
    sentences2['words'] = words_
    # sentences2['chars'] = chars_
    sentences2['max_sent_len'] = max_sent_len

    return sentences2


def process_input(input0, entities0, relations0, events2, params, dirpath):
    emissed = 0

    for pmid in input0:
        sentences_data = input0[pmid]
        relations_data = relations0[pmid]['data']
        events_data = events2['pmids'][pmid]
        # events2_data = events2[pmid]

        # selected = []
        # abst_sents_rels = []
        unk = 0
        added_events = []

        for sid, sentence in enumerate(sentences_data):
            eids = sentence['eids']
            readable_ents = sentence['readable_ents']

            cand_pairs = OrderedDict()
            for idR in relations_data:

                relation = relations_data[idR]
                rol1 = 'Arg1'
                rol2 = 'Arg2'
                arg1 = relation['arg1id']
                arg2 = relation['arg2id']
                typeR = relation['type']
                idR = relation['id']
                p = (idR, typeR)
                pair = [(rol1, arg1), (rol2, arg2)]

                if arg1 in eids and arg2 in eids:
                    # selected.append(p)
                    cand_pairs[p] = pair

            sentence['rels'] = cand_pairs
            true_relations = cand_pairs

            # RELATIONS
            readable_entsA = OrderedDict()
            read_temp = OrderedDict()

            for ee1 in eids:
                if ee1.startswith('TR'):
                    readable_entsA[ee1] = readable_ents[ee1]  # triggers
                else:
                    read_temp[ee1] = readable_ents[ee1]  # entities
            readable_entsB = OrderedDict()  # augment with triggers for trig-trig pairs
            readable_entsB.update(read_temp)
            readable_entsB.update(readable_entsA)

            r_idxs, readable_rels = process_relations(readable_entsA, readable_entsB, readable_ents, true_relations,
                                                      unk,
                                                      params)

            sentence['readable_r'] = readable_rels
            sentence['idx'] = r_idxs

            sent_evs = OrderedDict()
            for idE in events_data:
                event = events_data[idE]
                idTR = event['trid']
                if event['args_num'] == 0:
                    if idTR in sentence['idx']:
                        event['rel'] = {}
                        sent_evs[idE] = event
                else:

                    args_data = event['args_data']
                    isEvent = True
                    rels = OrderedDict()
                    for xx, arg1 in enumerate(args_data):
                        typeR = arg1[0]
                        idArg = arg1[1]
                        if idArg in events_data and idTR in sentence['idx']:
                            # argument is event
                            argEv = events_data[idArg]
                            idArg2 = argEv['trid']
                            if (idTR, idArg2) in readable_rels:
                                rel_data = readable_rels[(idTR, idArg2)]
                                if typeR in rel_data[1]:
                                    rels[(idTR, idArg2)] = [rel_data[0], typeR]
                                    continue
                                else:
                                    isEvent = False
                                    break
                            else:
                                isEvent = False
                                break

                        elif (idTR, idArg) in readable_rels:
                            rel_data = readable_rels[(idTR, idArg)]
                            if typeR in rel_data[1]:
                                rels[(idTR, idArg)] = [rel_data[0], typeR]
                                continue
                            else:
                                isEvent = False
                                break
                        else:
                            isEvent = False
                            break
                    if isEvent:
                        event['rel'] = rels
                        sent_evs[idE] = event

            sentence['readable_ev'] = sent_evs

            trigger_ev = collections.defaultdict(list)
            # idEvs = OrderedDict()
            for idE in sent_evs:
                event = sent_evs[idE]
                idTR = event['trid']
                trigger_ev[idTR].append(event)

            sentence['trigger_ev'] = trigger_ev
            # sentence['idEvs'] = idEvs

            added_events.extend([idE for idE in sent_evs])

    input1 = OrderedDict()
    for pmid in input0:
        for sid, sentence in enumerate(input0[pmid]):
            sent_id = pmid + ':' + 'sent' + str(sid)
            input1[sent_id] = sentence

    return input1
