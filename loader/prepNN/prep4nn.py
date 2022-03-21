"""Prepare data for training networks."""

import collections
from collections import OrderedDict

from bert.tokenization import BertTokenizer
from sklearn.preprocessing import MultiLabelBinarizer

from loader.prepNN.sent2net import prep_sentences
from loader.prepNN.ent2net import entity2network
from loader.prepNN.ev2net import event2network
from loader.prepNN.mapping import _elem2idx
from loader.prepNN.span4nn import get_nn_data


def data2network(data_struct, data_type, params):
    # input
    sent_words = data_struct['sentences']

    # words
    org_sent_words = sent_words['sent_words']
    sent_words = prep_sentences(sent_words, data_type, params)
    wordsIDs = _elem2idx(sent_words, params['mappings']['word_map'])

    all_sentences = []

    # C2T add:
    max_ev_per_layer = params['max_ev_per_layer']

    # nner: Using subwords:
    tokenizer = BertTokenizer.from_pretrained(
        params['bert_model'], do_lower_case=False
    )

    events_map = collections.defaultdict()

    for xx, sid in enumerate(data_struct['input']):

        # input
        sentence_data = data_struct['input'][sid]

        # document id
        fid = sid.split(':')[0]

        # words to ids
        # words = sentence_data['words']
        word_ids = wordsIDs[xx]
        words = org_sent_words[xx]

        # entity
        readable_e, idxs, ents, toks2, etypes2ids, entities, sw_sentence, sub_to_word, subwords, valid_starts, tagsIDs, tagsTR, terms = entity2network(
            sentence_data, words, params, tokenizer)

        # events
        events, truth_ev, max_ev_per_layer = event2network(sentence_data, fid, idxs, events_map, max_ev_per_layer,
                                                           readable_e, params)

        # return
        sentence_vector = OrderedDict()
        sentence_vector['fid'] = fid
        sentence_vector['ents'] = ents
        sentence_vector['word_ids'] = word_ids
        sentence_vector['words'] = words
        sentence_vector['offsets'] = sentence_data['offsets']
        sentence_vector['e_ids'] = idxs
        sentence_vector['tags'] = tagsIDs
        sentence_vector['tagsTR'] = tagsTR
        sentence_vector['etypes2'] = etypes2ids
        sentence_vector['toks2'] = toks2
        sentence_vector['raw_words'] = sentence_data['words']
        sentence_vector['truth_ev'] = truth_ev

        # nner
        sentence_vector['entities'] = entities
        sentence_vector['sw_sentence'] = sw_sentence
        sentence_vector['terms'] = terms
        sentence_vector['relations'] = sentence_data['readable_r']
        sentence_vector['events'] = events
        sentence_vector['sub_to_word'] = sub_to_word
        sentence_vector['subwords'] = subwords
        sentence_vector['valid_starts'] = valid_starts

        # ignore this sentence or not
        ignore_sent = False

        # filter sentence with no entity, for training set only (contains 'train' in path)
        if params['filter_no_ent_sents'] and data_type == 'train':

            # check number of entities in this sentence
            ents_no = len(sentence_vector['e_ids'])
            if ents_no == 0:
                ignore_sent = True

        if not ignore_sent:
            all_sentences.append(sentence_vector)

    # C2T add
    params['max_ev_per_layer'] = max_ev_per_layer

    return all_sentences, events_map


def torch_data_2_network(cdata2network, events_map, params, do_get_nn_data):
    """ Convert object-type data to torch.tensor type data, aim to use with Pytorch
    """
    etypes = [data['etypes2'] for data in cdata2network]

    # nner
    entitiess = [data['entities'] for data in cdata2network]
    sw_sentences = [data['sw_sentence'] for data in cdata2network]
    termss = [data['terms'] for data in cdata2network]
    valid_startss = [data['valid_starts'] for data in cdata2network]
    relationss = [data['relations'] for data in cdata2network]
    eventss = [data['events'] for data in cdata2network]

    fids = [data['fid'] for data in cdata2network]
    wordss = [data['words'] for data in cdata2network]
    offsetss = [data['offsets'] for data in cdata2network]
    sub_to_words = [data['sub_to_word'] for data in cdata2network]
    subwords = [data['subwords'] for data in cdata2network]

    tokenizer = BertTokenizer.from_pretrained(
        params['bert_model'], do_lower_case=False
    )

    # User-defined data
    if not params["predict"]:
        id_tag_mapping = params["mappings"]["nn_mapping"]["id_tag_mapping"]
        trigger_ids = params["mappings"]["nn_mapping"]["trTypes_Ids"]

        mlb = MultiLabelBinarizer()
        mlb.fit([sorted(id_tag_mapping)[1:]])  # [1:] skip label O

        params["mappings"]["nn_mapping"]["mlb"] = mlb
        params["mappings"]["nn_mapping"]["num_labels"] = len(mlb.classes_)

        params["max_span_width"] = max(params["max_entity_width"], params["max_trigger_width"])

        params["mappings"]["nn_mapping"]["full_labels"] = sorted([v for k, v in id_tag_mapping.items() if k > 0])
        params["mappings"]["nn_mapping"]["trigger_labels"] = sorted(
            [v for k, v in id_tag_mapping.items() if k in trigger_ids])

        params["mappings"]["nn_mapping"]["num_triggers"] = len(params["mappings"]["nn_mapping"]["trigger_labels"])
        params["mappings"]["nn_mapping"]["num_entities"] = params["mappings"]["nn_mapping"]["num_labels"] - \
                                                           params["mappings"]["nn_mapping"]["num_triggers"]

    if do_get_nn_data:
        nn_data = get_nn_data(fids, entitiess, termss, valid_startss, relationss, eventss, sw_sentences,
                              tokenizer, events_map,
                              params)

        return {'nn_data': nn_data, 'etypes': etypes, 'fids': fids, 'words': wordss, 'offsets': offsetss,
                'sub_to_words': sub_to_words, 'subwords': subwords, 'entities': entitiess}
    else:
        return {'termss': termss, 'relationss': relationss, 'eventss': eventss, 'sw_sentences': sw_sentences,
                'tokenizer': tokenizer, 'events_map': events_map, 'params': params, 'etypes': etypes, 'fids': fids,
                'words': wordss, 'offsets': offsetss, 'sub_to_words': sub_to_words, 'subwords': subwords,
                'entities': entitiess}
