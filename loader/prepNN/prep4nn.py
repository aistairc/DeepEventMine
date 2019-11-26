"""Prepare data for training networks."""

from collections import OrderedDict

from bert.tokenization import BertTokenizer
from sklearn.preprocessing import MultiLabelBinarizer

from loader.prepNN.sent2net import prep_sentences
from loader.prepNN.ent2net import entity2network, _elem2idx
from loader.prepNN.span4nn import get_nn_data


def data2network(data_struct, data_type, params):
    # input
    sent_words = data_struct['sentences']

    # words
    org_sent_words = sent_words['sent_words']
    sent_words = prep_sentences(sent_words, data_type, params)
    wordsIDs = _elem2idx(sent_words, params['mappings']['word_map'])

    all_sentences = []

    # nner: Using subwords:
    tokenizer = BertTokenizer.from_pretrained(
        params['bert_model'], do_lower_case=False
    )

    for xx, sid in enumerate(data_struct['input']):
        # input
        sentence_data = data_struct['input'][sid]

        # document id
        fid = sid.split(':')[0]

        # words to ids
        word_ids = wordsIDs[xx]
        words = org_sent_words[xx]

        # entity
        readable_e, idxs, ents, toks2, etypes2ids, entities, sw_sentence, sub_to_word, subwords, valid_starts, tagsIDs, terms = entity2network(
            sentence_data, words, params, tokenizer)

        # return
        sentence_vector = OrderedDict()
        sentence_vector['fid'] = fid
        sentence_vector['ents'] = ents
        sentence_vector['word_ids'] = word_ids
        sentence_vector['words'] = words
        sentence_vector['offsets'] = sentence_data['offsets']
        sentence_vector['e_ids'] = idxs
        sentence_vector['tags'] = tagsIDs
        sentence_vector['etypes2'] = etypes2ids
        sentence_vector['toks2'] = toks2
        sentence_vector['raw_words'] = sentence_data['words']

        sentence_vector['entities'] = entities
        sentence_vector['sw_sentence'] = sw_sentence
        sentence_vector['terms'] = terms
        sentence_vector['sub_to_word'] = sub_to_word
        sentence_vector['subwords'] = subwords
        sentence_vector['valid_starts'] = valid_starts

        all_sentences.append(sentence_vector)

    return all_sentences


def torch_data_2_network(cdata2network, params, do_get_nn_data):
    """ Convert object-type data to torch.tensor type data, aim to use with Pytorch
    """
    etypes = [data['etypes2'] for data in cdata2network]

    # nner
    entitiess = [data['entities'] for data in cdata2network]
    sw_sentences = [data['sw_sentence'] for data in cdata2network]
    termss = [data['terms'] for data in cdata2network]
    valid_startss = [data['valid_starts'] for data in cdata2network]

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

        mlb = MultiLabelBinarizer()
        mlb.fit([sorted(id_tag_mapping)[1:]])  # [1:] skip label O

        params["mappings"]["nn_mapping"]["mlb"] = mlb
        params["mappings"]["nn_mapping"]["num_labels"] = len(mlb.classes_)

        params["max_span_width"] = max(params["max_entity_width"], params["max_trigger_width"])

        params["mappings"]["nn_mapping"]["num_triggers"] = len(params["mappings"]["nn_mapping"]["trigger_labels"])
        params["mappings"]["nn_mapping"]["num_entities"] = params["mappings"]["nn_mapping"]["num_labels"] - \
                                                           params["mappings"]["nn_mapping"]["num_triggers"]

    if do_get_nn_data:
        nn_data = get_nn_data(fids, entitiess, termss, valid_startss, sw_sentences,
                              tokenizer, params)

        return {'nn_data': nn_data, 'etypes': etypes, 'fids': fids, 'words': wordss, 'offsets': offsetss,
                'sub_to_words': sub_to_words, 'subwords': subwords, 'entities': entitiess}
