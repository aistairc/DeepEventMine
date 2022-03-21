"""Prepare entity information for training networks."""

import collections

from loader.prepData.entity import extract_entities, convert_to_sub_words, convert_to_sub_words_lstm
from loader.prepNN.mapping import _elem2idx


def entity2network(sentence_data, words, params, tokenizer):
    # types -> ids
    tags = sentence_data['tags']
    tags_terms = sentence_data['tags_terms']

    # nner: Using subwords:
    if params['predict'] and params['pipelines']:
        if params['pipe_flag'] > 0:
            tokenizer = None

    # if use lstm
    if params['use_lstm']:
        sw_sentence, sub_to_word, subwords, valid_starts = convert_to_sub_words_lstm(words,
                                                                                list(map(list, zip(*tags))),
                                                                                list(map(list, zip(*tags_terms))),
                                                                                tokenizer=tokenizer)

    # or bert
    else:
        sw_sentence, sub_to_word, subwords, valid_starts = convert_to_sub_words(words,
                                                                            list(map(list, zip(*tags))),
                                                                            list(map(list, zip(*tags_terms))),
                                                                            tokenizer=tokenizer)

    entities, terms, sw_sentence = extract_entities(sw_sentence,
                                                    params['mappings']['tag_map'],
                                                    params['mappings']['rev_tag_map'],
                                                    params['mappings']['nn_mapping'])

    tagsIDs = _elem2idx(tags, params['mappings']['tag_map'])
    tagsIDs = list(map(list, zip(*tagsIDs)))

    tagsT = []
    tagsTR = []
    for tag in tagsIDs:
        if tag[0] in params['trTags_Ids']:
            tagsTR.append(tag)
        else:
            tagsT.append(tag)

    readable_e = sentence_data['readable_ents']
    idxs = sentence_data['idx']
    rev_idxs = {id: ent for ent, id in idxs.items()}
    toks2 = []
    etypes2 = []
    # ents = OrderedDict()
    ents = collections.defaultdict(list)
    # dup_ent_tag = False
    for xx in range(0, len(idxs)):

        ent = rev_idxs[xx]
        if "toks" not in readable_e[ent]:
            continue
        toks = readable_e[ent]['toks']
        toks2.append(toks)

        etypes2.append(readable_e[ent]['type'])
        toksid = tuple(toks)
        if len(toks) == 1:
            toksid = toks[0]
        ents[toksid].append([ent, readable_e[ent]['offs2'], readable_e[ent]['text']])

    # fix bug for mlee
    # etypes2ids = [params['mappings']['type_map'][etype] if etype in params['mappings']['type_map'] else params['mappings']['type_map']['Metabolism'] for etype in etypes2]
    etypes2ids = [params['mappings']['type_map'][etype] for etype in etypes2]

    return readable_e, idxs, ents, toks2, etypes2ids, entities, sw_sentence, sub_to_word, subwords, valid_starts, tagsIDs, tagsTR, terms
