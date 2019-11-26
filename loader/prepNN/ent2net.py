"""Prepare entity information for training networks."""

import collections

from loader.prepData.entity import extract_entities, convert_to_sub_words


def _elem2idx(list_of_elems, map_func):
    """
        :param list_of_elems: list of lists
        :param map_func: mapping dictionary
        :returns
            list with indexed elements
    """
    return [[map_func[x] for x in list_of] for list_of in list_of_elems]


def entity2network(sentence_data, words, params, tokenizer):
    # types -> ids
    tags = sentence_data['tags']
    tags_terms = sentence_data['tags_terms']

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
    for tag in tagsIDs:
        tagsT.append(tag)

    readable_e = sentence_data['readable_ents']
    idxs = sentence_data['idx']
    rev_idxs = {id: ent for ent, id in idxs.items()}
    toks2 = []
    etypes2 = []
    ents = collections.defaultdict(list)
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

    etypes2ids = [params['mappings']['type_map'][etype] for etype in etypes2]

    return readable_e, idxs, ents, toks2, etypes2ids, entities, sw_sentence, sub_to_word, subwords, valid_starts, tagsIDs, terms
