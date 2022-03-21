"""Process entity."""

from collections import defaultdict
from collections import OrderedDict
import numpy as np
import re


def process_etypes(entities0):
    entities1 = OrderedDict()
    etypes_ = []
    arguments = []
    for pmid in entities0:
        entities = entities0[pmid]
        etypes = entities['counted_types']
        for type in etypes:
            if type not in etypes_:
                etypes_.append(type)

        for idT in entities['data']:
            argument = entities['data'][idT]['text']
            arguments.append(argument)

    entities1['pmids'] = entities0
    entities1['types'] = etypes_
    entities1['arguments'] = arguments

    return entities1


def process_tags(entities1, triggers1):
    typesT = entities1['types']
    typesTR = triggers1['types']
    types2 = typesT + typesTR

    tags = []
    tagsTR = []

    tags2types = OrderedDict()
    tags2types['O'] = 'O'
    for type in types2:
        btag = 'B-' + type
        itag = 'I-' + type
        tags.append(btag)
        tags.append(itag)
        tags2types[btag] = type
        tags2types[itag] = type

        if type in typesTR:
            tagsTR.append(btag)
            tagsTR.append(itag)

    tags0 = OrderedDict()
    tags0['types'] = types2
    tags0['typesTR'] = typesTR
    tags0['typesT'] = typesT
    tags0['tags'] = tags
    tags0['tags2types'] = tags2types
    tags0['tagsTR'] = tagsTR

    return tags0


def assign_label(offsets, terms):
    """
    Assign BIO label to each word of the sentence.
    """
    terms_sentence = []

    if len(terms) == 0:
        lst = [['O'] * len(offsets)]
        return lst, lst, terms_sentence

    max_level = max([item[-1] for item in terms])
    lst = []
    # nner
    lst_term = []
    for _ in range(max_level):
        label = [['O'] * len(offsets)]
        label_term = [['O'] * len(offsets)]
        lst.extend(label)
        # nner
        lst_term.extend(label_term)

    for level in range(max_level):
        terms_level = [item for item in terms if item[-1] == level + 1]
        for i, offset in enumerate(offsets):
            for term in terms_level:
                t_start, t_end = int(term[2]), int(term[3])
                if offset[0] == t_start and offset[1] <= t_end:
                    lst[level][i] = 'B-' + term[1]
                    # nner
                    lst_term[level][i] = 'B-' + term[0]
                    terms_sentence.append(term)

                elif offset[0] > t_start and offset[1] <= t_end:
                    lst[level][i] = 'I-' + term[1]
                    # nner
                    lst_term[level][i] = 'I-' + term[0]

    # nner
    # return lst, terms_sentence
    return lst, lst_term, terms_sentence


def argsort(arr):
    return sorted(range(len(arr)), key=arr.__getitem__)


def count_nest_level(arr, _):
    """
    Calculate nest level of each term and
    get the max nest level.
    term: id, type, start, end, text, nest_level
    """
    # Nest level of flat entities and non-entities is 1
    max_level = 1
    if len(arr) == 0:
        return max_level, arr

    sorted_ids = argsort([[int(e[2]), int(e[3])] for e in arr])

    first_item = arr[sorted_ids[0]]
    first_item.append(max_level)

    levels = defaultdict(list, {max_level: [first_item]})
    for idx in map(lambda p: sorted_ids[p], range(1, len(arr))):
        level = 1
        while level <= max_level:
            if int(arr[idx][2]) >= int(levels[level][-1][3]):
                break
            level += 1

        arr[idx].append(level)
        levels[level].append(arr[idx])
        max_level = max(max_level, level)

    return max_level, arr


def spliter(line, _len=len):
    """
        Credits to https://stackoverflow.com/users/1235039/aquavitae
        Return a list of words and their indexes in a string.
    """
    words = line.split(' ')
    index = line.index
    offsets = []
    append = offsets.append
    running_offset = 0
    for word in words:
        word_offset = index(word, running_offset)
        word_len = _len(word)
        running_offset = word_offset + word_len
        append((word, word_offset, running_offset))
    return offsets


def process_entities(entities1, triggers1, sentences1, params, dirpath):
    entities0 = entities1['pmids']
    triggers0 = triggers1['pmids']

    input0 = OrderedDict()

    sentences0 = sentences1['doc_data']
    levels = []

    for pmid in entities0:
        entities = entities0[pmid]
        triggers = triggers0[pmid]
        sentences = sentences0[pmid]

        terms = entities['terms']
        terms.extend(triggers['terms'])

        nest_level, terms = count_nest_level(terms, params)
        # nest_level, terms = utils.count_nest_level(terms)
        # terms, file_discard_count = utils.dicard_invalid_nes(terms, sentences)
        levels.append(nest_level)

        abst_text = '\n'.join([sent['sentence'] for sent in sentences])
        spans = []
        init_char = 0
        next_char = 0
        for char in abst_text:
            if char != '\n':
                next_char += 1
            else:
                spans.append((init_char, next_char))
                next_char += 1
                init_char = next_char
        spans.append((init_char, next_char))

        # doc_data = []
        # tags_ = []
        # terms_ = []
        for xx, sentence in enumerate(sentences):
            # offsets, words = calculate_offset(sentences, xx)
            offsets = sentence['offsets']
            # words = sentence['words']
            # chars = sentence['chars']
            # sent = sentence['sentence']

            # nner
            # tags, terms_sentence = utils.assign_label(offsets, terms)
            tags, tags_terms, terms_sentence = assign_label(offsets, terms)
            # tags_.append(tags)
            # terms_.append(terms_sentence)

            # check sentence has no entity
            # if len(terms_sentence) == 0:
            #     print('NO ENTITY: ', pmid, xx, sentence['sentence'])

            # tags_.extend([tag for level in tags for tag in level]) # for nested

            sentence['tags'] = tags
            sentence['terms'] = terms_sentence
            # nner
            sentence['tags_terms'] = tags_terms

            eids = []
            for t1 in terms_sentence:
                eid = t1[0]
                eids.append(eid)
            sentence['eids'] = eids
            readable_ents = OrderedDict()
            for eid in eids:
                if eid in entities['data']:
                    readable_ents[eid] = entities['data'][eid]
                else:
                    readable_ents[eid] = triggers['data'][eid]
            # sentence['readable_ents'] = readable_ents

            # offsets2
            span = spans[xx]
            # offs2 = []
            # etypes2 = []
            for x, id_ in enumerate(eids):  # for every entity if it belongs to sentence span
                ent = readable_ents[id_]
                b = int(ent['pos1'])
                e = int(ent['pos2'])
                # b, e = offs[x]
                if (span[0] <= b <= span[1]) and (span[0] <= e <= span[1]):
                    b2 = b - span[0]
                    e2 = e - span[0]
                    # offs2.append([b2, e2])  #

                    # etypes2.append(ent['type'])
                    ent['offs2'] = [b2, e2]
                else:
                    print("SKIP ENTITY: " + str(b) + " --- " + str(e))

            sentence['readable_ents'] = readable_ents

            tokens = spliter(
                sentence['sentence'])  # we have the tokens of the sentence and their corresponding offsets
            tokensN = [tok for tok, b, e in tokens]

            for eid in eids:
                if "offs2" not in readable_ents[eid]:
                    print(readable_ents[eid])
                    continue
                offs = readable_ents[eid]['offs2']
                start = offs[0]
                end = offs[1]
                toks = []
                for tok_id, (tok, start0, end0) in enumerate(tokens):  # of the word token
                    if (start0, end0) == (start, end):
                        toks.append(tok_id)
                    elif start0 == start and end0 < end:
                        toks.append(tok_id)
                    elif start0 > start and end0 < end:
                        toks.append(tok_id)
                    elif start0 > start and end0 == end:
                        toks.append(tok_id)

                readable_ents[eid]['toks'] = toks

    max_nest_level = max(levels)
    max_nest_level += 1

    for pmid in sentences1['doc_data']:
        in_sentences = sentences1['doc_data'][pmid]
        out_sentences = []
        label_count = len(in_sentences[0]['tags'])
        pad_level = max_nest_level - label_count

        for xx, sentence in enumerate(in_sentences):
            tags = sentence['tags']
            pad_label = [['O'] * len(tags[0])]
            tags.extend(pad_label * pad_level)

            # nner
            tags_terms = sentence['tags_terms']
            pad_label = [['O'] * len(tags_terms[0])]
            tags_terms.extend(pad_label * pad_level)

            out_sentences.append(sentence)

        input0[pmid] = out_sentences

    return input0


def entity_tags(dico):
    """
    Create a dictionary and a mapping of tags
    """
    id_to_tag = {0: 'O'}
    id_to_type = {0: 'O'}
    # id_to_tag = {}
    # id_to_type = {}
    for i, (k, v) in enumerate(dico.items()):
        # if v != 'O':
        id_to_tag[2 * i + 1] = 'I-' + v
        id_to_tag[2 * i + 2] = 'B-' + v
        id_to_type[2 * i + 2] = v

    tag_to_id = {v: k for k, v in id_to_tag.items()}
    type_to_id = {v: k for k, v in id_to_type.items()}

    return id_to_tag, tag_to_id, id_to_type, type_to_id


def extract_entities(sw_sentence, tag2id_mapping, id2tag_mapping, nn_mapping):
    # For several edge cases
    max_depth = max(len(tags) for _, tags, _ in sw_sentence)

    # for sentence in sentences: (not using loop)
    entities = defaultdict(list)
    terms = defaultdict(list)

    tokens = [token for token, *_ in sw_sentence]

    num_tokens = len(tokens)

    begin_indices = np.arange(num_tokens)
    end_indices = begin_indices + 1

    token_indices = np.column_stack((begin_indices, end_indices))

    try:
        tags = np.asarray(
        [
            # bug: original
                        # [tag2id_mapping[tag] if tag in tag2id_mapping else tag2id_mapping["O"] for tag in tags + ["O"] * max_depth][
            [tag2id_mapping[tag] for tag in tags + ["O"] * max_depth][
            :max_depth
            ]
            for _, tags, tags_terms in sw_sentence
        ]
        ).T
    except KeyError as err:
        tags = np.asarray(
        [
            [tag2id_mapping[tag] if tag in tag2id_mapping else tag2id_mapping["O"] for tag in tags + ["O"] * max_depth][
            :max_depth
            ]
            for _, tags, tags_terms in sw_sentence
        ]
        ).T
        print(err)


    tags_terms = np.asarray(
        [
            [tag_term for tag_term in tags_terms + ["O"] * max_depth][
            :max_depth
            ]
            for _, _, tags_terms in sw_sentence
        ]
    ).T

    for idx, tag in enumerate(tags):
        tag_term = tags_terms[idx]
        bo_indices = np.where(tag % 2 == 0)[0]
        i_indices = np.where(tag % 2 == 1)[0]

        num_rows, num_cols = tag.shape[0], bo_indices.shape[0]

        # Build merging matrix
        merging_matrix = np.full((num_rows, num_cols), 0, dtype=np.int)
        merging_matrix[bo_indices, np.arange(num_cols)] = 1

        # Fill I tags
        sub_indices = [tag[:i] for i in i_indices]
        counts = np.asarray(
            [np.where(i % 2 == 0)[0].shape[0] for i in sub_indices],
            dtype=np.int,
        )

        merging_matrix[i_indices, counts - 1] = 1

        # Get all the start indices for each token
        token_start_indices = np.zeros(merging_matrix.shape, dtype=np.int)
        token_start_indices[
            np.argmax(merging_matrix, axis=0), np.arange(num_cols)
        ] = 1
        token_start_indices = np.matmul(
            token_indices[:, 0], token_start_indices
        )

        # Get all the end indices for each token
        flipped_merging_matrix = np.flipud(merging_matrix)
        token_end_indices = np.zeros(merging_matrix.shape, dtype=np.int)
        token_end_indices[
            num_rows - 1 - np.argmax(flipped_merging_matrix, axis=0),
            np.arange(num_cols),
        ] = 1
        token_end_indices = np.matmul(
            token_indices[:, 1], token_end_indices
        ) - 1

        for begin_pos, end_pos in zip(
                token_start_indices.flatten(), token_end_indices.flatten()
        ):
            tag_id = tag[begin_pos]
            term_name = tag_term[begin_pos]
            if tag_id > 0:  # Tag O is not an entity
                tag_name = re.sub("^[BI]-", "", id2tag_mapping[tag_id])
                term_name = re.sub("^[BI]-", "", term_name)

                entities[(begin_pos, end_pos)].append(nn_mapping['tag_id_mapping'][tag_name])
                terms[(begin_pos, end_pos)].append(term_name)

    return entities, terms, sw_sentence


def convert_to_sub_words(word_tokens, tags, tags_terms, tokenizer=None):
    subword_pos = 0
    subword_offset_mapping = {}
    subwords = []
    sw_sentence = []

    valid_starts = {0}

    for token_idx, token in enumerate(word_tokens):
        if tokenizer:
            subtokens = tokenizer.tokenize(token)
            if subtokens:
                sw_sentence.append(subtokens[:1] + [tags[token_idx], tags_terms[token_idx]])
                subword_offset_mapping[subword_pos] = token_idx
                subword_pos += 1
                subwords.append(subtokens[:1][0])

                labels = [re.sub("^B-", "I-", label) for label in tags[token_idx]]
                ids = [re.sub("^B-", "I-", _id) for _id in tags_terms[token_idx]]

                for subtoken in subtokens[1:]:
                    sw_sentence.append([subtoken] + [labels, ids])
                    subword_offset_mapping[subword_pos] = token_idx
                    subword_pos += 1
                    subwords.append(subtoken)

            valid_starts.add(len(subwords))
        else:
            sw_sentence.append([token] + [tags[token_idx], tags_terms[token_idx]])
            subword_offset_mapping[token_idx] = token_idx
    return sw_sentence, subword_offset_mapping, subwords, valid_starts

def convert_to_sub_words_lstm(word_tokens, tags, tags_terms, tokenizer=None):
    subword_pos = 0
    subword_offset_mapping = {}
    subwords = []
    sw_sentence = []

    valid_starts = {0}

    for token_idx, token in enumerate(word_tokens):
        if tokenizer:
            # subtokens = tokenizer.tokenize(token)
            subtokens = [token]
            if subtokens:
                sw_sentence.append(subtokens[:1] + [tags[token_idx], tags_terms[token_idx]])
                subword_offset_mapping[subword_pos] = token_idx
                subword_pos += 1
                # subwords.append(subtokens[:1][0])
                subwords.append(subtokens[0])

                labels = [re.sub("^B-", "I-", label) for label in tags[token_idx]]
                ids = [re.sub("^B-", "I-", _id) for _id in tags_terms[token_idx]]

                for subtoken in subtokens[1:]:
                    sw_sentence.append([subtoken] + [labels, ids])
                    subword_offset_mapping[subword_pos] = token_idx
                    subword_pos += 1
                    subwords.append(subtoken)

            valid_starts.add(len(subwords))
        else:
            sw_sentence.append([token] + [tags[token_idx], tags_terms[token_idx]])
            subword_offset_mapping[token_idx] = token_idx
    return sw_sentence, subword_offset_mapping, subwords, valid_starts
