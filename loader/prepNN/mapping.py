"""Generate mappings"""

import itertools
from collections import OrderedDict
from collections import Counter
import numpy as np

from loader.prepData.entity import entity_tags
from loader.prepData.relation import get_rtypes
from loader.prepNN.structure import process_structure


def _generate_mapping(list_of_elems):
    """
        :param list_of_elems: list of elements (single or nested)
        :returns
            dictionary with a unique id for each element
    """
    # list of lists
    elem_count = OrderedDict()
    if all(isinstance(el, list) for el in list_of_elems):
        for item in itertools.chain.from_iterable(list_of_elems):
            if item not in elem_count:
                elem_count[item] = 1
            else:
                elem_count[item] += 1
    # single lists
    else:
        for item in list_of_elems:
            if item not in elem_count:
                elem_count[item] = 1
            else:
                elem_count[item] += 1
    elem_count = sorted(elem_count.items(), key=lambda x: x[1])  # sort from low to high freq
    mapping = OrderedDict([(elem, i) for i, (elem, val) in enumerate(elem_count)])
    rev_mapping = OrderedDict([(v, k) for k, v in mapping.items()])
    return mapping, rev_mapping, len(elem_count)


def _find_singletons(list_of_elems, args, min_w_freq):
    """
        :param list_of_elems: list of all words in a train dataset
        :returns
            number of words with frequency = 1
    """
    elem_count = Counter([x for x in list_of_elems])
    unique_args = list(set(itertools.chain.from_iterable([a.split(' ') for a in args])))
    singles = [elem for elem, val in elem_count.items() if ((val <= min_w_freq) and (elem not in unique_args))]
    return singles


def generate_map(data_struct, data_struct_dev, data_struct_test, params): # add test for mlee

    # 1. words mapping
    words = data_struct['sentences']['sent_words']
    words_train = data_struct['sentences']['words']
    words.append(['<UNK>'])
    word_map, rev_word_map, word_size = _generate_mapping(words)

    # 2. ..
    # labels of entity (in .a1)
    argumentsT = data_struct['entities']['arguments']

    # labels of trigger (in .a2)
    argumentsTR = data_struct['triggers']['arguments']
    arguments = argumentsT + argumentsTR
    singlesW = _find_singletons(words_train, arguments, params['min_w_freq'])

    typesTR = data_struct['terms']['typesTR']
    typesTR.extend(data_struct_dev['terms']['typesTR'])

    typesT = data_struct['terms']['typesT']
    typesT.extend(data_struct_dev['terms']['typesT'])

    # add for test: fig bug for mlee
    typesTR.extend(data_struct_test['terms']['typesTR'])
    typesT.extend(data_struct_test['terms']['typesT'])

    all_types = []
    for type in typesTR:
        if type not in all_types:
            all_types.append(type)

    for type in typesT:
        if type not in all_types:
            all_types.append(type)

    type_map = {type: id for id, type in enumerate(all_types)}
    rev_type_map = {id: type for type, id in type_map.items()}
    type_size = len(type_map)

    typeTR_map = {}
    for type, id in type_map.items():
        if type in typesTR:
            typeTR_map[type] = id
    rev_typeTR_map = {id: type for type, id in typeTR_map.items()}
    # typeTR_size = len(typeTR_map)

    rev_tag_map, tag_map, _, _ = entity_tags(rev_type_map)

    tag_size = len(tag_map)

    trTypeIds = [id for id in rev_typeTR_map]

    tagsTR = data_struct['terms']['tagsTR']
    tagsTR2 = data_struct_dev['terms']['tagsTR']
    tagsTR.extend([tag for tag in tagsTR2 if tag not in tagsTR])
    rev_tag_mapTR = {tag_map[tag]: tag for tag in tagsTR}

    tag_mapTR = {tag: id for id, tag in rev_tag_mapTR.items()}
    trTagsIds = [tag for tag in rev_tag_mapTR]

    tag2type = data_struct['terms']['tags2types']
    tag2type2 = data_struct_dev['terms']['tags2types']
    for tag in tag2type2:
        if tag not in tag2type:
            tag2type[tag] = tag2type2[tag]
    tag2type_map = OrderedDict()
    for tag in tag2type:
        if tag != 'O':
            type = tag2type[tag]
            tag2type_map[tag_map[tag]] = type_map[type]
    tag2type_map[0] = -1  # tag O

    tag2type = np.zeros(tag_size, np.int32)
    for tag, type in tag2type_map.items():
        tag2type[tag] = type

    # 3. pos map
    all_sents = data_struct['sentences']['sentences']
    all_sents.extend(data_struct_dev['sentences']['sentences'])

    length = [len([w for w in s.split()]) for s in all_sents]
    ranges = [list(map(str, list(range(-l + 1, l)))) for l in length]
    if params['include_nested']:
        ranges.append(['inner'])  # encode nestedness embeddings
        ranges.append(['outer'])
    pos_map, rev_pos_map, pos_size = _generate_mapping(ranges)

    # 4. rel map
    rels = get_rtypes(data_struct, data_struct_dev)
    rel_map, rev_rel_map, rel_size = _generate_mapping(rels)

    # Generate relation maps with L R distinguishing
    rtype_map = {'Other': -1}
    rel2rtype_map = {}
    for rel in rel_map:
        relid = rel_map[rel]
        rtype = rel.split(':')[1]
        if '1:' in rel and rtype != 'Other':  # ony lef to right
            rtype_map[rtype] = relid

    for rel in rel_map:
        relid = rel_map[rel]
        rtype = rel.split(':')[1]
        rtypeid = rtype_map[rtype]
        rel2rtype_map[relid] = rtypeid

    rel2rtype_map2 = np.zeros((len(rel2rtype_map)), dtype=np.int32)
    for rel, rtype in rel2rtype_map.items():
        rel2rtype_map2[rel] = rtype

    rev_rtype_map = {id: type for type, id in rtype_map.items()}
    # rev_rtype_map[rel_size] = 'None'  # for the none relation in events

    # generate mappings for event structures
    flat_structs_map, nested_structs_map, flat_types_id_map, nested_types_id_map, etype_pairs = process_structure(
        data_struct, data_struct_dev, params, type_map, typeTR_map, rtype_map, type_size, rel_size)

    # modality
    modality_map = {'non-modality': 1, 'Speculation': 2, 'Negation': 3}
    rev_modality_map = {id: type for type, id in modality_map.items()}
    ev_size = len(modality_map)

    # return
    params['voc_sizes'] = {'word_size': word_size,
                           'etype_size': type_size,
                           'tag_size': tag_size,
                           'pos_size': pos_size,
                           'rel_size': rel_size,
                           'ev_size': ev_size
                           }
    params['mappings'] = {'word_map': word_map, 'rev_word_map': rev_word_map,
                          'type_map': type_map, 'rev_type_map': rev_type_map,
                          'typeTR_map': typeTR_map, 'rev_typeTR_map': rev_typeTR_map,
                          'tag_map': tag_map, 'rev_tag_map': rev_tag_map,
                          'tag_mapTR': tag_mapTR, 'rev_tag_mapTR': rev_tag_mapTR,
                          'tag2type_map': tag2type,
                          'pos_map': pos_map, 'rev_pos_map': rev_pos_map,
                          'rel_map': rel_map, 'rev_rel_map': rev_rel_map,
                          'rtype_map': rtype_map, 'rev_rtype_map': rev_rtype_map,
                          'rel2rtype_map': rel2rtype_map2,
                          'flat_structs_map': flat_structs_map, 'flat_types_id_map': flat_types_id_map,
                          'nested_structs_map': nested_structs_map, 'nested_types_id_map': nested_types_id_map,
                          'modality_map': modality_map, 'rev_modality_map': rev_modality_map,
                          'etype_pairs': etype_pairs
                          }
    params['trTags_Ids'] = trTagsIds
    params['trTypes_Ids'] = trTypeIds
    params['words_train'] = words_train
    params['singletons'] = singlesW
    params['max_sent_len'] = np.maximum(data_struct['sentences']['max_sent_len'],
                                        data_struct_dev['sentences']['max_sent_len'])
    params['rtype_trig_ev'] = rel_size

    return params


def find_ignore_label(params):
    """
        :return:
            id corresponds to the "Other" relation
            dictionary with directionality, e.g. relation_mapping['1:Rel:2'] = 3
                                                 relation_mapping['2:Rel:1'] = 8
                                                 lab_map[3] = 8, lab_map[8] = 3
    """
    lab2ign_id = params['mappings']['rel_map'][params['lab2ign']]

    # Map key of relation 1:REL:2 with 2:REL:1, else , map this key with itself, also map ignored keys
    lab_map = OrderedDict()
    for m, n in params['mappings']['rel_map'].items():
        for m2, n2 in params['mappings']['rel_map'].items():
            if m == m2:
                continue
            elif m == params['lab2ign'] or m2 == params['lab2ign']:
                continue
            elif m.split(':')[1] == m2.split(':')[1]:
                lab_map[n] = n2

    for m, n in params['mappings']['rel_map'].items():
        if n not in lab_map:
            lab_map[n] = n

    lab_map[lab2ign_id] = lab2ign_id
    params['lab_map'] = lab_map
    params['lab2ign_id'] = lab2ign_id
    return params


def _elem2idx(list_of_elems, map_func):
    """
        :param list_of_elems: list of lists
        :param map_func: mapping dictionary
        :returns
            list with indexed elements
    """
    # fix bug for mlee
    # return [[map_func[x] if x in map_func else map_func["O"] for x in list_of] for list_of in list_of_elems]
    return [[map_func[x] for x in list_of] for list_of in list_of_elems]



