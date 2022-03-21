from collections import defaultdict

from utils.utils import write_annotation_file


def get_entity_attrs(e_span_indice, words, offsets, sub_to_words):
    e_words = []
    e_offset = [-1, -1]
    curr_word_idx = -1
    for idx in range(e_span_indice[0], e_span_indice[1] + 1):
        if sub_to_words[idx] != curr_word_idx:
            e_words.append(words[sub_to_words[idx]])
            curr_word_idx = sub_to_words[idx]
        if idx == e_span_indice[0]:
            e_offset[0] = offsets[sub_to_words[idx]][0]
        if idx == e_span_indice[1]:
            e_offset[1] = offsets[sub_to_words[idx]][1]
    return ' '.join(e_words), (e_offset[0], e_offset[1])


def gen_sw_offsets(word_offsets, words, subwords, sub_to_words):
    sw_offsets = []
    last_sw_offsets = -1
    for sw_id, w_id in sub_to_words.items():
        subword = subwords[sw_id].replace('##', '')
        word = words[w_id]
        word_offset = word_offsets[w_id]
        sw_idx = word.index(subword,
                            0 if (last_sw_offsets == -1 or last_sw_offsets < word_offset[0]) else last_sw_offsets - 1 -
                                                                                                  word_offset[0])
        sw_offsets.append((word_offset[0] + sw_idx, word_offset[0] + sw_idx + len(subword)))
        last_sw_offsets = word_offset[0] + sw_idx + len(subword)
    return sw_offsets


def get_entity_sw_attrs(e_id, e_span_indice, words, offsets, sub_to_words, subwords, sw_offsets, org_mapping):
    e_words = []
    e_offset = [-1, -1]
    sw_text = []
    sw_offset = [-1, -1]

    curr_word_idx = -1
    for idx in range(e_span_indice[0], e_span_indice[1] + 1):
        if sub_to_words[idx] != curr_word_idx:
            e_words.append(words[sub_to_words[idx]])
            curr_word_idx = sub_to_words[idx]
        sw_text.append(subwords[idx])
        if idx == e_span_indice[0]:
            e_offset[0] = offsets[sub_to_words[idx]][0]
            sw_offset[0] = sw_offsets[idx][0]
        if idx == e_span_indice[1]:
            e_offset[1] = offsets[sub_to_words[idx]][1]
            sw_offset[1] = sw_offsets[idx][1]
    org_mapping[e_id] = (' '.join(e_words), (e_offset[0], e_offset[1]))
    return ' '.join(sw_text), (sw_offset[0], sw_offset[1])


def gen_ner_ann_files(fidss, ent_anns, params):
    dir2wr = params['pipeline_setting'] + params['pipe_ner']

    # Initial ent map
    map = defaultdict()
    org_map = defaultdict()
    params['pipeline_text_data'] = defaultdict()
    for fids in fidss:
        for fid in fids:
            map[fid] = {}
            org_map[fid] = {}
            params['pipeline_text_data'][fid] = []

    for xi, (fids, ent_ann) in enumerate(zip(fidss, ent_anns)):
        # Mapping entities
        entity_map = defaultdict()
        for xb, (fid) in enumerate(fids):
            span_indices = ent_ann['span_indices'][xb]
            ner_terms = ent_ann['ner_terms'][xb]
            ner_preds = ent_ann['ner_preds'][xb]
            words = ent_ann['words'][xb]
            offsets = ent_ann['offsets'][xb]
            sub_to_words = ent_ann['sub_to_words'][xb]
            subwords = ent_ann['subwords'][xb]
            sw_offsets = gen_sw_offsets(offsets, words, subwords, sub_to_words)
            params['pipeline_text_data'][fid].append(
                {'words': subwords, 'offsets': sw_offsets})
            entities = map[fid]
            org_mapping = org_map[fid]

            for x, pair in enumerate(span_indices):
                if pair[0].item() == -1:
                    break
                if ner_preds[x] > 0:
                    try:
                        e_id = ner_terms.id2term[x]
                        e_type = params['mappings']['rev_type_map'][
                            params['mappings']['nn_mapping']['tag2type_map'][ner_preds[x]]]
                        e_words, e_offset = get_entity_sw_attrs(e_id, pair, words, offsets, sub_to_words, subwords,
                                                                sw_offsets, org_mapping)
                        entity_map[(xb, (pair[0].item(), pair[1].item()))] = (
                            ner_preds[x], e_id, e_type, e_words, e_offset)
                        entities[e_id] = {"id": e_id, "type": e_type, "start": e_offset[0], "end": e_offset[1],
                                          "ref": e_words}
                    except KeyError as error:
                        print('pred not map term', error)

    params['pipeline_entity_org_map'] = org_map

    for fid, ners in map.items():
        write_annotation_file(ann_file=dir2wr + fid + '.ann', entities=ners)


def gen_rel_ann_files(fidss, ent_anns, rel_anns, params):
    dir2wr = params['pipeline_setting'] + params['pipe_rel']

    # Initial ent+rel map
    map = defaultdict()
    for fids in fidss:
        for fid in fids:
            map[fid] = {'ents': {}, 'rels': {}}

    for xi, (fids, ent_ann, rel_ann) in enumerate(zip(fidss, ent_anns, rel_anns)):
        # Mapping entities
        entity_map = defaultdict()
        for xb, (fid) in enumerate(fids):
            span_indices = ent_ann['span_indices'][xb]
            ner_terms = ent_ann['ner_terms'][xb]
            ner_preds = ent_ann['ner_preds'][xb]
            words = ent_ann['words'][xb]
            offsets = ent_ann['offsets'][xb]
            sub_to_words = ent_ann['sub_to_words'][xb]

            entities = map[fid]['ents']

            for x, pair in enumerate(span_indices):
                if pair[0].item() == -1:
                    break
                if ner_preds[x] > 0:
                    try:
                        e_id = ner_terms.id2term[x]
                        e_type = params['mappings']['rev_type_map'][
                            params['mappings']['nn_mapping']['tag2type_map'][ner_preds[x]]]
                        e_words, e_offset = get_entity_attrs(pair, words, offsets, sub_to_words)
                        entity_map[(xb, (pair[0].item(), pair[1].item()))] = (
                            ner_preds[x], e_id, e_type, e_words, e_offset)
                        entities[e_id] = {"id": e_id, "type": e_type, "start": e_offset[0], "end": e_offset[1],
                                          "ref": e_words}
                    except KeyError as error:
                        print('pred not map term', error)
        if len(rel_ann) > 0:
            # Mapping relations
            pairs_idx = rel_ann['pairs_idx']
            rel_preds = rel_ann['rel_preds']

            pairs_idx_i = pairs_idx[0]
            pairs_idx_j = pairs_idx[1]
            pairs_idx_k = pairs_idx[2]

            for x, i in enumerate(pairs_idx_i):
                relations = map[fids[i]]['rels']
                r_count = len(relations) + 1

                j = pairs_idx_j[x]
                k = pairs_idx_k[x]
                rel = rel_preds[x].item()
                role = params['mappings']['rev_rel_map'][rel].split(":")[1]
                if role != 'Other':
                    arg1s = entity_map[
                        (i.item(), (ent_ann['span_indices'][i][j][0].item(), ent_ann['span_indices'][i][j][1].item()))]
                    arg2s = entity_map[
                        (i.item(), (ent_ann['span_indices'][i][k][0].item(), ent_ann['span_indices'][i][k][1].item()))]

                    if int(params['mappings']['rev_rel_map'][rel].split(":")[0]) > int(
                            params['mappings']['rev_rel_map'][rel].split(":")[-1]):
                        arg1 = arg2s[1]
                        arg2 = arg1s[1]
                    else:
                        arg1 = arg1s[1]
                        arg2 = arg2s[1]
                    r_id = 'R' + str(r_count)
                    r_count += 1
                    relations[r_id] = {"id": r_id, "role": role,
                                       "left_arg": {"label": "Arg1", "id": arg1},
                                       "right_arg": {"label": "Arg2", "id": arg2}}

    for fid, ners_rels in map.items():
        write_annotation_file(ann_file=dir2wr + fid + '.ann', entities=ners_rels['ents'], relations=ners_rels['rels'])
