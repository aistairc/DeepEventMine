import torch
import os
import collections
from collections import defaultdict

from utils.utils import write_lines


class SelectClass:
    """
        Correct predictions: From 2 direction relations choose
        - if both 'no_relation' -> no_relation
        - if one positive, one negative -> positive
        - if both positive -> more confident (highest probability)
    """

    def __init__(self, params):
        self.params = params

    # Without L R distinguishing & on CPU
    def __call__(self, *inputs):
        labmap = torch.tensor(
            [self.params['lab_map'][e] for e in range(0, self.params['voc_sizes']['rel_size'])])
        ignore = torch.tensor(self.params['lab2ign_id'])

        cpu_device = torch.device("cpu")
        y_lr, y_rl = inputs
        y_lr = y_lr.to(cpu_device)
        y_rl = y_rl.to(cpu_device)
        if self.params['fp16']:
            y_lr = y_lr.float()
            y_rl = y_rl.float()

        labels_lr = y_lr.argmax(dim=1).view(-1)
        labels_rl = y_rl.argmax(dim=1).view(-1)

        m = torch.arange(labels_lr.shape[0])

        lr_probs = y_lr[m, labels_lr]
        rl_probs = y_rl[m, labels_rl]
        inv_lr = labmap[labels_lr]
        inv_rl = labmap[labels_rl]

        negative_val = torch.tensor(-1)

        # if both are negative --> keep negative class as prediction (1:Other:2)
        a_x1 = torch.where((labels_lr == labels_rl) & (labels_lr == ignore), ignore, negative_val)

        # if both are positive with same label (e.g. 1:rel:2) --> choose from probability
        a4 = torch.where((labels_lr != ignore) & (labels_rl != ignore) & (labels_lr == labels_rl),
                         lr_probs, negative_val.float())
        a5 = torch.where((labels_lr != ignore) & (labels_rl != ignore) & (labels_lr == labels_rl),
                         rl_probs, negative_val.float())
        a_x4 = torch.where((a4 >= a5) & (a4 != -1) & (a5 != -1), labels_lr, negative_val)
        a_x5 = torch.where((a4 < a5) & (a4 != -1) & (a5 != -1), inv_rl, negative_val)

        # # if both are positive with inverse 1:rel:2 & 2:rel:1 (this is correct) --> keep them the 'rel' label
        a_x6 = torch.where((labels_lr != labels_rl) & (labels_lr != ignore) &
                           (labels_rl != ignore) & (inv_lr == labels_rl), labels_lr, negative_val)

        # if one positive & one negative --> choose the positive class
        a_x2 = torch.where((labels_lr != labels_rl) & (labels_lr == ignore) & (labels_rl != ignore),
                           inv_rl, negative_val)
        a_x3 = torch.where((labels_lr != labels_rl) & (labels_lr != ignore) & (labels_rl == ignore),
                           labels_lr, negative_val)

        # if both are positive with different labels --> choose from probability
        a7 = torch.where(
            (labels_lr != ignore) & (labels_rl != ignore) & (labels_lr != labels_rl) & (inv_lr != labels_rl),
            lr_probs, negative_val.float())
        a8 = torch.where(
            (labels_lr != ignore) & (labels_rl != ignore) & (labels_lr != labels_rl) & (inv_lr != labels_rl), rl_probs,
            negative_val.float())

        a_x7 = torch.where((a7 >= a8) & (a7 != -1) & (a8 != -1), labels_lr, negative_val)
        a_x8 = torch.where((a7 < a8) & (a7 != -1) & (a8 != -1), inv_rl, negative_val)

        fin = torch.stack([a_x1, a_x2, a_x3, a_x4, a_x5, a_x6, a_x7, a_x8])
        assert (torch.sum(torch.clamp(fin, min=-1.0, max=0.0), dim=0) == -7).all(), "check evaluation"
        fin_preds = torch.max(fin, dim=0)

        return fin_preds[0]


def calc_stats(preds, params):
    new_preds = SelectClass(params)(preds[0], preds[1])
    return new_preds


def write_entity_relations(result_dir, fidss, ent_anns, rel_anns, params):
    # def gen_annotation(fidss, ent_anns, rel_anns, params, result_dir):
    """Generate entity and relation prediction"""

    dir2wr = ''.join([result_dir, 'rel-last/rel-ann/'])
    if not os.path.exists(dir2wr):
        os.makedirs(dir2wr)
    else:
        os.system('rm ' + dir2wr + '*.ann')

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
                        if 'pipeline_entity_org_map' in params:
                            if e_id in params['pipeline_entity_org_map'][fid]:
                                e_words, e_offset = params['pipeline_entity_org_map'][fid][e_id]
                            else:
                                print(e_id)
                                e_words, e_offset = get_entity_attrs(pair, words, offsets, sub_to_words)
                        else:
                            e_words, e_offset = get_entity_attrs(pair, words, offsets, sub_to_words)

                        # save entity map
                        entity_map[(xb, x)] = (
                            ner_preds[x], e_id, e_type, e_words, e_offset)

                        # save entity dic info
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

                    try:
                        arg1s = entity_map[(i.item(), j.item())]
                        arg2s = entity_map[(i.item(), k.item())]

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
                    except KeyError as error:
                        print('error relation', fids[i], error)

    for fid, ners_rels in map.items():
        write_annotation_file(dir2wr, fid, entities=ners_rels['ents'],
                              relations=ners_rels['rels'])


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


def mapping_entity_id(entities_):
    eid = 1
    enid_mapping = collections.OrderedDict()
    en_preds_out_ = []

    # create mapping for entity id first
    for en_id, en_data in entities_.items():

        if en_id.startswith('TR'):
            continue

        elif en_id.startswith('T'):
            enid_mapping[en_id] = 'T' + str(eid)
            eid += 1
            en_preds_out_.append(en_data)

    # creat mapping for trigger id
    for en_id, en_data in entities_.items():

        if en_id.startswith('TR'):
            enid_mapping[en_id] = 'T' + str(eid)
            eid += 1
            en_preds_out_.append(en_data)

    return enid_mapping, en_preds_out_


def write_annotation_file(dir2wr, fid, entities=None, relations=None):
    re_lines = []
    en_lines = []
    tr_lines = []

    # entity id mapping
    enid_mapping, en_preds_out_ = mapping_entity_id(entities)

    if entities:
        for entity in en_preds_out_:
            entity_annotation = "{}\t{} {} {}\t{}".format(
                enid_mapping[entity["id"]],
                entity["type"],
                entity["start"],
                entity["end"],
                entity["ref"],
            )

            re_lines.append(entity_annotation)

            if entity["id"].startswith('TR'):
                tr_lines.append(entity_annotation)

            elif entity["id"].startswith('T'):
                en_lines.append(entity_annotation)

    if relations:
        for relation in relations.values():
            relation_annotation = "{}\t{} {}:{} {}:{}".format(
                relation["id"],
                relation["role"],
                relation["left_arg"]["label"],
                enid_mapping[relation["left_arg"]["id"]],
                relation["right_arg"]["label"],
                enid_mapping[relation["right_arg"]["id"]],
            )
            re_lines.append(relation_annotation)

    # write to file
    re_file = ''.join([dir2wr, fid, '-RE.ann'])
    en_file = ''.join([dir2wr, fid, '-EN.ann'])
    tr_file = ''.join([dir2wr, fid, '-TR.ann'])

    write_lines(re_lines, re_file)
    write_lines(en_lines, en_file)
    write_lines(tr_lines, tr_file)
