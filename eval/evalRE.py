import os
import collections
from collections import defaultdict

import numpy as np
import torch
from tabulate import tabulate

from utils.utils import write_lines


class MeasureStatistics:
    """
    Calculate: True Positives (TP), False Positives (FP), False Negatives (FN)
    GPU & CPU code
    """

    def __init__(self, params, beta):
        self.params = params
        self.beta = beta

    def __call__(self, *inputs):
        label_num = self.params['voc_sizes']['rel_size']
        ignore_label = self.params['lab2ign_id']
        y, t = inputs

        if label_num is None:
            label_num = torch.max(t) + 1
        else:
            label_num = torch.tensor(label_num)

        mask_t = (t == ignore_label).view(-1)  # where the ground truth needs to be ignored
        true = torch.where(mask_t, label_num, t.view(-1))  # t: ground truth labels (replace ignored with 13)
        mask_p = (y == ignore_label).view(-1)  # where the predicted needs to be ignored
        pred = torch.where(mask_p, label_num, y.view(-1))  # y: output of neural network (replace ignored with 13)

        tp_mask = torch.where(pred == true, true, label_num)
        fp_mask = torch.where(pred != true, pred, label_num)
        fn_mask = torch.where(pred != true, true, label_num)

        try:
            tp = torch.bincount(tp_mask, minlength=label_num + 1)[:label_num]
            fp = torch.bincount(fp_mask, minlength=label_num + 1)[:label_num]
            fn = torch.bincount(fn_mask, minlength=label_num + 1)[:label_num]
        except:
            tp = torch.zeros(label_num)
            fp = torch.zeros(label_num)
            fn = torch.zeros(label_num)

        return tp, fp, fn


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
        y_lr, y_rl, truth_lr, truth_rl = inputs
        y_lr = y_lr.to(cpu_device)
        y_rl = y_rl.to(cpu_device)
        if self.params['fp16']:
            y_lr = y_lr.float()
            y_rl = y_rl.float()
        truth_lr = torch.tensor(truth_lr).long()
        truth_rl = torch.tensor(truth_rl).long()

        no_rel_matched_indices = 0
        no_rel_matched_types = 0

        try:
            labels_lr = y_lr.argmax(dim=1).view(-1)
            labels_rl = y_rl.argmax(dim=1).view(-1)
        except:
            return truth_lr, truth_lr, {'no_rel_matched_indices': no_rel_matched_indices,
                                        'no_rel_matched_types': no_rel_matched_types}
        m = torch.arange(labels_lr.shape[0])

        # count rel matched indices / types
        if not self.params['predict']:
            lr_ids = (truth_lr != -1).nonzero().transpose(0, 1)
            rl_ids = (truth_rl != -1).nonzero().transpose(0, 1)

            no_rel_matched_indices += (lr_ids.shape[1] + rl_ids.shape[1])

            lr_rel_matched_types = labels_lr[lr_ids] - truth_lr[lr_ids]
            rl_rel_matched_types = labels_rl[rl_ids] - truth_rl[rl_ids]

            no_rel_matched_types += (
                    (lr_rel_matched_types == 0).nonzero().shape[0] + (rl_rel_matched_types == 0).nonzero().shape[0])

        # split predictions into 2 arrays: relations + inv-relations
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
        # a_x5 = torch.where((a4 < a5) & (a4 != -1) & (a5 != -1), labels_rl, negative_val)

        # # if both are positive with inverse 1:rel:2 & 2:rel:1 (this is correct) --> keep them the 'rel' label
        a_x6 = torch.where((labels_lr != labels_rl) & (labels_lr != ignore) &
                           (labels_rl != ignore) & (inv_lr == labels_rl), labels_lr, negative_val)
        # If we don't care LR, we don't need a_x6
        # a_x6 = torch.where((labels_lr != labels_rl) & (labels_lr != ignore) &
        #                    (labels_rl != ignore) & (inv_even == odd_labels), even_labels, negative_val_long)

        # if one positive & one negative --> choose the positive class
        a_x2 = torch.where((labels_lr != labels_rl) & (labels_lr == ignore) & (labels_rl != ignore),
                           inv_rl, negative_val)
        # a_x2 = torch.where((labels_lr != labels_rl) & (labels_lr == ignore) & (labels_rl != ignore),
        #                    labels_rl, negative_val)
        a_x3 = torch.where((labels_lr != labels_rl) & (labels_lr != ignore) & (labels_rl == ignore),
                           labels_lr, negative_val)

        # if both are positive with different labels --> choose from probability
        a7 = torch.where(
            (labels_lr != ignore) & (labels_rl != ignore) & (labels_lr != labels_rl) & (inv_lr != labels_rl),
            lr_probs, negative_val.float())
        a8 = torch.where(
            (labels_lr != ignore) & (labels_rl != ignore) & (labels_lr != labels_rl) & (inv_lr != labels_rl), rl_probs,
            negative_val.float())
        # a7 = torch.where((labels_lr != ignore) & (labels_rl != ignore) & (labels_lr != labels_rl), lr_probs,
        #                  negative_val.float())
        # a8 = torch.where((labels_lr != ignore) & (labels_rl != ignore) & (labels_lr != labels_rl), rl_probs,
        #                  negative_val.float())
        a_x7 = torch.where((a7 >= a8) & (a7 != -1) & (a8 != -1), labels_lr, negative_val)
        a_x8 = torch.where((a7 < a8) & (a7 != -1) & (a8 != -1), inv_rl, negative_val)
        # a_x8 = torch.where((a7 < a8) & (a7 != -1) & (a8 != -1), labels_rl, negative_val)

        fin = torch.stack([a_x1, a_x2, a_x3, a_x4, a_x5, a_x6, a_x7, a_x8])
        # fin = torch.stack([a_x1, a_x2, a_x3, a_x4, a_x5, a_x7, a_x8])
        assert (torch.sum(torch.clamp(fin, min=-1.0, max=0.0), dim=0) == -7).all(), "check evaluation"
        # assert (torch.sum(torch.clamp(fin, min=-1.0, max=0.0), dim=0) == -6).all(), "check evaluation"
        fin_preds = torch.max(fin, dim=0)
        fin_truth = truth_lr

        return fin_preds[0], fin_truth, {'no_rel_matched_indices': no_rel_matched_indices,
                                         'no_rel_matched_types': no_rel_matched_types}


def calc_stats(preds, ts, params):
    new_preds, new_ts, no_matched_rels = SelectClass(params)(preds[0], preds[1], ts[0], ts[1])
    tp_, fp_, fn_ = MeasureStatistics(params, 1.0)(new_preds, new_ts)
    return new_preds, new_ts, no_matched_rels, tp_, fp_, fn_


def fbeta_score(precision, recall, beta=1.0):
    beta_square = beta * beta
    if (precision != 0.0) and (recall != 0.0):
        res = ((1 + beta_square) * precision * recall / (beta_square * precision + recall)).astype(precision.dtype)
    else:
        res = 0.0
    return res


def estimate_perf(all_tp, all_fp, all_fn, params):
    """
        Estimate performance: micro and macro average precision, recall, F1 score.
        CPU - based
    """
    lab_map = params['lab_map']
    class_size = params['voc_sizes']['rel_size']
    lab2ign = params['lab2ign_id']

    all_tp = np.sum(all_tp, axis=0)
    all_fp = np.sum(all_fp, axis=0)
    all_fn = np.sum(all_fn, axis=0)
    atp = np.sum(all_tp)
    afp = np.sum(all_fp)
    afn = np.sum(all_fn)
    micro_p = (1.0 * atp) / (atp + afp) if (atp + afp != 0) else 0.0
    micro_r = (1.0 * atp) / (atp + afn) if (atp + afn != 0) else 0.0
    micro_f = fbeta_score(micro_p, micro_r)

    # macro (merge directions l2r+r2l)
    ctp = []
    cfp = []
    cfn = []
    seen = []

    # Without L R distinguishing
    # for i in range(0, class_size):
    #     if i == lab2ign:  # don't include other class
    #         continue
    #     elif (i in seen):
    #         continue
    #     else:
    #         ctp.append(all_tp[i])
    #         cfp.append(all_fp[i])
    #         cfn.append(all_fn[i])
    #         seen.append(i)

    # With L R distinguishing
    for i in range(0, class_size):
        if i == lab2ign:  # don't include other class
            continue
        elif (i in seen) or (lab_map[i] in seen):
            continue
        else:
            ctp.append(all_tp[i] + all_tp[lab_map[i]])
            cfp.append(all_fp[i] + all_fp[lab_map[i]])
            cfn.append(all_fn[i] + all_fn[lab_map[i]])
            seen.append(i)
            seen.append(lab_map[i])

    pp = []
    rr = []
    ff = []
    for j in range(0, len(ctp)):
        pp.append((1.0 * ctp[j]) / (ctp[j] + cfp[j]) if (ctp[j] + cfp[j]) != 0 else 0.0)
        rr.append((1.0 * ctp[j]) / (ctp[j] + cfn[j]) if (ctp[j] + cfn[j]) != 0 else 0.0)
        ff.append(fbeta_score(pp[j], rr[j]))
    assert len(pp) == len(rr) == len(ff)

    # show performance on each class
    if params['show_macro']:
        gg = [ii for ii in range(0, class_size) if ii % 2 == 0][:-1]
        lab_val = []
        for i in range(0, len(pp)):
            lab_val.append([params['mappings']['rev_rel_map'][gg[i]].split(':')[1], pp[i], rr[i], ff[i]])
        print(tabulate(lab_val, headers=['Class', 'P', 'R', 'F1'], tablefmt='orgtbl'))

    macro_p = np.mean(pp)
    macro_r = np.mean(rr)
    macro_f = np.mean(ff)
    return {'micro_p': micro_p, 'micro_r': micro_r, 'micro_f': micro_f,
            'macro_p': macro_p, 'macro_r': macro_r, 'macro_f': macro_f}


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


def estimate_rel(ref_dir, result_dir, fids, ent_anns, rel_anns, params):
    """Evaluate entity and relation performance using n2c2 script"""

    # generate brat prediction
    gen_annotation(fids, ent_anns, rel_anns, params, result_dir)

    # calculate scores
    pred_dir = ''.join([result_dir, 'rel-last/rel-ann/'])
    pred_scores_file = ''.join([result_dir, 'rel-last/rel-scores-', params['ner_eval_corpus'], '.txt'])

    # run evaluation, output in the score file
    eval_performance(ref_dir, pred_dir, result_dir, pred_scores_file, params)

    # extract scores
    scores = extract_fscore(pred_scores_file)

    return scores


def gen_annotation(fidss, ent_anns, rel_anns, params, result_dir):
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
            # e_count = len(entities) + 1

            for x, pair in enumerate(span_indices):
                if pair[0].item() == -1:
                    break
                if ner_preds[x] > 0:
                    # e_id = 'T' + str(e_count)
                    # e_count += 1
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
                        # entity_map[(xb, (pair[0].item(), pair[1].item()))] = (
                        #     ner_preds[x], e_id, e_type, e_words, e_offset)
                        entity_map[(xb, x)] = (
                            ner_preds[x], e_id, e_type, e_words, e_offset)
                        entities[e_id] = {"id": e_id, "type": e_type, "start": e_offset[0], "end": e_offset[1],
                                          "ref": e_words}
                    except KeyError as error:
                        print('pred not map term', error)
        if len(rel_ann) > 0:
            # Mapping relations
            pairs_idx = rel_ann['pairs_idx']
            rel_preds = rel_ann['rel_preds']
            # positive_indices = rel_ann['positive_indices']

            # if positive_indices:
            # pairs_idx_i = pairs_idx[0][positive_indices]
            # pairs_idx_j = pairs_idx[1][positive_indices]
            # pairs_idx_k = pairs_idx[2][positive_indices]
            # else:
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
                # role = params['mappings']['rev_rtype_map'][rel]
                if role != 'Other':
                    # arg1s = entity_map[
                    #     (i.item(), (ent_ann['span_indices'][i][j][0].item(), ent_ann['span_indices'][i][j][1].item()))]
                    # arg2s = entity_map[
                    #     (i.item(), (ent_ann['span_indices'][i][k][0].item(), ent_ann['span_indices'][i][k][1].item()))]
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

                    # r_id = 'R' + str(r_count)
                    # r_count += 1
                    # relations[r_id] = {"id": r_id, "role": role,
                    #                    "left_arg": {"label": "Arg1", "id": arg2},
                    #                    "right_arg": {"label": "Arg2", "id": arg1}}

    for fid, ners_rels in map.items():
        write_annotation_file(ann_file=dir2wr + fid + '.ann', entities=ners_rels['ents'], relations=ners_rels['rels'])


def write_annotation_file(
        ann_file, entities=None, triggers=None, relations=None, events=None
):
    lines = []

    def annotate_text_bound(entities):
        for entity in entities.values():
            entity_annotation = "{}\t{} {} {}\t{}".format(
                entity["id"],
                entity["type"],
                entity["start"],
                entity["end"],
                entity["ref"],
            )
            lines.append(entity_annotation)

    if entities:
        annotate_text_bound(entities)

    if triggers:
        annotate_text_bound(triggers)

    if relations:
        for relation in relations.values():
            relation_annotation = "{}\t{} {}:{} {}:{}".format(
                relation["id"],
                relation["role"],
                relation["left_arg"]["label"],
                relation["left_arg"]["id"],
                relation["right_arg"]["label"],
                relation["right_arg"]["id"],
            )
            lines.append(relation_annotation)

    if events:
        for event in events.values():
            event_annotation = "{}\t{}:{}".format(
                event["id"], event["trigger_type"], event["trigger_id"]
            )
            for arg in event["args"]:
                event_annotation += " {}:{}".format(arg["role"], arg["id"])
            lines.append(event_annotation)

    write_lines(lines, ann_file)


def eval_performance(ref_dir, pred_dir, result_dir, pred_scores_file, params):
    # run evaluation script

    command = ''.join(
        ["python ", params['rel_eval_script_path'], " --ner-eval-corpus ", params['ner_eval_corpus'], " ", ref_dir, " ",
         pred_dir, " > ", pred_scores_file])
    os.system(command)

    # if predict: run for all config
    if params['predict'] == True:
        # entiy scores only
        ner_eval_corpus = ''.join([params['task_name'], '_en'])
        pred_scores_file = ''.join([result_dir, 'rel-last/rel-scores-', ner_eval_corpus, '.txt'])
        command = ''.join(
            ["python ", params['rel_eval_script_path'], " --ner-eval-corpus ", ner_eval_corpus, " ", ref_dir,
             " ",
             pred_dir, " > ", pred_scores_file])
        os.system(command)

        # trigger scores only
        ner_eval_corpus = ''.join([params['task_name'], '_tr'])
        pred_scores_file = ''.join([result_dir, 'rel-last/rel-scores-', ner_eval_corpus, '.txt'])
        command = ''.join(
            ["python ", params['rel_eval_script_path'], " --ner-eval-corpus ", ner_eval_corpus, " ", ref_dir,
             " ",
             pred_dir, " > ", pred_scores_file])
        os.system(command)


def extract_fscore(path):
    file = open(path, 'r')
    lines = file.readlines()
    report = defaultdict()
    report['NER'] = defaultdict()
    report['REL'] = defaultdict()

    ent_or_rel = ''
    for line in lines:
        if '*' in line and 'TRACK' in line:
            ent_or_rel = 'NER'
        elif '*' in line and 'RELATIONS' in line:
            ent_or_rel = 'REL'
        elif len(line.split()) > 0 and line.split()[0] == 'Overall':
            tokens = line.split()
            if len(tokens) > 8:
                strt_f, strt_r, strt_p, soft_f, soft_r, soft_p \
                    = tokens[-7], tokens[-8], tokens[-9], tokens[-4], tokens[-5], tokens[-6]
            else:
                strt_f, strt_r, strt_p, soft_f, soft_r, soft_p \
                    = tokens[-4], tokens[-5], tokens[-6], tokens[-1], tokens[-2], tokens[-3]
            if line.split()[1] == '(micro)':
                mi_or_mc = 'micro'
            elif line.split()[1] == '(macro)':
                mi_or_mc = 'macro'
            else:
                mi_or_mc = ''
            if mi_or_mc != '':
                report[ent_or_rel][mi_or_mc] = {'st_f': float(strt_f.strip()) * 100,
                                                'st_r': float(strt_r.strip()) * 100,
                                                'st_p': float(strt_p.strip()) * 100,
                                                'so_f': float(soft_f.strip()) * 100,
                                                'so_r': float(soft_r.strip()) * 100,
                                                'so_p': float(soft_p.strip()) * 100}

    return report

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
        write_annotation_file_bio(dir2wr, fid, entities=ners_rels['ents'],
                              relations=ners_rels['rels'])


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


def write_annotation_file_bio(dir2wr, fid, entities=None, relations=None):
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