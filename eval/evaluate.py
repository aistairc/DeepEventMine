import time

import torch
from tqdm import tqdm

from eval.evalEV import evaluate_ev
from eval.evalRE import estimate_perf, estimate_rel
from eval.evalNER import eval_nner
from scripts.pipeline_process import gen_ner_ann_files, gen_rel_ann_files
from utils import utils
from utils.utils import _humanized_time


def eval(model, eval_dir, result_dir, eval_dataloader, eval_data, params, epoch=0):
    mapping_id_tag = params['mappings']['nn_mapping']['id_tag_mapping']
    rel_tp_tr, rel_fp_tr, rel_fn_tr = [], [], []

    # store predicted entities
    ent_preds = []

    # store predicted events
    ev_preds = []

    fidss, wordss, offsetss, sub_to_wordss, span_indicess = [], [], [], [], []

    rel_anns = []
    ent_anns = []

    # Evaluation phase
    model.eval()

    # nner
    all_ner_preds, all_ner_golds, all_ner_terms = [], [], []
    total_rel_matched_indices = 0
    total_rel_matched_types = 0

    t_start = time.time()

    is_eval_rel = False
    is_eval_ev = False

    for step, batch in enumerate(
            tqdm(eval_dataloader, desc="Iteration", leave=False)
    ):
        eval_data_ids = batch
        tensors = utils.get_tensors(eval_data_ids, eval_data, params)

        nn_tokens, nn_ids, nn_token_mask, nn_attention_mask, nn_span_indices, nn_span_labels, nn_span_labels_match_rel, nn_entity_masks, nn_trigger_masks, nn_gtruth, nn_l2r, _, \
        nn_truth_ev, nn_ev_idxs, ev_lbls, etypes, _ = tensors

        fids = [
            eval_data["fids"][data_id] for data_id in eval_data_ids[0].tolist()
        ]
        offsets = [
            eval_data["offsets"][data_id]
            for data_id in eval_data_ids[0].tolist()
        ]
        words = [
            eval_data["words"][data_id] for data_id in eval_data_ids[0].tolist()
        ]
        sub_to_words = [
            eval_data["sub_to_words"][data_id]
            for data_id in eval_data_ids[0].tolist()
        ]
        subwords = [
            eval_data["subwords"][data_id]
            for data_id in eval_data_ids[0].tolist()
        ]
        gold_entities = [
            eval_data["entities"][data_id]
            for data_id in eval_data_ids[0].tolist()
        ]

        with torch.no_grad():
            if not params['predict']:
                ner_out, rel_out, ev_out, loss = model(tensors, epoch)
            else:
                ner_out, rel_out, ev_out, loss = model(tensors)

        ner_preds = ner_out['preds']

        if not params['predict']:  # Debug only
            # Case train REL only
            if params['skip_ner'] and params['rel_epoch'] >= (params['epoch'] - 1) and params['use_gold_ner']:
                ner_terms = ner_out['gold_terms']
                ner_preds = ner_out['golds']
            # Case train EV only
            elif params['skip_ner'] and params['skip_rel'] and params['use_gold_ner'] \
                    and params['use_gold_rel']:
                ner_terms = ner_out['gold_terms']
                ner_preds = ner_out['golds']
            else:
                ner_terms = ner_out['terms']
        else:
            if params['gold_eval'] or params['pipelines']:
                if params['pipelines'] and params['pipe_flag'] == 0:
                    ner_terms = ner_out['terms']
                else:
                    ner_terms = ner_out['gold_terms']
                    ner_preds = ner_out['golds']
            else:
                ner_terms = ner_out['terms']

        all_ner_terms.append(ner_terms)

        for sentence_idx, ner_pred in enumerate(ner_preds):
            all_ner_golds.append(
                [
                    (
                        sub_to_words[sentence_idx][span_start],
                        sub_to_words[sentence_idx][span_end],
                        mapping_id_tag[label_id],
                    )
                    for (
                            span_start,
                            span_end,
                        ), label_ids in gold_entities[sentence_idx].items()
                    for label_id in label_ids
                ]
            )

            pred_entities = []
            for span_id, ner_pred_id in enumerate(ner_pred):
                span_start, span_end = nn_span_indices[sentence_idx][span_id]
                span_start, span_end = span_start.item(), span_end.item()
                if (ner_pred_id > 0
                        and span_start in sub_to_words[sentence_idx]
                        and span_end in sub_to_words[sentence_idx]
                ):
                    pred_entities.append(
                        (
                            sub_to_words[sentence_idx][span_start],
                            sub_to_words[sentence_idx][span_end],
                            mapping_id_tag[ner_pred_id],
                        )
                    )
            all_ner_preds.append(pred_entities)

        fidss.append(fids)
        if params['predict']:
            if params['gold_eval'] or params['pipelines']:
                if params['pipelines'] and params['pipe_flag'] == 0:
                    ent_ann = {'span_indices': nn_span_indices, 'ner_preds': ner_out['preds'], 'words': words,
                               'offsets': offsets, 'sub_to_words': sub_to_words, 'subwords': subwords,
                               'ner_terms': ner_terms}
                else:
                    ent_ann = {'span_indices': nn_span_indices, 'ner_preds': ner_out['golds'], 'words': words,
                               'offsets': offsets, 'sub_to_words': sub_to_words, 'subwords': subwords,
                               'ner_terms': ner_terms}
            else:
                ent_ann = {'span_indices': nn_span_indices, 'ner_preds': ner_out['preds'], 'words': words,
                           'offsets': offsets, 'sub_to_words': sub_to_words, 'subwords': subwords,
                           'ner_terms': ner_terms}
        else:
            # Case only train REL
            if params['skip_ner'] and params['rel_epoch'] >= (params['epoch'] - 1) and params['use_gold_ner']:
                ent_ann = {'span_indices': nn_span_indices, 'ner_preds': ner_out['golds'], 'words': words,
                           'offsets': offsets, 'sub_to_words': sub_to_words, 'subwords': subwords,
                           'ner_terms': ner_terms}
            # Case only train EV
            elif params['skip_ner'] and params['skip_rel'] and params['use_gold_rel']:
                ent_ann = {'span_indices': nn_span_indices, 'ner_preds': ner_out['golds'], 'words': words,
                           'offsets': offsets, 'sub_to_words': sub_to_words, 'subwords': subwords,
                           'ner_terms': ner_terms}
            else:
                ent_ann = {'span_indices': nn_span_indices, 'ner_preds': ner_out['preds'], 'words': words,
                           'offsets': offsets, 'sub_to_words': sub_to_words, 'subwords': subwords,
                           'ner_terms': ner_terms}

        ent_anns.append(ent_ann)

        wordss.append(words)
        offsetss.append(offsets)
        sub_to_wordss.append(sub_to_words)

        if rel_out != None:
            rel_tp_tr.append(rel_out['true_pos'].tolist())
            rel_fp_tr.append(rel_out['false_pos'].tolist())
            rel_fn_tr.append(rel_out['false_neg'].tolist())
            total_rel_matched_indices += rel_out['no_matched_rel']['no_rel_matched_indices']
            total_rel_matched_types += rel_out['no_matched_rel']['no_rel_matched_types']

            if params['predict']:
                if params['gold_eval'] or params['pipelines']:
                    if params['pipelines'] and params['pipe_flag'] != 2:
                        pairs_idx = rel_out['pairs_idx']
                        rel_pred = rel_out['preds']
                    else:
                        pairs_idx = rel_out['l2r']
                        rel_pred = rel_out['truth']
                else:
                    pairs_idx = rel_out['pairs_idx']
                    rel_pred = rel_out['preds']
            else:
                # Case only train REL
                if params['skip_ner'] and params['rel_epoch'] >= (params['epoch'] - 1) \
                        and params['use_gold_ner']:
                    pairs_idx = rel_out['l2r']
                    rel_pred = rel_out['preds']
                # Case only train EV
                elif params['skip_ner'] and params['skip_rel'] and params['use_gold_rel']:
                    pairs_idx = rel_out['l2r']
                    rel_pred = rel_out['truth']
                else:
                    pairs_idx = rel_out['pairs_idx']
                    rel_pred = rel_out['preds']

            rel_ann = {'pairs_idx': pairs_idx, 'rel_preds': rel_pred}
            rel_anns.append(rel_ann)
            is_eval_rel = True
        else:
            rel_anns.append({})

        if ev_out != None:
            # add predicted entity
            ent_preds.append(ner_out["nner_preds"])

            # add predicted events
            ev_preds.append(ev_out['output'])

            span_indicess.append(
                [
                    indice.detach().cpu().numpy()
                    for indice in ner_out["span_indices"]
                ]
            )
            is_eval_ev = True
        else:
            ent_preds.append([])
            ev_preds.append([])

            span_indicess.append([])

        # Clear GPU unused RAM:
        if params['gpu'] >= 0:
            torch.cuda.empty_cache()

    if params['predict'] and params['pipelines']:
        if params['pipe_flag'] == 0:
            gen_ner_ann_files(fidss, ent_anns, params)
            return
        elif params['pipe_flag'] == 1:
            gen_rel_ann_files(fidss, ent_anns, rel_anns, params)
            return

    # Do estimations here
    labels = params["mappings"]["nn_mapping"]["trigger_labels"]
    if params["ner_predict_all"]:
        labels = params["mappings"]["nn_mapping"]["full_labels"]

    ner_res, ner_score = eval_nner(all_ner_preds, all_ner_golds, labels)
    scores = estimate_rel(ref_dir=eval_dir,
                          result_dir=result_dir,
                          fids=fidss,
                          ent_anns=ent_anns,
                          rel_anns=rel_anns,
                          params=params)
    if is_eval_rel:
        tr_scores = estimate_perf(rel_tp_tr, rel_fp_tr, rel_fn_tr, params)
    else:
        tr_scores = {'micro_p': 0, 'micro_r': 0, 'micro_f': 0}
    if is_eval_ev > 0:
        ev_scores = evaluate_ev(fids=fidss,
                                all_ent_preds=ent_preds,
                                all_words=wordss,
                                all_offsets=offsetss,
                                all_span_terms=all_ner_terms,
                                all_span_indices=span_indicess,
                                all_sub_to_words=sub_to_wordss,
                                all_ev_preds=ev_preds,
                                params=params,
                                gold_dir=eval_dir,
                                result_dir=result_dir)
    else:
        ev_scores = {}

    # Print estimation scores here
    if not params['predict'] or (params['predict'] and not params['gold_eval']):
        print()
        print('-----OUR EVALUATIONS (NOT RECOMMEND)-----')
        print()
        print(ner_res)
        print()
        print(
            "ENT: P/R/F1 = {:.02f}\t{:.02f}\t{:.02f} ".format(ner_score[-1][1], ner_score[-1][2],
                                                              ner_score[-1][3]), end="",
        )
        print()
        print(
            "REL: P/R/F1 = {:.02f}\t{:.02f}\t{:.02f} ".format(
                tr_scores["micro_p"] * 100,
                tr_scores["micro_r"] * 100,
                tr_scores["micro_f"] * 100,
            ),
            end="",
        )
        print()
        print('Total matched indice relations', total_rel_matched_indices)
        print('Total matched type relations', total_rel_matched_types)

    print()
    print('-----EVALUATING BY N2C2 SCRIPT (FOR ENT & REL)-----')
    print()
    print('STRICT_MATCHING:')
    print_scores('NER', scores['NER'], 'st')
    print()
    print('SOFT_MATCHING:')
    print_scores('NER', scores['NER'], 'so')
    if is_eval_rel:
        print()
        print('STRICT_MATCHING:')
        print_scores('REL', scores['REL'], 'st')
        print()
        print('SOFT_MATCHING:')
        print_scores('REL', scores['REL'], 'so')
    else:
        if params['skip_rel']:
            print('Not evaluate REL')
        else:
            print('No relation')
    print()
    print('-----EVALUATING BY SCRIPT (FOR EV)-----')
    print()
    if len(ev_scores) > 0:
        sub_p, sub_r, sub_f = ev_scores['sub_scores'][0], ev_scores['sub_scores'][1], ev_scores['sub_scores'][2]
        mod_p, mod_r, mod_f = ev_scores['mod_scores'][0], ev_scores['mod_scores'][1], ev_scores['mod_scores'][2]
        tot_p, tot_r, tot_f = ev_scores['tot_scores'][0], ev_scores['tot_scores'][1], ev_scores['tot_scores'][2]
        print('SUB : P/R/F1 = {:.02f}\t{:.02f}\t{:.02f} '.format(sub_p, sub_r, sub_f), end="")
        print()
        print('MOD : P/R/F1 = {:.02f}\t{:.02f}\t{:.02f} '.format(mod_p, mod_r, mod_f), end="")
        print()
        print('TOT : P/R/F1 = {:.02f}\t{:.02f}\t{:.02f} '.format(tot_p, tot_r, tot_f), end="")
        print()
    else:
        print('No event/Not evaluate EV/error when evaluating by CG script')
        print()
    print()
    print()
    t_end = time.time()
    print('Elapsed time: {}'.format(_humanized_time(t_end - t_start)))
    print()

    # Do saving models
    if not params['predict']:
        # ! ========== KHOA WAS HERE ==========
        ner_f1score = ner_score[-1][3]

        # ner_fscore = ner_f1score * 100
        # get the best score by n2c2 instead
        ner_fscore = scores['NER']['micro']['st_f']

        if is_eval_rel:
            rel_fscore = scores['REL']['micro']['st_f']
        else:
            rel_fscore = 0

        if len(ev_scores) > 0:
            ev_fscore = ev_scores['tot_scores'][2]
        else:
            ev_fscore = 0

        if params['ner_epoch'] >= (params['epoch'] - 1):
            best_score = ner_fscore
        elif params['rel_epoch'] >= (params['epoch'] - 1):
            best_score = rel_fscore
        else:
            best_score = ev_fscore
        # Save models:
        if params['save_ner']:
            ner_model_path = params['ner_model_dir']
            utils.handle_checkpoints(
                model=model.NER_layer,
                checkpoint_dir=ner_model_path,
                params={
                    "filename": "ner_base",
                    "epoch": epoch,
                    "fscore": ner_fscore,
                    "ner_fscore": ner_fscore,
                    "rel_fscore": rel_fscore,
                    "ev_fscore": ev_fscore,
                    'device': params['device']
                },
                filter_func=utils.save_best_fscore,
                num_saved=1
            )

        if params['save_rel']:
            rel_model_path = params['rel_model_dir']
            utils.handle_checkpoints(
                model=model.REL_layer,
                checkpoint_dir=rel_model_path,
                params={
                    "filename": "rel_base",
                    "epoch": epoch,
                    "fscore": rel_fscore,
                    "ner_fscore": ner_fscore,
                    "rel_fscore": rel_fscore,
                    "ev_fscore": ev_fscore,
                    'device': params['device']
                },
                filter_func=utils.save_best_fscore,
                num_saved=1
            )
            if params['save_model_pipeline']:
                ner_model_path = params['ner_model_dir']
                utils.handle_checkpoints(
                    model=model.NER_layer,
                    checkpoint_dir=ner_model_path,
                    params={
                        "filename": "rel_base",
                        "epoch": epoch,
                        "fscore": rel_fscore,
                        "ner_fscore": ner_fscore,
                        "rel_fscore": rel_fscore,
                        "ev_fscore": ev_fscore,
                        'device': params['device']
                    },
                    filter_func=utils.save_best_fscore,
                    num_saved=1
                )
        if params['save_ev']:
            ev_model_path = params['ev_model_dir']
            utils.handle_checkpoints(
                model=model.EV_layer,
                checkpoint_dir=ev_model_path,
                params={
                    "filename": "ev_base",
                    "epoch": epoch,
                    "fscore": ev_fscore,
                    "ner_fscore": ner_fscore,
                    "rel_fscore": rel_fscore,
                    "ev_fscore": ev_fscore,
                    'device': params['device']
                },
                filter_func=utils.save_best_fscore,
                num_saved=1
            )
            if params['save_model_pipeline']:
                ner_model_path = params['ner_model_dir']
                rel_model_path = params['rel_model_dir']
                utils.handle_checkpoints(
                    model=model.NER_layer,
                    checkpoint_dir=ner_model_path,
                    params={
                        "filename": "ev_base",
                        "epoch": epoch,
                        "fscore": ev_fscore,
                        "ner_fscore": ner_fscore,
                        "rel_fscore": rel_fscore,
                        "ev_fscore": ev_fscore,
                        'device': params['device']
                    },
                    filter_func=utils.save_best_fscore,
                    num_saved=1
                )
                utils.handle_checkpoints(
                    model=model.REL_layer,
                    checkpoint_dir=rel_model_path,
                    params={
                        "filename": "ev_base",
                        "epoch": epoch,
                        "fscore": ev_fscore,
                        "ner_fscore": ner_fscore,
                        "rel_fscore": rel_fscore,
                        "ev_fscore": ev_fscore,
                        'device': params['device']
                    },
                    filter_func=utils.save_best_fscore,
                    num_saved=1
                )

        if params['save_all_models']:
            deepee_model_path = params['joint_model_dir']
            utils.handle_checkpoints(
                model=model,
                checkpoint_dir=deepee_model_path,
                params={
                    "filename": "deepee_base",
                    "epoch": epoch,
                    "fscore": best_score,
                    "ner_fscore": ner_fscore,
                    "rel_fscore": rel_fscore,
                    "ev_fscore": ev_fscore,
                    'device': params['device']
                },
                filter_func=utils.save_best_fscore,
                num_saved=1
            )
            print("Saved all models")
        # ! ===================================

    if len(ev_scores) > 0:
        return ner_score, is_eval_rel, tr_scores, scores, {'p': ev_scores['tot_scores'][0],
                                                           'r': ev_scores['tot_scores'][1],
                                                           'f': ev_scores['tot_scores'][2]}
    else:
        return ner_score, is_eval_rel, tr_scores, scores, ev_scores


def print_scores(k, v, stoso):
    print(
        k + "(MICRO): P/R/F1 = {:.02f}\t{:.02f}\t{:.02f} , (MACRO): P/R/F1 = {:.02f}\t{:.02f}\t{:.02f} ".format(
            v['micro'][stoso + '_p'], v['micro'][stoso + '_r'], v['micro'][stoso + '_f'],
            v['macro'][stoso + '_p'], v['macro'][stoso + '_r'], v['macro'][stoso + '_f']), end="",
    )
    print()

def predict_bio(model, result_dir, eval_dataloader, eval_data, g_entity_ids_, params):
    mapping_id_tag = params['mappings']['nn_mapping']['id_tag_mapping']

    # store predicted entities
    ent_preds = []

    # store predicted events
    ev_preds = []

    fidss, wordss, offsetss, sub_to_wordss, span_indicess = [], [], [], [], []

    # entity and relation output
    ent_anns = []
    rel_anns = []

    # Evaluation phase
    model.eval()

    all_ner_preds, all_ner_golds, all_ner_terms = [], [], []

    is_eval_ev = False

    for step, batch in enumerate(
            tqdm(eval_dataloader, desc="Iteration", leave=False)
    ):
        eval_data_ids = batch
        tensors = utils.get_tensors(eval_data_ids, eval_data, params)

        nn_tokens, nn_ids, nn_token_mask, nn_attention_mask, nn_span_indices, nn_span_labels, nn_span_labels_match_rel, nn_entity_masks, nn_trigger_masks, nn_gtruth, nn_l2r, _, \
        nn_truth_ev, nn_ev_idxs, ev_lbls, etypes, _ = tensors

        fids = [
            eval_data["fids"][data_id] for data_id in eval_data_ids[0].tolist()
        ]
        offsets = [
            eval_data["offsets"][data_id]
            for data_id in eval_data_ids[0].tolist()
        ]
        words = [
            eval_data["words"][data_id] for data_id in eval_data_ids[0].tolist()
        ]
        sub_to_words = [
            eval_data["sub_to_words"][data_id]
            for data_id in eval_data_ids[0].tolist()
        ]
        subwords = [
            eval_data["subwords"][data_id]
            for data_id in eval_data_ids[0].tolist()
        ]
        gold_entities = [
            eval_data["entities"][data_id]
            for data_id in eval_data_ids[0].tolist()
        ]

        with torch.no_grad():
            ner_out, rel_out, ev_out = model(tensors, params)

        ner_preds = ner_out['preds']

        ner_terms = ner_out['terms']

        all_ner_terms.append(ner_terms)

        for sentence_idx, ner_pred in enumerate(ner_preds):
            all_ner_golds.append(
                [
                    (
                        sub_to_words[sentence_idx][span_start],
                        sub_to_words[sentence_idx][span_end],
                        mapping_id_tag[label_id],
                    )
                    for (
                            span_start,
                            span_end,
                        ), label_ids in gold_entities[sentence_idx].items()
                    for label_id in label_ids
                ]
            )

            pred_entities = []
            for span_id, ner_pred_id in enumerate(ner_pred):
                span_start, span_end = nn_span_indices[sentence_idx][span_id]
                span_start, span_end = span_start.item(), span_end.item()
                if (ner_pred_id > 0
                        and span_start in sub_to_words[sentence_idx]
                        and span_end in sub_to_words[sentence_idx]
                ):
                    pred_entities.append(
                        (
                            sub_to_words[sentence_idx][span_start],
                            sub_to_words[sentence_idx][span_end],
                            mapping_id_tag[ner_pred_id],
                        )
                    )
            all_ner_preds.append(pred_entities)

        # entity prediction
        ent_ann = {'span_indices': nn_span_indices, 'ner_preds': ner_out['preds'], 'words': words,
                   'offsets': offsets, 'sub_to_words': sub_to_words, 'subwords': subwords,
                   'ner_terms': ner_terms}
        ent_anns.append(ent_ann)

        fidss.append(fids)

        wordss.append(words)
        offsetss.append(offsets)
        sub_to_wordss.append(sub_to_words)

        # relation prediction
        if rel_out != None:
            pairs_idx = rel_out['pairs_idx']
            rel_pred = rel_out['preds']

            rel_ann = {'pairs_idx': pairs_idx, 'rel_preds': rel_pred}
            rel_anns.append(rel_ann)
        else:
            rel_anns.append({})

        # event prediction
        if ev_out != None:
            # add predicted entity
            ent_preds.append(ner_out["nner_preds"])

            # add predicted events
            ev_preds.append(ev_out)

            span_indicess.append(
                [
                    indice.detach().cpu().numpy()
                    for indice in ner_out["span_indices"]
                ]
            )
            is_eval_ev = True
        else:
            ent_preds.append([])
            ev_preds.append([])

            span_indicess.append([])

        # Clear GPU unused RAM:
        if params['gpu'] >= 0:
            torch.cuda.empty_cache()
    # write entity and relation prediction
    _ = write_entity_relations(
        result_dir=result_dir,
        fidss=fidss,
        ent_anns=ent_anns,
        rel_anns=rel_anns,
        params=params
    )

    if is_eval_ev > 0:
        write_events(fids=fidss,
                     all_ent_preds=ent_preds,
                     all_words=wordss,
                     all_offsets=offsetss,
                     all_span_terms=all_ner_terms,
                     all_span_indices=span_indicess,
                     all_sub_to_words=sub_to_wordss,
                     all_ev_preds=ev_preds,
                     g_entity_ids_=g_entity_ids_,
                     params=params,
                     result_dir=result_dir)