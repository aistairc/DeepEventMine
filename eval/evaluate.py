import torch
from tqdm import tqdm

from eval.evalRE import write_entity_relations
from eval.evalEV import write_events
from utils import utils


def predict(model, eval_dir, result_dir, eval_dataloader, eval_data, params):
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

        nn_tokens, nn_ids, nn_token_mask, nn_attention_mask, nn_span_indices, nn_span_labels, nn_span_labels_match_rel, nn_entity_masks, nn_trigger_masks, _, \
        etypes, _ = tensors

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
                     params=params,
                     result_dir=result_dir)
