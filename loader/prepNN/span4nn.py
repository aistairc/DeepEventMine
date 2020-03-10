"""Prepare data with span-based for training networks."""

import numpy as np
from collections import namedtuple

Term = namedtuple('Term', ['id2term', 'term2id', 'id2label'])

def get_span_index(
        span_start,
        span_end,
        max_span_width,
        max_sentence_length,
        index,
        limit
):
    assert span_start <= span_end
    assert index >= 0 and index < limit
    assert max_span_width > 0
    assert max_sentence_length > 0

    max_span_width = min(max_span_width, max_sentence_length)
    invalid_cases = max(
        0, span_start + max_span_width - max_sentence_length - 1
    )
    span_index = (
            (max_span_width - 1) * span_start
            + span_end
            - invalid_cases * (invalid_cases + 1) // 2
    )
    return span_index * limit + index


def get_batch_data(fid, entities, terms, valid_starts, sw_sentence, tokenizer, params):
    mlb = params["mappings"]["nn_mapping"]["mlb"]

    max_entity_width = params["max_entity_width"]
    max_trigger_width = params["max_trigger_width"]
    max_span_width = params["max_span_width"]

    tokens = [token for token, *_ in sw_sentence]

    num_tokens = len(tokens)

    token_mask = [1] * num_tokens

    # Account for [CLS] and [SEP] tokens
    if num_tokens > params['max_seq'] - 2:
        num_tokens = params['max_seq'] - 2
        tokens = tokens[:num_tokens]
        token_mask = token_mask[:num_tokens]

    ids = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens + ["[SEP]"])

    token_mask = [0] + token_mask + [0]

    # ! Whether use value 1 for [CLS] and [SEP]
    attention_mask = [1] * len(ids)

    # Generate spans here
    span_starts = np.tile(
        np.expand_dims(np.arange(num_tokens), 1), (1, max_span_width)
    )  # (num_tokens, max_span_width)

    span_ends = span_starts + np.expand_dims(
        np.arange(max_span_width), 0
    )  # (num_tokens, max_span_width)

    span_indices = []
    span_labels = []
    span_labels_match_rel = []
    entity_masks = []
    trigger_masks = []
    span_terms = Term({}, {}, {})

    for span_start, span_end in zip(
            span_starts.flatten(), span_ends.flatten()
    ):
        if span_start >= 0 and span_end < num_tokens:
            span_label = []  # No label
            span_term = []
            span_label_match_rel = 0

            entity_mask = 1
            trigger_mask = 1

            if span_end - span_start + 1 > max_entity_width:
                entity_mask = 0
            if span_end - span_start + 1 > max_trigger_width:
                trigger_mask = 0

            # Ignore spans containing incomplete words
            valid_span = True
            if not (params['predict'] and (params['pipelines'] and params['pipe_flag'] != 0)):
                if span_start not in valid_starts or (span_end + 1) not in valid_starts:
                    # Ensure that there is no entity label here
                    if not (params['predict'] and (params['pipelines'] and params['pipe_flag'] != 0)):
                        assert (span_start, span_end) not in entities

                        entity_mask = 0
                        trigger_mask = 0
                        valid_span = False

            if valid_span:
                if (span_start, span_end) in entities:
                    span_label = entities[(span_start, span_end)]
                    span_term = terms[(span_start, span_end)]

            if len(span_label) > params["ner_label_limit"]:
                print('over limit span_label', span_term)

            # For multiple labels
            for idx, (_, term_id) in enumerate(
                    sorted(zip(span_label, span_term), reverse=True)[:params["ner_label_limit"]]):
                span_index = get_span_index(span_start, span_end, max_span_width, num_tokens, idx,
                                            params["ner_label_limit"])
                span_terms.id2term[span_index] = term_id
                span_terms.term2id[term_id] = span_index

                # add entity type
                term_label = params['mappings']['nn_mapping']['id_tag_mapping'][span_label[0]]
                span_terms.id2label[span_index] = term_label

            span_label = mlb.transform([span_label])[-1]

            span_indices += [(span_start, span_end)] * params["ner_label_limit"]
            span_labels.append(span_label)
            span_labels_match_rel.append(span_label_match_rel)
            entity_masks.append(entity_mask)
            trigger_masks.append(trigger_mask)

    return {
        'tokens': tokens,
        'ids': ids,
        'token_mask': token_mask,
        'attention_mask': attention_mask,
        'span_indices': span_indices,
        'span_labels': span_labels,
        'span_labels_match_rel': span_labels_match_rel,
        'entity_masks': entity_masks,
        'trigger_masks': trigger_masks,
        'span_terms': span_terms
    }


def get_nn_data(fids, entitiess, termss, valid_startss, sw_sentences, tokenizer, params):
    samples = []

    for idx, sw_sentence in enumerate(sw_sentences):
        fid = fids[idx]
        entities = entitiess[idx]
        terms = termss[idx]
        valid_starts = valid_startss[idx]

        sample = get_batch_data(fid, entities, terms, valid_starts, sw_sentence, tokenizer,
                                params)
        samples.append(sample)

    all_tokens = []

    all_ids = [sample["ids"] for sample in samples]
    all_token_masks = [sample["token_mask"] for sample in samples]
    all_attention_masks = [sample["attention_mask"] for sample in samples]
    all_span_indices = [sample["span_indices"] for sample in samples]
    all_span_labels = [sample["span_labels"] for sample in samples]
    all_span_labels_match_rel = [sample["span_labels_match_rel"] for sample in samples]
    all_entity_masks = [sample["entity_masks"] for sample in samples]
    all_trigger_masks = [sample["trigger_masks"] for sample in samples]
    all_span_terms = [sample["span_terms"] for sample in samples]

    return {
        'tokens': all_tokens,
        'ids': all_ids,
        'token_mask': all_token_masks,
        'attention_mask': all_attention_masks,
        'span_indices': all_span_indices,
        'span_labels': all_span_labels,
        'span_labels_match_rel': all_span_labels_match_rel,
        'entity_masks': all_entity_masks,
        'trigger_masks': all_trigger_masks,
        'span_terms': all_span_terms
    }
