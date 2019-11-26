# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn

from bert.modeling import BertModel, BertPreTrainedModel


class NestedNERModel(BertPreTrainedModel):
    def __init__(self, config, params):
        super(NestedNERModel, self).__init__(config)

        self.params = params

        self.ner_label_limit = params["ner_label_limit"]
        self.thresholds = params["ner_threshold"]

        self.num_entities = params["mappings"]["nn_mapping"]["num_entities"]
        self.num_triggers = params["mappings"]["nn_mapping"]["num_triggers"]

        self.max_span_width = params["max_span_width"]

        self.bert = BertModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.entity_classifier = nn.Linear(config.hidden_size * 3, self.num_entities)
        self.trigger_classifier = nn.Linear(config.hidden_size * 3, self.num_triggers)

        self.register_buffer(
            "label_ids",
            torch.tensor(
                params["mappings"]["nn_mapping"]["mlb"].classes_, dtype=torch.uint8
            ),
        )

        self.apply(self.init_bert_weights)
        self.params = params

    def forward(
            self,
            all_tokens,
            all_ids,
            all_token_masks,
            all_attention_masks,
            all_entity_masks,
            all_trigger_masks,
            all_span_labels=None,
    ):
        device = all_ids.device
        max_span_width = self.max_span_width

        embeddings, sentence_embedding = self.bert(
            all_ids, attention_mask=all_attention_masks, output_all_encoded_layers=False
        )  # (B, S, H) (B, 128, 768)

        flattened_token_masks = all_token_masks.flatten()  # (B * S, )

        flattened_embedding_indices = torch.arange(
            flattened_token_masks.size(0), device=device
        ).masked_select(
            flattened_token_masks
        )  # (all_actual_tokens, )

        flattened_embeddings = torch.index_select(
            embeddings.view(-1, embeddings.size(-1)), 0, flattened_embedding_indices
        )  # (all_actual_tokens, H)

        span_starts = (
            torch.arange(flattened_embeddings.size(0), device=device)
                .view(-1, 1)
                .repeat(1, max_span_width)
        )  # (all_actual_tokens, max_span_width)

        flattened_span_starts = (
            span_starts.flatten()
        )  # (all_actual_tokens * max_span_width, )

        span_ends = span_starts + torch.arange(max_span_width, device=device).view(
            1, -1
        )  # (all_actual_tokens, max_span_width)

        flattened_span_ends = (
            span_ends.flatten()
        )  # (all_actual_tokens * max_span_width, )

        sentence_indices = (
            torch.arange(embeddings.size(0), device=device)
                .view(-1, 1)
                .repeat(1, embeddings.size(1))
        )  # (B, S)

        flattened_sentence_indices = sentence_indices.flatten().masked_select(
            flattened_token_masks
        )  # (all_actual_tokens, )

        span_start_sentence_indices = torch.index_select(
            flattened_sentence_indices, 0, flattened_span_starts
        )  # (all_actual_tokens * max_span_width, )

        span_end_sentence_indices = torch.index_select(
            flattened_sentence_indices,
            0,
            torch.min(
                flattened_span_ends,
                torch.ones(
                    flattened_span_ends.size(),
                    dtype=flattened_span_ends.dtype,
                    device=device,
                )
                * (span_ends.size(0) - 1),
            ),
        )  # (all_actual_tokens * max_span_width, )

        candidate_mask = torch.eq(
            span_start_sentence_indices,
            span_end_sentence_indices,  # Checking both indices is in the same sentence
        ) & torch.lt(
            flattened_span_ends, span_ends.size(0)
        )  # (all_actual_tokens * max_span_width, )

        flattened_span_starts = flattened_span_starts.masked_select(
            candidate_mask
        )  # (all_valid_spans, )

        flattened_span_ends = flattened_span_ends.masked_select(
            candidate_mask
        )  # (all_valid_spans, )

        span_start_embeddings = torch.index_select(
            flattened_embeddings, 0, flattened_span_starts
        )  # (all_valid_spans, H)

        span_end_embeddings = torch.index_select(
            flattened_embeddings, 0, flattened_span_ends
        )  # (all_valid_spans, H)

        # For computing embedding mean
        mean_indices = flattened_span_starts.view(-1, 1) + torch.arange(
            max_span_width, device=device
        ).view(
            1, -1
        )  # (all_valid_spans, max_span_width)

        mean_indices_criteria = torch.gt(
            mean_indices, flattened_span_ends.view(-1, 1).repeat(1, max_span_width)
        )  # (all_valid_spans, max_span_width)

        mean_indices = torch.min(
            mean_indices, flattened_span_ends.view(-1, 1).repeat(1, max_span_width)
        )  # (all_valid_spans, max_span_width)

        span_mean_embeddings = torch.index_select(
            flattened_embeddings, 0, mean_indices.flatten()
        ).view(
            *mean_indices.size(), -1
        )  # (all_valid_spans, max_span_width, H)

        coeffs = torch.ones(
            mean_indices.size(), dtype=embeddings.dtype, device=device
        )  # (all_valid_spans, max_span_width)

        coeffs[mean_indices_criteria] = 0

        span_mean_embeddings = span_mean_embeddings * coeffs.unsqueeze(
            -1
        )  # (all_valid_spans, max_span_width, H)

        span_mean_embeddings = torch.sum(span_mean_embeddings, dim=1) / torch.sum(
            coeffs, dim=-1
        ).view(
            -1, 1
        )  # (all_valid_spans, H)

        combined_embeddings = torch.cat(
            (
                span_start_embeddings,
                span_mean_embeddings,
                span_end_embeddings,
            ),
            dim=1,
        )  # (all_valid_spans, H * 3 + distance_dim)

        # ! REDUCE
        if self.params['ner_reduce']:
            combined_embeddings = self.reduce(combined_embeddings)

        entity_preds = self.entity_classifier(
            combined_embeddings
        )  # (all_valid_spans, num_entities)

        trigger_preds = self.trigger_classifier(
            combined_embeddings
        )  # (all_valid_spans, num_triggers)

        all_span_masks = (all_entity_masks > -1) | (
                all_trigger_masks > -1
        )  # (B, max_spans)

        all_entity_masks = all_entity_masks[all_span_masks] > 0  # (all_valid_spans, )

        all_trigger_masks = all_trigger_masks[all_span_masks] > 0  # (all_valid_spans, )

        sentence_sections = all_span_masks.sum(dim=-1).cumsum(dim=-1)  # (B, )

        # The number of possible spans is all_valid_spans = K * (2 * N - K + 1) / 2
        # K: max_span_width
        # N: number of tokens
        actual_span_labels = all_span_labels[
            all_span_masks
        ]  # (all_valid_spans, num_entities + num_triggers)

        all_preds = torch.cat(
            (trigger_preds, entity_preds), dim=-1
        )  # (all_valid_spans, num_entities + num_triggers)

        # We could do this due to the independence between variables
        all_preds = torch.sigmoid(
            all_preds
        )  # (all_valid_spans, num_entities + num_triggers)

        # Clear values at invalid positions
        all_preds[~all_trigger_masks, : self.num_triggers] = 0
        all_preds[~all_entity_masks, self.num_triggers:] = 0

        # Support for random-noise adding trick
        entity_coeff = all_entity_masks.sum().float()
        trigger_coeff = all_trigger_masks.sum().float()
        denominator = entity_coeff + trigger_coeff

        entity_coeff /= denominator
        trigger_coeff /= denominator

        _, all_preds_top_indices = torch.topk(all_preds, k=self.ner_label_limit, dim=-1)

        # Convert binary value to label ids
        all_preds = (all_preds > self.thresholds) * self.label_ids
        all_golds = (actual_span_labels > 0) * self.label_ids

        # Stupid trick
        all_golds, _ = torch.sort(all_golds, dim=-1, descending=True)
        all_golds = torch.narrow(all_golds, 1, 0, self.ner_label_limit)

        all_preds = torch.gather(all_preds, dim=1, index=all_preds_top_indices)

        all_preds = all_preds.detach().cpu().numpy()
        all_golds = all_golds.detach().cpu().numpy()

        all_aligned_preds = []
        trigger_indices = []
        for idx, (preds, golds) in enumerate(zip(all_preds, all_golds)):
            # check trigger in preds
            for pred in preds:
                if pred in self.params['mappings']['nn_mapping']['trTypes_Ids']:
                    trigger_indices.append(idx)
                    break
            aligned_preds = []
            pred_set = set(preds) - {0}
            gold_set = set(golds) - {0}
            shared = pred_set & gold_set
            diff = pred_set - shared
            for gold in golds:
                if gold in shared:
                    aligned_preds.append(gold)
                else:
                    aligned_preds.append(diff.pop() if diff else 0)
            all_aligned_preds.append(aligned_preds)

        all_aligned_preds = np.array(all_aligned_preds)

        return (
            all_aligned_preds,
            all_golds,
            sentence_sections,
            all_span_masks,
            combined_embeddings,
            sentence_embedding,
            trigger_indices
        )
