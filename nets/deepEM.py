from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as f
from torch import nn

from nets import EVNet
from nets import RELNet
from nets.NERNet import NestedNERModel
from loader.prepNN import utils

cpu_device = torch.device("cpu")


class DeepEM(nn.Module):
    """
    Network architecture
    """

    def __init__(self, params):
        super(DeepEM, self).__init__()

        sizes = params['voc_sizes']
        device = params['device']

        self.NER_layer = NestedNERModel.from_pretrained(params['bert_model'], params=params)
        self.REL_layer = RELNet.RELModel(params, sizes)
        self.EV_layer = EVNet.EVModel(params, sizes)

        self.trigger_id = -1

        if params['train']:
            self.beta = 1
        else:
            self.beta = params['beta']

        self.device = device
        self.params = params

    def is_tr(self, label):
        nn_tr_types_ids = self.params['mappings']['nn_mapping']['trTypes_Ids']
        return label in nn_tr_types_ids

    def gen_data_after_ner(self, bert_out, preds):

        lbls = preds

        labeled_spans = (lbls > 0).nonzero().transpose(0, 1).long()

        ent_types = torch.full((lbls.shape[0], lbls.shape[1]), -1, dtype=torch.int64, device=self.device)

        e_ids = torch.zeros((lbls.shape[0], lbls.shape[1]), dtype=torch.long)
        tr_ids = torch.zeros((lbls.shape), dtype=torch.int64, device=self.device)

        batch_eids_list = defaultdict(list)
        trig_list = []

        # store only entity in each batch
        batch_ent_list = defaultdict(list)

        for idx, i in enumerate(labeled_spans[0]):
            j = labeled_spans[1][idx]
            type_a1 = self.params['mappings']['nn_mapping']['tag2type_map'][lbls[i][j].item()]
            ent_types[i][j] = torch.tensor(type_a1, device=self.device)
            e_ids[i][j] = 1

            if type_a1 in self.params['trTypes_Ids']:
                tr_ids[i][j] = 1
                trig_list.append((i, j))
            else:
                batch_ent_list[i.item()].append(j)

            batch_eids_list[i.item()].append(j)

        ent_embeds = bert_out.clone()
        tr_embeds = bert_out.clone()
        ent_embeds[e_ids == 0] = torch.zeros((bert_out.shape[2]), dtype=bert_out.dtype, device=self.device)
        tr_embeds[tr_ids == 0] = torch.zeros((bert_out.shape[2]), dtype=bert_out.dtype, device=self.device)

        pairs_idx = []

        if len(trig_list):
            for batch_id, trig_id in trig_list:
                if len(batch_eids_list[batch_id.item()]) > 1:

                    # enable relation between triggers
                    if self.params['enable_triggers_pair']:
                        # get all entity ids in this batch
                        b_eids = batch_eids_list[batch_id.item()].copy()

                        # remove this trigger to avoid self relation
                        b_eids.remove(trig_id.clone().detach())

                    # or only between trigger and entity
                    else:
                        # pair with only entity
                        b_eids = batch_ent_list[batch_id.item()].copy()

                    # check empty
                    if len(b_eids) > 0:
                        # make pairs
                        batch_pair_idx = torch.tensor([[batch_id], [trig_id]]).repeat(1, len(b_eids))
                        batch_pair_idx = torch.cat(
                            (batch_pair_idx, torch.tensor(b_eids).view(1, len(b_eids))), dim=0)

                        # add to pairs
                        pairs_idx.append(batch_pair_idx)

            if len(pairs_idx) > 0:
                pairs_idx = torch.cat(pairs_idx, dim=-1)

        return ent_embeds, tr_embeds, ent_types, tr_ids, pairs_idx

    def calculate(self, batch_input):

        # for output
        ner_out = {}

        # input
        nn_tokens, nn_ids, nn_token_mask, nn_attention_mask, nn_span_indices, nn_span_labels, nn_span_labels_match_rel, nn_entity_masks, nn_trigger_masks, span_terms, \
        etypes, max_span_labels = batch_input

        # predict entity
        preds, golds, sentence_sections, span_masks, embeddings, sentence_emb, trigger_indices = self.NER_layer(
            all_tokens=nn_tokens,
            all_ids=nn_ids,
            all_token_masks=nn_token_mask,
            all_attention_masks=nn_attention_mask,
            all_entity_masks=nn_entity_masks,
            all_trigger_masks=nn_trigger_masks,
            all_span_labels=nn_span_labels,
        )

        # run on CPU
        sentence_sections = sentence_sections.detach().cpu().numpy()[:-1]
        all_span_masks = span_masks.detach() > 0

        # Embedding of each span
        embeddings = torch.split(embeddings, torch.sum(all_span_masks, dim=-1).tolist())

        # Pred of each span
        preds = np.split(preds.astype(int), sentence_sections)
        preds = [pred.flatten() for pred in preds]
        ner_out['preds'] = preds

        golds = np.split(golds.astype(int), sentence_sections)
        golds = [gold.flatten() for gold in golds]

        # Overwrite triggers
        if self.trigger_id == -1:
            self.trigger_id = utils.get_max_entity_id(span_terms) + 10000

        trigger_idx = self.trigger_id + 1
        for sentence_idx, span_preds in enumerate(preds):
            # Update gold labels
            for span_idx, span_term in span_terms[sentence_idx].id2term.items():
                if span_term != "O" and not span_term.startswith("TR") and span_preds[span_idx] != 255:
                    span_preds[span_idx] = golds[sentence_idx][span_idx]

            for pred_idx, label_id in enumerate(span_preds):
                span_term = span_terms[sentence_idx].id2term.get(pred_idx, "O")
                if span_term == "O" or span_term.startswith("TR"):
                    if self.is_tr(label_id):
                        term = "TR" + str(trigger_idx)
                        span_terms[sentence_idx].id2term[pred_idx] = term
                        span_terms[sentence_idx].term2id[term] = pred_idx
                        trigger_idx += 1
                    else:
                        span_preds[pred_idx] = 0
                        if span_term.startswith("TR"):
                            del span_terms[sentence_idx].id2term[pred_idx]
                            del span_terms[sentence_idx].term2id[span_term]

            span_preds[span_preds == 255] = 0
        self.trigger_id = trigger_idx

        num_padding = max_span_labels * self.params["ner_label_limit"]

        preds = [np.pad(pred, (0, num_padding - pred.shape[0]),
                        'constant', constant_values=-1) for pred in preds]
        golds = [np.pad(gold, (0, num_padding - gold.shape[0]),
                        'constant', constant_values=-1) for gold in golds]

        preds = torch.tensor(preds, device=self.device)

        embeddings = [f.pad(embedding, (0, 0, 0, max_span_labels - embedding.shape[0]),
                            'constant', value=0) for embedding in embeddings]

        embeddings = torch.stack(embeddings)
        embeddings = embeddings.unsqueeze(dim=2).expand(-1, -1, self.params["ner_label_limit"], -1)
        embeddings = embeddings.reshape(embeddings.size(0), -1, embeddings.size(-1))

        ent_embeds, tr_embeds, ent_types, tr_ids, pairs_idx = self.gen_data_after_ner(
            embeddings,
            preds=preds
        )
        ner_preds = {'preds': preds, 'golds': golds, 'embeddings': embeddings,
                     'ent_embeds': ent_embeds, 'tr_embeds': tr_embeds, 'tr_ids': tr_ids,
                     'ent_types': ent_types, 'pairs_idx': pairs_idx, 'e_types': etypes.long(),
                     'sentence_embeds': sentence_emb}

        rel_preds = self.REL_layer(ner_preds)
        if rel_preds['next']:

            ev_preds, empty_pred = self.EV_layer(ner_preds, rel_preds)

            if empty_pred == True:
                ev_preds = None


        else:
            rel_preds = None
            ev_preds = None

        ner_out['terms'] = span_terms
        ner_out['span_indices'] = nn_span_indices

        nner_preds = preds.detach().cpu().numpy()
        ner_out['nner_preds'] = nner_preds

        return ner_out, rel_preds, ev_preds

    def forward(self, batch_input, parameters):

        ner_preds, rel_preds, ev_preds = self.calculate(batch_input)

        return ner_preds, rel_preds, ev_preds
