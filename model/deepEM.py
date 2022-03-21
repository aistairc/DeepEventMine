import copy
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as f
from torch import nn
from torch.autograd import Variable

from model import EVNet
from model import RELNet
from model.NERNet import NestedNERModel
from utils import utils

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

    def process_ner_output(self, nn_tokens, nn_ids, nn_token_mask, nn_attention_mask, nn_entity_masks, nn_trigger_masks,
                           nn_span_labels, span_terms, max_span_labels, nn_span_indices):
        """Process NER output to prepare for training relation and event layers"""

        # entity output
        ner_preds = {}

        # predict entity
        ner_loss, e_preds, e_golds, sentence_sections, span_masks, embeddings, sentence_emb, trigger_indices = self.NER_layer(
            all_tokens=nn_tokens,
            all_ids=nn_ids,
            all_token_masks=nn_token_mask,
            all_attention_masks=nn_attention_mask,
            all_entity_masks=nn_entity_masks,
            all_trigger_masks=nn_trigger_masks,
            all_span_labels=nn_span_labels,
        )

        # ! Note that these below lines run on CPU
        sentence_sections = sentence_sections.detach().cpu().numpy()[:-1]
        all_span_masks = span_masks.detach() > 0

        # Embedding of each span
        embeddings = torch.split(embeddings, torch.sum(all_span_masks, dim=-1).tolist())

        # Pred of each span
        e_preds = np.split(e_preds.astype(int), sentence_sections)
        e_preds = [pred.flatten() for pred in e_preds]
        ner_preds['preds'] = e_preds

        e_golds = np.split(e_golds.astype(int), sentence_sections)
        e_golds = [gold.flatten() for gold in e_golds]
        ner_preds['golds'] = e_golds
        ner_preds['gold_terms'] = copy.deepcopy(span_terms)

        replace_term = True
        if self.params['predict']:
            if self.params['gold_eval'] or (self.params['pipelines'] and self.params['pipe_flag'] != 0):
                replace_term = False

        if self.params["ner_predict_all"]:
            if self.params['predict']:
                if self.params['gold_eval'] or (self.params['pipelines'] and self.params['pipe_flag'] != 0):
                    e_preds = e_golds
                    span_terms = ner_preds['gold_terms']
            else:
                if self.params['skip_ner'] and self.params['skip_rel'] and self.params['use_gold_ner'] and self.params[
                    'use_gold_rel']:
                    e_preds = e_golds
                    span_terms = ner_preds['gold_terms']

            if replace_term:
                for items in span_terms:
                    items.term2id.clear()
                    items.id2term.clear()

                # Overwrite triggers
                if self.trigger_id == -1:
                    self.trigger_id = utils.get_max_entity_id(span_terms) + 10000

                trigger_idx = self.trigger_id + 1
                for sentence_idx, span_preds in enumerate(e_preds):
                    for pred_idx, label_id in enumerate(span_preds):
                        if label_id > 0:
                            term = "T" + str(trigger_idx)

                            # check trigger
                            if label_id in self.params['mappings']['nn_mapping']['trTypes_Ids']:
                                term = "TR" + str(trigger_idx)

                            span_terms[sentence_idx].id2term[pred_idx] = term
                            span_terms[sentence_idx].term2id[term] = pred_idx
                            trigger_idx += 1

                self.trigger_id = trigger_idx
        else:
            if replace_term:
                # Overwrite triggers
                if self.trigger_id == -1:
                    self.trigger_id = utils.get_max_entity_id(span_terms) + 10000

                trigger_idx = self.trigger_id + 1
                for sentence_idx, span_preds in enumerate(e_preds):
                    # Update gold labels

                    # store gold entity index (a1)
                    a1ent_set = set()

                    for span_idx, span_term in span_terms[sentence_idx].id2term.items():

                        if span_term != "O" and not span_term.startswith("TR") and span_preds[span_idx] != 255:

                            # but do not replace for entity in a2 files
                            span_label = span_terms[sentence_idx].id2label[
                                span_idx]  # entity type, e.g: Gene_or_gene_product
                            if span_label not in self.params['a2_entities']:
                                # replace for entity (using gold entity)
                                span_preds[span_idx] = e_golds[sentence_idx][span_idx]

                                # save this index to ignore prediction
                                a1ent_set.add(span_idx)

                    for pred_idx, label_id in enumerate(span_preds):
                        span_term = span_terms[sentence_idx].id2term.get(pred_idx, "O")

                        # if this entity in a1: skip this span
                        if pred_idx in a1ent_set:
                            continue

                        remove_span = False

                        # add prediction for trigger or entity a2
                        if label_id > 0:
                            term = ''

                            # check trigger
                            if label_id in self.params['mappings']['nn_mapping']['trTypes_Ids']:
                                term = "TR" + str(trigger_idx)

                            # is entity
                            else:
                                etype_label = self.params['mappings']['nn_mapping']['id_tag_mapping'][label_id]

                                # check this entity type in a2 or not
                                if etype_label in self.params['a2_entities']:
                                    term = "T" + str(trigger_idx)
                                else:
                                    remove_span = True

                            if len(term) > 0:
                                span_terms[sentence_idx].id2term[pred_idx] = term
                                span_terms[sentence_idx].term2id[term] = pred_idx
                                trigger_idx += 1

                        # null prediction
                        if label_id == 0 or remove_span:
                            # do not write anything
                            span_preds[pred_idx] = 0

                            # remove this span
                            if span_term.startswith("T"):
                                del span_terms[sentence_idx].id2term[pred_idx]
                                del span_terms[sentence_idx].term2id[span_term]

                    span_preds[span_preds == 255] = 0
                self.trigger_id = trigger_idx

        num_padding = max_span_labels * self.params["ner_label_limit"]

        e_preds = [np.pad(pred, (0, num_padding - pred.shape[0]),
                          'constant', constant_values=-1) for pred in e_preds]
        e_golds = [np.pad(gold, (0, num_padding - gold.shape[0]),
                          'constant', constant_values=-1) for gold in e_golds]

        e_preds = torch.tensor(e_preds, device=self.device)
        nn_span_labels = torch.tensor(e_golds, device=self.device)

        embeddings = [f.pad(embedding, (0, 0, 0, max_span_labels - embedding.shape[0]),
                            'constant', value=0) for embedding in embeddings]

        embeddings = torch.stack(embeddings)
        embeddings = embeddings.unsqueeze(dim=2).expand(-1, -1, self.params["ner_label_limit"], -1)
        embeddings = embeddings.reshape(embeddings.size(0), -1, embeddings.size(-1))

        # output for ner
        ner_preds['loss'] = ner_loss
        ner_preds['terms'] = span_terms
        ner_preds['span_indices'] = nn_span_indices

        # For pre-train event layer
        use_gold = False
        if (not self.params['predict'] and self.params['skip_ner'] and self.params['skip_rel'] and self.params[
            'use_gold_ner'] and self.params['use_gold_rel']) or (self.params['gold_eval'] or self.params['pipelines']):
            use_gold = True
        if use_gold:
            ner_preds['nner_preds'] = e_golds
        else:
            ner_preds['nner_preds'] = e_preds.detach().cpu().numpy()

        return embeddings, e_preds, e_golds, nn_span_labels, sentence_emb, ner_preds

    def generate_entity_pairs_4rel(self, bert_embeds, p_span_indices, g_span_indices):
        """Prepare entity pairs for relation candidates"""

        # use gold or predicted span indices
        # training mode
        if not self.params['predict']:
            if self.training and self.params['use_gold_ner']:
                use_gold = True
            # train relation only
            elif not self.training and self.params['skip_ner'] and self.params['rel_epoch'] >= (
                    self.params['epoch'] - 1) and self.params['use_gold_ner']:
                use_gold = True
            # train event only
            elif not self.training and self.params['skip_ner'] and self.params['skip_rel'] and self.params[
                'use_gold_rel']:
                use_gold = True
            else:
                use_gold = False

        # predict mode
        else:
            if self.params['gold_eval'] or self.params['pipelines']:
                use_gold = True
            else:
                use_gold = False

        if use_gold:
            span_indices = g_span_indices
        else:
            span_indices = p_span_indices

        # positive indices
        pos_indices = (span_indices > 0).nonzero().transpose(0, 1).long()

        # entity types
        e_types = torch.full((span_indices.shape[0], span_indices.shape[1]), -1, dtype=torch.int64,
                             device=self.device)

        # entity and trigger indices
        e_indices = torch.zeros((span_indices.shape[0], span_indices.shape[1]), dtype=torch.long)
        tr_indices = torch.zeros((span_indices.shape), dtype=torch.int64, device=self.device)

        # store entity indices in batch and list of triggers
        batch_eids_list = defaultdict(list)
        tr_list = []

        # store entity in each batch
        batch_ent_list = defaultdict(list)

        for batch_id, a1id in enumerate(pos_indices[0]):

            # index
            a2id = pos_indices[1][batch_id]

            # entity type
            type_a1 = self.params['mappings']['nn_mapping']['tag2type_map'][span_indices[a1id][a2id].item()]
            e_types[a1id][a2id] = torch.tensor(type_a1, device=self.device)

            # masked
            e_indices[a1id][a2id] = 1

            # trigger
            if type_a1 in self.params['trTypes_Ids']:
                tr_indices[a1id][a2id] = 1
                tr_list.append((a1id, a2id))

            # entity
            else:
                batch_ent_list[a1id.item()].append(a2id)

            batch_eids_list[a1id.item()].append(a2id)

        # prepare for entity and trigger embeddings
        e_embeds = bert_embeds.clone()
        tr_embeds = bert_embeds.clone()
        e_embeds[e_indices == 0] = torch.zeros((bert_embeds.shape[2]), dtype=bert_embeds.dtype, device=self.device)
        tr_embeds[tr_indices == 0] = torch.zeros((bert_embeds.shape[2]), dtype=bert_embeds.dtype, device=self.device)

        # indices of pairs (trigger-entity OR trigger-trigger) for relation candidates
        pair_indices = []

        if len(tr_list):
            for batch_id, trig_id in tr_list:
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
                        pair_indices.append(batch_pair_idx)

            if len(pair_indices) > 0:
                pair_indices = torch.cat(pair_indices, dim=-1)

        return e_embeds, tr_embeds, e_types, tr_indices, pair_indices

    def _init_joint(self, n_epoch):
        """Flags to enable using the predicted from the previous output or not"""

        # init layer output
        rel_preds = None
        ev_preds = None

        # enable jointly training
        enable_rel = True
        enable_ev = True

        # training
        if not self.params['predict']:

            # pre-train ner only: unable relation and event layers
            if not self.params['skip_ner'] and n_epoch <= self.params['ner_epoch']:
                enable_rel = False
                enable_ev = False

            # pre-train relation only: unable event layer
            if not self.params['skip_rel'] and n_epoch <= self.params['rel_epoch']:
                enable_ev = False

        # predict on pipeline mode
        elif self.params['predict'] and self.params['pipelines']:

            # for ner
            if self.params['pipe_flag'] == 0:
                enable_rel = False
                enable_ev = False

            # for relation
            elif self.params['pipe_flag'] == 1:
                enable_rel = True
                enable_ev = False

            # for event
            else:
                enable_rel = False
                enable_ev = True

        return enable_rel, enable_ev, rel_preds, ev_preds

    def _accumulate_loss(self, ner_preds, rel_preds, ev_preds, n_epoch):
        """To calculate the total loss from the layers' loss"""

        # total loss
        acc_loss = 0

        if not self.params['predict']:
            # add ner loss
            if not self.params['skip_ner']:

                # add scaled loss according to the epoch range
                if n_epoch <= self.params['ner_epoch_limit']:
                    acc_loss = ner_preds['loss'] * self.params['ner_loss_weight_main']
                else:
                    acc_loss = ner_preds['loss'] * self.params['ner_loss_weight_minor']

            # add relation loss
            if not self.params['skip_rel'] and rel_preds != None:

                # check non-empty
                if rel_preds['valid']:

                    # add scaled loss according to the epoch range
                    if n_epoch <= self.params['rel_epoch_limit'] and n_epoch > self.params['ner_epoch_limit']:
                        acc_loss += rel_preds['loss'] * self.params['rel_loss_weight_main']
                    else:
                        acc_loss += rel_preds['loss'] * self.params['rel_loss_weight_minor']

            # add event loss
            if ev_preds != None:

                # add scaled loss according to the epoch range
                if n_epoch <= self.params['rel_epoch_limit']:
                    acc_loss += ev_preds['loss'] * self.params['ev_loss_weight_minor']
                else:
                    acc_loss += ev_preds['loss'] * self.params['ev_loss_weight_main']

        # zero
        if acc_loss == 0:
            acc_loss = Variable(torch.zeros(1, device=self.params['device']))

        return acc_loss

    def forward(self, batch_input, n_epoch=0):

        """Joint model interface."""

        # 1 - get input
        nn_tokens, nn_ids, nn_token_mask, nn_attention_mask, nn_span_indices, nn_span_labels, nn_span_labels_match_rel, nn_entity_masks, nn_trigger_masks, nn_gtruth, nn_l2r, span_terms, \
        nn_truth_ev, nn_ev_idxs, ev_lbls, etypes, max_span_labels = batch_input

        # 2 - predict entity and process output
        embeddings, e_preds, e_golds, nn_span_labels, sentence_emb, ner_preds = self.process_ner_output(
            nn_tokens, nn_ids,
            nn_token_mask,
            nn_attention_mask,
            nn_entity_masks,
            nn_trigger_masks,
            nn_span_labels,
            span_terms,
            max_span_labels,
            nn_span_indices
        )

        # 3 - initialize joint training
        enable_rel, enable_ev, rel_preds, ev_preds = self._init_joint(n_epoch)

        # 4 - joint training
        if enable_rel or enable_ev:

            # 4.1 - prepare input for joint model
            e_embeds, tr_embeds, e_types, tr_ids, pair_indices = self.generate_entity_pairs_4rel(bert_embeds=embeddings,
                                                                                                 p_span_indices=e_preds,
                                                                                                 g_span_indices=nn_span_labels)

            # check non-empty
            if len(pair_indices) > 0:

                joint_input = {'preds': e_preds, 'golds': e_golds, 'embeddings': embeddings,
                               'ent_embeds': e_embeds, 'tr_embeds': tr_embeds, 'tr_ids': tr_ids,
                               'ent_types': e_types, 'pairs_idx': pair_indices, 'e_types': etypes.long(),
                               'l2rs': nn_l2r,
                               'gtruths': nn_gtruth, 'truth_evs': nn_truth_ev, 'ev_idxs': nn_ev_idxs,
                               'ev_lbls': ev_lbls,
                               'sentence_embeds': sentence_emb}

                # 4.2 - training relation layer
                if enable_rel:
                    rel_preds = self.REL_layer(joint_input)

                # 4.4 - training event layer
                if enable_ev:

                    # get relation output
                    rel_preds = self.REL_layer(joint_input)

                    # check non-empty relation
                    if rel_preds['valid']:
                        # call event layer
                        ev_preds = self.EV_layer(joint_input, rel_preds, n_epoch)

        # joint model loss
        acc_loss = self._accumulate_loss(ner_preds, rel_preds, ev_preds, n_epoch)

        return ner_preds, rel_preds, ev_preds, acc_loss
