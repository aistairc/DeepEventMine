import numpy as np
import torch
import torch.nn.functional as f
from torch import nn

from eval.evalRE import calc_stats
from utils.utils import gelu


class RELModel(nn.Module):
    """Relation layer."""

    def __init__(self, params, sizes):
        super(RELModel, self).__init__()

        # entity type
        self.type_embed = nn.Embedding(num_embeddings=sizes['etype_size'] + 1,
                                       embedding_dim=params['etype_dim'],
                                       padding_idx=sizes['etype_size'])

        # entity dim
        if params['ner_reduce'] == False:
            ent_dim = params['bert_dim'] * 3 + params['etype_dim']
        else:
            ent_dim = params['ner_reduced_size'] + params['etype_dim']

        # layers
        self.hidden_layer1 = nn.Linear(in_features=2 * ent_dim + params['bert_dim'],
                                       out_features=params['hidden_dim'], bias=False)
        self.hidden_layer2 = nn.Linear(in_features=params['hidden_dim'],
                                       out_features=params['rel_reduced_size'], bias=False)
        self.l_class = nn.Linear(in_features=params['rel_reduced_size'],
                                 out_features=sizes['rel_size'])

        # others
        self.device = params['device']
        self.params = params
        self.sizes = sizes

    def _create_type_representation(self, bert_embeds, etypes_):
        """Create entity type embeddings"""

        # get dim
        self.b, self.w, _ = bert_embeds.shape
        self.e = etypes_.shape[1]

        # non-entity
        etypes_[etypes_ == -1] = self.sizes['etype_size']

        # type embeddings
        etype_embeds = self.type_embed(etypes_)  # (batch_size, entity_dim, type_dim)

        return etype_embeds

    def _create_pair_representation(self, etok_embeds, etype_embeds):
        """Create entity pair embeddings: Represent a sentence as a matrix of shape(B, E, E, dim)"""

        # concat: entities token and type embeddings
        pair_embeds = torch.cat((etok_embeds, etype_embeds), dim=2)

        # save for event layer
        type2_embeds = pair_embeds.clone()

        return pair_embeds, type2_embeds

    def _generate_l2r_pairs(self, pair_embeds, s_embeds, indices, rgtruth):
        """Generate left-to-right pair candidates embeddings"""

        # pair embeddings
        l2r_embeds = torch.cat(
            (pair_embeds[(indices[0], indices[1])], pair_embeds[(indices[0], indices[2])], s_embeds[indices[0]]),
            dim=-1)

        # pair labels
        l2r_truth = []
        for b, l, r in zip(indices[0], indices[1], indices[2]):
            l2r_truth.append(rgtruth[b.item()].get((l.item(), r.item()), -1))
        l2r_truth = np.asarray(l2r_truth)

        return l2r_embeds, l2r_truth

    def _generate_r2l_pairs(self, pair_embeds, s_embeds, indices, rgtruth):
        """Generate right-to-left pair candidates embeddings"""

        # pair embeddings
        r2l_embeds = torch.cat(
            (pair_embeds[(indices[0], indices[2])], pair_embeds[(indices[0], indices[1])], s_embeds[indices[0]]),
            dim=-1)

        # pair labels
        r2l_truth = []
        for b, r, l in zip(indices[0], indices[2], indices[1]):
            r2l_truth.append(rgtruth[b.item()].get((r.item(), l.item()), -1))
        r2l_truth = np.asarray(r2l_truth)

        return r2l_embeds, r2l_truth

    def _transpose_gold_indices(self, g_indices_):
        """Extract gold pairs indices"""
        # gold indices: batch, left, right
        gids_b = []
        gids_l = []
        gids_r = []
        for b_idx, l2r_batch in enumerate(g_indices_):
            if l2r_batch:
                gids_b.extend([b_idx] * len(l2r_batch[0]))
                gids_l.extend(l2r_batch[0])
                gids_r.extend(l2r_batch[1])
        g_indices = np.asarray([gids_b, gids_l, gids_r])
        return g_indices

    def predict(self, pair_embeds, g_indices_, p_indices, rgtruth_, sent_embeds):
        """Classify relations."""

        # 1-dropout
        if self.training:
            if self.params['dropout'] > 0:
                pair_embeds = f.dropout(pair_embeds, p=self.params['dropout'])

        # 2-transpose gold pairs indices
        g_indices = self._transpose_gold_indices(g_indices_)

        # 3-create left-to-right pairs
        # 3.1-training mode
        if not self.params['predict']:

            # i-gold ner
            if self.training and self.params['use_gold_ner']:
                use_gold = True

            # ii-train relation only: use gold ner
            elif not self.training and self.params['skip_ner'] and self.params['rel_epoch'] >= (
                    self.params['epoch'] - 1) and self.params['use_gold_ner']:
                use_gold = True

            # iii-train event only: use gold rel
            elif not self.training and self.params['skip_ner'] and self.params['skip_rel'] and self.params[
                'use_gold_rel']:
                use_gold = True

            # iv-
            else:
                use_gold = False

        # 3.2-predict mode
        else:

            # gold or pipeline
            if self.params['gold_eval'] or self.params['pipelines']:
                if self.params['pipelines'] and self.params['pipe_flag'] != 2:
                    use_gold = False
                else:
                    use_gold = True

            # joint
            else:
                use_gold = False

        # 3.3-get pair candidates embeddings and labels: from gold or predicted indices
        if use_gold:
            l2r_embeds, l2r_truth = self._generate_l2r_pairs(pair_embeds, sent_embeds, g_indices, rgtruth_)
        else:
            l2r_embeds, l2r_truth = self._generate_l2r_pairs(pair_embeds, sent_embeds, p_indices, rgtruth_)

        # 4-for non-relation label
        if not self.params['predict']:
            if np.ndim(l2r_truth) > 0:
                l2r_truth[l2r_truth == -1] = self.params['mappings']['rel_map']['1:Other:2']
            else:
                if l2r_truth == -1:
                    l2r_truth = self.params['mappings']['rel_map']['1:Other:2']
                l2r_truth = np.array([l2r_truth])

        # 5-NN on left-to-right pairs
        rel_l2r_embeds = gelu(self.hidden_layer1(l2r_embeds))
        rel_l2r_embeds = gelu(self.hidden_layer2(rel_l2r_embeds))
        l2r_preds = self.l_class(rel_l2r_embeds)  # (B*r, N)

        # 6-check dim
        if not self.params['predict']:
            assert (l2r_preds.shape[0] == l2r_truth.shape[0]), \
                "mismatch in ground-truth & prediction shapes left-to-right"

        # 7-both directions
        if self.params['direction'] != 'l2r':

            # training mode
            if not self.params['predict']:

                # i-gold ner
                if self.training and self.params['use_gold_ner']:
                    use_gold = True

                # ii-train rel only
                elif not self.training and self.params['skip_ner'] and self.params['rel_epoch'] >= (
                        self.params['epoch'] - 1) and self.params['use_gold_ner']:
                    use_gold = True

                # iii-train ev only
                elif not self.training and self.params['skip_ner'] and self.params['skip_rel'] and self.params[
                    'use_gold_rel']:
                    use_gold = True

                # iv
                else:
                    use_gold = False

            # predict mode
            else:

                # gold or pipeline
                if self.params['gold_eval'] or self.params['pipelines']:
                    if self.params['pipelines'] and self.params['pipe_flag'] != 2:
                        use_gold = False
                    else:
                        use_gold = True
                else:
                    use_gold = False

            # pair candidates embeddings and labels
            if use_gold:
                r2l_embeds, r2l_truth = self._generate_r2l_pairs(pair_embeds, sent_embeds, g_indices, rgtruth_)
            else:
                r2l_embeds, r2l_truth = self._generate_r2l_pairs(pair_embeds, sent_embeds, p_indices, rgtruth_)

            # non-relation type
            if not self.params['predict']:
                if np.ndim(r2l_truth) > 0:
                    r2l_truth[r2l_truth == -1] = self.params['mappings']['rel_map']['1:Other:2']
                else:
                    if r2l_truth == -1:
                        r2l_truth = self.params['mappings']['rel_map']['1:Other:2']
                    r2l_truth = np.array([r2l_truth])

            # NN for right-to-left pairs
            rel_r2l_embeds = gelu(self.hidden_layer1(r2l_embeds))
            rel_r2l_embeds = gelu(self.hidden_layer2(rel_r2l_embeds))
            r2l_preds = self.l_class(rel_r2l_embeds)

            # check dim
            if not self.params['predict']:
                assert (r2l_preds.shape[0] == r2l_truth.shape[0]), \
                    "mismatch in ground-truth & prediction shapes right-to-left"

            # both directions
            return rel_l2r_embeds, l2r_preds, l2r_truth, rel_r2l_embeds, r2l_preds, r2l_truth, pair_embeds, g_indices

        # only left-to-right
        else:
            return rel_l2r_embeds, l2r_preds, l2r_truth, pair_embeds, g_indices

    def forward(self, batch_input):

        # 1-entity type embeddings
        type_embeds = self._create_type_representation(batch_input['embeddings'], batch_input['ent_types'])

        # 2-create pair embeddings
        pair_embeds, type2_embeds = self._create_pair_representation(batch_input['ent_embeds'], type_embeds)
        pair_embeds = pair_embeds.view(self.b, self.e, pair_embeds.shape[2])

        # 3-predictions and labels
        predictions = self.predict(pair_embeds, batch_input['l2rs'], batch_input['pairs_idx'], batch_input['gtruths'],
                                   batch_input['sentence_embeds'])

        # 4-classify: use both directions
        acc_loss = 0
        if self.params['direction'] != 'lr2':

            # get output
            rel_l2r_embeds, l2r_preds, l2r_truth, rel_r2l_embeds, r2l_preds, r2l_truth, pair_embeds, g_indices = predictions

            # training
            if not self.params['predict']:
                if l2r_preds.shape[0] == 0:
                    return {'valid': False}

                # relation loss
                l2r_loss = f.cross_entropy(l2r_preds, torch.tensor(l2r_truth, device=self.device).long())
                r2l_loss = f.cross_entropy(r2l_preds, torch.tensor(r2l_truth, device=self.device).long())
                acc_loss = l2r_loss + r2l_loss

            # prediction and label
            r_preds = (f.softmax(l2r_preds, dim=1).data, f.softmax(r2l_preds, dim=1).data)
            r_gtruth = (l2r_truth, r2l_truth)

        # use only left-to-right direction
        else:

            # get output
            rel_l2r_embeds, l2r_preds, l2r_truth, pair_embeds, g_indices = predictions

            # training
            if not self.params['predict']:
                # relation loss
                acc_loss = f.cross_entropy(l2r_preds, torch.tensor(l2r_truth, device=self.device).long())

            # prediction and label
            r_preds = f.softmax(l2r_preds, dim=1).data
            r_gtruth = l2r_truth.data

        # get predicted type and scores
        new_rpreds, new_rgtruth, no_matched_rels, true_pos, false_pos, false_neg = calc_stats(r_preds, r_gtruth,
                                                                                              self.params)

        return {'valid': True, 'true_pos': true_pos, 'false_pos': false_pos, 'false_neg': false_neg,
                'preds': new_rpreds, 'enttoks_type_embeds': type2_embeds,
                'truth': new_rgtruth, 'no_matched_rel': no_matched_rels,
                'l2r': g_indices, 'pairs_idx': batch_input['pairs_idx'], 'rel_embeds': rel_l2r_embeds,
                'pair4class': pair_embeds, 'loss': acc_loss}
