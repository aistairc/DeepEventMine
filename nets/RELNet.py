import torch
import torch.nn.functional as f
from torch import nn

from eval.evalRE import calc_stats

import math


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(math.pi / 2) * (x + 0.044715 * x ** 3)))


class RELModel(nn.Module):

    def __init__(self, params, sizes):
        super(RELModel, self).__init__()

        self.type_embed = nn.Embedding(num_embeddings=sizes['etype_size'] + 1,
                                       embedding_dim=params['etype_dim'],
                                       padding_idx=sizes['etype_size'])

        ent_dim = params['bert_dim'] * 3 + params['etype_dim']  
        

        self.hidden_layer1 = nn.Linear(in_features=2 * ent_dim + params['bert_dim'],
                                       out_features=params['hidden_dim'], bias=False)

        self.hidden_layer2 = nn.Linear(in_features=params['hidden_dim'],
                                       out_features=params['rel_reduced_size'], bias=False)

        self.l_class = nn.Linear(in_features=params['rel_reduced_size'],
                                 out_features=sizes['rel_size'])

        self.device = params['device']
        self.params = params
        self.sizes = sizes

    def embedding_layer(self, bert_out, ents_etype_):

        self.b, self.w, _ = bert_out.shape
        self.e = ents_etype_.shape[1]

        ents_etype_[ents_etype_ == -1] = self.sizes['etype_size']
        type_embeds = self.type_embed(ents_etype_)  # (B, E, 10).

        return type_embeds

    def pair_representation(self, ent_embeds, tr_ids, type_embeds):

        pairs4class = torch.cat((ent_embeds, type_embeds), dim=2)

        enttoks_type_embeds = pairs4class.clone()

        return pairs4class, enttoks_type_embeds


    def get_pairs(self, pairs4class, pair_context, pairs_idx, direction, use_gold, use_context):
        indices = pairs_idx

        if direction == 'lr':
            if use_context:
                return torch.cat(
                    (pairs4class[(indices[0], indices[1])], pairs4class[(indices[0], indices[2])], pair_context),
                    dim=-1)
            else:
                return torch.cat((pairs4class[(indices[0], indices[1])], pairs4class[(indices[0], indices[2])]), dim=-1)
        else:
            if use_context:
                return torch.cat(
                    (pairs4class[(indices[0], indices[2])], pairs4class[(indices[0], indices[1])], pair_context),
                    dim=-1)
            else:
                return torch.cat((pairs4class[(indices[0], indices[2])], pairs4class[(indices[0], indices[1])]), dim=-1)

    def classification(self, pairs4class, pairs_idx_, sent_embeds):

    
        if self.params['predict']:
            
            pair_context = sent_embeds[pairs_idx_[0]]
               
            l2r_pairs = self.get_pairs(pairs4class, pair_context, pairs_idx_, 'lr', False,
                                           self.params['use_context'])

        l2r_pairs = gelu(self.hidden_layer1(l2r_pairs))
        l2r_pairs = gelu(self.hidden_layer2(l2r_pairs))

        pairs_preds_l2r = self.l_class(l2r_pairs)  # (B*r, N)


        if self.params['direction'] != 'l2r':
            
            if self.params['predict']:
                pair_context = sent_embeds[pairs_idx_[0]]
                r2l_pairs = self.get_pairs(pairs4class, pair_context, pairs_idx_, 'rl', False,
                                               self.params['use_context'])
               

            r2l_pairs = gelu(self.hidden_layer1(r2l_pairs))
            r2l_pairs = gelu(self.hidden_layer2(r2l_pairs))

            pairs_preds_r2l = self.l_class(r2l_pairs)

           

            return pairs_preds_l2r, pairs_preds_r2l, l2r_pairs, r2l_pairs, pairs4class, pairs_idx_
        else:
            return pairs_preds_l2r, pairs4class, pairs_idx_

    def calculate(self, batch_input):
        type_embeds = self.embedding_layer(batch_input['embeddings'], batch_input['ent_types'])

        sent_embeds = batch_input['sentence_embeds']

        pairs4class, enttoks_type_embeds = self.pair_representation(
            ent_embeds=batch_input['ent_embeds'], tr_ids=batch_input['tr_ids'],
            type_embeds=type_embeds)

        pairs4class = pairs4class.view(self.b, self.e, pairs4class.shape[2])

        forw_comp_res = self.classification(pairs4class=pairs4class,
                                            pairs_idx_=batch_input['pairs_idx'],
                                            sent_embeds=sent_embeds)

        return forw_comp_res, enttoks_type_embeds

    def forward(self, batch_input):
        if len(batch_input['pairs_idx']) > 0:
            fcomp_res, enttoks_type_embeds = self.calculate(batch_input)

            if self.params['direction'] != 'lr2':
                preds_l2r, preds_r2l, l2r_pairs, r2l_pairs, pair4class, pairs_idx = fcomp_res
                preds = (f.softmax(preds_l2r, dim=1).data, f.softmax(preds_r2l, dim=1).data)
            else:
                preds_l2r, l2r_pairs, pair4class, pairs_idx, positive_indices = fcomp_res
                preds = f.softmax(preds_l2r, dim=1).data

            new_preds = calc_stats(preds,self.params)

            return {'next': True,
                    'preds': new_preds, 'enttoks_type_embeds': enttoks_type_embeds,
                    'pairs_idx': pairs_idx, 'rel_embeds': l2r_pairs,
                    'pair4class': pair4class}

        else:
            return {'next': False}
