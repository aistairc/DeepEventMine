""" Build the EVENT prediction network.

"""
import numpy as np
import collections

import torch
from torch import nn
import torch.nn.functional as F

cpu_device = torch.device("cpu")

# use gelu instead of relu activation function
import math


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(math.pi / 2) * (x + 0.044715 * x ** 3)))


from nets.EVGen import EV_Generator


class EVModel(nn.Module):
    """CLASS FOR EVENT LAYERS."""

    def __init__(self, params, sizes):
        super(EVModel, self).__init__()

        # parameters
        self.params = params

        # dimensions
        if params['ner_reduce'] == False:
            ent_dim = params['bert_dim'] * 3 + params['etype_dim']  # no reduce
        else:
            ent_dim = params['ner_reduced_size'] + params['etype_dim']
        rel_dim = params['rel_reduced_size'] + params['rtype_dim'] + ent_dim
        ev_dim = ent_dim + params['role_dim']

        # to create event candidates
        self.ev_struct_generator = EV_Generator(params)

        # relation type embeddings
        self.rtype_layer = nn.Embedding(num_embeddings=sizes['rel_size'] + 1, embedding_dim=params['rtype_dim'])

        # IN argument embeddings: argument IN structure
        self.in_arg_layer = nn.Linear(in_features=rel_dim, out_features=params['role_dim'], bias=False)

        # OUT argument embeddings: argument NOT IN structure
        self.out_arg_layer = nn.Linear(in_features=rel_dim, out_features=params['role_dim'], bias=False)

        # for event classification
        self.hidden_layer1 = nn.Linear(in_features=ev_dim, out_features=params['hidden_dim'])
        self.hidden_layer2 = nn.Linear(in_features=params['hidden_dim'], out_features=params['ev_reduced_size'])
        self.l_class = nn.Linear(in_features=params['ev_reduced_size'], out_features=1)

        # reduce event embeds to replace entity
        self.ev2ent_reduce = nn.Linear(in_features=params['ev_reduced_size'], out_features=ent_dim)

        # predict modality
        self.modality_layer = nn.Linear(in_features=params['ev_reduced_size'], out_features=sizes['ev_size'])

        # others
        self.device = params['device']

    def get_rel_input(self, rel_preds):
        """Read relation input."""

        l2r = rel_preds['pairs_idx']
        rpreds_ = rel_preds['preds'].data

        # mapping relation type for 'OTHER' type to -1
        rpred_types = self.params['mappings']['rel2rtype_map'][rpreds_]

        # extract only relation type != 'OTHER' (valid relations)
        rpred_ids = (rpreds_ != self.params['voc_sizes']['rel_size'] - 1).nonzero().transpose(0, 1)[0]
        rpred_ids = rpred_ids.to(cpu_device)  # list: contain indices of the valid relations

        return l2r, rpred_types, rpred_ids

    def rtype_embedding_layer(self, rtype_):
        """Relation type embeddings."""

        # replace the -1 relation type by SPECIAL TYPE (rel-size) so that it is NO RELATION TYPE
        if np.ndim(rtype_) > 0:
            rtype_[rtype_ == -1] = self.params['voc_sizes']['rel_size']

            # relation type embedding
            rtype_embeds = self.rtype_layer(torch.tensor(rtype_, dtype=torch.long, device=self.device))  # (B, E, 10)
            has_no_rel = False
            for xx, rtypeid in enumerate(rtype_):
                if rtypeid == self.params['voc_sizes']['rel_size']:
                    # get an index for NO RELATION TYPE, using later for event with no-argument
                    no_rel_type_embed = rtype_embeds[xx]

                    has_no_rel = True
                    break
            if not has_no_rel:
                no_rel_type_embed = rtype_embeds[0]

        else:
            rtype_embeds = torch.zeros(self.params['rtype_dim'], dtype=torch.float32, device=self.device)
            no_rel_type_embed = torch.zeros(self.params['rtype_dim'], dtype=torch.float32, device=self.device)

        return rtype_embeds, no_rel_type_embed

    def get_arg_embeds(self, ent_embeds, rel_embeds, rtype_embeds, ev_arg_ids4nn):
        """Argument embeddings for each trigger.
            - Each trigger has a two-element tuple of
                1. trigger embedding
                2. a list of argument embedding: (relation emb, relation type emb, argument entity emb)
        """

        arg_embed_triggers = collections.OrderedDict()

        for trid, arg_data in ev_arg_ids4nn.items():
            tr_embeds = ent_embeds[trid]

            # no-argument
            if len(arg_data) == 1:
                arg_embed_triggers[trid] = [tr_embeds]

            # has argument:
            else:
                rids = arg_data[0]
                a2ids = arg_data[1]
                a2ids_ = np.vstack(a2ids).transpose()
                r_embeds = rel_embeds[rids]
                a2_embeds = ent_embeds[(a2ids_[0], a2ids_[1])]
                rt_embeds = rtype_embeds[rids]

                args_embeds = torch.cat([r_embeds, rt_embeds, a2_embeds],
                                        dim=-1)  # [number of arguments, rdim+rtypedim+edim]

                # store in a map to use later
                arg_embed_triggers[trid] = [tr_embeds, args_embeds]

        return arg_embed_triggers

    def event_representation(self, arg_embed_triggers, ev_cand_ids4nn, no_rel_type_embed):
        """Create event representation."""

        # get indices
        trids_ = ev_cand_ids4nn['trids_']
        io_ids_ = ev_cand_ids4nn['io_ids_']
        ev_structs_ = ev_cand_ids4nn['ev_structs_']

        # store event embeds in a list, return an array later
        ev_embeds_ = []

        # create embedding for each candidate indices
        for xx, trid in enumerate(trids_):

            # trigger embed
            tr_embed = arg_embed_triggers[trid][0]

            # store reduced argument embeds in a list
            args_embeds_list = []

            # get ev_struct
            ev_struct = ev_structs_[xx]

            # no-argument
            if len(ev_struct[1]) == 0:

                # since there is no argument, rel_embed is set as zeros
                no_rel_emb = torch.zeros((self.params['rel_reduced_size']), dtype=no_rel_type_embed.dtype,
                                         device=self.device)

                # argument emb is itself: trigger embed
                # then concatenate
                arg_embed = torch.cat([no_rel_emb, no_rel_type_embed, tr_embed])

                # put to IN ARGUMENT layer
                reduced_arg_embed = self.in_arg_layer(arg_embed)
                args_embeds_list.append(reduced_arg_embed)

                # check whether this trigger has other arguments, then set as OUT
                if len(arg_embed_triggers[trid]) > 1:

                    # argument embed
                    args_embeds = arg_embed_triggers[trid][1]

                    # calculate argument embedding
                    for xx, arg_embed in enumerate(args_embeds):
                        # OUT argument via OUT-ARG LAYER
                        reduced_arg_embed = self.out_arg_layer(arg_embed)

                        # store
                        args_embeds_list.append(reduced_arg_embed)

            # has argument
            else:

                # argument embed
                args_embeds = arg_embed_triggers[trid][1]

                # check IN/OUT
                io_ids = io_ids_[xx]

                # calculate argument embedding
                for ioid, arg_embed in enumerate(args_embeds):

                    # IN argument via IN-ARG LAYER
                    if ioid in io_ids:
                        reduced_arg_embed = self.in_arg_layer(arg_embed)

                    # OUT argument via OUT-ARG LAYER
                    else:
                        reduced_arg_embed = self.out_arg_layer(arg_embed)

                    # store
                    args_embeds_list.append(reduced_arg_embed)

            # calculate argument embed: by sum up all arguments or average, etc
            args_embed = torch.sum(torch.stack(args_embeds_list, dim=0), dim=0)

            # event embed: concatenate trigger embed and argument embed
            ev_embeds_.append(torch.cat([tr_embed, args_embed], dim=-1))

        # return tensor [number of event, dim]
        ev_embeds = torch.stack(ev_embeds_, dim=0)

        # dropout
        if self.training:
            if self.params['dropout'] > 0:
                ev_embeds = F.dropout(ev_embeds, p=self.params['dropout'])

        return ev_embeds

    def get_nest_arg_embeds(self, ent_embeds, rel_embeds, rtype_embeds, ev_arg_ids4nn, all_ev_embeds):
        """Argument embeddings for each trigger.
            - Each trigger has a two-element tuple of
                1. trigger embedding
                2. a list of argument embedding: (relation emb, relation type emb, argument entity emb)
        """

        # store a list of embeds for each trigger: trigger embeds, relation embeds, relation type embeds
        arg_embed_triggers = collections.OrderedDict()

        for trid, arg_data in ev_arg_ids4nn.items():
            tr_embeds = ent_embeds[trid]

            rids = arg_data[0]
            a2ids = arg_data[1]
            a2ids_ = np.vstack(a2ids).transpose()
            r_embeds = rel_embeds[rids]
            a2_embeds = ent_embeds[(a2ids_[0], a2ids_[1])]
            rt_embeds = rtype_embeds[rids]

            # replace event embeds for event arguments
            ev_argids_ = arg_data[2]

            # store event argument embeds by (argument id, event id)
            ev_arg_embeds_list = [[] for argid in range(len(rids))]

            for argid, ev_argids in enumerate(ev_argids_):

                # event argument
                if len(ev_argids) > 0:

                    # store event argument embeds with key is event id
                    ev_arg_embeds = collections.OrderedDict()

                    for pid in ev_argids:
                        # pid: (level, positive_event_id)

                        # store event argument embed by argument id and event id
                        ev_arg_emb = all_ev_embeds[pid[0]][pid[1]]
                        ev_rel_emb = r_embeds[argid]
                        ev_rtype_emb = rt_embeds[argid]

                        # concatenate with relation and relation type embeds
                        ev_arg_embeds[pid] = torch.cat([ev_rel_emb, ev_rtype_emb, ev_arg_emb], dim=-1)

                    # add to the list
                    ev_arg_embeds_list[argid] = ev_arg_embeds

            # concatenate for argument embeddings: [rel_embed, rel_type_embed, entity_embed]
            args_embeds = torch.cat([r_embeds, rt_embeds, a2_embeds],
                                    dim=-1)  # [number of arguments, rdim+rtypedim+edim]

            # store in a map to use later
            arg_embed_triggers[trid] = [tr_embeds, args_embeds, ev_arg_embeds_list]

        return arg_embed_triggers

    def event_nest_representation(self, arg_embed_triggers, ev_cand_ids4nn, no_rel_type_embed):
        """Create event representation."""

        # get indices
        trids_ = ev_cand_ids4nn['trids_']
        io_ids_ = ev_cand_ids4nn['io_ids_']
        ev_structs_ = ev_cand_ids4nn['ev_structs_']
        pos_ev_ids_ = ev_cand_ids4nn['pos_ev_ids_']

        # store event embeds in a list, return an array later
        ev_embeds_ = []

        # create embedding for each candidate indices
        for xx, trid in enumerate(trids_):

            # trigger embed
            tr_embed = arg_embed_triggers[trid][0]

            # store reduced argument embeds in a list
            args_embeds_list = []

            # get ev_struct
            ev_struct = ev_structs_[xx]

            # no-argument
            if len(ev_struct[1]) == 0:

                # since there is no argument, rel_embed is set as zeros
                no_rel_emb = torch.zeros((self.params['rel_reduced_size']), dtype=no_rel_type_embed.dtype,
                                         device=self.device)

                # argument emb is itself: trigger embed
                # then concatenate
                arg_embed = torch.cat([no_rel_emb, no_rel_type_embed, tr_embed])

                # put to IN ARGUMENT layer
                reduced_arg_embed = self.in_arg_layer(arg_embed)
                args_embeds_list.append(reduced_arg_embed)

                # check whether this trigger has other arguments, then set as OUT
                if len(arg_embed_triggers[trid]) > 1:

                    # argument embed
                    args_embeds = arg_embed_triggers[trid][1]

                    # calculate argument embedding
                    for xx, arg_embed in enumerate(args_embeds):
                        # OUT argument via OUT-ARG LAYER
                        reduced_arg_embed = self.out_arg_layer(arg_embed)

                        # store
                        args_embeds_list.append(reduced_arg_embed)

            # has argument
            else:

                # argument embed
                args_embeds = arg_embed_triggers[trid][1]

                # event argument embeds
                ev_args_embeds = arg_embed_triggers[trid][2]

                # check IN/OUT
                io_ids = io_ids_[xx]

                # positive event ids
                pos_ids = pos_ev_ids_[xx]

                # calculate argument embedding
                for ioid, arg_embed in enumerate(args_embeds):

                    # IN argument via IN-ARG LAYER
                    if ioid in io_ids:

                        for xx2, inid in enumerate(io_ids):
                            if inid == ioid:
                                pid = pos_ids[xx2]

                                # entity argument
                                if pid == (-1, -1):
                                    reduced_arg_embed = self.in_arg_layer(arg_embed)
                                    args_embeds_list.append(reduced_arg_embed)

                                # event argument
                                else:
                                    ev_arg_embed = ev_args_embeds[ioid][pid]
                                    reduced_arg_embed = self.in_arg_layer(ev_arg_embed)
                                    args_embeds_list.append(reduced_arg_embed)

                    # OUT argument via OUT-ARG LAYER
                    else:

                        # entity argument
                        if len(ev_args_embeds[ioid]) == 0:
                            reduced_arg_embed = self.out_arg_layer(arg_embed)
                            args_embeds_list.append(reduced_arg_embed)

                        # event arguments: run with all event arguments for this trigger
                        else:
                            for pid in ev_args_embeds[ioid]:
                                ev_arg_embed = ev_args_embeds[ioid][pid]
                                reduced_arg_embed = self.out_arg_layer(ev_arg_embed)
                                args_embeds_list.append(reduced_arg_embed)

                    # store
                    # args_embeds_list.append(reduced_arg_embed)

            # calculate argument embed: by sum up all arguments or average, etc
            args_embed = torch.sum(torch.stack(args_embeds_list, dim=0), dim=0)

            # event embed: concatenate trigger embed and argument embed
            ev_embeds_.append(torch.cat([tr_embed, args_embed], dim=-1))

        # return tensor [number of event, dim]
        ev_embeds = torch.stack(ev_embeds_, dim=0)

        # dropout
        if self.training:
            if self.params['dropout'] > 0:
                ev_embeds = F.dropout(ev_embeds, p=self.params['dropout'])

        return ev_embeds

    def predict(self, event_embeds):
        """Prediction."""

        threshold = self.params['ev_threshold']

        event4class = gelu(self.hidden_layer1(event_embeds))
        event4class = gelu(self.hidden_layer2(event4class))
        prediction = self.l_class(event4class)

        # modality
        ev_lbls = prediction.clone()
        norm_preds = torch.sigmoid(ev_lbls)
        norm_preds = norm_preds.detach().cpu().numpy()

        threshols = threshold * np.ones(ev_lbls.shape, dtype=np.float32)
        pred_mask = np.asarray(np.greater(norm_preds.data, threshols), dtype=np.int32)
        positive_idx = np.where(pred_mask.ravel() != 0)

        positive_ev_embs = event4class[positive_idx]

        prediction = prediction.flatten()

        # return prediction, modality_pred, positive_idx, positive_ev # revise
        return event4class, prediction, positive_idx, positive_ev_embs

    def predict_modality(self, positive_ev_embs, positive_ev_idx, mod_labels_):
        """Predict modality, return modality predictions."""

        # get labels
        mod_labels = np.vstack(mod_labels_).ravel()
        positive_labels = mod_labels.copy()
        positive_labels[positive_labels > 0] = 1
        possitive_lbl = torch.tensor((mod_labels[positive_ev_idx] - 1), dtype=torch.long,
                                     device=self.device)

        # prediction
        if possitive_lbl[possitive_lbl >= 0].shape[0] > 0:

            # prediction
            modality_preds_ = self.modality_layer(positive_ev_embs)
            modality_preds = modality_preds_[possitive_lbl >= 0]
            modality_pred = modality_preds_.detach().cpu().numpy()

            modality_pred = F.softmax(torch.tensor(modality_pred), dim=-1).data
            mod_preds = modality_pred.argmax(dim=-1)


        else:
            mod_preds = []

        return mod_preds

    def create_output(self, all_ev_preds):
        """Create output for writing events."""

        all_ev_output = []

        for level, ev_preds in enumerate(all_ev_preds):
            # store output in a list
            ev_output = []

            # get indices
            ev_cands_ids = ev_preds[0]
            ev_args_ids = ev_preds[1]
            positive_idx = ev_preds[2]
            modality_preds = ev_preds[3]

            # input indices
            trids_ = ev_cands_ids['trids_']
            io_ids_ = ev_cands_ids['io_ids_']
            ev_structs_ = ev_cands_ids['ev_structs_']

            # positive ev ids
            if level > 0:
                pos_ev_ids_ = ev_cands_ids['pos_ev_ids_']
            else:
                pos_ev_ids_ = []

            for xx1, pid in enumerate(positive_idx):

                # trigger id
                trid = trids_[pid]

                # structure
                ev_struct = ev_structs_[pid]

                # argument: relation id and entity id
                arg_data = ev_args_ids[trid]

                # check argument
                if len(arg_data) > 1:

                    # store argument list
                    # flat
                    if level == 0:
                        a2ids = [arg_data[1][inid] for inid in io_ids_[pid]]

                    # nested
                    else:
                        a2ids = []
                        for (inid, posid) in zip(io_ids_[pid], pos_ev_ids_[pid]):

                            # it is an entity argument
                            if posid == (-1, -1):
                                a2ids.append(arg_data[1][inid][0])

                            # or index of event
                            else:
                                a2ids.append((-1, -1, posid))  # add -1 to check later

                # no-argument: return empty list
                else:
                    a2ids = []

                # check modality
                if len(modality_preds) > 0:
                    mod_pred = modality_preds[xx1].item()
                else:
                    mod_pred = -1

                # store output
                ev_output.append([trid, ev_struct, a2ids, mod_pred])

            # store the output of this level
            all_ev_output.append(ev_output)

        return all_ev_output

    def calculate(self, ent_embeds, rel_embeds, rpred_types, ev_ids4nn):
        """
        Create embeddings, prediction.

        :param ent_embeds: [batch x a1id x embeds]
        :param rel_embeds: [rids x embeds]
        :param rpred_types: [rids] # predicted relation types
        :param ev_ids4nn: generated event canddiate indices
            + ev_cand_ids4nn: event candidates indices
                + trids_: list of trigger ids corresponding to the list of events
                + ev_labels_: list of corresponding labels
                + mod_labels_: modality labels
                io_ids_: in/out indices
            + ev_arg_ids4nn: event argument indices (a map of argument indices for each trigger)
                + list of rids
                + list of argument ids

        :return: prediction
        """

        # store output
        all_preds_output = []

        enable_nested_ev = True
        enable_modality = True

        # store all predictions for flat and nested, maximum as 3 nested levels
        # positive ids: the current predicted indices; tr_ids: trigger indices of the candidate list
        all_positive_ids = -1 * np.ones((self.params['max_ev_level'] + 1), dtype=np.object)
        all_positive_tr_ids = -1 * np.ones((self.params['max_ev_level'] + 1), dtype=np.object)

        # store predicted events embeds
        all_positive_ev_embs = []

        # for flat events
        # 1-candidate input
        ev_flat_cand_ids4nn = ev_ids4nn['ev_cand_ids4nn']
        ev_flat_arg_ids4nn = ev_ids4nn['ev_arg_ids4nn']

        # 2-relation type embeddings
        rtype_embeds, no_rel_type_embed = self.rtype_embedding_layer(rpred_types)

        # 3-argument embeddings for each trigger
        arg_embed_triggers = self.get_arg_embeds(ent_embeds, rel_embeds, rtype_embeds, ev_flat_arg_ids4nn)

        # 4-create event representation
        ev_embeds = self.event_representation(arg_embed_triggers, ev_flat_cand_ids4nn, no_rel_type_embed)

        # 5-prediction
        # positive_ev_embs: embedding of predicted events: using for the next nested level
        event4class, prediction, positive_idx, positive_ev_embs = self.predict(ev_embeds)

        empty_pred = True

        # for modality
        if enable_modality:
            mod_preds = self.predict_modality(positive_ev_embs, positive_idx,
                                              ev_flat_cand_ids4nn['mod_labels_'])
        else:
            mod_preds = []

        # init current nested level
        current_nested_level = 0
        current_tr_ids = ev_flat_cand_ids4nn['trids_']
        current_truth_ids = ev_flat_cand_ids4nn['truth_ids_']

        # store positive ids
        current_positive_ids = positive_idx[0]
        all_positive_ids[current_nested_level] = current_positive_ids
        ev_nest_cand_triggers = ev_ids4nn['ev_nest_cand_triggers']

        # for output
        all_preds_output.append([ev_flat_cand_ids4nn, ev_flat_arg_ids4nn, current_positive_ids, mod_preds])

        # loop until stop nested event prediction or no more events predicted, or in limited nested levels
        while enable_nested_ev and len(current_positive_ids) > 0 and current_nested_level < self.params['max_ev_level']:

            # update trigger indices and predicted positive indices
            # positive trigger indices
            current_positive_tr_ids = [current_tr_ids[pos_id] for pos_id in current_positive_ids]
            current_positive_truth_ids = [current_truth_ids[pos_id] for pos_id in current_positive_ids]
            all_positive_tr_ids[current_nested_level] = current_positive_tr_ids

            # reduce event embeds to replace entity
            reduced_ev_emb = self.ev2ent_reduce(positive_ev_embs)
            all_positive_ev_embs.append(reduced_ev_emb)

            # generate nested candidate indices
            ev_nest_ids4nn = self.ev_struct_generator._generate_nested_candidates(current_nested_level,
                                                                                  ev_nest_cand_triggers,
                                                                                  current_positive_tr_ids,
                                                                                  current_positive_truth_ids)

            # get candidate indices, updated by the previous level output
            ev_nest_cand_ids4nn = ev_nest_ids4nn['ev_nest_cand_ids']
            ev_nest_arg_ids4nn = ev_nest_ids4nn['ev_nest_arg_ids']
            ev_nest_cand_triggers = ev_nest_ids4nn['ev_nest_cand_triggers']
            current_tr_ids = ev_nest_cand_ids4nn['trids_']
            current_truth_ids = ev_nest_cand_ids4nn['truth_ids_']

            empty_pred = False

            # check non-empty
            if len(ev_nest_cand_ids4nn['trids_']) > 0:

                # 3-argument embeddings for each trigger
                arg_embed_triggers = self.get_nest_arg_embeds(ent_embeds, rel_embeds, rtype_embeds, ev_nest_arg_ids4nn,
                                                              all_positive_ev_embs)

                # event representation
                ev_embeds = self.event_nest_representation(arg_embed_triggers, ev_nest_cand_ids4nn, no_rel_type_embed)

                # check non-empty predictions

                # prediction
                event4class, prediction, positive_idx, positive_ev_embs = self.predict(ev_embeds)

                # for modality
                if enable_modality:
                    mod_preds = self.predict_modality(positive_ev_embs, positive_idx,
                                                      ev_nest_cand_ids4nn['mod_labels_'])
                else:
                    mod_preds = []

                # count nested level
                current_nested_level += 1
                current_positive_ids = positive_idx[0]

                # store positive ids
                all_positive_ids[current_nested_level] = current_positive_ids

                all_preds_output.append([ev_nest_cand_ids4nn, ev_nest_arg_ids4nn, current_positive_ids, mod_preds])

            # otherwise: stop loop
            else:
                enable_nested_ev = False

        # 7-create output for writing events
        pred_ev_output = self.create_output(all_preds_output)

        return pred_ev_output, empty_pred

    def forward(self, ner_preds, rel_preds):
        """Forward.
            Given entities and relations, event structures, return event prediction.
        """
        # check empty relation prediction
        if len(rel_preds['preds'].data) == 0:
            ev_preds = None
            empty_pred = True

        else:
            # 1-get input
            # entity
            etypes = ner_preds['ent_types']
            etypes = etypes.to(torch.device("cpu"))

            # entity and trigger embeddings [bert and type embeddings]
            ent_embeds = rel_preds['enttoks_type_embeds']

            # trigger
            tr_ids = (ner_preds['tr_ids'] == 1).nonzero().transpose(0, 1)
            tr_ids = list(zip(tr_ids[0], tr_ids[1]))

            # relation
            l2r, rpred_types, rpred_ids = self.get_rel_input(rel_preds)
            if np.ndim(rpred_types) > 0:
                rel_embeds = rel_preds['rel_embeds']
            else:
                rel_embeds = torch.zeros((1, self.params['rel_reduced_size']), dtype=torch.float32, device=self.device)

                # avoid scalar error
                rpred_types = np.array([rpred_types])

            # 2-generate event candidates
            ev_ids4nn = self.ev_struct_generator._generate(etypes, tr_ids, l2r, rpred_types, rpred_ids
                                                           )

            # 3-embeds, prediction
            # check empty
            if len(ev_ids4nn['ev_cand_ids4nn']['trids_']) > 0:
                ev_preds, empty_pred = self.calculate(ent_embeds, rel_embeds, rpred_types, ev_ids4nn)

            else:
                ev_preds = None
                empty_pred = True

        return ev_preds, empty_pred
