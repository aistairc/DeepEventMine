"""Prepare event data for training networks."""

import collections
import numpy as np
import logging

logger = logging.getLogger(__name__)


def create_ev_truth(idxs, readable_e, events, params):
    ev_num = 0
    ev_matched = 0
    max_ev_per_layer = 0

    truth_ev = -1 * np.ones((len(readable_e), 2), dtype=np.object)

    truth_ev0 = collections.defaultdict(list)
    type_debug = False

    for idTR in events:
        xxTR = idxs[idTR]

        ev_l_0 = 0
        ev_l_1 = 0

        for event in events[idTR]:
            event['modality'] = params['mappings']['modality_map'][event['modality']]
            ev_num += 1

            typeTR = event['trtype']

            struct = event['args_type']

            if typeTR in params['mappings']['flat_structs_map']:
                if struct in params['mappings']['flat_structs_map'][typeTR]:
                    ev_matched += 1
                    ev_l_0 += 1
                    ev_argtype = event['args_type']
                    args_ids = params['mappings']['flat_structs_map'][typeTR][ev_argtype]

                    rels = event['rel']
                    a2ids = []
                    for rel, reltype in rels.items():
                        a2id = idxs[rel[1]]
                        a2ids.append(a2id)

                    # self event

                    if event['is_zeroArg']:
                        a2ids.append(xxTR)

                    truth_ev0[(xxTR, 0)].append([args_ids, a2ids])

            if typeTR in params['mappings']['nested_structs_map']:
                if struct in params['mappings']['nested_structs_map'][typeTR]:
                    ev_matched += 1
                    ev_l_1 += 1
                    ev_argtype = event['args_type']
                    args_ids = params['mappings']['nested_structs_map'][typeTR][ev_argtype]

                    rels = event['rel']
                    a2ids = []
                    for rel, reltype in rels.items():
                        a2id = idxs[rel[1]]
                        a2ids.append(a2id)

                    truth_ev0[(xxTR, 1)].append([args_ids, a2ids])

        max_ev_per_layer = max(ev_l_0, ev_l_1, max_ev_per_layer)

    for trid, pairs in truth_ev0.items():
        truth_ev[trid[0]][trid[1]] = pairs

    no_event = False
    if len(truth_ev0) == 0:
        no_event = True

    ev_missed = ev_num - ev_matched

    return truth_ev, ev_num, ev_matched, ev_missed, no_event, type_debug, max_ev_per_layer


def event2network(sentence_data, fid, idxs, events_map, max_ev_per_layer, readable_e, params):
    # input
    events = sentence_data['trigger_ev']

    # create labels for events
    truth_ev, ev_num, ev_matched, ev_missed, no_event, type_debug, max_ev_per_layer_ = create_ev_truth(idxs,
                                                                                                       readable_e,
                                                                                                       events, params)

    # C2T add
    max_ev_per_layer = max(max_ev_per_layer_, max_ev_per_layer)

    # ev_num2 += ev_num
    # ev_matched2 += ev_matched
    # ev_missed2 += ev_missed

    # Add events to map:
    for _, events_list in enumerate(events.items()):
        for event in events_list[1]:
            if fid not in events_map:
                events_map[fid] = {event['id']: event}
            else:
                events_map[fid][event['id']] = event

    return events, truth_ev, max_ev_per_layer


def count_ev_truth(samples):
    """Count the number of created truth events."""

    # count total number of valid truth events
    total_count_valid_evs = 0

    # for each sentence
    for sample in samples:
        # get truth
        truth_ev = sample['truth_ev']

        # count the valid event truth
        valid_truth_ev = truth_ev[truth_ev != -1]
        count_valid_ev = sum([len(truth_list) for truth_list in valid_truth_ev])
        total_count_valid_evs += count_valid_ev

    print('Check created event truth')
    print('Valid truth events: ', total_count_valid_evs)

    return


def gen_nn_truth_nested_ev(fid, typeTR, struct, mapping_structs, event, span_terms, ev_idx, events_map, params,
                           self_event=False):
    try:
        if typeTR in mapping_structs:
            if struct in mapping_structs[typeTR]:
                ev_argtype = event['args_type']
                args_ids = mapping_structs[typeTR][ev_argtype]
                rels = event['rel']

                # store entity arguments and event arguments
                ent_args = []
                ev_args = []

                if len(event['nested_events']) > 0:
                    nested_evs = [events_map[fid][eid] if eid in events_map[fid] else -1 for eid in
                                  event['nested_events']]
                    nested_trIds = [ev['trid'] if ev != -1 else -1 for ev in nested_evs]
                for rel, reltype in rels.items():
                    argid = rel[1]

                    # is event argument
                    if len(event['nested_events']) > 0:

                        # is trigger
                        if argid in nested_trIds:
                            nested_ev = nested_evs[nested_trIds.index(argid)]
                            a2id = gen_nn_truth_nested_evs(fid, nested_ev, span_terms, events_map, params)
                            ev_args.append(a2id)

                        # is entity
                        else:
                            a2id = span_terms.term2id[argid]
                            ent_args.append(a2id)

                    # or flat
                    else:
                        a2id = span_terms.term2id[argid]
                        ent_args.append(a2id)

                # self event
                if self_event:
                    if event['is_zeroArg']:
                        ent_args.append(ev_idx)

                if len(ent_args) > 0:
                    ent_args = collections.Counter(ent_args)

                nested_ev_level = event['nested_ev_level']
                truth_out = (nested_ev_level, args_ids, ent_args, ev_args)
                return truth_out

    except (KeyError, ValueError) as err:
        logger.debug(err)
        return None


def gen_nn_truth_nested_evs(fid, nested_ev, span_terms, events_map, params):
    nested_idTR = nested_ev['trid']
    nested_ev_idx = span_terms.term2id[nested_idTR]
    typeTR = nested_ev['trtype']
    struct = nested_ev['args_type']
    nested_ev_present = gen_nn_truth_nested_ev(fid, typeTR, struct, params['mappings']['flat_structs_map'], nested_ev,
                                               span_terms, nested_ev_idx, events_map, params, self_event=True)
    if not nested_ev_present:
        nested_ev_present = gen_nn_truth_nested_ev(fid, typeTR, struct, params['mappings']['nested_structs_map'],
                                                   nested_ev,
                                                   span_terms, nested_ev_idx, events_map, params)

    return nested_ev_present


def gen_nn_truth_ev(fid, truth_ev_layer, typeTR, struct, mapping_structs, event, span_terms, ev_idx, events_map, params,
                    self_event=False):
    try:
        if typeTR in mapping_structs:
            if struct in mapping_structs[typeTR]:
                ev_argtype = event['args_type']
                struct_ids = mapping_structs[typeTR][ev_argtype]
                rels = event['rel']

                # store entity and event arguments
                ent_args_list = []
                ev_args_list = []

                if len(event['nested_events']) > 0:
                    nested_evs = [events_map[fid][eid] if eid in events_map[fid] else -1 for eid in
                                  event['nested_events']]
                    nested_trIds = [ev['trid'] if ev != -1 else -1 for ev in nested_evs]
                for rel, reltype in rels.items():
                    argid = rel[1]
                    if len(event['nested_events']) > 0:

                        # is trigger
                        if argid in nested_trIds:
                            nested_ev = nested_evs[nested_trIds.index(argid)]
                            a2id = gen_nn_truth_nested_evs(fid, nested_ev, span_terms, events_map, params)
                            ev_args_list.append(a2id)

                        # is entity
                        else:
                            a2id = span_terms.term2id[argid]
                            ent_args_list.append(a2id)

                    # is flat
                    else:
                        a2id = span_terms.term2id[argid]
                        ent_args_list.append(a2id)

                # self event
                if self_event:
                    if event['is_zeroArg']:
                        ent_args_list.append(ev_idx)

                mod_label = event['modality']
                if len(ent_args_list) > 0:
                    ent_args_list = collections.Counter(ent_args_list)
                truth_out = [(struct_ids, ent_args_list, ev_args_list), mod_label]
                truth_ev_layer.append(truth_out)

    except (KeyError, ValueError) as err:
        logger.debug(err)


def gen_nn_truth_evs(fid, span_terms, events, events_map, params):
    truth_ev = -1 * np.ones((len(events), params['max_ev_level'] + 1, params['max_ev_args'] + 1), dtype=np.object)
    ev_lbls = -1 * np.ones((len(events)), dtype=np.object)
    ev_idxs = {}

    truth_ev_dict = collections.defaultdict(list)
    ev_lbls_dict = collections.defaultdict(list)

    # store list of events for each trigger id
    ev_idxs_lst = []

    for idTR in events:
        if idTR in span_terms.term2id:

            # event trigger index
            ev_trid = span_terms.term2id[idTR]

            ev_idxs_lst.append(ev_trid)
            for i, event in enumerate(events[idTR]):
                mod_label = event['modality']
                typeTR = event['trtype']
                struct = event['args_type']

                # get the number of arguments, and nested level
                arg_num = event['args_num']
                nested_ev_level = event['nested_ev_level']

                # flat events
                gen_nn_truth_ev(fid, truth_ev_dict[(ev_trid, 0, arg_num)], typeTR, struct,
                                params['mappings']['flat_structs_map'], event,
                                span_terms, ev_trid, events_map, params, self_event=True)

                # nested events
                gen_nn_truth_ev(fid, truth_ev_dict[(ev_trid, nested_ev_level, arg_num)], typeTR, struct,
                                params['mappings']['nested_structs_map'], event,
                                span_terms, ev_trid, events_map, params)
                ev_lbls_dict[ev_trid].append(mod_label)

    for i, ev_trid in enumerate(ev_idxs_lst):
        ev_idxs[ev_trid] = i
        ev_lbls[i] = ev_lbls_dict[ev_trid]
        for level in range(params['max_ev_level'] + 1):
            for narg in range(params['max_ev_args'] + 1):
                try:
                    if len(truth_ev_dict[(ev_trid, level, narg)]) > 0:
                        truth_ev[i][level][narg] = truth_ev_dict[(ev_trid, level, narg)]
                    else:
                        truth_ev[i][level][narg] = -1
                except KeyError:
                    truth_ev[i][level][narg] = -1

    return truth_ev, ev_idxs, ev_lbls
