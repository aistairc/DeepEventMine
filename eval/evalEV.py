import collections
import os

from loguru import logger


def get_entity_attrs(e_idx, words, offset, span_indices, sub_to_words):
    e_span_indice = span_indices[e_idx]
    e_words = []
    e_offset = [-1, -1]
    curr_word_idx = -1
    for idx in range(e_span_indice[0], e_span_indice[1] + 1):
        if sub_to_words[idx] != curr_word_idx:
            e_words.append(words[sub_to_words[idx]])
            curr_word_idx = sub_to_words[idx]
        if idx == e_span_indice[0]:
            e_offset[0] = offset[sub_to_words[idx]][0]
        if idx == e_span_indice[1]:
            e_offset[1] = offset[sub_to_words[idx]][1]

    return ' '.join(e_words), (e_offset[0], e_offset[1])


# generate entity
def generate_entities(fids, all_e_preds, all_words, all_offsets, all_span_terms, all_span_indices, all_sub_to_words,
                      params):
    nn_tag2type_map = params['mappings']['nn_mapping']['tag2type_map']

    pred_ents_ = collections.defaultdict(list)

    for xi, (fids_, e_preds_, words_, offset_, span_indices_, sub_to_words_) in enumerate(
            zip(fids, all_e_preds, all_words, all_offsets, all_span_indices, all_sub_to_words)):

        for xb, (fid, e_preds, words, offset, span_indices, sub_to_words) in enumerate(
                zip(fids_, e_preds_, words_, offset_, span_indices_, sub_to_words_)):
            preds = collections.defaultdict(list)
            for xx, pred in enumerate(e_preds):
                if pred > 0:
                    e_term = all_span_terms[xi][xb].id2term[xx]
                    e_type_id = nn_tag2type_map[pred]
                    if 'pipeline_entity_org_map' in params:
                        if e_term in params['pipeline_entity_org_map'][fid]:
                            e_words, e_offset = params['pipeline_entity_org_map'][fid][e_term]
                        else:
                            print(e_term)
                            e_words, e_offset = get_entity_attrs(xx, words, offset, span_indices, sub_to_words)
                    else:
                        e_words, e_offset = get_entity_attrs(xx, words, offset, span_indices, sub_to_words)
                    preds[(xi, (xb, xx))] = [e_term, e_type_id, e_offset, e_words]
            pred_ents_[fid].append(preds)

    pred_ents = collections.OrderedDict()
    for fid in pred_ents_:
        pred_ents[fid] = collections.OrderedDict()
        for preds in pred_ents_[fid]:
            for trid, ent in preds.items():
                pred_ents[fid][trid] = ent

    return pred_ents


# generate event
def generate_events(fids, all_ev_preds, params):
    # store events in a map
    pred_evs = collections.defaultdict(list)

    for xi, (fids_, ev_preds_levels_) in enumerate(
            zip(fids, all_ev_preds)):

        # accumulated event numbers to count event id
        acc_evid = 0
        # ev_count = 0

        # store event ids
        evids_ = collections.OrderedDict()

        for level, ev_preds_ in enumerate(ev_preds_levels_):

            # process for each event level
            for xx1, ev_ in enumerate(ev_preds_):

                # store event data in a list
                ev_data = []

                # set event id
                ev_id = xx1 + acc_evid
                # if level == 0:
                # ev_id = xx1
                # ev_id = acc_evid + xx1
                # else:
                #     ev_id = level * len(ev_preds_levels_[level-1]) + xx1
                # ev_id = acc_evid + xx1

                ev_id_str = (str(xi) + '_' + str(ev_id))
                # ev_id_ = (xi, ev_id)

                # store evid for nested events
                evids_[(level, xx1)] = ev_id_str

                # add event id information
                ev_data.append(ev_id_str)

                # get trigger id, relation structure, entity ids
                trid = ev_[0]
                rel_struct_ = ev_[1]
                a2ids = ev_[2]

                # get document id, using for predicted event index to distinguish among mini-batches
                xb = trid[0]
                fid = fids_[int(xb)]

                # add trigger id
                ev_data.append((xi, (trid[0], trid[1])))

                # get relation structure
                # rel_struct_counter = rel_struct_[0]
                rel_struct_list = rel_struct_[1]

                # check no-argument
                # if len(a2ids) == 0:
                #     continue

                # has argument
                # if len(a2ids) > 0:
                if len(rel_struct_list) > 0:

                    # store args_data
                    args_data = []

                    # to check duplicate relation type
                    dup_rtypes = collections.OrderedDict()

                    for argid, a2id in enumerate(a2ids):

                        # print(argid, rel_struct_list, rel_struct_counter, trid, a2ids)

                        # get relation type id
                        rel_group = rel_struct_list[argid]  # (rtypeid, argtypeid)
                        rtypeid = rel_group[0]

                        # TODO: process duplicated relation types
                        # process duplicate relation type: +number to rtype: Theme, Theme2, Theme3 etc
                        if rtypeid not in dup_rtypes:
                            dup_rtypes[rtypeid] = 1
                        else:
                            dup_rtypes[rtypeid] += 1

                        # create id for a2
                        # check whether this is entity or event argument

                        # event argument
                        if level > 0 and len(a2id) > 2:
                            # evlevel = a2id[1]
                            # evxx1 = a2id[2]
                            evlevel_id = a2id[2]

                            # look up in the event ids list
                            # added_evid = evids_[(evlevel, evxx1)]
                            added_evid = evids_[evlevel_id]
                            a2bid = (added_evid, -1, -1)  # add -1 to mark the event argument

                        # entity argument
                        else:
                            a2bid = (xi, a2id)

                        # add argument (relation type, entity id, rtype-dup) to output
                        args_data.append((rtypeid, a2bid, dup_rtypes[rtypeid]))

                    # store ev_data
                    ev_data.append(args_data)

                # modality
                mod_pred = ev_[3]
                ev_data.append(mod_pred)

                # store this event
                pred_evs[fid].append(ev_data)

            # accumulate event number
            acc_evid += len(ev_preds_)

    return pred_evs


# generate event output
def generate_ev_output(pred_ents, pred_evs, params):
    # store output
    preds_output = collections.OrderedDict()

    # store event id to check nested event
    added_evid_list = []

    for fid, ents_ in pred_ents.items():

        # store output, events, triggers
        evs_ = pred_evs[fid]
        events_ = []
        triggers_ = []

        for ev_ in evs_:

            # event id and trigger id
            ev_id = ev_[0]
            trid = ev_[1]

            if len(ev_) > 3:
                args_data = ev_[2]

            # no-argument
            else:
                args_data = []

            # to check all argument entities exist
            valid_ev_flag = True

            if trid in ents_:

                # get trigger data: trigger id, type, offset, text
                trigger = ents_[trid]

                # store arguments (rtype, entity id)
                args_data_ = []

                # store argument entity ids
                arg2s = []

                # has argument
                if len(args_data) > 0:

                    for xx1, arg_ in enumerate(args_data):

                        # get relation type id
                        rtypeid = arg_[0]

                        # duplicated rtypeid
                        dup_rtype = arg_[2]

                        # get entity id
                        a2bid = arg_[1]

                        # get relation type
                        rtype = params['mappings']['rev_rtype_map'][rtypeid]

                        # add duplicated type:
                        if dup_rtype > 1:
                            rtype += str(dup_rtype)

                        # check if this is entity argument or event argument
                        # event argument
                        if len(a2bid) > 2:
                            # add to argument
                            pred_evid = a2bid[0]
                            if pred_evid in added_evid_list:
                                args_data_.append((rtype, pred_evid, -1))

                        # entity argument
                        else:

                            # if this entity has been predicted (exist)
                            if a2bid in ents_:

                                # get entity id and entity type
                                a2data = ents_[a2bid]
                                a2type = params['mappings']['rev_type_map'][a2data[1]]

                                # add to argument
                                args_data_.append((rtype, a2data))

                                # store if it is trigger
                                if a2data[0].startswith('TR'):
                                    arg2s.append(a2data)

                                # check if it is entity but should be written to *.a2 file
                                # CG: DNA_domain_or_region, Protein_domain_or_region
                                elif a2type in params['a2_entities']:
                                    arg2s.append(a2data)

                                # store if predict both entity and trigger
                                elif params["ner_predict_all"]:
                                    arg2s.append(a2data)

                            # otherwise, this event contains not existing entity, do not store this event
                            else:
                                valid_ev_flag = False

                # store event
                if valid_ev_flag:

                    # add trigger
                    if trigger not in triggers_:
                        triggers_.append(trigger)

                    # add if trigger is in argument
                    for arg2 in arg2s:
                        if arg2 not in triggers_:
                            triggers_.append(arg2)

                    # create event and store to the list
                    mod_pred = ev_[-1] + 1

                    event = [ev_id, trigger, args_data_, mod_pred]
                    added_evid_list.append(ev_id)
                    events_.append(event)

        # store triggers and events for this document
        preds_output[fid] = [triggers_, events_]

    return preds_output


def convert_evid_to_number(str_evid):
    evid = str_evid.split('_')
    if evid[0] == '0':
        return int('1000' + evid[1])
    return int(evid[0] + evid[1])


# write events to file
def write_ev_2file(pred_output, result_dir, params):
    rev_type_map = params['mappings']['rev_type_map']

    dir2wr = result_dir + 'ev-last/ev-ann/'
    if not os.path.exists(dir2wr):
        os.makedirs(dir2wr)
    else:
        os.system('rm ' + dir2wr + '*.a2')

    for fid, preds in pred_output.items():
        triggers = preds[0]
        events = preds[1]

        with open(dir2wr + fid + '.a2', 'w') as o2file:

            for trigger in triggers:
                o2file.write(trigger[0].replace('TR', 'T') + '\t' + rev_type_map[trigger[1]] + ' ' +
                             str(trigger[2][0]) + ' ' + str(trigger[2][1]) + '\t' + trigger[3] + '\n')

            # count event id
            f_evid = 0

            # mapping event id to incremental id
            f_evid_map = collections.OrderedDict()

            # store modality
            mod_list = []

            for event_ in events:

                # create event id
                evid = convert_evid_to_number(event_[0])

                # lookup in the map or create a new id
                if evid in f_evid_map:
                    evid_out = f_evid_map[evid]
                else:
                    f_evid += 1
                    evid_out = f_evid
                    f_evid_map[evid] = evid_out

                idTR = event_[1][0].replace('TR', 'T')
                typeEV = rev_type_map[event_[1][1]]
                args_data = event_[2]
                mod_pred = event_[3]

                args_output = ''
                for arg_ in args_data:

                    # relation type
                    typeR = arg_[0]

                    # check event or entity argument
                    if len(arg_) > 2:
                        argIdE = arg_[1]
                        nest_evid = convert_evid_to_number(argIdE)
                        if nest_evid in f_evid_map:
                            nest_evid_out = f_evid_map[nest_evid]
                            idT = 'E' + str(nest_evid_out)
                        else:
                            print('ERROR: NESTED EVENT BUT MISSING EVENT ARGUMENT.')

                    # entity argument
                    else:
                        a2data = arg_[1]
                        idT = a2data[0].replace('TR', 'T')

                    if len(args_output) > 0:
                        args_output += ' '

                    args_output += typeR + ':' + idT

                # if has argument
                if len(args_output) > 0:
                    o2file.write('E' + str(evid_out) + '\t' + typeEV + ':' + idTR + ' ' + args_output + '\n')

                # no argument
                else:
                    o2file.write('E' + str(evid_out) + '\t' + typeEV + ':' + idTR + '\n')

                # check and store modality
                if mod_pred > 1:
                    mod_value = params['mappings']['rev_modality_map'][mod_pred]
                    mod_list.append([mod_value, evid_out])

            # write modality
            if len(mod_list) > 0:
                for mod_id, mod_data in enumerate(mod_list):
                    mod_type = mod_data[0]
                    evid_out = mod_data[1]
                    o2file.write('M' + str(mod_id + 1) + '\t' + mod_type + ' ' + 'E' + str(evid_out) + '\n')

    return


# generate event output and evaluation
def evaluate_ev(fids, all_ent_preds, all_words, all_offsets, all_span_terms, all_span_indices, all_sub_to_words,
                all_ev_preds, params, gold_dir, result_dir):
    # generate predicted entities
    pred_ents = generate_entities(fids=fids,
                                  all_e_preds=all_ent_preds,
                                  all_words=all_words,
                                  all_offsets=all_offsets,
                                  all_span_terms=all_span_terms,
                                  all_span_indices=all_span_indices,
                                  all_sub_to_words=all_sub_to_words,
                                  params=params)

    # generate predicted events
    pred_evs = generate_events(fids=fids,
                               all_ev_preds=all_ev_preds,
                               params=params)

    # generate event output
    preds_output = generate_ev_output(pred_ents, pred_evs, params)

    # write output to file
    _ = write_ev_2file(preds_output, result_dir, params)

    # calculate score
    ev_scores = eval_performance(gold_dir, result_dir, params)

    return ev_scores


def eval_performance(ref_dir, result_dir, params):
    # create prediction paths
    pred_dir = ''.join([result_dir, 'ev-last/ev-ann/'])
    pred_scores_file = ''.join([result_dir, 'ev-last/', 'ev-scores-', params['task_name'], params['ev_matching'], '.txt'])

    try:

        command = ''.join(
            ["python " + params['ev_eval_script_path'], " -r ", ref_dir, " -d ", pred_dir, " ", params['ev_matching'],
             " > ", pred_scores_file])

        # exception for ezcat task
        if 'ezcat' in params['task_name']:
            command = ''.join(
                ["python " + params['ev_eval_script_path'], " -r ", ref_dir, " ", pred_dir, " ",
                 params['ev_matching'],
                 " > ", pred_scores_file])

        os.system(command)
        ev_scores = extract_fscore(pred_scores_file)
    except Exception as ex:
        ev_scores = {}
        logger.exception(ex)

    return ev_scores


def extract_fscore(path):
    file = open(path, 'r')
    lines = file.readlines()
    sub_fscore = '0'
    sub_recall = '0'
    sub_precision = '0'
    mod_fscore = '0'
    mod_recall = '0'
    mod_precision = '0'
    tot_fscore = '0'
    tot_recall = '0'
    tot_precision = '0'
    for line in lines:
        if line.split()[0] == '===[SUB-TOTAL]===':
            tokens = line.split()
            sub_recall = tokens[-3]
            sub_precision = tokens[-2]
            sub_fscore = tokens[-1]
        elif line.split()[0] == '==[MOD-TOTAL]==':
            tokens = line.split()
            mod_recall = tokens[-3]
            mod_precision = tokens[-2]
            mod_fscore = tokens[-1]
        elif line.split()[0] == '====[TOTAL]====':
            tokens = line.split()
            tot_recall = tokens[-3]
            tot_precision = tokens[-2]
            tot_fscore = tokens[-1]

    return {'sub_scores': (float(sub_precision.strip()), float(sub_recall.strip()), float(sub_fscore.strip())),
            'mod_scores': (float(mod_precision.strip()), float(mod_recall.strip()), float(mod_fscore.strip())),
            'tot_scores': (float(tot_precision.strip()), float(tot_recall.strip()), float(tot_fscore.strip()))}

# write events to file

# generate event output and evaluation
def write_events_bio(fids, all_ent_preds, all_words, all_offsets, all_span_terms, all_span_indices, all_sub_to_words,
                 all_ev_preds, g_entity_ids_, params, result_dir):
    # generate predicted entities
    pred_ents = generate_entities(fids=fids,
                                  all_e_preds=all_ent_preds,
                                  all_words=all_words,
                                  all_offsets=all_offsets,
                                  all_span_terms=all_span_terms,
                                  all_span_indices=all_span_indices,
                                  all_sub_to_words=all_sub_to_words,
                                  params=params)

    # generate predicted events
    pred_evs = generate_events(fids=fids,
                               all_ev_preds=all_ev_preds,
                               params=params)

    # generate event output
    preds_output = generate_ev_output(pred_ents, pred_evs, params)

    # write output to file
    write_ev_2file_bio(preds_output, pred_ents, result_dir, g_entity_ids_, params)

    return

def mapping_entity_id(en_preds_, g_entity_ids_, rev_type_map, params):
    # if gold entity, starting trigger id from max entity id + 1
    if not params['ner_predict_all'] and len(g_entity_ids_) > 0:
        eid = g_entity_ids_[-1] + 1

    # predict both entity and trigger
    else:
        eid = 1

    # mapping
    enid_mapping = collections.OrderedDict()
    en_preds_out_ = []

    # entity in a2
    a2_ents_ = []

    # create mapping for entity id first
    for pr_id, en_pred in en_preds_.items():

        # id
        en_id = en_pred[0]

        if en_id.startswith('TR'):
            continue

        elif en_id.startswith('T'):
            enid_mapping[en_id] = 'T' + str(eid)
            eid += 1
            en_preds_out_.append(en_pred)

        # using gold entity but in a2
        if not params['ner_predict_all']:
            etype = rev_type_map[en_pred[1]]

            # check entity type in a2
            if etype in params['a2_entities']:
                a2_ents_.append(en_id)

    # creat mapping for trigger id
    for pr_id, en_pred in en_preds_.items():
        # id
        en_id = en_pred[0]

        if en_id.startswith('TR'):
            enid_mapping[en_id] = 'T' + str(eid)
            eid += 1
            en_preds_out_.append(en_pred)

    return enid_mapping, en_preds_out_, a2_ents_

def write_ev_2file_bio(pred_output, pred_ents, result_dir, g_entity_ids_, params):
    a2dir = result_dir + 'ev-last/ev-tok-a2/'
    anndir = result_dir + 'ev-last/ev-tok-ann/'
    rev_type_map = params['mappings']['rev_type_map']

    # entity id mapping
    # feid_mapping = collections.OrderedDict()

    if not os.path.exists(a2dir):
        os.makedirs(a2dir)
    else:
        os.system('rm ' + a2dir + '*.a2')

    if not os.path.exists(anndir):
        os.makedirs(anndir)
    else:
        os.system('rm ' + anndir + '*.a2')
        os.system('rm ' + anndir + '*.a1')

    # write event and triggers, (and entity: if predict both entity and trigger)
    for fid, preds in pred_output.items():
        ev_en_preds_ = preds[0]
        events = preds[1]

        enid_mapping, en_preds_out_, a2_ents_ = mapping_entity_id(pred_ents[fid], g_entity_ids_[fid], rev_type_map, params)

        # entity and trigger for ann file
        ann_en_lines = []
        ann_tr_lines = []
        # write entity and trigger from entity predictions
        for pr_id, e_pred in pred_ents[fid].items():
            e0_id = e_pred[0]
            e_id = enid_mapping[e0_id]

            output = ''.join(
                [e_id, '\t', rev_type_map[e_pred[1]], ' ', str(e_pred[2][0]), ' ', str(e_pred[2][1]), '\t',
                 e_pred[3], '\n'])

            if e0_id.startswith('TR'):
                ann_tr_lines.append(output)

            # only write entity to a1
            elif e0_id.startswith('T'):

                # entity in a2
                if e0_id in a2_ents_:
                    ann_tr_lines.append(output)

                else:
                    ann_en_lines.append(output)


        # entity and trigger output for a2
        a2_en_lines_ = []
        a2_tr_lines_ = []

        # write entity and trigger only included event predictions
        # write entity and then trigger
        for e_pred in ev_en_preds_:
            e0_id = e_pred[0]
            e_id = enid_mapping[e0_id]

            output = ''.join(
                [e_id, '\t', rev_type_map[e_pred[1]], ' ', str(e_pred[2][0]), ' ', str(e_pred[2][1]), '\t',
                 e_pred[3], '\n'])

            if e0_id.startswith('TR'):
                a2_tr_lines_.append(output)
            elif e0_id.startswith('T'):

                # entity in a2
                if e0_id in a2_ents_:
                    a2_tr_lines_.append(output)

                else:
                    a2_en_lines_.append(output)

        # event output
        ev_lines = []

        # count event id
        f_evid = 0

        # mapping event id to incremental id
        f_evid_map = collections.OrderedDict()

        # store modality
        mod_list = []

        for event_ in events:

            # create event id
            evid = convert_evid_to_number(event_[0])

            # lookup in the map or create a new id
            if evid in f_evid_map:
                evid_out = f_evid_map[evid]
            else:
                f_evid += 1
                evid_out = f_evid
                f_evid_map[evid] = evid_out

            trid = event_[1][0]
            trid = enid_mapping[trid]
            typeEV = rev_type_map[event_[1][1]]
            args_data = event_[2]
            mod_pred = event_[3]

            args_output = ''
            for arg_ in args_data:

                # relation type
                typeR = arg_[0]

                # check event or entity argument
                if len(arg_) > 2:
                    argIdE = arg_[1]
                    nest_evid = convert_evid_to_number(argIdE)
                    if nest_evid in f_evid_map:
                        nest_evid_out = f_evid_map[nest_evid]
                        eid = 'E' + str(nest_evid_out)
                    else:
                        print('ERROR: NESTED EVENT BUT MISSING EVENT ARGUMENT.')

                # entity argument
                else:
                    a2data = arg_[1]
                    eid = a2data[0]

                    # mapping entity id: predict entity or entity in a2
                    if params['ner_predict_all'] or eid in a2_ents_:
                        eid = enid_mapping[eid]

                if len(args_output) > 0:
                    args_output += ' '

                args_output += typeR + ':' + eid

            # if has argument
            if len(args_output) > 0:
                output = ''.join(['E', str(evid_out), '\t', typeEV, ':', trid, ' ', args_output, '\n'])
                ev_lines.append(output)

            # no argument
            else:
                output = ''.join(['E', str(evid_out), '\t', typeEV, ':', trid, '\n'])
                ev_lines.append(output)

            # check and store modality
            if mod_pred > 1:
                mod_value = params['mappings']['rev_modality_map'][mod_pred]
                mod_list.append([mod_value, evid_out])

        # write modality
        if len(mod_list) > 0:
            for mod_id, mod_data in enumerate(mod_list):
                mod_type = mod_data[0]
                evid_out = mod_data[1]
                output = ''.join(['M', str(mod_id + 1), '\t', mod_type, ' ', 'E', str(evid_out), '\n'])
                ev_lines.append(output)

        # write a2 files
        with open(a2dir + fid + '.a2', 'w') as o2file:

            # write entity
            if params['ner_predict_all']:
                for entity in a2_en_lines_:
                    o2file.write(entity)

            for trigger in a2_tr_lines_:
                o2file.write(trigger)
            for event in ev_lines:
                o2file.write(event)

        # write ann file
        with open(anndir + fid + '.a1', 'w') as o1file:
            for entity in ann_en_lines:
                o1file.write(entity)

        with open(anndir + fid + '.a2', 'w') as annfile:
            for entity in ann_en_lines:
                annfile.write(entity)
            for trigger in ann_tr_lines:
                annfile.write(trigger)

            # events are the same for both a2 and ann
            for event in ev_lines:
                annfile.write(event)

    return