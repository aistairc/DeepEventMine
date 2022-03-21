"""Process event structures."""

from glob import glob
import os
import json
from loguru import logger
import collections
from collections import OrderedDict
import numpy as np

from utils import utils


def load_general_rules(cur_rules, params):

    num_dups = 0

    rule_fns = glob(os.path.join(params["rule_dir"], "*.rule"))

    for rule_fn in rule_fns:
        for rule_line in utils.read_lines(rule_fn):
            trigger_id, args = json.loads(rule_line)

            accumulative_level = 0

            rel_arg_pairs = []
            rel_arg_pair_strs = []

            if args:
                for relation_id, level, arg_id in args:
                    accumulative_level += level

                    rel_arg_pairs.append([relation_id, arg_id])
                    rel_arg_pair_strs.append("{}{}{}".format(relation_id, level, arg_id))
            else:
                rel_arg_pairs.append([str(None), trigger_id])
                rel_arg_pair_strs.append("{}{}{}".format(str(None), 0, trigger_id))

            rule_str = "+".join(rel_arg_pair_strs)

            if accumulative_level > 0:
                if trigger_id in cur_rules["structs1"]:
                    if rule_str in cur_rules["structs1"][trigger_id]:
                        num_dups += 1
                    else:
                        cur_rules["structs1"][trigger_id][rule_str] = [rel_arg_pairs]
            else:
                if trigger_id in cur_rules["structs0"]:
                    if rule_str in cur_rules["structs0"][trigger_id]:
                        num_dups += 1
                    else:
                        cur_rules["structs0"][trigger_id][rule_str] = [rel_arg_pairs]

    logger.debug("# Event rule duplicates: {}".format(num_dups))


def remove_invalid_rules(cur_rules):
    for rule_structures in cur_rules.values():
        for trigger_id in rule_structures:
            for rule_str in list(rule_structures[trigger_id]):
                has_relations = {relation_id for relation_id, _ in rule_structures[trigger_id][rule_str][0]}
                if trigger_id == "Mutation" and "Theme" not in has_relations and (
                        "CSite" in has_relations or "Site" in has_relations):
                    del rule_structures[trigger_id][rule_str]
                    logger.info("Removed an invalid rule: {} {}".format(trigger_id, rule_str))


def merge_struct(train_struct, dev_struct):
    for trigger_id, rule_structure in dev_struct.items():
        for rule_str, args in rule_structure.items():

            if trigger_id not in train_struct:
                # this trigger not in train set, create new
                train_struct[trigger_id] = OrderedDict()
            
            train_struct[trigger_id][rule_str] = args


def count_rules(train_struct):
    count = 0
    for type_tr, pairs in train_struct.items():
        count += len(pairs)
    return count


def prep_structs_mapping(structsTR, type_map, rtype_map, rel_size):
    structs_types = OrderedDict()
    structs_map = OrderedDict()

    max_ev_per_tr = 0
    max_rel_per_ev = 0

    for typeTR, structs in structsTR.items():
        typeTRid = type_map[typeTR]
        structs_id = []
        structs_map[typeTR] = OrderedDict()

        max_ev_per_tr = max(max_ev_per_tr, len(structs))

        for struct, struct_data in structs.items():

            rel_id = []
            for rel in struct_data[0]:
                if rel[0] == 'None':
                    r2id = (rel_size, typeTRid)
                    rel_id.append(r2id)
                else:
                    if rel[0] in rtype_map and rel[1] in type_map:
                        r2id = (rtype_map[rel[0]], type_map[rel[1]])
                        rel_id.append(r2id)

            max_rel_per_ev = max(max_rel_per_ev, len(rel_id))

            # structs_map[typeTR][struct] = rel_id
            if len(rel_id) == len(struct_data[0]):
                rel_id_count = collections.Counter(rel_id)
                structs_map[typeTR][struct] = rel_id_count

                # check to avoid duplicate
                if rel_id not in structs_id:
                    structs_id.append(rel_id)
        structs_types[typeTR] = structs_id

    return structs_map, structs_types, max_ev_per_tr, max_rel_per_ev


def prep_struct_map_ids(struct_map, typeTR_map, type_size, rel_size):
    """
    :param struct_map: mapping for each trigger type, there is a list of event structure, each structure is a list of arguments, each argument is a pair of relation type index, entity type index
    :param typeTR_map: mapping, each trigger type is assigned with an integer
    :param type_size: number of entity types + trigger types
    :param typeTR_size: number of trigger types
    :param rel_size: number of relation types
    :return:
        ev_structs_ids: array[type_size x 5], for the number of argument, each element is a list object for event structures of each trigger type
        ev_structs_args: array[type_size], list of arguments (pairs of (relation type, entity type)) for each trigger type
    """
    # convert event structure map into indices, size=[trigger_type_size x 5_arguments] (0 is for no argument)
    ev_structs_ids = -1 * np.ones((type_size + 1, 5), dtype=np.object)
    # struct_arg_map = OrderedDict()

    for typeTR, structs in struct_map.items():
        trid = typeTR_map[typeTR]

        # devide and store arguments separately by the number of arguments
        structs_0arg = []  # no argument
        structs_1arg = []  # 1 argument
        structs_2arg = []  # 2 arguments
        structs_3arg = []  # 3 arguments
        structs_4arg = []  # 4 arguments
        # struct_arg_map[typeTR] = OrderedDict()

        for struct in structs:
            args = []
            no_arg = len(struct)
            for arg in struct:
                args.append(arg)

            # check if it is no arg
            zero_arg = False
            if no_arg == 1 and args[0][0] == rel_size:
                zero_arg = True

            # convert list of arg to counter: compare easier
            args = collections.Counter(args)

            # store args to map
            # struct_arg_map[typeTR][struct] = args

            # check if there is one argument:
            if no_arg == 1:

                # it can be no argument (argument with the relation type is OTHER)
                if zero_arg:
                    if args not in structs_0arg:
                        structs_0arg.append(args)

                # or it can be one arguments
                else:
                    if args not in structs_1arg:
                        structs_1arg.append(args)

            # otherwise: 2, 3, 4 arguments
            elif no_arg == 2:
                if args not in structs_2arg:
                    structs_2arg.append(args)
            elif no_arg == 3:
                if args not in structs_3arg:
                    structs_3arg.append(args)
            elif no_arg == 4:
                if args not in structs_4arg:
                    structs_4arg.append(args)

        # store event structures, for each trigger type id, and for each number of argument: 0..4
        if len(structs_0arg) > 0:
            ev_structs_ids[trid][0] = structs_0arg
        if len(structs_1arg) > 0:
            ev_structs_ids[trid][1] = structs_1arg
        if len(structs_2arg) > 0:
            ev_structs_ids[trid][2] = structs_2arg
        if len(structs_3arg) > 0:
            ev_structs_ids[trid][3] = structs_3arg
        if len(structs_4arg) > 0:
            ev_structs_ids[trid][4] = structs_4arg

    return ev_structs_ids


def prep_pair_mapping(structsTR, type_map):
    etype_pairs = collections.defaultdict(set)
    for _, struct_level in structsTR.items():
        for typeTR, argStructs in struct_level.items():
            for _, argStruct in argStructs.items():
                for argPair in argStruct[0]:
                    typeT = argPair[1]
                    # pair_map[typeTR].add(typeT)
                    if typeT in type_map and typeTR in type_map:
                        etype_pairs[type_map[typeTR]].add(type_map[typeT])

    return etype_pairs


def prep_pair_mapping_from_file(entity_pairs, type_map):
    with open(entity_pairs, 'r') as stream:
        entity_pairs = utils._ordered_load(stream)

    etype_pairs = collections.defaultdict(set)
    for e, p in entity_pairs.items():
        try:
            etype = type_map[e]
            ps = p.split(',')
            for paired_e in ps:
                try:
                    paired_etype = type_map[paired_e]
                    etype_pairs[etype].add(paired_etype)
                except:
                    pass
        except:
            pass

    return etype_pairs


def process_structure(data_struct, data_struct_dev, params, type_map, typeTR_map, rtype_map, type_size, rel_size):
    structs_tr = data_struct['structsTR']
    structs_tr_dev = data_struct_dev['structsTR']

    if params['use_dev_rule']:
        merge_struct(structs_tr['structs0'], structs_tr_dev['structs0'])
        merge_struct(structs_tr['structs1'], structs_tr_dev['structs1'])

    if params['use_general_rule']:
        load_general_rules(structs_tr, params)
        remove_invalid_rules(structs_tr)

    print('Total FLAT rules', count_rules(structs_tr['structs0']))
    print('Total NESTED rules', count_rules(structs_tr['structs1']))

    # convert structure mapping into indices
    flat_structs_map, flat_types_map, max_ev_per_tr0, max_rel_per_ev0 = prep_structs_mapping(structs_tr['structs0'],
                                                                                             type_map, rtype_map,
                                                                                             rel_size)
    nested_structs_map, nested_types_map, max_ev_per_tr1, max_rel_per_ev1 = prep_structs_mapping(structs_tr['structs1'],
                                                                                                 type_map, rtype_map,
                                                                                                 rel_size)

    # create event structures for flat and nested events
    flat_types_id_map = prep_struct_map_ids(flat_types_map, typeTR_map, type_size, rel_size)
    nested_types_id_map = prep_struct_map_ids(nested_types_map, typeTR_map, type_size, rel_size)

    params['max_ev_per_tr'] = max(max_ev_per_tr0, max_ev_per_tr1, params['max_ev_per_tr'])
    params['max_rel_per_ev'] = max(max_rel_per_ev0, max_rel_per_ev1, params['max_rel_per_ev'])
    params['max_rel_per_ev'] += 1

    print('max_ev_per_tr', params['max_ev_per_tr'])
    print('max_rel_per_ev', params['max_rel_per_ev'])

    etype_pairs = prep_pair_mapping(structs_tr, type_map)

    # if params['using_entity_pairs_filter']:
    #     etype_pairs = prep_pair_mapping_from_file(params['entity_pairs'], type_map)

    return flat_structs_map, nested_structs_map, flat_types_id_map, nested_types_id_map, etype_pairs
