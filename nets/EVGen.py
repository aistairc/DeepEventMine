"""To generate events given triggers, entities, relations, and event structure."""

import numpy as np
import collections

from torch import nn


class EV_Generator(nn.Module):
    """CLASS FOR GENERATING FLAT EVENT CANDIDATE INDICES."""

    def __init__(self, params):
        super(EV_Generator, self).__init__()

        # parameters
        self.params = params

    def group_rels(self, l2r, rpred_types, rpred_ids, etypes):
        """For generating event candidates."""

        # store a list of relations for each trigger, flat events only contain these relations
        flat_rels_group = collections.defaultdict(list)

        # store a list of relations for each trigger with nested events (the relations between two triggers)
        nest_rels_group = collections.defaultdict(list)

        for xx, rid in enumerate(rpred_ids):
            # indices
            bid = l2r[0][rid]
            a1id = l2r[1][rid]
            a2id = l2r[2][rid]

            # rtype
            rtypeid = rpred_types[rid]

            # entities
            a1typeid = etypes[(bid, a1id)].item()
            a2typeid = etypes[(bid, a2id)].item()

            # if a1 is trigger
            if a1typeid in self.params['trTypes_Ids'] and a2typeid not in self.params['trTypes_Ids']:
                flat_rels_group[(bid.item(), a1id.item())].append(
                    [a1typeid, rid.item(), (rtypeid, a2typeid), (bid.item(), a2id.item())])

            # if a2 is trigger
            elif a2typeid in self.params['trTypes_Ids'] and a1typeid not in self.params['trTypes_Ids']:
                flat_rels_group[(bid.item(), a2id.item())].append(
                    [a2typeid, rid.item(), (rtypeid, a1typeid), (bid.item(), a1id.item())])

            # if both a1 and a2 are trigger: this can be for nested events
            if a1typeid in self.params['trTypes_Ids'] and a2typeid in self.params['trTypes_Ids']:
                nest_rels_group[(bid.item(), a1id.item())].append(
                    [a1typeid, rid.item(), (rtypeid, a2typeid), (bid.item(), a2id.item())])

        return flat_rels_group, nest_rels_group

    def add_no_arg_trigger(self, tr_ids, etypes, flat_structs_map):
        """Add no-argument triggers."""

        # store in a map: key is trigger id, value is a pair of (rtype, trigger type); rtype is a special type
        no_arg_group = collections.defaultdict(list)

        # consider all trigger
        for trid_ in tr_ids:
            trid = (trid_[0].item(), trid_[1].item())

            truth = [-1]
            mod_label = [-1]

            # rtype and trigger type
            rtype = self.params['voc_sizes']['rel_size']
            trtypeid = etypes[trid].item()

            # get structure via trigger type
            ev_structs = flat_structs_map[trtypeid][0]

            # store: [truth, modality label, ev structure]
            if ev_structs != -1:
                no_arg_group[trid] = [(rtype, trtypeid), truth, mod_label]

        return no_arg_group

    def add_truth_to_trigger(self, rels_group, structs_map, levelid=0):
        """For generating event candidates.
        # add event truth and labels to each trigger
        # levelid = 0: flat, levelid=1: nested events
        """

        for trid, rel_group in rels_group.items():

            truth = -1 * np.ones((self.params['max_ev_level'] + 1, self.params['max_ev_args'] + 1), dtype=np.object)
            mod_label = [-1]

            if levelid == 0:
                level_truth = truth[levelid]
            else:
                level_truth = truth

            # get event structure for this trigger via trigger type
            a1type = rel_group[0][0]

            # get all 5 cases of arguments
            ev_structs = structs_map[a1type]

            # add truth, label, and structures to the last element
            rel_group.append([level_truth, mod_label, ev_structs])

        return rels_group

    def create_no_arg_candidates(self, trid, rel_group, ev_truth, mod_labels):
        """Creating candidate with no argument.
        # store candidate in a list
        # format: [0=trig_id, 1-ev-structure-counter, 2-ev-structure-order, 3-ev_label, 4=modality label, 5=[list IN/OUT ids] ]
        """
        a1id = trid[1]
        cand_eids = [a1id]
        cand_struct = collections.Counter([rel_group])

        cand_output = self.generate_candidate_output(trid, cand_struct, [], [0], cand_eids, ev_truth[0], mod_labels)

        return cand_output

    def create_one_flat_arg_candidates(self, trid, rels_group, args_list, ev_truth, mod_labels, ev_structs):
        """Creating candidate with one argument for flat event.
        For flat events.
        """

        # store candidate in a list, there is only one candidate in this case
        # format: [0=trig_id, 1-ev-structure-counter, 2-ev-structure-order, 3-ev_label, 4=modality label, 5=[list IN/OUT ids] ]
        cand_output = []

        # convert list to Counter to compare with event structure
        cand_struct = collections.Counter(args_list)

        # get the list of event structures with one-argument (id=1)
        ev_structs = ev_structs[1]

        # check there is structure for this trigger type
        if ev_structs != -1:

            # compare if this pair (relation type, trigger type) is in the structure, then create candidate
            if cand_struct in ev_structs:
                # get argument id stored in rels_group[0][3] (0=only one relation, 3=id in the list)[1]=entity id
                a2id = rels_group[0][3][1]
                cand_eids = [a2id]  # add to a list for Counter

                # 3: arg_ids_list = empty list [], means there is only one argument
                cand_output = self.generate_candidate_output(trid, cand_struct, args_list, [0], cand_eids, ev_truth[1],
                                                             mod_labels)

        # otherwise: this cannot be a candidate, because no argument in structures

        return cand_output

    def create_multiple_flat_arg_candidates(self, trid, rels_group, args_list, n_args, ev_truth, mod_labels,
                                            ev_structs):
        """Creating candidate with more than one argument for flat event.
            For flat events.
        """

        # store candidate in a list
        cands = []

        # get the list of event structures with 1, 2, 3, 4 arguments
        ev_structs1 = ev_structs[1]
        ev_structs2 = ev_structs[2]
        ev_structs3 = ev_structs[3]
        ev_structs4 = ev_structs[4]

        # init for ignore empty structure
        isArg1 = ev_structs1 != -1
        isArg2 = ev_structs2 != -1
        isArg3 = ev_structs3 != -1
        isArg4 = ev_structs4 != -1

        # generate all possible combinations among arguments with limited by the maximum number of args
        max_n_args = self.params['max_ev_args']

        for xx1, arg1_ in enumerate(args_list):

            # check one arg first
            args1_ = [arg1_]
            args1 = collections.Counter(args1_)

            # argument id, rid
            a1id = rels_group[xx1][3][1]

            if isArg1 and args1 in ev_structs1:
                # get argument ids
                a2ids = [a1id]

                # generate candidate output
                # 3rd argument: number of event arguments
                # 4th argument: [IN indices] # the remain in the argument list will be OUT indices
                cand_output = self.generate_candidate_output(trid, args1, args1_, [xx1], a2ids, ev_truth[1], mod_labels)
                cands.append(cand_output)

            # check for 2, 3, 4 combined args
            if n_args > 1:

                for xx2 in range(xx1 + 1, len(args_list)):
                    arg2_ = args_list[xx2]

                    # combine two arguments
                    args12_ = [arg1_, arg2_]
                    args12 = collections.Counter(args12_)

                    # argument id
                    a2id = rels_group[xx2][3][1]

                    if isArg2 and args12 in ev_structs2:
                        # get argument ids
                        a2ids = [a1id, a2id]

                        # generate candidate output
                        cand_output = self.generate_candidate_output(trid, args12, args12_, [xx1, xx2], a2ids,
                                                                     ev_truth[2], mod_labels)
                        cands.append(cand_output)

                    # check for 3, 4 arguments
                    if n_args > 2:
                        for xx3 in range(xx2 + 1, len(args_list)):
                            arg3_ = args_list[xx3]

                            # combine three arguments
                            args123_ = [arg1_, arg2_, arg3_]
                            args123 = collections.Counter(args123_)

                            # argument id
                            a3id = rels_group[xx3][3][1]

                            if isArg3 and args123 in ev_structs3:
                                # get argument ids
                                a2ids = [a1id, a2id, a3id]

                                # generate candidate output
                                cand_output = self.generate_candidate_output(trid, args123, args123_, [xx1, xx2, xx3],
                                                                             a2ids, ev_truth[3],
                                                                             mod_labels)
                                cands.append(cand_output)

                            # check for 4 arguments
                            if n_args > 3 and max_n_args == 4:
                                for xx4 in range(xx3 + 1, len(args_list)):
                                    arg4_ = args_list[xx4]

                                    # combine four arguments
                                    args1234_ = [arg1_, arg2_, arg3_, arg4_]
                                    args1234 = collections.Counter(args1234_)

                                    # argument id
                                    a4id = rels_group[xx4][3][1]

                                    if isArg4 and args1234 in ev_structs4:
                                        # get argument ids
                                        a2ids = [a1id, a2id, a3id, a4id]

                                        # generate candidate output
                                        cand_output = self.generate_candidate_output(trid, args1234, args1234_,
                                                                                     [xx1, xx2, xx3, xx4],
                                                                                     a2ids, ev_truth[4], mod_labels)
                                        cands.append(cand_output)

        return cands

    def create_one_nest_arg_candidates(self, trid, rels_group, args_list, ev_truth, mod_labels, ev_structs,
                                       current_level):
        """Creating candidate with one argument for nested event.
        For nested events.
        """

        # store candidate in a list, there is only one candidate in this case
        # format: [0=trig_id, 1-ev-structure-counter, 2-ev-structure-order, 3-ev_label, 4=modality label, 5=[list IN/OUT ids] ]
        cands = []

        # convert list to Counter to compare with event structure
        cand_struct = collections.Counter(args_list)

        # get the list of event structures with one-argument (id=1)
        ev_structs1 = ev_structs[1]

        # check there is structure for this trigger type
        if ev_structs1 != -1:

            # compare if this pair (relation type, trigger type) is in the structure, then create candidate
            if cand_struct in ev_structs1:

                # get argument id stored in rels_group[0][3] (0=only one relation, 3=id in the list)[1]=entity id

                if len(rels_group[0]) > 5:

                    # check which level: it is the latest level, added in this level, or not
                    if rels_group[0][5] == 1:

                        # try for all predicted events with this trigger
                        # get the latest level
                        for level1, pred1_ in enumerate(rels_group[0][4]):
                            if level1 == current_level:
                                for (a1id, p1id) in pred1_:
                                    a2ids = [a1id]

                                    cand_output = self.generate_nest_candidate_output(trid, cand_struct, args_list, [0],
                                                                                      [p1id], a2ids, ev_truth,
                                                                                      mod_labels, current_level)
                                    cands.append(cand_output)

        # otherwise: this cannot be a candidate, because no argument in structures

        return cands

    def create_multiple_nest_arg_candidates(self, trid, rels_group, args_list, n_args, ev_truth, mod_labels, ev_structs,
                                            current_level):
        """Creating candidate with more than one argument for nested event.
            For nested events.
        """

        # store candidate in a list
        cands = []

        # get the list of event structures with 1, 2, 3, 4 arguments
        ev_structs1 = ev_structs[1]
        ev_structs2 = ev_structs[2]
        ev_structs3 = ev_structs[3]
        ev_structs4 = ev_structs[4]

        # init for ignore empty structure
        isArg1 = ev_structs1 != -1
        isArg2 = ev_structs2 != -1
        isArg3 = ev_structs3 != -1
        isArg4 = ev_structs4 != -1

        # generate all possible combinations among arguments with limited by the maximum number of args
        max_n_args = self.params['max_ev_args']

        for xx1, arg1_ in enumerate(args_list):

            # check one arg first
            args1_ = [arg1_]
            args1 = collections.Counter(args1_)

            # get pairs of (a2id, (level, positive id))
            preds1_ = rels_group[xx1][4]

            # check if this argument is a trigger
            if len(rels_group[xx1]) > 5 and rels_group[xx1][5] == 1:
                isNest1 = True
            else:
                isNest1 = False

            if isArg1 and args1 in ev_structs1:

                # get argument ids

                # generate candidate output
                # 3rd argument: number of event arguments
                # 4th argument: [IN indices] # the remain in the argument list will be OUT indices

                if isNest1:
                    # try for all predicted events with this trigger
                    for level1, pred1_ in enumerate(preds1_):
                        for (a1id, p1id) in pred1_:
                            a2ids = [a1id]
                            cand_output = self.generate_nest_candidate_output(trid, args1, args1_, [xx1], [p1id], a2ids,
                                                                              ev_truth[1], mod_labels, current_level)
                            cands.append(cand_output)

            # check for 2, 3, 4 combined args
            if n_args > 1:

                for xx2 in range(xx1 + 1, len(args_list)):
                    arg2_ = args_list[xx2]

                    # combine two arguments
                    args12_ = [arg1_, arg2_]
                    args12 = collections.Counter(args12_)

                    # get pairs of (a2id, (level, positive id))
                    preds2_ = rels_group[xx2][4]

                    # check if this argument is a trigger
                    if len(rels_group[xx2]) > 5 and rels_group[xx2][5] == 1:
                        isNest2 = True
                    else:
                        isNest2 = False

                    if isArg2 and args12 in ev_structs2:

                        # at least one argument is trigger
                        if isNest1 or isNest2:

                            # try for all predicted events with this trigger
                            for level1, pred1_ in enumerate(preds1_):
                                for level2, pred2_ in enumerate(preds2_):

                                    # make sure there is at least one new level to avoid duplicate
                                    if level1 == current_level or level2 == current_level:

                                        # try all combinations between any argument of any level
                                        for (a1id, p1id) in pred1_:
                                            for (a2id, p2id) in pred2_:
                                                a2ids = [a1id, a2id]

                                                # generate candidate output
                                                cand_output = self.generate_nest_candidate_output(trid, args12, args12_,
                                                                                                  [xx1, xx2],
                                                                                                  [p1id, p2id], a2ids,
                                                                                                  ev_truth[2],
                                                                                                  mod_labels,
                                                                                                  current_level)
                                                cands.append(cand_output)

                    # check for 3, 4 arguments
                    if n_args > 2:
                        for xx3 in range(xx2 + 1, len(args_list)):
                            arg3_ = args_list[xx3]

                            # combine three arguments
                            args123_ = [arg1_, arg2_, arg3_]
                            args123 = collections.Counter(args123_)

                            # get pairs of (a2id, (level, positive id))
                            preds3_ = rels_group[xx3][4]

                            # check if this argument is a trigger
                            if len(rels_group[xx3]) > 5 and rels_group[xx3][5] == 1:
                                isNest3 = True
                            else:
                                isNest3 = False

                            if isArg3 and args123 in ev_structs3:

                                # at least one argument is trigger
                                if isNest1 or isNest2 or isNest3:

                                    # try for all predicted events with this trigger
                                    for level1, pred1_ in enumerate(preds1_):
                                        for level2, pred2_ in enumerate(preds2_):
                                            for level3, pred3_ in enumerate(preds3_):

                                                # make sure there is at least one new level to avoid duplicate
                                                if level1 == current_level or level2 == current_level or level3 == current_level:

                                                    # try all combinations between any argument of any level
                                                    for (a1id, p1id) in pred1_:
                                                        for (a2id, p2id) in pred2_:
                                                            for (a3id, p3id) in pred3_:
                                                                a2ids = [a1id, a2id, a3id]

                                                                # generate candidate output
                                                                cand_output = self.generate_nest_candidate_output(trid,
                                                                                                                  args123,
                                                                                                                  args123_,
                                                                                                                  [xx1,
                                                                                                                   xx2,
                                                                                                                   xx3],
                                                                                                                  [p1id,
                                                                                                                   p2id,
                                                                                                                   p3id],
                                                                                                                  a2ids,
                                                                                                                  ev_truth[
                                                                                                                      3],
                                                                                                                  mod_labels,
                                                                                                                  current_level)
                                                                cands.append(cand_output)

                            # check for 4 arguments
                            if n_args > 3 and max_n_args == 4:
                                for xx4 in range(xx3 + 1, len(args_list)):
                                    arg4_ = args_list[xx4]

                                    # combine four arguments
                                    args1234_ = [arg1_, arg2_, arg3_, arg4_]
                                    args1234 = collections.Counter(args1234_)

                                    # get pairs of (a2id, (level, positive id))
                                    preds4_ = rels_group[xx4][4]

                                    # check if this argument is a trigger
                                    if len(rels_group[xx4]) > 5 and rels_group[xx4][5] == 1:
                                        isNest4 = True
                                    else:
                                        isNest4 = False

                                    if isArg4 and args1234 in ev_structs4:

                                        # at least one argument is trigger
                                        if isNest1 or isNest2 or isNest3 or isNest4:

                                            # try for all predicted events with this trigger
                                            for level1, pred1_ in enumerate(preds1_):
                                                for level2, pred2_ in enumerate(preds2_):
                                                    for level3, pred3_ in enumerate(preds3_):
                                                        for level4, pred4_ in enumerate(preds4_):

                                                            # make sure there is at least one new level to avoid duplicate
                                                            if level1 == current_level or level2 == current_level or level3 == current_level or level4 == current_level:

                                                                # try all combinations between any argument of any level
                                                                for (a1id, p1id) in pred1_:
                                                                    for (a2id, p2id) in pred2_:
                                                                        for (a3id, p3id) in pred3_:
                                                                            for (a4id, p4id) in pred4_:
                                                                                a2ids = [a1id, a2id, a3id, a4id]

                                                                                # generate candidate output
                                                                                cand_output = self.generate_nest_candidate_output(
                                                                                    trid, args1234, args1234_,
                                                                                    [xx1, xx2, xx3, xx4],
                                                                                    [p1id, p2id, p3id, p4id],
                                                                                    a2ids, ev_truth[4], mod_labels,
                                                                                    current_level)
                                                                                cands.append(cand_output)

        return cands

    def create_arg_candidates(self, trid, rels_group, ev_truth, mod_labels, ev_structs):
        """
        Creating candidates which have arguments (1,2,3,..).
        2 cases:
            1. for 1 argument
            2. for >1 argument
        :param trid:
        :param rels_group: a list of relation_group, each relation_group contains 4 elements
            + 0, trigger type
            + 1, relation id
            + 2, a tuple of (relation type, argument type)
            + 3, argument id (batch id, entity id)
        :param ev_truth:
        :param ev_label:
        :param ev_structs:
        :return:
        """

        # store candidate in a list [trig_id, ev_label]; there may have more than one candidate
        arg_cands = []

        # store rids
        relids = []

        # store a list of arguments: [ (rtype1, etype1), (rtype2, etype2), (rtype3, etype3), (rtype4, etype4), .. ]
        args_list = []

        # store argument entity ids
        argids = []

        for rel_group in rels_group:
            relids.append(rel_group[1])
            arg_type_pair = rel_group[2]
            args_list.append(arg_type_pair)
            argids.append(rel_group[3])

        # get the number of arguments, then process for each case
        n_args = len(args_list)

        # one argument
        if n_args == 1:

            one_arg_cand = self.create_one_flat_arg_candidates(trid, rels_group, args_list, ev_truth, mod_labels,
                                                               ev_structs)

            if len(one_arg_cand) > 0:
                arg_cands.append(one_arg_cand)

        # two arguments
        elif n_args >= 2:

            two_arg_cands = self.create_multiple_flat_arg_candidates(trid, rels_group, args_list, n_args, ev_truth,
                                                                     mod_labels,
                                                                     ev_structs)

            if len(two_arg_cands) > 0:
                arg_cands.extend(two_arg_cands)

        return arg_cands, relids, argids

    def create_ev_candidates(self, ev_cand_triggers, ev_no_arg_cand_triggers):
        """Creating event candidates from a group of relations for each trigger.
            1-create events with arguments
            2-add non-argument events
        """

        # store candidates in a list
        # format: [0=trig_id, 1-ev-structure-counter, 2-ev-structure-order, 3-ev_label, 4=modality label, 5=[list IN/OUT ids] ]
        ev_candidates = []

        # store relation id and entity argument id for each trigger
        args_triggers = collections.OrderedDict()

        # create candidates for each trigger
        for trid, cand_data in ev_cand_triggers.items():
            rels_group = cand_data[:-1]
            truth = cand_data[-1][0]
            mod_labels = cand_data[-1][1]
            ev_structs = cand_data[-1][2]

            # create candidates
            arg_cands, relids, argids = self.create_arg_candidates(trid, rels_group, truth, mod_labels, ev_structs)

            # check if there are candidates; add all candidates
            if len(arg_cands) > 0:
                ev_candidates.extend(arg_cands)
                args_triggers[trid] = [relids, argids]

        # create candidates for no argument: only for flat events
        for trid, cand_data in ev_no_arg_cand_triggers.items():
            rel_group = cand_data[0]

            truth = cand_data[1]
            mod_labels = cand_data[2]

            # create no arg candidates
            no_arg_cand = self.create_no_arg_candidates(trid, rel_group, truth, mod_labels)
            if len(no_arg_cand) > 0:
                ev_candidates.append(no_arg_cand)
                if trid not in args_triggers:
                    # add the special NONE rel-type
                    args_triggers[trid] = [self.params['voc_sizes']['rel_size']]

        return ev_candidates, args_triggers

    def create_nest_arg_candidates(self, trid, rels_group, ev_truth, mod_labels, ev_structs, current_nested_level):
        """
        Creating candidates which have arguments (1,2,3,..).
        2 cases:
            1. for 1 argument
            2. for >1 argument
        :param trid:
        :param rels_group: a list of relation_group, each relation_group contains 4 elements
            + 0, trigger type
            + 1, relation id
            + 2, a tuple of (relation type, argument type)
            + 3, argument id (batch id, entity id)
        :param ev_truth:
        :param ev_label:
        :param ev_structs:
        :return:
        """

        # store candidate in a list [trig_id, ev_label]; there may have more than one candidate
        arg_cands = []

        # store rids
        relids = []

        # store a list of arguments: [ (rtype1, etype1), (rtype2, etype2), (rtype3, etype3), (rtype4, etype4), .. ]
        args_list = []

        # store argument entity ids
        argids = []

        # store event argument ids
        ev_argids = []

        for rel_group in rels_group:
            relids.append(rel_group[1])
            arg_type_pair = rel_group[2]
            args_list.append(arg_type_pair)

            # if entity argument
            argids.append([rel_group[3]])

            # init for event arguments
            ev_argids.append([])

        # get the number of arguments, then process for each case
        n_args = len(args_list)

        # one argument
        if n_args == 1:

            one_arg_cands = self.create_one_nest_arg_candidates(trid, rels_group, args_list,
                                                                ev_truth[current_nested_level + 1][1], mod_labels,
                                                                ev_structs, current_nested_level)

            if len(one_arg_cands) > 0:
                arg_cands.extend(one_arg_cands)

                # add argument ids to create embeds later
                for arg_cand in one_arg_cands:
                    for (a2id, pid) in zip(arg_cand[5], arg_cand[7]):

                        # add event argument
                        if pid != (-1, -1):
                            if pid not in ev_argids[a2id]:
                                ev_argids[a2id].append(pid)
        # two arguments
        elif n_args >= 2:

            two_arg_cands = self.create_multiple_nest_arg_candidates(trid, rels_group, args_list, n_args,
                                                                     ev_truth[current_nested_level + 1],
                                                                     mod_labels,
                                                                     ev_structs, current_nested_level)

            if len(two_arg_cands) > 0:
                arg_cands.extend(two_arg_cands)

                # add argument ids to create embeds later
                for arg_cand in two_arg_cands:
                    for (a2id, pid) in zip(arg_cand[5], arg_cand[7]):

                        # add event argument
                        if pid != (-1, -1):
                            if pid not in ev_argids[a2id]:
                                ev_argids[a2id].append(pid)

        return arg_cands, relids, argids, ev_argids

    def create_nest_ev_candidates(self, ev_cand_triggers, current_nested_level):
        """Creating event candidates from a group of relations for each trigger.
            1-create events with arguments
            2-add non-argument events
        """

        # store candidates in a list
        # format: [0=trig_id, 1-ev-structure-counter, 2-ev-structure-order, 3-ev_label, 4=modality label, 5=[list IN/OUT ids] ]
        ev_candidates = []

        # store relation id and entity argument id for each trigger
        args_triggers = collections.OrderedDict()

        # create candidates for each trigger
        for trid, cand_data in ev_cand_triggers.items():
            rels_group = cand_data[:-1]
            truth = cand_data[-1][0]
            mod_labels = cand_data[-1][1]
            ev_structs = cand_data[-1][2]

            # create candidates
            arg_cands, relids, argids, ev_argids = self.create_nest_arg_candidates(trid, rels_group, truth, mod_labels,
                                                                                   ev_structs, current_nested_level)

            # check if there are candidates; add all candidates
            if len(arg_cands) > 0:
                ev_candidates.extend(arg_cands)
                args_triggers[trid] = [relids, argids, ev_argids]

        return ev_candidates, args_triggers

    def prepare4nn(self, ev_st_candidates):
        """Prepare indices for creating embeddings."""

        # store trigger indices
        trids_ = []
        ev_labels_ = []
        mod_labels_ = []
        io_ids_ = []
        ev_structs_ = []

        # store for checking nested argument later
        truth_ids_ = []
        pos_ev_ids_ = []

        for xx, ev_cand in enumerate(ev_st_candidates):
            trid = ev_cand[0]
            rel_group = ev_cand[1]
            rel_group_data = ev_cand[2]
            ev_label = ev_cand[3]
            mod_label = ev_cand[4]
            io_ids = ev_cand[5]
            truth_ids = ev_cand[6]

            # add to list
            trids_.append(trid)
            ev_labels_.append(ev_label)
            mod_labels_.append(mod_label)
            io_ids_.append(io_ids)
            ev_structs_.append([rel_group, rel_group_data])
            truth_ids_.append(truth_ids)

            # positive event ids for nested
            if len(ev_cand) > 7:
                pos_ev_ids_.append(ev_cand[7])

        return {'trids_': trids_, 'ev_labels_': ev_labels_, 'mod_labels_': mod_labels_, 'io_ids_': io_ids_,
                'ev_structs_': ev_structs_, 'truth_ids_': truth_ids_, 'pos_ev_ids_': pos_ev_ids_}

    def add_nest_arguments(self, nest_rels_group, ev_flat_arg_ids4nn, flat_rels_group):
        """Add entity arguments for nested trigger.
            Add reverse trigger pair if necessary.
        """

        # store new reversed arguments in a new dictionary
        rev_nest_rels_group = collections.defaultdict(list)

        # check whether trigger argument included in the flat candidate or not, otherwise, can add the reverse
        for trid, rel_groups in nest_rels_group.items():

            # store argument list
            args_list = []

            # store reverse list
            rev_args_list = []

            # store no-flat events
            no_ev_list = []

            # check in the list of arguments
            for rel_group in rel_groups:
                argid = rel_group[3]

                # whether this argument in the flat candidates: if yes, it is valid
                if argid in ev_flat_arg_ids4nn:
                    # continue
                    args_list.append(rel_group)

                # if this argument not in flat candidates, this is not a candidate, we can reverse
                else:

                    # create the reverse data
                    rev_arg = []
                    rev_arg.append(rel_group[2][1])  # trigger type
                    rev_arg.append(rel_group[1])  # relation id
                    rev_arg.append((rel_group[2][0], rel_group[0]))  # (rtype, argtype)
                    rev_arg.append(trid)

                    # check the reverse
                    if trid in ev_flat_arg_ids4nn:

                        rev_args_list.append((argid, rev_arg))



                    # the reverse also not in flat events
                    else:
                        no_ev_list.append([trid, rel_group])
                        no_ev_list.append([argid, rev_arg])

            # add to the map
            if len(args_list) > 0:
                if trid in rev_nest_rels_group:
                    rev_nest_rels_group[trid].extend(args_list)
                else:
                    rev_nest_rels_group[trid] = args_list

            # reverse list
            for rev_arg_ in rev_args_list:
                argid = rev_arg_[0]
                arg_data = rev_arg_[1]
                if argid not in rev_nest_rels_group:
                    rev_nest_rels_group[argid] = [arg_data]
                else:
                    rev_nest_rels_group[argid].append(arg_data)

            # add arguments with no event: add two directions
            if len(no_ev_list) > 0:

                for [trid, rel_group] in no_ev_list:
                    if trid not in rev_nest_rels_group:
                        rev_nest_rels_group[trid] = [rel_group]
                    else:
                        rev_nest_rels_group[trid].append(rel_group)

        # store entity arguments
        for trid, rel_groups in rev_nest_rels_group.items():

            # store the list of trigger and entity arguments
            trig_args = []
            ent_args_list = []
            for trig_arg_data in rel_groups:
                # argid = trig_arg_data[3]

                # add one more element (a list) to store nested positive ids and truth ids later:
                # 4th element: to mark this is an event argument
                trig_arg_data.append([])

                # 5th element: 0/1 to mark this event argument will be used to create next level event candidates
                trig_arg_data.append(0)

                trig_args.append(trig_arg_data)

            # store entities paired with this trigger
            if trid in flat_rels_group:
                ent_args_list = flat_rels_group[trid][:-1]

            # add the list of enity arguments
            if len(ent_args_list) > 0:
                for ent_arg_data in ent_args_list:
                    # add eid and posid to be the same format as trigger argument
                    a2id = ent_arg_data[3][1]
                    ent_arg_data.append([[[a2id, (-1, -1)]]])
                rel_groups.extend(ent_args_list)

        return rev_nest_rels_group

    def generate_event_candidate_structures(self, etypes, tr_ids, l2r, rpred_types, rpred_ids
                                            ):
        """ Generate event candidates structures.
            - Given a list of predicted/gold entities, triggers, relations
            - Given a set of EVENT STRUCTURES (rules by annotation), separated by event type (also trigger type)
            - For each trigger:
                + there are a list of entities and relations connected to this trigger
                + use trigger type to get corresponding EVENT STRUCTURES
                + create all combinations among (relation type, entity type) pairs: 0, 1, 2, 3, 4 arguments
            - Store combinations matched with the EVENT STRUCTURES: obtain CANDIDATES
                + for each CANDIDATE: get corresponding event truth, event label
                + return indices to creat embeddings for prediction, loss, and generate output

        :param etypes: entity types [batch, n_spans] # n_spans: number of spans
        :param tr_ids: trigger indices [batch, n_spans]
        :param l2r: left-to-right [3, n_rels] # n_rels: number of relations; 3: 0=batch id, 1=left id, 2=right id
        :param rtypes: relation types # [n_rels] # predicted relation types
        :param ev_idx: event indices # list for mapping event indices
        :param ev_truth: event truth [batch, n_entities] # n_entities: number of of entities in the mini-batch
        :param ev_lbls: event labels [batch, n_entities x 3] # 3: 0=non-event, 1=event and modality Speculation, 2=event and modality Negation
        :return:
        """

        # get the event structures
        flat_structs_map = self.params['mappings']['flat_types_id_map']
        nest_structs_map = self.params['mappings']['nested_types_id_map']

        # group rels for each trigger: one for flat and one for nested events
        flat_rels_group, nest_rels_group = self.group_rels(l2r, rpred_types, rpred_ids, etypes)

        # add truth, labels, and event structure to each trigger
        # the mapping: key=trigger id, values = a list of[ [list of relations], [truth, label, ev-structures] ]
        ev_flat_cand_triggers = self.add_truth_to_trigger(flat_rels_group, flat_structs_map,
                                                          levelid=0)

        # prepare for no argument candidates
        ev_no_arg_cand_triggers = self.add_no_arg_trigger(tr_ids, etypes, flat_structs_map)

        # create flat event candidates using event structures
        ev_flat_st_candidates, ev_flat_arg_ids4nn = self.create_ev_candidates(ev_flat_cand_triggers,
                                                                              ev_no_arg_cand_triggers)

        # for nested candidates
        # add reverse trigger pairs, entity arguments for nested candidates
        rev_nest_rels_group = self.add_nest_arguments(nest_rels_group, ev_flat_arg_ids4nn, flat_rels_group)

        # add truth: do it later after flat prediction
        ev_nest_cand_triggers = self.add_truth_to_trigger(rev_nest_rels_group,
                                                          nest_structs_map, levelid=1)

        # prepare for creating embeddings from event structure candidates
        ev_flat_cands_ids4nn = self.prepare4nn(ev_flat_st_candidates)

        return {'ev_cand_ids4nn': ev_flat_cands_ids4nn, 'ev_arg_ids4nn': ev_flat_arg_ids4nn,
                'ev_nest_cand_triggers': ev_nest_cand_triggers}

    def _generate(self, etypes, tr_ids, l2r, rpred_types, rpred_ids):
        """Generate event candidates indices for creating embeddings."""

        # a map with two output:
        # 1-event candidate indices: a list of event candidate, [trigger id, event label, modality label, in/out ids]
        # 2-event argument indices for each trigger: a map (key: trigger id, values: ids of relations and entity arguments)
        ev_ids4nn = self.generate_event_candidate_structures(etypes, tr_ids, l2r, rpred_types, rpred_ids
                                                             )

        return ev_ids4nn

    def select_nest_arguments(self, nest_group_rels, flat_pos_tr_ids, flat_pos_truth_ids, current_nested_level):
        """
        Creat nested event candidates given rel-groups and predicted flat event trigger indices.
        :param nest_group_rels: from relations, for each nested event trigger, there is a list of trigger arguments and truth, labels
        :param nest_args_list: for each trigger, there are two lists: 0-trigger arguments, 1-entity arguments
        :param flat_pos_tr_ids: positive trigger indices from prediction of previous level
        :param flat_pos_truth_ids: positive truth from prediction of previous level,
        len(flat_pos_truth_ids) = len(flat_pos_tr_ids)=number of predicted positive events
        :return:
        """

        for trid, args_data in nest_group_rels.items():

            # get arguments
            args_list = args_data[:-1]

            # check whether the trigger argument ids included in the predicted positive tr_ids
            for trig_arg_data in args_list:

                # process for trigger arguments
                if len(trig_arg_data) > 5:
                    argid = trig_arg_data[3]

                    # store which event id will replace trigger argument, and its truth
                    posid_list = []

                    # check all possible appearance of this trigger in the predicted events
                    for posid, pos_trid in enumerate(flat_pos_tr_ids):
                        if argid == pos_trid:
                            pos_truth = flat_pos_truth_ids[posid]

                            # only add positive truth for training
                            if pos_truth != -1 or not self.training:
                                # positive id: index of (level, event id)
                                posid_list.append([pos_truth, (current_nested_level, posid)])

                    # add to the list of arguments by level: 4th element
                    trig_arg_data[4].append(posid_list)
                    # if there is predicted events
                    if len(posid_list) > 0:

                        # mark this is used to make the next level nested candidate
                        trig_arg_data[5] = 1


                    # otherwise: mark this event argument is not used to search for next level nested candidates
                    else:
                        trig_arg_data[5] = 0

        return nest_group_rels

    def generate_candidate_output(self, trid, rel_group_counter, rel_group_list, arg_ids_list, a2_ids, ev_truth,
                                  mod_labels):
        """Create a flat event candidate."""

        # store the output in a list
        # format: [0=trig_id, 1-ev-structure-counter, 2-ev-structure-order, 3-ev_label, 4=modality label, 5=[list IN/OUT ids] ]
        cand_output = []

        # convert ids to Counter to compare
        cand_eids_count = collections.Counter(a2_ids)
        # cand_eids_count = tuple(a2_ids)

        ev_label = 0
        mod_label = 1

        # store truth for nested event later
        matched_truth = -1

        # compare all values in the truths: if training and if truth exists
        if self.training:
            if ev_truth != -1:
                for xx, truth_ in enumerate(ev_truth):
                    if truth_ != -1:

                        # check matching event structure
                        if rel_group_counter == truth_[0][0]:

                            truth_ids_count = truth_[0][1]

                            # check this is a true candidate, for calculating loss
                            if cand_eids_count == truth_ids_count:
                                ev_label = 1
                                mod_label = truth_[1]
                                matched_truth = (0, truth_[0][0], truth_[0][1], truth_[0][2])
                                break

        # store the output
        cand_output.append(trid)
        cand_output.append(rel_group_counter)
        cand_output.append(rel_group_list)
        cand_output.append(ev_label)
        cand_output.append(mod_label)
        cand_output.append(arg_ids_list)
        cand_output.append(matched_truth)

        return cand_output

    def generate_nest_candidate_output(self, trid, rel_group_counter, rel_group_list, arg_ids_list, pos_ids, a2_ids,
                                       ev_truth, mod_labels, current_level):
        """Create a nested event candidate."""

        # store the output in a list
        # format: [0=trig_id, 1-ev-structure-counter, 2-ev-structure-order, 3-ev_label, 4=modality label, 5=[list IN/OUT ids] ]
        cand_output = []

        ev_label = 0
        mod_label = 1

        # store truth for nested event later
        matched_truth = -1

        # compare all values in the truths: if training and if truth exists
        if self.training:

            if ev_truth != -1:

                for xx, truth_ in enumerate(ev_truth):
                    if truth_ != -1:

                        # store entity arguments and event arguments
                        ent_args = []
                        ev_args = []
                        for a2id in a2_ids:
                            if type(a2id) == tuple:
                                ev_args.append(a2id)
                            else:
                                ent_args.append(a2id)
                        if len(ent_args) > 0:
                            ent_args = collections.Counter(ent_args)

                        # get truth argument ids to compare
                        truth_eids_ = truth_[0]

                        # compare event struct
                        if rel_group_counter == truth_eids_[0]:

                            # compare entity arguments
                            if truth_eids_[1] == ent_args:

                                is_matched = True
                                # compare event arguments
                                for truth_ev_arg in truth_eids_[2]:
                                    if truth_ev_arg not in ev_args:
                                        is_matched = False
                                        break
                                if is_matched:
                                    ev_label = 1
                                    mod_label = truth_[1]
                                    matched_truth = (current_level + 1, truth_eids_[0], truth_eids_[1], truth_eids_[2])
                                    break

        # store the output
        cand_output.append(trid)
        cand_output.append(rel_group_counter)
        cand_output.append(rel_group_list)
        cand_output.append(ev_label)
        cand_output.append(mod_label)
        cand_output.append(arg_ids_list)
        cand_output.append(matched_truth)
        cand_output.append(pos_ids)

        return cand_output

    def _generate_nested_candidates(self, current_nested_level, ev_nest_cand_triggers, current_positive_tr_ids,
                                    current_positive_truth_ids):
        """Generate candidate indices for nested events"""

        # create list of arguments by checking the positive predicted event triggers
        ev_nest_cand_triggers = self.select_nest_arguments(ev_nest_cand_triggers, current_positive_tr_ids,
                                                           current_positive_truth_ids, current_nested_level)

        # create flat event candidates using event structures
        ev_nest_st_candidates, ev_nest_arg_ids4nn = self.create_nest_ev_candidates(ev_nest_cand_triggers,
                                                                                   current_nested_level=current_nested_level)

        # prepare for creating embeddings from event structure candidates
        ev_nest_cands_ids4nn = self.prepare4nn(ev_nest_st_candidates)

        return {'ev_nest_cand_ids': ev_nest_cands_ids4nn, 'ev_nest_arg_ids': ev_nest_arg_ids4nn,
                'ev_nest_cand_triggers': ev_nest_cand_triggers}
