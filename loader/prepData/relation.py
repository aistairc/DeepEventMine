"""Process relation information."""

from collections import OrderedDict


def process_relations(readable_entsA, readable_entsB, readable_ents, true_relations, unk, params):
    r_idxs = OrderedDict()
    readable_rels = OrderedDict()

    for e1, ent1 in enumerate(readable_entsA):  # ent1 is A
        if ent1 not in r_idxs:
            r_idxs[ent1] = list(readable_ents.keys()).index(
                ent1)  # find to which index corresponds from all entities
        for e2, ent2 in enumerate(readable_entsB):
            if ent2 not in r_idxs:  # ent2 is B
                r_idxs[ent2] = list(readable_ents.keys()).index(ent2)

            if (ent1, ent2) not in readable_rels:
                readable_rels[(ent1, ent2)] = []
            if (ent2, ent1) not in readable_rels:
                readable_rels[(ent2, ent1)] = []

            # A before B (in text)
            Apos = readable_ents[ent1]['pos2']
            Bpos = readable_ents[ent2]['pos1']

            # if readable_ents[ent1][4][-1] <= readable_ents[ent2][4][0]:
            if Apos <= Bpos:
                pref_f = ''
                pref_b = '_INV'
                arg1 = ent1
                arg2 = ent2
            # B before A (in text)
            else:
                pref_f = '_INV'
                pref_b = ''
                arg1 = ent2
                arg2 = ent1

            Fpair = [('Arg1', arg1), ('Arg2', arg2)]  # forward
            Rpair = [('Arg1', arg2), ('Arg2', arg1)]  # reverse

            total_rels = len(true_relations)
            not_found = 0
            for rel in true_relations:  # existing relations

                if rel[1] == 'Other':  # in case negative relations are already labeled
                    # left-to-right
                    readable_rels[(arg1, arg2)] = (rel[0] + pref_f, '1:Other:2')
                    # right-to-left
                    if params['direction'] != 'l2r':
                        readable_rels[(arg2, arg1)] = (rel[0] + pref_b, '1:Other:2')

                # AB existing relation
                if Fpair == true_relations[rel]:
                    # left-to-right
                    if len(readable_rels[(arg1, arg2)]) == 0:
                        readable_rels[(arg1, arg2)] = (rel[0] + pref_f, '1:' + rel[1] + ':2')
                    # right-to-left
                    if params['direction'] == 'neg':
                        readable_rels[(arg2, arg1)] = (rel[0] + pref_b, '1:Other:2')
                    elif params['direction'] == 'l2r+r2l':
                        if len(readable_rels[(arg2, arg1)]) == 0:
                            readable_rels[(arg2, arg1)] = (rel[0] + pref_b, '2:' + rel[1] + ':1')
                # BA existing relation
                elif Rpair == true_relations[rel]:
                    # left-to-right
                    if len(readable_rels[(arg1, arg2)]) == 0:
                        readable_rels[(arg1, arg2)] = (rel[0] + pref_f, '2:' + rel[1] + ':1')
                    # right-to-left
                    if params['direction'] == 'neg':
                        readable_rels[(arg2, arg1)] = (rel[0] + pref_b, '1:Other:2')
                    elif params['direction'] == 'l2r+r2l':
                        if len(readable_rels[(arg2, arg1)]) == 0:
                            readable_rels[(arg2, arg1)] = (rel[0] + pref_b, '1:' + rel[1] + ':2')
                else:
                    not_found += 1

            # this pair does not have a relation
            if not_found == total_rels:
                if readable_rels[(arg1, arg2)] or readable_rels[
                    (arg2, arg1)]:  # if pair already there, don't do anything
                    continue

                rel_new_id = 'R-' + str(unk)

                # left-to-right
                readable_rels[(arg1, arg2)] = (rel_new_id + pref_f, '1:Other:2')
                # right-to-left
                if params['direction'] != 'l2r' and (ent1 != ent2):
                    readable_rels[(arg2, arg1)] = (rel_new_id + pref_b, '1:Other:2')
                unk += 1

    return r_idxs, readable_rels


def get_rtypes(data_struct, data_struct_dev):
    rel_len = []
    rels = []
    for sid in data_struct['input']:
        sent = data_struct['input'][sid]
        rels2 = []
        for (e1, e2) in sent['readable_r']:
            if sent['readable_r'][(e1, e2)]:
                rels2.append(sent['readable_r'][(e1, e2)][1])
        rels.append(rels2)
        rel_len.append(len(rels2))

    for sid in data_struct_dev['input']:
        sent = data_struct_dev['input'][sid]
        rels2 = []
        for (e1, e2) in sent['readable_r']:
            if sent['readable_r'][(e1, e2)]:
                rels2.append(sent['readable_r'][(e1, e2)][1])
        rels.append(rels2)
        rel_len.append(len(rels2))

    return rels
