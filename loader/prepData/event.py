"""Process events."""

import collections
from collections import OrderedDict


def count_nested_events(events):
    for pmid, fevents in events.items():
        count_nested_evs_level(fevents['data'])


def count_nested_evs_level(fevents):
    for evid, evdata in fevents.items():
        level = count_nested_ev_level(evdata, fevents, current_level=0)
        evdata['nested_ev_level'] = level


def count_nested_ev_level(evdata, fevents, current_level):
    """Nested event level"""

    # to avoid loop forever
    if current_level > 20:
        return current_level

    # flat
    if evdata['is_flat_ev']:
        return current_level

    # nested
    elif 'nested_ev_level' in evdata:
        return current_level + evdata['nested_ev_level']

    else:
        levels = []
        args_ids = evdata['args_ids']
        for arg_id in args_ids:
            if arg_id.startswith('E'):
                arg_evdata = fevents[arg_id]
                if arg_evdata['is_flat_ev']:
                    levels.append(current_level + 1)
                else:
                    arg_level = count_nested_ev_level(arg_evdata, fevents, current_level + 1)
                    levels.append(arg_level)

        level = max(levels)
        return level


def extract_events(events0, entities1):
    """Extract event data"""

    nflat = 0
    n1nested = 0
    nevents = 0

    events1 = OrderedDict()
    for pmid in events0:
        events = events0[pmid]['data']
        idsE = events0[pmid]['ids']
        entities = entities1['pmids'][pmid]['data']
        ev2_ = OrderedDict()

        nevents += len(idsE)

        # Read event data
        for idE in events:
            event = events[idE]
            args_data = event['args_data']

            nestedE = OrderedDict()
            if event['is_nested_ev']:
                n1nested += 1
                for idnE in event['nested_events']:
                    nE = events[idnE]
                    nestedE[idnE] = nE
            else:
                nflat += 1

            event['nested_events_info'] = nestedE

            argTypes = []
            argEntities = []
            if event['args_num'] > 0:
                for arg in args_data:
                    typeR = arg[0]
                    typeR = ''.join([i for i in typeR if not i.isdigit()])
                    eid = arg[1]
                    if eid in entities:
                        typeT = entities[eid]['type']
                        typeArg = typeR + '->' + typeT
                        eArg = typeR + '->' + eid
                    else:
                        typeT = 'E'
                        typeArg = (typeR, typeT)
                        eArg = (typeR, eid)
                    argTypes.append(typeArg)
                    argEntities.append(eArg)

            event['args_types'] = argTypes
            event['args_entities'] = argEntities
            ev2_[idE] = event

        # Process nested events
        for idE in ev2_:
            event = ev2_[idE]
            nestedE2 = False
            if event['is_nested_ev']:
                argsTypes = event['args_types']
                argsTypes2 = []
                for xx, arg in enumerate(event['args_data']):
                    typeR = arg[0]
                    eid = arg[1]
                    typeArg = argsTypes[xx]
                    if eid not in entities:
                        nEvent = events[eid]
                        if nEvent['is_nested_ev']:
                            typeArg = (typeArg[0], 'nestedEV')
                            nestedE2 = True
                        else:
                            if nEvent['is_zeroArg']:
                                typenEvent = ('Nested1', nEvent['trtype'], ['None'])
                            else:
                                typenEvent = ('Nested1', nEvent['trtype'], nEvent['args_types'])

                            typeArg = (typeR, typenEvent)

                    argsTypes2.append(typeArg)

                event['args_types'] = argsTypes2
            event['is_nested_ev_level2'] = nestedE2
            ev2_[idE] = event

        events1[pmid] = ev2_

    evNums = OrderedDict()
    evNums['ev_num'] = nevents
    evNums['ev_flat'] = nflat
    evNums['nested_level1'] = n1nested

    events3 = OrderedDict()
    events3['pmids'] = events1
    events3['evNum'] = evNums

    return events3


def string2pair(st):
    """Parse line to event structure"""

    pairs = []

    pairs0 = st.split('+')
    for pair in pairs0:
        if '0' in pair:
            pair0 = pair.split('0')
            pairs.append(pair0)
        elif '1' in pair:
            pair0 = pair.split('1')
            pairs.append(pair0)
        elif '2' in pair:
            pair0 = pair.split('2')
            pairs.append(pair0)
        elif '3' in pair:
            pair0 = pair.split('3')
            pairs.append(pair0)

    return pairs


def count_structures(structs0):
    """Event structure"""

    for typeTR, structs in structs0.items():

        # store structures by each trigger type
        structs_counts = dict(collections.Counter(structs))
        structs_data = OrderedDict()

        for struct, count in structs_counts.items():
            pairs = string2pair(struct)
            # structs_data[struct] = [pairs, count]
            structs_data[struct] = [pairs]

        # store structure data
        structs0[typeTR] = structs_data

    return structs0


def extract_trigger_structures(events1, entities1):
    """Event structure by trigger type"""

    structs0 = collections.defaultdict(list)
    structs1 = collections.defaultdict(list)

    n_events = 0
    n_1events = 0

    for pmid in events1['pmids']:
        events = events1['pmids'][pmid]
        entities = entities1['pmids'][pmid]['data']

        for idE in events:
            event = events[idE]
            trtype = event['trtype']
            args_data = event['args_data']

            n_events += 1

            # nested event
            if event['is_nested_ev']:
                n_1events += 1
                trtype = event['trtype']
                args_data = event['args_data']

                args_type = ''
                for pair in args_data:
                    if len(pair) > 0:
                        typeR = pair[0]

                        # event argument
                        A2 = pair[1]

                        # argument is entity: flat
                        if A2 in entities:
                            typeA2 = entities[A2]['type']
                            type1 = typeR + '0' + typeA2

                        # argument is event: nested
                        else:
                            typeA2 = events[A2]['trtype']
                            type1 = typeR + '1' + typeA2

                    else:
                        type1 = 'None' + '0' + trtype
                    if len(args_type) > 0:
                        args_type += '+'
                    args_type += type1
                    event['args_type'] = args_type

                structs1[trtype].append(args_type)

            # flat event
            else:
                args_type = ''
                for pair in args_data:
                    if len(pair) > 0:
                        typeR = pair[0]
                        if pair[1] not in entities:
                            print(pmid, pair[1])
                            continue
                        typeT = entities[pair[1]]['type']
                        type1 = typeR + '0' + typeT
                    else:
                        type1 = 'None' + '0' + trtype
                    if len(args_type) > 0:
                        args_type += '+'
                    args_type += type1

                    event['args_type'] = args_type

                structs0[trtype].append(args_type)

    structs0 = count_structures(structs0)
    structs1 = count_structures(structs1)

    print('events: ', n_events, ' flat events: ', (n_events - n_1events))
    print('nested: ', n_1events)

    return {'structs0': structs0, 'structs1': structs1}, events1
