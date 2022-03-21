"""Read brat format input files."""

import glob
import collections
from collections import OrderedDict
import os


def brat_loader(files_fold, params):
    file_list = glob.glob(files_fold + '*' + '.txt')

    triggers = OrderedDict()
    entities = OrderedDict()
    relations = OrderedDict()
    events = OrderedDict()
    sentences = OrderedDict()

    for filef in sorted(file_list):
        if filef.split("/")[-1].startswith("."):
            continue
        filename = filef.split('/')[-1].split('.txt')[0]
        ffolder = '/'.join(filef.split('/')[:-1]) + '/'



        # store data for each document
        ftriggers = OrderedDict()
        fentities = OrderedDict()
        frelations = OrderedDict()
        fevents = OrderedDict()

        idsTR = []
        typesTR = []
        infoTR = OrderedDict()
        termsTR = []

        idsT = []
        typesT = []
        infoT = OrderedDict()
        termsT = []

        idsR = []
        typesR = []
        infoR = OrderedDict()

        idsE = []
        infoE = OrderedDict()
        infoM = OrderedDict()

        # # check empty file, otherwise, create an empty file to fix bug pipeline (temporarily)
        # filepath = ffolder + filename + '.ann'
        # if not os.path.isfile(filepath):
        #     with open(filepath, 'w') as f:
        #         print('EMPTY FILE: ', filepath)

        with open(ffolder + filename + '.ann', encoding="UTF-8") as infile:
            for line in infile:

                if line.startswith('TR'):
                    line = line.rstrip().split('\t')
                    trId = line[0]
                    tr1 = line[1].split()
                    trType = tr1[0]
                    pos1 = tr1[1]
                    pos2 = tr1[2]
                    text = line[2]

                    idsTR.append(trId)
                    typesTR.append(trType)
                    trigger_info = OrderedDict()
                    trigger_info['id'] = trId
                    trigger_info['type'] = trType
                    trigger_info['pos1'] = pos1
                    trigger_info['pos2'] = pos2
                    trigger_info['text'] = text
                    infoTR[trId] = trigger_info
                    termsTR.append([trId, trType, pos1, pos2, text])

                elif line.startswith('T'):
                    line = line.rstrip().split('\t')
                    eid = line[0]
                    e1 = line[1].split()
                    etype = e1[0]
                    pos1 = e1[1]
                    pos2 = e1[2]
                    text = line[2]

                    idsT.append(eid)
                    typesT.append(etype)
                    ent_info = OrderedDict()
                    ent_info['id'] = eid
                    ent_info['type'] = etype
                    ent_info['pos1'] = pos1
                    ent_info['pos2'] = pos2
                    ent_info['text'] = text
                    infoT[eid] = ent_info
                    termsT.append([eid, etype, pos1, pos2, text])

                elif line.startswith('R'):
                    line = line.rstrip().split('\t')
                    idR = line[0]
                    typeR = line[1].split()[0]
                    typeR = ''.join([i for i in typeR if not i.isdigit()])
                    args = line[1].split()[1:]
                    arg1id = args[0].split(':')[1]
                    arg2id = args[1].split(':')[1]

                    trig2 = False
                    trig1 = False
                    if arg1id.startswith('TR') and arg2id.startswith('TR'):
                        trig2 = True
                        trig1 = True
                    elif arg1id.startswith('TR'):
                        trig1 = True

                    r_info = OrderedDict()
                    r_info['id'] = idR
                    r_info['type'] = typeR
                    r_info['arg1id'] = arg1id
                    r_info['arg2id'] = arg2id
                    r_info['2trigger'] = trig2
                    r_info['1trigger'] = trig1

                    idsR.append(idR)
                    typesR.append(typeR)
                    infoR[idR] = r_info

                elif line.startswith('E'):
                    line = line.rstrip().split('\t')
                    idE = line[0]
                    args = line[1].split()
                    tr1 = args[0].split(':')
                    trType = tr1[0]
                    trId = tr1[1]
                    args_num = len(args) - 1

                    nestedEv_ = []
                    args2 = []
                    args_ids = []
                    for xx, arg in enumerate(args[1:]):
                        role, eid = arg.split(':')
                        role = ''.join([i for i in role if not i.isdigit()])
                        args2.append((role, eid))
                        args_ids.append(eid)
                        if eid.startswith('E'):
                            nestedEv_.append(eid)

                    zeroArg = False
                    if len(args2) == 0:
                        args2 = [()]
                        zeroArg = True

                    if len(nestedEv_) > 0:
                        evArg = True
                    else:
                        evArg = False

                    idsE.append(idE)
                    e_info = OrderedDict()
                    e_info['id'] = idE
                    e_info['trid'] = trId
                    e_info['trtype'] = trType
                    e_info['args_num'] = args_num
                    e_info['args_data'] = args2
                    e_info['is_zeroArg'] = zeroArg
                    e_info['is_nested_ev'] = evArg
                    e_info['nested_events'] = nestedEv_
                    e_info['is_flat_ev'] = len(nestedEv_) == 0
                    e_info['args_ids'] = args_ids

                    e_info['modality'] = 'non-modality'

                    infoE[idE] = e_info

                elif line.startswith('M'):
                    line = line.rstrip().split('\t')
                    modals = line[1].split(' ')
                    idev = modals[1]
                    modal_type = modals[0]
                    infoM[idev] = modal_type

            typesTR2 = dict(collections.Counter(typesTR))
            typesT2 = dict(collections.Counter(typesT))
            typesR2 = dict(collections.Counter(typesR))

            ftriggers['data'] = infoTR
            ftriggers['types'] = typesTR
            ftriggers['counted_types'] = typesTR2
            ftriggers['ids'] = idsTR
            ftriggers['terms'] = termsTR

            fentities['data'] = infoT
            fentities['types'] = typesT
            fentities['counted_types'] = typesT2
            fentities['ids'] = idsT
            fentities['terms'] = termsT

            frelations['data'] = infoR
            frelations['types'] = typesR
            frelations['ids'] = idsR
            frelations['counted_types'] = typesR2

            for evid, modal_type in infoM.items():
                infoE[evid]['modality'] = modal_type

            fevents['data'] = infoE
            fevents['ids'] = idsE

        # check empty
        if len(idsT) == 0 and not params['raw_text']:
            continue

        else:
            entities[filename] = fentities
            triggers[filename] = ftriggers
            relations[filename] = frelations
            events[filename] = fevents

            lowerc = params['lowercase']
            with open(ffolder + filename + '.txt', encoding="UTF-8") as infile:
                lines = []
                for line in infile:
                    line = line.strip()
                    if len(line) > 0:
                        if lowerc:
                            line = line.lower()
                        lines.append(line)
                sentences[filename] = lines

    return triggers, entities, relations, events, sentences
