"""Read brat format input files."""

import glob
import collections
from collections import OrderedDict


def brat_loader(files_fold, params):
    file_list = glob.glob(files_fold + '*' + '.txt')

    entities = OrderedDict()

    sentences = OrderedDict()

    for filef in sorted(file_list):
        if filef.split("/")[-1].startswith("."):
            continue

        filename = filef.split('/')[-1].split('.txt')[0]
        ffolder = '/'.join(filef.split('/')[:-1]) + '/'

        fentities = OrderedDict()

        idsT = []
        typesT = []
        infoT = OrderedDict()
        termsT = []

        with open(ffolder + filename + '.ann', encoding="UTF-8") as infile:
            for line in infile:

                if line.startswith('T'):
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

            typesT2 = dict(collections.Counter(typesT))

            fentities['data'] = infoT
            fentities['types'] = typesT
            fentities['counted_types'] = typesT2
            fentities['ids'] = idsT
            fentities['terms'] = termsT

        # check empty
        if len(idsT) == 0:
            continue

        else:
            entities[filename] = fentities

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

    return entities, sentences
