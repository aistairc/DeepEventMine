"""Load data from brat format and process for entity, trigger, relation, events."""
from collections import OrderedDict

from loader.prepData.brat import brat_loader
from loader.prepData.sentence import prep_sentence_offsets, process_input
from loader.prepData.entity import process_etypes, process_tags, process_entities
from loader.prepData.event import extract_events, count_nested_events, extract_trigger_structures


def prep_input_data(files_fold, params):
    # load data from *.ann files
    triggers0, entities0, relations0, events0, sentences0 = brat_loader(files_fold, params)

    # sentence offsets
    sentences1 = prep_sentence_offsets(sentences0)
    if 'pipeline_text_data' in params:
        sent_words = []
        for pmid in sentences0:
            doc_data = params['pipeline_text_data'][pmid]
            for sent, pipe_sent in zip(sentences1['doc_data'][pmid], doc_data):
                sent['words'] = pipe_sent['words']
                sent_words.append(sent['words'])
                sent['offsets'] = pipe_sent['offsets']
        sentences1['sent_words'] = sent_words

    # entity
    entities1 = process_etypes(entities0)  # all entity types
    triggers1 = process_etypes(triggers0)  # all trigger types
    terms0 = process_tags(entities1, triggers1)  # terms, offset, tags, etypes
    input0 = process_entities(entities1, triggers1, sentences1, params, files_fold)

    # event
    count_nested_events(events0)
    events1 = extract_events(events0, entities1)
    structsTR, events2 = extract_trigger_structures(events1, entities1)

    # prepare for training batch data for each sentence
    input1 = process_input(input0, entities0, relations0, events2, params, files_fold)

    #
    print("Missing gold entities:")
    for doc_name, doc in sorted(input0.items(), key=lambda x: x[0]):
        entities = set()
        num_entities_per_doc = 0
        for sentence in doc:
            eids = sentence["eids"]
            entities |= set(eids)
            num_entities_per_doc += len(eids)

        full_entities = set(entities1["pmids"][doc_name]["ids"])
        diff = full_entities.difference(entities)
        if diff:
            print(doc_name, sorted(diff, key=lambda _id: int(_id.replace("T", ""))))

    # entity indices
    g_entity_ids_ = OrderedDict()
    for fid, fdata in entities0.items():
        # get max entity id
        eid_ = [eid for eid in fdata['ids'] if not eid.startswith('TR')]
        ids_ = [int(eid.replace('T', '')) for eid in eid_]
        if len(ids_) > 0:
            max_id = max(ids_)
        else:
            max_id = 0
        eid_.append(max_id)
        g_entity_ids_[fid] = eid_

    return {'entities': entities1, 'triggers': triggers1, 'terms': terms0, 'relations': relations0, 'events': events0,
            'sentences': sentences1, 'input': input1, 'structsTR': structsTR, 'g_entity_ids_': g_entity_ids_}
