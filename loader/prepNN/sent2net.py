"""Prepare sentence data for training networks."""

import numpy as np


def prep_sentences(data_struct, data_type, params):
    """
        :param data_struct: data
        :param data_type: train/demo/dev/
        :param params: input parameters
        :returns: list of lists with ids
        In train data, replace words with frequency = 1 with UNK id
        In dev/demo data, replace with UNK if not found in train or pre-trained words
    """
    # MAPPINGS
    singlesW = params['singletons']
    words_train = params['words_train']
    uw_prob = params['unk_w_prob']
    if data_type == 'train':
        singlesW = set(singlesW)
        singles_replaced = 0
        words_real = []
        for sid, s in enumerate(data_struct['sentences']):
            ff = []
            for w in s.split():  # for word in sentence
                if w not in params['mappings']['word_map']:
                    ff.append('<UNK>')

                elif (w in singlesW) and (np.random.uniform() < uw_prob):
                    ff.append('<UNK>')
                    singles_replaced += 1
                else:
                    ff.append(w)
            words_real.append(ff)
        if params['stats']:
            print('\tSingletons in train: %d' % len(singlesW))
            print('\tSingleton words replaced with UNK: %d' % singles_replaced)
    else:
        words_real = []
        in_train = 0
        in_pretrain = 0
        nowhere = 0
        words_train = set(words_train)
        for sid, s in enumerate(data_struct['sentences']):
            ff = []
            for w in s.split():
                if w in words_train:
                    in_train += 1
                    ff.append(w)

                else:
                    nowhere += 1
                    ff.append('<UNK>')
            words_real.append(ff)
        if params['stats']:
            print('\tWords found in train: %d' % in_train)
            print('\tWords found in pre-trained only: %d' % in_pretrain)
            print('\tWords not found anywhere: %d' % nowhere)
    return words_real
