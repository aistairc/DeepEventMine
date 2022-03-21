import argparse
import copy
import json
import logging
import os
import pickle
import pprint
import random
import re
import shutil
from collections import OrderedDict
from datetime import datetime
from glob import glob
import math

import numpy as np
import torch
# C2T
import yaml

from utils import c2t_utils

logger = logging.getLogger(__name__)


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(math.pi / 2) * (x + 0.044715 * x ** 3)))


def path(*paths):
    return os.path.normpath(os.path.join(os.path.dirname(__file__), *paths))


def make_dirs(*paths):
    os.makedirs(path(*paths), exist_ok=True)


def makedir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def deserialize(filename):
    with open(path(filename), "rb") as f:
        return pickle.load(f)


def serialize(obj, filename):
    make_dirs(os.path.dirname(filename))
    with open(path(filename), "wb") as f:
        pickle.dump(obj, f)


def _parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type=str, required=True, help='yaml file')
    args = parser.parse_args()
    return args


def _parsing_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type=str, required=True, help='yaml file')
    parser.add_argument('--opt', type=str, required=True, help='yaml opt file')
    args = parser.parse_args()
    return args


def _ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    """
        Load parameters from yaml in order
    """

    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))

    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)

    # print(dict(yaml.load(stream, OrderedLoader).items()))

    return yaml.load(stream, OrderedLoader)


def _print_config(config, config_path):
    """Print config in dictionary format"""
    print("\n====================================================================\n")
    print('RUNNING CONFIG: ', config_path)
    print('TIME: ', datetime.now())

    for key, value in config.items():
        print(key, value)

    return


def dicard_invalid_nes(terms, sentences):
    """
    Discard incomplete tokenized entities.
    """
    text = ' '.join(sentences)
    valid_terms = []
    count = 0
    for term in terms:
        start, end = int(term[2]), int(term[3])
        if start == 0:
            if text[end] == ' ':
                valid_terms.append(term)
            else:
                count += 1
            #    print('Context:{}\t{}'.format(text[start:end + 1], term))
        elif text[start - 1] == ' ' and text[end] == ' ':
            valid_terms.append(term)
        else:
            count += 1
        #    print('Context:{}\t{}'.format(text[start-1:end+1], term))
    return valid_terms, count


def _humanized_time(second):
    """
        Returns a human readable time.
    """
    m, s = divmod(second, 60)
    h, m = divmod(m, 60)
    return "%dh %02dm %02ds" % (h, m, s)


def is_best_epoch(prf_):
    fs = []
    for epoch, (p, r, f) in enumerate(prf_):
        fs.append(f)

    if len(fs) == 1:
        return True

    elif max(fs[:-1]) < fs[-1]:
        return True

    else:
        return False


def extract_scores(task, prf_):
    ps = []
    rs = []
    fs = []
    for epoch, (p, r, f) in enumerate(prf_):
        ps.append(p)
        rs.append(r)
        fs.append(f)

    maxp = max(ps)
    maxr = max(rs)
    maxf = max(fs)

    maxp_index = ps.index(maxp)
    maxr_index = rs.index(maxr)
    maxf_index = fs.index(maxf)

    print('TASK: ', task)
    print('precision: ', ps)
    print('recall:    ', rs)
    print('fscore:    ', fs)
    print('best precision/recall/fscore [epoch]: ', maxp, ' [', maxp_index, ']', '\t', maxr, ' [', maxr_index, ']',
          '\t', maxf, ' [', maxf_index, ']')
    print()

    return (maxp, maxr, maxf)


def write_best_epoch(result_dir):
    # best_dir = params['ev_setting'] + params['ev_eval_best']
    best_dir = result_dir + 'ev-best/'

    if os.path.exists(best_dir):
        os.system('rm -rf ' + best_dir)
    # else:
    #     os.makedirs(best_dir)

    current_dir = result_dir + 'ev-last/'

    shutil.copytree(current_dir, best_dir)


def dumps(obj):
    if isinstance(obj, dict):
        return json.dumps(obj, indent=4, ensure_ascii=False)
    elif isinstance(obj, list):
        return pprint.pformat(obj)
    return obj


def debug(*args, **kwargs):
    print(*map(dumps, args), **kwargs)


def get_max_entity_id(span_terms):
    max_id = 0
    for items in span_terms:
        for item in items.term2id:
            matcher = re.search(r"^T(?!R)\S*?(\d+)(?=\s)", item)
            if matcher:
                max_id = max(max_id, int(matcher.group(1)))
    return max_id


def gen_nn_mapping(tag2id_mapping, tag2type_map, trTypes_Ids):
    nn_tr_types_ids = []
    nn_tag_2_type = {}
    tag_names = []
    for tag, _id in tag2id_mapping.items():
        if tag.startswith("I-"):
            continue
        tag_names.append(re.sub("^B-", "", tag))
        if tag2type_map[_id] in trTypes_Ids:
            nn_tr_types_ids.append(len(tag_names) - 1)

        nn_tag_2_type[len(tag_names) - 1] = tag2type_map[_id]

    id_tag_mapping = {k: v for k, v in enumerate(tag_names)}
    tag_id_mapping = {v: k for k, v in id_tag_mapping.items()}

    # For multi-label nner
    assert all(_id == tr_id for _id, tr_id in
               zip(sorted(id_tag_mapping)[1:], nn_tr_types_ids)), "Trigger IDS must be continuous and on the left side"
    return {'id_tag_mapping': id_tag_mapping, 'tag_id_mapping': tag_id_mapping, 'trTypes_Ids': nn_tr_types_ids,
            'tag2type_map': nn_tag_2_type}


def padding_samples_lstm(tokens_, ids_, token_mask_, attention_mask_, span_indices_, span_labels_,
                         span_labels_match_rel_,
                         entity_masks_, trigger_masks_, gtruth_, l2r_, ev_idxs_, params):
    # count max lengths:
    max_seq = 0
    for ids in ids_:
        max_seq = max(max_seq, len(ids))

    max_span_labels = 0
    for span_labels in span_labels_:
        max_span_labels = max(max_span_labels, len(span_labels))

    for idx, (
            tokens, ids, token_mask, attention_mask, span_indices, span_labels, span_labels_match_rel, entity_masks,
            trigger_masks, gtruth, l2r, ev_idxs) in enumerate(
        zip(tokens_,
            ids_,
            token_mask_,
            attention_mask_,
            span_indices_,
            span_labels_,
            span_labels_match_rel_,
            entity_masks_,
            trigger_masks_,
            gtruth_,
            l2r_,
            ev_idxs_)):
        padding_size = max_seq - len(ids)

        tokens += ["<pad>"] * padding_size

        # Zero-pad up to the sequence length
        ids += [0] * padding_size
        token_mask += [0] * padding_size
        attention_mask += [0] * padding_size

        # Padding for gtruth and l2r
        # gtruth = np.pad(gtruth, (
        #     (0, max_span_labels - len(span_indices)), (0, max_span_labels - len(span_indices))),
        #                 'constant', constant_values=-1)

        # l2r = np.pad(l2r,
        #              ((0, max_span_labels - len(span_indices)),
        #               (0, max_span_labels - len(span_indices))),
        #              'constant', constant_values=-1)

        # Padding for span indices and labels
        num_padding_spans = max_span_labels - len(span_labels)

        span_indices += [(-1, -1)] * (num_padding_spans * params["ner_label_limit"])
        span_labels += [np.zeros(params["mappings"]["nn_mapping"]["num_labels"])] * num_padding_spans
        span_labels_match_rel += [-1] * num_padding_spans
        entity_masks += [-1] * num_padding_spans
        trigger_masks += [-1] * num_padding_spans

        # ev_idxs = np.pad(ev_idxs, (0, params['max_span_labels'] - len(ev_idxs)), 'constant', constant_values=-1)
        # ev_idxs = np.array(ev_idxs)

        gtruth_[idx] = gtruth
        l2r_[idx] = l2r
        ev_idxs_[idx] = ev_idxs

        assert len(ids) == max_seq
        assert len(token_mask) == max_seq
        assert len(attention_mask) == max_seq
        assert len(span_indices) == max_span_labels * params["ner_label_limit"]
        assert len(span_labels) == max_span_labels
        assert len(span_labels_match_rel) == max_span_labels
        assert len(entity_masks) == max_span_labels
        assert len(trigger_masks) == max_span_labels
        # assert len(gtruth_[idx][0]) == max_span_labels
        # assert len(l2r_[idx][0]) == max_span_labels

    return max_span_labels


def padding_samples(ids_, token_mask_, attention_mask_, span_indices_, span_labels_, span_labels_match_rel_,
                    entity_masks_, trigger_masks_, gtruth_, l2r_, ev_idxs_, params):
    # count max lengths:
    max_seq = 0
    for ids in ids_:
        max_seq = max(max_seq, len(ids))

    max_span_labels = 0
    for span_labels in span_labels_:
        max_span_labels = max(max_span_labels, len(span_labels))

    for idx, (
            ids, token_mask, attention_mask, span_indices, span_labels, span_labels_match_rel, entity_masks,
            trigger_masks, gtruth, l2r, ev_idxs) in enumerate(
        zip(
            ids_,
            token_mask_,
            attention_mask_,
            span_indices_,
            span_labels_,
            span_labels_match_rel_,
            entity_masks_,
            trigger_masks_,
            gtruth_,
            l2r_,
            ev_idxs_)):
        padding_size = max_seq - len(ids)

        # Zero-pad up to the sequence length
        ids += [0] * padding_size
        token_mask += [0] * padding_size
        attention_mask += [0] * padding_size

        # Padding for gtruth and l2r
        # gtruth = np.pad(gtruth, (
        #     (0, max_span_labels - len(span_indices)), (0, max_span_labels - len(span_indices))),
        #                 'constant', constant_values=-1)

        # l2r = np.pad(l2r,
        #              ((0, max_span_labels - len(span_indices)),
        #               (0, max_span_labels - len(span_indices))),
        #              'constant', constant_values=-1)

        # Padding for span indices and labels
        num_padding_spans = max_span_labels - len(span_labels)

        span_indices += [(-1, -1)] * (num_padding_spans * params["ner_label_limit"])
        span_labels += [np.zeros(params["mappings"]["nn_mapping"]["num_labels"])] * num_padding_spans
        span_labels_match_rel += [-1] * num_padding_spans
        entity_masks += [-1] * num_padding_spans
        trigger_masks += [-1] * num_padding_spans

        # ev_idxs = np.pad(ev_idxs, (0, params['max_span_labels'] - len(ev_idxs)), 'constant', constant_values=-1)
        # ev_idxs = np.array(ev_idxs)

        gtruth_[idx] = gtruth
        l2r_[idx] = l2r
        ev_idxs_[idx] = ev_idxs

        assert len(ids) == max_seq
        assert len(token_mask) == max_seq
        assert len(attention_mask) == max_seq
        assert len(span_indices) == max_span_labels * params["ner_label_limit"]
        assert len(span_labels) == max_span_labels
        assert len(span_labels_match_rel) == max_span_labels
        assert len(entity_masks) == max_span_labels
        assert len(trigger_masks) == max_span_labels
        # assert len(gtruth_[idx][0]) == max_span_labels
        # assert len(l2r_[idx][0]) == max_span_labels

    return max_span_labels


def partialize_optimizer_models_parameters(model):
    """
    Partialize entity, relation and event models parameters from optimizer's parameters
    """
    ner_params = list(model.NER_layer.named_parameters())
    rel_params = list(model.REL_layer.named_parameters())
    ev_params = list(model.EV_layer.named_parameters())

    return ner_params, rel_params, ev_params


def gen_optimizer_grouped_parameters(param_optimizers, name, params):
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    if not params['bert_warmup_lr']:
        lr = float(params['ner_lr'])
        if name == 'rel':
            lr = float(params['rel_lr'])
        if name == 'ev':
            lr = float(params['ev_lr'])
    else:
        lr = params['learning_rate']

    optimizer_grouped_parameters = [
        {
            "name": name,
            "params": [
                p
                for n, p in param_optimizers
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
            "lr": lr
        },
        {
            "name": name,
            "params": [
                p
                for n, p in param_optimizers
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": lr
        },
    ]

    return optimizer_grouped_parameters


def prepare_optimizer_parameters(optimizer, rel_params, ev_params, conf_params, epoch):
    if not conf_params['skip_ner']:
        if epoch == conf_params['ner_epoch'] + 1:
            print("Adding optimizer's REL model params")
            rel_grouped_params = gen_optimizer_grouped_parameters(rel_params, "rel", conf_params)
            optimizer.add_param_group(rel_grouped_params[0])
            optimizer.add_param_group(rel_grouped_params[1])
    if not conf_params['skip_rel']:
        if epoch == conf_params['rel_epoch'] + 1:
            print("Adding optimizer's EV model params")
            ev_grouped_params = gen_optimizer_grouped_parameters(ev_params, "ev", conf_params)
            optimizer.add_param_group(ev_grouped_params[0])
            optimizer.add_param_group(ev_grouped_params[1])
    else:
        pass


def get_tensors(data_ids, data, params):
    # for lstm
    if params['use_lstm']:
        tokens = [
            data["nn_data"]["tokens"][tr_data_id]
            for tr_data_id in data_ids[0].tolist()
        ]
    else:
        tokens = []

    ids = [
        data["nn_data"]["ids"][tr_data_id]
        for tr_data_id in data_ids[0].tolist()
    ]
    token_masks = [
        data["nn_data"]["token_mask"][tr_data_id]
        for tr_data_id in data_ids[0].tolist()
    ]
    attention_masks = [
        data["nn_data"]["attention_mask"][tr_data_id]
        for tr_data_id in data_ids[0].tolist()
    ]
    span_indices = [
        data["nn_data"]["span_indices"][tr_data_id]
        for tr_data_id in data_ids[0].tolist()
    ]
    span_labels = [
        data["nn_data"]["span_labels"][tr_data_id]
        for tr_data_id in data_ids[0].tolist()
    ]
    span_labels_match_rel = [
        data["nn_data"]["span_labels_match_rel"][tr_data_id]
        for tr_data_id in data_ids[0].tolist()
    ]

    entity_masks = [
        data["nn_data"]["entity_masks"][tr_data_id]
        for tr_data_id in data_ids[0].tolist()
    ]
    trigger_masks = [
        data["nn_data"]["trigger_masks"][tr_data_id]
        for tr_data_id in data_ids[0].tolist()
    ]
    gtruths = [
        data["nn_data"]["gtruth"][tr_data_id]
        for tr_data_id in data_ids[0].tolist()
    ]
    l2rs = [
        data["nn_data"]["l2r"][tr_data_id]
        for tr_data_id in data_ids[0].tolist()
    ]

    span_terms = [
        data["nn_data"]["span_terms"][tr_data_id]
        for tr_data_id in data_ids[0].tolist()
    ]

    truth_evs = [
        data["nn_data"]["truth_ev"][tr_data_id]
        for tr_data_id in data_ids[0].tolist()
    ]
    ev_idxs = [
        data["nn_data"]["ev_idxs"][tr_data_id]
        for tr_data_id in data_ids[0].tolist()
    ]
    ev_lbls = [
        data["nn_data"]["ev_lbls"][tr_data_id]
        for tr_data_id in data_ids[0].tolist()
    ]
    etypes = [data["etypes"][tr_data_id] for tr_data_id in data_ids[0].tolist()]

    tokens = copy.deepcopy(tokens)
    ids = copy.deepcopy(ids)
    token_masks = copy.deepcopy(token_masks)
    attention_masks = copy.deepcopy(attention_masks)
    span_indices = copy.deepcopy(span_indices)
    span_labels = copy.deepcopy(span_labels)
    span_labels_match_rel = copy.deepcopy(span_labels_match_rel)
    entity_masks = copy.deepcopy(entity_masks)
    trigger_masks = copy.deepcopy(trigger_masks)
    gtruths = copy.deepcopy(gtruths)
    l2rs = copy.deepcopy(l2rs)
    span_terms = copy.deepcopy(span_terms)
    truth_evs = copy.deepcopy(truth_evs)
    ev_idxs = copy.deepcopy(ev_idxs)
    etypes = copy.deepcopy(etypes)

    # use lstm
    if params['use_lstm']:
        max_span_labels = padding_samples_lstm(
            tokens,
            ids,
            token_masks,
            attention_masks,
            span_indices,
            span_labels,
            span_labels_match_rel,
            entity_masks,
            trigger_masks,
            gtruths,
            l2rs,
            ev_idxs,
            params
        )

    # use bert
    else:
        max_span_labels = padding_samples(
            ids,
            token_masks,
            attention_masks,
            span_indices,
            span_labels,
            span_labels_match_rel,
            entity_masks,
            trigger_masks,
            gtruths,
            l2rs,
            ev_idxs,
            params
        )

    # Padding etypes
    etypes = c2t_utils._to_torch_data(etypes, max_span_labels, params)

    batch_ids = torch.tensor(ids, dtype=torch.long, device=params["device"])
    batch_token_masks = torch.tensor(
        token_masks, dtype=torch.uint8, device=params["device"]
    )
    batch_attention_masks = torch.tensor(
        attention_masks, dtype=torch.long, device=params["device"]
    )
    batch_span_indices = torch.tensor(
        span_indices, dtype=torch.long, device=params["device"]
    )
    batch_span_labels = torch.tensor(
        span_labels, dtype=torch.float, device=params["device"]
    )
    batch_span_labels_match_rel = torch.tensor(
        span_labels_match_rel, dtype=torch.float, device=params["device"]
    )
    batch_entity_masks = torch.tensor(
        entity_masks, dtype=torch.int8, device=params["device"]
    )
    batch_trigger_masks = torch.tensor(
        trigger_masks, dtype=torch.int8, device=params["device"]
    )

    batch_gtruths = gtruths
    batch_l2rs = l2rs
    batch_truth_evs = truth_evs
    batch_ev_idxs = ev_idxs

    return (
        tokens,
        batch_ids,
        batch_token_masks,
        batch_attention_masks,
        batch_span_indices,
        batch_span_labels,
        batch_span_labels_match_rel,
        batch_entity_masks,
        batch_trigger_masks,
        batch_gtruths,
        batch_l2rs,
        span_terms,  # ! << KHOA WAS HERE
        batch_truth_evs,
        batch_ev_idxs,
        ev_lbls,
        etypes,
        max_span_labels
    )


def save_best_fscore(current_params, last_params):
    # This means that we skip epochs having fscore <= previous fscore
    return current_params["fscore"] <= last_params["fscore"]


def save_best_loss(current_params, last_params):
    # This means that we skip epochs having loss >= previous loss
    return current_params["loss"] >= last_params["loss"]


def handle_checkpoints(
        model,
        checkpoint_dir,
        resume=False,
        params={},
        filter_func=None,
        num_saved=-1,
        filename_fmt="${filename}_${epoch}_${fscore}.pt",
):
    if resume:
        # List all checkpoints in the directory
        checkpoint_files = sorted(
            glob(os.path.join(checkpoint_dir, "*.*")), reverse=True
        )

        # There is no checkpoint to resume
        if len(checkpoint_files) == 0:
            return None

        last_checkpoint = None

        if isinstance(resume, dict):
            for previous_checkpoint_file in checkpoint_files:
                previous_checkpoint = torch.load(previous_checkpoint_file, map_location=params['device'])
                previous_params = previous_checkpoint["params"]
                if all(previous_params[k] == v for k, v in resume.items()):
                    last_checkpoint = previous_checkpoint
        else:
            # Load the last checkpoint for comparison
            last_checkpoint = torch.load(checkpoint_files[0], map_location=params['device'])

        print(checkpoint_files[0])

        # There is no appropriate checkpoint to resume
        if last_checkpoint is None:
            return None

        print('Loading model from checkpoint', checkpoint_dir)

        # Restore parameters
        model.load_state_dict(last_checkpoint["model"])
        return last_checkpoint["params"]
    else:
        # Validate params
        varname_pattern = re.compile(r"\${([^}]+)}")
        for varname in varname_pattern.findall(filename_fmt):
            assert varname in params, (
                    "Params must include variable '%s'" % varname
            )

        # Create a new directory to store checkpoints if not exist
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # Make the checkpoint unique
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")

        # Store the current status
        random_states = {}
        random_states["random_state"] = random.getstate()
        random_states["np_random_state"] = np.random.get_state()
        random_states["torch_random_state"] = torch.get_rng_state()

        for device_id in range(torch.cuda.device_count()):
            random_states[
                "cuda_random_state_" + str(device_id)
                ] = torch.cuda.get_rng_state(device=device_id)

        # List all checkpoints in the directory
        checkpoint_files = sorted(
            glob(os.path.join(checkpoint_dir, "*.*")), reverse=True
        )

        # Now, we can define filter_func to save the best model
        if filter_func and len(checkpoint_files):
            # Load the last checkpoint for comparison
            last_checkpoint = torch.load(checkpoint_files[0], map_location=params['device'])

            if timestamp <= last_checkpoint["timestamp"] or filter_func(
                    params, last_checkpoint["params"]
            ):
                return None

        checkpoint_file = (
                timestamp  # For sorting easily
                + "_"
                + varname_pattern.sub(
            lambda m: str(params[m.group(1)]), filename_fmt
        )
        )
        checkpoint_file = os.path.join(checkpoint_dir, checkpoint_file)

        # In case of using DataParallel
        model = model.module if hasattr(model, "module") else model

        print("***** Saving model *****")

        # Save the new checkpoint
        torch.save(
            {
                "model": model.state_dict(),
                "random_states": random_states,
                "params": params,
                "timestamp": timestamp,
            },
            checkpoint_file,
        )

        print("Saved checkpoint as `%s`" % checkpoint_file)

        # Remove old checkpoints
        if num_saved > 0:
            for old_checkpoint_file in checkpoint_files[num_saved - 1:]:
                os.remove(old_checkpoint_file)


def abs_path(*paths):
    return os.path.normpath(
        os.path.join(os.path.dirname(__file__), os.pardir, *paths)
    )


def read_lines(filename):
    with open(abs_path(filename), "r", encoding="UTF-8") as f:
        for line in f:
            yield line.rstrip("\r\n\v")


def write_lines(lines, filename, linesep="\n"):
    is_first_line = True
    # make_dirs(os.path.dirname(filename))
    # os.makedirs(filename)
    # with open(abs_path(filename), "w", encoding="UTF-8") as f:
    with open(filename, "w", encoding="UTF-8") as f:
        for line in lines:
            if is_first_line:
                is_first_line = False
            else:
                f.write(linesep)
            f.write(line)

        # fig bug that not write file with empty prediction
        # if len(lines) == 0:
        #     print(filename)
        #     f.write(linesep)


def list_compare(left, right):
    """
    Failed cases:
    a = np.array([[1,2,3], [4,5,6]])
    b = np.array([[1,2,3], [4,5,6]])
    # => Expected value: True

    a = np.array([[1,2,3], [4,5,6]])
    b = np.array([[1,2,3], np.array([4,5,6])])
    # => Expected value: True

    a = [np.array([1,2,3]), np.array([4,5,6])]
    b = [np.array([1,2,3]), np.array([4,5,6])]
    # => Expected value: True

    a = np.array([[1,2,3], [1,2,3]])
    b = np.array([[1,2,3]])
    # => Expected value: False
    """
    if isinstance(left, np.ndarray):
        left = left.tolist()

    if isinstance(right, np.ndarray):
        right = right.tolist()

    if (isinstance(right, list) and not isinstance(left, list)) or (
            isinstance(left, list) and not isinstance(right, list)):
        return False

    try:
        return left == right
    except:
        try:
            if len(left) == len(right):
                for left_, right_ in zip(left, right):
                    if not list_compare(left_, right_):
                        return False
                return True
            else:
                return False
        except:
            return False


def compare_event_truth(ev, truth):
    if isinstance(ev, list) and isinstance(truth, list):
        ev_args = sort_ev_args(ev, truth)
        if ev_args:
            truth_args = truth[1]
            return compare_args(ev_args, truth_args)
    else:
        return list_compare(ev, truth)


def sort_ev_args(ev, truth):
    if len(ev[0]) != len(truth[0]):
        return None
    ev_can = ev[0]
    truth_can = truth[0]
    ev_args = ev[1]
    ev_sorted_args = []
    for can in truth_can:
        if can in ev_can:
            ev_sorted_args.append(ev_args[ev_can.index(can)])
        else:
            return None

    return ev_sorted_args


def compare_args(ev_args, truth_args):
    if isinstance(ev_args, np.ndarray):
        ev_args = ev_args.tolist()

    if isinstance(truth_args, np.ndarray):
        truth_args = truth_args.tolist()

    if isinstance(ev_args, list) and isinstance(truth_args, list):
        if len(ev_args) != len(truth_args):
            return False
        for ev_arg, truth_arg in zip(ev_args, truth_args):
            if not compare_event_truth(ev_arg, truth_arg):
                return False
        return True
    else:
        return False


def write_annotation_file(
        ann_file, entities=None, triggers=None, relations=None, events=None
):
    lines = []

    def annotate_text_bound(entities):
        for entity in entities.values():
            entity_annotation = "{}\t{} {} {}\t{}".format(
                entity["id"],
                entity["type"],
                entity["start"],
                entity["end"],
                entity["ref"],
            )
            lines.append(entity_annotation)

    if entities:
        annotate_text_bound(entities)

    if triggers:
        annotate_text_bound(triggers)

    if relations:
        for relation in relations.values():
            relation_annotation = "{}\t{} {}:{} {}:{}".format(
                relation["id"],
                relation["role"],
                relation["left_arg"]["label"],
                relation["left_arg"]["id"],
                relation["right_arg"]["label"],
                relation["right_arg"]["id"],
            )
            lines.append(relation_annotation)

    if events:
        for event in events.values():
            event_annotation = "{}\t{}:{}".format(
                event["id"], event["trigger_type"], event["trigger_id"]
            )
            for arg in event["args"]:
                event_annotation += " {}:{}".format(arg["role"], arg["id"])
            lines.append(event_annotation)

    write_lines(lines, ann_file)
