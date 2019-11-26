import argparse
import copy
import json
import logging
import os
import pickle
import pprint
import random
import re
from collections import OrderedDict
from datetime import datetime
from glob import glob

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


def _to_torch_data(arr, max_length, params, padding_idx=-1):
    for e in arr:
        _truncate(e, max_length)
        _padding(e, max_length, padding_idx=padding_idx)
    return _to_tensor(arr, params)


def _truncate(arr, max_length):
    while True:
        total_length = len(arr)
        if total_length <= max_length:
            break
        else:
            arr.pop()


def _padding(arr, max_length, padding_idx=-1):
    while len(arr) < max_length:
        arr.append(padding_idx)


def _to_tensor(arr, params):
    return torch.tensor(arr, device=params['device'])


def path(*paths):
    return os.path.normpath(os.path.join(os.path.dirname(__file__), *paths))


def make_dirs(*paths):
    os.makedirs(path(*paths), exist_ok=True)


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
    return yaml.load(stream, OrderedLoader)


def dumps(obj):
    if isinstance(obj, dict):
        return json.dumps(obj, indent=4, ensure_ascii=False)
    elif isinstance(obj, list):
        return pprint.pformat(obj)
    return obj


def get_max_entity_id(span_terms):
    max_id = 0
    for items in span_terms:
        for item in items.term2id:
            matcher = re.search(r"^T(?!R)\S*?(\d+)(?=\s)", item)
            if matcher:
                max_id = max(max_id, int(matcher.group(1)))
    return max_id


def padding_samples(ids_, token_mask_, attention_mask_, span_indices_, span_labels_, span_labels_match_rel_,
                    entity_masks_, trigger_masks_, params):
    # count max lengths:
    max_seq = 0
    for ids in ids_:
        max_seq = max(max_seq, len(ids))

    max_span_labels = 0
    for span_labels in span_labels_:
        max_span_labels = max(max_span_labels, len(span_labels))

    for idx, (
            ids, token_mask, attention_mask, span_indices, span_labels, span_labels_match_rel, entity_masks,
            trigger_masks) in enumerate(
        zip(
            ids_,
            token_mask_,
            attention_mask_,
            span_indices_,
            span_labels_,
            span_labels_match_rel_,
            entity_masks_,
            trigger_masks_,
        )):
        padding_size = max_seq - len(ids)

        # Zero-pad up to the sequence length
        ids += [0] * padding_size
        token_mask += [0] * padding_size
        attention_mask += [0] * padding_size

        # Padding for span indices and labels
        num_padding_spans = max_span_labels - len(span_labels)

        span_indices += [(-1, -1)] * (num_padding_spans * params["ner_label_limit"])
        span_labels += [np.zeros(params["mappings"]["nn_mapping"]["num_labels"])] * num_padding_spans
        span_labels_match_rel += [-1] * num_padding_spans
        entity_masks += [-1] * num_padding_spans
        trigger_masks += [-1] * num_padding_spans

        assert len(ids) == max_seq
        assert len(token_mask) == max_seq
        assert len(attention_mask) == max_seq
        assert len(span_indices) == max_span_labels * params["ner_label_limit"]
        assert len(span_labels) == max_span_labels
        assert len(span_labels_match_rel) == max_span_labels
        assert len(entity_masks) == max_span_labels
        assert len(trigger_masks) == max_span_labels

    return max_span_labels


def get_tensors(data_ids, data, params):
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

    span_terms = [
        data["nn_data"]["span_terms"][tr_data_id]
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
    span_terms = copy.deepcopy(span_terms)

    max_span_labels = padding_samples(
        ids,
        token_masks,
        attention_masks,
        span_indices,
        span_labels,
        span_labels_match_rel,
        entity_masks,
        trigger_masks,
        params
    )

    # Padding etypes
    etypes = _to_torch_data(etypes, max_span_labels, params)

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
        span_terms,
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
    with open(filename, "w", encoding="UTF-8") as f:
        for line in lines:
            if is_first_line:
                is_first_line = False
            else:
                f.write(linesep)
            f.write(line)
