import torch
from tqdm import tqdm, trange

import os
import pickle

from eval.evaluate import eval
from utils import utils
from utils.utils import debug, path
from utils.utils import (
    extract_scores,
    is_best_epoch,
    write_best_epoch,
)


# try:
#     from apex import amp
# except ImportError:
#     pass


def train(
        train_data_loader,
        dev_data_loader,
        train_data,
        dev_data,
        params,
        model,
        optimizer
):
    is_params_saved = False
    global_steps = 0

    gradient_accumulation_steps = params["gradient_accumulation_steps"]

    ner_prf_dev, rel_prf_dev, ev_prf_dev = [], [], []

    ner_prf_dev_str, ner_prf_dev_sof, rel_prf_dev_str, rel_prf_dev_sof = [], [], [], []

    # create output directory for results
    result_dir = params['result_dir']
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    if params['freeze_ner']:
        for p in model.NER_layer.parameters():
            p.requires_grad = False

    if params['freeze_rel']:
        for p in model.REL_layer.parameters():
            p.requires_grad = False

    if params['freeze_bert']:
        for p in model.NER_layer.bert.parameters():
            p.requires_grad = False

    # Save params:
    if params['save_params']:
        if not is_params_saved:
            saved_params_path = result_dir + params['task_name'] + '.param'
            with open(saved_params_path, "wb") as f:
                pickle.dump(params, f)
            # is_params_saved = True
            print('SAVED PARAMETERS!')

    for epoch in trange(int(params["epoch"]), desc="Epoch"):
        # TRAIN loop
        model.train()
        tr_loss = 0
        nb_tr_steps = 0

        print()
        print(
            "====================================================================================================================")
        print()
        debug(f"[1] Epoch: {epoch}\n")

        for step, batch in enumerate(
                tqdm(train_data_loader, desc="Iteration", leave=False)
        ):

            # Start training batch
            tr_data_ids = batch
            tensors = utils.get_tensors(tr_data_ids, train_data, params)

            ner_preds, rel_preds, ev_preds, loss = model(tensors, epoch)

            if gradient_accumulation_steps > 1:
                loss /= gradient_accumulation_steps

            tr_loss += loss.item()
            nb_tr_steps += 1

            if loss != 0:
                if params["fp16"]:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

            if (step + 1) % params["gradient_accumulation_steps"] == 0:

                optimizer.step()
                optimizer.zero_grad()
                global_steps += 1

                # Clear GPU unused RAM:
                if params['gpu'] >= 0:
                    torch.cuda.empty_cache()

        print()
        debug(f"[2] Train loss: {tr_loss / nb_tr_steps}\n")
        debug(f"[3] Global steps: {global_steps}\n")

        print(
            "+" * 10 + "RUN EVALUATION" + "+" * 10
        )
        ner_score, is_eval_rel, tr_scores, scores, ev_scores = eval(
            model=model,
            eval_dir=params['dev_data'],
            result_dir=result_dir,
            eval_dataloader=dev_data_loader,
            eval_data=dev_data,
            params=params,
            epoch=epoch
        )

        ner_prf_dev.append(
            [
                float("{0:.2f}".format(ner_score[-1][1])),
                float("{0:.2f}".format(ner_score[-1][2])),
                float("{0:.2f}".format(ner_score[-1][3])),
            ]
        )
        ner_prf_dev_str.append(
            [
                float("{0:.2f}".format(scores['NER']['micro']['st_p'])),
                float("{0:.2f}".format(scores['NER']['micro']['st_r'])),
                float("{0:.2f}".format(scores['NER']['micro']['st_f'])),
            ]
        )
        ner_prf_dev_sof.append(
            [
                float("{0:.2f}".format(scores['NER']['micro']['so_p'])),
                float("{0:.2f}".format(scores['NER']['micro']['so_r'])),
                float("{0:.2f}".format(scores['NER']['micro']['so_f'])),
            ]
        )
        extract_scores('DEV NER', ner_prf_dev)
        ner_max_scores = extract_scores('n2c2 ner strict (micro)', ner_prf_dev_str)
        extract_scores('n2c2 ner soft (micro)', ner_prf_dev_sof)

        if is_eval_rel:
            rel_prf_dev.append(
                [
                    float("{0:.2f}".format(tr_scores["micro_p"] * 100)),
                    float("{0:.2f}".format(tr_scores["micro_r"] * 100)),
                    float("{0:.2f}".format(tr_scores["micro_f"] * 100)),
                ]
            )
            rel_prf_dev_str.append(
                [
                    float("{0:.2f}".format(scores['REL']['micro']['st_p'])),
                    float("{0:.2f}".format(scores['REL']['micro']['st_r'])),
                    float("{0:.2f}".format(scores['REL']['micro']['st_f'])),
                ]
            )
            rel_prf_dev_sof.append(
                [
                    float("{0:.2f}".format(scores['REL']['micro']['so_p'])),
                    float("{0:.2f}".format(scores['REL']['micro']['so_r'])),
                    float("{0:.2f}".format(scores['REL']['micro']['so_f'])),
                ]
            )
            extract_scores('DEV REL', rel_prf_dev)
            rel_max_scores = extract_scores('n2c2 rel strict (micro)', rel_prf_dev_str)
            extract_scores('n2c2 rel soft (micro)', rel_prf_dev_sof)
        else:
            rel_prf_dev.append(
                [
                    float("{0:.2f}".format(0)),
                    float("{0:.2f}".format(0)),
                    float("{0:.2f}".format(0)),
                ]
            )
            rel_prf_dev_str.append(
                [
                    float("{0:.2f}".format(0)),
                    float("{0:.2f}".format(0)),
                    float("{0:.2f}".format(0)),
                ]
            )
            rel_prf_dev_sof.append(
                [
                    float("{0:.2f}".format(0)),
                    float("{0:.2f}".format(0)),
                    float("{0:.2f}".format(0)),
                ]
            )
        if len(ev_scores) > 0:
            ev_prf_dev.append([ev_scores["p"], ev_scores["r"], ev_scores["f"]])
            ev_max_scores = extract_scores('DEV EV', ev_prf_dev)
            best_epoch = is_best_epoch(ev_prf_dev)
            if best_epoch:
                write_best_epoch(result_dir)
        else:
            ev_prf_dev.append(
                [
                    float("{0:.2f}".format(0)),
                    float("{0:.2f}".format(0)),
                    float("{0:.2f}".format(0)),
                ]
            )

        # Clear GPU unused RAM:
        if params['gpu'] >= 0:
            torch.cuda.empty_cache()

    # if params['optimize_type'] == 0:
    #     return ner_max_scores
    # elif params['optimize_type'] == 1:
    #     return rel_max_scores
    # else:
    #     return ev_max_scores
