import os
import random
import time

import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
)

from model import training

from loader.prepData import prepdata
from loader.prepNN import mapping
from loader.prepNN import prep4nn

from bert.optimization import BertAdam
from model import deepEM
from utils import utils


def main():
    # check running time
    t_start = time.time()

    # set config path by command line
    inp_args = utils._parsing()
    config_path = getattr(inp_args, 'yaml')

    # set config path manually
    # config_path = 'configs/default.yaml'

    with open(config_path, 'r') as stream:
        parameters = utils._ordered_load(stream)

    # print config
    utils._print_config(parameters, config_path)

    parameters['learning_rate'] = float(parameters['learning_rate'])

    if parameters['gpu'] >= 0:
        device = torch.device("cuda:" + str(parameters['gpu']) if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(parameters['gpu'])
    else:
        device = torch.device("cpu")

    print('device', device)

    parameters['device'] = device

    # Fix seed for reproducibility
    os.environ["PYTHONHASHSEED"] = str(parameters['seed'])
    random.seed(parameters['seed'])
    np.random.seed(parameters['seed'])
    torch.manual_seed(parameters['seed'])

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Init needed params
    parameters['max_ev_per_batch'] = 0
    parameters['max_ev_per_layer'] = 0
    parameters['max_rel_per_ev'] = 0
    parameters['max_ev_per_tr'] = 0

    # Force predict = False
    parameters['predict'] = False

    # 1. process data
    train_data = prepdata.prep_input_data(parameters['train_data'], parameters)
    dev_data = prepdata.prep_input_data(parameters['dev_data'], parameters)

    # fix bug for mlee
    test_data = prepdata.prep_input_data(parameters['test_data'], parameters)

    # mapping
    parameters = mapping.generate_map(train_data, dev_data, test_data, parameters)  # add test data for mlee
    if len(parameters['mappings']['rel_map']) > 0:
        parameters = mapping.find_ignore_label(parameters)

    # nner:
    parameters['mappings']['nn_mapping'] = utils.gen_nn_mapping(parameters['mappings']['tag_map'],
                                                                parameters['mappings']['tag2type_map'],
                                                                parameters['trTypes_Ids'])

    train, train_events_map = prep4nn.data2network(train_data, 'train', parameters)
    dev, dev_events_map = prep4nn.data2network(dev_data, 'demo', parameters)

    if len(train) == 0:
        raise ValueError("Train set empty.")
    if len(dev) == 0:
        raise ValueError("Test set empty.")

    # For ranking REL labels weight
    parameters['statistics'] = {'rel': np.zeros(parameters['voc_sizes']['rel_size'])}

    train_data = prep4nn.torch_data_2_network(cdata2network=train, events_map=train_events_map, params=parameters,
                                              do_get_nn_data=True)
    dev_data = prep4nn.torch_data_2_network(cdata2network=dev, events_map=dev_events_map, params=parameters,
                                            do_get_nn_data=True)

    trn_data_size = len(train_data['nn_data']['ids'])
    dev_data_size = len(dev_data['nn_data']['ids'])

    train_data_ids = TensorDataset(torch.arange(trn_data_size))
    dev_data_ids = TensorDataset(torch.arange(dev_data_size))
    train_sampler = RandomSampler(train_data_ids)
    train_dataloader = DataLoader(train_data_ids, sampler=train_sampler, batch_size=parameters['batchsize'])
    dev_sampler = SequentialSampler(dev_data_ids)
    dev_dataloader = DataLoader(dev_data_ids, sampler=dev_sampler, batch_size=parameters['batchsize'])

    # 2. model
    model = deepEM.DeepEM(parameters)

    # Continue training joint model
    if not parameters['predict']:
        # Load pre-trained models
        if 'joint_model_dir' in parameters:
            print('Continue training joint model from', parameters['joint_model_dir'])
            utils.handle_checkpoints(model=model,
                                     checkpoint_dir=parameters['joint_model_dir'],
                                     params={
                                         'device': device
                                     },
                                     resume=True)
        if 'ner_model_dir' in parameters:
            print('pre-load NER model from', parameters['ner_model_dir'])
            utils.handle_checkpoints(model=model.NER_layer,
                                     checkpoint_dir=parameters['ner_model_dir'],
                                     params={
                                         'device': device
                                     },
                                     resume=True)

        if 'rel_model_dir' in parameters:
            print('pre-load REL model from', parameters['rel_model_dir'])
            utils.handle_checkpoints(model=model.REL_layer,
                                     checkpoint_dir=parameters['rel_model_dir'],
                                     params={
                                         'device': device
                                     },
                                     resume=True)

        if 'ev_model_dir' in parameters:
            print('pre-load EV model from', parameters['ev_model_dir'])
            utils.handle_checkpoints(model=model.EV_layer,
                                     checkpoint_dir=parameters['ev_model_dir'],
                                     params={
                                         'device': device
                                     },
                                     resume=True)

    # 3. optimizer
    assert (
            parameters['gradient_accumulation_steps'] >= 1
    ), "Invalid gradient_accumulation_steps parameter, should be >= 1."

    parameters['batchsize'] //= parameters['gradient_accumulation_steps']

    num_train_steps = parameters['epoch'] * (
            (trn_data_size - 1) // (parameters['batchsize'] * parameters['gradient_accumulation_steps']) + 1)
    parameters['voc_sizes']['num_train_steps'] = num_train_steps

    model.to(device)

    # Prepare optimizer

    ner_params, rel_params, ev_params = utils.partialize_optimizer_models_parameters(model)
    param_optimizers = ner_params
    optimizer_grouped_parameters = utils.gen_optimizer_grouped_parameters(param_optimizers, "ner", parameters)
    rel_grouped_params = utils.gen_optimizer_grouped_parameters(rel_params, "rel", parameters)
    ev_grouped_params = utils.gen_optimizer_grouped_parameters(ev_params, "ev", parameters)

    if parameters['bert_warmup_lr']:
        t_total = num_train_steps
    else:
        t_total = -1

    optimizer = BertAdam(
        optimizer_grouped_parameters,
        lr=parameters['learning_rate'],
        warmup=parameters['warmup_proportion'],
        t_total=t_total
    )

    optimizer.add_param_group(rel_grouped_params[0])
    optimizer.add_param_group(rel_grouped_params[1])
    optimizer.add_param_group(ev_grouped_params[0])
    optimizer.add_param_group(ev_grouped_params[1])

    if parameters['train']:
        # 4. training

        if parameters['fp16']:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

        training.train(train_data_loader=train_dataloader, dev_data_loader=dev_dataloader,
                       train_data=train_data, dev_data=dev_data, params=parameters, model=model,
                       optimizer=optimizer)

    print('TRAINING: DONE!')

    # calculate running time
    t_end = time.time()
    print('TOTAL RUNNING TIME: {}'.format(utils._humanized_time(t_end - t_start)))

    return


if __name__ == '__main__':

    main()
