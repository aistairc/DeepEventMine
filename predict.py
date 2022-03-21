import os
import random
import pickle
import time

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from eval.evaluate import eval
from model import deepEM
from loader.prepData import prepdata
from loader.prepNN import prep4nn
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

    pred_params = parameters

    # Fix seed for reproducibility
    os.environ["PYTHONHASHSEED"] = str(parameters['seed'])
    random.seed(parameters['seed'])
    np.random.seed(parameters['seed'])
    torch.manual_seed(parameters['seed'])

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load configurations for prediction only
    test_data_dir = parameters['test_data']
    params_dir = parameters['params']
    pipelines = parameters['pipelines']
    t_gpu = parameters['t_gpu']
    t_fp16 = parameters['t_fp16']
    t_batch_size = parameters['t_batch_size']

    rel_eval_script_path = parameters['rel_eval_script_path']

    ev_eval_script_path = parameters['ev_eval_script_path']

    gold_eval = parameters['gold_eval']

    bert_model = parameters['bert_model']

    # Load pre-trained parameters
    with open(params_dir, "rb") as f:
        parameters = pickle.load(f)

    parameters['predict'] = True

    parameters['gpu'] = t_gpu
    parameters['fp16'] = t_fp16
    parameters['batchsize'] = t_batch_size
    if parameters['gpu'] >= 0:
        device = torch.device("cuda:" + str(parameters['gpu']) if torch.cuda.is_available() else "cpu")

        torch.cuda.set_device(parameters['gpu'])
    else:
        device = torch.device("cpu")
    parameters['device'] = device

    # Set evaluation settings
    parameters['test_data'] = test_data_dir
    parameters['rel_eval_script_path'] = rel_eval_script_path
    parameters['ev_eval_script_path'] = ev_eval_script_path

    parameters['gold_eval'] = gold_eval
    parameters['bert_model'] = bert_model
    parameters['pipelines'] = pipelines

    result_dir = pred_params['result_dir']
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    parameters['pipeline_setting'] = result_dir
    parameters['pipe_ner'] = 'pipe_ner/'
    parameters['pipe_rel'] = 'pipe_rel/'
    parameters['pipe_ev'] = 'pipe_ev/'
    pipe_ner = parameters['pipeline_setting'] + parameters['pipe_ner']
    pipe_rel = parameters['pipeline_setting'] + parameters['pipe_rel']
    preprocess_pipe_dir(parameters['test_data'], pipe_ner)
    preprocess_pipe_dir(parameters['test_data'], pipe_rel)

    parameters['result_dir'] = pred_params['result_dir']

    parameters['ner_predict_all'] = pred_params['ner_predict_all']

    if pipelines:

        ner_model_dir = pred_params['ner_model_dir']
        rel_model_dir = pred_params['rel_model_dir']
        ev_model_dir = pred_params['ev_model_dir']

        model_dir = {'NER': ner_model_dir, 'REL': rel_model_dir, 'EV': ev_model_dir}
        data_dir = {'NER': test_data_dir, 'REL': pipe_ner, 'EV': pipe_rel}
        # 1.Run NER model
        print("Start running NER model")
        run_pipeline('NER', model_dir, data_dir, parameters, device)
        # 2.Run REL model
        print("Start running REL model")
        run_pipeline('REL', model_dir, data_dir, parameters, device)
        # 3. Run EV model (final)
        print("Start running EV model")
        run_pipeline('EV', model_dir, data_dir, parameters, device)

    else:
        # 1. process data
        test_data = prepdata.prep_input_data(test_data_dir, parameters)
        test_data, test_dataloader = read_test_data(test_data, parameters)

        # 2. model
        # Init zero model
        deepee_model = deepEM.DeepEM(parameters)

        deepee_model_dir = pred_params['joint_model_dir']

        # Load all models
        utils.handle_checkpoints(model=deepee_model,
                                 checkpoint_dir=deepee_model_dir,
                                 params={
                                     'device': device
                                 },
                                 resume=True)

        deepee_model.to(device)

        # create output directory for results
        result_dir = parameters['result_dir']
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        eval(model=deepee_model,
             eval_dir=parameters['test_data'],
             result_dir=result_dir,
             eval_dataloader=test_dataloader,
             eval_data=test_data,
             params=parameters)

    print('PREDICT: DONE!')

    # calculate running time
    t_end = time.time()
    print('TOTAL RUNNING TIME: {}'.format(utils._humanized_time(t_end - t_start)))


def run_pipeline(model_type, model_dir, test_data_dir, params, device):
    if model_type == 'NER':
        deepee_model = deepEM.DeepEM(params)
        utils.handle_checkpoints(model=deepee_model.NER_layer,
                                 checkpoint_dir=model_dir[model_type] + 'ner_model/',
                                 params={
                                     'device': device
                                 },
                                 resume=True)
        params['pipe_flag'] = 0
    elif model_type == 'REL':
        deepee_model = deepEM.DeepEM(params)
        utils.handle_checkpoints(model=deepee_model.REL_layer,
                                 checkpoint_dir=model_dir[model_type] + 'rel_model/',
                                 params={
                                     'device': device
                                 },
                                 resume=True)
        utils.handle_checkpoints(model=deepee_model.NER_layer,
                                 checkpoint_dir=model_dir[model_type] + 'ner_model/',
                                 params={
                                     'device': device
                                 },
                                 resume=True)
        params['pipe_flag'] = 1
    elif model_type == 'EV':
        deepee_model = deepEM.DeepEM(params)
        utils.handle_checkpoints(model=deepee_model.EV_layer,
                                 checkpoint_dir=model_dir[model_type] + 'ev_model/',
                                 params={
                                     'device': device
                                 },
                                 resume=True)
        utils.handle_checkpoints(model=deepee_model.REL_layer,
                                 checkpoint_dir=model_dir[model_type] + 'rel_model/',
                                 params={
                                     'device': device
                                 },
                                 resume=True)
        utils.handle_checkpoints(model=deepee_model.NER_layer,
                                 checkpoint_dir=model_dir[model_type] + 'ner_model/',
                                 params={
                                     'device': device
                                 },
                                 resume=True)
        params['pipe_flag'] = 2
    test_data = prepdata.prep_input_data(test_data_dir[model_type], params)
    test_data, test_dataloader = read_test_data(test_data, params)
    deepee_model.to(device)

    # create output directory for results
    result_dir = 'results/' + params['experiment_name'] + '/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    eval(model=deepee_model,
         eval_dir=test_data_dir['NER'],
         result_dir=result_dir,
         eval_dataloader=test_dataloader,
         eval_data=test_data,
         params=params)


def read_test_data(test_data, params):
    test, test_events_map = prep4nn.data2network(test_data, 'predict', params)

    if len(test) == 0:
        raise ValueError("Test set empty.")

    test_data = prep4nn.torch_data_2_network(cdata2network=test, events_map=test_events_map, params=params,
                                             do_get_nn_data=True)
    te_data_size = len(test_data['nn_data']['ids'])

    test_data_ids = TensorDataset(torch.arange(te_data_size))
    test_sampler = SequentialSampler(test_data_ids)
    test_dataloader = DataLoader(test_data_ids, sampler=test_sampler, batch_size=params['batchsize'])
    return test_data, test_dataloader


def preprocess_pipe_dir(test_dir, pipe_dir):
    if not os.path.exists(pipe_dir):
        os.makedirs(pipe_dir)
    else:
        os.system('rm ' + pipe_dir + '*')
    command = 'cp ' + test_dir + '*.txt ' + pipe_dir
    os.system(command)


if __name__ == '__main__':
    main()
