import os
import random
import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from eval.evaluate import predict

from nets import deepEM
from loader.prepData import prepdata
from loader.prepNN import prep4nn
from utils import utils


def main():
    # read predict config
    # set config path by command line
    inp_args = utils._parsing()
    config_path = getattr(inp_args, 'yaml')

    # set config path manually
    # config_path = 'configs/default.yaml'

    with open(config_path, 'r') as stream:
        pred_params = utils._ordered_load(stream)

    # Fix seed for reproducibility
    os.environ["PYTHONHASHSEED"] = str(pred_params['seed'])
    random.seed(pred_params['seed'])
    np.random.seed(pred_params['seed'])
    torch.manual_seed(pred_params['seed'])

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load pre-trained parameters
    with open(pred_params['saved_params'], "rb") as f:
        parameters = pickle.load(f)

    parameters['predict'] = True

    # Set predict settings value for params
    parameters['gpu'] = pred_params['gpu']
    parameters['batchsize'] = pred_params['batchsize']
    if parameters['gpu'] >= 0:
        device = torch.device("cuda:" + str(parameters['gpu']) if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(parameters['gpu'])
    else:
        device = torch.device("cpu")
    parameters['device'] = device

    # Set evaluation settings
    parameters['test_data'] = pred_params['test_data']

    parameters['bert_model'] = pred_params['bert_model']

    result_dir = pred_params['result_dir']
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    parameters['result_dir'] = pred_params['result_dir']

    # raw text
    parameters['raw_text'] = pred_params['raw_text']
    parameters['ner_predict_all'] = pred_params['raw_text']

    # process data
    test_data = prepdata.prep_input_data(pred_params['test_data'], parameters)
    nntest_data, test_dataloader = read_test_data(test_data, parameters)

    # model
    deepee_model = deepEM.DeepEM(parameters)

    model_path = pred_params['model_path']

    # Load all models
    utils.handle_checkpoints(model=deepee_model,
                             checkpoint_dir=model_path,
                             params={
                                 'device': device
                             },
                             resume=True)

    deepee_model.to(device)

    predict(model=deepee_model,
            result_dir=result_dir,
            eval_dataloader=test_dataloader,
            eval_data=nntest_data,
            g_entity_ids_=test_data['g_entity_ids_'],
            params=parameters)

    # print('Done!')


def read_test_data(test_data, params):
    test = prep4nn.data2network(test_data, 'predict', params)

    if len(test) == 0:
        raise ValueError("Test set empty.")

    test_data = prep4nn.torch_data_2_network(cdata2network=test, params=params, do_get_nn_data=True)
    te_data_size = len(test_data['nn_data']['ids'])

    test_data_ids = TensorDataset(torch.arange(te_data_size))
    test_sampler = SequentialSampler(test_data_ids)
    test_dataloader = DataLoader(test_data_ids, sampler=test_sampler, batch_size=params['batchsize'])
    return test_data, test_dataloader


if __name__ == '__main__':
    main()
