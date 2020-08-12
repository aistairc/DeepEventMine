"""Generating configs for training and evaluating models."""

import os
import sys

sys.path.insert(0, '.')
from utils import utils


def write_config(datapath, config):
    """Write config to file"""

    with open(datapath, 'w') as outfile:
        for key, value in config.items():

            # format
            if key == 'bert_model' or key == 'test_data' or key == 'ev_eval_script_path' or key == 'result_dir' or key == 'gpu':
                outfile.write('\n')

            outfile.write('{}: {}'.format(key, value))
            outfile.write('\n')


def gen_predict_config(predict_config, specific_config, eval_set, config_dir, task, taskdir):
    """For joint prediction"""

    predict_config['test_data'] = ''.join(["data/corpora/", task, "/", eval_set, "/"])
    predict_config['result_dir'] = ''.join([taskdir, 'predict-gold-', eval_set, '/'])

    # overwrite task config
    overwrite_task_config(predict_config, specific_config)

    write_config(os.path.join(config_dir, ''.join(['predict-gold-', eval_set, '.yaml'])), predict_config)


def overwrite_task_config(config, specific_config):
    """Overwrite config for specific task."""

    # add specific task config
    for key, value in specific_config.items():
        if key in config:
            config[key] = value

    return config


def read_specific_config(task):
    """Specific config for specific task."""

    # open specific config
    task_config_path = ''.join(['../configs/', task, '.yaml'])

    specific_config = {}

    # check exist and read config
    if os.path.exists(task_config_path):
        with open(task_config_path, 'r') as stream:
            specific_config = utils._ordered_load(stream)

    return specific_config


def generate_configs(expdir, tasks):
    """Generate configs for all."""

    tasks_ = tasks.split('-')

    for task in tasks_:
        # create experiment dir
        taskdir = os.path.join(expdir, '/'.join([task, '']))
        config_dir = os.path.join(expdir, '/'.join([task, 'configs', '']))
        utils.makedir(config_dir)

        # default setting
        default_config_path = '../configs/default.yaml'
        with open(default_config_path, 'r') as stream:
            default_config = utils._ordered_load(stream)

        # read config for specific task
        specific_config = read_specific_config(task)

        # generate config for each task
        task_config = default_config.copy()
        task_config['ev_eval_script_path'] = task_config['ev_eval_script_path'].replace('cg', task)

        # predict config
        predict_dev_config = task_config.copy()
        gen_predict_config(predict_dev_config, specific_config, 'dev', config_dir, task, taskdir)

        predict_test_config = task_config.copy()
        gen_predict_config(predict_test_config, specific_config, 'test', config_dir, task, taskdir)

    print('Generate configs: Done!')

    return


if __name__ == '__main__':
    generate_configs("experiments/", "cg-ge11-ge13-epi-id-pc-mlee")
    # generate_configs(sys.argv[1], sys.argv[2], sys.argv[3])
