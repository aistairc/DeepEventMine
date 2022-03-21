"""Generating configs for training and evaluating models."""

import os
import sys
import argparse

sys.path.insert(0, '.')
from utils import utils


def write_config(datapath, config):
    """Write config to file"""

    # with open(datapath, 'w') as outfile:
    #     yaml.dump(config, outfile, default_flow_style=False, sort_keys=False)

    with open(datapath, 'w') as outfile:
        for key, value in config.items():

            # format
            if key == 'result_dir':
                outfile.write('\n')
            if key == 'epoch':
                outfile.write('\n')
            if key == 'bert_dim':
                outfile.write('\n')
            if key == 'ner_reduce':
                outfile.write('\n')
            if key == 'seed':
                outfile.write('\n')
            if key == 'ner_label_limit':
                outfile.write('\n')
            if key == 'ev_threshold':
                outfile.write('\n')
            if key == 'use_gold_ner':
                outfile.write('\n')
            if key == 'freeze_bert':
                outfile.write('\n')
            if key == 'ner_epoch_limit':
                outfile.write('\n')
            if key == 'direction':
                outfile.write('\n')
            if key == 'ner_eval_corpus':
                outfile.write('\n')
            if key == 'predict':
                outfile.write('\n')

            outfile.write('{}: {}'.format(key, value))
            outfile.write('\n')


def gen_ner_config(ner_config, task_config, config_dir, taskdir):
    """For entity"""

    ner_config['result_dir'] = ''.join([taskdir, 'ner/'])
    ner_config['ner_model_dir'] = ''.join([taskdir, 'ner/model/'])
    ner_config['save_ner'] = True

    # overwrite task config
    overwrite_task_config(ner_config, task_config)

    write_config(os.path.join(config_dir, 'train-ner.yaml'), ner_config)


def gen_rel_config(rel_config, task_config, config_dir, taskdir):
    """For relation"""

    rel_config['result_dir'] = ''.join([taskdir, 'rel/'])
    rel_config['rel_model_dir'] = ''.join([taskdir, 'rel/model/'])
    rel_config['save_rel'] = True
    rel_config['use_gold_ner'] = True
    rel_config['use_gold_rel'] = False
    rel_config['ner_predict_all'] = False
    rel_config['skip_ner'] = True
    rel_config['ner_epoch'] = -1

    # overwrite task config
    overwrite_task_config(rel_config, task_config)

    write_config(os.path.join(config_dir, 'train-rel.yaml'), rel_config)


def gen_ev_config(ev_config, task_config, config_dir, taskdir):
    """For event"""

    ev_config['result_dir'] = ''.join([taskdir, 'ev/'])
    ev_config['ev_model_dir'] = ''.join([taskdir, 'ev/model/'])
    ev_config['save_ev'] = True
    ev_config['ev_nested_epoch'] = 20
    ev_config['modality_epoch'] = 20
    ev_config['use_general_rule'] = True
    ev_config['use_gold_ner'] = True
    ev_config['use_gold_rel'] = True
    ev_config['ner_predict_all'] = False
    ev_config['skip_ner'] = True
    ev_config['skip_rel'] = True
    ev_config['ner_epoch'] = -1
    ev_config['rel_epoch'] = -1
    ev_config['freeze_bert'] = True
    ev_config['freeze_ner'] = True
    ev_config['freeze_rel'] = True

    # overwrite task config
    overwrite_task_config(ev_config, task_config)

    write_config(os.path.join(config_dir, 'train-ev.yaml'), ev_config)


def gen_joint_config(joint_config, task_config, config_dir, taskdir):
    """For joint"""

    joint_config['result_dir'] = ''.join([taskdir, 'joint-gold/'])
    joint_config['joint_model_dir'] = ''.join([taskdir, 'joint-gold/model/'])
    joint_config['save_params'] = True
    joint_config['save_all_models'] = True
    joint_config['use_general_rule'] = True
    joint_config['ner_model_dir'] = ''.join([taskdir, 'ner/model/'])
    joint_config['rel_model_dir'] = ''.join([taskdir, 'rel/model/'])
    joint_config['ev_model_dir'] = ''.join([taskdir, 'ev/model/'])
    joint_config['ner_predict_all'] = False
    joint_config['ner_epoch'] = -1
    joint_config['rel_epoch'] = -1
    joint_config['ner_epoch_limit'] = 70
    joint_config['rel_epoch_limit'] = 90
    joint_config['rel_loss_weight_minor'] = 0.001
    joint_config['ev_loss_weight_minor'] = 0.001
    joint_config['ner_loss_weight_minor'] = 0.5
    joint_config['rel_loss_weight_main'] = 0.5
    joint_config['ev_loss_weight_main'] = 0.1

    # overwrite task config
    overwrite_task_config(joint_config, task_config)

    write_config(os.path.join(config_dir, 'train-joint-gold.yaml'), joint_config)


def gen_joint_e2e_config(joint_e2e_config, task_config, config_dir, taskdir):
    """For joint end-to-end"""

    joint_e2e_config['result_dir'] = ''.join([taskdir, 'joint-e2e/'])
    joint_e2e_config['joint_model_dir'] = ''.join([taskdir, 'joint-e2e/model/'])
    joint_e2e_config['ner_predict_all'] = True

    # overwrite task config
    overwrite_task_config(joint_e2e_config, task_config)

    write_config(os.path.join(config_dir, 'train-joint-e2e.yaml'), joint_e2e_config)


def gen_predict_config(args, predict_config, eval_set, config_dir, taskdir):
    """For joint prediction"""

    predict_config['test_data'] = predict_config['test_data'].replace('dev', eval_set)
    set_debug_mode(predict_config, args)
    predict_config['result_dir'] = ''.join([taskdir, 'predict-gold-', eval_set, '/'])
    predict_config['save_params'] = False
    predict_config['joint_model_dir'] = ''.join([taskdir, 'joint-gold/model/'])
    predict_config['params'] = ''.join([taskdir, 'joint-gold/', predict_config['task_name'], '.param'])
    predict_config['predict'] = True
    predict_config['ner_predict_all'] = False

    write_config(os.path.join(config_dir, ''.join(['predict-gold-', eval_set, '.yaml'])), predict_config)


def gen_predict_e2e_config(args, predict_e2e_config, eval_set, config_dir, taskdir):
    """For joint end-to-end prediction"""

    predict_e2e_config['result_dir'] = ''.join([taskdir, 'predict-e2e-', eval_set, '/'])
    set_debug_mode(predict_e2e_config, args)
    predict_e2e_config['joint_model_dir'] = ''.join([taskdir, 'joint-e2e/model/'])
    predict_e2e_config['params'] = ''.join([taskdir, 'joint-e2e/', predict_e2e_config['task_name'], '.param'])
    predict_e2e_config['ner_predict_all'] = True

    write_config(os.path.join(config_dir, ''.join(['predict-e2e-', eval_set, '.yaml'])), predict_e2e_config)


def read_specific_config(task):
    """Specific config for specific task."""

    # open specific config
    task_config_path = ''.join(['configs/', task, '.yaml'])

    specific_config = {}

    # check exist and read config
    if os.path.exists(task_config_path):
        with open(task_config_path, 'r') as stream:
            specific_config = utils._ordered_load(stream)

    return specific_config


def overwrite_task_config(config, specific_config):
    """Overwrite config for specific task."""

    # add specific task config
    for key, value in specific_config.items():
        if key in config:
            config[key] = value

    return config


def set_debug_mode(configs, args):
    if args.debug_mode:
        if "train_data" in configs:
            configs['train_data'] = configs['train_data'].replace('train', "debug")
        if "dev_data" in configs:
            configs['dev_data'] = configs['dev_data'].replace('dev', "debug")
            configs['dev_data'] = configs['dev_data'].replace('test', "debug")
        if "test_data" in configs:
            configs['test_data'] = configs['test_data'].replace('dev', "debug")
            configs['test_data'] = configs['test_data'].replace('test', "debug")
        if "epoch" in configs:
            configs["epoch"] = 2

def generate_configs(args, expdir, task, exp_name):
    """Generate configs for all."""

    # create experiment dir
    taskdir = os.path.join(expdir, '/'.join([task, exp_name, '']))
    config_dir = os.path.join(expdir, '/'.join([task, exp_name, 'configs', '']))
    utils.makedir(config_dir)

    # default exp_name
    default_config_path = 'configs/default.yaml'
    with open(default_config_path, 'r') as stream:
        default_config = utils._ordered_load(stream)

    # read config for specific task
    specific_config = read_specific_config(task)

    # generate config for each task
    task_config = default_config.copy()

    # generate data path
    task_config['train_data'] = ''.join(["data/corpora/", task, "/train/"])
    task_config['dev_data'] = ''.join(["data/corpora/", task, "/dev/"])
    task_config['test_data'] = ''.join(["data/corpora/", task, "/dev/"])

    # debug mode
    set_debug_mode(task_config, args)

    # bert
    task_config['bert_model'] = "data/bert/scibert_scivocab_cased"

    # task specific
    task_config['task_name'] = task
    task_config['ner_eval_corpus'] = task
    task_config['ev_eval_script_path'] = task_config['ev_eval_script_path'].replace('cg', task)
    task_config['rule_dir'] = task_config['rule_dir'].replace('cg', task)

    # ner config
    ner_config = task_config.copy()
    gen_ner_config(ner_config, specific_config, config_dir, taskdir)

    # rel config
    rel_config = task_config.copy()
    gen_rel_config(rel_config, specific_config, config_dir, taskdir)

    # ev config
    ev_config = task_config.copy()
    gen_ev_config(ev_config, specific_config, config_dir, taskdir)

    # joint gold config
    joint_config = task_config.copy()
    gen_joint_config(joint_config, specific_config, config_dir, taskdir)

    # joint end-to-end config (predict entity)
    joint_e2e_config = joint_config.copy()
    gen_joint_e2e_config(joint_e2e_config, specific_config, config_dir, taskdir)

    # predict config
    predict_dev_config = task_config.copy()
    gen_predict_config(args, predict_dev_config, 'dev', config_dir, taskdir)

    predict_test_config = task_config.copy()
    gen_predict_config(args, predict_test_config, 'test', config_dir, taskdir)

    # predict end-to-end config

    predict_e2e_dev_config = predict_dev_config.copy()
    gen_predict_e2e_config(args, predict_e2e_dev_config, 'dev', config_dir, taskdir)

    predict_e2e_test_config = predict_test_config.copy()
    gen_predict_e2e_config(args, predict_e2e_test_config, 'test', config_dir, taskdir)

    print('Generate configs: Done!')

    return


def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir', help='Directory for experiments', type=str, default='experiments')
    parser.add_argument('--task_name', help='Name of task', type=str, default='cg')
    parser.add_argument('--experiment_name', help='Name of this experiment', type=str,
                        default='basic')
    parser.add_argument("--debug_mode", action='store_true',
                        help="Run experiments on a small data for debugging quickly")
    args = parser.parse_args(arguments)

    generate_configs(args, args.experiment_dir, args.task_name, args.experiment_name)

if __name__ == '__main__':
    # generate_configs("experiments/", "cg", "debug_mode")
    main(sys.argv[1:])
