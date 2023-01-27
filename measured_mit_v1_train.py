import argparse
import sys
import pdb
import comet_ml

from typing import Any, Dict

seed = 1
import torch
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

import torch.backends.cudnn
import torch.cuda
import torch.nn
import torch.utils.data
from torchpack import distributed as dist
from torchpack.callbacks import (InferenceRunner, MeanAbsoluteError, MeanSquaredError, MinSaver,
                                 Saver, SaverRestore)
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs
from torchpack.utils.logging import logger
from tensorboardX import SummaryWriter

from core import builder
from core.trainers import MAPTrainer
from core.callbacks import CometWriter

import numpy as np
np.random.seed(seed)

import random
random.seed(seed)

import os
import subprocess
import json

from get_split_folds import get_folds


def main(config_file=None, fold=None) -> Dict[str, Any]:
    os.unsetenv("COMET_EXPERIMENT_KEY")

    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE', help='config file')
    parser.add_argument('--fold', metavar='INT', help='which fold to use as test')
    parser.add_argument('--seed', metavar='INT', help='random seed to use for folds split')
    parser.add_argument('--fold_file', metavar='DIR', help='temp file to write folds')
    parser.add_argument('--run_eval', metavar='STRING', help='true or false, whether to run eval command', default='true')
    parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    parser.add_argument('--pdb', action='store_true', help='pdb')

    if config_file is not None:
        args = argparse.Namespace(config=config_file, fold=fold, pdb=False, run_dir=None)
        opts = []
    else:
        args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)
    if configs.pdb or args.pdb:
        pdb.set_trace()

    fold_num = int(args.fold)
    split_seed = int(args.seed)
    configs.dataset.custom_split_indices = get_folds(fold_num, split_seed)
    configs.dataset.fold = fold_num
    configs.dataset.seed = split_seed

    if configs.device == 'gpu':
        device = torch.device('cuda')
    elif configs.device == 'cpu':
        device = torch.device('cpu')

    if isinstance(configs.optimizer.lr, str):
        configs.optimizer.lr = eval(configs.optimizer.lr)

    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)

    logger.info(' '.join([sys.executable] + sys.argv))
    logger.info(f'Experiment started: "{args.run_dir}".' + '\n' + f'{configs}')

    # determine whether to use individual beats or ensembled beat
    ensembled = True
    for feat in configs.model.feats + [configs.dataset.target]:
        if 'complete' in feat:
            ensembled = False
            break
    print("ensembled:", ensembled)

    dataset = builder.make_dataset(ensembled)
    dataflow = dict()
    for split in dataset:
        if split == 'train':
            sampler = torch.utils.data.RandomSampler(dataset[split])
            dataflow[split] = torch.utils.data.DataLoader(
                dataset[split],
                batch_size=configs.batch_size,
                sampler=sampler,
                num_workers=configs.workers_per_gpu,
                pin_memory=True)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset[split])
            dataflow[split] = torch.utils.data.DataLoader(
                dataset[split],
                batch_size=configs.batch_size,
                sampler=sampler,
                num_workers=configs.workers_per_gpu,
                pin_memory=True)
    model = builder.make_model()
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Model Size: {total_params}')

    criterion = builder.make_criterion()
    optimizer = builder.make_optimizer(model)
    scheduler = builder.make_scheduler(optimizer)

    trainer = MAPTrainer(model=model,
                         criterion=criterion,
                         optimizer=optimizer,
                         scheduler=scheduler)
    trainer.train_with_defaults(
        dataflow['train'],
        num_epochs=configs.num_epochs,
        callbacks=[
            InferenceRunner(
                dataflow=dataflow['valid'],
                callbacks=[MeanSquaredError(name='error/valid')],
                ),
            InferenceRunner(
                dataflow=dataflow['test'],
                callbacks=[MeanSquaredError(name='error/test')],
                ),
            MinSaver('error/valid'),
            Saver(max_to_keep=1),
            CometWriter(save_dir=args.run_dir, configs=configs),
        ])

    experiment_key = os.environ["COMET_EXPERIMENT_KEY"]
    experiment_name = os.environ["COMET_EXPERIMENT_NAME"]
    print("experiment_key:", experiment_key)
    print("experiment_name:", experiment_name)
    print("run dir:", args.run_dir)

    if args.run_eval == "true":  # run eval command via subprocess
        eval_configs = args.config.replace("train", "eval")
        ensembled_arg = "true" if ensembled else "false"
        eval_command = subprocess.run(["python3", "measured_mit_v1_eval.py", eval_configs,
                                       f"--run_dir={args.run_dir}",
                                       f"--experiment_key={experiment_key}",
                                       "--part=1",
                                       f"--fold={args.fold}",
                                       f"--seed={args.seed}",
                                       f"--ensembled={ensembled_arg}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("The exit code was:", eval_command.returncode)

    os.unsetenv("COMET_EXPERIMENT_KEY")
    os.unsetenv("COMET_EXPERIMENT_NAME")
    fold_info = {'configs': configs,
            'experiment_key': experiment_key,
            'run_dir': args.run_dir,
            'ensembled': ensembled,
            'fold': fold_num,
            'seed': split_seed,
            'experiment_name': experiment_name,
            }

    if args.fold_file is not None:  # write the fold information for this run into the specified fold_file
        with open(args.fold_file, 'w') as f:
            f.write(json.dumps(fold_info))

    # need this as the last print statement to get fold info from stdout in train_fold_wrapper.py
    print(f"FOLD INFO{json.dumps(fold_info)}END OF INFO")
    return fold_info


if __name__ == '__main__':
    main()