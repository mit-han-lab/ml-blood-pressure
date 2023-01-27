import argparse
import os
import pdb
import comet_ml

seed = 1
import numpy as np
import torch
import random
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.use_deterministic_algorithms(True)

import torch.backends.cudnn
import torch.cuda
import torch.nn
import torch.utils.data
import tqdm
from torchpack.utils import fs, io
from torchpack.utils.config import configs
from torchpack.utils.logging import logger
from torchpack.environ import set_run_dir

from core import builder
from tools import augment, augment_merge, augment_multi, augment_merge_multi
import matplotlib.pyplot as plt
from torchprofile import profile_macs
from sklearn.metrics import r2_score
from get_split_folds import get_folds


def main() -> None:
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE', help='config file')
    parser.add_argument('--run_dir', metavar='DIR', help='run directory')
    parser.add_argument('--fold', metavar='INT', help='fold number to use as testing fold')
    parser.add_argument('--seed', metavar='INT', help='random seed to use for folds split')
    parser.add_argument('--experiment_key', metavar='STRING', help='comet experiment key')
    parser.add_argument('--ensembled', metavar='STRING', help='true or false, whether to use ensembled or individual beat', default='true')
    parser.add_argument('--metric_mode', metavar='STRING', help='mean=use avg target/output, full=use individual', default='full')
    parser.add_argument('--part', metavar='INT', help='dataset part to use for evaluation', default=1)
    args, opts = parser.parse_known_args()

    configs.load(os.path.join(args.run_dir, 'metainfo', 'configs.yaml'))
    model_configs = configs.model.copy()
    target_configs = configs.dataset.target
    dataset_root = configs.dataset.root
    configs.load(args.config, recursive=True)
    configs.update(opts)
    configs.update({'model': model_configs})
    configs['dataset']['target'] = target_configs
    configs['dataset']['root'] = dataset_root

    if args.part is not None:
        configs.dataset.part = int(args.part)

    if configs.pdb:
        pdb.set_trace()

    if args.ensembled == "true":
        ensembled = True
    elif args.ensembled == "false":
        ensembled = False
    else:
        raise ValueError("ensembled argument must be either true or false")

    if configs.device == 'gpu':
        device = torch.device('cuda')
    elif configs.device == 'cpu':
        device = torch.device('cpu')

    set_run_dir(args.run_dir)

    comet_api_key = os.environ["COMET_API_KEY"]
    resume = True

    if resume:
        experiment = comet_ml.ExistingExperiment(
            api_key=comet_api_key,
            experiment_key=args.experiment_key,
        )

    else:
        experiment = comet_ml.Experiment(
            api_key=comet_api_key,
            project_name="healthcare-dev",
        )

    if configs.dataset.name == 'js':
        if configs.batch_size != 1:
            logger.warning(f'For evaluation on JS, batch size will be set to '
                           f'1 due to collate issue of varying length a_whole '
                           f'and v_whole!')
            configs.batch_size = 1
        if configs.eval.mode == 'multi':
            if configs.dataset.augment_setting['cyclic'] is None:
                raise ValueError(
                    f'For multiple beat mode evaluation on JS, '
                    f'must specify the cyclic augmentation settings!')

    if configs.dataset.name == 'pwdb' or configs.dataset.name == 'pwdb_pert' or configs.dataset.name == 'pwdb_measured_v1':
        if configs.eval.mode != 'single':
            logger.warning(f'For evaluation on PWDB, only single beat mode '
                           f'is supported since the dataset only contains '
                           f'single beat data!')
            configs.eval.mode = 'single'

    logger.info(f'Evaluation started: "{args.run_dir}".' + '\n' + f'{configs}')

    split = 'test'

    # optional: can pass in different mean and std for normalization
    # dataset = builder.make_dataset(ensembled, mean=mean, std=std)
    dataset = builder.make_dataset(ensembled)
    mean = dataset['test'].mean
    std = dataset['test'].std
    sampler = torch.utils.data.SequentialSampler(dataset[split])
    dataflow = torch.utils.data.DataLoader(
        dataset[split],
        sampler=sampler,
        batch_size=2500,
        num_workers=configs.workers_per_gpu,
        pin_memory=True)
    model = builder.make_model()
    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Model Size: {total_params}')
    experiment.log_metrics({"num_params": total_params})

    checkpoint_file = 'min-error-valid.pt'
    state_dict = io.load(
        os.path.join(args.run_dir, 'checkpoints', checkpoint_file), map_location=torch.device('cpu'))

    model.load_state_dict(state_dict['model'])

    fs.makedir(os.path.join(args.run_dir, 'plot'))
    with torch.no_grad():
        target_all = None
        output_all = None
        for feed_dict in tqdm.tqdm(dataflow):
            inputs = dict()
            for key, value in feed_dict.items():
                if key in ['area', 'v', 'pwv', 'age', 'comp', 'z0', 'deltat',
                           'pp', 'id', 'bp', 'flow', 'shape', 'weight', 'height', 'gender', 'heartrate',
                           'diameter_complete_avg_beats', 'velocity_complete_avg_beats',
                           'bp_shape_complete_avg_beats', 'area_complete_avg_beats',
                           'bp_shape_complete_min', 'bp_shape_complete_mean', 'bp_shape_complete_max',
                           'velocity_complete_min', 'velocity_complete_mean', 'velocity_complete_max']:
                    if configs.device == 'gpu':
                        inputs[key] = value.cuda()
                    else:
                        inputs[key] = value

            if configs.device == 'gpu':
                targets = feed_dict[configs.dataset.target].cuda(non_blocking=True)
            else:
                targets = feed_dict[configs.dataset.target]

            outputs = model(inputs)

            output_dict = {'outputs': outputs, 'targets': targets}

            outputs = output_dict['outputs']
            targets = output_dict['targets']

            if configs.model.name == 'attn':
                outputs = outputs[0]  # discard attn weights

            if target_all is None:
                target_all = targets.cpu().numpy()
                output_all = outputs.cpu().numpy()
            else:
                target_all = np.concatenate([target_all,
                                             targets.cpu().numpy()])
                output_all = np.concatenate([output_all,
                                             outputs.cpu().numpy()])


        target_all_meanap = target_all * std[configs.dataset.target] + \
            mean[configs.dataset.target]
        output_all_meanap = output_all * std[configs.dataset.target] + \
            mean[configs.dataset.target]

        A = np.vstack([target_all_meanap, np.ones(len(target_all_meanap))]).T
        m, c = np.linalg.lstsq(A, output_all_meanap, rcond=None)[0]
        # best fit line: output = m*target + c
        logger.info(f"orig best fit line for {configs.model.name} part {configs.dataset.part} m: {m} c: {c}")
        experiment.log_metrics({f"orig_bestfit_m_part{configs.dataset.part}": m,
                                f"orig_bestfit_c_part{configs.dataset.part}": c},)

        np.save('target_denorm_all.npy', target_all_meanap)
        np.save('output_denorm_all.npy', output_all_meanap)
        print("targets:", target_all_meanap)
        print("outputs:", output_all_meanap)

        # loss
        logger.info(f'Loss: {np.mean((output_all - target_all) ** 2)}')
        experiment.log_metrics({f'Loss_part{configs.dataset.part}': np.mean((output_all - target_all) ** 2)},)

        # bias
        bias = output_all_meanap - target_all_meanap

        bias_mean = np.mean(bias)
        bias_std = np.std(bias)

        logger.info(f'Bias Mean: {bias_mean:.2f}')
        logger.info(f'Bias Std: {bias_std:.2f}')
        experiment.log_metrics({f'Bias Mean part{configs.dataset.part}': bias_mean,
                                f'Bias Std part{configs.dataset.part}': bias_std},)
        # bias_abs mean and std
        bias_abs_mean = np.mean(np.abs(bias))
        bias_abs_std = np.std(np.abs(bias))

        # RMSE
        rmse = np.sqrt(np.mean(bias ** 2))

        logger.info(f'Abs Bias Mean: {bias_abs_mean:.2f}')
        logger.info(f'Abs Bias Std: {bias_abs_std:.2f}')
        logger.info(f'RMSE: {rmse:.2f}')
        logger.info(f'R Squared: {r2_score(output_all_meanap, target_all_meanap)}')
        experiment.log_metrics({f'Abs Bias Mean part{configs.dataset.part}': bias_abs_mean,
                                f'Abs Bias Std part{configs.dataset.part}': bias_abs_std,
                                f'RMSE part{configs.dataset.part}': rmse,
                                f'R Squared part{configs.dataset.part}': r2_score(output_all_meanap, target_all_meanap)
                                },)
        for i in range(output_all_meanap.shape[0]):
            experiment.log_metrics({f'output_all_meanap_part{configs.dataset.part}': output_all_meanap[i],
                                    f'target_all_meanap_part{configs.dataset.part}': target_all_meanap[i],
                                   }, step=i)

        # get subject indices of splits for plotting
        fold_num = int(args.fold)
        seed = int(args.seed)
        train, val, test = get_folds(fold_num, seed)

        if ensembled:
            # log ensembled metrics
            bias_mean_test = np.mean(bias[test])
            bias_std_test = np.std(bias[test])
            bias_abs_mean_test = np.mean(np.abs(bias[test]))
            bias_abs_std_test = np.std(np.abs(bias[test]))
            logger.info(f'Bias Std test ensembled: {bias_std_test:.2f}')
            logger.info(f'Bias Mean test ensembled: {bias_mean_test:.2f}')
            logger.info(f'Abs Bias Std test ensembled: {bias_abs_std_test:.2f}')
            logger.info(f'Abs Bias Mean test ensembled: {bias_abs_mean_test:.2f}')
            experiment.log_metrics({f'Bias Std test ensembled part{configs.dataset.part}': bias_std_test,
                                    f'Bias Mean test ensembled part{configs.dataset.part}': bias_mean_test,
                                    f'Abs Bias Std test ensembled part{configs.dataset.part}': bias_abs_std_test,
                                    f'Abs Bias Mean test ensembled part{configs.dataset.part}': bias_abs_mean_test},)

            plt.scatter(target_all_meanap[train], output_all_meanap[train], c='blue')
            plt.scatter(target_all_meanap[val], output_all_meanap[val], c='green')
            plt.scatter(target_all_meanap[test], output_all_meanap[test], c='red')
        else:  # beat-to-beat
            # transform test subject indices into beat indices
            train_beats, val_beats, test_beats = [], [], []
            # collect avg target and avg output for each subject
            train_target_avgs, val_target_avgs, test_target_avgs = [], [], []
            train_output_avgs, val_output_avgs, test_output_avgs = [], [], []
            train_color = iter(plt.cm.Blues(np.linspace(0.1, 1, len(train))))
            val_color = iter(plt.cm.Greens(np.linspace(0.2, 1, len(val))))
            test_color = iter(plt.cm.Reds(np.linspace(0.2, 1, len(test))))
            tups = [(train_beats, train, train_color, train_target_avgs, train_output_avgs),
                    (val_beats, val, val_color, val_target_avgs, val_output_avgs),
                    (test_beats, test, test_color, test_target_avgs, test_output_avgs)]
            for beats, subjs, color, target_avgs, output_avgs in tups:
                for subj in subjs:
                    start_ind = sum(dataset.beats_per_subject_indiv[:subj])
                    num_beats = dataset.beats_per_subject_indiv[subj]
                    subj_beats = list(range(start_ind, start_ind + num_beats))
                    beats.extend(subj_beats)
                    target_avgs.append(np.mean(target_all_meanap[subj_beats]))
                    output_avgs.append(np.mean(output_all_meanap[subj_beats]))
                    subj_color = next(color)
                    if args.metric_mode == 'full':
                        plt.scatter(target_all_meanap[subj_beats], output_all_meanap[subj_beats], color=subj_color)


            # train, val, test = train_beats, val_beats, test_beats

            # log full metrics
            bias_mean_test = np.mean(bias[test_beats])
            bias_std_test = np.std(bias[test_beats])
            bias_abs_mean_test = np.mean(np.abs(bias[test_beats]))
            bias_abs_std_test = np.std(np.abs(bias[test_beats]))
            logger.info(f'Bias Std test full: {bias_std_test:.2f}')
            logger.info(f'Bias Mean test full: {bias_mean_test:.2f}')
            logger.info(f'Abs Bias Std test full: {bias_abs_std_test:.2f}')
            logger.info(f'Abs Bias Mean test full: {bias_abs_mean_test:.2f}')
            experiment.log_metrics({f'Bias Std test full part{configs.dataset.part}': bias_std_test,
                                    f'Bias Mean test full part{configs.dataset.part}': bias_mean_test,
                                    f'Abs Bias Std test full part{configs.dataset.part}': bias_abs_std_test,
                                    f'Abs Bias Mean test full part{configs.dataset.part}': bias_abs_mean_test},)

            # log averaged metrics
            bias_test = np.array(test_output_avgs) - np.array(test_target_avgs)
            bias_mean_test = np.mean(bias_test)
            bias_std_test = np.std(bias_test)
            bias_abs_mean_test = np.mean(np.abs(bias_test))
            bias_abs_std_test = np.std(np.abs(bias_test))
            logger.info(f'Bias Std test avged: {bias_std_test:.2f}')
            logger.info(f'Bias Mean test avged: {bias_mean_test:.2f}')
            logger.info(f'Abs Bias Std test avged: {bias_abs_std_test:.2f}')
            logger.info(f'Abs Bias Mean test avged: {bias_abs_mean_test:.2f}')
            experiment.log_metrics({f'Bias Std test avged part{configs.dataset.part}': bias_std_test,
                                    f'Bias Mean test avged part{configs.dataset.part}': bias_mean_test,
                                    f'Abs Bias Std test avged part{configs.dataset.part}': bias_abs_std_test,
                                    f'Abs Bias Mean test avged part{configs.dataset.part}': bias_abs_mean_test},
                                   )
            if args.metric_mode == 'mean':
                plt.scatter(train_target_avgs, train_output_avgs, c='blue')
                plt.scatter(val_target_avgs, val_output_avgs, c='green')
                plt.scatter(test_target_avgs, test_output_avgs, c='red')
        print(target_all_meanap.shape, output_all_meanap.shape)

        lim_min = min(min(target_all_meanap), min(output_all_meanap)) - 2
        lim_max = max(max(target_all_meanap), max(output_all_meanap)) + 2
        plt.xlim([lim_min, lim_max])
        plt.ylim([lim_min, lim_max])
        # plot identity line
        x = np.linspace(lim_min, lim_max, 100)
        y = x  # identity line
        plt.plot(x, y, c='orange')
        y = m*x+c  # best fit line
        plt.plot(x, y, c='purple')

        logger.info(f"output_all_meanap range: {min(output_all_meanap)}, {max(output_all_meanap)}")
        logger.info(f"target_all_meanap range: {min(target_all_meanap)}, {max(target_all_meanap)}")

        plt.xlabel('Ground Truth MAP')
        plt.ylabel('Predicted MAP')

        plt.gca().set_aspect('equal', adjustable='box')

        prefix = "part1norm"
        figure_name = f'plot/{prefix}_{args.config.replace("/", "_")}_{split}_{len(bias)}_scatter_new_{configs.dataset.target}.png'
        experiment.log_figure(figure_name=f"Output vs Target MAP Part {configs.dataset.part}", figure=plt)
        plt.title(f"Output vs Target MAP Part {configs.dataset.part} {args.metric_mode}")
        plt.savefig(os.path.join(
            args.run_dir,
            figure_name,
            ), dpi=1200, bbox_inches='tight'
        )


if __name__ == '__main__':
    main()


