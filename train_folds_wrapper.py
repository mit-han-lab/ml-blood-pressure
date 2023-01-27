import argparse
import os
import comet_ml
import json
import tempfile
import subprocess

NUM_FOLDS = 5


def run_folds():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', metavar='FILE', help='config file')
    parser.add_argument('--seed', metavar='INT', help='random seed for folds split')

    args, opts = parser.parse_known_args()

    fold_info = {i: None for i in range(NUM_FOLDS)}
    # start all fold runs
    train_commands = []
    fold_files = {i: tempfile.NamedTemporaryFile() for i in range(NUM_FOLDS)}

    if "attn" in args.config:  # train transformer 2-3 at a time b/c not enough gpu
        print("starting attn, half folds")
        folds_to_run = NUM_FOLDS//2
    else:
        folds_to_run = NUM_FOLDS

    for fold in range(folds_to_run):
        print(f"fold file name {fold}:", fold_files[fold].name)
        train_commands.append(subprocess.Popen(["python3", "measured_mit_v1_train.py", args.config, f"--fold={fold}",
                                                f"--seed={args.seed}", f"--fold_file={fold_files[fold].name}",
                                                "--run_eval=false"],
                                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT))

    print(f"initiated {folds_to_run} folds")

    # wait for fold runs to finish and capture fold info
    eval_commands = []
    for fold in range(folds_to_run):
        output, err = train_commands[fold].communicate()
        print(f"output fold {fold}:", output)

        # read fold run info from temp file
        output_str = fold_files[fold].read()
        print("output", output_str)
        fold_info[fold] = json.loads(output_str)
        print(fold, fold_info[fold])
        fold_files[fold].close()

        # initiate eval commands
        eval_configs = args.config.replace("train", "eval")
        ensembled_arg = "true" if fold_info[fold]['ensembled'] else "false"  # note: is ensembled_arg a str or bool
        eval_commands.append(subprocess.Popen(["python3", "measured_mit_v1_eval.py", eval_configs,
                                       f"--run_dir={fold_info[fold]['run_dir']}",
                                       f"--experiment_key={fold_info[fold]['experiment_key']}",
                                       "--part=1",
                                       f"--fold={fold_info[fold]['fold']}",
                                       f"--seed={fold_info[fold]['seed']}",
                                       f"--ensembled={ensembled_arg}"],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))

    # wait for eval commands to finish
    for fold in range(folds_to_run):
        eval_commands[fold].wait()
        print(f"finished eval fold {fold}")
        print(f"fold {fold} eval returncode:", eval_commands[fold].returncode)

    # run second half of folds (if attn model)
    if folds_to_run < NUM_FOLDS:
        for fold in range(folds_to_run, NUM_FOLDS):
            print(f"fold file name {fold}:", fold_files[fold].name)
            train_commands.append(subprocess.Popen(["python3", "measured_mit_v1_train.py", args.config, f"--fold={fold}",
                                                    f"--seed={args.seed}", f"--fold_file={fold_files[fold].name}",
                                                    "--run_eval=false"],
                                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT))

        print("initiated second half")
        # wait for fold runs to finish and capture fold info
        for fold in range(folds_to_run, NUM_FOLDS):
            output, err = train_commands[fold].communicate()
            print(f"output fold {fold}:", output)

            output_str = fold_files[fold].read()
            print("output", output_str)
            fold_info[fold] = json.loads(output_str)
            print(fold, fold_info[fold])
            fold_files[fold].close()

            eval_configs = args.config.replace("train", "eval")
            ensembled_arg = "true" if fold_info[fold]['ensembled'] else "false"  # note: is ensembled_arg a str or bool
            eval_commands.append(subprocess.Popen(["python3", "measured_mit_v1_eval.py", eval_configs,
                                                   f"--run_dir={fold_info[fold]['run_dir']}",
                                                   f"--experiment_key={fold_info[fold]['experiment_key']}",
                                                   "--part=1",
                                                   f"--fold={fold_info[fold]['fold']}",
                                                   f"--seed={fold_info[fold]['seed']}",
                                                   f"--ensembled={ensembled_arg}"],
                                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))

        for fold in range(folds_to_run, NUM_FOLDS):
            eval_commands[fold].wait()
            print(f"finished eval fold {fold}")
            print(f"fold {fold} eval returncode:", eval_commands[fold].returncode)

    return fold_info


def run_summary(fold_info):
    """
    run a summary Comet experiment to get metrics from each fold run and compute averages
    fold_info is a nested dictionary mapping fold number to a dict with the following keys:
    'configs': configs,
    'experiment_key': experiment_key,
    'run_dir': args.run_dir,
    'ensembled': bool,
    'fold': args.fold,
    'experiment_name': experiment_name,
    """
    experiment = comet_ml.Experiment(
        api_key=os.environ['COMET_API_KEY'],
        project_name='healthcare-dev',
    )

    # log model hyperparameters - assumes all params are same across folds
    model_params = dict(fold_info[0]['configs'])
    experiment.log_parameters(model_params)

    all_metrics = dict()  # mapping metric name to list of metric values from each fold
    # gather and log metrics from each fold
    tags = None
    for fold in fold_info:
        fold_expt = comet_ml.APIExperiment(previous_experiment=fold_info[fold]['experiment_key'])
        if tags is None:
            tags = fold_expt.get_tags()

        experiment.log_metric(f'fold{fold}_experiment_name', fold_expt.get_name())
        experiment.log_metric(f'fold{fold}_experiment_url', fold_expt.url)

        # log fold metrics
        metrics = fold_expt.get_metrics()
        part = 1
        metric_names = [f"Abs Bias Mean part{part}",
                        f"Abs Bias Mean test ensembled part{part}",
                        f"Abs Bias Std part{part}",
                        f"Abs Bias Std test ensembled part{part}",
                        f"Bias Mean part{part}",
                        f"Bias Mean test ensembled part{part}",
                        f"Bias Std part{part}",
                        f"Bias Std test ensembled part{part}",
                        f"Loss_part{part}",
                        f"num_params",
                        f"R Squared part{part}",
                        f"RMSE part{part}"]
        for metric in metrics:
            metric_name = metric['metricName']
            if metric_name not in metric_names:
                continue
            try:
                metric_value = float(metric['metricValue'])
            except ValueError:
                metric_value = metric['metricValue']
            experiment.log_metric(f'{metric_name}_fold{fold}', metric_value)
            if metric_name not in all_metrics:
                all_metrics[metric_name] = []
            all_metrics[metric_name].append(metric_value)

        # log fold figures
        fig_list = fold_expt.get_asset_list("image")
        print("Fold", fold)
        print(fig_list)
        for fig in fig_list:
            experiment.log_metric(f'img_{fig["fileName"]}_fold{fold}', fig["link"])

    # compute average metrics
    for metric in all_metrics:
        print("metric", metric, type(all_metrics[metric][0]), all_metrics[metric])
        try:
            avg_val = sum(all_metrics[metric]) / len(all_metrics[metric])
            experiment.log_metric(f'avg_{metric}', avg_val)
            print(f'avg_{metric}', avg_val, all_metrics[metric])
        except TypeError:
            print(f"metric {metric} is not a float")

    # add all expt tags except fold num
    experiment.add_tags([tag for tag in tags if 'fold' not in tag])
    experiment.add_tag("summary")

    print("expt name", experiment.get_name())


if __name__ == '__main__':
    fold_info = run_folds()
    print("finished running folds")
    run_summary(fold_info)
    
