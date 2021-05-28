from definitions import *
import json
import pandas as pd
import torch
import os
from rnn_module import RNNModel


def load_json(path: str):
    """Loads a json object
    Parameters
    ----------
    path: str
        Location of the json file.
    """
    with open(path) as file:
        return json.load(file)


def extract_config(loc):
    """ Extracts the metrics from the directory."""
    config = load_json(loc + '/config.json')
    return config


def extract_metrics(loc):
    """ Extracts the metrics from the directory. """
    metrics = load_json(loc + '/metrics.json')

    # Strip of non-necessary entries
    metrics = {key: value['values'] for key, value in metrics.items()}

    return metrics


def extract_run(loc):
    """ Extracts some metrics from the run. """
    run = load_json(loc + '/run.json')
    try:
        return {'start_time': run['start_time'], 'stop_time': run['stop_time']}
    except:
        return {'start_time': None, 'stop_time': None}


def get_ex_results(dirname, run_nums):
    """Extract all result of a configuration grid.

    Parameters
    ----------
    dirname: str
        Name of the directory where the experiments are stored.

    run_nums: list of strings or 'all'
        Numbers of the runs to load

    Returns
    -------
    df: pandas DataFrame
        Dataframe with all the experiments results
    """
    dirname = RESULTS_DIR + '/' + dirname

    frames = []
    if run_nums == 'all':
        not_in = ['_sources', '.DS_Store']
        run_nums = [x for x in os.listdir(dirname) if x not in not_in]

    for run_num in run_nums:
        loc = dirname + '/' + run_num
        config = extract_config(loc)
        metrics = extract_metrics(loc)
        run = extract_run(loc)

        # Create a config and metrics frame and concat them
        config = {str(k): str(v) for k, v in config.items()}
        df_config = pd.DataFrame.from_dict(config, orient='index').T
        df_metrics = pd.DataFrame.from_dict(metrics, orient='index').T
        df_run = pd.DataFrame.from_dict(run, orient='index').T
        df = pd.concat([df_config, df_metrics, df_run], axis=1)
        df.index = [int(run_num)]
        frames.append(df)

    # Concat for a full frame
    df = pd.concat(frames, axis=0, sort=True)
    df.sort_index(inplace=True)
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['stop_time'] = pd.to_datetime(df['stop_time'])
    df['running_time'] = (df['stop_time'] - df['start_time']).dt.total_seconds()
    return df


def get_ex_best_model_params(dirname, run_num, epoch):
    dirname = RESULTS_DIR + '/' + dirname + '/' + run_num
    checkpoint = torch.load('{}/rnn_model_{}.pt'.format(dirname, epoch), map_location=torch.device('cpu'))
    return checkpoint


def get_training_loss(dirname, run_num, n_epoch=None):
    df = get_ex_results(dirname, [str(run_num)])
    if not n_epoch:
        n_epoch = int(df.iloc[0]['n_epoch'])

    loss = []
    for epoch in range(n_epoch):
        checkpoint = get_ex_best_model_params(dirname, str(run_num), epoch)
        loss.append(checkpoint['loss'])
    return loss


def get_RNN_model(df, experiment, run_num, n_epoch=None, non_linearity='tanh'):
    if n_epoch is None:
        n_epoch = int(df.loc[int(run_num)]['n_epoch']) - 1
    checkpoint = get_ex_best_model_params(experiment, run_num, n_epoch)
    model = RNNModel(int(df.loc[int(run_num)]['input_channels']), int(df.loc[int(run_num)]['hidden_channels']),
                     int(df.loc[int(run_num)]['output_channels']), non_linearity=non_linearity)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

