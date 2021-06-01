import json
import os
import shutil

import pandas as pd
from sklearn.model_selection import ParameterGrid
from sacred.observers import FileStorageObserver
import torch

from rnn import RNNModel

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
    config = load_json(os.path.join(loc, 'config.json'))
    return config


def extract_metrics(loc):
    """ Extracts the metrics from the directory. """
    metrics = load_json(os.path.join(loc, 'metrics.json'))

    # Strip of non-necessary entries
    metrics = {key: value['values'] for key, value in metrics.items()}

    return metrics


def extract_run(loc):
    """ Extracts some metrics from the run. """
    run = load_json(os.path.join(loc, 'run.json'))
    try:
        return {'start_time': run['start_time'], 'stop_time': run['stop_time']}
    except:
        return {'start_time': None, 'stop_time': None}


def get_ex_results(experiment_dir, run_nums=None):
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
    frames = []
    if run_nums is None:
        run_nums = [x for x in os.listdir(experiment_dir) if x.isdigit()]

    for run_num in run_nums:
        loc = os.path.join(experiment_dir, run_num)
        config = extract_config(loc)
        metrics = extract_metrics(loc)
        run = extract_run(loc)

        # Trick to handle lists in pandas dataframe.
        config_without_nested_list = {k: v for (k, v) in config.items() if type(v) != list}
        df_config = pd.DataFrame.from_dict(config_without_nested_list, orient='index').T
        for k, v in config.items():
            if type(v) == list:
                df_config[k] = [v]
        df_metrics = pd.DataFrame.from_dict(metrics, orient='index').T
        df_run = pd.DataFrame.from_dict(run, orient='index').T
        df = pd.concat([df_config, df_metrics, df_run], axis=1)
        df.index = [int(run_num)]
        df['save_dir'] = os.path.join(config['save_dir'], str(run_num))
        frames.append(df)

    # Concat for a full frame
    df = pd.concat(frames, axis=0, sort=True)
    df.sort_index(inplace=True)
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['stop_time'] = pd.to_datetime(df['stop_time'])
    df['running_time'] = (df['stop_time'] - df['start_time']).dt.total_seconds()
    return df


def get_RNN_model(experiment):
    checkpoint = torch.load(os.path.join(experiment['save_dir'], 'rnn_model_{}.pt'.format(experiment['n_epoch']-1)), map_location=torch.device('cpu'))
    model = RNNModel(int(experiment['input_channels']), experiment['hidden_channels'],
                     int(experiment['output_channels']), experiment['non_linearity'])
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def total_variation(paths: torch.Tensor) -> torch.Tensor:
    """Computes the total variation of a batch of multidimensional paths.

    :param paths: torch.Tensor of shape (batch, step, channel)
    :return: torch.Tensor of shape (batch)
    """
    paths_shifted = paths[:, 1:, :]
    return torch.sum(torch.norm(paths[:, :-1, :] - paths_shifted, dim=2), dim=1)


def number_of_params(model: torch.nn.Module) -> int:
    """Returns the number of trainable parameters of a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_multiclass_accuracy(pred_y: torch.Tensor, true_y: torch.Tensor) -> float:
    """Returns the average accuracy of predictions for classification.

    :param pred_y: output of the model, of shape (batch, class). For each element of the batch, the predicted class is the argmax of the outputs.
    :param true_y: ground truth, of shape (batch).
    :return: the proportion of data for which the correct class was predicted.
    """
    label_predictions = pred_y.argmax(dim=1)
    prediction_matches = (label_predictions == true_y)
    proportion_correct = prediction_matches.sum().float() / float(true_y.size(0))
    return proportion_correct


def gridsearch(ex, config_grid, save_dir):
    ex.observers.append(FileStorageObserver(save_dir))
    param_grid = list(ParameterGrid(config_grid))
    for params in param_grid:
        ex.run(config_updates=params, info={})


def evaluate_model(model, test_dataloader, device=torch.device('cpu')):
    count = 0
    eval = 0.
    for X_test, y_test in test_dataloader:
        X_test, y_test = X_test.to(device), y_test.to(device)
        count += len(y_test)
        y_pred = model.forward(X_test)
        eval += compute_multiclass_accuracy(y_pred, y_test) * len(y_test)
    return eval / count

def add_time(paths: torch.Tensor) -> torch.Tensor:
    """Adds a time channel to a batch of multidimensional paths.

    :param paths: torch.Tensor of shape (batch, step, channel)
    :return: torch.Tensor of shape (batch, step, channel+1), where the last channel is the time normalized between 0 and 1.
    """
    t = torch.linspace(0., 1, paths.shape[1])
    paths = torch.cat([paths, t.unsqueeze(0).repeat(paths.shape[0], 1).unsqueeze(-1)], dim=2)
    return paths

def clean_dir(dir):
    if os.path.exists(dir) and os.path.isdir(dir):
        print('Removing previous runs')
        shutil.rmtree(dir)
    if not os.path.exists(dir):
        os.makedirs(dir)