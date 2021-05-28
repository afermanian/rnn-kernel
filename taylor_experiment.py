import os
from typing import Dict

import numpy as np
import pandas as pd
import scipy
import torch

import generate_data
import rnn
import taylor_expansion
import utils


def linear_approximation(X: np.array, t: float) -> np.array:
    """Linear approximation of X."""
    assert t >= 0
    assert t <= 1
    nb_steps = len(X) - 1
    k = int(nb_steps * t)
    if t == 1:
        return X[nb_steps]
    return X[k] + nb_steps * (t - k / nb_steps) * (X[k+1]-X[k])


def f(t: float, h: np.array, X: np.array, Wh: np.array, Wi: np.array, b: np.array, non_linearity: str) -> np.array:
    """Evolution function of the RNN cell given to the scipy solver."""
    if non_linearity == 'sigmoid':
        return 1 / (1 + np.exp(-(Wh @ h + Wi @ linear_approximation(X, t) + b)))
    elif non_linearity == 'tanh':
        return np.tanh(Wh @ h + Wi @ linear_approximation(X, t) + b)


def approximation(model: torch.nn.Module, N: int, X: torch.Tensor) -> np.array:
    """Computes the distance between the solution of the CDE with scipy solver and the Taylor expansion truncated at N.

    :param model: RNN model
    :param N: truncation order of the Taylor expansion
    :param X: driving path, of shape (batch_size, length, channels)
    :return: numpy array of the distance between the two solutions.
    """
    hidden_state = torch.randn(model.hidden_channels + model.input_channels)
    hidden_state[-model.input_channels:] = X[0,:-1]
    Wh = model.weight_hh.detach().numpy()
    Wi = model.weight_ih.detach().numpy()
    b = model.bias.detach().numpy()
    ode_result = scipy.integrate.solve_ivp(lambda t,h: f(t, h, X[:,:-1].detach().numpy(), Wh, Wi, b, model.non_linearity), 
                                           (0, 1), 
                                           method='LSODA', 
                                           y0=hidden_state[:-model.input_channels].detach().numpy(),
                                           rtol=10**-12, 
                                           atol=10**-14).y[:,-1]
    _, euler_coeff_sparse = taylor_expansion.model_approximation(model, N, X, hidden_state, is_sparse=True)
    return np.linalg.norm(euler_coeff_sparse[:,:-2].detach().numpy() - np.expand_dims(ode_result, axis=0), axis=1)


def compute_taylor_convergence(experiment_dir: str, config: Dict):
    """Compares the solution of a CDE obtained with a classical solver to its Taylor approximation with signatures for
    various truncations, when the tensor field of the CDE are random RNNs, with either tanh or sigmoid activations, and
    the driving path is a 2d spiral. Saves the results in a dataframe.

    :param experiment_dir: directory where the experiment is saved
    :param config: configuration values
    :return: None
    """
    X, _ = generate_data.generate_spirals(1, config['length'])
    input_channels = X.shape[2] # dimension d
    Xtime = utils.add_time(X / utils.total_variation(X))[0]
    n_classes = 2

    df = pd.DataFrame(columns=['Step N', 'Error', 'Activation', 'Weight L2 norm'])
    for k in range(config['n_realisations']):
        if k % 10 == 0:
            print('Realisation: {}/{}'.format(k, config['n_realisations']))
        for activation in ['sigmoid', 'tanh']:
            model = rnn.RNNModel(input_channels, config['hidden_channels'], output_channels=n_classes,
                                 non_linearity=activation)
            weight = torch.norm(torch.cat([model.weight_hh, model.weight_ih])).detach().numpy()
            result = approximation(model, config['order'], Xtime)
            for n in range(config['order']):
                df = df.append({'Step N': n+1, 'Error': result[n], 'Weight L2 norm': weight, 'Activation': activation},
                               ignore_index=True)
    
    df.to_csv(os.path.join(experiment_dir, 'taylor_convergence.csv'))
