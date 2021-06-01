import os

import numpy as np
import pandas as pd
import scipy
import torch

from rnn import RNNModel
from taylor_expansion import model_approximation
from generate_data import generate_spirals
from utils import total_variation, add_time


def x(X, t):
    assert t >= 0
    assert t <= 1
    k = int(100 * t)
    if t == 1:
        return X[100]
    return X[k] + 100 * (t - k / 100) * (X[k+1]-X[k])


def f(t, h, X, Wh, Wi, b, non_linearity):
    if non_linearity == 'sigmoid':
        return 1 / (1 + np.exp(-(Wh @ h + Wi @ x(X, t) + b)))
    elif non_linearity == 'tanh':
        return np.tanh(Wh @ h + Wi @ x(X, t) + b)


def approximation(model, N, X):
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
    _, euler_coeff_sparse = model_approximation(model, N, X, hidden_state, is_sparse=True)
    return np.linalg.norm(euler_coeff_sparse[:,:-2].detach().numpy() - np.expand_dims(ode_result, axis=0), axis=1)


def compute_taylor_convergence(experiment_dir, config):
    X, _ = generate_spirals(1, 101)
    input_channels = X.shape[2] # dimension d
    Xtime = add_time(X / total_variation(X))[0]
    n_classes = 2

    df = pd.DataFrame(columns=['Step N', 'Error', 'Activation', 'Weight L2 norm'])
    for k in range(config['n_realisations']):
        if k % 10 == 0:
            print('Realisation: {}/{}'.format(k, config['n_realisations']))
        for activation in ['sigmoid', 'tanh']:
            model = RNNModel(input_channels, config['hidden_channels'], output_channels=n_classes, non_linearity=activation)
            weight = torch.norm(torch.cat([model.weight_hh, model.weight_ih])).detach().numpy()
            result = approximation(model, config['order'], Xtime)
            for n in range(config['order']):
                df = df.append({'Step N': n+1, 'Error': result[n], 'Weight L2 norm': weight, 'Activation': activation}, ignore_index=True)
    
    df.to_csv(os.path.join(experiment_dir, 'taylor_convergence.csv'))
