import os

import pandas as pd
from sacred import Experiment
import torch

import adversarial
import generate_data
import rnn
import utils

ex = Experiment()

# Configuration
@ex.config
def my_config():
    non_linearity = 'tanh'
    batch_size = 64
    reg_lambda = 0.
    hidden_channels = 8
    n_epoch = 10
    order = 0
    length = 100
    seed = 17
    n_train = 100
    n_test = 100
    lr = None
    save_dir = ''
    adversarial_epsilon = [0.]
    adversarial_steps = 10


@ex.main
def train_model(_run: Experiment, non_linearity: str, batch_size: int, reg_lambda: float, hidden_channels: int,
            n_epoch: int, order: int, length: int, seed: int, n_train: int, n_test: int,
            lr: float, save_dir: str):
    """Main function that executes a run. The results of the run are stored in the files config.json and metrics.json.

    :param _run: run ID
    :param non_linearity: activation function of the RNN must be 'tanh' or 'sigmoid'
    :param batch_size: batch size
    :param reg_lambda: regularization parameter, can be None or float. If None, no regularization is applied.
    :param hidden_channels: size of the hidden state of the RNN
    :param n_epoch: number of training epochs
    :param order: truncation order for computing the norm in the RKHS as a N-step Taylor expansion
    :param length: number of sampling points of the spirals
    :param seed: random seed
    :param n_train: number of training samples
    :param n_test: number of test samples
    :param lr: learning rate
    :param save_dir: path to directory where the experiment is saved
    :return: Trained RNN.
    """
    if torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu"
    device = torch.device(dev)
    ex_save_dir = save_dir + '/{}'.format(_run._id)

    train_dataloader, test_dataloader, output_channels = generate_data.get_data(
        length, batch_size, random_seed=seed, n_train=n_train, n_test=n_test)

    input_channels = next(iter(train_dataloader))[0].shape[2] # get last dimension of one batch
    _run.log_scalar('input_channels', input_channels)
    _run.log_scalar('output_channels', output_channels)

    model = rnn.RNNModel(input_channels, hidden_channels, output_channels, non_linearity=non_linearity, device=device)
    model.to(device)

    rnn.train_penalized_rnn(model, train_dataloader, n_epoch=n_epoch, verbose=True, reg_lambda=reg_lambda, order=order,
                    save_dir=ex_save_dir, device=device, lr=lr)

    test_acc = utils.evaluate_model(model, test_dataloader, device=device)
    train_acc = utils.evaluate_model(model, train_dataloader, device=device)
    _run.log_scalar('accuracy_test', test_acc.item())
    _run.log_scalar('accuracy_train', train_acc.item())
    return model


def compute_adversarial_accuracy(experiment_dir, run_nums=None):
    df = utils.get_ex_results(experiment_dir, run_nums)

    df_adv = pd.DataFrame(columns=['acc_test_adv', 'epsilon', 'reg_lambda'])
    n_runs = df.shape[0]
    for index, exp in df.iterrows():
        print('Computing adversarial accuracy on experiment {}/{}'.format(index, n_runs))
        for epsilon in exp['adversarial_epsilon']:
            print('epsilon: {}'.format(epsilon))
            model = utils.get_RNN_model(exp)
            X_test, y_test = generate_data.generate_spirals(exp['n_test'], length=exp['length'])
            if epsilon > 0.:
                pgd_1 = adversarial.PGDL2(model, epsilon, steps=exp['adversarial_steps'])
                X_test = pgd_1(X_test, y_test)
            test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=exp['batch_size'])
            acc_test = utils.evaluate_model(model, test_dataloader)
            df_adv = df_adv.append({'epsilon': epsilon, 'acc_test_adv': float(acc_test), 'reg_lambda': r'$\lambda$: ' + str(exp['reg_lambda'])}, ignore_index=True)

    df_adv.to_csv(os.path.join(experiment_dir, 'adversarial_accuracy.csv'))
