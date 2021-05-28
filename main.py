from configurations import configs
from definitions import *
from generate_data import get_data
from rnn_module import RNNModel
from train_models import train_penalized, train_rnn_validation_set, evaluate_model
from sacred import Experiment
import sys
import torch
from utils import gridsearch

ex = Experiment()

# Configuration
@ex.config
def my_config():
    non_linearity = 'tanh'
    batch_size = 64
    reg_lambda = None
    hidden_channels = 8
    n_epoch = 10
    order = 0
    length = None
    seed = 17
    n_train = 100
    n_test = 100
    lr = None
    save_dir = ''


@ex.main
def my_main(_run: Experiment, non_linearity: str, batch_size: int, reg_lambda: float, hidden_channels: int,
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

    train_dataloader, test_dataloader, output_channels = get_data(
        length, batch_size, random_seed=seed, n_train=n_train, n_test=n_test)

    input_channels = next(iter(train_dataloader))[0].shape[2] # get last dimension of one batch
    _run.log_scalar('input_channels', int(input_channels))
    _run.log_scalar('output_channels', int(output_channels))


    _run.log_scalar('reg_lambda', reg_lambda)
    model = RNNModel(input_channels, hidden_channels, output_channels, non_linearity=non_linearity, device=device)

    print('--------------------------------')
    print(f'FINAL MODEL')
    print('--------------------------------')
    model.to(device)
    train_penalized(model, train_dataloader, n_epoch=n_epoch, verbose=True, reg_lambda=reg_lambda, order=order,
                    save_dir=ex_save_dir, device=device, lr=lr)

    test_acc = evaluate_model(model, test_dataloader, device=device)
    train_acc = evaluate_model(model, train_dataloader, device=device)
    _run.log_scalar('accuracy_test', test_acc.item())
    _run.log_scalar('accuracy_train', train_acc.item())
    return model


if __name__ == '__main__':
    # Check if GPU is used
    print('GPU available: ', torch.cuda.is_available())

    # Run a configuration
    config = configs[str(sys.argv[1])]
    save_dir = RESULTS_DIR + '/{}'.format(str(sys.argv[1]))
    config['save_dir'] = [save_dir]
    gridsearch(ex, config, save_dir)



