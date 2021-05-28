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
    grid_lambda = None
    hidden_channels = 8
    n_epoch = 10
    n_epoch_lambda = None
    order = 0
    save_dir = None
    length = None
    seed = 17
    n_val = 100
    n_train = 100
    n_test = 100
    lr = None


@ex.main
def my_main(_run, non_linearity, batch_size, grid_lambda, hidden_channels, n_epoch, order, save_dir, length, seed,
            n_epoch_lambda, n_train, n_val, n_test, lr):
    if torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu"
    device = torch.device(dev)
    ex_save_dir = save_dir + '/{}'.format(_run._id)

    train_dataloader, val_dataloader, test_dataloader, output_channels = get_data(
        length, batch_size, random_seed=seed, n_train=n_train, n_test=n_test, n_val=n_val)

    input_channels = next(iter(train_dataloader))[0].shape[2] # get last dimension of one batch
    _run.log_scalar('input_channels', int(input_channels))
    _run.log_scalar('output_channels', int(output_channels))

    if grid_lambda is None:
        reg_lambda = None
    elif len(grid_lambda) == 1:
        reg_lambda = grid_lambda[0]
    else:
        if n_epoch_lambda is None:
            n_epoch_lambda = n_epoch
        reg_lambda, metrics_lambda = train_rnn_validation_set(
            train_dataloader, val_dataloader, input_channels, hidden_channels, output_channels,
            grid_lambda=grid_lambda, non_linearity=non_linearity, n_epoch=n_epoch_lambda, device=device, order=order)
        torch.save({'cv_accuracy': metrics_lambda}, ex_save_dir + '/cv_acc_info.pt')

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



