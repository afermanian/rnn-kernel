import os
import torch
from rnn_module import RNNModel
import utils
import euler_scheme


def train_penalized(model, train_dataloader, n_epoch=30, save_dir=None, verbose=False, reg_lambda=None,
                    order=1, device=torch.device("cpu"), lr=None):
    if lr:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
    else:
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    for epoch in range(n_epoch):
        for X_batch, y_batch in train_dataloader:
            optimizer.zero_grad()
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            y_pred = model.forward(X_batch)
            criterion = model.loss(y_pred, y_batch)
            if reg_lambda is not None:
                criterion += reg_lambda * get_kernel_penalization(model, order, device=device)
            criterion.backward()
            optimizer.step()
            scheduler.step()

        if save_dir is not None:
            grad_dict = {k: v.grad for k, v in zip(model.state_dict(), model.parameters())}
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'loss': criterion,
                        'optimizer': optimizer.state_dict(), 'grad_dict': grad_dict},
                        '{}/rnn_model_{}.pt'.format(save_dir, epoch))
        if verbose:
            print('Epoch: {}   Training loss: {}'.format(epoch, criterion.item()))


def train_rnn_validation_set(train_dataloader, val_dataloader, input_channels, hidden_channels,
                             output_channels, grid_lambda=[1e-05, 1e-04, 1e-03], n_epoch=30, non_linearity='tanh',
                             order=3, device=torch.device('cpu'), lr=None):
    acc = []

    for reg_lambda in grid_lambda:
        print('--------------------------------')
        print(f'REGULARIZATION PARAM {reg_lambda}')
        print('--------------------------------')
        model = RNNModel(input_channels, hidden_channels, output_channels, non_linearity=non_linearity, device=device)
        model.to(device)
        train_penalized(model, train_dataloader, n_epoch=n_epoch, verbose=True, reg_lambda=reg_lambda, order=order,
                        device=device, lr=lr)
        acc.append(evaluate_model(model, val_dataloader, device=device))

    acc = torch.stack(acc)
    best_reg_lambda = grid_lambda[torch.argmax(acc)]
    return best_reg_lambda, acc


def evaluate_model(model, test_dataloader, device=torch.device('cpu')):
    count = 0
    eval = 0.
    for X_test, y_test in test_dataloader:
        X_test, y_test = X_test.to(device), y_test.to(device)
        count += len(y_test)
        y_pred = model.forward(X_test)
        eval += utils.compute_multiclass_accuracy(y_pred, y_test) * len(y_test)
    return eval / count


def get_kernel_penalization(model, order, device=torch.device('cpu')):
    """
    :param order: int
        Order of truncation of the jacobians.
    :param X_0: torch.Tensor, shape (batch_size, input_channels)
        Initial value of the input paths: it is not batched! so all paths should begin at the same point. It is the
        result of X_train[0, 0, :]
    :return: float
        Norm in the RKHS of the RNN, averages over all initial values in X_0 and each class
            (there is one function per class)
    """
    hidden_state = torch.cat([model.hidden_state_0, torch.zeros(model.input_channels, device=device)])
    model_jacobian_vectorized = euler_scheme.iterated_jacobian(model, order, hidden_state, model.non_linearity,
                                                               is_sparse=False, device=device)

    # Initialize norm for order 0
    norm = model.readout(hidden_state[:model.hidden_channels]) ** 2

    for i in range(order):
        # Reshape the jacobian to have shape (input_channels * (i+1), hidden_state.shape), then take only the
        # model.hidden_channels first values
        model_jacobian_i = model_jacobian_vectorized[i].flatten(start_dim=1).permute(1, 0)[:, :model.hidden_channels]
        model_jacobian_i = model.readout(model_jacobian_i)
        norm += (torch.norm(model_jacobian_i, dim=0) / (
                torch.tensor(i + 2, device=device, dtype=torch.float).lgamma().exp())) ** 2
    return torch.mean(norm)

