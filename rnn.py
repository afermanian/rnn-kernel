import os

import torch

import taylor_expansion


class RNNModel(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, non_linearity='tanh',
                 device=torch.device("cpu")):
        super(RNNModel, self).__init__()
        self.name = 'RNN'

        self.hidden_channels = hidden_channels
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.device = device

        self.non_linearity = non_linearity

        self.readout = torch.nn.Linear(hidden_channels, output_channels)
        if non_linearity in ['tanh', 'relu']:
            self.rnn_cell = torch.nn.RNNCell(input_channels, hidden_channels, non_linearity)
            self.weight_ih = self.rnn_cell.weight_ih
            self.weight_hh = self.rnn_cell.weight_hh
            self.bias = self.rnn_cell.bias_ih + self.rnn_cell.bias_hh
        else:
            self.rnn_cell = RNNCell(input_channels, hidden_channels, non_linearity)
            self.weight_ih = self.rnn_cell.weight_ih.weight
            self.weight_hh = self.rnn_cell.weight_hh.weight
            self.bias = self.rnn_cell.weight_ih.bias + self.rnn_cell.weight_hh.bias
        self.rnn_cell.to(device)

    def initialize_rnn(self, batch_size):
        return torch.zeros((batch_size, self.hidden_channels,), device=self.device)

    def forward(self, inputs):
        # inputs is of shape (batch_size, timesteps, input_channels)
        # Initialize hidden state
        hidden_state = self.initialize_rnn(inputs.shape[0])
        for i in range(inputs.shape[1]):
            hidden_state = self.rnn_cell(inputs[:, i, :], hidden_state) / inputs.shape[1] + hidden_state
        return self.readout(hidden_state)

    def get_kernel_penalization(self, order, device=torch.device('cpu')):
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
        hidden_state = torch.zeros(self.hidden_channels + self.input_channels, device=device)
        model_jacobian_vectorized = taylor_expansion.iterated_jacobian(self, order, hidden_state,
                                                                is_sparse=False, device=device)

        # Initialize norm for order 0
        norm = self.readout(hidden_state[:self.hidden_channels]) ** 2

        for i in range(order):
            # Reshape the jacobian to have shape (input_channels * (i+1), hidden_state.shape), then take only the
            # self.hidden_channels first values
            model_jacobian_i = model_jacobian_vectorized[i].flatten(start_dim=1).permute(1, 0)[:, :self.hidden_channels]
            model_jacobian_i = self.readout(model_jacobian_i)
            norm += (torch.norm(model_jacobian_i, dim=0) / (
                    torch.tensor(i + 2, device=device, dtype=torch.float).lgamma().exp())) ** 2
        return torch.mean(norm)


class RNNCell(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, non_linearity):
        super(RNNCell, self).__init__()
        self.hidden_channels = hidden_channels
        self.weight_ih = torch.nn.Linear(input_channels, hidden_channels)
        self.weight_hh = torch.nn.Linear(hidden_channels, hidden_channels)
        if non_linearity == 'tanh':
            self.non_linearity = torch.tanh
        elif non_linearity == 'sigmoid':
            self.non_linearity = torch.sigmoid
        else:
            raise ValueError('The non linearity is not well specified')

    def forward(self, input, hidden_state):
        return self.non_linearity(self.weight_hh(hidden_state) + self.weight_ih(input))


def train_penalized_rnn(model, train_dataloader, n_epoch=30, save_dir=None, verbose=False, reg_lambda=None,
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

    loss = torch.nn.CrossEntropyLoss()
    for epoch in range(n_epoch):
        for X_batch, y_batch in train_dataloader:
            optimizer.zero_grad()
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            y_pred = model.forward(X_batch)
            criterion = loss(y_pred, y_batch)
            if reg_lambda > 0:
                criterion += reg_lambda * model.get_kernel_penalization(order, device=device)
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
