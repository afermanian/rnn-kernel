import os

import torch

import taylor_expansion


class RNNModel(torch.nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, output_channels: int, non_linearity: str = 'tanh',
                 device=torch.device("cpu")):
        """Feedforward RNN, that can be penalized with its RKHS norm.

        :param input_channels: dimension of the data
        :param hidden_channels: size of the hidden state
        :param output_channels: size of the prediction. In a classification setting it is the number of classes.
        :param non_linearity: Activation function, can be 'tanh', 'relu' or 'sigmoid'.
        :param device: device on which the model is stored
        """
        super(RNNModel, self).__init__()
        self.name = 'RNN'

        self.hidden_channels = hidden_channels
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.device = device

        self.non_linearity = non_linearity

        self.readout = torch.nn.Linear(hidden_channels, output_channels)

        self.hidden_state_0 = torch.nn.Parameter(torch.zeros(self.hidden_channels, device=device))

        if non_linearity in ['tanh', 'relu']:
            self.rnn_cell = torch.nn.RNNCell(input_channels, hidden_channels, non_linearity)
            self.rnn_cell.to(device)
            self.weight_ih = self.rnn_cell.weight_ih
            self.weight_hh = self.rnn_cell.weight_hh
            self.bias = self.rnn_cell.bias_ih + self.rnn_cell.bias_hh
        else:
            self.rnn_cell = RNNCell(input_channels, hidden_channels, non_linearity)
            self.rnn_cell.to(device)
            self.weight_ih = self.rnn_cell.weight_ih.weight
            self.weight_hh = self.rnn_cell.weight_hh.weight
            self.bias = self.rnn_cell.weight_ih.bias + self.rnn_cell.weight_hh.bias
        self.rnn_cell.to(device)

    def initialize_rnn(self, batch_size: int) -> torch.Tensor:
        """Initialize the hidden state of the RNN.
        :param batch_size:
        :return: torch.Tensor of shape (batch_size, hidden_channels)
        """
        return torch.cat([self.hidden_state_0.unsqueeze(0)] * batch_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass of the RNN.

        :param inputs:  data, of shape (batch_size, length, input_channels)
        :return: last hidden state of the RNN, torch.Tensor of shape (batch_size, hidden_channels)
        """
        hidden_state = self.initialize_rnn(inputs.shape[0])
        for i in range(inputs.shape[1]):
            hidden_state = self.rnn_cell(inputs[:, i, :], hidden_state) / inputs.shape[1] + hidden_state
        return self.readout(hidden_state)

    def get_kernel_penalization(self, order: int, device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """Computes the norm of the RNN in the RKHS, valid only if non_linearity is tanh or sigmoid.

        :param order:  Order of truncation of the taylor expansion
        :param device: device on which the model is stored
        :return: torch.Tensor of shape (1), norm in the RKHS of the RNN
        """
        hidden_state = torch.cat([self.hidden_state_0, torch.zeros(self.input_channels, device=device)])
        model_jacobian_vectorized = taylor_expansion.iterated_jacobian(self, order, hidden_state, is_sparse=False,
                                                                       device=device)

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
    def __init__(self, input_channels: int, hidden_channels: int, non_linearity: str):
        """Manual implementation of a cell of the RNN, necessary when non_linearity is 'sigmoid' since torch.nn.RNNCell
        accepts only 'tanh' or 'relu' activations

        :param input_channels: dimension of the data
        :param hidden_channels: size of the hidden state
        :param non_linearity: Activation function, can be 'tanh' or 'sigmoid'.
        """
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

    def forward(self, input: torch.Tensor, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.non_linearity(self.weight_hh(hidden_state) + self.weight_ih(input))


def train_penalized_rnn(model: RNNModel, train_dataloader: torch.utils.data.DataLoader, n_epoch: int = 30,
                        save_dir: str = None, verbose: bool = False, reg_lambda: float = None, order: int = 1,
                        device: torch.device = torch.device("cpu"), lr: float = None):
    """Train the RNN, eventually with a kernel penalization

    :param model: instance of RNNMOdel
    :param train_dataloader: training data
    :param n_epoch: number of epochs
    :param save_dir: directory where the trained models are stored
    :param verbose: if True, prints the loss at each epoch
    :param reg_lambda: regularization parameter. If 0., no penalization is used
    :param order: Order of truncation of the taylor expansion
    :param device: device on which the model is stored
    :param lr: learning rate of the optimizer. If None, the default value of torch.optim.Adam is used.
    :return: None
    """
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
                        os.path.join(save_dir, 'rnn_model_{}.pt'.format(epoch)))
        if verbose:
            print('Epoch: {}   Training loss: {}'.format(epoch, criterion.item()))
