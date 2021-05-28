import torch


class RNNModel(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, non_linearity='tanh'):
        super(RNNModel, self).__init__()
        self.name = 'RNN'

        self.hidden_channels = hidden_channels
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.non_linearity = non_linearity

        self.readout = torch.nn.Linear(hidden_channels, output_channels)
        self.rnn_cell = RNNCell(input_channels, hidden_channels, non_linearity)

        self.weight_ih = self.rnn_cell.weight_ih.weight
        self.weight_hh = self.rnn_cell.weight_hh.weight
        self.bias = self.rnn_cell.weight_ih.bias + self.rnn_cell.weight_hh.bias

    def initialize_rnn(self, batch_size):
        return torch.zeros((batch_size, self.hidden_channels))

    def forward(self, inputs):
        # inputs is of shape (batch_size, timesteps, input_channels)
        # Initialize hidden state
        hidden_state = self.initialize_rnn(inputs.shape[0])
        for i in range(inputs.shape[1]):
            hidden_state = self.rnn_cell(inputs[:, i, :], hidden_state) / inputs.shape[1] + hidden_state
        return self.readout(hidden_state)


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

