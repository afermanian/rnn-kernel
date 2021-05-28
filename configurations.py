configs = {
    'test': {
        'non_linearity': ['tanh'],
        'batch_size': [32],
        'n_epoch': [10],
        'grid_lambda': [None, [0.1]],
        'order': [3],
        'n_train': [50]
    },

    'spirals_no_penalization_200_epochs': {
        'non_linearity': ['tanh'],
        'batch_size': [64],
        'hidden_channels': [32],
        'n_epoch': [200],
        'grid_lambda': [None],
        'n_val': [1000],
        'length': [100],
        'n_train': [50] * 20,
        'n_test': [1000],
        'order': [3],
        'seed': [None],
        'lr': [0.1],
    },

    'spirals_penalization_200_epochs': {
        'non_linearity': ['tanh'],
        'batch_size': [64],
        'hidden_channels': [32],
        'n_epoch': [200],
        'grid_lambda': [[0.1]],
        'n_val': [1000],
        'length': [100],
        'n_train': [50] * 20,
        'n_test': [1000],
        'order': [3],
        'seed': [None],
        'lr': [0.1]
    },

}
