configs = {
    'test': {
        'non_linearity': ['tanh'],
        'batch_size': [32],
        'n_epoch': [10],
        'reg_lambda': [None, 0.1],
        'order': [3],
        'n_train': [50]
    },

    'spirals_penalization': {
        'non_linearity': ['tanh'],
        'batch_size': [64],
        'hidden_channels': [32],
        'n_epoch': [200],
        'reg_lambda': [None, 0.1],
        'length': [100],
        'n_train': [50] * 20,
        'n_test': [1000],
        'order': [3],
        'seed': [None],
        'lr': [0.1],
    },

}
