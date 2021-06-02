configs_taylor_exp = {
    'test': {
        'n_realisations': 10,
        'order': 4,
        'hidden_channels': 2,
        'length': 100
    },

    'final': {
        'n_realisations': 10**3,
        'order': 5,
        'hidden_channels': 2,
        'length': 100
    }
}


configs_adversarial_exp = {
    'test': {
        'non_linearity': ['tanh'],
        'batch_size': [32],
        'n_epoch': [10],
        'reg_lambda': [0., 0.1],
        'order': [3],
        'n_train': [50],
        'adversarial_epsilon': [[0., 0.2]],
        'adversarial_steps': [5]
    },

    'spirals_penalization': {
        'non_linearity': ['tanh'],
        'batch_size': [64],
        'hidden_channels': [32],
        'n_epoch': [200],
        'reg_lambda': [0., 0.1],
        'length': [100],
        'n_train': [50] * 20,
        'n_test': [1000],
        'order': [3],
        'lr': [0.1],
        'adversarial_epsilon': [[0., 0.2, 0.4, 0.6, 0.8, 1.0]],
        'adversarial_steps': [50]
    },
}
