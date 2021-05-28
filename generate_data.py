import math
import torch


def generate_spirals(n_samples, length=100):
    t = torch.linspace(0., 4 * math.pi, length)

    start = torch.rand(n_samples) * 2 * math.pi
    width = torch.rand(n_samples)
    speed = torch.rand(n_samples)
    x_pos = torch.cos(start.unsqueeze(1) + speed.unsqueeze(1) * t.unsqueeze(0)) / (1 + width.unsqueeze(1) * t)

    x_pos[:(n_samples // 2)] *= -1
    y_pos = torch.sin(start.unsqueeze(1) + speed.unsqueeze(1) * t.unsqueeze(0)) / (1 + width.unsqueeze(1) * t)

    X = torch.stack([x_pos, y_pos], dim=2)
    y = torch.zeros(n_samples, dtype=torch.int64)
    y[:(n_samples // 2)] = 1

    perm = torch.randperm(n_samples)

    X = X[perm]
    y = y[perm]

    return X, y


def get_data(length, batch_size, add_time=False, n_train=1000, n_test=1000, n_val=1000, random_seed=None):
    if random_seed:
        torch.manual_seed(random_seed)

    # The noise is added only on the test set!
    X_train, y_train = generate_spirals(n_train, length=length)
    X_test, y_test = generate_spirals(n_test, length=length)
    X_val, y_val = generate_spirals(n_val, length=length)

    if add_time:
        t = torch.linspace(0., 1, X_train.shape[1])
        X_train = torch.cat([t.unsqueeze(0).repeat(X_train.shape[0], 1).unsqueeze(-1), X_train], dim=2)
        X_test = torch.cat([t.unsqueeze(0).repeat(X_test.shape[0], 1).unsqueeze(-1), X_test], dim=2)
        X_val = torch.cat([t.unsqueeze(0).repeat(X_val.shape[0], 1).unsqueeze(-1), X_val], dim=2)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader, 2



