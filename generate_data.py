import math
import torch


def generate_spirals(n_samples: int, length: int = 100):
    """

    :param n_samples: number of samples to generate
    :param length: length of the time sampling of the spirals
    :return: X, y data samples. X is a torch.Tensor of shape (n_samples, length, 2), y is of shape (n_samples) with 0/1
    labels
    """
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


def get_data(length: int, batch_size: int, n_train: int = 1000, n_test: int = 1000, random_seed: int = None):
    """Generate train and test dataloaders for the spirals dataset.

    :param length: length of the time sampling of the spirals
    :param batch_size: batch size
    :param n_train: number of training samples
    :param n_test: number of test samples
    :param random_seed: random seed. If None the seed is not fixed
    :return: train_dataloader, test_dataloader, output_channels. The train and test dataloader are of type
    torch.utils.dataset.DataLoader, output_channels is the number of classes, equal to 2 for the spirals.
    """
    if random_seed:
        torch.manual_seed(random_seed)

    X_train, y_train = generate_spirals(n_train, length=length)
    X_test, y_test = generate_spirals(n_test, length=length)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)

    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, test_dataloader, 2
