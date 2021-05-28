import torch
from sklearn.model_selection import ParameterGrid
from sacred.observers import FileStorageObserver


def total_variation(paths: torch.Tensor) -> torch.Tensor:
    """Computes the total variation of a batch of multidimensional paths.

    :param paths: torch.Tensor of shape (batch, step, channel)
    :return: torch.Tensor of shape (batch)
    """
    paths_shifted = paths[:, 1:, :]
    return torch.sum(torch.norm(paths[:, :-1, :] - paths_shifted, dim=2), dim=1)


def number_of_params(model: torch.nn.Module) -> int:
    """Returns the number of trainable parameters of a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_multiclass_accuracy(pred_y: torch.Tensor, true_y: torch.Tensor) -> float:
    """Returns the average accuracy of predictions for classification.

    :param pred_y: output of the model, of shape (batch, class). For each element of the batch, the predicted class is the argmax of the outputs.
    :param true_y: ground truth, of shape (batch).
    :return: the proportion of data for which the correct class was predicted.
    """
    label_predictions = pred_y.argmax(dim=1)
    prediction_matches = (label_predictions == true_y)
    proportion_correct = prediction_matches.sum().float() / float(true_y.size(0))
    return proportion_correct


def gridsearch(ex, config_grid, save_dir):
    ex.observers.append(FileStorageObserver(save_dir))
    param_grid = list(ParameterGrid(config_grid))
    for params in param_grid:
        ex.run(config_updates=params, info={})



