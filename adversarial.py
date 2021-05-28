# Disclaimer: this code is a slight modification of https://github.com/albietz/kernel_reg

import torch


def l2_project(X: torch.Tensor, r: float):
    """Project data X onto l2 ball of radius r.

    :param X: batch of data, of shape (batch_size, length, input_channels), to project onto a L2 ball
    :param r: radius of the ball
    :return: torch.Tensor X, the projected data
    """
    n = X.shape[0]
    norms = X.data.view(n, -1).norm(dim=1).view(n, 1, 1)
    X.data *= norms.clamp(0., r) / norms
    return X


class PGDL2(object):
    """Projected gradient descent with l2 perturbations."""
    def __init__(self, model: torch.nn.Module, epsilon: float, step_size: float = None, steps: int = 5,
                 rand: bool = False):
        """

        :param model: trained model over which to generate adversarial examples
        :param epsilon: radius of the ball for the perturbations
        :param step_size: size of the gradient step
        :param steps: number of gradient steps
        :param rand: if True, the noise is initialized randomly, otherwise it is initialized at zero
        """
        self.model = model
        self.epsilon = epsilon
        self.steps = steps
        self.rand = rand
        if step_size is not None:
            self.step_size = step_size
        else:
            self.step_size = 1.5 * epsilon / steps

    def __call__(self, ims: torch.Tensor, labels: torch.Tensor):
        """

        :param ims: input data, tensor of shape (batch_size, lenght, input_channels)
        :param labels: labels of the data, tensor of shape (batch_size)
        :return: torch.Tensor of adversarial examples, same shape as ims.
        """
        n = ims.shape[0]
        if self.rand:
            deltas = torch.randn_like(ims, requires_grad=True)
            deltas.data *= self.epsilon
            l2_project(deltas, self.epsilon)
        else:
            deltas = torch.zeros_like(ims, requires_grad=True)
        for step in range(self.steps):
            if deltas.grad is not None:
                deltas.grad.zero_()
            preds = self.model(ims + deltas)
            loss = -torch.nn.CrossEntropyLoss()(preds, labels)
            loss.backward()

            # normalized (constant step-length) gradient step
            deltas.data.sub_(self.step_size * deltas.grad / deltas.grad.view(n, -1).norm(dim=1).view(n, 1, 1).clamp(min=1e-7))

            # projection on L2 ball
            l2_project(deltas, self.epsilon)

        return (ims + deltas).detach()
