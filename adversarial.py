import torch


def l2_project(X, r):
    '''project data X onto l2 ball of radius r.'''
    n = X.shape[0]
    norms = X.data.view(n, -1).norm(dim=1).view(n, 1, 1)
    X.data *= norms.clamp(0., r) / norms
    return X


class PGDL2(object):
    '''PGD with l2 perturbations.'''
    def __init__(self, model, epsilon, step_size=None, steps=5, rand=False):
        self.model = model
        self.epsilon = epsilon
        self.steps = steps
        self.rand = rand
        if step_size is not None:
            self.step_size = step_size
        else:
            self.step_size = 1.5 * epsilon / steps

    def __call__(self, ims, labels):
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
            loss = -self.model.loss(preds, labels)
            loss.backward()

            # normalized (constant step-length) gradient step
            deltas.data.sub_(self.step_size * deltas.grad / deltas.grad.view(n, -1).norm(dim=1).view(n, 1, 1).clamp(min=1e-7))

            # projection on L2 ball
            l2_project(deltas, self.epsilon)

        return (ims + deltas).detach()



