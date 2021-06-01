from logging import warn
import math
from typing import Callable, List
import warnings

import numpy as np
import signatory
import sparse
import sympy
import torch


def derivative_tanh(m, x):
    # see https://arxiv.org/ftp/arxiv/papers/0903/0903.0117.pdf
    return (-2)**m * (torch.tanh(x) + 1) * \
        sum([math.factorial(k) / (2**k) * int(sympy.functions.combinatorial.numbers.stirling(m, k))
             * (torch.tanh(x) - 1)**k for k in range(m+1)])


def sigmoid(x):
    return 0.5 * (1 + np.tanh(0.5 * x))


def derivative_sigmoid(m, x):
    return derivative_tanh(m, x / 2) / (2 ** (m+1))


def sparse_jacobians(C, b, h, e, d, derivation_order, is_sparse, non_linearity, device=torch.device('cpu')):
    """Computes [J^k(F)(h) for k in range(N)]"""
    if is_sparse:
        x = np.matmul(C, h) + b
        E = np.zeros([d, e+d])
        E[:, e:] = np.eye(d)
        y = np.zeros([e + d])
        if non_linearity == 'tanh':
            y[:e] = np.tanh(x[:e])
        elif non_linearity == 'sigmoid':
            y[:e] = sigmoid(x[:e])
        result = [sparse.COO(np.concatenate([E.T, y[:, None]], axis=1))]
    else:
        x = torch.matmul(C, h) + b
        E = torch.zeros([d, e+d], device=device)
        E[:, e:] = torch.eye(d, device=device)
        y = torch.zeros([e + d], device=device)
        if non_linearity == 'tanh':
            y[:e] = torch.tanh(x[:e])
        elif non_linearity == 'sigmoid':
            y[:e] = sigmoid(x[:e])
        result = [torch.cat([E.T, y[:, None]], dim=1)]
    R = C
    for k in range(1, derivation_order + 1):
        if non_linearity == 'tanh':
            I = derivative_tanh(k, x)
        elif non_linearity == 'sigmoid':
            I = derivative_sigmoid(k, x)
        if is_sparse:
            result.append(
                sparse.COO(
                    np.concatenate(
                        [np.zeros([e + d, d] + [e + d] * k),
                         np.einsum('b...,b->b...', R, I)[:, None, ...]],
                        axis=1)))
            R = np.einsum('b...,bi->b...i', R, C)
        else:
            result.append(
                torch.cat(
                    [torch.zeros([e + d, d] + [e + d] * k, device=device),
                     torch.einsum('b...,b->b...', R, I)[:, None, ...]],
                    dim=1))
            R = torch.einsum('b...,bi->b...i', R, C)
    return result


def star_product(tensor_field_1: Callable[[torch.Tensor],
                                          torch.Tensor],
                 tensor_field_2: Callable[[torch.Tensor],
                                          torch.Tensor],
                 create_graph) -> Callable[[torch.Tensor], torch.Tensor]:
    """ Computes the star product between two tensor fields, returned as a tensor field itself.

    The star product between two vector fields $f: R^e -> R^e$ and $g: R^e -> R^e$ is the vector field
    $f * g$ such that

    $$f * g(h) = \sum_{i} \frac{dg}{dh_i}(h) f_i(h).$$

    This operation can be generalized to tensor fields. The star product is applied on the first axis
    of both tensor fields. The subsequent axes are multiplied with the standard tensor product.

    The operation is batched, in the sense that all vector fields input and output tensors.
    All vector fields take as input a tensor of shape (batch_size, $e$).

    :param tensor_field_1: function that returns a tensor of shape (batch_size, $e$, ..._1).
    :param tensor_field_2: function that returns a tensor of shape (batch_size, $e$, ..._2).
    :return: function that returns a torch.Tensor of shape (batch_size, $e$, ..._1, ..._2).
    """
    def result(inputs: torch.Tensor):
        # create_graph=True is necessary for recursive gradient computation.
        jacobian = torch.autograd.functional.jacobian(tensor_field_2, inputs, create_graph=create_graph)
        output_1 = tensor_field_1(inputs)
        nb_axes_1 = len(output_1.shape) - 2
        return torch.movedim(
            torch.tensordot(output_1, jacobian, dims=([0, 1], [-2, -1])),
            (nb_axes_1, nb_axes_1 + 1),
            (0, 1))
    return result


def identity_product(X, shapes, contraction, transposed, is_sparse, device=torch.device('cpu')):
    results = []
    number_identities = len(shapes)
    if is_sparse:
        identities_product = sparse.eye(int(np.prod(shapes))).reshape(shapes + shapes)
    else:
        identities_product = torch.eye(int(np.prod(shapes)), device=device).reshape(shapes + shapes)

    if transposed:
        number_first_axes, number_second_axes = (contraction, len(X[0].shape) - contraction)
        tensor_slicing = (None, ) * number_identities + (slice(None),) * number_second_axes + \
            (None,) * number_identities + (slice(None),) * number_first_axes
        id_slicing = (slice(None),) * number_identities + (None,) * number_second_axes + \
            (slice(None),) * number_identities + (None,) * number_first_axes
    else:
        number_first_axes, number_second_axes = (len(X[0].shape) - contraction, contraction)
        tensor_slicing = (slice(None),) * number_first_axes + (None, ) * number_identities + \
            (slice(None),) * number_second_axes + (None, ) * number_identities
        id_slicing = (None,) * number_first_axes + (slice(None),) * number_identities + \
            (None,) * number_second_axes + (slice(None),) * number_identities

    for derivation_order, tensor in enumerate(X):
        if transposed:
            if is_sparse:
                tensor_with_good_axes = sparse.moveaxis(
                    tensor, range(number_first_axes, number_first_axes + number_second_axes),
                    range(number_second_axes))
            else:
                tensor_with_good_axes = torch.movedim(
                    tensor, tuple(range(number_first_axes, number_first_axes + number_second_axes)),
                    tuple(range(number_second_axes)))
        else:
            tensor_with_good_axes = tensor
        results.append(
            tensor_with_good_axes[tensor_slicing + (slice(None),) * derivation_order] *
            identities_product[id_slicing + (None,) * derivation_order])
    return results


def plus(A, B):
    return [x + y for (x, y) in zip(A, B)]


def moveaxis(A, old, new, is_sparse):
    if is_sparse:
        return [sparse.moveaxis(t, old, new) for t in A]
    else:
        return [torch.movedim(t, old, new) for t in A]


def tensordot(A, B, contraction, derivation_order, is_sparse, device=torch.device('cpu')):
    if is_sparse:
        zeroth_order_result = sparse.tensordot(A[0], B[0], axes=contraction)
    else:
        zeroth_order_result = torch.tensordot(A[0], B[0], dims=contraction)
    if derivation_order == 0:
        return [zeroth_order_result]
    else:
        first_tensordot = tensordot(
            identity_product(B, A[0].shape[: -contraction], contraction, True, is_sparse, device=device),
            A[1:],
            len(A[0].shape),
            derivation_order - 1, is_sparse)
        second_tensordot = tensordot(
            identity_product(A, B[0].shape[contraction:], contraction, False, is_sparse, device=device),
            B[1:],
            len(B[0].shape),
            derivation_order - 1, is_sparse)
        return [zeroth_order_result] + plus(first_tensordot, second_tensordot)


def iterated_jacobian(model, order, evaluation_point, is_sparse=False, device=torch.device('cpu')):
    if order >= 5 and not is_sparse:
        raise ValueError('Cannot call non sparse mode with an order greater than 4.')
    if is_sparse:
        warnings.warn('Computation with sparse iterated jacobians is not stabilized. Use at your own risk.')
    if len(evaluation_point.shape) > 1:
        raise ValueError('iterated_jacobian is not batched.')
    if model.non_linearity not in ['tanh', 'sigmoid']:
        raise ValueError('iterated_jacobian only works for tanh nonlinearity.')

    d = model.input_channels
    e = model.hidden_channels

    C = torch.zeros([e+d, e+d], device=device)
    C[:e, :e] = model.weight_hh
    C[:e, e:] = model.weight_ih
    b = torch.zeros([e + d], device=device)
    b[:e] = model.bias
    if is_sparse:
        C = C.detach().numpy()
        b = b.detach().numpy()

    F = sparse_jacobians(C, b, evaluation_point, e, d, order-1, is_sparse, model.non_linearity, device=device)
    if is_sparse:
        results = [torch.Tensor(F[0], device=device)]
    else:
        results = [F[0]]
    star_result = F
    for k in range(order-2, -1, -1):
        star_result = moveaxis(tensordot(star_result[1:], F, 1, k, is_sparse, device=device), order-k, 1, is_sparse)
        if is_sparse:
            results.append(torch.Tensor(star_result[0], device=device))
        else:
            results.append(star_result[0])
    return results


def model_approximation(
        model: Callable[[torch.Tensor],
                        torch.Tensor],
        truncation_order: int, control: torch.Tensor, initial_value: torch.Tensor,
        is_sparse=False) -> torch.Tensor:
    """Computes the approximation of the solution of a controlled differential equation (CDE) dh = F(h)dX,
    where F is a tensor field and X is a  control.

    The approximation is obtained by taking the scalar product between the signature of the control and
    the iterated jacobians of the tensor field. More precisely, we have the approximation:

    $$h_t \simeq h_0 + \sum_{k=1}^N S^{(i_1, \dots, i_k)}_{[0,t]}(X) F_{i_1} * (F_{i_2} * ... * (F_{i_{k-1}} * F_{i_k})) (h_0),$$

    where $N$ is the order of truncation and $h_0$ is the initial value.

    :param tensor_field: tensor field, taking as input a tensor of shape (batch_size, $e$) and returns a
    tensor of shape (batch_size, $e$, $d$).
    :param truncation_order: int, order of truncation of the signature and the iterated jacobian.
    :param control: tensor of shape (length, $d$)
    :param initial_value: tensor of shape ($e$)
    :return: an approximation of the value of the solution of the CDE, a tensor of shape (batch_size, $e$)
    """
    result = initial_value.clone().detach()
    all_results = []
    signature = signatory.signature(control.unsqueeze(0), truncation_order)

    jacobians = iterated_jacobian(model, truncation_order, initial_value, is_sparse=is_sparse)

    for k in range(truncation_order):
        reshaped_jacobian = jacobians[k].reshape((initial_value.shape[0], -1))
        signature_tensor = signatory.extract_signature_term(signature, control.shape[1], k + 1)
        result += torch.sum(reshaped_jacobian * signature_tensor, dim=-1)
        all_results.append(result.clone())
    return result, torch.stack(all_results)
