import math
from typing import Callable, List

import numpy as np
import signatory
import sparse
import sympy
import torch


def derivative_tanh(m, x):
    """Computes the m-th derivative of tanh at x."""
    # see https://arxiv.org/ftp/arxiv/papers/0903/0903.0117.pdf
    return (-2)**m * (torch.tanh(x) + 1) * \
        sum([math.factorial(k) / (2**k) * int(sympy.functions.combinatorial.numbers.stirling(m, k))
             * (torch.tanh(x) - 1)**k for k in range(m+1)])


def sigmoid(x):
    return 0.5 * (1 + np.tanh(0.5 * x))


def derivative_sigmoid(m, x):
    """Computes the m-th derivative of sigmoid at x."""
    return derivative_tanh(m, x / 2) / (2 ** (m+1))


def jacobians(C, b, h, e, d, derivation_order, is_sparse, non_linearity, device=torch.device('cpu')):
    """Computes the higher-order Jacobians of a RNN cell.
    
    :param C: weights of the cell
    :param b: bias of the cell
    :param h: hidden state
    :param e: size of the hidden state
    :param d: size of the data
    :param derivation_order: order of the Jacobians
    :param is_sparse: whether sparse tensors or PyTorch tensors should be used
    :param non_linearity: name of the activation function
    :param device: used device
    :return: list of Jacobians up to order derivation_order
    
    """
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
            y[:e] = torch.sigmoid(x[:e])
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


def identity_product(X: List, shapes: List, contraction: int, transposed: bool, is_sparse: bool, device: torch.device = torch.device('cpu')):
    """Technical function used for the implementation of the tensor dot with higher-order forward automatic differentiation."""
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


def plus(A: List, B: List) -> List:
    """Implements a sum operation with higher-order forward automatic differentiation."""
    return [x + y for (x, y) in zip(A, B)]


def moveaxis(A: List, old: int, new: int, is_sparse: bool) -> List:
    """Implements a moveaxis operation with higher-order forward automatic differentiation."""
    if is_sparse:
        return [sparse.moveaxis(t, old, new) for t in A]
    else:
        return [torch.movedim(t, old, new) for t in A]


def tensordot(A: List, B: List, contraction: int, derivation_order: int, is_sparse: bool, device: torch.device = torch.device('cpu')):
    """Implements a tensordot operation with higher-order forward automatic differentiation."""
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


def iterated_jacobian(model: torch.nn.Module, 
                      order: int, 
                      evaluation_point: torch.Tensor, 
                      is_sparse: bool = False, 
                      device=torch.device('cpu')) -> List[torch.Tensor]:
    """Computes the iterated jacobians of the tensor field associated to a model.

    :param model: RNN model, taking as input a tensor of shape (batch_size, $e$) and returns a
    tensor of shape (batch_size, $e$, $d$)
    :param order: int, order of truncation of the signature and the iterated jacobian
    :param evaluation_point: tensor of shape ($e$)
    :param is_sparse: whether sparse tensors or PyTorch tensors should be used
    :return: a list of iterated jacobians up to the specified order
    """
    if order >= 5 and not is_sparse:
        raise ValueError('Cannot call non sparse mode with an order greater than 4.')
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

    F = jacobians(C, b, evaluation_point, e, d, order-1, is_sparse, model.non_linearity, device=device)
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
        model: torch.nn.Module,
        truncation_order: int, control: torch.Tensor, initial_value: torch.Tensor,
        is_sparse=False):
    """Computes the approximation of the solution of a controlled differential equation (CDE) dh = F(h)dX,
    where F is the tensor field associated to a model and X is a control.

    The approximation is obtained by taking the scalar product between the signature of the control and
    the iterated jacobians of the tensor field. More precisely, we have the approximation:

    $$h_t \simeq h_0 + \sum_{k=1}^N S^{(i_1, \dots, i_k)}_{[0,t]}(X) F_{i_1} * (F_{i_2} * ... * (F_{i_{k-1}} * F_{i_k})) (h_0),$$

    where $N$ is the order of truncation and $h_0$ is the initial value.

    :param model: RNN model, taking as input a tensor of shape (batch_size, $e$) and returns a
    tensor of shape (batch_size, $e$, $d$)
    :param truncation_order: int, order of truncation of the signature and the iterated jacobian.
    :param control: tensor of shape (length, $d$)
    :param initial_value: tensor of shape ($e$)
    :param is_sparse: whether sparse tensors or PyTorch tensors should be used
    :return: an approximation of the value of the solution of the CDE, a tensor of shape (batch_size, $e$)
        at order trunction_order, and a list of approximations up to this order
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
