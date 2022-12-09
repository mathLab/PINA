"""Utils module"""
from functools import reduce

from .label_tensor import LabelTensor

import torch


def number_parameters(model, aggregate=True, only_trainable=True):  # TODO: check
    """
    Return the number of parameters of a given `model`.

    :param torch.nn.Module model: the torch module to inspect.
    :param bool aggregate: if True the return values is an integer corresponding
        to the total amount of parameters of whole model. If False, it returns a
        dictionary whose keys are the names of layers and the values the
        corresponding number of parameters. Default is True.
    :param bool trainable: if True, only trainable parameters are count,
        otherwise no. Default is True.
    :return: the number of parameters of the model
    :rtype: dict or int
    """
    tmp = {}
    for name, parameter in model.named_parameters():
        if only_trainable and not parameter.requires_grad:
            continue

        tmp[name] = parameter.numel()

    if aggregate:
        tmp = sum(tmp.values())

    return tmp


def merge_tensors(tensors):  # name to be changed
    if tensors:
        return reduce(merge_two_tensors, tensors[1:], tensors[0])
    raise ValueError("Expected at least one tensor")


def merge_two_tensors(tensor1, tensor2):
    n1 = tensor1.shape[0]
    n2 = tensor2.shape[0]

    tensor1 = LabelTensor(tensor1.repeat(n2, 1), labels=tensor1.labels)
    tensor2 = LabelTensor(tensor2.repeat_interleave(n1, dim=0),
                          labels=tensor2.labels)
    return tensor1.append(tensor2)


def torch_lhs(n, dim):
    """Latin Hypercube Sampling torch routine.
    Sampling in range $[0, 1)^d$.

    :param int n: number of samples
    :param int dim: dimensions of latin hypercube
    :return: samples
    :rtype: torch.tensor
    """

    if not isinstance(n, int):
        raise TypeError('number of point n must be int')

    if not isinstance(dim, int):
        raise TypeError('dim must be int')

    if dim < 1:
        raise ValueError('dim must be greater than one')

    samples = torch.rand(size=(n, dim))

    perms = torch.tile(torch.arange(1, n + 1), (dim, 1))

    for row in range(dim):
        idx_perm = torch.randperm(perms.shape[-1])
        perms[row, :] = perms[row, idx_perm]

    perms = perms.T

    samples = (perms - samples) / n

    return samples
