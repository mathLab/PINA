"""Utils module"""

from torch.utils.data import Dataset, DataLoader
from functools import reduce
import types

import torch
from torch.utils.data import DataLoader, default_collate, ConcatDataset

from .label_tensor import LabelTensor

import torch


def check_consistency(object, object_instance, subclass=False):
    """Helper function to check object inheritance consistency.
       Given a specific ``'object'`` we check if the object is
       instance of a specific ``'object_instance'``, or in case
       ``'subclass=True'`` we check if the object is subclass
       if the ``'object_instance'``.

    :param (iterable or class object) object: The object to check the inheritance
    :param Object object_instance: The parent class from where the object
        is expected to inherit
    :param str object_name: The name of the object
    :param bool subclass: Check if is a subclass and not instance
    :raises ValueError: If the object does not inherit from the
        specified class
    """
    if not isinstance(object, (list, set, tuple)):
        object = [object]

    for obj in object:
        try:
            if not subclass:
                assert isinstance(obj, object_instance)
            else:
                assert issubclass(obj, object_instance)
        except AssertionError:
            raise ValueError(f"{type(obj).__name__} must be {object_instance}.")

def labelize_forward(forward, input_variables, output_variables):
    def wrapper(x):
        x = x.extract(input_variables)
        output = forward(x.tensor)
        # keep it like this, directly using LabelTensor(...) raises errors
        # when compiling the code
        output = output.as_subclass(LabelTensor)
        output.labels = output_variables
        return output
    return wrapper

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
        raise TypeError("number of point n must be int")

    if not isinstance(dim, int):
        raise TypeError("dim must be int")

    if dim < 1:
        raise ValueError("dim must be greater than one")

    samples = torch.rand(size=(n, dim))

    perms = torch.tile(torch.arange(1, n + 1), (dim, 1))

    for row in range(dim):
        idx_perm = torch.randperm(perms.shape[-1])
        perms[row, :] = perms[row, idx_perm]

    perms = perms.T

    samples = (perms - samples) / n

    return samples


def is_function(f):
    """
    Checks whether the given object `f` is a function or lambda.

    :param object f: The object to be checked.
    :return: `True` if `f` is a function, `False` otherwise.
    :rtype: bool
    """
    return type(f) == types.FunctionType or type(f) == types.LambdaType


def chebyshev_roots(n):
    """
    Return the roots of *n* Chebyshev polynomials (between [-1, 1]).

    :param int n: number of roots
    :return: roots
    :rtype: torch.tensor
    """
    pi = torch.acos(torch.zeros(1)).item() * 2
    k = torch.arange(n)
    nodes = torch.sort(torch.cos(pi * (k + 0.5) / n))[0]
    return nodes