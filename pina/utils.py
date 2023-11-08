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


def number_parameters(model,
                      aggregate=True,
                      only_trainable=True):  # TODO: check
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


# class PinaDataset():

#     def __init__(self, pinn) -> None:
#         self.pinn = pinn

#     @property
#     def dataloader(self):
#         return self._create_dataloader()

#     @property
#     def dataset(self):
#         return [self.SampleDataset(key, val)
#                 for key, val in self.input_pts.items()]

#     def _create_dataloader(self):
#         """Private method for creating dataloader

#         :return: dataloader
#         :rtype: torch.utils.data.DataLoader
#         """
#         if self.pinn.batch_size is None:
#             return {key: [{key: val}] for key, val in self.pinn.input_pts.items()}

#         def custom_collate(batch):
#             # extracting pts labels
#             _, pts = list(batch[0].items())[0]
#             labels = pts.labels
#             # calling default torch collate
#             collate_res = default_collate(batch)
#             # save collate result in dict
#             res = {}
#             for key, val in collate_res.items():
#                 val.labels = labels
#                 res[key] = val
#             __init__(self, location, tensor):
#             self._tensor = tensor
#             self._location = location
#             self._len = len(tensor)

#         def __getitem__(self, index):
#             tensor = self._tensor.select(0, index)
#             return {self._location: tensor}

#         def __len__(self):
#             return self._len


class LabelTensorDataLoader(DataLoader):

    def collate_fn(self, data):
        pass
