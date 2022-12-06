"""Utils module"""
from functools import reduce
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


class MyDataSet(torch.utils.data.Dataset):

    def __init__(self, location, tensor):
        self._tensor = tensor
        self._location = location
        self._len = len(tensor)

    def __getitem__(self, index):
        tensor = self._tensor.select(0, index)
        return {self._location: tensor}

    def __len__(self):
        return self._len
