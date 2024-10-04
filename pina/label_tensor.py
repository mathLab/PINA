""" Module for LabelTensor """

import torch
from torch import Tensor

def issubset(a, b):
    """
    Check if a is a subset of b.
    """
    return set(a).issubset(set(b))


class LabelTensor(torch.Tensor):
    """Torch tensor with a label for any column."""

    @staticmethod
    def __new__(cls, x, labels, *args, **kwargs):
        return super().__new__(cls, x, *args, **kwargs)

    @property
    def tensor(self):
        return self.as_subclass(Tensor)

    def __len__(self) -> int:
        return super().__len__()

    def __init__(self, x, labels):
        """
        Construct a `LabelTensor` by passing a dict of the labels

        :Example:
            >>> from pina import LabelTensor
            >>> tensor = LabelTensor(
            >>>     torch.rand((2000, 3)),
                    {1: {"name": "space"['a', 'b', 'c'])

        """
        self.labels = labels

    @property
    def labels(self):
        """Property decorator for labels

        :return: labels of self
        :rtype: list
        """
        return self._labels

    @labels.setter
    def labels(self, labels):
        """"
        Set properly the parameter _labels

        :param labels: Labels to assign to the class variable _labels.
        :type: labels: str | list(str) | dict
        """
        if hasattr(self, 'labels') is False:
            self.init_labels()
        if isinstance(labels, dict):
            self.update_labels_from_dict(labels)
        elif isinstance(labels, list):
            self.update_labels_from_list(labels)
        elif isinstance(labels, str):
            labels = [labels]
            self.update_labels_from_list(labels)
        else:
            raise ValueError(f"labels must be list, dict or string.")

    def extract(self, label_to_extract):
        """
        Extract the subset of the original tensor by returning all the columns
        corresponding to the passed ``label_to_extract``.

        :param label_to_extract: The label(s) to extract.
        :type label_to_extract: str | list(str) | tuple(str)
        :raises TypeError: Labels are not ``str``.
        :raises ValueError: Label to extract is not in the labels ``list``.
        """
        from copy import deepcopy
        if isinstance(label_to_extract, (str, int)):
            label_to_extract = [label_to_extract]
        if isinstance(label_to_extract, (tuple, list)):
            last_dim_label = self._labels[self.tensor.ndim - 1]['dof']
            if set(label_to_extract).issubset(last_dim_label) is False:
                raise ValueError('Cannot extract a dof which is not in the original LabelTensor')
            idx_to_extract = [last_dim_label.index(i) for i in label_to_extract]
            new_tensor = self.tensor
            new_tensor = new_tensor[..., idx_to_extract]
            new_labels = deepcopy(self._labels)
            last_dim_new_label = {self.tensor.ndim - 1: {
                'dof': label_to_extract,
                'name': self._labels[self.tensor.ndim - 1]['name']
            }}
            new_labels.update(last_dim_new_label)
        elif isinstance(label_to_extract, dict):
            new_labels = (deepcopy(self._labels))
            new_tensor = self.tensor
            for k, v in label_to_extract.items():
                idx_dim = None
                for kl, vl in self._labels.items():
                    if vl['name'] == k:
                        idx_dim = kl
                        break
                dim_labels = self._labels[idx_dim]['dof']
                if isinstance(label_to_extract[k], (int, str)):
                    label_to_extract[k] = [label_to_extract[k]]
                if set(label_to_extract[k]).issubset(dim_labels) is False:
                    raise ValueError('Cannot extract a dof which is not in the original LabelTensor')
                idx_to_extract = [dim_labels.index(i) for i in label_to_extract[k]]
                indexer = [slice(None)] * idx_dim + [idx_to_extract] + [slice(None)] * (self.tensor.ndim - idx_dim - 1)
                new_tensor = new_tensor[indexer]
                dim_new_label = {idx_dim: {
                    'dof': label_to_extract[k],
                    'name': self._labels[idx_dim]['name']
                }}
                new_labels.update(dim_new_label)
        else:
            raise ValueError('labels_to_extract must be str or list or dict')
        return LabelTensor(new_tensor, new_labels)

    def __str__(self):
        """
        returns a string with the representation of the class
        """

        s = ''
        for key, value in self._labels.items():
            s += f"{key}: {value}\n"
        s += '\n'
        s += super().__str__()
        return s

    @staticmethod
    def cat(tensors, dim=0):
        """
        Stack a list of tensors. For example, given a tensor `a` of shape `(n,m,dof)` and a tensor `b` of dimension `(n',m,dof)`
        the resulting tensor is of shape `(n+n',m,dof)`

        :param tensors: tensors to concatenate
        :type tensors: list(LabelTensor)
        :param dim: dimensions on which you want to perform the operation (default 0)
        :type dim: int
        :rtype: LabelTensor
        :raises ValueError: either number dof or dimensions names differ
        """
        if len(tensors) == 0:
            return []
        if len(tensors) == 1:
            return tensors[0]
        n_dims = tensors[0].ndim
        new_labels_cat_dim = []
        for i in range(n_dims):
            name = tensors[0].labels[i]['name']
            if i != dim:
                dof = tensors[0].labels[i]['dof']
                for tensor in tensors:
                    dof_to_check = tensor.labels[i]['dof']
                    name_to_check = tensor.labels[i]['name']
                    if dof != dof_to_check or name != name_to_check:
                        raise ValueError('dimensions must have the same dof and name')
            else:
                for tensor in tensors:
                    new_labels_cat_dim += tensor.labels[i]['dof']
                    name_to_check = tensor.labels[i]['name']
                    if name != name_to_check:
                        raise ValueError('dimensions must have the same dof and name')
        new_tensor = torch.cat(tensors, dim=dim)
        labels = tensors[0].labels
        labels.pop(dim)
        new_labels_cat_dim = new_labels_cat_dim if len(set(new_labels_cat_dim)) == len(new_labels_cat_dim) \
            else range(new_tensor.shape[dim])
        labels[dim] = {'dof': new_labels_cat_dim,
                       'name': tensors[1].labels[dim]['name']}
        return LabelTensor(new_tensor, labels)

    def requires_grad_(self, mode=True):
        lt = super().requires_grad_(mode)
        lt.labels = self._labels
        return lt

    @property
    def dtype(self):
        return super().dtype

    def to(self, *args, **kwargs):
        """
        Performs Tensor dtype and/or device conversion. For more details, see
        :meth:`torch.Tensor.to`.
        """
        tmp = super().to(*args, **kwargs)
        new = self.__class__.clone(self)
        new.data = tmp.data
        return new

    def clone(self, *args, **kwargs):
        """
        Clone the LabelTensor. For more details, see
        :meth:`torch.Tensor.clone`.

        :return: A copy of the tensor.
        :rtype: LabelTensor
        """

        out = LabelTensor(super().clone(*args, **kwargs), self._labels)
        return out


    def init_labels(self):
        self._labels = {
            idx_: {
                'dof': range(self.tensor.shape[idx_]),
                'name': idx_
            } for idx_ in range(self.tensor.ndim)
        }

    def update_labels_from_dict(self, labels):
        """
        Update the internal label representation according to the values passed as input.

        :param labels: The label(s) to update.
        :type labels: dict
        :raises ValueError: dof list contain duplicates or number of dof does not match with tensor shape
        """

        tensor_shape = self.tensor.shape
        for k, v in labels.items():
            if len(v['dof']) != len(set(v['dof'])):
                raise ValueError("dof must be unique")
            if len(v['dof']) != tensor_shape[k]:
                raise ValueError('Number of dof does not match with tensor dimension')
        self._labels.update(labels)

    def update_labels_from_list(self, labels):
        """
        Given a list of dof, this method update the internal label representation

        :param labels: The label(s) to update.
        :type labels: list
        """
        last_dim_labels = {self.tensor.ndim - 1: {'dof': labels, 'name': self.tensor.ndim - 1}}
        self.update_labels_from_dict(last_dim_labels)

    @staticmethod
    def summation(tensors):
        if len(tensors) == 0:
            raise ValueError('tensors list must not be empty')
        if len(tensors) == 1:
            return tensors[0]
        labels = tensors[0].labels
        for j in range(tensors[0].ndim):
            for i in range(1, len(tensors)):
                if labels[j] != tensors[i].labels[j]:
                    labels.pop(j)
                    break

        data = torch.zeros(tensors[0].tensor.shape)
        for i in range(len(tensors)):
            data += tensors[i].tensor
        new_tensor = LabelTensor(data, labels)
        return new_tensor

    def last_dim_dof(self):
        return self._labels[self.tensor.ndim - 1]['dof']

    def append(self, tensor, mode='std'):
        print(self.labels)
        print(tensor.labels)
        if mode == 'std':
            new_label_tensor = LabelTensor.cat([self, tensor], dim=self.tensor.ndim - 1)

        return new_label_tensor
