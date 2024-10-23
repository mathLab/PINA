""" Module for LabelTensor """
from copy import deepcopy, copy
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

    def __init__(self, x, labels):
        """
        Construct a `LabelTensor` by passing a dict of the labels

        :Example:
            >>> from pina import LabelTensor
            >>> tensor = LabelTensor(
            >>>     torch.rand((2000, 3)),
                    {1: {"name": "space"['a', 'b', 'c'])

        """
        self.dim_names = None
        self.labels = labels

    @property
    def labels(self):
        """Property decorator for labels

        :return: labels of self
        :rtype: list
        """
        return self._labels[self.tensor.ndim - 1]['dof']

    @property
    def full_labels(self):
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
            raise ValueError("labels must be list, dict or string.")
        self.set_names()

    def set_names(self):
        labels = self.full_labels
        self.dim_names = {}
        for dim in range(self.tensor.ndim):
            self.dim_names[labels[dim]['name']] = dim

    def extract(self, label_to_extract):
        """
        Extract the subset of the original tensor by returning all the columns
        corresponding to the passed ``label_to_extract``.

        :param label_to_extract: The label(s) to extract.
        :type label_to_extract: str | list(str) | tuple(str)
        :raises TypeError: Labels are not ``str``.
        :raises ValueError: Label to extract is not in the labels ``list``.
        """
        if isinstance(label_to_extract, (str, int)):
            label_to_extract = [label_to_extract]
        if isinstance(label_to_extract, (tuple, list)):
            return self._extract_from_list(label_to_extract)
        if isinstance(label_to_extract, dict):
            return self._extract_from_dict(label_to_extract)
        raise ValueError('labels_to_extract must be str or list or dict')

    def _extract_from_list(self, labels_to_extract):
        # Store locally all necessary obj/variables
        ndim = self.tensor.ndim
        labels = self.full_labels
        tensor = self.tensor
        last_dim_label = self.labels

        # Verify if all the labels in labels_to_extract are in last dimension
        if set(labels_to_extract).issubset(last_dim_label) is False:
            raise ValueError(
                'Cannot extract a dof which is not in the original LabelTensor')

        # Extract index to extract
        idx_to_extract = [last_dim_label.index(i) for i in labels_to_extract]

        # Perform extraction
        new_tensor = tensor[..., idx_to_extract]

        # Manage labels
        new_labels = copy(labels)

        last_dim_new_label = {ndim - 1: {
            'dof': list(labels_to_extract),
            'name': labels[ndim - 1]['name']
        }}
        new_labels.update(last_dim_new_label)
        return LabelTensor(new_tensor, new_labels)

    def _extract_from_dict(self, labels_to_extract):
        labels = self.full_labels
        tensor = self.tensor
        ndim = tensor.ndim
        new_labels = deepcopy(labels)
        new_tensor = tensor
        for k, _ in labels_to_extract.items():
            idx_dim = self.dim_names[k]
            dim_labels = labels[idx_dim]['dof']
            if isinstance(labels_to_extract[k], (int, str)):
                labels_to_extract[k] = [labels_to_extract[k]]
            if set(labels_to_extract[k]).issubset(dim_labels) is False:
                raise ValueError(
                    'Cannot extract a dof which is not in the original '
                    'LabelTensor')
            idx_to_extract = [dim_labels.index(i) for i in labels_to_extract[k]]
            indexer = [slice(None)] * idx_dim + [idx_to_extract] + [
                slice(None)] * (ndim - idx_dim - 1)
            new_tensor = new_tensor[indexer]
            dim_new_label = {idx_dim: {
                'dof': labels_to_extract[k],
                'name': labels[idx_dim]['name']
            }}
            new_labels.update(dim_new_label)
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
        Stack a list of tensors. For example, given a tensor `a` of shape
        `(n,m,dof)` and a tensor `b` of dimension `(n',m,dof)`
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
        new_labels_cat_dim = LabelTensor._check_validity_before_cat(tensors,
                                                                    dim)

        # Perform cat on tensors
        new_tensor = torch.cat(tensors, dim=dim)

        # Update labels
        labels = tensors[0].full_labels
        labels.pop(dim)
        new_labels_cat_dim = new_labels_cat_dim if len(
            set(new_labels_cat_dim)) == len(new_labels_cat_dim) \
            else range(new_tensor.shape[dim])
        labels[dim] = {'dof': new_labels_cat_dim,
                       'name': tensors[1].full_labels[dim]['name']}
        return LabelTensor(new_tensor, labels)

    @staticmethod
    def _check_validity_before_cat(tensors, dim):
        n_dims = tensors[0].ndim
        new_labels_cat_dim = []
        # Check if names and dof of the labels are the same in all dimensions
        # except in dim
        for i in range(n_dims):
            name = tensors[0].full_labels[i]['name']
            if i != dim:
                dof = tensors[0].full_labels[i]['dof']
                for tensor in tensors:
                    dof_to_check = tensor.full_labels[i]['dof']
                    name_to_check = tensor.full_labels[i]['name']
                    if dof != dof_to_check or name != name_to_check:
                        raise ValueError(
                            'dimensions must have the same dof and name')
            else:
                for tensor in tensors:
                    new_labels_cat_dim += tensor.full_labels[i]['dof']
                    name_to_check = tensor.full_labels[i]['name']
                    if name != name_to_check:
                        raise ValueError(
                            'Dimensions to concatenate must have the same name')
        return new_labels_cat_dim

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
        Update the internal label representation according to the values passed
        as input.

        :param labels: The label(s) to update.
        :type labels: dict
        :raises ValueError: dof list contain duplicates or number of dof does
        not match with tensor shape
        """
        tensor_shape = self.tensor.shape
        # Check dimensionality
        for k, v in labels.items():
            if len(v['dof']) != len(set(v['dof'])):
                raise ValueError("dof must be unique")
            if len(v['dof']) != tensor_shape[k]:
                raise ValueError(
                    'Number of dof does not match with tensor dimension')
        # Perform update
        self._labels.update(labels)

    def update_labels_from_list(self, labels):
        """
        Given a list of dof, this method update the internal label
        representation

        :param labels: The label(s) to update.
        :type labels: list
        """
        # Create a dict with labels
        last_dim_labels = {
            self.tensor.ndim - 1: {'dof': labels, 'name': self.tensor.ndim - 1}}
        self.update_labels_from_dict(last_dim_labels)

    @staticmethod
    def summation(tensors):
        if len(tensors) == 0:
            raise ValueError('tensors list must not be empty')
        if len(tensors) == 1:
            return tensors[0]
        # Collect all labels
        labels = tensors[0].full_labels
        # Check labels of all the tensors in each dimension
        for j in range(tensors[0].ndim):
            for i in range(1, len(tensors)):
                if labels[j] != tensors[i].full_labels[j]:
                    labels.pop(j)
                    break
        # Sum tensors
        data = torch.zeros(tensors[0].tensor.shape)
        for tensor in tensors:
            data += tensor.tensor
        new_tensor = LabelTensor(data, labels)
        return new_tensor

    def append(self, tensor, mode='std'):
        if mode == 'std':
            # Call cat on last dimension
            new_label_tensor = LabelTensor.cat([self, tensor],
                                               dim=self.tensor.ndim - 1)
        elif mode == 'cross':
            # Crete tensor and call cat on last dimension
            tensor1 = self
            tensor2 = tensor
            n1 = tensor1.shape[0]
            n2 = tensor2.shape[0]
            tensor1 = LabelTensor(tensor1.repeat(n2, 1), labels=tensor1.labels)
            tensor2 = LabelTensor(tensor2.repeat_interleave(n1, dim=0),
                                  labels=tensor2.labels)
            new_label_tensor = LabelTensor.cat([tensor1, tensor2],
                                               dim=self.tensor.ndim - 1)
        else:
            raise ValueError('mode must be either "std" or "cross"')
        return new_label_tensor

    @staticmethod
    def vstack(label_tensors):
        """
        Stack tensors vertically. For more details, see
        :meth:`torch.vstack`.

        :param list(LabelTensor) label_tensors: the tensors to stack. They need
            to have equal labels.
        :return: the stacked tensor
        :rtype: LabelTensor
        """
        return LabelTensor.cat(label_tensors, dim=0)

    def __getitem__(self, index):
        """
        TODO: Complete docstring
        :param index:
        :return:
        """

        if isinstance(index, str) or (isinstance(index, (tuple, list)) and all(
                isinstance(a, str) for a in index)):
            return self.extract(index)
        selected_lt = super().__getitem__(index)

        if isinstance(index, (int, slice)):
            return self._getitem_int_slice(index, selected_lt)

        if len(index) == self.tensor.ndim:
            return self._getitem_full_dim_indexing(index, selected_lt)

        if isinstance(index, torch.Tensor) or (
                isinstance(index, (tuple, list)) and all(
            isinstance(x, int) for x in index)):
            return self._getitem_permutation(index, selected_lt)
        raise ValueError('Not recognized index type')

    def _getitem_int_slice(self, index, selected_lt):
        """
        :param index:
        :param selected_lt:
        :return:
        """
        if selected_lt.ndim == 1:
            selected_lt = selected_lt.reshape(1, -1)
        if hasattr(self, "labels"):
            new_labels = deepcopy(self.full_labels)
            to_update_dof = new_labels[0]['dof'][index]
            to_update_dof = to_update_dof if isinstance(to_update_dof, (
                tuple, list, range)) else [to_update_dof]
            new_labels.update(
                {0: {'dof': to_update_dof, 'name': new_labels[0]['name']}}
            )
            selected_lt.labels = new_labels
        return selected_lt

    def _getitem_full_dim_indexing(self, index, selected_lt):
        new_labels = {}
        old_labels = self.full_labels
        if selected_lt.ndim == 1:
            selected_lt = selected_lt.reshape(-1, 1)
            new_labels = deepcopy(old_labels)
            new_labels[1].update({'dof': old_labels[1]['dof'][index[1]],
                                  'name': old_labels[1]['name']})
        idx = 0
        for j in range(selected_lt.ndim):
            if not isinstance(index[j], int):
                if hasattr(self, "labels"):
                    new_labels.update(
                        self._update_label_for_dim(old_labels, index[j], idx))
                idx += 1
        selected_lt.labels = new_labels
        return selected_lt

    def _getitem_permutation(self, index, selected_lt):
        new_labels = deepcopy(self.full_labels)
        new_labels.update(self._update_label_for_dim(self.full_labels, index,
                                                     0))
        selected_lt.labels = self.labels
        return selected_lt

    @staticmethod
    def _update_label_for_dim(old_labels, index, dim):
        """
        TODO
        :param old_labels:
        :param index:
        :param dim:
        :return:
        """
        if isinstance(index, torch.Tensor):
            index = index.nonzero()
        if isinstance(index, list):
            return {dim: {'dof': [old_labels[dim]['dof'][i] for i in index],
                          'name': old_labels[dim]['name']}}
        else:
            return {dim: {'dof': old_labels[dim]['dof'][index],
                          'name': old_labels[dim]['name']}}

    def sort_labels(self, dim=None):
        def argsort(lst):
            return sorted(range(len(lst)), key=lambda x: lst[x])

        if dim is None:
            dim = self.tensor.ndim - 1
        labels = self.full_labels[dim]['dof']
        sorted_index = argsort(labels)
        indexer = [slice(None)] * self.tensor.ndim
        indexer[dim] = sorted_index
        new_labels = deepcopy(self.full_labels)
        new_labels[dim] = {'dof': sorted(labels),
                           'name': new_labels[dim]['name']}
        return LabelTensor(self.tensor[indexer], new_labels)
