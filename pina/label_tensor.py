""" Module for LabelTensor """
from copy import copy, deepcopy
import torch
from torch import Tensor


def issubset(a, b):
    """
    Check if a is a subset of b.
    """
    if isinstance(a, list) and isinstance(b, list):
        return set(a).issubset(set(b))
    if isinstance(a, range) and isinstance(b, range):
        return a.start <= b.start and a.stop >= b.stop
    return False


class LabelTensor(torch.Tensor):
    """Torch tensor with a label for any column."""

    @staticmethod
    def __new__(cls, x, labels, *args, **kwargs):
        if isinstance(x, LabelTensor):
            return x
        else:
            return super().__new__(cls, x, *args, **kwargs)

    @property
    def tensor(self):
        return self.as_subclass(Tensor)

    def __init__(self, x, labels, **kwargs):
        """
        Construct a `LabelTensor` by passing a dict of the labels

        :Example:
            >>> from pina import LabelTensor
            >>> tensor = LabelTensor(
            >>>     torch.rand((2000, 3)),
                    {1: {"name": "space"['a', 'b', 'c'])

        """
        self.dim_names = None
        self.full = kwargs.get('full', True)
        self.labels = labels

    @classmethod
    def __internal_init__(cls,
                          x,
                          labels,
                          dim_names,
                          *args,
                          **kwargs):
        lt = cls.__new__(cls, x, labels, *args, **kwargs)
        lt._labels = labels
        lt.full = kwargs.get('full', True)
        lt.dim_names = dim_names
        return lt

    @property
    def labels(self):
        """Property decorator for labels

        :return: labels of self
        :rtype: list
        """
        if self.ndim - 1 in self._labels.keys():
            return self._labels[self.ndim - 1]['dof']

    @property
    def full_labels(self):
        """Property decorator for labels

        :return: labels of self
        :rtype: list
        """
        to_return_dict = {}
        shape_tensor = self.shape
        for i in range(len(shape_tensor)):
            if i in self._labels.keys():
                to_return_dict[i] = self._labels[i]
            else:
                to_return_dict[i] = {'dof': range(shape_tensor[i]), 'name': i}
        return to_return_dict

    @property
    def stored_labels(self):
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
        if not hasattr(self, '_labels'):
            self._labels = {}
        if isinstance(labels, dict):
            self._init_labels_from_dict(labels)
        elif isinstance(labels, list):
            self._init_labels_from_list(labels)
        elif isinstance(labels, str):
            labels = [labels]
            self._init_labels_from_list(labels)
        else:
            raise ValueError("labels must be list, dict or string.")
        self.set_names()

    def _init_labels_from_dict(self, labels):
        """
            Update the internal label representation according to the values
            passed as input.

            :param labels: The label(s) to update.
            :type labels: dict
            :raises ValueError: dof list contain duplicates or number of dof
            does not match with tensor shape
            """
        tensor_shape = self.shape

        if hasattr(self, 'full') and self.full:
            labels = {
                i: labels[i] if i in labels else {
                    'name': i
                }
                for i in labels.keys()
            }
        for k, v in labels.items():
            # Init labels from str
            if isinstance(v, str):
                v = {'name': v, 'dof': range(tensor_shape[k])}
            # Init labels from dict
            elif isinstance(v, dict) and list(v.keys()) == ['name']:
                # Init from dict with only name key
                v['dof'] = range(tensor_shape[k])
                # Init from dict with both name and dof keys
            elif isinstance(v, dict) and sorted(list(
                    v.keys())) == ['dof', 'name']:
                dof_list = v['dof']
                dof_len = len(dof_list)
                if dof_len != len(set(dof_list)):
                    raise ValueError("dof must be unique")
                if dof_len != tensor_shape[k]:
                    raise ValueError(
                        'Number of dof does not match tensor shape')
            else:
                raise ValueError('Illegal labels initialization')
            # Perform update
            self._labels[k] = v

    def _init_labels_from_list(self, labels):
        """
        Given a list of dof, this method update the internal label
        representation

        :param labels: The label(s) to update.
        :type labels: list
        """
        # Create a dict with labels
        last_dim_labels = {
            self.ndim - 1: {
                'dof': labels,
                'name': self.ndim - 1
            }
        }
        self._init_labels_from_dict(last_dim_labels)

    def set_names(self):
        labels = self.stored_labels
        self.dim_names = {}
        for dim in labels.keys():
            self.dim_names[labels[dim]['name']] = dim

    def extract(self, labels_to_extract):
        """
        Extract the subset of the original tensor by returning all the columns
        corresponding to the passed ``label_to_extract``.

        :param label_to_extract: The label(s) to extract.
        :type label_to_extract: str | list(str) | tuple(str)
        :raises TypeError: Labels are not ``str``.
        :raises ValueError: Label to extract is not in the labels ``list``.
        """
        # Convert str/int to string
        if isinstance(labels_to_extract, (str, int)):
            labels_to_extract = [labels_to_extract]

        # Store useful variables
        labels = self.stored_labels
        stored_keys = labels.keys()
        dim_names = self.dim_names
        ndim = len(super().shape)

        # Convert tuple/list to dict
        if isinstance(labels_to_extract, (tuple, list)):
            if not ndim - 1 in stored_keys:
                raise ValueError(
                    "LabelTensor does not have labels in last dimension")
            name = labels[max(stored_keys)]['name']
            labels_to_extract = {name: list(labels_to_extract)}

        # If labels_to_extract is not dict then rise error
        if not isinstance(labels_to_extract, dict):
            raise ValueError('labels_to_extract must be str or list or dict')

        # Make copy of labels (avoid issue in consistency)
        updated_labels = {k: copy(v) for k, v in labels.items()}

        # Initialize list used to perform extraction
        extractor = [slice(None) for _ in range(ndim)]

        # Loop over labels_to_extract dict
        for k, v in labels_to_extract.items():

            # If label is not find raise value error
            idx_dim = dim_names.get(k)
            if idx_dim is None:
                raise ValueError(
                    'Cannot extract label with is not in original labels')

            dim_labels = labels[idx_dim]['dof']
            v = [v] if isinstance(v, (int, str)) else v

            if not isinstance(v, range):
                extractor[idx_dim] = [dim_labels.index(i)
                                      for i in v] if len(v) > 1 else slice(
                                          dim_labels.index(v[0]),
                                          dim_labels.index(v[0]) + 1)
            else:
                extractor[idx_dim] = slice(v.start, v.stop)

            updated_labels.update({idx_dim: {'dof': v, 'name': k}})

        tensor = self.tensor
        tensor = tensor[extractor]
        return LabelTensor.__internal_init__(tensor, updated_labels, dim_names)

    def __str__(self):
        """
        returns a string with the representation of the class
        """
        s = ''
        for key, value in self._labels.items():
            s += f"{key}: {value}\n"
        s += '\n'
        s += self.tensor.__str__()
        return s

    @staticmethod
    def cat(tensors, dim=0):
        """
        Stack a list of tensors. For example, given a tensor `a` of shape
        `(n,m,dof)` and a tensor `b` of dimension `(n',m,dof)`
        the resulting tensor is of shape `(n+n',m,dof)`

        :param tensors: tensors to concatenate
        :type tensors: list(LabelTensor)
        :param dim: dimensions on which you want to perform the operation
        (default 0)
        :type dim: int
        :rtype: LabelTensor
        :raises ValueError: either number dof or dimensions names differ
        """
        if len(tensors) == 0:
            return []
        if len(tensors) == 1 or isinstance(tensors, LabelTensor):
            return tensors[0]
        # Perform cat on tensors
        new_tensor = torch.cat(tensors, dim=dim)

        # Update labels
        labels = LabelTensor.__create_labels_cat(tensors, dim)

        return LabelTensor.__internal_init__(new_tensor, labels,
                                             tensors[0].dim_names)

    @staticmethod
    def __create_labels_cat(tensors, dim):
        # Check if names and dof of the labels are the same in all dimensions
        # except in dim
        stored_labels = [tensor.stored_labels for tensor in tensors]

        # check if:
        # - labels dict have same keys
        # - all labels are the same expect for dimension dim
        if not all(
                all(stored_labels[i][k] == stored_labels[0][k]
                    for i in range(len(stored_labels)))
                for k in stored_labels[0].keys() if k != dim):
            raise RuntimeError('tensors must have the same shape and dof')

        labels = {k: copy(v) for k, v in tensors[0].stored_labels.items()}
        if dim in labels.keys():
            last_dim_dof = [i for j in stored_labels for i in j[dim]['dof']]
            labels[dim]['dof'] = last_dim_dof
        return labels

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
        labels = {k: copy(v) for k, v in self._labels.items()}
        out = LabelTensor(super().clone(*args, **kwargs), labels)
        return out

    @staticmethod
    def summation(tensors):
        if len(tensors) == 0:
            raise ValueError('tensors list must not be empty')
        if len(tensors) == 1:
            return tensors[0]
        # Collect all labels

        # Check labels of all the tensors in each dimension
        if not all(tensor.shape == tensors[0].shape for tensor in tensors) or \
                not all(tensor.full_labels[i] == tensors[0].full_labels[i] for
                        tensor in tensors for i in range(tensors[0].ndim - 1)):
            raise RuntimeError('Tensors must have the same shape and labels')

        last_dim_labels = []
        data = torch.zeros(tensors[0].tensor.shape)
        for tensor in tensors:
            data += tensor.tensor
            last_dim_labels.append(tensor.labels)

        last_dim_labels = ['+'.join(items) for items in zip(*last_dim_labels)]
        labels = {k: copy(v) for k, v in tensors[0].stored_labels.items()}
        labels.update({
            tensors[0].ndim - 1: {
                'dof': last_dim_labels,
                'name': tensors[0].name
            }
        })
        return LabelTensor(data, labels)

    def append(self, tensor, mode='std'):
        if mode == 'std':
            # Call cat on last dimension
            new_label_tensor = LabelTensor.cat([self, tensor],
                                               dim=self.ndim - 1)
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
                                               dim=self.ndim - 1)
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
        if isinstance(index,
                      str) or (isinstance(index, (tuple, list))
                               and all(isinstance(a, str) for a in index)):
            return self.extract(index)

        selected_lt = super().__getitem__(index)

        if isinstance(index, (int, slice)):
            index = [index]

        if index[0] == Ellipsis:
            index = [slice(None)] * (self.ndim - 1) + [index[1]]

        if hasattr(self, "labels"):
            labels = {k: copy(v) for k, v in self.stored_labels.items()}
            for j, idx in enumerate(index):
                if isinstance(idx, int):
                    selected_lt = selected_lt.unsqueeze(j)
                if j in labels.keys() and idx != slice(None):
                    self._update_single_label(labels, labels, idx, j)
            selected_lt = LabelTensor.__internal_init__(selected_lt, labels,
                                                        self.dim_names)
        return selected_lt

    @staticmethod
    def _update_single_label(old_labels, to_update_labels, index, dim):
        """
        TODO
        :param old_labels: labels from which retrieve data
        :param to_update_labels: labels to update
        :param index: index of dof to retain
        :param dim: label index
        :return:
        """
        old_dof = old_labels[dim]['dof']
        if not isinstance(
                index,
            (int, slice)) and len(index) == len(old_dof) and isinstance(
                old_dof, range):
            return
        if isinstance(index, torch.Tensor):
            index = index.nonzero(
                as_tuple=True
            )[0] if index.dtype == torch.bool else index.tolist()
        if isinstance(index, list):
            to_update_labels.update({
                dim: {
                    'dof': [old_dof[i] for i in index],
                    'name': old_labels[dim]['name']
                }
            })
        else:
            to_update_labels.update(
                {dim: {
                    'dof': old_dof[index],
                    'name': old_labels[dim]['name']
                }})

    def sort_labels(self, dim=None):

        def arg_sort(lst):
            return sorted(range(len(lst)), key=lambda x: lst[x])

        if dim is None:
            dim = self.ndim - 1
        labels = self.stored_labels[dim]['dof']
        sorted_index = arg_sort(labels)
        indexer = [slice(None)] * self.ndim
        indexer[dim] = sorted_index
        return self.__getitem__(indexer)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls(deepcopy(self.tensor), deepcopy(self.stored_labels))
        return result

    def permute(self, *dims):
        tensor = super().permute(*dims)
        stored_labels = self.stored_labels
        keys_list = list(*dims)
        labels = {
            keys_list.index(k): copy(stored_labels[k])
            for k in stored_labels.keys()
        }
        return LabelTensor.__internal_init__(tensor, labels, self.dim_names)
