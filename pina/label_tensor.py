""" Module for LabelTensor """
from copy import copy, deepcopy
import torch
from torch import Tensor


full_labels = False
MATH_FUNCTIONS = {torch.sin, torch.cos}

class LabelTensor(torch.Tensor):
    """Torch tensor with a label for any column."""

    @staticmethod
    def __new__(cls, x, labels, *args, **kwargs):
        full = kwargs.pop("full", full_labels)

        if isinstance(x, LabelTensor):
            x.full = full
            return x
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
        self.full = kwargs.get('full', full_labels)
        if labels is not None:
            self.labels = labels
        else:
            self._labels = {}

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
        elif isinstance(labels, (list, range)):
            self._init_labels_from_list(labels)
        elif isinstance(labels, str):
            labels = [labels]
            self._init_labels_from_list(labels)
        else:
            raise ValueError("labels must be list, dict or string.")

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

        # Set all labels if full_labels is True
        if hasattr(self, 'full') and self.full:
            labels = {
                i: labels[i] if i in labels else {
                    'name': i, 'dof': range(tensor_shape[i])
                }
                for i in range(len(tensor_shape))
            }

        for k, v in labels.items():

            # Init labels from str
            if isinstance(v, str):
                v = {'name': v, 'dof': range(tensor_shape[k])}

            # Init labels from dict
            elif isinstance(v, dict):
                # Only name of the dimension if provided
                if list(v.keys()) == ['name']:
                    v['dof'] = range(tensor_shape[k])
                # Both name and dof are provided
                elif sorted(list(v.keys())) == ['dof', 'name']:
                    dof_list = v['dof']
                    dof_len = len(dof_list)
                    if dof_len != len(set(dof_list)):
                        raise ValueError("dof must be unique")
                    if dof_len != tensor_shape[k]:
                        raise ValueError(
                            'Number of dof does not match tensor shape')
            else:
                raise ValueError('Illegal labels initialization')
            # Assign labels values
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

    def extract(self, labels_to_extract):
        """
        Extract the subset of the original tensor by returning all the columns
        corresponding to the passed ``label_to_extract``.

        :param labels_to_extract: The label(s) to extract.
        :type labels_to_extract: str | list(str) | tuple(str)
        :raises TypeError: Labels are not ``str``.
        :raises ValueError: Label to extract is not in the labels ``list``.
        """
        # Convert str/int to string
        def find_names(labels):
            dim_names = {}
            for dim in labels.keys():
                dim_names[labels[dim]['name']] = dim
            return dim_names

        if isinstance(labels_to_extract, (str, int)):
            labels_to_extract = [labels_to_extract]

        # Store useful variables
        labels = copy(self._labels)
        stored_keys = labels.keys()
        dim_names = find_names(labels)
        ndim = len(super().shape)

        # Convert tuple/list to dict (having a list as input
        # means that we want to extract a values from the last dimension)
        if isinstance(labels_to_extract, (tuple, list)):
            if not ndim - 1 in stored_keys:
                raise ValueError(
                    "LabelTensor does not have labels in last dimension")
            name = labels[ndim-1]['name']
            labels_to_extract = {name: list(labels_to_extract)}

        # If labels_to_extract is not dict then rise error
        if not isinstance(labels_to_extract, dict):
            raise ValueError('labels_to_extract must be str or list or dict')

        # Initialize list used to perform extraction
        extractor = [slice(None)]*ndim

        # Loop over labels_to_extract dict
        for dim_name, labels_te in labels_to_extract.items():

            # If label is not find raise value error
            idx_dim = dim_names.get(dim_name, None)
            if idx_dim is None:
                raise ValueError(
                    'Cannot extract label with is not in original labels')

            dim_labels = labels[idx_dim]['dof']
            labels_te = [labels_te] if isinstance(labels_te, (int, str)) else labels_te
            if not isinstance(labels_te, range):
                #If is done to keep the dimension if there is only one extracted label
                extractor[idx_dim] = [dim_labels.index(i) for i in labels_te] \
                    if len(labels_te)>1 else slice(dim_labels.index(labels_te[0]), dim_labels.index(labels_te[0])+1)
            else:
                extractor[idx_dim] = slice(labels_te.start, labels_te.stop)

            labels.update({idx_dim: {'dof': labels_te, 'name': dim_name}})

        tensor = super().__getitem__(extractor).as_subclass(LabelTensor)
        tensor._labels = labels
        return tensor

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

        # --------- Start definition auxiliary function ------
        # Compute and update labels
        def create_labels_cat(tensors, dim, tensor_shape):
            stored_labels = [tensor.stored_labels for tensor in tensors]
            keys = stored_labels[0].keys()

            if any(not all(stored_labels[i][k] == stored_labels[0][k] for i in
                        range(len(stored_labels))) for k in keys if k != dim):
                raise RuntimeError('tensors must have the same shape and dof')

            # Copy labels from the first tensor and update the 'dof' for dimension `dim`
            labels = copy(stored_labels[0])
            if dim in labels:
                labels_list = [tensor[dim]['dof'] for tensor in stored_labels]
                last_dim_dof = range(tensor_shape[dim]) if all(isinstance(label, range)
                                    for label in labels_list) else sum(labels_list, [])
                labels[dim]['dof'] = last_dim_dof
            return labels
        # --------- End definition auxiliary function ------

        # Update labels
        if dim in tensors[0].stored_labels.keys():
            new_tensor_shape = new_tensor.shape
            labels = create_labels_cat(tensors, dim, new_tensor_shape)
        else:
            labels = tensors[0].stored_labels
        new_tensor._labels = labels
        return new_tensor

    @staticmethod
    def stack(tensors):
        new_tensor = torch.stack(tensors)
        labels = tensors[0]._labels
        labels = {key + 1: value for key, value in labels.items()}
        if full_labels:
            new_tensor.labels = labels
        else:
            new_tensor._labels = labels
        return new_tensor

    def requires_grad_(self, mode=True):
        lt = super().requires_grad_(mode)
        lt._labels = self._labels
        return lt

    @property
    def dtype(self):
        return super().dtype

    def to(self, *args, **kwargs):
        """
        Performs Tensor dtype and/or device conversion. For more details, see
        :meth:`torch.Tensor.to`.
        """
        lt = super().to(*args, **kwargs)
        lt._labels = self._labels
        return lt

    def clone(self, *args, **kwargs):
        """
        Clone the LabelTensor. For more details, see
        :meth:`torch.Tensor.clone`.

        :return: A copy of the tensor.
        :rtype: LabelTensor
        """
        out = LabelTensor(super().clone(*args, **kwargs), deepcopy(self._labels))
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
        data = torch.zeros(tensors[0].tensor.shape).to(tensors[0].device)
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

        # ---------------------- Start auxiliary function definition -----
        # This method is used to update labels
    def _update_single_label(self, old_labels, to_update_labels, index, dim,
                             to_update_dim):
            """
            TODO
                :param old_labels: labels from which retrieve data
                :param to_update_labels: labels to update
                :param index: index of dof to retain
                :param dim: label index
                :return:
            """
            old_dof = old_labels[to_update_dim]['dof']
            if isinstance(index, slice):
                to_update_labels.update({
                    dim: {
                        'dof': old_dof[index],
                        'name': old_labels[dim]['name']
                    }
                })
                return
            if isinstance(index, int):
                index = [index]
            if isinstance(index, (list, torch.Tensor)):
                to_update_labels.update({
                    dim: {
                        'dof': [old_dof[i] for i in index] if isinstance(old_dof, list) else index,
                        'name': old_labels[dim]['name']
                    }
                })
                return
            raise NotImplementedError(f'Getitem not implemented for '
                                      f'{type(index)} values')
        # ---------------------- End auxiliary function definition -----


    def __getitem__(self, index):
        """
        TODO: Complete docstring
        :param index:
        :return:
        """
        # Index are str --> call extract
        if isinstance(index, str) or (isinstance(index, (tuple, list))
                                      and all(
                    isinstance(a, str) for a in index)):
            return self.extract(index)

        # Store important variables
        selected_lt = super().__getitem__(index)
        stored_labels = self._labels
        labels = copy(stored_labels)

        # Put here because it is the most common case (int as index).
        # Used by DataLoader -> put here for efficiency purpose
        if isinstance(index, list):
            if 0 in labels.keys():
                self._update_single_label(stored_labels, labels, index,
                                          0, 0)
            selected_lt._labels = labels
            return selected_lt

        if isinstance(index, int):
            labels.pop(0, None)
            labels = {key - 1 if key > 0 else key: value for key, value in
                      labels.items()}
            selected_lt._labels = labels
            return selected_lt

        if not isinstance(index, (tuple, torch.Tensor)):
            index = [index]

        # Ellipsis are used to perform operation on the last dimension
        if index[0] == Ellipsis:
            if len(self.shape) in labels:
                self._update_single_label(stored_labels, labels, index, 0, 0)
            selected_lt._labels = labels
            return selected_lt

        i = 0
        for j, idx in enumerate(index):
            if j in self.stored_labels.keys():
                if isinstance(idx, int) or (
                        isinstance(idx, torch.Tensor) and idx.ndim == 0):
                    selected_lt = selected_lt.unsqueeze(j)
                if idx != slice(None):
                    self._update_single_label(stored_labels, labels, idx, j, i)
            else:
                if isinstance(idx, int):
                    labels = {key - 1 if key > j else key:
                                  value for key, value in labels.items()}
                    continue
            i += 1
        selected_lt._labels = labels
        return selected_lt

    def sort_labels(self, dim=None):
        def arg_sort(lst):
            return sorted(range(len(lst)), key=lambda x: lst[x])
        if dim is None:
            dim = self.ndim - 1
        if self.shape[dim] == 1:
            return self
        labels = self.stored_labels[dim]['dof']
        sorted_index = arg_sort(labels)
        indexer = [slice(None)] * self.ndim
        indexer[dim] = sorted_index
        return self.__getitem__(tuple(indexer))

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls(deepcopy(self.tensor), deepcopy(self.stored_labels))
        return result

    def permute(self, *dims):
        tensor = super().permute(*dims)
        labels = self._labels
        keys_list = list(*dims)
        labels = {
            keys_list.index(k): labels[k]
            for k in labels.keys()
        }
        tensor._labels = labels
        return tensor

    def detach(self):
        lt = super().detach()
        lt._labels = self.stored_labels
        return lt