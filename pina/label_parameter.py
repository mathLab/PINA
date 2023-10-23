""" Module for LabelTensor """
from typing import Any
import torch
from torch import Tensor


class LabelParameter(torch.nn.Parameter):
    """Torch parameter with a label for any element."""

    @staticmethod
    def __new__(cls, x, labels, *args, **kwargs):
        return super().__new__(cls, x, *args, **kwargs)

    def __init__(self, x, labels):
        '''
        Construct a `LabelParameter` by passing a tensor and a list of
        labels. Such labels uniquely identify the elements of the tensor,
        allowing for an easier manipulation.

        :param torch.Tensor x: the tensor of parameters.
        :param labels: the labels of the elements.
        :type labels: str or iterable(str)

        :Example:
            >>> from pina import LabelParameter
            >>> tensor = LabelParameter(torch.rand(3), ['a', 'b', 'c'])
            >>> tensor
            Parameter containing:
            Parameter(LabelParameter([0.4842, 0.7341, 0.4789],
                requires_grad=True))
            >>> tensor.extract('a')
            Parameter containing:
            Parameter(LabelParameter([0.4842],
                grad_fn=<AliasBackward0>))
            >>> tensor.extract(['a', 'b'])
            Parameter containing:
            Parameter(LabelParameter([0.4842, 0.7341],
                grad_fn=<AliasBackward0>))
            >>> tensor.extract(['b', 'a'])
            Parameter containing:
            Parameter(LabelParameter([0.7341, 0.4842],
                grad_fn=<AliasBackward0>))
        '''
        if isinstance(labels, str):
            labels = [labels]

        if len(labels) != x.shape[-1]:
            raise ValueError(
                'the tensor has not the same number of elements of '
                'the passed labels.'
            )
        self._labels = labels

    @property
    def labels(self):
        """Property decorator for labels

        :return: labels of self
        :rtype: list
        """
        return self._labels

    @labels.setter
    def labels(self, labels):
        self._labels = labels   # assign the label

    def clone(self, *args, **kwargs):
        """
        Clone the LabelParameter. For more details, see
        :meth:`torch.Tensor.clone`.

        :return: a copy of the tensor
        :rtype: LabelTensor
        """
        try:
            out = LabelParameter(super().clone(*args, **kwargs), self.labels)
        except:
            out = super().clone(*args, **kwargs)

        return out

    def to(self, *args, **kwargs):
        """
        Performs Tensor dtype and/or device conversion. For more details, see
        :meth:`torch.Tensor.to`.
        """
        tmp = super().to(*args, **kwargs)
        new = self.__class__.clone(self)
        new.data = tmp.data
        return new

    def select(self, *args, **kwargs):
        """
        Performs Tensor selection. For more details, see :meth:`torch.Tensor.select`.
        """
        tmp = super().select(*args, **kwargs)
        tmp._labels = self._labels
        return tmp

    def extract(self, label_to_extract):
        """
        Extract the subset of the original tensor by returning all the elements
        corresponding to the passed `label_to_extract`.

        :param label_to_extract: the label(s) to extract.
        :type label_to_extract: str or iterable(str)
        :raises TypeError: labels are not str
        :raises ValueError: label to extract is not in the labels list
        """

        if isinstance(label_to_extract, str):
            label_to_extract = [label_to_extract]
        elif isinstance(label_to_extract, (tuple, list)):  # TODO
            pass
        else:
            raise TypeError(
                '`label_to_extract` should be a str, or a str iterator')

        indeces = []
        for f in label_to_extract:
            try:
                indeces.append(self.labels.index(f))
            except ValueError:
                raise ValueError(f'`{f}` not in the labels list')

        extracted_tensor = super(torch.nn.Parameter, self).select(0, indeces[0])
        if len(indeces) > 1:
            for ind in indeces[1:]:
                new_data = torch.cat((new_data, super(torch.nn.Parameter, self).select(0, ind)))

        # For now each extracted tensor is NOT a LabelParameter.
        # The reason is that if we use LabelParameter we have tensors with
        # grad_fn=<AliasBackward0> and the clamp ranges do not affect them.
        # If we keep torch.nn.Parameter we have tensors with
        # grad_fn=<SelectBackward0> and the clamp works properly during training.
        # TODO: let the LabelParameter work with grad_fn=<SelectBackward0>.
        new_labels = [self.labels[idx] for idx in indeces]

        extracted_tensor.labels = new_labels

        return extracted_tensor

    def append(self, lt):
        """
        Return a copy of the merged tensors of parameters.

        :param LabelParameter lt: the tensor to merge.
        :return: the merged tensors
        :rtype: LabelTensor
        """
        if set(self.labels).intersection(lt.labels):
            raise RuntimeError('The tensors to merge have common labels')

        new_labels = self.labels + lt.labels
        new_tensor = torch.cat((self, lt))

        new_tensor = new_tensor.as_subclass(LabelParameter)
        new_tensor.labels = new_labels
        return new_tensor

    def __getitem__(self, index):
        """
        Return a copy of the selected tensor.
        """
        selected_lt = super(torch.nn.Parameter, self).__getitem__(index)

        try:
            len_index = len(index)
        except TypeError:
            len_index = 1

        if isinstance(index, int) or len_index == 1:
            if selected_lt.ndim == 1:
                selected_lt = selected_lt
            if hasattr(self, 'labels'):
                selected_lt.labels = self.labels
        elif len_index == 2:
            if selected_lt.ndim == 1:
                selected_lt = selected_lt
            if hasattr(self, 'labels'):
                if isinstance(index[1], list):
                    selected_lt.labels = [self.labels[i] for i in index[1]]
                else:
                    selected_lt.labels = self.labels[index[1]]

        return selected_lt

    def __len__(self) -> int:
        return super().__len__()

    def __str__(self):
        if hasattr(self, 'labels'):
            s = f'labels({str(self.labels)})\n'
        else:
            s = 'no labels\n'
        s += super().__str__()
        return s
