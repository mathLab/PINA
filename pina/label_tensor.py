""" Module for LabelTensor """

from typing import Any
import torch
from torch import Tensor


class LabelTensor(torch.Tensor):
    """Torch tensor with a label for any column."""

    @staticmethod
    def __new__(cls, x, labels, *args, **kwargs):
        return super().__new__(cls, x, *args, **kwargs)

    def __init__(self, x, labels):
        """
        Construct a :class:`~pina.label_tensor.LabelTensor` by passing a tensor and a list of column
        labels. Such labels uniquely identify the columns of the tensor,
        allowing for an easier manipulation.

        :param torch.Tensor x: The data tensor.
        :param labels: The labels of the columns.
        :type labels: str | list(str) | tuple(str)

        :Example:
            >>> from pina import LabelTensor
            >>> tensor = LabelTensor(torch.rand((2000, 3)), ['a', 'b', 'c'])
            >>> tensor
            tensor([[6.7116e-02, 4.8892e-01, 8.9452e-01],
                [9.2392e-01, 8.2065e-01, 4.1986e-04],
                [8.9266e-01, 5.5446e-01, 6.3500e-01],
                ...,
                [5.8194e-01, 9.4268e-01, 4.1841e-01],
                [1.0246e-01, 9.5179e-01, 3.7043e-02],
                [9.6150e-01, 8.0656e-01, 8.3824e-01]])
            >>> tensor.extract('a')
            tensor([[0.0671],
                    [0.9239],
                    [0.8927],
                    ...,
                    [0.5819],
                    [0.1025],
                    [0.9615]])
            >>> tensor['a']
            tensor([[0.0671],
                    [0.9239],
                    [0.8927],
                    ...,
                    [0.5819],
                    [0.1025],
                    [0.9615]])
            >>> tensor.extract(['a', 'b'])
            tensor([[0.0671, 0.4889],
                    [0.9239, 0.8207],
                    [0.8927, 0.5545],
                    ...,
                    [0.5819, 0.9427],
                    [0.1025, 0.9518],
                    [0.9615, 0.8066]])
            >>> tensor.extract(['b', 'a'])
            tensor([[0.4889, 0.0671],
                    [0.8207, 0.9239],
                    [0.5545, 0.8927],
                    ...,
                    [0.9427, 0.5819],
                    [0.9518, 0.1025],
                    [0.8066, 0.9615]])
        """
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        if isinstance(labels, str):
            labels = [labels]

        if len(labels) != x.shape[-1]:
            raise ValueError(
                "the tensor has not the same number of columns of "
                "the passed labels."
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
        if len(labels) != self.shape[self.ndim - 1]:  # small check
            raise ValueError(
                "The tensor has not the same number of columns of "
                "the passed labels."
            )

        self._labels = labels  # assign the label

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
        if len(label_tensors) == 0:
            return []

        all_labels = [label for lt in label_tensors for label in lt.labels]
        if set(all_labels) != set(label_tensors[0].labels):
            raise RuntimeError("The tensors to stack have different labels")

        labels = label_tensors[0].labels
        tensors = [lt.extract(labels) for lt in label_tensors]
        return LabelTensor(torch.vstack(tensors), labels)

    def clone(self, *args, **kwargs):
        """
        Clone the LabelTensor. For more details, see
        :meth:`torch.Tensor.clone`.

        :return: A copy of the tensor.
        :rtype: LabelTensor
        """
        # # used before merging
        # try:
        #     out = LabelTensor(super().clone(*args, **kwargs), self.labels)
        # except:
        #     out = super().clone(*args, **kwargs)
        out = LabelTensor(super().clone(*args, **kwargs), self.labels)
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

    def cuda(self, *args, **kwargs):
        """
        Send Tensor to cuda. For more details, see :meth:`torch.Tensor.cuda`.
        """
        tmp = super().cuda(*args, **kwargs)
        new = self.__class__.clone(self)
        new.data = tmp.data
        return tmp

    def cpu(self, *args, **kwargs):
        """
        Send Tensor to cpu. For more details, see :meth:`torch.Tensor.cpu`.
        """
        tmp = super().cpu(*args, **kwargs)
        new = self.__class__.clone(self)
        new.data = tmp.data
        return tmp

    def extract(self, label_to_extract):
        """
        Extract the subset of the original tensor by returning all the columns
        corresponding to the passed ``label_to_extract``.

        :param label_to_extract: The label(s) to extract.
        :type label_to_extract: str | list(str) | tuple(str)
        :raises TypeError: Labels are not ``str``.
        :raises ValueError: Label to extract is not in the labels ``list``.
        """

        if isinstance(label_to_extract, str):
            label_to_extract = [label_to_extract]
        elif isinstance(label_to_extract, (tuple, list)):  # TODO
            pass
        else:
            raise TypeError(
                "`label_to_extract` should be a str, or a str iterator"
            )

        indeces = []
        for f in label_to_extract:
            try:
                indeces.append(self.labels.index(f))
            except ValueError:
                raise ValueError(f"`{f}` not in the labels list")

        new_data = super(Tensor, self.T).__getitem__(indeces).T
        new_labels = [self.labels[idx] for idx in indeces]

        extracted_tensor = new_data.as_subclass(LabelTensor)
        extracted_tensor.labels = new_labels

        return extracted_tensor

    def detach(self):
        detached = super().detach()
        if hasattr(self, "_labels"):
            detached._labels = self._labels
        return detached

    def requires_grad_(self, mode=True):
        lt = super().requires_grad_(mode)
        lt.labels = self.labels
        return lt

    def append(self, lt, mode="std"):
        """
        Return a copy of the merged tensors.

        :param LabelTensor lt: The tensor to merge.
        :param str mode: {'std', 'first', 'cross'}
        :return: The merged tensors.
        :rtype: LabelTensor
        """
        if set(self.labels).intersection(lt.labels):
            raise RuntimeError("The tensors to merge have common labels")

        new_labels = self.labels + lt.labels
        if mode == "std":
            new_tensor = torch.cat((self, lt), dim=1)
        elif mode == "first":
            raise NotImplementedError
        elif mode == "cross":
            tensor1 = self
            tensor2 = lt
            n1 = tensor1.shape[0]
            n2 = tensor2.shape[0]

            tensor1 = LabelTensor(tensor1.repeat(n2, 1), labels=tensor1.labels)
            tensor2 = LabelTensor(
                tensor2.repeat_interleave(n1, dim=0), labels=tensor2.labels
            )
            new_tensor = torch.cat((tensor1, tensor2), dim=1)

        new_tensor = new_tensor.as_subclass(LabelTensor)
        new_tensor.labels = new_labels
        return new_tensor

    def __getitem__(self, index):
        """
        Return a copy of the selected tensor.
        """

        if isinstance(index, str) or (
            isinstance(index, (tuple, list))
            and all(isinstance(a, str) for a in index)
        ):
            return self.extract(index)

        selected_lt = super(Tensor, self).__getitem__(index)

        try:
            len_index = len(index)
        except TypeError:
            len_index = 1

        if isinstance(index, int) or len_index == 1:
            if selected_lt.ndim == 1:
                selected_lt = selected_lt.reshape(1, -1)
            if hasattr(self, "labels"):
                selected_lt.labels = self.labels
        elif len_index == 2:
            if selected_lt.ndim == 1:
                selected_lt = selected_lt.reshape(-1, 1)
            if hasattr(self, "labels"):
                if isinstance(index[1], list):
                    selected_lt.labels = [self.labels[i] for i in index[1]]
                else:
                    selected_lt.labels = self.labels[index[1]]
        else:
            selected_lt.labels = self.labels

        return selected_lt

    @property
    def tensor(self):
        return self.as_subclass(Tensor)

    def __len__(self) -> int:
        return super().__len__()

    def __str__(self):
        if hasattr(self, "labels"):
            s = f"labels({str(self.labels)})\n"
        else:
            s = "no labels\n"
        s += super().__str__()
        return s
