"""Module for LabelTensor"""

from copy import copy, deepcopy
import torch
from torch import Tensor


class LabelTensor(torch.Tensor):
    """
    Extension of the :class:`torch.Tensor` class that includes labels for
    each dimension.
    """

    @staticmethod
    def __new__(cls, x, labels, *args, **kwargs):
        """
        Create a new instance of the :class:`~pina.label_tensor.LabelTensor`
        class.

        :param torch.Tensor x: :class:`torch.tensor` instance to be casted as a
            :class:`~pina.label_tensor.LabelTensor`.
        :param labels: Labels to assign to the tensor.
        :type labels: str | list[str] | dict
        :return: The instance of the :class:`~pina.label_tensor.LabelTensor`
            class.
        :rtype: LabelTensor
        """

        if isinstance(x, LabelTensor):
            return x
        return super().__new__(cls, x, *args, **kwargs)

    @property
    def tensor(self):
        """
        Give the tensor part of the :class:`~pina.label_tensor.LabelTensor`
        object.

        :return: tensor part of the :class:`~pina.label_tensor.LabelTensor`.
        :rtype: torch.Tensor
        """

        return self.as_subclass(Tensor)

    def __init__(self, x, labels):
        """
        Construct a :class:`~pina.label_tensor.LabelTensor` by passing a dict of
        the labels and a :class:`torch.Tensor`. Internally, the initialization
        method will store check the compatibility of the labels with the tensor
        shape.

        :Example:
            >>> from pina import LabelTensor
            >>> tensor = LabelTensor(
            >>>     torch.rand((2000, 3)),
            ...     {1: {"name": "space", "dof": ['a', 'b', 'c'])
            >>> tensor = LabelTensor(
            >>>     torch.rand((2000, 3)),
            ...     ["a", "b", "c"])

        """
        # Avoid unused argument warning. x is not used in the constructor
        # of the parent class.
        # pylint: disable=unused-argument
        super().__init__()
        if labels is not None:
            self.labels = labels
        else:
            self._labels = {}

    @property
    def full_labels(self):
        """
        Gives the full labels of the tensor, even for the dimensions that are
        not labeled.

        :return: The full labels of the tensor
        :rtype: dict
        """
        to_return_dict = {}
        shape_tensor = self.shape
        for i, value in enumerate(shape_tensor):
            if i in self._labels:
                to_return_dict[i] = self._labels[i]
            else:
                to_return_dict[i] = {"dof": range(value), "name": i}
        return to_return_dict

    @property
    def stored_labels(self):
        """
        Gives the labels stored inside the instance.

        :return: The labels stored inside the instance.
        :rtype: dict
        """
        return self._labels

    @property
    def labels(self):
        """
        Give the labels of the last dimension of the instance.

        :return: labels of last dimension
        :rtype: list
        """
        if self.ndim - 1 in self._labels:
            return self._labels[self.ndim - 1]["dof"]
        return None

    @labels.setter
    def labels(self, labels):
        """
        Set the parameter ``_labels`` by checking the type of the input labels
        and handling it accordingly. The following types are accepted:

        - **list**: The list of labels is assigned to the last dimension.
        - **dict**: The dictionary of labels is assigned to the tensor.
        - **str**: The string is assigned to the last dimension.

        :param labels: Labels to assign to the class variable _labels.
        :type labels: str | list[str] | dict
        """

        if not hasattr(self, "_labels"):
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

    def _init_labels_from_dict(self, labels: dict):
        """
        Store the internal label representation according to the values
        passed as input.

        :param dict labels: The label(s) to update.
        :raises ValueError: If the dof list contains duplicates or the number of
                            dof does not match the tensor shape.
        """

        tensor_shape = self.shape

        def validate_dof(dof_list, dim_size: int):
            """Validate the 'dof' list for uniqueness and size."""
            if len(dof_list) != len(set(dof_list)):
                raise ValueError("dof must be unique")
            if len(dof_list) != dim_size:
                raise ValueError(
                    f"Number of dof ({len(dof_list)}) does not match "
                    f"tensor shape ({dim_size})"
                )

        for dim, label in labels.items():
            if isinstance(label, dict):
                if "name" not in label:
                    label["name"] = dim
                if "dof" not in label:
                    label["dof"] = range(tensor_shape[dim])
                if "dof" in label and "name" in label:
                    dof = label["dof"]
                    dof_list = dof if isinstance(dof, (list, range)) else [dof]
                    if not isinstance(dof_list, (list, range)):
                        raise ValueError(
                            f"'dof' should be a list or range, not"
                            f" {type(dof_list)}"
                        )
                    validate_dof(dof_list, tensor_shape[dim])
                else:
                    raise ValueError(
                        "Labels dictionary must contain either "
                        " both 'name' and 'dof' keys"
                    )
            else:
                raise ValueError(
                    f"Invalid label format for {dim}: Expected "
                    f"list or dictionary, got {type(label)}"
                )

            # Assign validated label data to internal labels
            self._labels[dim] = label

    def _init_labels_from_list(self, labels):
        """
        Given a ``list`` of dof, this method update the internal label
        representation by assigning the dof to the last dimension.

        :param labels: The label(s) to update.
        :type labels: list
        """

        # Create a dict with labels
        last_dim_labels = {
            self.ndim - 1: {"dof": labels, "name": self.ndim - 1}
        }
        self._init_labels_from_dict(last_dim_labels)

    def extract(self, labels_to_extract):
        """
        Extract the subset of the original tensor by returning all the positions
        corresponding to the passed ``label_to_extract``.

        :param labels_to_extract: The label(s) to extract. If a single label or
            a list of labels is passed, the last dimension is considered.
            If a dictionary is passed, the keys are the dimension names and the
            values are the labels to extract.
        :type labels_to_extract: str | list[str] | tuple[str] | dict
        :return: The extracted tensor with the updated labels.
        :rtype: LabelTensor

        :raises TypeError: Labels are not ``str``, ``list of str`` or ``dict``
            properly setted.
        :raises ValueError: Label to extract is not in the labels ``list``.
        """

        def get_label_indices(dim_labels, labels_te):
            if isinstance(labels_te, (int, str)):
                labels_te = [labels_te]
            return (
                [dim_labels.index(label) for label in labels_te]
                if len(labels_te) > 1
                else slice(
                    dim_labels.index(labels_te[0]),
                    dim_labels.index(labels_te[0]) + 1,
                )
            )

        # Ensure labels_to_extract is a list or dict
        if isinstance(labels_to_extract, (str, int)):
            labels_to_extract = [labels_to_extract]

        labels = copy(self._labels)

        # Get the dimension names and the respective dimension index
        dim_names = {labels[dim]["name"]: dim for dim in labels}
        ndim = super().ndim
        tensor = self.tensor.as_subclass(torch.Tensor)

        # Convert list/tuple to a dict for the last dimension if applicable
        if isinstance(labels_to_extract, (list, tuple)):
            last_dim = ndim - 1
            dim_name = labels[last_dim]["name"]
            labels_to_extract = {dim_name: list(labels_to_extract)}

        # Validate the labels_to_extract type
        if not isinstance(labels_to_extract, dict):
            raise ValueError(
                "labels_to_extract must be a string, list, or dictionary."
            )

        # Perform the extraction for each specified dimension
        for dim_name, labels_te in labels_to_extract.items():
            if dim_name not in dim_names:
                raise ValueError(
                    f"Cannot extract labels for dimension '{dim_name}' as it is"
                    f" not present in the original labels."
                )

            idx_dim = dim_names[dim_name]
            dim_labels = labels[idx_dim]["dof"]
            indices = get_label_indices(dim_labels, labels_te)

            extractor = [slice(None)] * ndim
            extractor[idx_dim] = indices
            tensor = tensor[tuple(extractor)]

            labels[idx_dim] = {"dof": labels_te, "name": dim_name}

        return LabelTensor(tensor, labels)

    def __str__(self):
        """
        The string representation of the :class:`~pina.label_tensor.LabelTensor`.

        :return: String representation of the
            :class:`~pina.label_tensor.LabelTensor` instance.
        :rtype: str
        """

        s = ""
        for key, value in self._labels.items():
            s += f"{key}: {value}\n"
        s += "\n"
        s += self.tensor.__str__()
        return s

    @staticmethod
    def cat(tensors, dim=0):
        """
        Concatenate a list of tensors along a specified dimension. For more
        details, see :meth:`torch.cat`.

        :param list[LabelTensor] tensors: :class:`~pina.label_tensor.LabelTensor`
            instances to concatenate
        :param int dim: dimensions on which you want to perform the operation
            (default is 0)
        :return: A new :class:`LabelTensor' instance obtained by concatenating
            the input instances, with the updated labels.

        :rtype: LabelTensor
        :raises ValueError: either number dof or dimensions names differ
        """

        if not tensors:
            return []  # Handle empty list
        if len(tensors) == 1:
            return tensors[0]  # Return single tensor as-is

            # Perform concatenation
        cat_tensor = torch.cat(tensors, dim=dim)
        tensors_labels = [tensor.stored_labels for tensor in tensors]

        # Check label consistency across tensors, excluding the
        # concatenation dimension
        for key in tensors_labels[0]:
            if key != dim:
                if any(
                    tensors_labels[i][key] != tensors_labels[0][key]
                    for i in range(len(tensors_labels))
                ):
                    raise RuntimeError(
                        f"Tensors must have the same labels along all "
                        f"dimensions except {dim}."
                    )

        # Copy and update the 'dof' for the concatenation dimension
        cat_labels = {k: copy(v) for k, v in tensors_labels[0].items()}

        # Update labels if the concatenation dimension has labels
        if dim in tensors[0].stored_labels:
            if dim in cat_labels:
                cat_dofs = [label[dim]["dof"] for label in tensors_labels]
                cat_labels[dim]["dof"] = sum(cat_dofs, [])
        else:
            cat_labels = tensors[0].stored_labels

        # Assign updated labels to the concatenated tensor
        cat_tensor._labels = cat_labels
        return cat_tensor

    @staticmethod
    def stack(tensors):
        """
        Stacks a list of tensors along a new dimension. For more details, see
        :meth:`torch.stack`.

        :param list[LabelTensor] tensors: A list of tensors to stack.
            All tensors must have the same shape.
        :return: A new :class:`~pina.label_tensor.LabelTensor` instance obtained
            by stacking the input tensors, with the updated labels.
        :rtype: LabelTensor
        """

        # Perform stacking in torch
        new_tensor = torch.stack(tensors)

        # Increase labels keys by 1
        labels = tensors[0]._labels
        labels = {key + 1: value for key, value in labels.items()}
        new_tensor._labels = labels
        return new_tensor

    def requires_grad_(self, mode=True):
        """
        Override the requires_grad_ method to handle the labels in the new
        tensor. For more details, see :meth:`torch.Tensor.requires_grad_`.

        :param bool mode: A boolean value indicating whether the tensor should
            track gradients.If `True`, the tensor will track gradients;
            if `False`, it will not.
        :return: The :class:`~pina.label_tensor.LabelTensor` itself with the
            updated `requires_grad` state and retained labels.
        :rtype: LabelTensor
        """

        lt = super().requires_grad_(mode)
        lt._labels = self._labels
        return lt

    @property
    def dtype(self):
        """
        Give the ``dtype`` of the tensor. For more details, see
        :meth:`torch.dtype`.

        :return: dtype of the tensor
        :rtype: torch.dtype
        """

        return super().dtype

    def to(self, *args, **kwargs):
        """
        Performs Tensor dtype and/or device conversion. For more details, see
        :meth:`torch.Tensor.to`.

        :return: A new :class:`~pina.label_tensor.LabelTensor` instance with the
            updated dtype and/or device and retained labels.
        :rtype: LabelTensor
        """

        lt = super().to(*args, **kwargs)
        lt._labels = self._labels
        return lt

    def clone(self, *args, **kwargs):
        """
        Clone the :class:`~pina.label_tensor.LabelTensor`. For more details, see
        :meth:`torch.Tensor.clone`.

        :return: A new :class:`~pina.label_tensor.LabelTensor` instance with the
            same data and labels but allocated in a different memory location.
        :rtype: LabelTensor
        """

        out = LabelTensor(
            super().clone(*args, **kwargs), deepcopy(self._labels)
        )
        return out

    def append(self, tensor, mode="std"):
        """
        Appends a given tensor to the current tensor along the last dimension.

        This method supports two types of appending operations:
        1. **Standard append** ("std"): Concatenates the input tensor with the
            current tensor along the last dimension.
        2. **Cross append** ("cross"): Creates a cross-product of the current
            tensor and the input tensor by repeating them in a cross-product
            fashion, then concatenates the result along the last dimension.

        :param tensor: The tensor to append to the current tensor.
        :type tensor: LabelTensor
        :param mode: The append mode to use. Defaults to "std".
        :type mode: str, optional
        :return: A new `LabelTensor` obtained by appending the input tensor.
        :rtype: LabelTensor

        :raises ValueError: If the mode is not "std" or "cross".
        """

        if mode == "std":
            # Call cat on last dimension
            new_label_tensor = LabelTensor.cat(
                [self, tensor], dim=self.ndim - 1
            )
            return new_label_tensor
        if mode == "cross":
            # Crete tensor and call cat on last dimension
            tensor1 = self
            tensor2 = tensor
            n1 = tensor1.shape[0]
            n2 = tensor2.shape[0]
            tensor1 = LabelTensor(tensor1.repeat(n2, 1), labels=tensor1.labels)
            tensor2 = LabelTensor(
                tensor2.repeat_interleave(n1, dim=0), labels=tensor2.labels
            )
            new_label_tensor = LabelTensor.cat(
                [tensor1, tensor2], dim=self.ndim - 1
            )
            return new_label_tensor
        raise ValueError('mode must be either "std" or "cross"')

    @staticmethod
    def vstack(label_tensors):
        """
        Stack tensors vertically. For more details, see :meth:`torch.vstack`.

        :param list of LabelTensor label_tensors: The
            :class:`~pina.label_tensor.LabelTensor` instances to stack. They need
            to have equal labels.
        :return: A new :class:`~pina.label_tensor.LabelTensor` instance obtained
            by stacking the input tensors vertically.
        :rtype: LabelTensor
        """

        return LabelTensor.cat(label_tensors, dim=0)

    # This method is used to update labels
    def _update_single_label(
        self, old_labels, to_update_labels, index, dim, to_update_dim
    ):
        """
        Update the labels of the tensor based on the index (or list of indices).

        :param dict old_labels: Labels from which retrieve data.
        :param dict to_update_labels: Labels to update.
        :param index: Index of dof to retain.
        :type index: int | slice | list[int] | tuple[int] | torch.Tensor
        :param int dim: The dimension to update.

        :raises: ValueError: If the index type is not supported.
        """

        old_dof = old_labels[to_update_dim]["dof"]
        label_name = old_labels[dim]["name"]
        # Handle slicing
        if isinstance(index, slice):
            to_update_labels[dim] = {"dof": old_dof[index], "name": label_name}
        # Handle single integer index
        elif isinstance(index, int):
            to_update_labels[dim] = {
                "dof": [old_dof[index]],
                "name": label_name,
            }
        # Handle lists or tensors
        elif isinstance(index, (list, torch.Tensor)):
            # Handle list of bools
            if isinstance(index, torch.Tensor) and index.dtype == torch.bool:
                index = index.nonzero().squeeze()
            to_update_labels[dim] = {
                "dof": (
                    [old_dof[i] for i in index]
                    if isinstance(old_dof, list)
                    else index
                ),
                "name": label_name,
            }
        else:
            raise NotImplementedError(
                f"Unsupported index type: {type(index)}. Expected slice, int, "
                f"list, or torch.Tensor."
            )

    def __getitem__(self, index):
        """ "
        Override the __getitem__ method to handle the labels of the
        :class:`~pina.label_tensor.LabelTensor` instance. It first performs
        __getitem__ operation on the :class:`torch.Tensor` part of the instance,
        then updates the labels based on the index.

        :param index: The index used to access the item
        :type index: int | str | tuple of int  | list ot int | torch.Tensor
        :return: A new :class:`~pina.label_tensor.LabelTensor` instance obtained
            `__getitem__` operation on :class:`torch.Tensor` part of the
            instance, with the updated labels.
        :rtype: LabelTensor

        :raises KeyError: If an invalid label index is provided.
        :raises IndexError: If an invalid index is accessed in the tensor.
        """

        # Handle string index
        if isinstance(index, str) or (
            isinstance(index, (tuple, list))
            and all(isinstance(i, str) for i in index)
        ):
            return self.extract(index)

        # Retrieve selected tensor and labels
        selected_tensor = super().__getitem__(index)
        if not hasattr(self, "_labels"):
            return selected_tensor

        original_labels = self._labels
        updated_labels = copy(original_labels)

        # Ensure the index is iterable
        if not isinstance(index, tuple):
            index = [index]

        # Update labels based on the index
        offset = 0
        for dim, idx in enumerate(index):
            if dim in self.stored_labels:
                if isinstance(idx, int):
                    selected_tensor = selected_tensor.unsqueeze(dim)
                if idx != slice(None):
                    self._update_single_label(
                        original_labels, updated_labels, idx, dim, offset
                    )
            else:
                # Adjust label keys if dimension is reduced (case of integer
                # index on a non-labeled dimension)
                if isinstance(idx, int):
                    updated_labels = {
                        key - 1 if key > dim else key: value
                        for key, value in updated_labels.items()
                    }
                    continue
            offset += 1

        # Update the selected tensor's labels
        selected_tensor._labels = updated_labels
        return selected_tensor

    def sort_labels(self, dim=None):
        """
        Sort the labels along the specified dimension and apply the same sorting
        to the :class:`torch.Tensor` part of the instance.

        :param int dim: The dimension along which to sort the labels.
            If ``None``, the last dimension (``ndim - 1``) is used.
        :return: A new tensor with sorted labels along the specified dimension.
        :rtype: LabelTensor
        """

        def arg_sort(lst):
            return sorted(range(len(lst)), key=lambda x: lst[x])

        if dim is None:
            dim = self.ndim - 1
        if self.shape[dim] == 1:
            return self
        labels = self.stored_labels[dim]["dof"]
        sorted_index = arg_sort(labels)
        # Define an indexer to sort the tensor along the specified dimension
        indexer = [slice(None)] * self.ndim
        # Assigned the sorted index to the specified dimension
        indexer[dim] = sorted_index
        return self[tuple(indexer)]

    def __deepcopy__(self, memo):
        """
        Creates a deep copy of the object. For more details, see
        :meth:`copy.deepcopy`.

        :param memo: LabelTensor object to be copied.
        :type memo: LabelTensor
        :return: A deep copy of the original LabelTensor object.
        :rtype: LabelTensor
        """

        cls = self.__class__
        result = cls(deepcopy(self.tensor), deepcopy(self.stored_labels))
        return result

    def permute(self, *dims):
        """
        Permutes the dimensions of the tensor and the associated labels
        accordingly. For more details, see :meth:`torch.Tensor.permute`.

        :param dims: The dimensions to permute the tensor to.
        :type dims: tuple[int] | list[int]
        :return: A new object with permuted dimensions and reordered labels.
        :rtype: LabelTensor
        """
        # Call the base class permute method
        tensor = super().permute(*dims)

        # Update lables
        labels = self._labels
        keys_list = list(*dims)
        labels = {keys_list.index(k): v for k, v in labels.items()}

        # Assign labels to the new tensor
        tensor._labels = labels
        return tensor

    def detach(self):
        """
        Detaches the tensor from the computation graph and retains the stored
        labels. For more details, see :meth:`torch.Tensor.detach`.

        :return: A new tensor detached from the computation graph.
        :rtype: LabelTensor
        """

        lt = super().detach()

        # Copy the labels to the new tensor only if present
        if hasattr(self, "_labels"):
            lt._labels = self.stored_labels
        return lt

    @staticmethod
    def summation(tensors):
        """
        Computes the summation of a list of
        :class:`~pina.label_tensor.LabelTensor` instances.


        :param list[LabelTensor] tensors: A list of tensors to sum. All
            tensors must have the same shape and labels.
        :return: A new `LabelTensor` containing the element-wise sum of the
                 input tensors.
        :rtype: LabelTensor

        :raises ValueError: If the input `tensors` list is empty.
        :raises RuntimeError: If the tensors have different shapes and/or
                              mismatched labels.
        """

        if not tensors:
            raise ValueError("The tensors list must not be empty.")

        if len(tensors) == 1:
            return tensors[0]

        # Initialize result tensor and labels
        data = torch.zeros_like(tensors[0].tensor).to(tensors[0].device)
        last_dim_labels = []

        # Accumulate tensors
        for tensor in tensors:
            data += tensor.tensor
            last_dim_labels.append(tensor.labels)

        # Construct last dimension labels
        last_dim_labels = ["+".join(items) for items in zip(*last_dim_labels)]

        # Update the labels for the resulting tensor
        labels = {k: copy(v) for k, v in tensors[0].stored_labels.items()}
        labels[tensors[0].ndim - 1] = {
            "dof": last_dim_labels,
            "name": tensors[0].name,
        }

        return LabelTensor(data, labels)

    def reshape(self, *shape):
        """
        Override the reshape method to update the labels of the tensor.
        For more details, see :meth:`torch.Tensor.reshape`.

        :param tuple of int shape: The new shape of the tensor.
        :return: A new :class:`~pina.label_tensor.LabelTensor` instance with the
            updated shape and labels.
        :rtype: LabelTensor
        """

        # As for now the reshape method is used only in the context of the
        # dataset, the labels are not
        tensor = super().reshape(*shape)
        if not hasattr(self, "_labels") or shape != (-1, *self.shape[2:]):
            return tensor
        tensor.labels = self.labels
        return tensor
