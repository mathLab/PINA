"""Module for OrthogonalBlock."""

import torch
from ...utils import check_consistency


class OrthogonalBlock(torch.nn.Module):
    """
    Module to make the input orthonormal.
    The module takes a tensor of size :math:`[N, M]` and returns a tensor of
    size :math:`[N, M]` where the columns are orthonormal. The block performs a
    Gram Schmidt orthogonalization process for the input, see
    `here <https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process>` for
    details.
    """

    def __init__(self, dim=-1, requires_grad=True):
        """
        Initialize the OrthogonalBlock module.

        :param int dim: The dimension where to orthogonalize.
        :param bool requires_grad: If autograd should record operations on
            the returned tensor, defaults to True.
        """
        super().__init__()
        # store dim
        self.dim = dim
        # store requires_grad
        check_consistency(requires_grad, bool)
        self._requires_grad = requires_grad

    def forward(self, X):
        """
        Forward pass of the OrthogonalBlock module using a Gram-Schmidt
        algorithm.

        :raises Warning: If the dimension is greater than the other dimensions.

        :param torch.Tensor X: The input tensor to orthogonalize. The input must
            be of dimensions :math:`[N, M]`.
        :return: The orthonormal tensor.
        """
        # check dim is less than all the other dimensions
        if X.shape[self.dim] > min(X.shape):
            raise Warning(
                "The dimension where to orthogonalize is greater"
                " than the other dimensions"
            )

        result = torch.zeros_like(X, requires_grad=self._requires_grad)
        X_0 = torch.select(X, self.dim, 0).clone()
        result_0 = X_0 / torch.linalg.norm(X_0)
        result = self._differentiable_copy(result, 0, result_0)

        # iterate over the rest of the basis with Gram-Schmidt
        for i in range(1, X.shape[self.dim]):
            v = torch.select(X, self.dim, i).clone()
            for j in range(i):
                vj = torch.select(result, self.dim, j).clone()
                v = v - torch.sum(v * vj, dim=self.dim, keepdim=True) * vj
            # result_i = torch.select(result, self.dim, i)
            result_i = v / torch.linalg.norm(v)
            result = self._differentiable_copy(result, i, result_i)
        return result

    def _differentiable_copy(self, result, idx, value):
        """
        Perform a differentiable copy operation on a tensor.

        :param torch.Tensor result: The tensor where values will be copied to.
        :param int idx: The index along the specified dimension where the
            value will be copied.
        :param torch.Tensor value: The tensor value to copy into the
            result tensor.
        :return: A new tensor with the copied values.
        :rtype: torch.Tensor
        """
        return result.index_copy(
            self.dim, torch.tensor([idx]), value.unsqueeze(self.dim)
        )

    @property
    def dim(self):
        """
        Get the dimension along which operations are performed.

        :return: The current dimension value.
        :rtype: int
        """
        return self._dim

    @dim.setter
    def dim(self, value):
        """
        Set the dimension along which operations are performed.

        :param value: The dimension to be set, which must be 0, 1, or -1.
        :type value: int
        :raises IndexError: If the provided dimension is not in the
            range [-1, 1].
        """
        # check consistency
        check_consistency(value, int)
        if value not in [0, 1, -1]:
            raise IndexError(
                "Dimension out of range (expected to be in "
                f"range of [-1, 1], but got {value})"
            )
        # assign value
        self._dim = value

    @property
    def requires_grad(self):
        """
        Indicates whether gradient computation is required for operations
        on the tensors.

        :return: True if gradients are required, False otherwise.
        :rtype: bool
        """
        return self._requires_grad
