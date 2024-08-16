"""Module for OrthogonalBlock layer, to make the input orthonormal."""

import torch


class OrthogonalBlock(torch.nn.Module):
    """
    Module to make the input orthonormal.
    The module takes a tensor of size [N, M] and returns a tensor of
    size [N, M] where the columns are orthonormal.
    """
    def __init__(self, dim=-1):
        """
        Initialize the OrthogonalBlock module.

        :param int dim: The dimension where to orthogonalize.
        """
        super().__init__()
        self.dim = dim

    def forward(self, X):
        """
        Forward pass of the OrthogonalBlock module using a Gram-Schmidt
        algorithm.

        :raises Warning: If the dimension is greater than the other dimensions.

        :param torch.Tensor X: The input tensor to orthogonalize.
        :return: The orthonormal tensor.
        """
        # check dim is less than all the other dimensions
        if X.shape[self.dim] > min(X.shape):
            raise Warning("The dimension where to orthogonalize is greater\
                          than the other dimensions")

        result = torch.zeros_like(X)
        # normalize first basis
        X_0 = torch.select(X, self.dim, 0)
        result_0 = torch.select(result, self.dim, 0)
        result_0 += X_0/torch.norm(X_0)
        # iterate over the rest of the basis with Gram-Schmidt
        for i in range(1, X.shape[self.dim]):
            v = torch.select(X, self.dim, i)
            for j in range(i):
                v -= torch.sum(v * torch.select(result, self.dim, j),
                               dim=self.dim, keepdim=True) * torch.select(
                                   result, self.dim, j)
            result_i = torch.select(result, self.dim, i)
            result_i += v/torch.norm(v)
        return result