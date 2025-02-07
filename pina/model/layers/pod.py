"""Module for Base Continuous Convolution class."""

from abc import ABCMeta, abstractmethod
import torch
from .stride import Stride
from .utils_convolution import optimizing


class PODBlock(torch.nn.Module):
    """
    POD layer: it projects the input field on the proper orthogonal
    decomposition basis.  It needs to be fitted to the data before being used
    with the method :meth:`fit`, which invokes the singular value decomposition.
    The layer is not trainable.

    .. note::
        All the POD modes are stored in memory, avoiding to recompute them when the rank changes but increasing the memory usage.
    """

    def __init__(self, rank, scale_coefficients=True):
        """
        Build the POD layer with the given rank.

        :param int rank: The rank of the POD layer.
        :param bool scale_coefficients: If True, the coefficients are scaled
            after the projection to have zero mean and unit variance.
        """
        super().__init__()
        self.__scale_coefficients = scale_coefficients
        self._basis = None
        self._scaler = None
        self._rank = rank

    @property
    def rank(self):
        """
        The rank of the POD layer.

        :rtype: int
        """
        return self._rank

    @rank.setter
    def rank(self, value):
        if value < 1 or not isinstance(value, int):
            raise ValueError("The rank must be positive integer")

        self._rank = value

    @property
    def basis(self):
        """
        The POD basis. It is a matrix whose columns are the first `self.rank` POD modes.

        :rtype: torch.Tensor
        """
        if self._basis is None:
            return None

        return self._basis[: self.rank]

    @property
    def scaler(self):
        """
        The scaler. It is a dictionary with the keys `'mean'` and `'std'` that
        store the mean and the standard deviation of the coefficients.

        :rtype: dict
        """
        if self._scaler is None:
            return

        return {
            "mean": self._scaler["mean"][: self.rank],
            "std": self._scaler["std"][: self.rank],
        }

    @property
    def scale_coefficients(self):
        """
        If True, the coefficients are scaled after the projection to have zero
        mean and unit variance.

        :rtype: bool
        """
        return self.__scale_coefficients

    def fit(self, X):
        """
        Set the POD basis by performing the singular value decomposition of the
        given tensor. If `self.scale_coefficients` is True, the coefficients
        are scaled after the projection to have zero mean and unit variance.

        :param torch.Tensor X: The tensor to be reduced.
        """
        self._fit_pod(X)

        if self.__scale_coefficients:
            self._fit_scaler(torch.matmul(self._basis, X.T))

    def _fit_scaler(self, coeffs):
        """
        Private merhod that computes the mean and the standard deviation of the
        given coefficients, allowing to scale them to have zero mean and unit
        variance. Mean and standard deviation are stored in the private member
        `_scaler`.

        :param torch.Tensor coeffs: The coefficients to be scaled.
        """
        self._scaler = {
            "std": torch.std(coeffs, dim=1),
            "mean": torch.mean(coeffs, dim=1),
        }

    def _fit_pod(self, X):
        """
        Private method that computes the POD basis of the given tensor and stores it in the private member `_basis`.

        :param torch.Tensor X: The tensor to be reduced.
        """
        self._basis = torch.svd(X.T)[0].T

    def forward(self, X):
        """
        The forward pass of the POD layer. By default it executes the
        :meth:`reduce` method, reducing the input tensor to its POD
        representation. The POD layer needs to be fitted before being used.

        :param torch.Tensor X: The input tensor to be reduced.
        :return: The reduced tensor.
        :rtype: torch.Tensor
        """
        return self.reduce(X)

    def reduce(self, X):
        """
        Reduce the input tensor to its POD representation. The POD layer needs
        to be fitted before being used.

        :param torch.Tensor X: The input tensor to be reduced.
        :return: The reduced tensor.
        :rtype: torch.Tensor
        """
        if self._basis is None:
            raise RuntimeError(
                "The POD layer needs to be fitted before being used."
            )

        coeff = torch.matmul(self.basis, X.T)
        if coeff.ndim == 1:
            coeff = coeff.unsqueeze(1)

        coeff = coeff.T
        if self.__scale_coefficients:
            coeff = (coeff - self.scaler["mean"]) / self.scaler["std"]

        return coeff

    def expand(self, coeff):
        """
        Expand the given coefficients to the original space. The POD layer needs
        to be fitted before being used.

        :param torch.Tensor coeff: The coefficients to be expanded.
        :return: The expanded tensor.
        :rtype: torch.Tensor
        """
        if self._basis is None:
            raise RuntimeError(
                "The POD layer needs to be trained before being used."
            )

        if self.__scale_coefficients:
            coeff = coeff * self.scaler["std"] + self.scaler["mean"]
        predicted = torch.matmul(self.basis.T, coeff.T).T

        if predicted.ndim == 1:
            predicted = predicted.unsqueeze(0)

        return predicted
