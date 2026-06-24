"""Module for the Adaptive Function Interface."""

from abc import ABCMeta, abstractmethod


class AdaptiveFunctionInterface(metaclass=ABCMeta):
    """
    Abstract interface for all adaptive functions.

    :Example:

        >>> import torch
        >>> from pina.adaptive_function import AdaptiveTanh
        >>> act = AdaptiveTanh()
        >>> x = torch.randn(10, 3)
        >>> out = act(x)
        >>> out.shape
        torch.Size([10, 3])
    """

    @abstractmethod
    def forward(self, x):
        """
        Compute the transformation of the adaptive function on the input.

        :param x: The input tensor to evaluate the adaptive function.
        :type x: torch.Tensor | LabelTensor
        :return: The output of the adaptive function.
        :rtype: torch.Tensor | LabelTensor
        """

    @property
    @abstractmethod
    def alpha(self):
        """
        The output scaling parameter of the adaptive function.

        :return: The alpha parameter.
        :rtype: torch.nn.Parameter | torch.Tensor
        """

    @property
    @abstractmethod
    def beta(self):
        """
        The input scaling parameter of the adaptive function.

        :return: The beta parameter.
        :rtype: torch.nn.Parameter | torch.Tensor
        """

    @property
    @abstractmethod
    def gamma(self):
        """
        The input shifting parameter of the adaptive function.

        :return: The gamma parameter.
        :rtype: torch.nn.Parameter | torch.Tensor
        """
