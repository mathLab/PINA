"""Module for the Adaptive Function Interface."""

from abc import ABCMeta, abstractmethod


class AdaptiveFunctionInterface(metaclass=ABCMeta):
    """
    Abstract interface for all adaptive functions.
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

    @abstractmethod
    @property
    def alpha(self):
        """
        The output scaling parameter of the adaptive function.

        :return: The alpha parameter.
        :rtype: torch.nn.Parameter | torch.Tensor
        """

    @abstractmethod
    @property
    def beta(self):
        """
        The input scaling parameter of the adaptive function.

        :return: The beta parameter.
        :rtype: torch.nn.Parameter | torch.Tensor
        """

    @abstractmethod
    @property
    def gamma(self):
        """
        The input shifting parameter of the adaptive function.

        :return: The gamma parameter.
        :rtype: torch.nn.Parameter | torch.Tensor
        """

    @abstractmethod
    @property
    def func(self):
        """
        The adaptive function.

        :return: The adaptive function.
        :rtype: callable
        """

    @abstractmethod
    @func.setter
    def func(self, value):
        """
        Set the adaptive function.

        :param value: The adaptive function.
        :type value: callable
        """
