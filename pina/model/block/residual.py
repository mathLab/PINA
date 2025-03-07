"""
TODO: Add title.
"""

import torch
from torch import nn
from ...utils import check_consistency


class ResidualBlock(nn.Module):
    """Residual block base class. Implementation of a residual block.

    .. seealso::

        **Original reference**: He, Kaiming, et al.
        *Deep residual learning for image recognition.*
        Proceedings of the IEEE conference on computer vision
        and pattern recognition. 2016..
        DOI: `<https://arxiv.org/pdf/1512.03385.pdf>`_.

    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        spectral_norm=False,
        activation=torch.nn.ReLU(),
    ):
        """
        Initializes the ResidualBlock module.

        :param int input_dim: Dimension of the input to pass to the
            feedforward linear layer.
        :param int output_dim: Dimension of the output from the
            residual layer.
        :param int hidden_dim: Hidden dimension for mapping the input
            (first block).
        :param bool spectral_norm: Apply spectral normalization to feedforward
            layers, defaults to False.
        :param torch.nn.Module activation: Cctivation function after first
            block.

        """
        super().__init__()
        # check consistency
        check_consistency(spectral_norm, bool)
        check_consistency(input_dim, int)
        check_consistency(output_dim, int)
        check_consistency(hidden_dim, int)
        check_consistency(activation, torch.nn.Module)

        # assign variables
        self._spectral_norm = spectral_norm
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._hidden_dim = hidden_dim
        self._activation = activation

        # create layers
        self._l1 = self._spect_norm(nn.Linear(input_dim, hidden_dim))
        self._l2 = self._spect_norm(nn.Linear(hidden_dim, output_dim))
        self._l3 = self._spect_norm(nn.Linear(input_dim, output_dim))

    def forward(self, x):
        """Forward pass for residual block layer.

        :param torch.Tensor x: Input tensor for the residual layer.
        :return: Output tensor for the residual layer.
        :rtype: torch.Tensor
        """
        y = self._activation(self._l1(x))
        y = self._l2(y)
        x = self._l3(x)
        return y + x

    def _spect_norm(self, x):
        """Perform spectral norm on the layers.

        :param x: A torch.nn.Module Linear layer
        :type x: torch.nn.Module
        :return: The spectral norm of the layer
        :rtype: torch.nn.Module
        """
        return nn.utils.spectral_norm(x) if self._spectral_norm else x


class EnhancedLinear(torch.nn.Module):
    """
    A wrapper class for enhancing a linear layer with activation and/or dropout.

    :param layer: The linear layer to be enhanced.
    :type layer: torch.nn.Module
    :param activation: The activation function to be applied after the linear
        layer.
    :type activation: torch.nn.Module
    :param dropout: The dropout probability to be applied after the activation
        (if provided).
    :type dropout: float

    :Example:

    >>> linear_layer = torch.nn.Linear(10, 20)
    >>> activation = torch.nn.ReLU()
    >>> dropout_prob = 0.5
    >>> enhanced_linear = EnhancedLinear(linear_layer, activation, dropout_prob)
    """

    def __init__(self, layer, activation=None, dropout=None):
        """
        Initializes the EnhancedLinear module.

        :param layer: The linear layer to be enhanced.
        :type layer: torch.nn.Module
        :param activation: The activation function to be applied after the
            linear layer.
        :type activation: torch.nn.Module
        :param dropout: The dropout probability to be applied after the
            activation (if provided).
        :type dropout: float
        """
        super().__init__()

        # check consistency
        check_consistency(layer, nn.Module)
        if activation is not None:
            check_consistency(activation, nn.Module)
        if dropout is not None:
            check_consistency(dropout, float)

        # assign forward
        if (dropout is None) and (activation is None):
            self._model = torch.nn.Sequential(layer)

        elif (dropout is None) and (activation is not None):
            self._model = torch.nn.Sequential(layer, activation)

        elif (dropout is not None) and (activation is None):
            self._model = torch.nn.Sequential(layer, self._drop(dropout))

        elif (dropout is not None) and (activation is not None):
            self._model = torch.nn.Sequential(
                layer, activation, self._drop(dropout)
            )

    def forward(self, x):
        """
        Forward pass through the enhanced linear module.

        :param x: Input tensor.
        :type x: torch.Tensor

        :return: Output tensor after passing through the enhanced linear module.
        :rtype: torch.Tensor
        """
        return self._model(x)

    def _drop(self, p):
        """
        Applies dropout with probability p.

        :param p: Dropout probability.
        :type p: float

        :return: Dropout layer with the specified probability.
        :rtype: torch.nn.Dropout
        """
        return torch.nn.Dropout(p)
