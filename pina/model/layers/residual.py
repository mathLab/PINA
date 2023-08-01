import torch
import torch.nn as nn
from ...utils import check_consistency


class ResidualBlock(nn.Module):
    """Residual block base class. Implementation of a residual block.

    .. seealso::

        **Original reference**: He, Kaiming, et al.
        "Deep residual learning for image recognition."
        Proceedings of the IEEE conference on computer vision
        and pattern recognition. 2016..
        <https://arxiv.org/pdf/1512.03385.pdf>`_.

    """

    def __init__(self, input_dim, output_dim,
                 hidden_dim, spectral_norm=False, 
                 activation=torch.nn.ReLU()):
        """Residual block constructor

        :param int input_dim: Dimension of the input to pass to the
            feedforward linear layer.
        :param int output_dim: Dimension of the output from the 
            residual layer.
        :param int hidden_dim: Hidden dimension for mapping the input
            (first block).
        :param bool spectral_norm: Apply spectral normalization to feedforward
            layers, defaults to False.
        :param torch.nn.Module activation: Cctivation function after first block.

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
        self.l1 = self._spect_norm(nn.Linear(input_dim, hidden_dim))
        self.l2 = self._spect_norm(nn.Linear(hidden_dim, output_dim))
        self.l3 = self._spect_norm(nn.Linear(input_dim, output_dim))

    def forward(self, x):
        """Forward pass for residual block layer.

        :param torch.Tensor x: Input tensor for the residual layer.
        :return: Output tensor for the residual layer.
        :rtype: torch.Tensor
        """
        y = self.activation(self.l1(x))
        y = self.l2(y)
        x = self.l3(x)
        return y + x

    def _spect_norm(self, x):
        """Perform spectral norm on the layers.

        :param x: A torch.nn.Module Linear layer
        :type x: torch.nn.Module
        :return: The spectral norm of the layer
        :rtype: torch.nn.Module
        """
        return nn.utils.spectral_norm(x) if self._spectral_norm else x

    @ property
    def spectral_norm(self):
        return self._spectral_norm

    @ property
    def input_dim(self):
        return self._input_dim

    @ property
    def output_dim(self):
        return self._output_dim

    @ property
    def hidden_dim(self):
        return self._hidden_dim
    
    @ property
    def activation(self):
        return self._activation