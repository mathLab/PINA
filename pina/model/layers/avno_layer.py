"""Module for Averaging Neural Operator Layer class."""

from torch import nn, mean


class AVNOBlock(nn.Module):
    """
    The PINA implementation of the inner layer 
        of the Averaging Neural Operator . 

    :param int hidden_size: size of the layer.
        Defaults to 100.
    :param func: the activation function to use. 
        Default to nn.GELU.
        
    """

    def __init__(self, hidden_size=100, func=nn.GELU):
        super().__init__()
        self._nn = nn.Linear(hidden_size, hidden_size)
        self._func = func()

    def forward(self, batch):
        """Forward pass of the layer."""
        return self._func(self._nn(batch) + mean(batch, dim=1, keepdim=True))
