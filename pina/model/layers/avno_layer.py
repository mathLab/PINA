"""Module for Averaging Neural Operator class."""
from torch import nn, mean


class AVNOLayer(nn.Module):
    """
    The PINA implementation of the inner layer of the Averaging Neural Operator . 

    :param int hidden_size: size of the layer.
    :param func: the activation function to use. 
    
    """

    def __init__(self, hidden_size, func):
        super().__init__()
        self.nn = nn.Linear(hidden_size, hidden_size)
        self.func = func

    def forward(self, batch):
        return self.func()(self.nn(batch) + mean(batch, dim=1).unsqueeze(1))
