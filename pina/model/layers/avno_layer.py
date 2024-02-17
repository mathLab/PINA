"""Module for Averaging Neural Operator Layer class."""
from torch import nn, mean
import dataclasses
from collections.abc import Callable

@dataclasses.dataclass
class AVNOLayer(nn.Module):
    """
    The PINA implementation of the inner layer 
        of the Averaging Neural Operator . 

    :param int hidden_size: size of the layer.
    :param func: the activation function to use. 
    """

    hidden_size: int
    func: Callable

    def __post_init__(self):
        super().__init__()
        self.nn = nn.Linear(self.hidden_size, self.hidden_size)


    def forward(self, batch):
        """Forward pass of the layer."""
        return self.func()(self.nn(batch) + mean(batch, dim=1).unsqueeze(1))
