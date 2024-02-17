"""Module Averaging Neural Operator."""

from torch import nn, concatenate
from . import FeedForward
from .layers import AVNOLayer


class AVNO(nn.Module):
    """
    The PINA implementation of the inner layer 
        of the Averaging Neural Operator.

    :param int input_features: The number of input components of the model.
    :param int output_features: The number of output components of the model.
    :param int points_size: the dimension of the domain of the functions.
    :param int inner_size: number of neurons in the hidden layer(s). 
        Default is 100.
    :param int n_layers: number of hidden layers. Default is 4.
    :param func: the activation function to use. Default to nn.GELU.
    :param str features_label: the label of the features in the input tensor. 
        Default to 'v'.
    :param str points_label: the label of the points in the input tensor. 
        Default to 'p'.
    """

    def __init__(
        self,
        input_features,
        output_features,
        features_label='v',
        points_label='p',
        points_size=3,
        inner_size=100,
        n_layers=4,
        func=nn.GELU,
    ):

        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.points_size = points_size
        self.points_label = points_label
        self.features_label = features_label
        self._lifting = FeedForward(input_features + self.points_size,
                                   inner_size, inner_size, n_layers, func)
        self._nn = nn.Sequential(
            *[AVNOLayer(inner_size, func) for _ in range(n_layers)])
        self._projection = FeedForward(inner_size + self.points_size,
                                      output_features, inner_size, n_layers,
                                      func)

    def forward(self, batch):
        """
        Computes the forward pass of the model with the points 
            specified when calling the function.

        :param torch.Tensor batch: the input tensor.
        :param torch.Tensor points: the points tensor.
        
        """
        points_tmp = concatenate([
            batch.extract(f"{self.points_label}_{i}")
            for i in range(self.points_size)
        ],
                                 axis=2)
        features_tmp = concatenate([
            batch.extract(f"{self.features_label}_{i}")
            for i in range(self.input_features)
        ],
                                   axis=2)
        new_batch = concatenate((features_tmp, points_tmp), dim=2)
        new_batch = self._lifting(new_batch)
        new_batch = self._nn(new_batch)
        new_batch = concatenate((new_batch, points_tmp), dim=2)
        new_batch = self._projection(new_batch)
        return new_batch

    @property
    def lifting(self):
        "Lifting operator of the AVNO"
        return self._lifting

    @property
    def nn(self):
        "Integral operator of the AVNO"
        return self._nn

    @property
    def projection(self):
        "Projection operator of the AVNO"
        return self._projection
