"""Module Averaging Neural Operator."""

from torch import nn, concatenate
from . import FeedForward
from .layers import AVNOBlock
from .base_no import KernelNeuralOperator


class AVNO(KernelNeuralOperator):
    """
    The PINA implementation of the inner layer 
        of the Averaging Neural Operator.

    :param int input_features: The number of input components of the model.
    :param int output_features: The number of output components of the model.
    :param int points_size: the dimension of the domain of the functions.
    :param int inner_size: number of neurons in the hidden layer(s). 
        Defaults to 100.
    :param int n_layers: number of hidden layers. Default is 4.
    :param func: the activation function to use. Default to nn.GELU.
    :param str features_label: the label of the features in the input tensor. 
        Defaults to 'v'.
    :param str points_label: the label of the points in the input tensor. 
        Defaults to 'p'.
    """

    def __init__(
        self,
        input_features,
        output_features,
        field_indices,
        coordinates_indices,
        points_size=3,
        inner_size=100,
        n_layers=4,
        func=nn.GELU,
    ):

        self.input_features = input_features
        self.output_features = output_features
        self.points_size = points_size
        self.coordinates_indices = coordinates_indices
        self.field_indices = field_indices
        integral_net = nn.Sequential(
            *[AVNOBlock(inner_size, func) for _ in range(n_layers)])
        lifting_net = FeedForward(input_features + points_size, inner_size,
                                  inner_size, n_layers, func)
        projection_net = FeedForward(inner_size + points_size, output_features,
                                     inner_size, n_layers, func)
        super().__init__(lifting_net, integral_net, projection_net)

    def forward(self, batch):
        points_tmp = concatenate([
            batch.extract(f"{self.coordinates_indices}_{i}")
            for i in range(self.points_size)
        ],
                                 axis=2)
        features_tmp = concatenate([
            batch.extract(f"{self.field_indices}_{i}")
            for i in range(self.input_features)
        ],
                                   axis=2)
        new_batch = concatenate((features_tmp, points_tmp), dim=2)
        new_batch = self._lifting_operator(new_batch)
        new_batch = self._integral_kernels(new_batch)
        new_batch = concatenate((new_batch, points_tmp), dim=2)
        new_batch = self._projection_operator(new_batch)
        return new_batch
