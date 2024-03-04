"""Module Averaging Neural Operator."""

from torch import nn, concatenate
from . import FeedForward
from .layers import AVNOLayer




class AVNO(nn.Module):
    """
    The PINA implementation of the inner layer of the Averaging Neural Operator.

    :param int input_features: The number of input components of the model.
    :param int output_features: The number of output components of the model.
    :param ndarray points: the points in the training set.
    :param int inner_size: number of neurons in the hidden layer(s). Default is 100.
    :param int n_layers: number of hidden layers. Default is 4.
    :param func: the activation function to use. Default to nn.GELU.

    """

    def __init__(
        self,
        input_features,
        output_features,
        points,
        inner_size=100,
        n_layers=4,
        func=nn.GELU,
    ):

        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.num_points = points.shape[0]
        self.points_size = points.shape[1]
        self.lifting = FeedForward(input_features + self.points_size,
                                   inner_size, inner_size, n_layers, func)
        self.nn = nn.Sequential(
            *[AVNOLayer(inner_size, func) for _ in range(n_layers)])
        self.projection = FeedForward(inner_size + self.points_size,
                                      output_features, inner_size, n_layers,
                                      func)
        self.points = points

    def forward(self, batch):
        """
        Computes the forward pass of the model with the points specified in init.

        :param torch.Tensor batch: the input tensor.

        """
        points_tmp = self.points.unsqueeze(0).repeat(batch.shape[0], 1, 1)
        new_batch = concatenate((batch, points_tmp), dim=2)
        new_batch = self.lifting(new_batch)
        new_batch = self.nn(new_batch)
        new_batch = concatenate((new_batch, points_tmp), dim=2)
        new_batch = self.projection(new_batch)
        return new_batch

    def forward_eval(self, batch, points):
        """
        Computes the forward pass of the model with the points specified when calling the function.

        :param torch.Tensor batch: the input tensor.
        :param torch.Tensor points: the points tensor.
        
        """
        points_tmp = points.unsqueeze(0).repeat(batch.shape[0], 1, 1)
        new_batch = concatenate((batch, points_tmp), dim=2)
        new_batch = self.lifting(new_batch)
        new_batch = self.nn(new_batch)
        new_batch = concatenate((new_batch, points_tmp), dim=2)
        new_batch = self.projection(new_batch)
        return new_batch
