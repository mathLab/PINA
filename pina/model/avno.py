"""Module Averaging Neural Operator."""

from torch import nn, concatenate
from . import FeedForward
from .layers import AVNOBlock
from .base_no import KernelNeuralOperator

class Lifting_Net(nn.Module):
    """
    The PINA implementation of the lifting layer of the AVNO
    
    :param int input_features: The number of input components of the model.
    :param int points_size: the dimension of the domain of the functions.
    :param int inner_size: number of neurons in the hidden layer(s).
    :param int n_layers: number of hidden layers. Default is 4.
    :param func: the activation function to use. Default to nn.GELU.
    :param str features_label: the label of the features in the input tensor.
    :param str points_label: the label of the points in the input tensor.

    """
    
    def __init__(self, input_features, points_size, inner_size,n_layers, func, points_label, features_label):
        super(Lifting_Net, self).__init__()
        self._lifting = FeedForward(input_features + points_size,
                                    inner_size, inner_size, n_layers, func)
        self.points_size = points_size
        self.inner_size = inner_size
        self.input_features = input_features
        self.points_label = points_label
        self.features_label = features_label

    def forward(self, batch):
        """Forward pass of the layer."""
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
        return [new_batch, points_tmp]

    def net(self):
        """Returns the net"""
        return self._lifting

    
class Integral_Net(nn.Module):
    def __init__(self, inner_size, n_layers, func):
        super(Integral_Net, self).__init__()
        self._nn = nn.Sequential(
            *[AVNOBlock(inner_size, func) for _ in range(n_layers)])

    def forward(self, batch):
        """Forward pass of the layer."""

        new_batch, points_tmp = batch
        new_batch = self._nn(new_batch)
        return [new_batch, points_tmp]

    def net(self):
        """Returns the net"""
        return self._nn
    
class Projection_Net(nn.Module):
    """
    The PINA implementation of the projection 
        layer of the AVNO.

    :param int inner_size: number of neurons in the hidden layer(s).
    :param int points_size: the dimension of the domain of the functions.
    :param int output_features: The number of output components of the model.
    :param int n_layers: number of hidden layers. Default is 4.
    :param func: the activation function to use. Default to nn.GELU.

    """
    
    def __init__(self, inner_size, points_size, output_features, n_layers, func):
        super(Projection_Net, self).__init__()
        self._projection = FeedForward(inner_size + points_size,
                                        output_features, inner_size, n_layers,
                                        func)

    def forward(self, batch):
        """Forward pass of the layer."""
        new_batch, points_tmp = batch
        new_batch = concatenate((new_batch, points_tmp), dim=2)
        new_batch = self._projection(new_batch)
        return new_batch

    def net(self):
        """Returns the net"""
        return self._projection


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
        features_label='v',
        points_label='p',
        points_size=3,
        inner_size=100,
        n_layers=4,
        func=nn.GELU,
    ):

        self.input_features = input_features
        self.output_features = output_features
        self.points_size = points_size
        self.points_label = points_label
        self.features_label = features_label

        nn_net=Integral_Net(inner_size=inner_size, n_layers=n_layers, func=func)
        lifting_net=Lifting_Net(input_features=input_features, 
                                points_size=points_size, 
                                inner_size=inner_size, 
                                n_layers=n_layers, func=func, 
                                points_label=points_label, 
                                features_label=features_label)
        projection_net=Projection_Net(inner_size=inner_size, 
                                      points_size=points_size, 
                                      output_features=output_features, 
                                      n_layers=n_layers, func=func)
        super().__init__(lifting_net, nn_net, projection_net)
