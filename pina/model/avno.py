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

        class Lifting_Net(nn.Module):
            def __init__(self):
                super(Lifting_Net, self).__init__()
                self._lifting = FeedForward(input_features + points_size,
                                           inner_size, inner_size, n_layers, func)
                
            def forward(self, batch):
                points_tmp = concatenate([
                    batch.extract(f"{points_label}_{i}")
                    for i in range(points_size)
                ],
                                        axis=2)
                features_tmp = concatenate([
                    batch.extract(f"{features_label}_{i}")
                    for i in range(input_features)
                ],
                                        axis=2)
                new_batch = concatenate((features_tmp, points_tmp), dim=2)
                new_batch = self._lifting(new_batch)
                return [new_batch, points_tmp]
            
        class NN_Net(nn.Module):
            def __init__(self):
                super(NN_Net, self).__init__()
                self._nn = nn.Sequential(
                    *[AVNOBlock(inner_size, func) for _ in range(n_layers)])
                
            def forward(self, batch):
                new_batch, points_tmp = batch
                new_batch = self._nn(new_batch)
                return [new_batch, points_tmp]
            
        class Projection_Net(nn.Module):
            def __init__(self):
                super(Projection_Net, self).__init__()
                self._projection = FeedForward(inner_size + points_size,
                                              output_features, inner_size, n_layers,
                                              func)
                
            def forward(self, batch):
                new_batch, points_tmp = batch
                new_batch = concatenate((new_batch, points_tmp), dim=2)
                new_batch = self._projection(new_batch)
                return new_batch

        nn_net=NN_Net()
        lifting_net=Lifting_Net()
        projection_net=Projection_Net()
        super().__init__(lifting_net, nn_net, projection_net)
