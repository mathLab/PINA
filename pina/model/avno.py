"""Module Averaging Neural Operator."""

from torch import nn, concatenate
from . import FeedForward
from .layers import AVNOBlock
from .base_no import KernelNeuralOperator


class AveragingNeuralOperator(KernelNeuralOperator):
    """
    Implementation of Averaging Neural Operator.

    This class implements the Averaging Neural Operator.

    .. seealso::

        **Original reference**: Lanthaler S. Li, Z., Kovachki,
        Stuart, A.
        (2020). *The Nonlocal Neural Operator:
            Universal Approximation*.
        DOI: `arXiv preprint arXiv:2304.13221.
        <https://arxiv.org/abs/2304.13221>`_
    """

    def __init__(
        self,
        input_numb_fields,
        output_numb_fields,
        field_indices,
        coordinates_indices,
        dimension=3,
        inner_size=100,
        n_layers=4,
        func=nn.GELU,
    ):
        """
        :param int input_numb_fields: The number of input components 
            of the model.
        :param int output_numb_fields: The number of output components 
            of the model.
        :param int dimension: the dimension of the domain of the functions.
        :param int inner_size: number of neurons in the hidden layer(s). 
            Defaults to 100.
        :param int n_layers: number of hidden layers. Default is 4.
        :param func: the activation function to use. Default to nn.GELU.
        :param str field_indices: the label of the fields 
            in the input tensor. 
        :param str coordinates_indices: the label of the 
            coordinates in the input tensor. 
        """

        self.input_numb_fields = input_numb_fields
        self.output_numb_fields = output_numb_fields
        self.dimension = dimension
        self.coordinates_indices = coordinates_indices
        self.field_indices = field_indices
        integral_net = nn.Sequential(
            *[AVNOBlock(inner_size, func) for _ in range(n_layers)])
        lifting_net = FeedForward(dimension + input_numb_fields, inner_size,
                                  inner_size, n_layers, func)
        projection_net = FeedForward(inner_size + dimension, output_numb_fields,
                                     inner_size, n_layers, func)
        super().__init__(lifting_net, integral_net, projection_net)

    def forward(self, x):
        points_tmp = x.extract(self.coordinates_indices)
        features_tmp = x.extract(self.field_indices)
        new_batch = concatenate((features_tmp, points_tmp), dim=2)
        new_batch = self._lifting_operator(new_batch)
        new_batch = self._integral_kernels(new_batch)
        new_batch = concatenate((new_batch, points_tmp), dim=2)
        new_batch = self._projection_operator(new_batch)
        return new_batch
