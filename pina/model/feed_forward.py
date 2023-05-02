"""Module for FeedForward model"""
import torch
import torch.nn as nn

from pina.label_tensor import LabelTensor


class FeedForward(torch.nn.Module):
    """
    The PINA implementation of feedforward network, also refered as multilayer
    perceptron.

    :param int input_variables: The number of input components of the model.
        Expected tensor shape of the form (*, input_variables), where *
        means any number of dimensions including none.
    :param int output_variables: The number of output components of the model.
        Expected tensor shape of the form (*, output_variables), where *
        means any number of dimensions including none.
    :param int inner_size: number of neurons in the hidden layer(s). Default is
        20.
    :param int n_layers: number of hidden layers. Default is 2.
    :param func: the activation function to use. If a single
        :class:`torch.nn.Module` is passed, this is used as activation function
        after any layers, except the last one. If a list of Modules is passed,
        they are used as activation functions at any layers, in order.
    :param iterable(int) layers: a list containing the number of neurons for
        any hidden layers. If specified, the parameters `n_layers` e
        `inner_size` are not considered.
    :param bool bias: If `True` the MLP will consider some bias.
    """
    def __init__(self, input_variables, output_variables, inner_size=20,
                 n_layers=2, func=nn.Tanh, layers=None, bias=True):
        """
        """
        super().__init__()


        if not isinstance(input_variables, int):
            raise ValueError('input_variables expected to be int.')
        self.input_dimension = input_variables

        if not isinstance(output_variables, int):
            raise ValueError('output_variables expected to be int.')
        self.output_dimension = output_variables
        if layers is None:
            layers = [inner_size] * n_layers

        tmp_layers = layers.copy()
        tmp_layers.insert(0, self.input_dimension)
        tmp_layers.append(self.output_dimension)

        self.layers = []
        for i in range(len(tmp_layers) - 1):
            self.layers.append(
                nn.Linear(tmp_layers[i], tmp_layers[i + 1], bias=bias)
            )

        if isinstance(func, list):
            self.functions = func
        else:
            self.functions = [func for _ in range(len(self.layers) - 1)]

        if len(self.layers) != len(self.functions) + 1:
            raise RuntimeError('uncosistent number of layers and functions')

        unique_list = []
        for layer, func in zip(self.layers[:-1], self.functions):
            unique_list.append(layer)
            if func is not None:
                unique_list.append(func())
        unique_list.append(self.layers[-1])

        self.model = nn.Sequential(*unique_list)

    def forward(self, x):
        """
        Defines the computation performed at every call.

        :param x: .
        :type x: :class:`pina.LabelTensor`
        :return: the output computed by the model.
        :rtype: LabelTensor
        """
        return self.model(x)
