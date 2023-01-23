import torch
import torch.nn as nn


class NeuralNet(torch.nn.Module):

    def __init__(self, input_channel, output_channel, inner_size=20,
                 n_layers=2, func=nn.Tanh, layers=None):
        """Deep neural network model

        :param input_channel: input channel for the network
        :type input_channel: int
        :param output_channel: output channel for the network
        :type output_channel: int
        :param inner_size: inner size of each hidden layer, defaults to 20
        :type inner_size: int, optional
        :param n_layers: number of layers in the network, defaults to 2
        :type n_layers: int, optional
        :param func: function(s) to pass to the network, defaults to nn.Tanh
        :type func: (list of) torch.nn function(s), optional
        :param layers: list of layers for the network, defaults to None
        :type layers: list[int], optional
        """
        super().__init__()

        self._input_channel = input_channel
        self._output_channel = output_channel
        self._inner_size = inner_size
        self._n_layers = n_layers
        self._layers = layers

        if layers is None:
            layers = [inner_size] * n_layers

        tmp_layers = layers.copy()
        tmp_layers.insert(0, self._input_channel)
        tmp_layers.append(self._output_channel)

        self._layers = []
        for i in range(len(tmp_layers) - 1):
            self._layers.append(nn.Linear(tmp_layers[i], tmp_layers[i + 1]))

        if isinstance(func, list):
            self._functions = func
        else:
            self._functions = [func for _ in range(len(self._layers) - 1)]

        unique_list = []
        for layer, func in zip(self._layers[:-1], self._functions):
            unique_list.append(layer)
            if func is not None:
                unique_list.append(func())
        unique_list.append(self._layers[-1])

        self.model = nn.Sequential(*unique_list)

    def forward(self, x):
        """Forward method for NeuralNet class

        :param x: network input data
        :type x: torch.tensor
        :return: network output
        :rtype: torch.tensor
        """
        return self.model(x)

    @property
    def input_channel(self):
        return self._input_channel

    @property
    def output_channel(self):
        return self._output_channel

    @property
    def inner_size(self):
        return self._inner_size

    @property
    def n_layers(self):
        return self._n_layers

    @property
    def functions(self):
        return self._functions

    @property
    def layers(self):
        return self._layers


def check_point(x, current_stride, dim):
    max_stride = current_stride + dim
    indeces = torch.logical_and(x[..., :-1] < max_stride,
                                x[..., :-1] >= current_stride).all(dim=-1)
    return indeces


def map_points_(x, filter_position):
    """Mapping function n dimensional case

    :param x: input data of two dimension
    :type x: torch.tensor
    :param filter_position: position of the filter
    :type dim: list[numeric]
    :return: data mapped inplace
    :rtype: torch.tensor
    """
    x.add_(-filter_position)

    return x
