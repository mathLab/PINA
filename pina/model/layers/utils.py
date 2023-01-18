import torch
import torch.nn as nn


def prod(iterable):
    p = 1
    for n in iterable:
        p *= n
    return p


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


class Integral(object):

    def __init__(self, param):

        if param == 'discrete':
            self.make_integral = self.integral_param_disc
        elif param == 'continuous':
            self.make_integral = self.integral_param_cont
        else:
            raise TypeError

    def __call__(self, *args, **kwds):
        return self.make_integral(*args, **kwds)

    def _prepend_zero(self, x):
        return torch.cat((torch.zeros(1, dtype=x.dtype, device=x.device), x))

    def integral_param_disc(self, x, y, idx):
        cs_idxes = self._prepend_zero(torch.cumsum(torch.tensor(idx), 0))
        cs = self._prepend_zero(torch.cumsum(x.flatten()*y.flatten(), 0))
        return cs[cs_idxes[1:]] - cs[cs_idxes[:-1]]

    def integral_param_cont(self, x, y, idx):
        raise NotImplementedError


def create_stride(my_dict):
    """Creating the list for applying the filter

    :param my_dict: Dictionary with the following arguments:
    domain size, starting position of the filter, jump size
    for the filter and direction of the filter
    :type my_dict: dict
    :raises IndexError: Values in the dict must have all same length
    :raises ValueError: Domain values must be greater than 0
    :raises ValueError: Direction must be either equal to 1, -1 or 0
    :raises IndexError: Direction and jumps must have zero in the same index
    :return: list of positions for the filter
    :rtype: list
    :Example:


            >>> stride = {"domain": [4, 4],
                          "start": [-4, 2],
                          "jump": [2, 2],
                          "direction": [1, 1],
                          }
            >>> create_stride(stride)
            [[-4.0, 2.0], [-4.0, 4.0], [-2.0, 2.0], [-2.0, 4.0]]
    """

    # we must check boundaries of the input as well

    domain, start, jumps, direction = my_dict.values()

    # checking

    if not all([len(s) == len(domain) for s in my_dict.values()]):
        raise IndexError("values in the dict must have all same length")

    if not all(v >= 0 for v in domain):
        raise ValueError("domain values must be greater than 0")

    if not all(v == 1 or v == -1 or v == 0 for v in direction):
        raise ValueError("direction must be either equal to 1, -1 or 0")

    seq_jumps = [i for i, e in enumerate(jumps) if e == 0]
    seq_direction = [i for i, e in enumerate(direction) if e == 0]

    if seq_direction != seq_jumps:
        raise IndexError(
            "direction and jumps must have zero in the same index")

    if seq_jumps:
        for i in seq_jumps:
            jumps[i] = domain[i]
            direction[i] = 1

    # creating the stride grid
    values_mesh = [torch.arange(0, i, step).float()
                   for i, step in zip(domain, jumps)]

    values_mesh = [single * dim for single, dim in zip(values_mesh, direction)]

    mesh = torch.meshgrid(values_mesh)
    coordinates_mesh = [x.reshape(-1, 1) for x in mesh]

    stride = torch.cat(coordinates_mesh, dim=1) + torch.tensor(start)

    return stride


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
