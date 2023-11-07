"""Module for FeedForward model"""
import torch
import torch.nn as nn
from ..utils import check_consistency
from .layers.residual import EnhancedLinear


class FeedForward(torch.nn.Module):
    """
    The PINA implementation of feedforward network, also refered as multilayer
    perceptron.

    :param int input_dimensions: The number of input components of the model.
        Expected tensor shape of the form :math:`(*, d)`, where *
        means any number of dimensions including none, and :math:`d` the ``input_dimensions``.
    :param int output_dimensions: The number of output components of the model.
        Expected tensor shape of the form :math:`(*, d)`, where *
        means any number of dimensions including none, and :math:`d` the ``output_dimensions``.
    :param int inner_size: number of neurons in the hidden layer(s). Default is
        20.
    :param int n_layers: number of hidden layers. Default is 2.
    :param func: the activation function to use. If a single
        :class:`torch.nn.Module` is passed, this is used as activation function
        after any layers, except the last one. If a list of Modules is passed,
        they are used as activation functions at any layers, in order.
    :param list(int) | tuple(int) layers: a list containing the number of neurons for
        any hidden layers. If specified, the parameters ``n_layers`` e
        ``inner_size`` are not considered.
    :param bool bias: If ``True`` the MLP will consider some bias.
    """

    def __init__(self,
                 input_dimensions,
                 output_dimensions,
                 inner_size=20,
                 n_layers=2,
                 func=nn.Tanh,
                 layers=None,
                 bias=True):
        """
        """
        super().__init__()

        if not isinstance(input_dimensions, int):
            raise ValueError('input_dimensions expected to be int.')
        self.input_dimension = input_dimensions

        if not isinstance(output_dimensions, int):
            raise ValueError('output_dimensions expected to be int.')
        self.output_dimension = output_dimensions
        if layers is None:
            layers = [inner_size] * n_layers

        tmp_layers = layers.copy()
        tmp_layers.insert(0, self.input_dimension)
        tmp_layers.append(self.output_dimension)

        self.layers = []
        for i in range(len(tmp_layers) - 1):
            self.layers.append(
                nn.Linear(tmp_layers[i], tmp_layers[i + 1], bias=bias))

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

        :param x: The tensor to apply the forward pass.
        :type x: torch.Tensor
        :return: the output computed by the model.
        :rtype: torch.Tensor
        """
        return self.model(x)


class ResidualFeedForward(torch.nn.Module):
    """
    The PINA implementation of feedforward network, also with skipped connection
    and transformer network, as presented in **Understanding and mitigating gradient
    pathologies in physics-informed neural networks**

    .. seealso::

        **Original reference**: Wang, Sifan, Yujun Teng, and Paris Perdikaris. 
        "Understanding and mitigating gradient flow pathologies in physics-informed
        neural networks." SIAM Journal on Scientific Computing 43.5 (2021): A3055-A3081.
        DOI: `10.1137/20M1318043
        <https://epubs.siam.org/doi/abs/10.1137/20M1318043>`_


    :param int input_dimensions: The number of input components of the model.
        Expected tensor shape of the form :math:`(*, d)`, where *
        means any number of dimensions including none, and :math:`d` the ``input_dimensions``.
    :param int output_dimensions: The number of output components of the model.
        Expected tensor shape of the form :math:`(*, d)`, where *
        means any number of dimensions including none, and :math:`d` the ``output_dimensions``.
    :param int inner_size: number of neurons in the hidden layer(s). Default is
        20.
    :param int n_layers: number of hidden layers. Default is 2.
    :param func: the activation function to use. If a single
        :class:`torch.nn.Module` is passed, this is used as activation function
        after any layers, except the last one. If a list of Modules is passed,
        they are used as activation functions at any layers, in order.
    :param bool bias: If ``True`` the MLP will consider some bias.
    :param list | tuple transformer_nets: a list or tuple containing the two
        torch.nn.Module which act as transformer network. The input dimension
        of the network must be the same as ``input_dimensions``, and the output
        dimension must be the same as ``inner_size``.
    """

    def __init__(self,
                 input_dimensions,
                 output_dimensions,
                 inner_size=20,
                 n_layers=2,
                 func=nn.Tanh,
                 bias=True,
                 transformer_nets=None):
        """
        """
        super().__init__()

        # check type consistency
        check_consistency(input_dimensions, int)
        check_consistency(output_dimensions, int)
        check_consistency(inner_size, int)
        check_consistency(n_layers, int)
        check_consistency(func, torch.nn.Module, subclass=True)
        check_consistency(bias, bool)

        # check transformer nets
        if transformer_nets is None:
            transformer_nets = [
                EnhancedLinear(
                    nn.Linear(in_features=input_dimensions,
                              out_features=inner_size), nn.Tanh()),
                EnhancedLinear(
                    nn.Linear(in_features=input_dimensions,
                              out_features=inner_size), nn.Tanh())
            ]
        elif isinstance(transformer_nets, (list, tuple)):
            if len(transformer_nets) != 2:
                raise ValueError(
                    'transformer_nets needs to be a list of len two.')
            for net in transformer_nets:
                if not isinstance(net, nn.Module):
                    raise ValueError(
                        'transformer_nets needs to be a list of torch.nn.Module.'
                    )
                x = torch.rand(10, input_dimensions)
                try:
                    out = net(x)
                except RuntimeError:
                    raise ValueError(
                        'transformer network input incompatible with input_dimensions.'
                    )
                if out.shape[-1] != inner_size:
                    raise ValueError(
                        'transformer network output incompatible with inner_size.'
                    )
        else:
            RuntimeError(
                'Runtime error for transformer nets, check official documentation.'
            )

        # assign variables
        self.input_dimension = input_dimensions
        self.output_dimension = output_dimensions
        self.transformer_nets = nn.ModuleList(transformer_nets)

        # build layers
        layers = [inner_size] * n_layers

        tmp_layers = layers.copy()
        tmp_layers.insert(0, self.input_dimension)

        self.layers = []
        for i in range(len(tmp_layers) - 1):
            self.layers.append(
                nn.Linear(tmp_layers[i], tmp_layers[i + 1], bias=bias))
        self.last_layer = nn.Linear(tmp_layers[len(tmp_layers) - 1],
                                    output_dimensions,
                                    bias=bias)

        if isinstance(func, list):
            self.functions = func()
        else:
            self.functions = [func() for _ in range(len(self.layers))]

        if len(self.layers) != len(self.functions):
            raise RuntimeError('uncosistent number of layers and functions')

        unique_list = []
        for layer, func in zip(self.layers, self.functions):
            unique_list.append(EnhancedLinear(layer=layer, activation=func))
        self.inner_layers = torch.nn.Sequential(*unique_list)

    def forward(self, x):
        """
        Defines the computation performed at every call.

        :param x: The tensor to apply the forward pass.
        :type x: torch.Tensor
        :return: the output computed by the model.
        :rtype: torch.Tensor
        """
        # enhance the input with transformer
        input_ = []
        for nets in self.transformer_nets:
            input_.append(nets(x))

        # skip connections pass
        for layer in self.inner_layers.children():
            x = layer(x)
            x = (1. - x) * input_[0] + x * input_[1]

        # last layer
        return self.last_layer(x)
