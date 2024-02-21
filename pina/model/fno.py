"""
Fourier Neural Operator Module.
"""

import torch
import torch.nn as nn
from pina import LabelTensor
import warnings
from ..utils import check_consistency
from .layers.fourier import FourierBlock1D, FourierBlock2D, FourierBlock3D
from .base_no import KernelNeuralOperator


class FourierIntegralKernel(torch.nn.Module):
    """
    Implementation of Fourier Integral Kernel network.

    This class implements the Fourier Integral Kernel network, which is a
    PINA implementation of Fourier Neural Operator kernel network.
    It performs global convolution by operating in the Fourier space.

    .. seealso::

        **Original reference**: Li, Z., Kovachki, N., Azizzadenesheli,
        K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A.
        (2020). *Fourier neural operator for parametric partial
        differential equations*.
        DOI: `arXiv preprint arXiv:2010.08895.
        <https://arxiv.org/abs/2010.08895>`_
    """
    def __init__(self,
                 input_numb_fields,
                 output_numb_fields,
                 n_modes,
                 dimensions=3,
                 padding=8,
                 padding_type="constant",
                 inner_size=20,
                 n_layers=2,
                 func=nn.Tanh,
                 layers=None):
        """
        :param int input_numb_fields: Number of input fields.
        :param int output_numb_fields: Number of output fields.
        :param int | list[int] n_modes: Number of modes.
        :param int dimensions: Number of dimensions (1, 2, or 3).
        :param int padding: Padding size, defaults to 8.
        :param str padding_type: Type of padding, defaults to "constant".
        :param int inner_size: Inner size, defaults to 20.
        :param int n_layers: Number of layers, defaults to 2.
        :param torch.nn.Module func: Activation function, defaults to nn.Tanh.
        :param list[int] layers: List of layer sizes, defaults to None.
        """
        super().__init__()

        # check type consistency
        check_consistency(dimensions, int)
        check_consistency(padding, int)
        check_consistency(padding_type, str)
        check_consistency(inner_size, int)
        check_consistency(n_layers, int)
        check_consistency(func, nn.Module, subclass=True)

        if layers is not None:
            if isinstance(layers, (tuple, list)):
                check_consistency(layers, int)
            else:
                raise ValueError("layers must be tuple or list of int.")
        if not isinstance(n_modes, (list, tuple, int)):
            raise ValueError(
                "n_modes must be a int or list or tuple of valid modes."
                " More information on the official documentation.")

        # assign padding
        self._padding = padding

        # initialize fourier layer for each dimension
        if dimensions == 1:
            fourier_layer = FourierBlock1D
        elif dimensions == 2:
            fourier_layer = FourierBlock2D
        elif dimensions == 3:
            fourier_layer = FourierBlock3D
        else:
            raise NotImplementedError(
                "FNO implemented only for 1D/2D/3D data."
                )

        # Here we build the FNO kernels by stacking Fourier Blocks

        # 1. Assign output dimensions for each FNO layer
        if layers is None:
            layers = [inner_size] * n_layers

        # 2. Assign activation functions for each FNO layer
        if isinstance(func, list):
            if len(layers) != len(func):
                raise RuntimeError(
                    'Uncosistent number of layers and functions.')
            _functions = func
        else:
            _functions = [func for _ in range(len(layers) - 1)]
        _functions.append(torch.nn.Identity)

        # 3. Assign modes functions for each FNO layer
        if isinstance(n_modes, list):
            if all(isinstance(i, list)
                   for i in n_modes) and len(layers) != len(n_modes):
                raise RuntimeError(
                    "Uncosistent number of layers and functions.")
            elif all(isinstance(i, int) for i in n_modes):
                n_modes = [n_modes] * len(layers)
        else:
            n_modes = [n_modes] * len(layers)

        # 4. Build the FNO network
        _layers = []
        tmp_layers = [input_numb_fields] + layers + [output_numb_fields]
        for i in range(len(layers)):
            _layers.append(
                fourier_layer(input_numb_fields=tmp_layers[i],
                              output_numb_fields=tmp_layers[i + 1],
                              n_modes=n_modes[i],
                              activation=_functions[i]))
        self._layers = nn.Sequential(*_layers)

        # 5. Padding values for spectral conv
        if isinstance(padding, int):
            padding = [padding] * dimensions
        self._ipad = [-pad if pad > 0 else None for pad in padding[:dimensions]]
        self._padding_type = padding_type
        self._pad = [
            val for pair in zip([0] * dimensions, padding) for val in pair
        ]

    def forward(self, x):
        """
        Forward computation for Fourier Neural Operator. It performs a
        lifting of the input by the ``lifting_net``. Then different layers
        of Fourier Blocks are applied. Finally the output is projected
        to the final dimensionality by the ``projecting_net``.

        :param torch.Tensor x: The input tensor for fourier block,
            depending on ``dimension`` in the initialization.
            In particular it is expected:

            * 1D tensors: ``[batch, X, channels]``
            * 2D tensors: ``[batch, X, Y, channels]``
            * 3D tensors: ``[batch, X, Y, Z, channels]``
        :return: The output tensor obtained from the kernels convolution.
        :rtype: torch.Tensor
        """
        if isinstance(x, LabelTensor):  #TODO remove when Network is fixed
            warnings.warn(
                'LabelTensor passed as input is not allowed,'
                ' casting LabelTensor to Torch.Tensor'
            )
            x = x.as_subclass(torch.Tensor)
        # permuting the input [batch, channels, x, y, ...]
        permutation_idx = [0, x.ndim - 1, *[i for i in range(1, x.ndim - 1)]]
        x = x.permute(permutation_idx)

        # padding the input
        x = torch.nn.functional.pad(x, pad=self._pad, mode=self._padding_type)

        # apply fourier layers
        x = self._layers(x)

        # remove padding
        idxs = [slice(None), slice(None)] + [slice(pad) for pad in self._ipad]
        x = x[idxs]

        # permuting back [batch, x, y, ..., channels]
        permutation_idx = [0, *[i for i in range(2, x.ndim)], 1]
        x = x.permute(permutation_idx)

        return x


class FNO(KernelNeuralOperator):
    """
    The PINA implementation of Fourier Neural Operator network.

    Fourier Neural Operator (FNO) is a general architecture for
    learning Operators. Unlike traditional machine learning methods
    FNO is designed to map entire functions to other functions. It
    can be trained with Supervised learning strategies. FNO does global
    convolution by performing the operation on the Fourier space.

    .. seealso::

        **Original reference**: Li, Z., Kovachki, N., Azizzadenesheli,
        K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A.
        (2020). *Fourier neural operator for parametric partial
        differential equations*.
        DOI: `arXiv preprint arXiv:2010.08895.
        <https://arxiv.org/abs/2010.08895>`_
    """
    def __init__(self,
                 lifting_net,
                 projecting_net,
                 n_modes,
                 dimensions=3,
                 padding=8,
                 padding_type="constant",
                 inner_size=20,
                 n_layers=2,
                 func=nn.Tanh,
                 layers=None):
        """
        :param torch.nn.Module lifting_net: The neural network for lifting
            the input.
        :param torch.nn.Module projecting_net: The neural network for
            projecting the output.
        :param int | list[int] n_modes: Number of modes.
        :param int dimensions: Number of dimensions (1, 2, or 3).
        :param int padding: Padding size, defaults to 8.
        :param str padding_type: Type of padding, defaults to `constant`.
        :param int inner_size: Inner size, defaults to 20.
        :param int n_layers: Number of layers, defaults to 2.
        :param torch.nn.Module func: Activation function, defaults to nn.Tanh.
        :param list[int] layers: List of layer sizes, defaults to None.
        """
        lifting_operator_out = lifting_net(
            torch.rand(size=next(lifting_net.parameters()).size())).shape[-1]
        super().__init__(lifting_operator=lifting_net,
                         projection_operator=projecting_net,
                         integral_kernels=FourierIntegralKernel(
                             input_numb_fields=lifting_operator_out,
                             output_numb_fields=next(
                                 projecting_net.parameters()).size(),
                             n_modes=n_modes,
                             dimensions=dimensions,
                             padding=padding,
                             padding_type=padding_type,
                             inner_size=inner_size,
                             n_layers=n_layers,
                             func=func,
                             layers=layers))

    def forward(self, x):
        """
        Forward computation for Fourier Neural Operator. It performs a
        lifting of the input by the ``lifting_net``. Then different layers
        of Fourier Blocks are applied. Finally the output is projected
        to the final dimensionality by the ``projecting_net``.

        :param torch.Tensor x: The input tensor for fourier block,
            depending on ``dimension`` in the initialization. In
            particular it is expected:
            
            * 1D tensors: ``[batch, X, channels]``
            * 2D tensors: ``[batch, X, Y, channels]``
            * 3D tensors: ``[batch, X, Y, Z, channels]``
        :return: The output tensor obtained from FNO.
        :rtype: torch.Tensor
        """
        return super().forward(x)
