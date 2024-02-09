import torch
import torch.nn as nn
from ..utils import check_consistency
from .layers.fourier import FourierBlock1D, FourierBlock2D, FourierBlock3D
from pina import LabelTensor
import warnings


class FNO(torch.nn.Module):
    """
    The PINA implementation of Fourier Neural Operator network.

    Fourier Neural Operator (FNO) is a general architecture for learning Operators.
    Unlike traditional machine learning methods FNO is designed to map
    entire functions to other functions. It can be trained both with
    Supervised learning strategies. FNO does global convolution by performing the
    operation on the Fourier space.

    .. seealso::

        **Original reference**: Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B.,
        Bhattacharya, K., Stuart, A., & Anandkumar, A. (2020). *Fourier neural operator for
        parametric partial differential equations*.
        DOI: `arXiv preprint arXiv:2010.08895.
        <https://arxiv.org/abs/2010.08895>`_
    """

    def __init__(
        self,
        lifting_net,
        projecting_net,
        n_modes,
        dimensions=3,
        padding=8,
        padding_type="constant",
        inner_size=20,
        n_layers=2,
        func=nn.Tanh,
        layers=None,
    ):
        super().__init__()

        # check type consistency
        check_consistency(lifting_net, nn.Module)
        check_consistency(projecting_net, nn.Module)
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
                " More information on the official documentation."
            )

        # assign variables
        # TODO check input lifting net and input projecting net
        self._lifting_net = lifting_net
        self._projecting_net = projecting_net
        self._padding = padding

        # initialize fourier layer for each dimension
        if dimensions == 1:
            fourier_layer = FourierBlock1D
        elif dimensions == 2:
            fourier_layer = FourierBlock2D
        elif dimensions == 3:
            fourier_layer = FourierBlock3D
        else:
            raise NotImplementedError("FNO implemented only for 1D/2D/3D data.")

        # Here we build the FNO by stacking Fourier Blocks

        # 1. Assign output dimensions for each FNO layer
        if layers is None:
            layers = [inner_size] * n_layers

        # 2. Assign activation functions for each FNO layer
        if isinstance(func, list):
            if len(layers) != len(func):
                raise RuntimeError(
                    "Uncosistent number of layers and functions."
                )
            self._functions = func
        else:
            self._functions = [func for _ in range(len(layers))]

        # 3. Assign modes functions for each FNO layer
        if isinstance(n_modes, list):
            if all(isinstance(i, list) for i in n_modes) and len(layers) != len(
                n_modes
            ):
                raise RuntimeError(
                    "Uncosistent number of layers and functions."
                )
            elif all(isinstance(i, int) for i in n_modes):
                n_modes = [n_modes] * len(layers)
        else:
            n_modes = [n_modes] * len(layers)

        # 4. Build the FNO network
        tmp_layers = layers.copy()
        first_parameter = next(lifting_net.parameters())
        input_shape = first_parameter.size()
        out_feats = lifting_net(torch.rand(size=input_shape)).shape[-1]
        tmp_layers.insert(0, out_feats)

        self._layers = []
        for i in range(len(tmp_layers) - 1):
            self._layers.append(
                fourier_layer(
                    input_numb_fields=tmp_layers[i],
                    output_numb_fields=tmp_layers[i + 1],
                    n_modes=n_modes[i],
                    activation=self._functions[i],
                )
            )
        self._layers = nn.Sequential(*self._layers)

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

        :param torch.Tensor x: The input tensor for fourier block, depending on
            ``dimension`` in the initialization. In particular it is expected
            * 1D tensors: ``[batch, X, channels]``
            * 2D tensors: ``[batch, X, Y, channels]``
            * 3D tensors: ``[batch, X, Y, Z, channels]``
        :return: The output tensor obtained from the FNO.
        :rtype: torch.Tensor
        """
        if isinstance(x, LabelTensor):  # TODO remove when Network is fixed
            warnings.warn(
                "LabelTensor passed as input is not allowed, casting LabelTensor to Torch.Tensor"
            )
            x = x.as_subclass(torch.Tensor)

        # lifting the input in higher dimensional space
        x = self._lifting_net(x)

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

        # apply projecting operator and return
        return self._projecting_net(x)
