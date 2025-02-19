import torch
import torch.nn as nn
from ...utils import check_consistency

from pina.model.block import (
    SpectralConvBlock1D,
    SpectralConvBlock2D,
    SpectralConvBlock3D,
)


class FourierBlock1D(nn.Module):
    """
    Fourier block implementation for three dimensional
    input tensor. The combination of Fourier blocks
    make up the Fourier Neural Operator

    .. seealso::

        **Original reference**: Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B.,
        Bhattacharya, K., Stuart, A., & Anandkumar, A. (2020). *Fourier neural operator for
        parametric partial differential equations*.
        DOI: `arXiv preprint arXiv:2010.08895.
        <https://arxiv.org/abs/2010.08895>`_

    """

    def __init__(
        self,
        input_numb_fields,
        output_numb_fields,
        n_modes,
        activation=torch.nn.Tanh,
    ):
        super().__init__()
        """
        PINA implementation of Fourier block one dimension. The module computes
        the spectral convolution of the input with a linear kernel in the
        fourier space, and then it maps the input back to the physical
        space. The output is then added to a Linear tranformation of the
        input in the physical space. Finally an activation function is
        applied to the output. 

        The block expects an input of size ``[batch, input_numb_fields, N]``
        and returns an output of size ``[batch, output_numb_fields, N]``.

        :param int input_numb_fields: The number of channels for the input.
        :param int output_numb_fields: The number of channels for the output.
        :param list | tuple n_modes: Number of modes to select for each dimension.
            It must be at most equal to the ``floor(N/2)+1``.
        :param torch.nn.Module activation: The activation function.
        """
        # check type consistency
        check_consistency(activation(), nn.Module)

        # assign variables
        self._spectral_conv = SpectralConvBlock1D(
            input_numb_fields=input_numb_fields,
            output_numb_fields=output_numb_fields,
            n_modes=n_modes,
        )
        self._activation = activation()
        self._linear = nn.Conv1d(input_numb_fields, output_numb_fields, 1)

    def forward(self, x):
        """
        Forward computation for Fourier Block. It performs a spectral
        convolution and a linear transformation of the input and sum the
        results.

        :param x: The input tensor for fourier block, expect of size
            ``[batch, input_numb_fields, x]``.
        :type x: torch.Tensor
        :return: The output tensor obtained from the
            fourier block of size ``[batch, output_numb_fields, x]``.
        :rtype: torch.Tensor
        """
        return self._activation(self._spectral_conv(x) + self._linear(x))


class FourierBlock2D(nn.Module):
    """
    Fourier block implementation for two dimensional
    input tensor. The combination of Fourier blocks
    make up the Fourier Neural Operator

    .. seealso::

        **Original reference**: Li, Zongyi, et al.
        *Fourier neural operator for parametric partial
        differential equations*. arXiv preprint
        arXiv:2010.08895 (2020)
        <https://arxiv.org/abs/2010.08895.pdf>`_.

    """

    def __init__(
        self,
        input_numb_fields,
        output_numb_fields,
        n_modes,
        activation=torch.nn.Tanh,
    ):
        """
        PINA implementation of Fourier block two dimensions. The module computes
        the spectral convolution of the input with a linear kernel in the
        fourier space, and then it maps the input back to the physical
        space. The output is then added to a Linear tranformation of the
        input in the physical space. Finally an activation function is
        applied to the output.

        The block expects an input of size ``[batch, input_numb_fields, Nx, Ny]``
        and returns an output of size ``[batch, output_numb_fields, Nx, Ny]``.

        :param int input_numb_fields: The number of channels for the input.
        :param int output_numb_fields: The number of channels for the output.
        :param list | tuple n_modes: Number of modes to select for each dimension.
            It must be at most equal to the ``floor(Nx/2)+1`` and ``floor(Ny/2)+1``.
        :param torch.nn.Module activation: The activation function.
        """
        super().__init__()

        # check type consistency
        check_consistency(activation(), nn.Module)

        # assign variables
        self._spectral_conv = SpectralConvBlock2D(
            input_numb_fields=input_numb_fields,
            output_numb_fields=output_numb_fields,
            n_modes=n_modes,
        )
        self._activation = activation()
        self._linear = nn.Conv2d(input_numb_fields, output_numb_fields, 1)

    def forward(self, x):
        """
        Forward computation for Fourier Block. It performs a spectral
        convolution and a linear transformation of the input and sum the
        results.

        :param x: The input tensor for fourier block, expect of size
            ``[batch, input_numb_fields, x, y]``.
        :type x: torch.Tensor
        :return: The output tensor obtained from the
            fourier block of size ``[batch, output_numb_fields, x, y, z]``.
        :rtype: torch.Tensor
        """
        return self._activation(self._spectral_conv(x) + self._linear(x))


class FourierBlock3D(nn.Module):
    """
    Fourier block implementation for three dimensional
    input tensor. The combination of Fourier blocks
    make up the Fourier Neural Operator

    .. seealso::

        **Original reference**: Li, Zongyi, et al.
        *Fourier neural operator for parametric partial
        differential equations*. arXiv preprint
        arXiv:2010.08895 (2020)
        <https://arxiv.org/abs/2010.08895.pdf>`_.

    """

    def __init__(
        self,
        input_numb_fields,
        output_numb_fields,
        n_modes,
        activation=torch.nn.Tanh,
    ):
        """
        PINA implementation of Fourier block three dimensions. The module computes
        the spectral convolution of the input with a linear kernel in the
        fourier space, and then it maps the input back to the physical
        space. The output is then added to a Linear tranformation of the
        input in the physical space. Finally an activation function is
        applied to the output.

        The block expects an input of size ``[batch, input_numb_fields, Nx, Ny, Nz]``
        and returns an output of size ``[batch, output_numb_fields, Nx, Ny, Nz]``.

        :param int input_numb_fields: The number of channels for the input.
        :param int output_numb_fields: The number of channels for the output.
        :param list | tuple n_modes: Number of modes to select for each dimension.
            It must be at most equal to the ``floor(Nx/2)+1``, ``floor(Ny/2)+1``
            and ``floor(Nz/2)+1``.
        :param torch.nn.Module activation: The activation function.
        """
        super().__init__()

        # check type consistency
        check_consistency(activation(), nn.Module)

        # assign variables
        self._spectral_conv = SpectralConvBlock3D(
            input_numb_fields=input_numb_fields,
            output_numb_fields=output_numb_fields,
            n_modes=n_modes,
        )
        self._activation = activation()
        self._linear = nn.Conv3d(input_numb_fields, output_numb_fields, 1)

    def forward(self, x):
        """
        Forward computation for Fourier Block. It performs a spectral
        convolution and a linear transformation of the input and sum the
        results.

        :param x: The input tensor for fourier block, expect of size
            ``[batch, input_numb_fields, x, y, z]``.
        :type x: torch.Tensor
        :return: The output tensor obtained from the
            fourier block of size ``[batch, output_numb_fields, x, y, z]``.
        :rtype: torch.Tensor
        """
        return self._activation(self._spectral_conv(x) + self._linear(x))
