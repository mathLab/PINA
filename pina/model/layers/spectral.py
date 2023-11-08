import torch
import torch.nn as nn
from ...utils import check_consistency
import warnings


######## 1D Spectral Convolution ###########
class SpectralConvBlock1D(nn.Module):
    """
    PINA implementation of Spectral Convolution Block for one
    dimensional tensors.
    """

    def __init__(self, input_numb_fields, output_numb_fields, n_modes):
        """
        The module computes the spectral convolution of the input with a linear kernel in the
        fourier space, and then it maps the input back to the physical
        space.

        The block expects an input of size ``[batch, input_numb_fields, N]``
        and returns an output of size ``[batch, output_numb_fields, N]``.

        :param int input_numb_fields: The number of channels for the input.
        :param int output_numb_fields: The number of channels for the output.
        :param int n_modes: Number of modes to select, it must be at most equal
            to the ``floor(N/2)+1``.
        """
        super().__init__()

        # check type consistency
        check_consistency(input_numb_fields, int)
        check_consistency(output_numb_fields, int)

        # assign variables
        self._modes = n_modes
        self._input_channels = input_numb_fields
        self._output_channels = output_numb_fields

        # scaling factor
        scale = (1. / (self._input_channels * self._output_channels))
        self._weights = nn.Parameter(scale * torch.rand(self._input_channels,
                                                        self._output_channels,
                                                        self._modes,
                                                        dtype=torch.cfloat))

    def _compute_mult1d(self, input, weights):
        """
        Compute the matrix multiplication of the input
        with the linear kernel weights.

        :param input: The input tensor, expect of size 
            ``[batch, input_numb_fields, x]``.
        :type input: torch.Tensor
        :param weights: The kernel weights, expect of
            size ``[input_numb_fields, output_numb_fields, x]``.
        :type weights: torch.Tensor
        :return: The matrix multiplication of the input
            with the linear kernel weights.
        :rtype: torch.Tensor
        """
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        """
        Forward computation for Spectral Convolution.

        :param x: The input tensor, expect of size 
            ``[batch, input_numb_fields, x]``.
        :type x: torch.Tensor
        :return: The output tensor obtained from the
            spectral convolution of size ``[batch, output_numb_fields, x]``.
        :rtype: torch.Tensor
        """
        batch_size = x.shape[0]

        # Compute Fourier transform of the input
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batch_size,
                             self._output_channels,
                             x.size(-1) // 2 + 1,
                             device=x.device,
                             dtype=torch.cfloat)
        out_ft[:, :, :self._modes] = self._compute_mult1d(
            x_ft[:, :, :self._modes], self._weights)

        # Return to physical space
        return torch.fft.irfft(out_ft, n=x.size(-1))


######## 2D Spectral Convolution ###########
class SpectralConvBlock2D(nn.Module):
    """
    PINA implementation of spectral convolution block for two
    dimensional tensors.
    """

    def __init__(self, input_numb_fields, output_numb_fields, n_modes):
        """
        The module computes the spectral convolution of the input with a linear kernel in the
        fourier space, and then it maps the input back to the physical
        space.

        The block expects an input of size ``[batch, input_numb_fields, Nx, Ny]``
        and returns an output of size ``[batch, output_numb_fields, Nx, Ny]``.

        :param int input_numb_fields: The number of channels for the input.
        :param int output_numb_fields: The number of channels for the output.
        :param list | tuple n_modes: Number of modes to select for each dimension.
            It must be at most equal to the ``floor(Nx/2)+1`` and ``floor(Ny/2)+1``.
        """
        super().__init__()

        # check type consistency
        check_consistency(input_numb_fields, int)
        check_consistency(output_numb_fields, int)
        check_consistency(n_modes, int)
        if isinstance(n_modes, (tuple, list)):
            if len(n_modes) != 2:
                raise ValueError(
                    'Expected n_modes to be a list or tuple of len two, '
                    'with each entry corresponding to the number of modes '
                    'for each dimension ')
        elif isinstance(n_modes, int):
            n_modes = [n_modes] * 2
        else:
            raise ValueError(
                'Expected n_modes to be a list or tuple of len two, '
                'with each entry corresponding to the number of modes '
                'for each dimension; or an int value representing the '
                'number of modes for all dimensions')

        # assign variables
        self._modes = n_modes
        self._input_channels = input_numb_fields
        self._output_channels = output_numb_fields

        # scaling factor
        scale = (1. / (self._input_channels * self._output_channels))
        self._weights1 = nn.Parameter(scale * torch.rand(self._input_channels,
                                                         self._output_channels,
                                                         self._modes[0],
                                                         self._modes[1],
                                                         dtype=torch.cfloat))
        self._weights2 = nn.Parameter(scale * torch.rand(self._input_channels,
                                                         self._output_channels,
                                                         self._modes[0],
                                                         self._modes[1],
                                                         dtype=torch.cfloat))

    def _compute_mult2d(self, input, weights):
        """
        Compute the matrix multiplication of the input
        with the linear kernel weights.

        :param input: The input tensor, expect of size 
            ``[batch, input_numb_fields, x, y]``.
        :type input: torch.Tensor
        :param weights: The kernel weights, expect of
            size ``[input_numb_fields, output_numb_fields, x, y]``.
        :type weights: torch.Tensor
        :return: The matrix multiplication of the input
            with the linear kernel weights.
        :rtype: torch.Tensor
        """
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        """
        Forward computation for Spectral Convolution.

        :param x: The input tensor, expect of size 
            ``[batch, input_numb_fields, x, y]``.
        :type x: torch.Tensor
        :return: The output tensor obtained from the
            spectral convolution of size ``[batch, output_numb_fields, x, y]``.
        :rtype: torch.Tensor
        """

        batch_size = x.shape[0]

        # Compute Fourier transform of the input
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batch_size,
                             self._output_channels,
                             x.size(-2),
                             x.size(-1) // 2 + 1,
                             device=x.device,
                             dtype=torch.cfloat)
        out_ft[:, :, :self._modes[0], :self._modes[1]] = self._compute_mult2d(
            x_ft[:, :, :self._modes[0], :self._modes[1]], self._weights1)
        out_ft[:, :, -self._modes[0]:, :self._modes[1]:] = self._compute_mult2d(
            x_ft[:, :, -self._modes[0]:, :self._modes[1]], self._weights2)

        # Return to physical space
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


######## 3D Spectral Convolution ###########
class SpectralConvBlock3D(nn.Module):
    """
    PINA implementation of spectral convolution block for three
    dimensional tensors.
    """

    def __init__(self, input_numb_fields, output_numb_fields, n_modes):
        """
        The module computes the spectral convolution of the input with a linear kernel in the
        fourier space, and then it maps the input back to the physical
        space.

        The block expects an input of size ``[batch, input_numb_fields, Nx, Ny, Nz]``
        and returns an output of size ``[batch, output_numb_fields, Nx, Ny, Nz]``.

        :param int input_numb_fields: The number of channels for the input.
        :param int output_numb_fields: The number of channels for the output.
        :param list | tuple n_modes: Number of modes to select for each dimension.
            It must be at most equal to the ``floor(Nx/2)+1``, ``floor(Ny/2)+1``
            and ``floor(Nz/2)+1``.
        """
        super().__init__()

        # check type consistency
        check_consistency(input_numb_fields, int)
        check_consistency(output_numb_fields, int)
        check_consistency(n_modes, int)
        if isinstance(n_modes, (tuple, list)):
            if len(n_modes) != 3:
                raise ValueError(
                    'Expected n_modes to be a list or tuple of len three, '
                    'with each entry corresponding to the number of modes '
                    'for each dimension ')
        elif isinstance(n_modes, int):
            n_modes = [n_modes] * 3
        else:
            raise ValueError(
                'Expected n_modes to be a list or tuple of len three, '
                'with each entry corresponding to the number of modes '
                'for each dimension; or an int value representing the '
                'number of modes for all dimensions')

        # assign variables
        self._modes = n_modes
        self._input_channels = input_numb_fields
        self._output_channels = output_numb_fields

        # scaling factor
        scale = (1. / (self._input_channels * self._output_channels))
        self._weights1 = nn.Parameter(scale * torch.rand(self._input_channels,
                                                         self._output_channels,
                                                         self._modes[0],
                                                         self._modes[1],
                                                         self._modes[2],
                                                         dtype=torch.cfloat))
        self._weights2 = nn.Parameter(scale * torch.rand(self._input_channels,
                                                         self._output_channels,
                                                         self._modes[0],
                                                         self._modes[1],
                                                         self._modes[2],
                                                         dtype=torch.cfloat))
        self._weights3 = nn.Parameter(scale * torch.rand(self._input_channels,
                                                         self._output_channels,
                                                         self._modes[0],
                                                         self._modes[1],
                                                         self._modes[2],
                                                         dtype=torch.cfloat))
        self._weights4 = nn.Parameter(scale * torch.rand(self._input_channels,
                                                         self._output_channels,
                                                         self._modes[0],
                                                         self._modes[1],
                                                         self._modes[2],
                                                         dtype=torch.cfloat))

    def _compute_mult3d(self, input, weights):
        """
        Compute the matrix multiplication of the input
        with the linear kernel weights.

        :param input: The input tensor, expect of size 
            ``[batch, input_numb_fields, x, y, z]``.
        :type input: torch.Tensor
        :param weights: The kernel weights, expect of
            size ``[input_numb_fields, output_numb_fields, x, y, z]``.
        :type weights: torch.Tensor
        :return: The matrix multiplication of the input
            with the linear kernel weights.
        :rtype: torch.Tensor
        """
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x):
        """
        Forward computation for Spectral Convolution.

        :param x: The input tensor, expect of size 
            ``[batch, input_numb_fields, x, y, z]``.
        :type x: torch.Tensor
        :return: The output tensor obtained from the
            spectral convolution of size ``[batch, output_numb_fields, x, y, z]``.
        :rtype: torch.Tensor
        """

        batch_size = x.shape[0]

        # Compute Fourier transform of the input
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batch_size,
                             self._output_channels,
                             x.size(-3),
                             x.size(-2),
                             x.size(-1) // 2 + 1,
                             device=x.device,
                             dtype=torch.cfloat)

        slice0 = (
            slice(None),
            slice(None),
            slice(self._modes[0]),
            slice(self._modes[1]),
            slice(self._modes[2]),
        )
        out_ft[slice0] = self._compute_mult3d(x_ft[slice0], self._weights1)

        slice1 = (
            slice(None),
            slice(None),
            slice(self._modes[0]),
            slice(-self._modes[1], None),
            slice(self._modes[2]),
        )
        out_ft[slice1] = self._compute_mult3d(x_ft[slice1], self._weights2)

        slice2 = (
            slice(None),
            slice(None),
            slice(-self._modes[0], None),
            slice(self._modes[1]),
            slice(self._modes[2]),
        )
        out_ft[slice2] = self._compute_mult3d(x_ft[slice2], self._weights3)

        slice3 = (
            slice(None),
            slice(None),
            slice(-self._modes[0], None),
            slice(-self._modes[1], None),
            slice(self._modes[2]),
        )
        out_ft[slice3] = self._compute_mult3d(x_ft[slice3], self._weights4)

        # Return to physical space
        return torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
