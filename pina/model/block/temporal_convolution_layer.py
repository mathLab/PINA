import torch
from torch import nn
from pina.utils import check_consistency


class ParameterizedSpectralConvBlock3D(nn.Module):
    def __init__(self, upper_left_size, bottom_right_size, n_modes):
        
        super().__init__()

        # Check type consistency
        check_consistency(upper_left_size, int)
        check_consistency(bottom_right_size, int)


        # assign variables
        self._modes = n_modes
        self.upper_left_size = upper_left_size
        self.bottom_right_size = bottom_right_size


        # scaling factor
        #! Why do we need scaling factor?
        scale_ul = 1.0 / (self.upper_left_size ** 2)
        scale_br = 1.0 / (self.bottom_right_size ** 2)


        self._upper_left_weights = nn.Parameter(
            scale_ul * torch.rand(
                self.upper_left_size,
                self.upper_left_size,
                self._modes, #! what's up with modes
                dtype=torch.cfloat
            )
        )
        self._bottom_right_weights = nn.Parameter(
            scale_br * torch.rand(
                self.bottom_right_size,
                self.bottom_right_size,
                self._modes,
                dtype=torch.cfloat
            )
        )

    def _compute_mult(self, input, weights):
        #! this is not the right matrix mul, understand size of tensors
        #! that we are working with
        return torch.einsum("ij, jk -> ik", input, weights)

    def forward(self, x):
        # Would like to split the input x into f_h and f_z like the paper
        f_h, f_z = torch.split(x, [self.upper_left_size, self.bottom_right_size])

        # Fourier Transform on both
        f_h_ft = torch.fft.rfftn(f_h,dim=[-3,-2,-1])
        f_z_ft = torch.fft.rfftn(f_z, dim=[-3,-2,-1])

        # Learnable Matrix Multiply

        # avoid block matrix issues by separating
        # block_matrix = torch.block_diag(self._upper_left_weights, self._bottom_right_weights)

        out_f_h = self._compute_mult(f_h_ft, self._upper_left_weights)
        out_f_z = self._compute_mult(f_z_ft, self._bottom_right_weights)

        # Inverse FFT
        out_f_h = torch.fft.irfftn(out_f_h, dim=[-3,-2,-1])
        out_f_z = torch.fft.irfftn(out_f_z, dim=[-3,-2,-1])

        return torch.concat([out_f_h, out_f_z])



        

#? Temporal Convolution Layer $T_{theta}$
#? $(T_{\theta} f)(t) = f(t) + \sigma\left( (K_{\theta} f)(t) \right)$


class TemporalConvolutionLayer3D(nn.Module):
    """
    The inner block of a Equivariant Graph Neural Operator for 1-dimensional input tensors.

    """

    def __init__(
        self,
        upper_left_size,
        bottom_right_size,
        n_modes,
        activation=torch.nn.Tanh,
    ):

        #! need to recreate spectral_conv to ensure equivariance
        self._spectral_conv = ParameterizedSpectralConvBlock3D(
            bottom_right_size=bottom_right_size,
            upper_left_size=upper_left_size,
            n_modes=n_modes,
        )
        self._activation = activation
        self._linear = nn.Linear(input_numb_fields, output_numb_fields)

    def forward(self, x):
        """
        Forward pass of the block. It performs a #?spectral convolution and a
        #?linear transformation of the input. Then, it sums the results.

        :param torch.Tensor x: The input tensor for performing the computation.
        :return: The output tensor.
        :rtype: torch.Tensor
        """

        #? Here we have the residual connection as well as everything else
        return x + self.activation(self._spectral_conv(x) + self._linear(x))