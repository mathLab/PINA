import torch
from torch import nn
from pina.utils import check_consistency


class M_H(nn.Module):
    def __init__(self, k, n_modes):
        super().__init__()

        self._modes = n_modes
        self._k = k

        self._weights = nn.Parameter(
            torch.rand(
                self._modes,
                self._k,
                self._k,
                dtype=torch.cfloat
            )
        )


    def forward(self, fh):
        batch_size = fh.shape[0]

        #todo What is the correct Fourier transform?
        F_h = torch.fft.fft(fh)

        # largely copied from spectral.py
        out_ft = torch.zeros(
            batch_size,
            fh.size(1) // 2 + 1, # allowed because FFT is Hermitian
            self._k,
            dtype = torch.cfloat
        )

        # (mode, channel, channel), (batch, mode, channel) -> (batch, mode, channel)
        out = torch.einsum("ijl, bil -> bij", self._weights, F_h[:,:self._modes,...])

        out_ft[:,:self._modes,...] = out

        return torch.fft.ifft(out)


     
class M_Z(nn.Module):
    def __init__(self, m, n_modes):
        super().__init__()

        self._modes = n_modes
        self.m = m

        self._weights = nn.Parameter(
            torch.rand(
                self._m,
                self._m,
                self._modes,
                dtype=torch.cfloat
            )
        )

    def forward(self, fz):
        batch_size = fz.shape[0]

        #todo What is the correct Fourier transform?
        F_z = torch.fft(fz)

        # largely copied from spectral.py
        out_ft = torch.zeros(
            batch_size,
            fz.size(1) // 2 + 1, # allowed because FFT is Hermitian
            self._k,
            dtype = torch.cfloat
        )

        # (mode, channel, channel), (batch, mode, channel, Coords) -> (batch, mode, channel, Coords)
        out = torch.einsum("isl, bilT -> bisT", self._weights, F_z)

        out_ft[:,:self._modes,...] = out

        return torch.fft.ifft(out)

#? Temporal Convolution Layer $T_{theta}$
#? $(T_{\theta} f)(t) = f(t) + \sigma\left( (K_{\theta} f)(t) \right)$

class TemporalConvolutionLayer(nn.Module):
    def __init__(self,
                 f_h_size,
                 f_z_size,
                 n_modes,
                 equivariant_activation,
                 other_activation
    ):
        
        super().__init__()

        # Check type consistency
        check_consistency(f_h_size, int)
        check_consistency(f_z_size, int)

        # assign variables
        self._modes = n_modes
        self.f_h_size = f_h_size
        self.f_z_size = f_z_size

        self.M_H = M_H(f_h_size, n_modes)
        self.M_Z = M_Z(f_z_size, n_modes)

        self._equivariant_activation = equivariant_activation
        self._other_activation = other_activation


    def forward(self, x):
        # Would like to split the input x into f_h and f_z like the paper
        f_h, f_z = torch.split(x, [self.upper_left_size, self.bottom_right_size])

        out_f_h = self.M_H(f_h)
        out_f_z = self.M_Z(f_z)

        out_f_h = self._other_activation(out_f_h)
        out_f_z = self._equivariant_activation(out_f_z)
        
        return x + torch.concat([out_f_h, out_f_z])