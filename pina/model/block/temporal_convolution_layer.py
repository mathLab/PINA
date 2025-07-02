import torch
from torch import nn
from pina.utils import check_consistency


class M_H(nn.Module):
    def __init__(self, time_discretizations, num_nodes, num_scalar_features, num_modes):
        super().__init__()

        self._P = time_discretizations
        self._N = num_nodes
        self._k = num_scalar_features
        self._I = num_modes

        self._weights = nn.Parameter(
            torch.rand(
                self._I,
                self._k,
                self._k,
                dtype=torch.cfloat
            )
        )

    def forward(self, f_h):
        '''
        Args: Input tensor (batches, time_discretizations, nodes, scalar features)
        Returns: Output tensor (batches, nodes, modes, scalar features)
        '''

        fft_f_h = torch.fft.fft(f_h, dim=1)
        fft_f_h_truncated = fft_f_h[:,:self._I,...]

        # could be potentially speed up using hermitian property
        # seems to already do this in torch
        M_h_dot_f_h = torch.einsum("ijl, binl -> bnij", self._weights, fft_f_h_truncated)

        return torch.fft.ifft(M_h_dot_f_h)


class M_Z(nn.Module):
    def __init__(self, time_discretizations, num_nodes, num_3d_tensors, num_modes):
        super().__init__()

        self._P = time_discretizations
        self._N = num_nodes
        self._m = num_3d_tensors
        self._I = num_modes

        self._weights = nn.Parameter(
            torch.rand(
                self._I,
                self._m,
                self._m,
                dtype=torch.cfloat
            )
        )

    def forward(self, f_z):
        '''
        Args: Input tensor (batches, time_discretizations, nodes, 3d_tensors features)
        Returns: Output tensor (batches, nodes, modes, 3d_tensor_features)
        '''

        fft_f_z = torch.fft.fft(f_z, dim=1)
        fft_f_z_truncated = fft_f_z[:,:self._I,...]

        M_z_dot_f_z = torch.einsum("isl, binlT -> bnisT", self._weights, fft_f_z_truncated)

        return torch.fft.ifft(M_z_dot_f_z)


class TemporalConvolutionLayer(nn.Module):
    def __init__(self,
                 time_discretizations,
                 n_nodes,
                 f_h_size,
                 f_z_size,
                 n_modes,
                 equivariant_activation,
                 other_activation
    ):
        
        super().__init__()

        # Check type consistency
        check_consistency(n_nodes, int)
        check_consistency(f_h_size, int)
        check_consistency(f_z_size, int)
        check_consistency(time_discretizations, int)

        # assign variables
        self._nodes = n_nodes
        self._modes = n_modes
        self._f_h_size = f_h_size
        self._f_z_size = f_z_size
        self._time_discretizations = time_discretizations

        self.M_H = M_H(time_discretizations, n_nodes, f_h_size, n_modes)
        self.M_Z = M_Z(time_discretizations, n_nodes, f_z_size, n_modes)

        self._equivariant_activation = equivariant_activation
        self._other_activation = other_activation

    def forward(self, x):
        '''
        Args: Input tensor x is dim = 
        (Batches, Time Discretization, Nodes, (Scalar Features + 3d Vect Features * 3))
        '''
        f_h, f_z = x.split([self._f_h_size, 3 * self._f_z_size], dim=-1)

        # reshape to have a coordinate dimension
        batch_size = x.shape[0]
        f_z = f_z.reshape(batch_size, self._time_discretizations, self._nodes, self._f_z_size, 3)

        # apply weights
        out_f_h = self.M_H(f_h)
        out_f_z = self.M_Z(f_z)

        # reshape to remove compress coordinate dimension
        out_f_z = f_z.reshape(batch_size, self._time_discretizations, self._nodes, -1)

        out_f_h = self._other_activation(out_f_h)
        out_f_z = self._equivariant_activation(out_f_z)
        
        return x + torch.concat([out_f_h, out_f_z], dim=-1)