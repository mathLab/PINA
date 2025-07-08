import torch
from torch import nn
from ...utils import check_consistency


class M_H(nn.Module):
    def __init__(self, num_scalar_features, num_modes):
        super().__init__()

        scale = 1.0 / (num_scalar_features**2)
        self.weights = nn.Parameter(
            scale
            * torch.rand(
                num_modes,
                num_scalar_features,
                num_scalar_features,
                dtype=torch.cfloat,
            )
        )

    def forward(self, f_h):
        """
        Args: Input tensor (time_discretizations, nodes, scalar features)
        Returns: Output tensor (modes, nodes, scalar features)
        """
        num_modes = self.weights.shape[0]

        fft_f_h = torch.fft.fft(f_h, dim=0)
        fft_f_h = fft_f_h[:, :num_modes, ...]

        M_h_dot_f_h = torch.einsum("ijl, inl -> inj", self.weights, fft_f_h)

        return torch.fft.ifft(M_h_dot_f_h)


class M_Z(nn.Module):
    def __init__(self, num_3d_tensors, num_modes):
        super().__init__()

        scale = 1.0 / (num_3d_tensors**2)
        self.weights = nn.Parameter(
            scale
            * torch.rand(
                num_modes, num_3d_tensors, num_3d_tensors, dtype=torch.cfloat
            )
        )

    def forward(self, f_z):
        """
        Args: Input tensor (time_discretizations, nodes, 3d_tensors features)
        Returns: Output tensor (modes, nodes, 3d_tensor_features)
        """
        num_modes = self.weights.shape[0]

        fft_f_z = torch.fft.fft(f_z, dim=0)
        fft_f_z = fft_f_z[:, :num_modes, ...]

        M_z_dot_f_z = torch.einsum("isl, inlT -> insT", self.weights, fft_f_z)

        return torch.fft.ifft(M_z_dot_f_z)


class TemporalConvolutionLayer(nn.Module):
    def __init__(
        self,
        time_discretizations,
        n_nodes,
        f_h_size,
        f_z_size,
        n_modes,
        equivariant_activation_tensor_features,
        activation_scalar_features,
    ):

        super().__init__()

        # Check type consistency
        check_consistency(n_nodes, int)
        check_consistency(f_h_size, int)
        check_consistency(f_z_size, int)
        check_consistency(time_discretizations, int)

        # assign variables
        self._f_h_size = f_h_size
        self._f_z_size = f_z_size

        self.M_H = M_H(time_discretizations, n_nodes, f_h_size, n_modes)
        self.M_Z = M_Z(time_discretizations, n_nodes, f_z_size, n_modes)

        self._equivariant_activation_tensor_features = (
            equivariant_activation_tensor_features
        )
        self._activation_scalar_features = activation_scalar_features

    def forward(self, x):
        """
        Args: Input tensor x is dim =
        (Time Discretization, Nodes, (Scalar Features + 3d Vect Features * 3))
        """
        f_h, f_z = x.split([self._f_h_size, 3 * self._f_z_size], dim=-1)

        time_discretizations = self.M_H.shape[0]
        n_nodes = self.M_H.shape[1]

        f_z = f_z.reshape(
            time_discretizations,
            n_nodes,
            self._f_z_size,
            3,
        )

        # apply weights
        out_f_h = self.M_H(f_h)
        out_f_z = self.M_Z(f_z)

        # reshape to remove compress coordinate dimension
        out_f_z = f_z.reshape(time_discretizations, n_nodes, -1)

        out_f_h = self._activation_scalar_features(out_f_h)
        out_f_z = self._equivariant_activation_tensor_features(out_f_z)

        return x + torch.concat([out_f_h, out_f_z], dim=-1)

    def forward(self, f_h, f_z):
        """
        Args: Input tensors: scalar feature tensor (Time Discretization, Nodes, Scalar Features) and 3d feature tensor (Time Discretization, Nodes, 3d Vec Features, 3)
        Returns: Tuple: Scalar feature tensor and 3d feature tensor
        """

        # apply weights
        out_f_h = self.M_H(f_h)
        out_f_z = self.M_Z(f_z)

        out_f_h = self._activation_scalar_features(out_f_h)
        out_f_z = self._equivariant_activation_tensor_features(out_f_z)

        return f_h + out_f_h, f_z + out_f_z
