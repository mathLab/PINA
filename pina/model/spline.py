"""Module for the Spline model class."""

import torch
from ..utils import check_consistency


class Spline(torch.nn.Module):
    """
    Spline model class.
    """

    def __init__(
        self, order=4, knots=None, control_points=None, grid_extension=True
    ):
        """
        Initialization of the :class:`Spline` class.

        :param int order: The order of the spline. Default is ``4``.
        :param torch.Tensor knots: The tensor representing knots. If ``None``,
            the knots will be initialized automatically. Default is ``None``.
        :param torch.Tensor control_points: The control points. Default is
            ``None``.
        :raises ValueError: If the order is negative.
        :raises ValueError: If both knots and control points are ``None``.
        :raises ValueError: If the knot tensor is not one or two dimensional.
        """
        super().__init__()

        check_consistency(order, int)

        if order < 0:
            raise ValueError("Spline order cannot be negative.")
        if knots is None and control_points is None:
            raise ValueError("Knots and control points cannot be both None.")

        self.order = order
        self.k = order - 1
        self.grid_extension = grid_extension

        # Cache for performance optimization
        self._boundary_interval_idx = None

        if knots is not None and control_points is not None:
            self.knots = knots
            self.control_points = control_points

        elif knots is not None:
            print("Warning: control points will be initialized automatically.")
            print("         experimental feature")

            self.knots = knots
            n = len(knots) - order
            self.control_points = torch.nn.Parameter(
                torch.zeros(n), requires_grad=True
            )

        elif control_points is not None:
            print("Warning: knots will be initialized automatically.")
            print("         experimental feature")

            self.control_points = control_points

            n = len(self.control_points) - 1
            self.knots = {
                "type": "auto",
                "min": 0,
                "max": 1,
                "n": n + 2 + self.order,
            }

        else:
            raise ValueError("Knots and control points cannot be both None.")

        if self.knots.ndim > 2:
            raise ValueError("Knot vector must be one or two-dimensional.")

        # Precompute boundary interval index for performance
        self._compute_boundary_interval()

    def _compute_boundary_interval(self):
        """
        Precompute the rightmost non-degenerate interval index for performance.
        This avoids the search loop in the basis function on every call.
        """
        # Handle multi-dimensional knots
        if self.knots.ndim > 1:
            # For multi-dimensional knots, we'll handle boundary detection in
            # the basis function
            self._boundary_interval_idx = None
            return

        # For 1D knots, find the rightmost non-degenerate interval
        for i in range(len(self.knots) - 2, -1, -1):
            if self.knots[i] < self.knots[i + 1]:  # Non-degenerate interval found
                self._boundary_interval_idx = i
                return

        self._boundary_interval_idx = len(self.knots) - 2 if len(self.knots) > 1 else 0

    def basis(self, x, k, knots):
        """
        Compute the basis functions for the spline using an iterative approach.
        This is a vectorized implementation based on the Cox-de Boor recursion.

        :param torch.Tensor x: The points to be evaluated.
        :param int k: The spline degree.
        :param torch.Tensor knots: The tensor of knots.
        :return: The basis functions evaluated at x
        :rtype: torch.Tensor
        """

        if x.ndim == 1:
            x = x.unsqueeze(1)  # (batch_size, 1)
        if x.ndim == 2:
            x = x.unsqueeze(2)  # (batch_size, in_dim, 1)

        if knots.ndim == 1:
            knots = knots.unsqueeze(0)  # (1, n_knots)
        if knots.ndim == 2:
            knots = knots.unsqueeze(0)  # (1, in_dim, n_knots)

        # Base case: k=0
        basis = (x >= knots[..., :-1]) & (x < knots[..., 1:])
        basis = basis.to(x.dtype)

        if self._boundary_interval_idx is not None:
            i = self._boundary_interval_idx
            tolerance = 1e-10
            x_squeezed = x.squeeze(-1)
            knot_left = knots[..., i]
            knot_right = knots[..., i + 1]

            at_right_boundary = torch.abs(x_squeezed - knot_right) <= tolerance
            in_rightmost_interval = (
                x_squeezed >= knot_left
            ) & at_right_boundary

            if torch.any(in_rightmost_interval):
                # For points at the boundary, ensure they're included in the
                # rightmost interval
                basis[..., i] = torch.logical_or(
                    basis[..., i].bool(), in_rightmost_interval
                ).to(basis.dtype)

        # Iterative step (Cox-de Boor recursion)
        for i in range(1, k + 1):
            # First term of the recursion
            denom1 = knots[..., i:-1] - knots[..., : -(i + 1)]
            denom1 = torch.where(
                torch.abs(denom1) < 1e-8, torch.ones_like(denom1), denom1
            )
            numer1 = x - knots[..., : -(i + 1)]
            term1 = (numer1 / denom1) * basis[..., :-1]

            denom2 = knots[..., i + 1 :] - knots[..., 1:-i]
            denom2 = torch.where(
                torch.abs(denom2) < 1e-8, torch.ones_like(denom2), denom2
            )
            numer2 = knots[..., i + 1 :] - x
            term2 = (numer2 / denom2) * basis[..., 1:]

            basis = term1 + term2

        return basis

    def compute_control_points(self, x_eval, y_eval):
        """
        Compute control points from given evaluations using least squares.
        This method fits the control points to match the target y_eval values.
        """
        # (batch, in_dim)
        A = self.basis(x_eval, self.k, self.knots)
        # (batch, in_dim, n_basis)

        in_dim = A.shape[1]
        out_dim = y_eval.shape[2]
        n_basis = A.shape[2]
        c = torch.zeros(in_dim, out_dim, n_basis).to(A.device)

        for i in range(in_dim):
            # A_i is (batch, n_basis)
            # y_i is (batch, out_dim)
            A_i = A[:, i, :]
            y_i = y_eval[:, i, :]
            c_i = torch.linalg.lstsq(A_i, y_i).solution  # (n_basis, out_dim)
            c[i, :, :] = c_i.T  # (out_dim, n_basis)

        self.control_points = torch.nn.Parameter(c)

    def forward(self, x):
        """
        Forward pass for the :class:`Spline` model.

        :param torch.Tensor x: The input tensor.
        :return: The output tensor.
        :rtype: torch.Tensor
        """
        t = self.knots
        k = self.k
        c = self.control_points

        # Create the basis functions
        # B will have shape (batch, in_dim, n_basis)
        B = self.basis(x, k, t)

        # KAN case where control points are (in_dim, out_dim, n_basis)
        if c.ndim == 3:
            y_ij = torch.einsum(
                "bil,iol->bio", B, c
            )  # (batch, in_dim, out_dim)
            # sum over input dimensions
            y = torch.sum(y_ij, dim=1)  # (batch, out_dim)
        # Original test case
        else:
            B = B.squeeze(1)  # (batch, n_basis)
            if c.ndim == 1:
                y = torch.einsum("bi,i->b", B, c)
            else:
                y = torch.einsum("bi,ij->bj", B, c)

        return y

    @property
    def control_points(self):
        """
        The control points of the spline.

        :return: The control points.
        :rtype: torch.Tensor
        """
        return self._control_points

    @control_points.setter
    def control_points(self, value):
        """
        Set the control points of the spline.

        :param value: The control points.
        :type value: torch.Tensor | dict
        :raises ValueError: If invalid value is passed.
        """
        if isinstance(value, dict):
            if "n" not in value:
                raise ValueError("Invalid value for control_points")
            n = value["n"]
            dim = value.get("dim", 1)
            value = torch.zeros(n, dim)

        if not isinstance(value, torch.nn.Parameter):
            value = torch.nn.Parameter(value)

        if not isinstance(value, torch.Tensor):
            raise ValueError("Invalid value for control_points")
        self._control_points = value

    @property
    def knots(self):
        """
        The knots of the spline.

        :return: The knots.
        :rtype: torch.Tensor
        """
        return self._knots

    @knots.setter
    def knots(self, value):
        """
        Set the knots of the spline.

        :param value: The knots.
        :type value: torch.Tensor | dict
        :raises ValueError: If invalid value is passed.
        """
        if isinstance(value, dict):

            type_ = value.get("type", "auto")
            min_ = value.get("min", 0)
            max_ = value.get("max", 1)
            n = value.get("n", 10)

            if type_ == "uniform":
                value = torch.linspace(min_, max_, n + self.k + 1)
            elif type_ == "auto":
                initial_knots = torch.ones(self.order + 1) * min_
                final_knots = torch.ones(self.order + 1) * max_

                if n < self.order + 1:
                    value = torch.concatenate((initial_knots, final_knots))
                elif n - 2 * self.order + 1 == 1:
                    value = torch.Tensor([(max_ + min_) / 2])
                else:
                    value = torch.linspace(min_, max_, n - 2 * self.order - 1)

                value = torch.concatenate((initial_knots, value, final_knots))

        if not isinstance(value, torch.Tensor):
            raise ValueError("Invalid value for knots")

        self._knots = value

        # Recompute boundary interval when knots change
        if hasattr(self, "_boundary_interval_idx"):
            self._compute_boundary_interval()