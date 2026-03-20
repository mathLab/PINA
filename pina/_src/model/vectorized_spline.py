"""Vectorized univariate B-spline model."""

import torch
import torch.nn as nn


class VectorizedSpline(nn.Module):
    """
    Vectorized univariate B-spline model (shared knots, many splines).

    Notation:
      - knots: shape (m,)
      - order: k (degree = k-1)
      - n_ctrl = m - k
      - control_points:
          * (S, n_ctrl)            -> S splines, scalar output each
          * (S, O, n_ctrl)         -> S splines, O outputs each (like multiple channels)
    Input:
      - x: shape (...,) or (..., B)
    Output:
      - if control_points is (S, n_ctrl):      shape (..., S)
      - if control_points is (S, O, n_ctrl):  shape (..., S, O)
    """

    def __init__(
        self,
        order: int,
        knots: torch.Tensor,
        control_points: torch.Tensor | None = None,
    ):
        super().__init__()
        if not isinstance(order, int) or order <= 0:
            raise ValueError("order must be a positive integer.")
        if not isinstance(knots, torch.Tensor):
            raise ValueError("knots must be a torch.Tensor.")
        if knots.ndim != 1:
            raise ValueError("knots must be 1D.")

        self.order = order

        # store sorted knots as buffer
        knots_sorted = knots.sort().values
        self.register_buffer("knots", knots_sorted)

        n_ctrl = knots_sorted.numel() - order
        if n_ctrl <= 0:
            raise ValueError(
                f"Need #knots > order. Got #knots={knots_sorted.numel()} order={order}."
            )

        # boundary interval idx for rightmost inclusion
        self._boundary_interval_idx = self._compute_boundary_interval_idx(
            knots_sorted
        )

        # # control points init
        # if control_points is None:
        #     # default: one spline
        #     cp = torch.zeros(1, n_ctrl, dtype=knots_sorted.dtype, device=knots_sorted.device)
        #     self.control_points = nn.Parameter(cp, requires_grad=True)
        # else:
        #     if not isinstance(control_points, torch.Tensor):
        #         raise ValueError("control_points must be a torch.Tensor or None.")
        #     if control_points.ndim not in (2, 3):
        #         raise ValueError("control_points must have shape (S, n_ctrl) or (S, O, n_ctrl).")
        #     if control_points.shape[-1] != n_ctrl:
        #         raise ValueError(
        #             f"Last dim of control_points must be n_ctrl={n_ctrl}. Got {control_points.shape[-1]}."
        #         )
        self.control_points = nn.Parameter(control_points, requires_grad=True)

    @staticmethod
    def _compute_boundary_interval_idx(knots: torch.Tensor) -> int:
        if knots.numel() < 2:
            return 0
        diffs = knots[1:] - knots[:-1]
        valid = torch.nonzero(diffs > 0, as_tuple=False)
        if valid.numel() == 0:
            return 0
        return int(valid[-1])

    def basis(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute B-spline basis functions of order self.order at x.

        Returns:
          basis: shape (..., n_ctrl)
        """
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)

        # ensure float dtype consistent
        x = x.to(dtype=self.knots.dtype, device=self.knots.device)

        # make x shape (..., 1) for broadcasting
        x_exp = x.unsqueeze(-1)  # (..., 1)

        # knots as (1, ..., 1, m) via unsqueeze to broadcast
        # (m,) -> (1,)*x.ndim + (m,)
        knots = self.knots.view(*([1] * x.ndim), -1)

        # order-1 base: indicator on intervals [t_i, t_{i+1})
        basis = ((x_exp >= knots[..., :-1]) & (x_exp < knots[..., 1:])).to(
            x_exp.dtype
        )  # (..., m-1)

        # include rightmost boundary in the last non-degenerate interval
        j = self._boundary_interval_idx
        knot_left = knots[..., j]
        knot_right = knots[..., j + 1]
        at_right = (x >= knot_left.squeeze(-1)) & torch.isclose(
            x, knot_right.squeeze(-1), rtol=1e-8, atol=1e-10
        )
        if torch.any(at_right):
            basis_j = basis[..., j].bool() | at_right
            basis[..., j] = basis_j.to(basis.dtype)

        # Cox-de Boor recursion up to order k
        # after i-th iteration, basis has length (m-1 - i)
        for i in range(1, self.order):
            denom1 = knots[..., i:-1] - knots[..., : -(i + 1)]
            denom2 = knots[..., i + 1 :] - knots[..., 1:-i]

            denom1 = torch.where(
                denom1.abs() < 1e-8, torch.ones_like(denom1), denom1
            )
            denom2 = torch.where(
                denom2.abs() < 1e-8, torch.ones_like(denom2), denom2
            )

            term1 = ((x_exp - knots[..., : -(i + 1)]) / denom1) * basis[
                ..., :-1
            ]
            term2 = ((knots[..., i + 1 :] - x_exp) / denom2) * basis[..., 1:]
            basis = term1 + term2

        # final basis length is n_ctrl = m - order
        return basis  # (..., n_ctrl)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate spline(s) at x.

        If control_points is (S, n_ctrl): output (..., S)
        If control_points is (S, O, n_ctrl): output (..., S, O)
        """
        B = self.basis(x)  # (..., n_ctrl)

        cp = self.control_points
        if cp.ndim == 2:
            # (S, n_ctrl)
            # want (..., S) = (..., n_ctrl) @ (n_ctrl, S)
            out = B @ cp.transpose(0, 1)
            return out
        else:
            # (S, O, n_ctrl)
            # Compute for each S: (..., n_ctrl) @ (n_ctrl, O) -> (..., O), then stack over S
            # vectorized using einsum (yes, this one is actually appropriate)
            # (..., n) * (S, O, n) -> (..., S, O)
            # out = torch.einsum("...n, son -> ...so", B, cp)
            out = torch.einsum("bsc,sco->bso", B, cp)

            return out

    def forward_basis(self, basis):
        """
        Evaluate spline(s) given precomputed basis.

        """
        cp = self.control_points
        if cp.ndim == 2:
            # (S, n_ctrl)
            # want (..., S) = (..., n_ctrl) @ (n_ctrl, S)
            out = basis @ cp.transpose(0, 1)
            return out
        else:
            # (S, O, n_ctrl)
            # Compute for each S: (..., n_ctrl) @ (n_ctrl, O) -> (..., O), then stack over S
            # vectorized using einsum (yes, this one is actually appropriate)
            # (..., n) * (S, O, n) -> (..., S, O)
            # out = torch.einsum("...n, son -> ...so", B, cp)
            out = torch.einsum("bsc,sco->bso", basis, cp)

            return out
