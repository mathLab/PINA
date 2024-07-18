"""Module for Radial Basis kernel (RBF) interpolation class."""

from abc import ABC, abstractmethod
import torch
import math
import warnings
from ...label_tensor import LabelTensor
from itertools import combinations_with_replacement

def linear(r):
    return -r

def thin_plate_spline(r, eps=1e-7):
    r = torch.clamp(r, min=eps)
    return r**2 * torch.log(r)

def cubic(r):
    return r**3

def quintic(r):
    return -r**5

def multiquadric(r):
    return -torch.sqrt(r**2 + 1)

def inverse_multiquadric(r):
    return 1/torch.sqrt(r**2 + 1)

def inverse_quadratic(r):
    return 1/(r**2 + 1)

def gaussian(r):
    return torch.exp(-r**2)

RADIAL_FUNCS = {
   "linear": linear,
   "thin_plate_spline": thin_plate_spline,
   "cubic": cubic,
   "quintic": quintic,
   "multiquadric": multiquadric,
   "inverse_multiquadric": inverse_multiquadric,
   "inverse_quadratic": inverse_quadratic,
   "gaussian": gaussian
   }

SCALE_INVARIANT = {"linear", "thin_plate_spline", "cubic", "quintic"}

MIN_DEGREE = {
    "multiquadric": 0,
    "linear": 0,
    "thin_plate_spline": 1,
    "cubic": 1,
    "quintic": 2
    }


class RBFLayer(torch.nn.Module):
    """
    Radial Basis Function (RBF) interpolation layer.

    Parameters
    ----------
    neighbors : int or None
        Number of neighbors to use in the interpolation. If None, all points
        are used.
    smoothing : float or tensor
        Smoothing parameter for the RBF interpolation. If a scalar, the same
        smoothing is used for all points. If a tensor, it must have the same
        length as the first dimension of `y`.
    kernel : str
        Radial basis function to use. Must be one of "linear", "thin_plate_spline",
        "cubic", "quintic", "multiquadric", "inverse_multiquadric", "inverse_quadratic",
        or "gaussian".
    epsilon : float
        Scaling parameter for the kernel. Must be specified for some kernels.
    degree : int or None
        Degree of the polynomial to include in the interpolation. If None, the
        degree is chosen automatically.
    device : str
        Device on which to perform calculations. Default is "cpu".

    Attributes
    ----------
    y : (n, d) tensor
        Data points.
    d : (n, m) tensor
        Data values.
    smoothing : (n,) tensor
        Smoothing parameter for each data point.
    kernel : str
        Radial basis function to use.
    epsilon : float
        Scaling parameter for the kernel.
    degree : int
        Degree of the polynomial to include in the interpolation.
    device : str
        Device on which to perform calculations.
    powers : (r, d) tensor
        Powers for each monomial in the polynomial.
    _shift : (d,) tensor
        Shift for the data.
    _scale : (d,) tensor
        Scale for the data.
    _coeffs : (n + r, m) tensor
        Coefficients for the RBF interpolation.

    """
    def __init__(
        self,
        neighbors=None,
        smoothing=0.0,
        kernel="thin_plate_spline",
        epsilon=None,
        degree=None,
        device="cpu",
    ):
        super().__init__()

        self.device = device
        self.smoothing = smoothing
        self.neighbors = neighbors
        self.kernel = kernel
        self.epsilon = epsilon
        self.degree = degree

    def _check_data(self, y, d):
        if y.ndim != 2:
            raise ValueError("y must be a 2-dimensional tensor.")

        self.ny, self.ndim = y.shape
        if d.shape[0] != self.ny:
            raise ValueError(
                "The first dim of d must have the same length as the first dim of y."
            )

        self.d_shape = d.shape[1:]
        d = d.reshape((self.ny, -1))

        if isinstance(self.smoothing, (int, float)):
            self.smoothing = torch.full((self.ny,), self.smoothing,
                    device=self.device).float()
        elif not isinstance(self.smoothing, torch.Tensor):
            raise ValueError("`smoothing` must be a scalar or a 1-dimensional tensor.")

        self.kernel = self.kernel.lower()
        if self.kernel not in RADIAL_FUNCS:
            raise ValueError(f"Unknown kernel: {self.kernel}")

        if self.epsilon is None:
            if self.kernel in SCALE_INVARIANT:
                self.epsilon = 1.0
            else:
                raise ValueError("Must specify `epsilon` for this kernel.")
        else:
            self.epsilon = float(self.epsilon)

        min_degree = MIN_DEGREE.get(self.kernel, -1)
        if self.degree is None:
            self.degree = max(min_degree, 0)
        else:
            self.degree = int(self.degree)
            if self.degree < -1:
                raise ValueError("`degree` must be at least -1.")
            elif self.degree < min_degree:
                warnings.warn(
                    f"`degree` is too small for this kernel. Setting to {min_degree}.",
                    UserWarning,
                )

    def fit(self, y, d):
        self._check_data(y, d)
        if self.neighbors is None:
            nobs = self.ny
        else:
            raise ValueError("neighbors currently not supported")

        powers = monomial_powers(self.ndim, self.degree).to(device=self.device)
        if powers.shape[0] > nobs:
            raise ValueError("The data is not compatible with the requested degree.")

        if self.neighbors is None:
            self._shift, self._scale, self._coeffs = solve(y, d,
                    self.smoothing, self.kernel, self.epsilon, powers)

        self.powers = powers
        self.y = y
        self.d = d

    def forward(self, x):
        """
        Returns the interpolated data at the given points `x`.

        @param x: (n, d) tensor of points at which to query the interpolator
        @param use_grad (optional): bool, whether to use Torch autograd when
            querying the interpolator. Default is False.

        Returns a (n, m) tensor of interpolated data.
        """
        if x.ndim != 2:
            raise ValueError("`x` must be a 2-dimensional tensor.")

        nx, ndim = x.shape
        if ndim != self.y.shape[1]:
            raise ValueError(
                "Expected the second dim of `x` to have length "
                f"{self.y.shape[1]}."
            )

        kernel_func = RADIAL_FUNCS[self.kernel]

        yeps = self.y * self.epsilon
        xeps = x * self.epsilon
        xhat = (x - self._shift) / self._scale

        kv = kernel_vector(xeps, yeps, kernel_func)
        p = polynomial_matrix(xhat, self.powers)
        vec = torch.cat([kv, p], dim=1)
        out = torch.matmul(vec, self._coeffs)
        out = out.reshape((nx,) + self.d_shape)
        return out


def kernel_vector(x, y, kernel_func):
    """Evaluate radial functions with centers `y` for all points in `x`."""
    return kernel_func(torch.cdist(x, y))


def polynomial_matrix(x, powers):
    """Evaluate monomials at `x` with given `powers`"""
    x_ = torch.repeat_interleave(x, repeats=powers.shape[0], dim=0)
    powers_ = powers.repeat(x.shape[0], 1)
    return torch.prod(x_**powers_, dim=1).view(x.shape[0], powers.shape[0])


def kernel_matrix(x, kernel_func):
    """Returns radial function values for all pairs of points in `x`."""
    return kernel_func(torch.cdist(x, x))


def monomial_powers(ndim, degree):
    """Return the powers for each monomial in a polynomial.

    Parameters
    ----------
    ndim : int
        Number of variables in the polynomial.
    degree : int
        Degree of the polynomial.

    Returns
    -------
    (nmonos, ndim) int ndarray
        Array where each row contains the powers for each variable in a
        monomial.

    """
    nmonos = math.comb(degree + ndim, ndim)
    out = torch.zeros((nmonos, ndim), dtype=torch.int32)
    count = 0
    for deg in range(degree + 1):
        for mono in combinations_with_replacement(range(ndim), deg):
            for var in mono:
                out[count, var] += 1
            count += 1

    return out


def build(y, d, smoothing, kernel, epsilon, powers):
    """Build the RBF linear system"""

    p = d.shape[0]
    s = d.shape[1]
    r = powers.shape[0]
    kernel_func = RADIAL_FUNCS[kernel]

    mins = torch.min(y, dim=0).values
    maxs = torch.max(y, dim=0).values
    shift = (maxs + mins) / 2
    scale = (maxs - mins) / 2

    scale[scale == 0.0] = 1.0

    yeps = y * epsilon
    yhat = (y - shift) / scale

    lhs = torch.empty((p + r, p + r), device=d.device).float()
    lhs[:p, :p] = kernel_matrix(yeps, kernel_func)
    lhs[:p, p:] = polynomial_matrix(yhat, powers)
    lhs[p:, :p] = lhs[:p, p:].T
    lhs[p:, p:] = 0.0
    lhs[:p, :p] += torch.diag(smoothing)

    rhs = torch.empty((r + p, s), device=d.device).float()
    rhs[:p] = d
    rhs[p:] = 0.0

    return lhs, rhs, shift, scale


def solve(y, d, smoothing, kernel, epsilon, powers):
    """Build then solve the RBF linear system"""

    lhs, rhs, shift, scale = build(y, d, smoothing, kernel, epsilon, powers)
    try:
        coeffs = torch.linalg.solve(lhs, rhs)
    except RuntimeError:  # singular matrix
        if coeffs is None:
            msg = "Singular matrix."
            nmonos = powers.shape[0]
            if nmonos > 0:
                pmat = polynomial_matrix((y - shift) / scale, powers)
                rank = torch.linalg.matrix_rank(pmat)
                if rank < nmonos:
                    msg = (
                        "Singular matrix. The matrix of monomials evaluated at "
                        "the data point coordinates does not have full column "
                        f"rank ({rank}/{nmonos})."
                    )

            raise ValueError(msg)

    return shift, scale, coeffs
