"""Module for Radial Basis Function Interpolation layer."""

import math
import warnings
from itertools import combinations_with_replacement
import torch
from ...utils import check_consistency

def linear(r):
    '''
    Linear radial basis function.
    '''
    return -r

def thin_plate_spline(r, eps=1e-7):
    '''
    Thin plate spline radial basis function.
    '''
    r = torch.clamp(r, min=eps)
    return r**2 * torch.log(r)

def cubic(r):
    '''
    Cubic radial basis function.
    '''
    return r**3

def quintic(r):
    '''
    Quintic radial basis function.
    '''
    return -r**5

def multiquadric(r):
    '''
    Multiquadric radial basis function.
    '''
    return -torch.sqrt(r**2 + 1)

def inverse_multiquadric(r):
    '''
    Inverse multiquadric radial basis function.
    '''
    return 1/torch.sqrt(r**2 + 1)

def inverse_quadratic(r):
    '''
    Inverse quadratic radial basis function.
    '''
    return 1/(r**2 + 1)

def gaussian(r):
    '''
    Gaussian radial basis function.
    '''
    return torch.exp(-r**2)

radial_functions = {
   "linear": linear,
   "thin_plate_spline": thin_plate_spline,
   "cubic": cubic,
   "quintic": quintic,
   "multiquadric": multiquadric,
   "inverse_multiquadric": inverse_multiquadric,
   "inverse_quadratic": inverse_quadratic,
   "gaussian": gaussian
   }

scale_invariant = {"linear", "thin_plate_spline", "cubic", "quintic"}

min_degree_funcs = {
    "multiquadric": 0,
    "linear": 0,
    "thin_plate_spline": 1,
    "cubic": 1,
    "quintic": 2
    }


class RBFBlock(torch.nn.Module):
    """
    Radial Basis Function (RBF) interpolation layer. It need to be fitted with
    the data with the method :meth:`fit`, before it can be used to interpolate
    new points. The layer is not trainable.

    .. note::
        It reproduces the implementation of ``scipy.interpolate.RBFBlock`` and
        it is inspired from the implementation in `torchrbf.
        <https://github.com/ArmanMaesumi/torchrbf>`_
    """
    def __init__(
        self,
        neighbors=None,
        smoothing=0.0,
        kernel="thin_plate_spline",
        epsilon=None,
        degree=None,
    ):
        """
        :param int neighbors: Number of neighbors to use for the
            interpolation.
            If ``None``, use all data points.
        :param float smoothing: Smoothing parameter for the interpolation.
            if 0.0, the interpolation is exact and no smoothing is applied.
        :param str kernel: Radial basis function to use. Must be one of
            ``linear``, ``thin_plate_spline``, ``cubic``, ``quintic``,
            ``multiquadric``, ``inverse_multiquadric``, ``inverse_quadratic``,
            or ``gaussian``.
        :param float epsilon: Shape parameter that scaled the input to
            the RBF. This defaults to 1 for kernels in ``scale_invariant``
            dictionary, and must be specified for other kernels.
        :param int degree: Degree of the added polynomial.
            For some kernels, there exists a minimum degree of the polynomial
            such that the RBF is well-posed. Those minimum degrees are specified
            in the `min_degree_funcs` dictionary above. If `degree` is less than
            the minimum degree, a warning is raised and the degree is set to the
            minimum value.
        """

        super().__init__()
        check_consistency(neighbors, (int, type(None)))
        check_consistency(smoothing, (int, float, torch.Tensor))
        check_consistency(kernel, str)
        check_consistency(epsilon, (float, type(None)))
        check_consistency(degree, (int, type(None)))

        self.neighbors = neighbors
        self.smoothing = smoothing
        self.kernel = kernel
        self.epsilon = epsilon
        self.degree = degree
        self.powers = None
        # initialize data points and values
        self.y = None
        self.d = None
        # initialize attributes for the fitted model
        self._shift = None
        self._scale = None
        self._coeffs = None

    @property
    def smoothing(self):
        """
        Smoothing parameter for the interpolation.

        :rtype: float
        """
        return self._smoothing

    @smoothing.setter
    def smoothing(self, value):
        self._smoothing = value

    @property
    def kernel(self):
        """
        Radial basis function to use.

        :rtype: str
        """
        return self._kernel

    @kernel.setter
    def kernel(self, value):
        if value not in radial_functions:
            raise ValueError(f"Unknown kernel: {value}")
        self._kernel = value.lower()

    @property
    def epsilon(self):
        """
        Shape parameter that scaled the input to the RBF.

        :rtype: float
        """
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        if value is None:
            if self.kernel in scale_invariant:
                value = 1.0
            else:
                raise ValueError("Must specify `epsilon` for this kernel.")
        else:
            value = float(value)
        self._epsilon = value

    @property
    def degree(self):
        """
        Degree of the added polynomial.

        :rtype: int
        """
        return self._degree

    @degree.setter
    def degree(self, value):
        min_degree = min_degree_funcs.get(self.kernel, -1)
        if value is None:
            value = max(min_degree, 0)
        else:
            value = int(value)
            if value < -1:
                raise ValueError("`degree` must be at least -1.")
            if value < min_degree:
                warnings.warn(
                    "`degree` is too small for this kernel. Setting to "
                    f"{min_degree}.", UserWarning,
                )
        self._degree = value

    def _check_data(self, y, d):
        if y.ndim != 2:
            raise ValueError("y must be a 2-dimensional tensor.")

        if d.shape[0] != y.shape[0]:
            raise ValueError(
                "The first dim of d must have the same length as "
                "the first dim of y."
            )

        if isinstance(self.smoothing, (int, float)):
            self.smoothing = torch.full((y.shape[0],), self.smoothing
                    ).float().to(y.device)

    def fit(self, y, d):
        """
        Fit the RBF interpolator to the data.

        :param torch.Tensor y: (n, d) tensor of data points.
        :param torch.Tensor d: (n, m) tensor of data values.
        """
        self._check_data(y, d)

        self.y = y
        self.d = d

        if self.neighbors is None:
            nobs = self.y.shape[0]
        else:
            raise NotImplementedError("neighbors currently not supported")

        powers = RBFBlock.monomial_powers(self.y.shape[1], self.degree).to(
                y.device)
        if powers.shape[0] > nobs:
            raise ValueError("The data is not compatible with the "
                "requested degree.")

        if self.neighbors is None:
            self._shift, self._scale, self._coeffs = RBFBlock.solve(self.y,
                    self.d.reshape((self.y.shape[0], -1)),
                    self.smoothing, self.kernel, self.epsilon, powers)

        self.powers = powers

    def forward(self, x):
        """
        Returns the interpolated data at the given points `x`.

        :param torch.Tensor x: `(n, d)` tensor of points at which
            to query the interpolator

        :rtype: `(n, m)` torch.Tensor of interpolated data.
        """
        if x.ndim != 2:
            raise ValueError("`x` must be a 2-dimensional tensor.")

        nx, ndim = x.shape
        if ndim != self.y.shape[1]:
            raise ValueError(
                "Expected the second dim of `x` to have length "
                f"{self.y.shape[1]}."
            )

        kernel_func = radial_functions[self.kernel]

        yeps = self.y * self.epsilon
        xeps = x * self.epsilon
        xhat = (x - self._shift) / self._scale

        kv = RBFBlock.kernel_vector(xeps, yeps, kernel_func)
        p = RBFBlock.polynomial_matrix(xhat, self.powers)
        vec = torch.cat([kv, p], dim=1)
        out = torch.matmul(vec, self._coeffs)
        out = out.reshape((nx,) + self.d.shape[1:])
        return out

    @staticmethod
    def kernel_vector(x, y, kernel_func):
        """
        Evaluate radial functions with centers `y` for all points in `x`.

        :param torch.Tensor x: `(n, d)` tensor of points.
        :param torch.Tensor y: `(m, d)` tensor of centers.
        :param str kernel_func: Radial basis function to use.

        :rtype: `(n, m)` torch.Tensor of radial function values.
        """
        return kernel_func(torch.cdist(x, y))

    @staticmethod
    def polynomial_matrix(x, powers):
        """
        Evaluate monomials at `x` with given `powers`.

        :param torch.Tensor x: `(n, d)` tensor of points.
        :param torch.Tensor powers: `(r, d)` tensor of powers for each monomial.

        :rtype: `(n, r)` torch.Tensor of monomial values.
        """
        x_ = torch.repeat_interleave(x, repeats=powers.shape[0], dim=0)
        powers_ = powers.repeat(x.shape[0], 1)
        return torch.prod(x_**powers_, dim=1).view(x.shape[0], powers.shape[0])

    @staticmethod
    def kernel_matrix(x, kernel_func):
        """
        Returns radial function values for all pairs of points in `x`.

        :param torch.Tensor x: `(n, d`) tensor of points.
        :param str kernel_func: Radial basis function to use.

        :rtype: `(n, n`) torch.Tensor of radial function values.
        """
        return kernel_func(torch.cdist(x, x))

    @staticmethod
    def monomial_powers(ndim, degree):
        """
        Return the powers for each monomial in a polynomial.

        :param int ndim: Number of variables in the polynomial.
        :param int degree: Degree of the polynomial.

        :rtype: `(nmonos, ndim)` torch.Tensor where each row contains the powers
            for each variable in a monomial.

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

    @staticmethod
    def build(y, d, smoothing, kernel, epsilon, powers):
        """
        Build the RBF linear system.

        :param torch.Tensor y: (n, d) tensor of data points.
        :param torch.Tensor d: (n, m) tensor of data values.
        :param torch.Tensor smoothing: (n,) tensor of smoothing parameters.
        :param str kernel: Radial basis function to use.
        :param float epsilon: Shape parameter that scaled the input to the RBF.
        :param torch.Tensor powers: (r, d) tensor of powers for each monomial.

        :rtype: (lhs, rhs, shift, scale) where `lhs` and `rhs` are the
            left-hand side and right-hand side of the linear system, and
            `shift` and `scale` are the shift and scale parameters.
        """
        p = d.shape[0]
        s = d.shape[1]
        r = powers.shape[0]
        kernel_func = radial_functions[kernel]

        mins = torch.min(y, dim=0).values
        maxs = torch.max(y, dim=0).values
        shift = (maxs + mins) / 2
        scale = (maxs - mins) / 2

        scale[scale == 0.0] = 1.0

        yeps = y * epsilon
        yhat = (y - shift) / scale

        lhs = torch.empty((p + r, p + r), device=d.device).float()
        lhs[:p, :p] = RBFBlock.kernel_matrix(yeps, kernel_func)
        lhs[:p, p:] = RBFBlock.polynomial_matrix(yhat, powers)
        lhs[p:, :p] = lhs[:p, p:].T
        lhs[p:, p:] = 0.0
        lhs[:p, :p] += torch.diag(smoothing)

        rhs = torch.empty((r + p, s), device=d.device).float()
        rhs[:p] = d
        rhs[p:] = 0.0
        return lhs, rhs, shift, scale

    @staticmethod
    def solve(y, d, smoothing, kernel, epsilon, powers):
        """
        Build then solve the RBF linear system.

        :param torch.Tensor y: (n, d) tensor of data points.
        :param torch.Tensor d: (n, m) tensor of data values.
        :param torch.Tensor smoothing: (n,) tensor of smoothing parameters.

        :param str kernel: Radial basis function to use.
        :param float epsilon: Shape parameter that scaled the input to the RBF.
        :param torch.Tensor powers: (r, d) tensor of powers for each monomial.

        :raises ValueError: If the linear system is singular.

        :rtype: (shift, scale, coeffs) where `shift` and `scale` are the
            shift and scale parameters, and `coeffs` are the coefficients
            of the interpolator
        """

        lhs, rhs, shift, scale = RBFBlock.build(y, d, smoothing, kernel,
                epsilon, powers)
        try:
            coeffs = torch.linalg.solve(lhs, rhs)
        except RuntimeError as e:
            msg = "Singular matrix."
            nmonos = powers.shape[0]
            if nmonos > 0:
                pmat = RBFBlock.polynomial_matrix((y - shift) / scale, powers)
                rank = torch.linalg.matrix_rank(pmat)
                if rank < nmonos:
                    msg = (
                        "Singular matrix. The matrix of monomials evaluated at "
                        "the data point coordinates does not have full column "
                        f"rank ({rank}/{nmonos})."
                    )

            raise ValueError(msg) from e

        return shift, scale, coeffs
