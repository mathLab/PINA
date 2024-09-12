import torch
import torch

from .domain_interface import DomainInterface
from ..label_tensor import LabelTensor
from ..utils import torch_lhs, chebyshev_roots


class CartesianDomain(DomainInterface):
    """PINA implementation of Hypercube domain."""

    def __init__(self, cartesian_dict):
        """
        :param cartesian_dict: A dictionary with dict-key a string representing
            the input variables for the pinn, and dict-value a list with
            the domain extrema.
        :type cartesian_dict: dict

        :Example:
            >>> spatial_domain = CartesianDomain({'x': [0, 1], 'y': [0, 1]})
        """
        self.fixed_ = {}
        self.range_ = {}

        for k, v in cartesian_dict.items():
            if isinstance(v, (int, float)):
                self.fixed_[k] = v
            elif isinstance(v, (list, tuple)) and len(v) == 2:
                self.range_[k] = v
            else:
                raise TypeError

    @property
    def variables(self):
        """Spatial variables.

        :return: Spatial variables defined in ``__init__()``
        :rtype: list[str]
        """
        return sorted(list(self.fixed_.keys()) + list(self.range_.keys()))

    def update(self, new_domain):
        """Adding new dimensions on the ``CartesianDomain``

        :param CartesianDomain new_domain: A new ``CartesianDomain`` object to merge

        :Example:
            >>> spatial_domain = CartesianDomain({'x': [0, 1], 'y': [0, 1]})
            >>> spatial_domain.variables
            ['x', 'y']
            >>> spatial_domain_2 = CartesianDomain({'z': [3, 4], 'w': [0, 1]})
            >>> spatial_domain.update(spatial_domain_2)
            >>> spatial_domain.variables
            ['x', 'y', 'z', 'w']
        """
        self.fixed_.update(new_domain.fixed_)
        self.range_.update(new_domain.range_)

    def _sample_range(self, n, mode, bounds):
        """Rescale the samples to the correct bounds

        :param n: Number of points to sample, see Note below
            for reference.
        :type n: int
        :param mode: Mode for sampling, defaults to ``random``.
            Available modes include: random sampling, ``random``;
            latin hypercube sampling, ``latin`` or ``lh``;
            chebyshev sampling, ``chebyshev``; grid sampling ``grid``.
        :type mode: str
        :param bounds: Bounds to rescale the samples.
        :type bounds: torch.Tensor
        :return: Rescaled sample points.
        :rtype: torch.Tensor
        """
        dim = bounds.shape[0]
        if mode in ["chebyshev", "grid"] and dim != 1:
            raise RuntimeError("Something wrong in Span...")

        if mode == "random":
            pts = torch.rand(size=(n, dim))
        elif mode == "chebyshev":
            pts = chebyshev_roots(n).mul(0.5).add(0.5).reshape(-1, 1)
        elif mode == "grid":
            pts = torch.linspace(0, 1, n).reshape(-1, 1)
        # elif mode == 'lh' or mode == 'latin':
        elif mode in ["lh", "latin"]:
            pts = torch_lhs(n, dim)

        pts *= bounds[:, 1] - bounds[:, 0]
        pts += bounds[:, 0]

        return pts

    def sample(self, n, mode="random", variables="all"):
        """Sample routine.

        :param n: Number of points to sample, see Note below
            for reference.
        :type n: int
        :param mode: Mode for sampling, defaults to ``random``.
            Available modes include: random sampling, ``random``;
            latin hypercube sampling, ``latin`` or ``lh``;
            chebyshev sampling, ``chebyshev``; grid sampling ``grid``.
        :type mode: str
        :param variables: pinn variable to be sampled, defaults to ``all``.
        :type variables: str | list[str]
        :return: Returns ``LabelTensor`` of n sampled points.
        :rtype: LabelTensor

        .. note::
            The total number of points sampled in case of multiple variables
            is not ``n``, and it depends on the chosen ``mode``. If ``mode`` is
            'grid' or ``chebyshev``, the points are sampled independentely
            across the variables and the results crossed together, i.e. the
            final number of points is ``n`` to the power of the number of
            variables. If 'mode' is 'random', ``lh`` or ``latin``, the variables
            are sampled all together, and the final number of points

        .. warning::
            The extrema values of Span are always sampled only for ``grid`` mode.

        :Example:
            >>> spatial_domain = Span({'x': [0, 1], 'y': [0, 1]})
            >>> spatial_domain.sample(n=4, mode='random')
                tensor([[0.0108, 0.7643],
                        [0.4477, 0.8015],
                        [0.2063, 0.8087],
                        [0.8735, 0.6349]])
            >>> spatial_domain.sample(n=4, mode='grid')
                tensor([[0.0000, 0.0000],
                        [0.3333, 0.0000],
                        [0.6667, 0.0000],
                        [1.0000, 0.0000],
                        [0.0000, 0.3333],
                        [0.3333, 0.3333],
                        [0.6667, 0.3333],
                        [1.0000, 0.3333],
                        [0.0000, 0.6667],
                        [0.3333, 0.6667],
                        [0.6667, 0.6667],
                        [1.0000, 0.6667],
                        [0.0000, 1.0000],
                        [0.3333, 1.0000],
                        [0.6667, 1.0000],
                        [1.0000, 1.0000]])
        """

        def _1d_sampler(n, mode, variables):
            """Sample independentely the variables and cross the results"""
            tmp = []
            for variable in variables:
                if variable in self.range_.keys():
                    bound = torch.tensor([self.range_[variable]])
                    pts_variable = self._sample_range(n, mode, bound)
                    pts_variable = pts_variable.as_subclass(LabelTensor)
                    pts_variable.labels = [variable]

                    tmp.append(pts_variable)

            result = tmp[0]
            for i in tmp[1:]:
                result = result.append(i, mode="cross")

            for variable in variables:
                if variable in self.fixed_.keys():
                    value = self.fixed_[variable]
                    pts_variable = torch.tensor([[value]]).repeat(
                        result.shape[0], 1
                    )
                    pts_variable = pts_variable.as_subclass(LabelTensor)
                    pts_variable.labels = [variable]

                    result = result.append(pts_variable, mode="std")

            return result

        def _Nd_sampler(n, mode, variables):
            """Sample all the variables together

            :param n: Number of points to sample.
            :type n: int
            :param mode: Mode for sampling, defaults to ``random``.
                Available modes include: random sampling, ``random``;
                latin hypercube sampling, ``latin`` or ``lh``;
                chebyshev sampling, ``chebyshev``; grid sampling ``grid``.
            :type mode: str.
            :param variables: pinn variable to be sampled, defaults to ``all``.
            :type variables: str or list[str].
            :return: Sample points.
            :rtype: list[torch.Tensor]
            """
            pairs = [(k, v) for k, v in self.range_.items() if k in variables]
            keys, values = map(list, zip(*pairs))
            bounds = torch.tensor(values)
            result = self._sample_range(n, mode, bounds)
            result = result.as_subclass(LabelTensor)
            result.labels = keys

            for variable in variables:
                if variable in self.fixed_.keys():
                    value = self.fixed_[variable]
                    pts_variable = torch.tensor([[value]]).repeat(
                        result.shape[0], 1
                    )
                    pts_variable = pts_variable.as_subclass(LabelTensor)
                    pts_variable.labels = [variable]

                    result = result.append(pts_variable, mode="std")
            return result

        def _single_points_sample(n, variables):
            """Sample a single point in one dimension.

            :param n: Number of points to sample.
            :type n: int
            :param variables: Variables to sample from.
            :type variables: list[str]
            :return: Sample points.
            :rtype: list[torch.Tensor]
            """
            tmp = []
            for variable in variables:
                if variable in self.fixed_.keys():
                    value = self.fixed_[variable]
                    pts_variable = torch.tensor([[value]]).repeat(n, 1)
                    pts_variable = pts_variable.as_subclass(LabelTensor)
                    pts_variable.labels = [variable]
                    tmp.append(pts_variable)

            result = tmp[0]
            for i in tmp[1:]:
                result = result.append(i, mode="std")

            return result

        if variables == "all":
            variables = self.variables
        elif isinstance(variables, (list, tuple)):
            variables = sorted(variables)

        if self.fixed_ and (not self.range_):
            return _single_points_sample(n, variables)

        if mode in ["grid", "chebyshev"]:
            return _1d_sampler(n, mode, variables).extract(variables)
        elif mode in ["random", "lh", "latin"]:
            return _Nd_sampler(n, mode, variables).extract(variables)
        else:
            raise ValueError(f"mode={mode} is not valid.")

    def is_inside(self, point, check_border=False):
        """Check if a point is inside the ellipsoid.

        :param point: Point to be checked
        :type point: LabelTensor
        :param check_border: Check if the point is also on the frontier
            of the hypercube, default ``False``.
        :type check_border: bool
        :return: Returning ``True`` if the point is inside, ``False`` otherwise.
        :rtype: bool
        """
        is_inside = []

        # check fixed variables
        for variable, value in self.fixed_.items():
            if variable in point.labels:
                is_inside.append(point.extract([variable]) == value)

        # check not fixed variables
        for variable, bound in self.range_.items():
            if variable in point.labels:

                if check_border:
                    check = bound[0] <= point.extract([variable]) <= bound[1]
                else:
                    check = bound[0] < point.extract([variable]) < bound[1]

                is_inside.append(check)

        return all(is_inside)
