"""Module for CartesianDomain class."""

import torch

from .domain_interface import DomainInterface
from ..label_tensor import LabelTensor
from ..utils import torch_lhs, chebyshev_roots


class CartesianDomain(DomainInterface):
    """
    Implementation of the hypercube domain.
    """

    def __init__(self, cartesian_dict):
        """
        Initialize the :class:`~pina.domain.CartesianDomain` class.

        :param dict cartesian_dict: A dictionary where the keys are the 
            variable names and the values are the domain extrema. The domain 
            extrema can be either a list with two elements or a single number. 
            If the domain extrema is a single number, the variable is fixed to 
            that value.

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
    def sample_modes(self):
        """
        List of available sampling modes.

        :return: List of available sampling modes.
        :rtype: list[str]
        """
        return ["random", "grid", "lh", "chebyshev", "latin"]

    @property
    def variables(self):
        """
        List of variables of the domain.

        :return: List of variables of the domain.
        :rtype: list[str]
        """
        return sorted(list(self.fixed_.keys()) + list(self.range_.keys()))

    def update(self, new_domain):
        """
        Add new dimensions to an existing :class:`~pina.domain.CartesianDomain` 
        object.

        :param :class:`~pina.domain.CartesianDomain` new_domain: New domain to 
            be added to an existing :class:`~pina.domain.CartesianDomain` object.

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
        """
        Rescale the samples to fit within the specified bounds.

        :param int n: Number of points to sample.
        :param str mode: Sampling method. Default is ``random``.
        :param torch.Tensor bounds: Bounds of the domain.
        :raises RuntimeError: Wrong bounds initialization.
        :raises ValueError: Invalid sampling mode.
        :return: Rescaled sample points.
        :rtype: torch.Tensor
        """
        dim = bounds.shape[0]
        if mode in ["chebyshev", "grid"] and dim != 1:
            raise RuntimeError("Wrong bounds initialization")

        if mode == "random":
            pts = torch.rand(size=(n, dim))
        elif mode == "chebyshev":
            pts = chebyshev_roots(n).mul(0.5).add(0.5).reshape(-1, 1)
        elif mode == "grid":
            pts = torch.linspace(0, 1, n).reshape(-1, 1)
        elif mode in ["lh", "latin"]:
            pts = torch_lhs(n, dim)
        else:
            raise ValueError("Invalid mode")

        return pts * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]

    def sample(self, n, mode="random", variables="all"):
        """
        Sampling routine.

        :param int n: Number of points to sample, see Note below for reference.
        :param str mode: Sampling method. Default is ``random``.
            Available modes: random sampling, ``random``; 
            latin hypercube sampling, ``latin`` or ``lh``; 
            chebyshev sampling, ``chebyshev``; grid sampling ``grid``.
        :param variables: variables to be sampled. Default is ``all``.
        :type variables: str | list[str]
        :return: Sampled points.
        :rtype: LabelTensor

        .. note::
            When multiple variables are involved, the total number of sampled 
            points may differ from ``n``, depending on the chosen ``mode``. 
            If ``mode`` is ``grid`` or ``chebyshev``, points are sampled 
            independently for each variable and then combined, resulting in a 
            total number of points equal to ``n`` raised to the power of the 
            number of variables. If 'mode' is 'random', ``lh`` or ``latin``, 
            all variables are sampled together, and the total number of points 
            remains ``n``.

        .. warning::
            The extrema of CartesianDomain are only sampled when using the 
            ``grid`` mode.

        :Example:
            >>> spatial_domain = CartesianDomain({'x': [0, 1], 'y': [0, 1]})
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
            """
            Sample each variable independently.

            :param int n: Number of points to sample.
            :param str mode: Sampling method.
            :param variables: variables to be sampled.
            :type variables: str | list[str]
            :return: Sampled points.
            :rtype: list[LabelTensor]
            """
            tmp = []
            for variable in variables:
                if variable in self.range_:
                    bound = torch.tensor([self.range_[variable]])
                    pts_variable = self._sample_range(n, mode, bound)
                    pts_variable = pts_variable.as_subclass(LabelTensor)
                    pts_variable.labels = [variable]

                    tmp.append(pts_variable)
            if tmp:
                result = tmp[0]
                for i in tmp[1:]:
                    result = result.append(i, mode="cross")

            for variable in variables:
                if variable in self.fixed_:
                    value = self.fixed_[variable]
                    pts_variable = torch.tensor([[value]]).repeat(
                        result.shape[0], 1
                    )
                    pts_variable = pts_variable.as_subclass(LabelTensor)
                    pts_variable.labels = [variable]

                    result = result.append(pts_variable, mode="std")

            return result

        def _Nd_sampler(n, mode, variables):
            """
            Sample all variables together.

            :param int n: Number of points to sample.
            :param str mode: Sampling method.
            :param variables: variables to be sampled.
            :type variables: str | list[str]
            :return: Sampled points.
            :rtype: list[LabelTensor]
            """
            pairs = [(k, v) for k, v in self.range_.items() if k in variables]
            keys, values = map(list, zip(*pairs))
            bounds = torch.tensor(values)
            result = self._sample_range(n, mode, bounds)
            result = result.as_subclass(LabelTensor)
            result.labels = keys

            for variable in variables:
                if variable in self.fixed_:
                    value = self.fixed_[variable]
                    pts_variable = torch.tensor([[value]]).repeat(
                        result.shape[0], 1
                    )
                    pts_variable = pts_variable.as_subclass(LabelTensor)
                    pts_variable.labels = [variable]

                    result = result.append(pts_variable, mode="std")
            return result

        def _single_points_sample(n, variables):
            """
            Sample a single point in one dimension.

            :param int n: Number of points to sample.
            :param variables: variables to be sampled.
            :type variables: str | list[str]
            :return: Sampled points.
            :rtype: list[torch.Tensor]
            """
            tmp = []
            for variable in variables:
                if variable in self.fixed_:
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
        if isinstance(variables, str) and variables in self.fixed_:
            return _single_points_sample(n, variables)

        if mode in ["grid", "chebyshev"]:
            return _1d_sampler(n, mode, variables).extract(variables)
        if mode in ["random", "lh", "latin"]:
            return _Nd_sampler(n, mode, variables).extract(variables)
        raise ValueError(f"mode={mode} is not valid.")

    def is_inside(self, point, check_border=False):
        """
        Check if a point is inside the hypercube.

        :param LabelTensor point: Point to be checked.
        :param bool check_border: Determines whether to check if the point lies 
            on the boundary of the hypercube. Default is ``False``.
        :return: ``True`` if the point is inside the domain, ``False`` otherwise.
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
