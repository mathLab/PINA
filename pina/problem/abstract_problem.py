""" Module for AbstractProblem class """

from abc import ABCMeta, abstractmethod
from ..utils import check_consistency
from ..domain import DomainInterface, CartesianDomain
from ..condition.domain_equation_condition import DomainEquationCondition
from ..condition import InputPointsEquationCondition
from copy import deepcopy
from .. import LabelTensor
from ..utils import merge_tensors


class AbstractProblem(metaclass=ABCMeta):
    """
    The abstract `AbstractProblem` class. All the class defining a PINA Problem
    should be inherited from this class.

    In the definition of a PINA problem, the fundamental elements are:
    the output variables, the condition(s), and the domain(s) where the
    conditions are applied.
    """

    def __init__(self):

        self._discretised_domains = {}
        # create collector to manage problem data

        # create hook conditions <-> problems
        for condition_name in self.conditions:
            self.conditions[condition_name].problem = self

        self._batching_dimension = 0

        # Store in domains dict all the domains object directly passed to
        # ConditionInterface. Done for back compatibility with PINA <0.2
        if not hasattr(self, "domains"):
            self.domains = {}
        for cond_name, cond in self.conditions.items():
            if isinstance(cond, (DomainEquationCondition,
                                 InputPointsEquationCondition)):
                if isinstance(cond.domain, DomainInterface):
                    self.domains[cond_name] = cond.domain
                    cond.domain = cond_name

    # @property
    # def collector(self):
    #     return self._collector

    @property
    def batching_dimension(self):
        return self._batching_dimension

    @batching_dimension.setter
    def batching_dimension(self, value):
        self._batching_dimension = value

    @property
    def discretised_domains(self):
        return self._discretised_domains

    # TODO this should be erase when dataloading will interface collector,
    # kept only for back compatibility
    @property
    def input_pts(self):
        to_return = {}
        for cond_name, cond in self.conditions.items():
            if hasattr(cond, "input_points"):
                to_return[cond_name] = cond.input_points
            elif hasattr(cond, "domain"):
                to_return[cond_name] = self._discretised_domains[cond.domain]
        return to_return

    def __deepcopy__(self, memo):
        """
        Implements deepcopy for the
        :class:`~pina.problem.abstract_problem.AbstractProblem` class.

        :param dict memo: Memory dictionary, to avoid excess copy
        :return: The deep copy of the
            :class:`~pina.problem.abstract_problem.AbstractProblem` class
        :rtype: AbstractProblem
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    @property
    def are_all_domains_discretised(self):
        """
        Check if all the domains are discretised.

        :return: True if all the domains are discretised, False otherwise
        :rtype: bool
        """
        return all(
            [
                domain in self.discretised_domains
                for domain in self.domains.keys()
            ]
        )

    @property
    def input_variables(self):
        """
        The input variables of the AbstractProblem, whose type depends on the
        type of domain (spatial, temporal, and parameter).

        :return: the input variables of self
        :rtype: list

        """
        variables = []

        if hasattr(self, "spatial_variables"):
            variables += self.spatial_variables
        if hasattr(self, "temporal_variable"):
            variables += self.temporal_variable
        if hasattr(self, "parameters"):
            variables += self.parameters

        return variables

    @input_variables.setter
    def input_variables(self, variables):
        raise RuntimeError

    @property
    @abstractmethod
    def output_variables(self):
        """
        The output variables of the problem.
        """
        pass

    @property
    @abstractmethod
    def conditions(self):
        """
        The conditions of the problem.
        """
        return self.conditions

    def discretise_domain(self,
                          n=None,
                          mode="random",
                          domains="all",
                          sample_rules=None):
        """
        Generate a set of points to span the `Location` of all the conditions of
        the problem.

        :param n: Number of points to sample, see Note below
            for reference.
        :type n: int
        :param mode: Mode for sampling, defaults to ``random``.
            Available modes include: random sampling, ``random``;
            latin hypercube sampling, ``latin`` or ``lh``;
            chebyshev sampling, ``chebyshev``; grid sampling ``grid``.
        :param variables: variable(s) to sample, defaults to 'all'.
        :type variables: str | list[str]
        :param domains: problem's domain from where to sample, defaults to 'all'.
        :type domains: str | list[str]

        :Example:
            >>> pinn.discretise_domain(n=10, mode='grid')
            >>> pinn.discretise_domain(n=10, mode='grid', domain=['bound1'])
            >>> pinn.discretise_domain(n=10, mode='grid', variables=['x'])

        .. warning::
            ``random`` is currently the only implemented ``mode`` for all geometries, i.e.
            ``EllipsoidDomain``, ``CartesianDomain``, ``SimplexDomain`` and the geometries
            compositions ``Union``, ``Difference``, ``Exclusion``, ``Intersection``. The
            modes ``latin`` or ``lh``,  ``chebyshev``, ``grid`` are only implemented for
            ``CartesianDomain``.
        """

        # check consistecy n, mode, variables, locations
        if sample_rules is not None:
            check_consistency(sample_rules, dict)
        if mode is not None:
            check_consistency(mode, str)
        check_consistency(domains, (list, str))

        # check correct location
        if domains == "all":
            domains = self.domains.keys()
        elif not isinstance(domains, (list)):
            domains = [domains]
        if n is not None and sample_rules is None:
            self._apply_default_discretization(n, mode, domains)
        if n is None and sample_rules is not None:
            self._apply_custom_discretization(sample_rules, domains)
        elif n is not None and sample_rules is not None:
            raise RuntimeError(
                "You can't specify both n and sample_rules at the same time."
            )
        elif n is None and sample_rules is None:
            raise RuntimeError(
                "You have to specify either n or sample_rules."
            )

    def _apply_default_discretization(self, n, mode, domains):
        for domain in domains:
            self.discretised_domains[domain] = (
                self.domains[domain].sample(n, mode).sort_labels()
            )

    def _apply_custom_discretization(self, sample_rules, domains):
        if sorted(list(sample_rules.keys())) != sorted(self.input_variables):
            raise RuntimeError(
                "The keys of the sample_rules dictionary must be the same as "
                "the input variables."
            )
        for domain in domains:
            if not isinstance(self.domains[domain], CartesianDomain):
                raise RuntimeError(
                    "Custom discretisation can be applied only on Cartesian "
                    "domains")
            discretised_tensor = []
            for var, rules in sample_rules.items():
                n, mode = rules['n'], rules['mode']
                points = self.domains[domain].sample(n, mode, var)
                discretised_tensor.append(points)

            self.discretised_domains[domain] = merge_tensors(
                discretised_tensor).sort_labels()

    def add_points(self, new_points_dict):
        """
        Add input points to a sampled condition
        :param new_points_dict: Dictionary of input points (condition_name:
        LabelTensor)
        :raises RuntimeError: if at least one condition is not already sampled
        """
        for k, v in new_points_dict.items():
            self.discretised_domains[k] = LabelTensor.vstack(
                [self.discretised_domains[k], v])
