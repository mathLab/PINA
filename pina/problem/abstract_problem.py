""" Module for AbstractProblem class """

from abc import ABCMeta, abstractmethod
from ..utils import check_consistency
from ..domain import DomainInterface
from ..condition.domain_equation_condition import DomainEquationCondition
from ..collector import Collector
from copy import deepcopy

class AbstractProblem(metaclass=ABCMeta):
    """
    The abstract `AbstractProblem` class. All the class defining a PINA Problem
    should be inheritied from this class.

    In the definition of a PINA problem, the fundamental elements are:
    the output variables, the condition(s), and the domain(s) where the
    conditions are applied.
    """

    def __init__(self):

        # create collector to manage problem data
        self._collector = Collector(self)

        # create hook conditions <-> problems
        for condition_name in self.conditions:
            self.conditions[condition_name].problem = self

        # store in collector all the available fixed points
        # note that some points could not be stored at this stage (e.g. when
        # sampling locations). To check that all data points are ready for
        # training all type self.collector.full, which returns true if all
        # points are ready.
        self.collector.store_fixed_data()

    @property
    def collector(self):
        return self._collector

    # TODO this should be erase when dataloading will interface collector,
    # kept only for back compatibility
    @property
    def input_pts(self):
        to_return = {}
        for k, v in self.collector.data_collections.items():
            if 'input_points' in v.keys():
                to_return[k] = v['input_points']
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
        if hasattr(self, "unknown_parameters"):
            variables += self.parameters
        if hasattr(self, "custom_variables"):
            variables += self.custom_variables

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
        return self._conditions

    def discretise_domain(
        self, n, mode="random", variables="all", locations="all"
    ):
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
        :param variables: problem's variables to be sampled, defaults to 'all'.
        :type variables: str | list[str]
        :param locations: problem's locations from where to sample, defaults to 'all'.
        :type locations: str

        :Example:
            >>> pinn.discretise_domain(n=10, mode='grid')
            >>> pinn.discretise_domain(n=10, mode='grid', location=['bound1'])
            >>> pinn.discretise_domain(n=10, mode='grid', variables=['x'])

        .. warning::
            ``random`` is currently the only implemented ``mode`` for all geometries, i.e.
            ``EllipsoidDomain``, ``CartesianDomain``, ``SimplexDomain`` and the geometries
            compositions ``Union``, ``Difference``, ``Exclusion``, ``Intersection``. The
            modes ``latin`` or ``lh``,  ``chebyshev``, ``grid`` are only implemented for
            ``CartesianDomain``.
        """

        # check consistecy n, mode, variables, locations
        check_consistency(n, int)
        check_consistency(mode, str)
        check_consistency(variables, str)
        check_consistency(locations, str)

        # check correct sampling mode
        if mode not in DomainInterface.available_sampling_modes:
            raise TypeError(f"mode {mode} not valid.")

        # check correct variables
        if variables == "all":
            variables = self.input_variables
        for variable in variables:
            if variable not in self.input_variables:
                TypeError(
                    f"Wrong variables for sampling. Variables ",
                    f"should be in {self.input_variables}.",
                )

        # check correct location
        if locations == "all":
            locations = [name for name in self.conditions.keys()
                         if isinstance(self.conditions[name],
                                       DomainEquationCondition)]
        else:
            if not isinstance(locations, (list)):
                locations = [locations]
        for loc in locations:
            if not isinstance(self.conditions[loc], DomainEquationCondition):
                raise TypeError(
                    f"Wrong locations passed, locations for sampling "
                    f"should be in {[loc for loc in locations if isinstance(self.conditions[loc], DomainEquationCondition)]}.",
                )

        # store data
        self.collector.store_sample_domains(n, mode, variables, locations)

    def add_points(self, new_points_dict):
        self.collector.add_points(new_points_dict)
