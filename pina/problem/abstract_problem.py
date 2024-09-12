""" Module for AbstractProblem class """

from abc import ABCMeta, abstractmethod
from ..utils import merge_tensors, check_consistency
from copy import deepcopy
import torch

from .. import LabelTensor


class AbstractProblem(metaclass=ABCMeta):
    """
    The abstract `AbstractProblem` class. All the class defining a PINA Problem
    should be inheritied from this class.

    In the definition of a PINA problem, the fundamental elements are:
    the output variables, the condition(s), and the domain(s) where the
    conditions are applied.
    """

    def __init__(self):

        self._discretized_domains = {}

        for name, domain in self.domains.items():
            if isinstance(domain, (torch.Tensor, LabelTensor)):
                self._discretized_domains[name] = domain

        for condition_name in self.conditions:
            self.conditions[condition_name].set_problem(self)

        # # variable storing all points
        self.input_pts = {}

        # # varible to check if sampling is done. If no location
        # # element is presented in Condition this variable is set to true
        # self._have_sampled_points = {}
        for condition_name in self.conditions:
            self._discretized_domains[condition_name] = False

        # # put in self.input_pts all the points that we don't need to sample
        self._span_condition_points()

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
    def domains(self):
        """
        The domain(s) where the conditions of the AbstractProblem are valid.
        If more than one domain type is passed, a list of Location is
        retured.

        :return: the domain(s) of ``self``
        :rtype: list[Location]
        """
        pass

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

        

    def _span_condition_points(self):
        """
        Simple function to get the condition points
        """
        for condition_name in self.conditions:
            condition = self.conditions[condition_name]
            if hasattr(condition, "input_points"):
                samples = condition.input_points
                self.input_pts[condition_name] = samples
                self._discretized_domains[condition_name] = True
            if hasattr(self, "unknown_parameter_domain"):
                # initialize the unknown parameters of the inverse problem given
                # the domain the user gives
                self.unknown_parameters = {}
                for i, var in enumerate(self.unknown_variables):
                    range_var = self.unknown_parameter_domain.range_[var]
                    tensor_var = (
                        torch.rand(1, requires_grad=True) * range_var[1]
                        + range_var[0]
                    )
                    self.unknown_parameters[var] = torch.nn.Parameter(
                        tensor_var
                    )

    def discretise_domain(
        self, n, mode="random", variables="all", domains="all"
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

        # check consistecy n
        check_consistency(n, int)

        # check consistency mode
        check_consistency(mode, str)
        if mode not in ["random", "grid", "lh", "chebyshev", "latin"]:
            raise TypeError(f"mode {mode} not valid.")

        # check consistency variables
        if variables == "all":
            variables = self.input_variables
        else:
            check_consistency(variables, str)

        if sorted(variables) != sorted(self.input_variables):
            TypeError(
                f"Wrong variables for sampling. Variables ",
                f"should be in {self.input_variables}.",
            )

        # check consistency location
        if domains == "all":
            domains = [condition for condition in self.conditions]
        else:
            check_consistency(domains, str)
        print(domains)
        if sorted(domains) != sorted(self.conditions):
            TypeError(
                f"Wrong locations for sampling. Location ",
                f"should be in {self.conditions}.",
            )

        # sampling
        for d in domains:
            condition = self.conditions[d]

            # we try to check if we have already sampled
            try:
                already_sampled = [self.input_pts[d]]
            # if we have not sampled, a key error is thrown
            except KeyError:
                already_sampled = []

            # if we have already sampled fully the condition
            # but we want to sample again we set already_sampled
            # to an empty list since we need to sample again, and
            # self._have_sampled_points to False.
            if self._discretized_domains[d]:
                already_sampled = []
                self._discretized_domains[d] = False
            print(condition.domain)
            print(d)
            # build samples
            samples = [
                self.domains[d].sample(n=n, mode=mode, variables=variables)
            ] + already_sampled
            pts = merge_tensors(samples)
            self.input_pts[d] = pts

            # the condition is sampled if input_pts contains all labels
            if sorted(self.input_pts[d].labels) == sorted(
                self.input_variables
            ):
                self._have_sampled_points[d] = True

    def add_points(self, new_points):
        """
        Adding points to the already sampled points.

        :param dict new_points: a dictionary with key the location to add the points
            and values the torch.Tensor points.
        """

        if sorted(new_points.keys()) != sorted(self.conditions):
            TypeError(
                f"Wrong locations for new points. Location ",
                f"should be in {self.conditions}.",
            )

        for location in new_points.keys():
            # extract old and new points
            old_pts = self.input_pts[location]
            new_pts = new_points[location]

            # if they don't have the same variables error
            if sorted(old_pts.labels) != sorted(new_pts.labels):
                TypeError(
                    f"Not matching variables for old and new points "
                    f"in condition {location}."
                )
            if old_pts.labels != new_pts.labels:
                new_pts = torch.hstack(
                    [new_pts.extract([i]) for i in old_pts.labels]
                )
                new_pts.labels = old_pts.labels

            # merging
            merged_pts = torch.vstack([old_pts, new_points[location]])
            merged_pts.labels = old_pts.labels
            self.input_pts[location] = merged_pts