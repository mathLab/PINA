"""Module for the Problem Interface."""

from abc import ABCMeta, abstractmethod


class ProblemInterface(metaclass=ABCMeta):
    """
    Abstract interface for all problems.
    """

    @abstractmethod
    def __deepcopy__(self, memo):
        """
        Create a deep copy of the problem instance.

        :param dict memo: The memorization dictionary used by the deepcopy
            function.
        :return: A deep copy of the problem instance.
        :rtype: ProblemInterface
        """

    @abstractmethod
    def discretise_domain(
        self, n=None, mode="random", domains=None, sample_rules=None
    ):
        """
        Discretise the problem's domains by sampling a specified number of
        points according to the selected sampling mode.

        :param int n: The number of points to sample. This is ignored if
            ``sample_rules`` is provided. Default is ``None``.
        :param str mode: The sampling method. Available modes include:
            ``"random"`` for random sampling, ``"latin"`` or ``"lh"`` for latin
            hypercube sampling, ``"chebyshev"`` for Chebyshev sampling, and
            ``"grid"`` for grid sampling. Default is ``"random"``.
        :param domains: The domains from which to sample. If ``None``, all
            domains are considered for sampling. Default is ``None``.
        :type domains: str | list[str]
        :param dict sample_rules: The dictionary specifying custom sampling
            rules for each input variable. When provided, it overrides the
            global ``n`` and ``mode`` arguments. Each key in the dictionary must
            match one of the variables defined in :meth:`input_variables`, and
            each value must be a dictionary containing two keys: ``n`` for the
            number of points to sample for that variable, and ``mode`` for the
            sampling method to use. If ``None``, the global ``n`` and ``mode``
            parameters are used for all variables. Default is ``None``.

        .. warning::
            ``"random"`` is the only supported ``mode`` across all geometries:
            :class:`~pina.domain.cartesian_domain.CartesianDomain`,
            :class:`~pina.domain.ellipsoid_domain.EllipsoidDomain`, and
            :class:`~pina.domain.simplex_domain.SimplexDomain`.
            Sampling modes such as ``"latin"``, ``"chebyshev"``, and ``"grid"``
            are only implemented for
            :class:`~pina.domain.cartesian_domain.CartesianDomain`.
            When custom discretisation is specified via ``sample_rules``, the
            domain to be discretised must be an instance of
            :class:`~pina.domain.cartesian_domain.CartesianDomain`.

        :Example:
            >>> problem.discretise_domain(n=10, mode="random")
            >>> problem.discretise_domain(n=10, mode="lh", domains=["boundary"])
            >>> problem.discretise_domain(
            ...     sample_rules={
            ...         'x': {'n': 10, 'mode': 'grid'},
            ...         'y': {'n': 100, 'mode': 'grid'}
            ...     },
            ... )
        """

    @abstractmethod
    def add_points(self, new_points_dict):
        """
        Append additional points to an already discretised domain.

        :param dict new_points_dict: The dictionary mapping each domain to the
            corresponding set of new points to be added. Each key in the
            dictionary must match one of the domains defined in :attr:`domains`,
            and each value must be a :class:`~pina.tensor.LabelTensor`
            containing the new points to be added to that domain. The labels of
            the points to be added must correspond to those of the domain to
            which they are being added.

        :Example:
            >>> additional_points = {
            ...     "boundary": LabelTensor(torch.rand(5, 2), labels=["x", "y"])
            ... }
            >>> problem.add_points(additional_points)
        """

    @abstractmethod
    def move_discretisation_into_conditions(self):
        """
        Move the sampled points from the discretised domains into their
        corresponding conditions. This ensures that the conditions are evaluated
        on the correct set of points after discretisation.
        """

    @property
    @abstractmethod
    def input_variables(self):
        """
        The input variables of the problem.

        :return: The input variables of the problem.
        :rtype: list[str]
        """

    @property
    @abstractmethod
    def output_variables(self):
        """
        The output variables of the problem.

        :return: The output variables of the problem.
        :rtype: list[str]
        """

    @property
    @abstractmethod
    def conditions(self):
        """
        The conditions associated with the problem.

        :return: The conditions associated with the problem.
        :rtype: dict
        """

    @property
    @abstractmethod
    def discretised_domains(self):
        """
        The dictionary containing the discretised domains of the problem. Each
        key corresponds to a domain defined in :attr:`domains`, and each value
        is a :class:`~pina.tensor.LabelTensor` containing the sampled points for
        that domain.

        :return: The discretised domains.
        :rtype: dict
        """

    @property
    @abstractmethod
    def are_all_domains_discretised(self):
        """
        Whether all domains of the problem have been discretised.

        :return: ``True`` if all domains are discretised, ``False`` otherwise.
        :rtype: bool
        """
