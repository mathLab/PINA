"""Module for the InverseProblem class."""

from abc import abstractmethod
import torch
from pina._src.problem.base_problem import BaseProblem


class InverseProblem(BaseProblem):
    """
    Base class for all inverse problems, extending the standard problem
    definition with unknown parameters to be determined through training.

    An inverse problem is defined by a set of unknown parameters that need to be
    estimated from observed data.

    This class is not meant to be instantiated directly.

    :Example:

        >>> import torch
        >>> from pina.problem import InverseProblem
        >>> from pina.domain import CartesianDomain
        >>> class MyInverseProblem(InverseProblem):
        ...     @property
        ...     def unknown_parameter_domain(self):
        ...         return CartesianDomain({"k": [0.1, 5.0]})
        ...     @property
        ...     def conditions(self): return {}
        >>> problem = MyInverseProblem()
        >>> problem.unknown_variables
        ['k']
    """

    def __init__(self):
        """
        Initialization of the :class:`InverseProblem` class.
        """
        super().__init__()

        # Set the unknown parameters as trainable parameters
        self.unknown_parameters = {}
        for var in self.unknown_variables:
            low, high = self.unknown_parameter_domain._range[var]
            tensor_var = low + (high - low) * torch.rand(1)
            self.unknown_parameters[var] = torch.nn.Parameter(tensor_var)

    @property
    @abstractmethod
    def unknown_parameter_domain(self):
        """
        The domain of the unknown parameters of the problem.
        """

    @property
    def unknown_variables(self):
        """
        The unknown variables of the problem.

        :return: The unknown variables of the problem.
        :rtype: list[str]
        """
        return self.unknown_parameter_domain.variables

    @property
    def unknown_parameters(self):
        """
        The unknown parameters of the problem.

        :return: The unknown parameters of the problem.
        :rtype: torch.nn.Parameter
        """
        return self.__unknown_parameters

    @unknown_parameters.setter
    def unknown_parameters(self, value):
        """
        Set the unknown parameters of the problem.

        :param torch.nn.Parameter value: The unknown parameters of the problem.
        """
        self.__unknown_parameters = value
