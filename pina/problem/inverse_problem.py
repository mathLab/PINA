"""Module for the ParametricProblem class"""

from abc import abstractmethod
import torch
from .abstract_problem import AbstractProblem


class InverseProblem(AbstractProblem):
    """
    The class for the definition of inverse problems, i.e., problems
    with unknown parameters that have to be learned during the training process
    from given data.

    Here's an example of a spatial inverse ODE problem, i.e., a spatial
    ODE problem with an unknown parameter `alpha` as coefficient of the
    derivative term.

    :Example:
        TODO
    """

    def __init__(self):
        super().__init__()
        # storing unknown_parameters for optimization
        self.unknown_parameters = {}
        for var in self.unknown_variables:
            range_var = self.unknown_parameter_domain.range_[var]
            tensor_var = (
                torch.rand(1, requires_grad=True) * range_var[1] + range_var[0]
            )
            self.unknown_parameters[var] = torch.nn.Parameter(tensor_var)

    @abstractmethod
    def unknown_parameter_domain(self):
        """
        The parameters' domain of the problem.
        """

    @property
    def unknown_variables(self):
        """
        The parameters of the problem.
        """
        return self.unknown_parameter_domain.variables

    @property
    def unknown_parameters(self):
        """
        The parameters of the problem.
        """
        return self.__unknown_parameters

    @unknown_parameters.setter
    def unknown_parameters(self, value):
        self.__unknown_parameters = value
