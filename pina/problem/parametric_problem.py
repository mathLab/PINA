from abc import abstractmethod

from .abstract_problem import AbstractProblem


class ParametricProblem(AbstractProblem):

    @abstractmethod
    def parameter_domain(self):
        pass

    @property
    def parameters(self):
        return self.parameter_domain.variables
