from abc import abstractmethod

from .abstract_problem import AbstractProblem


class TimeDependentProblem(AbstractProblem):

    @abstractmethod
    def temporal_domain(self):
        pass

    @property
    def temporal_variables(self):
        return self.temporal_domain.variables
