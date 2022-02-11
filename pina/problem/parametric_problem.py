from abc import abstractmethod

from .abstract_problem import AbstractProblem


class ParametricProblem(AbstractProblem):

    @property
    @abstractmethod
    def parameters(self):
        pass
