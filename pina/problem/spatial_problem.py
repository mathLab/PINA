from abc import abstractmethod

from .abstract_problem import AbstractProblem


class SpatialProblem(AbstractProblem):

    @abstractmethod
    def spatial_variables(self):
        pass
