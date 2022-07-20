from abc import abstractmethod

from .abstract_problem import AbstractProblem


class SpatialProblem(AbstractProblem):

    @abstractmethod
    def spatial_domain(self):
        pass

    @property
    def spatial_variables(self):
        return self.spatial_domain.variables
