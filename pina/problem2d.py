from .problem import Problem


class Problem2D(Problem):

    spatial_dimensions = 2

    @property
    def boundary_condition(self):
        return self._boundary_condition

    @boundary_condition.setter
    def boundary_condition(self, bc):
        self._boundary_condition = bc
        

