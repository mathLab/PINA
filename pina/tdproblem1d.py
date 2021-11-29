import numpy as np
from .problem1d import Problem1D
from .segment import Segment

class TimeDepProblem1D(Problem1D):

    def __init__(self, variables=None, bc=None, initial=None, tend=None, domain_bound=None):
        self.variables = variables
        self._spatial_dimensions = 2
        self.tend = tend
        self.tstart = 0
        if domain_bound is None:
            bound_pts = [bc[0] for bc in self.boundary_conditions]
            domain_bound = np.array([
                [min(bound_pts), max(bound_pts)],
                [self.tstart,    self.tend     ]
            ])

        self.domain_bound = np.array([[-1, 1],[0, 1]])#domain_bound
        print(domain_bound)
        self.boundary_conditions = (
            (Segment((bc[0][0], self.tstart), (bc[1][0], self.tstart)), initial),
            (Segment((bc[0][0], self.tstart), (bc[0][0], self.tend)), bc[0][1]),
            (Segment((bc[1][0], self.tstart), (bc[1][0], self.tend)), bc[1][1])
        )


