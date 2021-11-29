
from .problem import Problem
import numpy as np

class Problem1D(Problem):

    def __init__(self, variables=None, bc=None):
        self._spatial_dimensions = 1
        self.variables = variables
        print(bc)
        self.bc = bc
