from .problem2d import Problem2D
import numpy as np

class ParametricProblem2D(Problem2D):

    def __init__(self, variables=None, bc=None, params_bound=None, domain_bound=None):
        
        Problem2D.__init__(self, variables=variables, bc=bc, domain_bound=domain_bound)
        self.params_domain = params_bound
