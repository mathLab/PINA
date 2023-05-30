import numpy as np
import torch

from pina import Span, Condition
from pina.problem import SpatialProblem, ParametricProblem
from pina.operators import grad, nabla

# ===================================================== #
#                                                       #
#  This script implements the two dimensional           #
#  Parametric Elliptic Optimal Control problem.         #
#  The ParametricEllipticOptimalControl class is        #
#  inherited from TimeDependentProblem, SpatialProblem  #
#  and we denote:                                       #
#           u --> field variable                        #
#           p --> field variable                        #
#           y --> field variable                        #
#           x1, x2 --> spatial variables                #
#           mu, alpha --> problem parameters            #
#                                                       #
#  More info in https://arxiv.org/pdf/2110.13530.pdf    #
#  Section 4.2 of the article                           #
# ===================================================== #


class ParametricEllipticOptimalControl(SpatialProblem, ParametricProblem):

    # setting spatial variables ranges
    xmin, xmax, ymin, ymax = -1, 1, -1, 1
    x_range = [xmin, xmax]
    y_range = [ymin, ymax]
    # setting parameters range
    amin, amax = 0.0001, 1
    mumin, mumax = 0.5, 3
    mu_range = [mumin, mumax]
    a_range = [amin, amax]
    # setting field variables
    output_variables = ['u', 'p', 'y']
    # setting spatial and parameter domain
    spatial_domain = Span({'x1': x_range, 'x2': y_range})
    parameter_domain = Span({'mu': mu_range, 'alpha': a_range})

    # equation terms as in https://arxiv.org/pdf/2110.13530.pdf
    def term1(input_, output_):
        laplace_p = nabla(output_, input_, components=['p'], d=['x1', 'x2'])
        return output_.extract(['y']) - input_.extract(['mu']) - laplace_p

    def term2(input_, output_):
        laplace_y = nabla(output_, input_, components=['y'], d=['x1', 'x2'])
        return - laplace_y - output_.extract(['u_param'])

    def state_dirichlet(input_, output_):
        y_exp = 0.0
        return output_.extract(['y']) - y_exp

    def adj_dirichlet(input_, output_):
        p_exp = 0.0
        return output_.extract(['p']) - p_exp

    # setting problem condition formulation
    conditions = {
        'gamma1': Condition(
            location=Span({'x1': x_range, 'x2':  1, 'mu': mu_range, 'alpha': a_range}),
            function=[state_dirichlet, adj_dirichlet]),
        'gamma2': Condition(
            location=Span({'x1': x_range, 'x2': -1, 'mu': mu_range, 'alpha': a_range}),
            function=[state_dirichlet, adj_dirichlet]),
        'gamma3': Condition(
            location=Span({'x1':  1, 'x2': y_range, 'mu': mu_range, 'alpha': a_range}),
            function=[state_dirichlet, adj_dirichlet]),
        'gamma4': Condition(
            location=Span({'x1': -1, 'x2': y_range, 'mu': mu_range, 'alpha': a_range}),
            function=[state_dirichlet, adj_dirichlet]),
        'D': Condition(
            location=Span({'x1': x_range, 'x2': y_range,
                  'mu': mu_range, 'alpha': a_range}),
            function=[term1, term2]),
    }