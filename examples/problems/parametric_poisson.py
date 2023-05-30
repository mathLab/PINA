import torch

from pina.problem import SpatialProblem, ParametricProblem
from pina.operators import nabla
from pina import Span, Condition

# ===================================================== #
#                                                       #
#  This script implements the two dimensional           #
#  Parametric Poisson problem. The ParametricPoisson    #
#  class is defined inheriting from SpatialProblem and  #
#  ParametricProblem. We  denote:                       #
#           u --> field variable                        #
#           x,y --> spatial variables                   #
#           mu1, mu2 --> parameter variables            #
#                                                       #
# ===================================================== #

class ParametricPoisson(SpatialProblem, ParametricProblem):

    # assign output/ spatial and parameter variables
    output_variables = ['u']
    spatial_domain = Span({'x': [-1, 1], 'y': [-1, 1]})
    parameter_domain = Span({'mu1': [-1, 1], 'mu2': [-1, 1]})

    # define the laplace equation
    def laplace_equation(input_, output_):
        force_term = torch.exp(
                - 2*(input_.extract(['x']) - input_.extract(['mu1']))**2
                - 2*(input_.extract(['y']) - input_.extract(['mu2']))**2)
        return nabla(output_.extract(['u']), input_) - force_term

    # define nill dirichlet boundary conditions
    def nil_dirichlet(input_, output_):
        value = 0.0
        return output_.extract(['u']) - value

    # problem condition statement
    conditions = {
        'gamma1': Condition(
            location=Span({'x': [-1, 1], 'y': 1, 'mu1': [-1, 1], 'mu2': [-1, 1]}),
            function=nil_dirichlet),
        'gamma2': Condition(
            location=Span({'x': [-1, 1], 'y': -1, 'mu1': [-1, 1], 'mu2': [-1, 1]}),
            function=nil_dirichlet),
        'gamma3': Condition(
            location=Span({'x': 1, 'y': [-1, 1], 'mu1': [-1, 1], 'mu2': [-1, 1]}),
            function=nil_dirichlet),
        'gamma4': Condition(
            location=Span({'x': -1, 'y': [-1, 1], 'mu1': [-1, 1], 'mu2': [-1, 1]}),
            function=nil_dirichlet),
        'D': Condition(
            location=Span({'x': [-1, 1], 'y': [-1, 1], 'mu1': [-1, 1], 'mu2': [-1, 1]}),
            function=laplace_equation),
    }
