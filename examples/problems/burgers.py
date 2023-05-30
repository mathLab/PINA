import torch

from pina.problem import TimeDependentProblem, SpatialProblem
from pina.operators import grad
from pina import Condition
from pina.span import Span

# ===================================================== #
#                                                       #
#  This script implements the one dimensional Burger    #
#  problem. The Burgers1D class is defined inheriting   #
#  from TimeDependentProblem, SpatialProblem and we     #
#  denote:                                              #
#           u --> field variable                        #
#           x --> spatial variable                      #
#           t --> temporal variable                     #
#                                                       #
# ===================================================== #

class Burgers1D(TimeDependentProblem, SpatialProblem):

    # assign output/ spatial and temporal variables
    output_variables = ['u']
    spatial_domain = Span({'x': [-1, 1]})
    temporal_domain = Span({'t': [0, 1]})

    # define the burger equation
    def burger_equation(input_, output_):
        du = grad(output_, input_)
        ddu = grad(du, input_, components=['dudx'])
        return (
            du.extract(['dudt']) +
            output_.extract(['u'])*du.extract(['dudx']) -
            (0.01/torch.pi)*ddu.extract(['ddudxdx'])
        )

    # define nill dirichlet boundary conditions
    def nil_dirichlet(input_, output_):
        u_expected = 0.0
        return output_.extract(['u']) - u_expected

    # define initial condition
    def initial_condition(input_, output_):
        u_expected = -torch.sin(torch.pi*input_.extract(['x']))
        return output_.extract(['u']) - u_expected

    # problem condition statement
    conditions = {
        'gamma1': Condition(location=Span({'x': -1, 't': [0, 1]}), function=nil_dirichlet),
        'gamma2': Condition(location=Span({'x':  1, 't': [0, 1]}), function=nil_dirichlet),
        't0': Condition(location=Span({'x': [-1, 1], 't': 0}), function=initial_condition),
        'D': Condition(location=Span({'x': [-1, 1], 't': [0, 1]}), function=burger_equation),
    }
