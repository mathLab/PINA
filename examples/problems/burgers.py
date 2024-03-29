""" Burgers' problem. """


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


import torch
from pina.geometry import CartesianDomain
from pina import Condition
from pina.problem import TimeDependentProblem, SpatialProblem
from pina.operators import grad
from pina.equation import FixedValue, Equation

        
class Burgers1D(TimeDependentProblem, SpatialProblem):

    # define the burger equation
    def burger_equation(input_, output_):
        du = grad(output_, input_)
        ddu = grad(du, input_, components=['dudx'])
        return (
            du.extract(['dudt']) +
            output_.extract(['u'])*du.extract(['dudx']) -
            (0.01/torch.pi)*ddu.extract(['ddudxdx'])
        )

    # define initial condition
    def initial_condition(input_, output_):
        u_expected = -torch.sin(torch.pi*input_.extract(['x']))
        return output_.extract(['u']) - u_expected

    # assign output/ spatial and temporal variables
    output_variables = ['u']
    spatial_domain = CartesianDomain({'x': [-1, 1]})
    temporal_domain = CartesianDomain({'t': [0, 1]})

    # problem condition statement
    conditions = {
        'gamma1': Condition(location=CartesianDomain({'x': -1, 't': [0, 1]}), equation=FixedValue(0.)),
        'gamma2': Condition(location=CartesianDomain({'x':  1, 't': [0, 1]}), equation=FixedValue(0.)),
        't0': Condition(location=CartesianDomain({'x': [-1, 1], 't': 0}), equation=Equation(initial_condition)),
        'D': Condition(location=CartesianDomain({'x': [-1, 1], 't': [0, 1]}), equation=Equation(burger_equation)),
    }