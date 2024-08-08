""" Wave equation Problem """


import torch
from pina.domain import CartesianDomain
from pina import Condition
from pina.problem import SpatialProblem, TimeDependentProblem
from pina.operators import laplacian, grad
from pina.equation import FixedValue, Equation


# ===================================================== #
#                                                       #
#  This script implements the two dimensional           #
#  Wave equation. The Wave class is defined inheriting  #
#  from SpatialProblem and TimeDependentProblem. Let    #
#           u --> field variable                        #
#           x,y --> spatial variables                   #
#           t --> temporal variables                    #
# the velocity coefficient is set to one.               #
#                                                       #
# ===================================================== #



class Wave(TimeDependentProblem, SpatialProblem):
    output_variables = ['u']
    spatial_domain = CartesianDomain({'x': [0, 1], 'y': [0, 1]})
    temporal_domain = CartesianDomain({'t': [0, 1]})

    def wave_equation(input_, output_):
        u_t = grad(output_, input_, components=['u'], d=['t'])
        u_tt = grad(u_t, input_, components=['dudt'], d=['t'])
        nabla_u = laplacian(output_, input_, components=['u'], d=['x', 'y'])
        return nabla_u - u_tt

    def initial_condition(input_, output_):
        u_expected = (torch.sin(torch.pi*input_.extract(['x'])) *
                      torch.sin(torch.pi*input_.extract(['y'])))
        return output_.extract(['u']) - u_expected

    conditions = {
        'gamma1': Condition(location=CartesianDomain({'x': [0, 1], 'y':  1, 't': [0, 1]}), equation=FixedValue(0.)),
        'gamma2': Condition(location=CartesianDomain({'x': [0, 1], 'y': 0, 't': [0, 1]}), equation=FixedValue(0.)),
        'gamma3': Condition(location=CartesianDomain({'x':  1, 'y': [0, 1], 't': [0, 1]}), equation=FixedValue(0.)),
        'gamma4': Condition(location=CartesianDomain({'x': 0, 'y': [0, 1], 't': [0, 1]}), equation=FixedValue(0.)),
        't0': Condition(location=CartesianDomain({'x': [0, 1], 'y': [0, 1], 't': 0}), equation=Equation(initial_condition)),
        'D': Condition(location=CartesianDomain({'x': [0, 1], 'y': [0, 1], 't': [0, 1]}), equation=Equation(wave_equation)),
    }

    def wave_sol(self, pts):
        sqrt_2 = torch.sqrt(torch.tensor(2.))
        return (torch.sin(torch.pi*pts.extract(['x'])) *
                torch.sin(torch.pi*pts.extract(['y'])) *
                torch.cos(sqrt_2*torch.pi*pts.extract(['t'])))

    truth_solution = wave_sol