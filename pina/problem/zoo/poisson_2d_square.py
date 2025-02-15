""" Definition of the Poisson problem on a square domain."""

from pina.problem import SpatialProblem
from pina.operators import laplacian
from pina import LabelTensor, Condition
from pina.domain import CartesianDomain
from pina.equation.equation import Equation
from pina.equation.equation_factory import FixedValue
import torch

def laplace_equation(input_, output_):
    force_term = (torch.sin(input_.extract(['x']) * torch.pi) *
                  torch.sin(input_.extract(['y']) * torch.pi))
    delta_u = laplacian(output_.extract(['u']), input_)
    return delta_u - force_term


my_laplace = Equation(laplace_equation)


class Poisson2DSquareProblem(SpatialProblem):
    output_variables = ['u']
    spatial_domain = CartesianDomain({'x': [0, 1], 'y': [0, 1]})

    domains = {
        'D': CartesianDomain({'x': [0, 1], 'y': [0, 1]}),
        'g1': CartesianDomain({'x': [0, 1], 'y': 1}),
        'g2': CartesianDomain({'x': [0, 1], 'y': 0}),
        'g3': CartesianDomain({'x': 1, 'y': [0, 1]}),
        'g4': CartesianDomain({'x': 0, 'y': [0, 1]}),
    }

    conditions = {
        'nil_g1': Condition(domain='D', equation=FixedValue(0.0)),
        'nil_g2': Condition(domain='D', equation=FixedValue(0.0)),
        'nil_g3': Condition(domain='D', equation=FixedValue(0.0)),
        'nil_g4': Condition(domain='D', equation=FixedValue(0.0)),
        'laplace_D': Condition(domain='D', equation=my_laplace),
    }

    def poisson_sol(self, pts):
        return -(torch.sin(pts.extract(['x']) * torch.pi) *
                 torch.sin(pts.extract(['y']) * torch.pi))
    
