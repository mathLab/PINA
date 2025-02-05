""" Definition of the inverse Poisson problem on a square domain."""

from pina.problem import SpatialProblem, InverseProblem
from pina.operators import laplacian
from pina import Condition, LabelTensor
from pina.domain import CartesianDomain
from pina.equation.equation import Equation
from pina.equation.equation_factory import FixedValue
import torch

def laplace_equation(input_, output_, params_):
    force_term = torch.exp(- 2*(input_.extract(['x']) - params_['mu1'])**2
                            - 2*(input_.extract(['y']) - params_['mu2'])**2)
    delta_u = laplacian(output_, input_, components=['u'], d=['x', 'y'])
    return delta_u - force_term

class InversePoisson2DSquareProblem(SpatialProblem, InverseProblem):
    output_variables = ['u']
    x_min, x_max = -2, 2
    y_min, y_max = -2, 2
    data_input = LabelTensor(torch.rand(10, 2), ['x', 'y'])
    data_output = LabelTensor(torch.rand(10, 1), ['u'])
    spatial_domain = CartesianDomain({'x': [x_min, x_max], 'y': [y_min, y_max]})
    unknown_parameter_domain = CartesianDomain({'mu1': [-1, 1], 'mu2': [-1, 1]})

    domains = {
        'g1': CartesianDomain({'x': [x_min, x_max], 'y':  y_max}),
        'g2': CartesianDomain({'x': [x_min, x_max], 'y': y_min}),
        'g3': CartesianDomain({'x':  x_max, 'y': [y_min, y_max]}),
        'g4': CartesianDomain({'x': x_min, 'y': [y_min, y_max]}),
        'D': CartesianDomain({'x': [x_min, x_max], 'y': [y_min, y_max]}),
    }

    conditions = {
        'nil_g1': Condition(domain='g1', equation=FixedValue(0.0)),
        'nil_g2': Condition(domain='g2', equation=FixedValue(0.0)),
        'nil_g3': Condition(domain='g3', equation=FixedValue(0.0)),
        'nil_g4': Condition(domain='g4', equation=FixedValue(0.0)),
        'laplace_D': Condition(domain='D', equation=Equation(laplace_equation)),
        'data': Condition(
            input_points=data_input.extract(['x', 'y']),
            output_points=data_output)
    }