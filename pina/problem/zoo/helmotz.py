""" Definition of the diffusion-reaction problem."""

import torch
from pina import Condition, LabelTensor
from pina.problem import SpatialProblem
from pina.equation import Equation, FixedValue
from pina.domain import CartesianDomain
from pina.operators import laplacian


class HelmotzEquation(Equation):
    def __init__(self, alpha):
        self.alpha = alpha
        def equation(input_, output_):
            x = input_.extract('x')
            y = input_.extract('y')
            laplacian_u = laplacian(output_, input_, components=['u'])
            q = (1- 2 * (self.alpha * torch.pi)**2) * torch.sin(
                self.alpha*torch.pi*x)*torch.sin(self.alpha*torch.pi*y)
            return laplacian_u + output_ - q
        super().__init__(equation)

class HelmholtzProblem(SpatialProblem):
    """
    Taken from https://www.arxiv.org/pdf/2502.04917
    """
    output_variables = ['u']
    spatial_domain = CartesianDomain({'x': [-1, 1], 'y' : [-1, 1]})

    domains = {
        'D' : CartesianDomain({'x': [-1, 1], 'y' : [-1, 1]}),
        'g1': CartesianDomain({'x': [-1, 1], 'y': 1}),
        'g2': CartesianDomain({'x': [-1, 1], 'y': -1}),
        'g3': CartesianDomain({'x': 1, 'y': [-1, 1]}),
        'g4': CartesianDomain({'x': -1, 'y': [-1, 1]}),
    }

    conditions = dict()

    def __init__(self, alpha=3):
        super().__init__()
        self.alpha = alpha
        self.conditions = {
            'D': Condition(domain='D', equation=HelmotzEquation(self.alpha)),
            'g1' : Condition(domain='g1', equation=FixedValue(0)),
            'g2' : Condition(domain='g2', equation=FixedValue(0)),
            'g3' : Condition(domain='g1', equation=FixedValue(0)),
            'g4' : Condition(domain='g2', equation=FixedValue(0)),
        }

    def solution(self, pts):
        x = pts.extract('x')
        y = pts.extract('y')
        return LabelTensor(
            torch.sin(self.alpha * torch.pi * x) * 
            torch.sin(self.alpha * torch.pi * y)
            ,labels=self.output_variables)
