""" Definition of the diffusion-reaction problem."""

import torch
from pina import Condition, LabelTensor
from pina.problem import SpatialProblem, TimeDependentProblem, InverseProblem
from pina.equation.equation import Equation
from pina.domain import CartesianDomain
from pina.operator import grad

def diffusion_reaction(input_, output_):
    """
    Implementation of the diffusion-reaction equation.
    """
    x = input_.extract('x')
    t = input_.extract('t')
    u_t = grad(output_, input_, d='t')
    u_x = grad(output_, input_, d='x')
    u_xx = grad(u_x, input_, d='x')
    r = torch.exp(-t) * (1.5 * torch.sin(2*x) + (8/3) * torch.sin(3*x) +
                         (15/4) * torch.sin(4*x) + (63/8) * torch.sin(8*x))
    return u_t - u_xx - r

class InverseDiffusionReactionProblem(TimeDependentProblem,
                                      SpatialProblem,
                                      InverseProblem):
    """
    Implementation of the diffusion-reaction inverse problem on the spatial 
    interval [-pi, pi] and temporal interval [0,1], with unknown parameters 
    in the interval [-1,1].
    """
    output_variables = ['u']
    spatial_domain = CartesianDomain({'x': [-torch.pi, torch.pi]})
    temporal_domain = CartesianDomain({'t': [0, 1]})
    unknown_parameter_domain = CartesianDomain({'mu': [-1, 1]})

    conditions = {
        'D': Condition(
            domain=CartesianDomain({'x': [-torch.pi, torch.pi], 't': [0, 1]}),
            equation=Equation(diffusion_reaction)),
        'data' : Condition(
            input_points=LabelTensor(torch.randn(10, 2), ['x', 't']),
            output_points=LabelTensor(torch.randn(10, 1), ['u'])),
    }

    def _solution(self, pts):
        t = pts.extract('t')
        x = pts.extract('x')
        return torch.exp(-t) * (
            torch.sin(x) + (1/2)*torch.sin(2*x) + (1/3)*torch.sin(3*x) +
            (1/4)*torch.sin(4*x) + (1/8)*torch.sin(8*x)
        )
