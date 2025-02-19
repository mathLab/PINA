"""Definition of the diffusion-reaction problem."""

import torch
from pina import Condition, LabelTensor
from pina.problem import SpatialProblem, TimeDependentProblem, InverseProblem
from pina.equation import Equation, FixedValue
from pina.domain import CartesianDomain
from pina.operator import grad


def diffusion_reaction(input_, output_):
    """
    Implementation of the diffusion-reaction equation.
    """
    x = input_.extract("x")
    t = input_.extract("t")
    u_t = grad(output_, input_, d="t")
    u_x = grad(output_, input_, d="x")
    u_xx = grad(u_x, input_, d="x")
    r = torch.exp(-t) * (
        1.5 * torch.sin(2 * x)
        + (8 / 3) * torch.sin(3 * x)
        + (15 / 4) * torch.sin(4 * x)
        + (63 / 8) * torch.sin(8 * x)
    )
    return u_t - u_xx - r


def initial_condition(input_, output_):
    t = input_.extract('t')
    x = input_.extract('x')
    u_0 = (torch.sin(x) + (1/2)*torch.sin(2*x) + 
           (1/3)*torch.sin(3*x) + (1/4)*torch.sin(4*x) + (1/8)*torch.sin(8*x))
    return output_ - u_0

class InverseDiffusionReactionProblem(TimeDependentProblem,
                                      SpatialProblem,
                                      InverseProblem):
    """
    Implementation of the diffusion-reaction inverse problem on the spatial 
    interval [-pi, pi] and temporal interval [0,1], with unknown parameters 
    in the interval [-1,1]. Taken from https://www.arxiv.org/pdf/2502.04917
>>>>>>> e218673 (adding problems)
    """

    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [-torch.pi, torch.pi]})
    temporal_domain = CartesianDomain({"t": [0, 1]})
    unknown_parameter_domain = CartesianDomain({"mu": [-1, 1]})

    domains = {
        'D' : CartesianDomain({'x': [-torch.pi, torch.pi], 't': [0, 1]}),
        'g1' : CartesianDomain({'x': -torch.pi, 't': [0, 1]}),
        'g2' : CartesianDomain({'x': torch.pi, 't': [0, 1]}),
        't0' : CartesianDomain({'x': [-torch.pi, torch.pi], 't': 0}),
    }
    conditions = {
        'D': Condition(domain='D', equation=Equation(diffusion_reaction)),
        'g1' : Condition(domain='g1', equation=FixedValue(0)),
        'g2' : Condition(domain='g2', equation=FixedValue(0)),
        't0' : Condition(domain='t0', equation=Equation(initial_condition)),

    }

    def __init__(self):
        super().__init__()
        pts = self.spatial_domain.sample(100)
        self.conditions['data'] = Condition(
            input_points=pts,
            output_points=self.solution(pts)
            )
        
    def solution(self, pts):
        t = pts.extract('t')
        x = pts.extract('x')
        return LabelTensor(
            torch.exp(-t) * (
            torch.sin(x) + (1/2)*torch.sin(2*x) + (1/3)*torch.sin(3*x) +
            (1/4)*torch.sin(4*x) + (1/8)*torch.sin(8*x)
        ), labels=self.output_variables)
