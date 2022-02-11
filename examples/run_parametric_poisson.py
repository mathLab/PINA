import argparse
import torch
from torch.nn import Softplus

from pina import PINN as pPINN
from problems.parametric_poisson import ParametricPoisson
from pina.model import FeedForward


class myFeature(torch.nn.Module):
    """
    """
    def __init__(self):
        super(myFeature, self).__init__()

    def forward(self, x):
        return torch.exp(- 2*(x['x'] - x['mu1'])**2 - 2*(x['y'] - x['mu2'])**2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run PINA")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-s", "-save", action="store_true")
    group.add_argument("-l", "-load", action="store_true")
    parser.add_argument("id_run", help="number of run", type=int)
    parser.add_argument("features", help="extra features", type=int)
    args = parser.parse_args()

    feat = [myFeature()] if args.features else []

    poisson_problem = ParametricPoisson()
    model = FeedForward(
        layers=[200, 40, 10],
        output_variables=poisson_problem.output_variables,
        input_variables=poisson_problem.input_variables,
        func=Softplus,
        extra_features=feat
    )

    pinn = pPINN(
        poisson_problem,
        model,
        lr=0.0006,
        regularizer=1e-6,
        lr_accelerate=None)

    if args.s:

        pinn.span_pts(2000, 'random', ['D'])
        pinn.span_pts(200, 'random', ['gamma1', 'gamma2', 'gamma3', 'gamma4'])
        pinn.train(10000, 10)
        pinn.save_state('pina.poisson_param')

    else:
        pinn.load_state('pina.poisson_param')
