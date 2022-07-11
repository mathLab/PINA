import argparse
import torch
from torch.nn import Softplus
from pina import Plotter, LabelTensor, PINN
from parametric_poisson2 import ParametricPoisson
from pina.model import FeedForward


class myFeature(torch.nn.Module):
    """
    """
    def __init__(self):
        super(myFeature, self).__init__()

    def forward(self, x):
        t = (
            torch.exp(
                - 2*(x.extract(['x']) - x.extract(['mu1']))**2
                - 2*(x.extract(['y']) - x.extract(['mu2']))**2
            )
        )
        return LabelTensor(t, ['k0'])


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
        layers=[10, 10, 10],
        output_variables=poisson_problem.output_variables,
        input_variables=poisson_problem.input_variables,
        func=Softplus,
        extra_features=feat
    )

    pinn = PINN(
        poisson_problem,
        model,
        lr=0.006,
        regularizer=1e-6)

    if args.s:

        pinn.span_pts(
            {'variables': ['x', 'y'], 'mode': 'random', 'n': 100},
            {'variables': ['mu1', 'mu2'], 'mode': 'grid', 'n': 5},
            locations=['D'])
        pinn.span_pts(
            {'variables': ['x', 'y'], 'mode': 'grid', 'n': 20},
            {'variables': ['mu1', 'mu2'], 'mode': 'grid', 'n': 5},
            locations=['gamma1', 'gamma2', 'gamma3', 'gamma4'])
        pinn.train(10000, 100)
        pinn.save_state('pina.poisson_param')

    else:
        pinn.load_state('pina.poisson_param')
        plotter = Plotter()
        plotter.plot(pinn, component='u', parametric=True, params_value=0)
