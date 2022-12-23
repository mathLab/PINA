import torch
from torch.nn import Softplus

from pina import PINN, Plotter, LabelTensor
from pina.model import FeedForward
from problems.burgers import Burgers1D

from utils import setup_generic_run_parser, setup_extra_features_parser


class myFeature(torch.nn.Module):
    """
    Feature: sin(pi*x)
    """
    def __init__(self, idx):
        super(myFeature, self).__init__()
        self.idx = idx

    def forward(self, x):
        return LabelTensor(torch.sin(torch.pi * x.extract(['x'])), ['sin(x)'])


if __name__ == "__main__":
    # fmt: off
    args = setup_extra_features_parser(
        setup_generic_run_parser()
    ).parse_args()
    # fmt: on

    feat = [myFeature(0)] if args.extra else []

    burgers_problem = Burgers1D()
    model = FeedForward(
        layers=[30, 20, 10, 5],
        output_variables=burgers_problem.output_variables,
        input_variables=burgers_problem.input_variables,
        func=Softplus,
        extra_features=feat,
    )

    pinn = PINN(
        burgers_problem,
        model,
        lr=0.006,
        error_norm='mse',
        regularizer=0)

    if args.save:
        pinn.span_pts(
            {'n': 200, 'mode': 'grid', 'variables': 't'},
            {'n': 20, 'mode': 'grid', 'variables': 'x'},
            locations=['D'])
        pinn.span_pts(150, 'random', location=['gamma1', 'gamma2', 't0'])
        pinn.train(5000, 100)
        pinn.save_state(f'pina.burger{args.id_run}')
    if args.load:
        pinn.load_state(f'pina.burger{args.id_run}')
        plotter = Plotter()
        plotter.plot(pinn)
