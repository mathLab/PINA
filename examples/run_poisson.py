import sys
import numpy as np
import torch
from torch.nn import ReLU, Tanh, Softplus

from pina import PINN, LabelTensor, Plotter
from pina.model import FeedForward
from pina.adaptive_functions import AdaptiveSin, AdaptiveCos, AdaptiveTanh
from problems.poisson import Poisson

from utils import setup_generic_run_parser, setup_extra_features_parser


class myFeature(torch.nn.Module):
    """
    Feature: sin(x)
    """

    def __init__(self):
        super(myFeature, self).__init__()

    def forward(self, x):
        t = (torch.sin(x.extract(['x'])*torch.pi) *
             torch.sin(x.extract(['y'])*torch.pi))
        return LabelTensor(t, ['sin(x)sin(y)'])

if __name__ == "__main__":
    # fmt: off
    args = setup_extra_features_parser(
        setup_generic_run_parser()
    ).parse_args()
    # fmt: on

    feat = [myFeature()] if args.extra else []

    poisson_problem = Poisson()
    model = FeedForward(
        layers=[20, 20],
        output_variables=poisson_problem.output_variables,
        input_variables=poisson_problem.input_variables,
        func=Softplus,
        extra_features=feat
    )

    pinn = PINN(
        poisson_problem,
        model,
        lr=0.03,
        error_norm='mse',
        regularizer=1e-8)

    if args.save:
        pinn.span_pts(20, 'grid', locations=['gamma1', 'gamma2', 'gamma3', 'gamma4'])
        pinn.span_pts(20, 'grid', locations=['D'])
        pinn.train(5000, 100)
        pinn.save_state(f'pina.poisson{args.id_run}')
    if args.load:
        pinn.load_state(f'pina.poisson{args.id_run}')
        plotter = Plotter()
        plotter.plot(pinn)
