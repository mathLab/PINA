"""Run PINA on Burgers equation"""
import argparse
import torch
from torch.nn import Softplus

from pina import PINN, Plotter, LabelTensor
from pina.model import FeedForward
from problems.burgers import Burgers1D


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

    parser = argparse.ArgumentParser(description="Run PINA")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-s", "-save", action="store_true")
    group.add_argument("-l", "-load", action="store_true")
    parser.add_argument("id_run", help="number of run", type=int)
    parser.add_argument("features", help="extra features", type=int)
    args = parser.parse_args()

    feat = [myFeature(0)] if args.features else []

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

    if args.s:
        pinn.span_pts(
            {'n': 200, 'mode': 'grid', 'variables': 't'},
            {'n': 20, 'mode': 'grid', 'variables': 'x'},
            locations=['D'])
        pinn.span_pts(150, 'random', location=['gamma1', 'gamma2', 't0'])
        pinn.train(5000, 100)
        pinn.save_state('pina.burger.{}.{}'.format(args.id_run, args.features))
    else:
        pinn.load_state('pina.burger.{}.{}'.format(args.id_run, args.features))
        plotter = Plotter()
        plotter.plot(pinn)
