import argparse
import sys
import numpy as np
import torch
from torch.nn import ReLU, Tanh, Softplus

from pina import PINN, LabelTensor, Plotter
from pina.model import FeedForward
from pina.adaptive_functions import AdaptiveSin, AdaptiveCos, AdaptiveTanh
from problems.poisson import Poisson


class myFeature(torch.nn.Module):
    """
    Feature: sin(x)
    """

    def __init__(self):
        super(myFeature, self).__init__()

    def forward(self, x):
        return torch.sin(x[:, 0]*torch.pi) * torch.sin(x[:, 1]*torch.pi)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run PINA")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-s", "-save", action="store_true")
    group.add_argument("-l", "-load", action="store_true")
    parser.add_argument("id_run", help="number of run", type=int)
    parser.add_argument("features", help="extra features", type=int)
    args = parser.parse_args()

    feat = [myFeature()] if args.features else []

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

    if args.s:

        print(pinn)
        pinn.span_pts(20, 'grid', ['gamma1', 'gamma2', 'gamma3', 'gamma4'])
        pinn.span_pts(20, 'grid', ['D'])
        #pinn.plot_pts()
        pinn.train(5000, 100)
        with open('poisson_history_{}_{}.txt'.format(args.id_run, args.features), 'w') as file_:
            for i, losses in enumerate(pinn.history):
                file_.write('{} {}\n'.format(i, sum(losses)))
        pinn.save_state('pina.poisson')

    else:
        pinn.load_state('pina.poisson')
        plotter = Plotter()
        plotter.plot(pinn)


