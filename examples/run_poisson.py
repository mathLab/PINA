import sys
import numpy as np
import torch
import argparse
from pina import PINN
from pina.ppinn import ParametricPINN as pPINN
from pina.label_tensor import LabelTensor
from torch.nn import ReLU, Tanh, Softplus
from problems.poisson2D import Poisson2D
from pina.deep_feed_forward import DeepFeedForward

from pina.adaptive_functions import AdaptiveSin, AdaptiveCos, AdaptiveTanh

from pina import Plotter

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

    poisson_problem = Poisson2D()
    model = DeepFeedForward(
        layers=[10, 10],
        output_variables=poisson_problem.output_variables,
        input_variables=poisson_problem.input_variables,
        func=Softplus,
        extra_features=feat
    )

    pinn = PINN(
        poisson_problem,
        model,
        lr=0.003,
        error_norm='mse',
        regularizer=1e-8,
        lr_accelerate=None)

    if args.s:

        pinn.span_pts(20, 'grid', ['D'])
        pinn.span_pts(20, 'grid', ['gamma1', 'gamma2', 'gamma3', 'gamma4'])
        #pinn.plot_pts()
        pinn.train(1000, 100)
        with open('poisson_history_{}_{}.txt'.format(args.id_run, args.features), 'w') as file_:
            for i, losses in enumerate(pinn.history):
                file_.write('{} {}\n'.format(i, sum(losses)))
        pinn.save_state('pina.poisson')

    else:
        pinn.load_state('pina.poisson')
        plotter = Plotter()
        plotter.plot(pinn)


