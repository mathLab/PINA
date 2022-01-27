import sys
import numpy as np
import torch
import argparse
from pina.pinn import PINN
from pina.ppinn import ParametricPINN as pPINN
from pina.label_tensor import LabelTensor
from torch.nn import ReLU, Tanh, Softplus
from problems.burgers import Burgers1D
from pina.deep_feed_forward import DeepFeedForward

from pina import Plotter


class myFeature(torch.nn.Module):
    """
    Feature: sin(pi*x)
    """
    def __init__(self, idx):
        super(myFeature, self).__init__()
        self.idx = idx

    def forward(self, x):
        return torch.sin(torch.pi * x[:, self.idx])

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
    model = DeepFeedForward(
        layers=[30, 20, 10, 5],
        #layers=[8, 8, 8],
        #layers=[16, 8, 4, 4],
        #layers=[20, 4, 4, 4],
        output_variables=burgers_problem.output_variables,
        input_variables=burgers_problem.input_variables,
        func=Softplus,
        extra_features=feat
    )

    pinn = PINN(
        burgers_problem,
        model,
        lr=0.006,
        error_norm='mse',
        regularizer=0,
        lr_accelerate=None)

    if args.s:
        pinn.span_pts(2000, 'latin', ['D'])
        pinn.span_pts(150, 'random', ['gamma1', 'gamma2', 'initia'])
        pinn.train(5000, 100)
        pinn.save_state('pina.burger.{}.{}'.format(args.id_run, args.features))
    else:
        pinn.load_state('pina.burger.{}.{}'.format(args.id_run, args.features))
        plotter = Plotter()
        plotter.plot(pinn)
