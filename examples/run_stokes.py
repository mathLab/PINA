import argparse
import sys
import numpy as np
import torch
from torch.nn import ReLU, Tanh, Softplus

from pina import PINN, LabelTensor, Plotter
from pina.model import FeedForward
from pina.adaptive_functions import AdaptiveSin, AdaptiveCos, AdaptiveTanh
from problems.stokes import Stokes

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run PINA")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-s", "-save", action="store_true")
    group.add_argument("-l", "-load", action="store_true")
    parser.add_argument("id_run", help="number of run", type=int)
    args = parser.parse_args()


    stokes_problem = Stokes()
    model = FeedForward(
        layers=[40, 20, 20, 10],
        output_variables=stokes_problem.output_variables,
        input_variables=stokes_problem.input_variables,
        func=Softplus,
    )

    pinn = PINN(
        stokes_problem,
        model,
        lr=0.006,
        error_norm='mse',
        regularizer=1e-8,
        lr_accelerate=None)

    if args.s:

        #pinn.span_pts(200, 'grid', ['gamma_out'])
        pinn.span_pts(200, 'grid', ['gamma_top', 'gamma_bot', 'gamma_in', 'gamma_out'])
        pinn.span_pts(2000, 'random', ['D'])
        #plotter = Plotter()
        #plotter.plot_samples(pinn)
        pinn.train(10000, 100)
        pinn.save_state('pina.stokes')

    else:
        pinn.load_state('pina.stokes')
        plotter = Plotter()
        plotter.plot_samples(pinn)
        plotter.plot(pinn)


