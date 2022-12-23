import sys
import numpy as np
import torch
from torch.nn import ReLU, Tanh, Softplus

from pina import PINN, LabelTensor, Plotter
from pina.model import FeedForward
from pina.adaptive_functions import AdaptiveSin, AdaptiveCos, AdaptiveTanh
from problems.stokes import Stokes

from utils import setup_generic_run_parser


if __name__ == "__main__":
    args = setup_generic_run_parser().parse_args()

    stokes_problem = Stokes()
    model = FeedForward(
        layers=[10, 10, 10, 10],
        output_variables=stokes_problem.output_variables,
        input_variables=stokes_problem.input_variables,
        func=Softplus,
    )

    pinn = PINN(
        stokes_problem,
        model,
        lr=0.006,
        error_norm='mse',
        regularizer=1e-8)

    if args.save:
        pinn.span_pts(200, 'grid', locations=['gamma_top', 'gamma_bot', 'gamma_in', 'gamma_out'])
        pinn.span_pts(2000, 'random', locations=['D'])
        pinn.train(10000, 100)
        with open('stokes_history_{}.txt'.format(args.id_run), 'w') as file_:
            for i, losses in enumerate(pinn.history):
                file_.write('{} {}\n'.format(i, sum(losses)))
        pinn.save_state(f'pina.stokes{args.id_run}')
    if args.load:
        pinn.load_state(f'pina.stokes{args.id_run}')
        plotter = Plotter()
        plotter.plot(pinn, components='ux')
        plotter.plot(pinn, components='uy')
        plotter.plot(pinn, components='p')


