import argparse
import numpy as np
import torch
from torch.nn import ReLU, Tanh, Softplus

from pina import PINN, Plotter
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

    if args.s:

        pinn.span_pts(200, 'grid', locations=['gamma_top', 'gamma_bot', 'gamma_in', 'gamma_out'])
        # pinn.span_pts(2000, 'random', locations=['D'])
        pinn.span_pts(2000, 'random', locations=['D1'])
        pinn.span_pts(2000, 'random', locations=['D2'])
        pinn.train(10000, 100)
        with open('stokes_history_{}.txt'.format(args.id_run), 'w') as file_:
            for i, losses in pinn.history_loss.items():
                file_.write('{} {}\n'.format(i, sum(losses)))
        pinn.save_state('pina.stokes')

    else:
        pinn.load_state('pina.stokes')
        plotter = Plotter()
        plotter.plot(pinn, components='ux')
        plotter.plot(pinn, components='uy')
        plotter.plot(pinn, components='p')


