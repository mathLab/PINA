""" Run PINA on ODE equation. """
import argparse
import torch
from torch.nn import Softplus

from pina.model import FeedForward
from pina.solvers import PINN
from pina.plotter import Plotter
from pina.trainer import Trainer
from problems.first_order_ode import FirstOrderODE



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run PINA")
    parser.add_argument("--load", help="directory to save or load file", type=str)
    parser.add_argument("--epochs", help="extra features", type=int, default=3000)
    args = parser.parse_args()


    # create problem and discretise domain
    problem = FirstOrderODE()
    problem.discretise_domain(n=500, mode='grid', variables = 'x', locations=['D'])
    problem.discretise_domain(n=1, mode='grid', variables = 'x', locations=['BC'])

    # create model
    model = FeedForward(
        layers=[10, 10],
        output_dimensions=len(problem.output_variables),
        input_dimensions=len(problem.input_variables),
        func=Softplus
    )

    # create solver
    pinn = PINN(
        problem=problem,
        model=model,
        extra_features=None,
        optimizer_kwargs={'lr' : 0.001}
    )

    # create trainer
    directory = 'pina.ode'
    trainer = Trainer(solver=pinn, accelerator='cpu', max_epochs=args.epochs, default_root_dir=directory)


    if args.load:
        pinn = PINN.load_from_checkpoint(checkpoint_path=args.load, problem=problem, model=model)
        plotter = Plotter()
        plotter.plot(pinn)
    else:
        trainer.train()