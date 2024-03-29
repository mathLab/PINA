""" Run PINA on Burgers equation. """

import argparse
import torch
from torch.nn import Softplus

from pina import LabelTensor
from pina.model import FeedForward
from pina.solvers import PINN
from pina.plotter import Plotter
from pina.trainer import Trainer
from problems.burgers import Burgers1D


class myFeature(torch.nn.Module):
    """
    Feature: sin(pi*x)
    """

    def __init__(self):
        super(myFeature, self).__init__()

    def forward(self, x):
        return LabelTensor(torch.sin(torch.pi * x.extract(['x'])), ['sin(x)'])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run PINA")
    parser.add_argument("--load", help="directory to save or load file", type=str)
    parser.add_argument("--features", help="extra features", type=int)
    parser.add_argument("--epochs", help="extra features", type=int, default=1000)
    args = parser.parse_args()

    if args.features is None:
        args.features = 0

    # extra features
    feat = [myFeature()] if args.features else []

    # create problem and discretise domain
    burgers_problem = Burgers1D()
    burgers_problem.discretise_domain(n=200, mode='grid', variables = 't', locations=['D'])
    burgers_problem.discretise_domain(n=20, mode='grid', variables = 'x', locations=['D'])
    burgers_problem.discretise_domain(n=150, mode='random', locations=['gamma1', 'gamma2', 't0'])

    # create model
    model = FeedForward(
        layers=[30, 20, 10, 5],
        output_dimensions=len(burgers_problem.output_variables),
        input_dimensions=len(burgers_problem.input_variables) + len(feat),
        func=Softplus
    )

    # create solver
    pinn = PINN(
        problem=burgers_problem,
        model=model,
        extra_features=feat,
        optimizer_kwargs={'lr' : 0.006}
    )

    # create trainer
    directory = 'pina.burger_extrafeats_{}'.format(bool(args.features))
    trainer = Trainer(solver=pinn, accelerator='cpu', max_epochs=args.epochs, default_root_dir=directory)


    if args.load:
        pinn = PINN.load_from_checkpoint(checkpoint_path=args.load, problem=burgers_problem, model=model)
        plotter = Plotter()
        plotter.plot(pinn)
    else:
        trainer.train()
