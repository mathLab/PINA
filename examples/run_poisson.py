""" Run PINA on ODE equation. """

import argparse
import torch
from torch.nn import Softplus

from pina import LabelTensor
from pina.model import FeedForward
from pina.solvers import PINN
from pina.plotter import Plotter
from pina.trainer import Trainer
from problems.poisson import Poisson


class myFeature(torch.nn.Module):
    """
    Feature: sin(x)
    """

    def __init__(self):
        super(myFeature, self).__init__()

    def forward(self, x):
        t = torch.sin(x.extract(["x"]) * torch.pi) * torch.sin(
            x.extract(["y"]) * torch.pi
        )
        return LabelTensor(t, ["sin(x)sin(y)"])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run PINA")
    parser.add_argument(
        "--load", help="directory to save or load file", type=str
    )
    parser.add_argument("--features", help="extra features", type=int)
    parser.add_argument(
        "--epochs", help="extra features", type=int, default=1000
    )
    args = parser.parse_args()

    if args.features is None:
        args.features = 0

    # extra features
    feat = [myFeature()] if args.features else []
    args = parser.parse_args()

    # create problem and discretise domain
    problem = Poisson()
    problem.discretise_domain(n=20, mode="grid", locations=["D"])
    problem.discretise_domain(
        n=100, mode="random", locations=["gamma1", "gamma2", "gamma3", "gamma4"]
    )

    # create model
    model = FeedForward(
        layers=[10, 10],
        output_dimensions=len(problem.output_variables),
        input_dimensions=len(problem.input_variables) + len(feat),
        func=Softplus,
    )

    # create solver
    pinn = PINN(
        problem=problem,
        model=model,
        extra_features=feat,
        optimizer_kwargs={"lr": 0.001},
    )

    # create trainer
    directory = f"pina.parametric_poisson_extrafeats_{bool(args.features)}"
    trainer = Trainer(
        solver=pinn,
        accelerator="cpu",
        max_epochs=args.epochs,
        default_root_dir=directory,
    )

    if args.load:
        pinn = PINN.load_from_checkpoint(
            checkpoint_path=args.load,
            problem=problem,
            model=model,
            extra_features=feat,
        )
        plotter = Plotter()
        plotter.plot(pinn)
    else:
        trainer.train()
