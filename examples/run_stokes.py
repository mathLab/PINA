import argparse
from torch.nn import Softplus

from pina import Plotter, Trainer
from pina.model import FeedForward
from pina.solvers import PINN
from problems.stokes import Stokes

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run PINA")
    parser = argparse.ArgumentParser(description="Run PINA")
    parser.add_argument(
        "--load", help="directory to save or load file", type=str
    )
    parser.add_argument(
        "--epochs", help="extra features", type=int, default=1000
    )
    args = parser.parse_args()

    # create problem and discretise domain
    stokes_problem = Stokes()
    stokes_problem.discretise_domain(
        n=1000, locations=["gamma_top", "gamma_bot", "gamma_in", "gamma_out"]
    )
    stokes_problem.discretise_domain(n=2000, locations=["D"])

    # make the model
    model = FeedForward(
        layers=[10, 10, 10, 10],
        output_dimensions=len(stokes_problem.output_variables),
        input_dimensions=len(stokes_problem.input_variables),
        func=Softplus,
    )

    # make the pinn
    pinn = PINN(stokes_problem, model, optimizer_kwargs={"lr": 0.001})

    # create trainer
    directory = "pina.navier_stokes"
    trainer = Trainer(
        solver=pinn,
        accelerator="cpu",
        max_epochs=args.epochs,
        default_root_dir=directory,
    )

    if args.load:
        pinn = PINN.load_from_checkpoint(
            checkpoint_path=args.load, problem=stokes_problem, model=model
        )
        plotter = Plotter()
        plotter.plot(pinn, components="ux")
        plotter.plot(pinn, components="uy")
        plotter.plot(pinn, components="p")
    else:
        trainer.train()
