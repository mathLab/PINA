import argparse
import torch
from pina import Plotter, LabelTensor, Trainer
from pina.solvers import PINN
from pina.model import DeepONet, FeedForward
from problems.parametric_poisson import ParametricPoisson


class myFeature(torch.nn.Module):
    """
    """
    def __init__(self):
        super(myFeature, self).__init__()

    def forward(self, x):
        t = (
            torch.exp(
                - 2*(x.extract(['x']) - x.extract(['mu1']))**2
                - 2*(x.extract(['y']) - x.extract(['mu2']))**2
            )
        )
        return LabelTensor(t, ['k0'])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run PINA")
    parser.add_argument("--load", help="directory to save or load file", type=str)
    parser.add_argument("--epochs", help="extra features", type=int, default=1000)
    args = parser.parse_args()


    # create problem and discretise domain
    ppoisson_problem = ParametricPoisson()
    ppoisson_problem.discretise_domain(n=100, mode='random', variables = ['x', 'y'], locations=['D'])
    ppoisson_problem.discretise_domain(n=100, mode='random', variables = ['mu1', 'mu2'], locations=['D'])
    ppoisson_problem.discretise_domain(n=20, mode='random', variables = ['x', 'y'], locations=['gamma1', 'gamma2', 'gamma3', 'gamma4'])
    ppoisson_problem.discretise_domain(n=5, mode='random', variables = ['mu1', 'mu2'], locations=['gamma1', 'gamma2', 'gamma3', 'gamma4'])

    # create model
    trunck = FeedForward(
        layers=[40, 40],
        output_dimensions=1,
        input_dimensions=2,
        func=torch.nn.ReLU
    )
    branch = FeedForward(
        layers=[40, 40],
        output_dimensions=1,
        input_dimensions=2,
        func=torch.nn.ReLU
    )
    model = DeepONet(branch_net=branch,
                     trunk_net=trunck,
                     input_indeces_branch_net=['x', 'y'],
                     input_indeces_trunk_net=['mu1', 'mu2'])

    # create solver
    pinn = PINN(
        problem=ppoisson_problem,
        model=model,
        optimizer_kwargs={'lr' : 0.006}
    )

    # create trainer
    directory = 'pina.parametric_poisson_deeponet'
    trainer = Trainer(solver=pinn, accelerator='cpu', max_epochs=args.epochs, default_root_dir=directory)


    if args.load:
        pinn = PINN.load_from_checkpoint(checkpoint_path=args.load, problem=ppoisson_problem, model=model)
        plotter = Plotter()
        plotter.plot(pinn, fixed_variables={'mu1': 1, 'mu2': -1})
    else:
        trainer.train()
