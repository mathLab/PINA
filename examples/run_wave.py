""" Run PINA on Burgers equation. """

import argparse
import torch
from torch.nn import Softplus

from pina import LabelTensor
from pina.model import FeedForward
from pina.solvers import PINN
from pina.plotter import Plotter
from pina.trainer import Trainer
from problems.wave import Wave

class HardMLP(torch.nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        self.layers = FeedForward(**kwargs)
        
    # here in the foward we implement the hard constraints
    def forward(self, x):
        hard_space = x.extract(['x'])*(1-x.extract(['x']))*x.extract(['y'])*(1-x.extract(['y']))
        hard_t = torch.sin(torch.pi*x.extract(['x'])) * torch.sin(torch.pi*x.extract(['y'])) * torch.cos(torch.sqrt(torch.tensor(2.))*torch.pi*x.extract(['t']))
        return hard_space * self.layers(x) * x.extract(['t']) + hard_t

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run PINA")
    parser.add_argument("--load", help="directory to save or load file", type=str)
    parser.add_argument("--epochs", help="extra features", type=int, default=1000)
    args = parser.parse_args()


    # create problem and discretise domain
    wave_problem = Wave()
    wave_problem.discretise_domain(1000, 'random', locations=['D', 't0', 'gamma1', 'gamma2', 'gamma3', 'gamma4'])

    # create model
    model = HardMLP(
        layers=[40, 40, 40],
        output_dimensions=len(wave_problem.output_variables),
        input_dimensions=len(wave_problem.input_variables),
        func=Softplus
    )

    # create solver
    pinn = PINN(
        problem=wave_problem,
        model=model,
        optimizer_kwargs={'lr' : 0.006}
    )

    # create trainer
    directory = 'pina.wave'
    trainer = Trainer(solver=pinn, accelerator='cpu', max_epochs=args.epochs, default_root_dir=directory)


    if args.load:
        pinn = PINN.load_from_checkpoint(checkpoint_path=args.load, problem=wave_problem, model=model)
        plotter = Plotter()
        plotter.plot(pinn)
    else:
        trainer.train()
