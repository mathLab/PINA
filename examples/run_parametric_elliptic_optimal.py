import argparse
import numpy as np
import torch
from torch.nn import Softplus

from pina import LabelTensor
from pina.solvers import PINN
from pina.model import MultiFeedForward, FeedForward
from pina.plotter import Plotter
from pina.trainer import Trainer
from problems.parametric_elliptic_optimal_control import (
    ParametricEllipticOptimalControl)


class myFeature(torch.nn.Module):
    """
    Feature: sin(x)
    """

    def __init__(self):
        super(myFeature, self).__init__()

    def forward(self, x):
        t = (-x.extract(['x1'])**2+1) * (-x.extract(['x2'])**2+1)
        return LabelTensor(t, ['k0'])


class PIArch(MultiFeedForward):

    def __init__(self, dff_dict):
        super().__init__(dff_dict)

    def forward(self, x):
        out = self.uy(x)
        out.labels = ['u', 'y']
        z = LabelTensor(
            (out.extract(['u']) * x.extract(['alpha'])), ['z'])
        return out.append(z)

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
    args = parser.parse_args()

    # create problem and discretise domain
    opc = ParametricEllipticOptimalControl()
    opc.discretise_domain(n= 900, mode='random', variables=['x1', 'x2'], locations=['D'])
    opc.discretise_domain(n= 5,   mode='random', variables=['mu', 'alpha'], locations=['D'])
    opc.discretise_domain(n= 200, mode='random', variables=['x1', 'x2'], locations=['gamma1', 'gamma2', 'gamma3', 'gamma4'])
    opc.discretise_domain(n= 5,   mode='random', variables=['mu', 'alpha'], locations=['gamma1', 'gamma2', 'gamma3', 'gamma4'])

    # create model
    model = PIArch(
        {
            'uy': {
                'input_dimensions': 4 + len(feat),
                'output_dimensions': 2,
                'layers': [40, 40, 20],
                'func': Softplus,
            },
        }
    )

    # create PINN
    pinn = PINN(problem=opc, model=model, optimizer_kwargs={'lr' : 0.002}, extra_features=feat)

    # create trainer
    directory = 'pina.parametric_optimal_control_{}'.format(bool(args.features))
    trainer = Trainer(solver=pinn, accelerator='cpu', max_epochs=args.epochs, default_root_dir=directory)


    if args.load:
        pinn = PINN.load_from_checkpoint(checkpoint_path=args.load, problem=opc, model=model, extra_features=feat)
        plotter = Plotter()
        plotter.plot(pinn, fixed_variables={'mu' : 3 , 'alpha' : 1}, components='y')
        plotter.plot(pinn, fixed_variables={'mu' : 3 , 'alpha' : 1}, components='u')
        plotter.plot(pinn, fixed_variables={'mu' : 3 , 'alpha' : 1}, components='z')
    else:
        trainer.train()
