import sys
import numpy as np
import torch
import argparse
from pina.pinn import PINN
from pina.ppinn import ParametricPINN as pPINN
from pina.label_tensor import LabelTensor
from torch.nn import ReLU, Tanh, Softplus
from problems.burgers import Burgers1D
from pina.deep_feed_forward import DeepFeedForward

from pina.adaptive_functions import AdaptiveSin, AdaptiveCos, AdaptiveTanh


class myFeature(torch.nn.Module):
    """
    Feature: sin(x)
    """

    def __init__(self, idx):

        super(myFeature, self).__init__()
        self.idx = idx

    def forward(self, x):

        return torch.sin(torch.pi * x[:, self.idx])

class myExp(torch.nn.Module):
    def __init__(self, idx):

        super().__init__()
        self.idx = idx

    def forward(self, x):

        return torch.exp(x[:, self.idx])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run PINA")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-s", "-save", action="store_true")
    group.add_argument("-l", "-load", action="store_true")
    parser.add_argument("id_run", help="number of run", type=int)
    parser.add_argument("features", help="extra features", type=int)
    args = parser.parse_args()

    feat = [myFeature(0)] if args.features else []

    burgers_problem = Burgers1D()
    model = DeepFeedForward(
        layers=[20, 10, 5],
        #layers=[8, 4, 2],
        #layers=[16, 8, 4, 4],
        #layers=[20, 4, 4, 4],
        output_variables=burgers_problem.output_variables,
        input_variables=burgers_problem.input_variables,
        func=Tanh,
        extra_features=feat
    )

    pinn = PINN(
        burgers_problem,
        model,
        lr=0.006,
        error_norm='mse',
        regularizer=0,
        lr_accelerate=None)

    if args.s:

        pinn.span_pts(8000, 'latin', ['D'])
        pinn.span_pts(50, 'random', ['gamma1', 'gamma2', 'initia'])
        #pinn.plot_pts()
        pinn.train(10000, 1000)
        #with open('burgers_history_{}_{}.txt'.format(args.id_run, args.features), 'w') as file_:
        #    for i, losses in enumerate(pinn.history):
        #        file_.write('{} {}\n'.format(i, sum(losses).item()))
        pinn.save_state('pina.burger.{}.{}'.format(args.id_run, args.features))

    else:
        pinn.load_state('pina.burger.{}.{}'.format(args.id_run, args.features))
        #pinn.plot(256,filename='pina.burger.{}.{}.jpg'.format(args.id_run, args.features))


        print(pinn.history)
        with open('burgers_history_{}_{}.txt'.format(args.id_run, args.features), 'w') as file_:
            for i, losses in enumerate(pinn.history):
                print(losses)
                file_.write('{} {}\n'.format(i, sum(losses)))
        import scipy
        data = scipy.io.loadmat('Data/burgers_shock.mat')
        data_solution = {'grid': np.meshgrid(data['x'], data['t']), 'grid_solution': data['usol'].T}
        import matplotlib
        matplotlib.use('Qt5Agg')
        import matplotlib.pyplot as plt

        t =75
        for t in [25, 50, 75]:
            input = torch.cat([
                    torch.linspace(-1, 1, 256).reshape(-1, 1),
                    torch.ones(size=(256, 1)) * t /100],
                    axis=1).double()
            output = pinn.model(input)
            fout = 'pina.burgers.{}.{}.t{}.dat'.format(args.id_run, args.features, t)
            with open(fout, 'w') as f_:
                f_.write('x utruth upinn\n')
                for x, utruth, upinn in zip(data['x'], data['usol'][:, t], output.tensor.detach()):
                    f_.write('{} {} {}\n'.format(x[0], utruth, upinn.item()))
            plt.plot(data['usol'][:, t], label='truth')
            plt.plot(output.tensor.detach(), 'x', label='pinn')
            plt.legend()
            plt.show()
