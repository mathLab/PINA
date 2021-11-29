import sys
import numpy as np
import torch
import argparse
from pina.pinn import PINN
from pina.ppinn import ParametricPINN as pPINN
from pina.label_tensor import LabelTensor
from torch.nn import ReLU, Tanh, Softplus
from problems.parametric_poisson import ParametricPoisson2DProblem as Poisson2D
from pina.deep_feed_forward import DeepFeedForward

from pina.adaptive_functions import AdaptiveSin, AdaptiveCos, AdaptiveTanh


class myFeature(torch.nn.Module):
    """
    Feature: sin(x)
    """

    def __init__(self):
        super(myFeature, self).__init__()

    def forward(self, x):
        return torch.exp(- 2*(x[:, 0] - x[:, 2])**2 - 2*(x[:, 1] - x[:, 3])**2)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run PINA")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-s", "-save", action="store_true")
    group.add_argument("-l", "-load", action="store_true")
    parser.add_argument("id_run", help="number of run", type=int)
    parser.add_argument("features", help="extra features", type=int)
    args = parser.parse_args()

    feat = [myFeature()] if args.features else []

    poisson_problem = Poisson2D()
    model = DeepFeedForward(
        layers=[200, 40, 10],
        output_variables=poisson_problem.output_variables,
        input_variables=poisson_problem.input_variables+['mu1', 'mu2'],
        func=Softplus,
        extra_features=feat
    )

    pinn = pPINN(
        poisson_problem,
        model,
        lr=0.0006,
        regularizer=1e-6,
        lr_accelerate=None)

    if args.s:

        pinn.span_pts(30, 'chebyshev', ['D'])
        pinn.span_pts(50, 'grid', ['gamma1', 'gamma2', 'gamma3', 'gamma4'])
        #pinn.plot_pts()
        pinn.train(10000, 10)
        pinn.save_state('pina.poisson_param')

    else:
        pinn.load_state('pina.poisson_param')
        #pinn.plot(40, torch.tensor([-0.8, -0.8]))
        #pinn.plot(40, torch.tensor([ 0.8,  0.8]))

        from smithers.io import VTUHandler
        from scipy.interpolate import griddata
        import matplotlib
        matplotlib.use('GTK3Agg')
        import matplotlib.pyplot as plt

        res = 64
        fname_minus = 'Poisson_param_08minus000000.vtu'
        param = torch.tensor([-0.8, -0.8])
        pts_container = []
        for mn, mx in [[-1, 1], [-1, 1]]:
            pts_container.append(np.linspace(mn, mx, res))
        grids_container = np.meshgrid(*pts_container)
        unrolled_pts = torch.tensor([t.flatten() for t in grids_container]).T
        unrolled_pts = torch.cat([unrolled_pts, param.double().repeat(unrolled_pts.shape[0]).reshape(-1, 2)], axis=1)

        #unrolled_pts.to(dtype=self.dtype)
        unrolled_pts = LabelTensor(unrolled_pts, ['x1', 'x2', 'mu1', 'mu2'])

        Z_pred = pinn.model(unrolled_pts.tensor)
        data = VTUHandler.read(fname_minus)


        print(data['points'][:, :2].shape)
        print(data['point_data']['f_16'].shape)
        print(grids_container[0].shape)
        print(grids_container[1].shape)
        Z_truth = griddata(data['points'][:, :2], data['point_data']['f_16'], (grids_container[0], grids_container[1]))


        err = np.abs(Z_truth + Z_pred.tensor.reshape(res, res).detach().numpy())

        plt.subplot(1, 3, 1)
        plt.pcolor(-Z_pred.tensor.reshape(res, res).detach())
        plt.colorbar()
        plt.subplot(1, 3, 2)
        plt.pcolor(Z_truth)
        plt.colorbar()
        plt.subplot(1, 3, 3)
        plt.pcolor(err, vmin=0, vmax=0.009)
        plt.colorbar()
        plt.show()

        print(unrolled_pts.tensor.shape)
        with open('parpoisson_minus_plot.txt', 'w') as f_:
            f_.write('x y truth pred e\n')
            for (x, y), tru, pre, e in zip(unrolled_pts[:, :2], Z_truth.reshape(-1, 1), -Z_pred.tensor.reshape(-1, 1), err.reshape(-1, 1)):
                f_.write('{} {} {} {} {}\n'.format(x.item(), y.item(), tru.item(), pre.item(), e.item()))

        fname_plus = 'Poisson_param_08plus000000.vtu'
        param = torch.tensor([0.8, 0.8])
        pts_container = []
        for mn, mx in [[-1, 1], [-1, 1]]:
            pts_container.append(np.linspace(mn, mx, res))
        grids_container = np.meshgrid(*pts_container)
        unrolled_pts = torch.tensor([t.flatten() for t in grids_container]).T
        unrolled_pts = torch.cat([unrolled_pts, param.double().repeat(unrolled_pts.shape[0]).reshape(-1, 2)], axis=1)

        #unrolled_pts.to(dtype=self.dtype)
        unrolled_pts = LabelTensor(unrolled_pts, ['x1', 'x2', 'mu1', 'mu2'])

        Z_pred = pinn.model(unrolled_pts.tensor)
        data = VTUHandler.read(fname_plus)


        print(data['points'][:, :2].shape)
        print(data['point_data']['f_16'].shape)
        print(grids_container[0].shape)
        print(grids_container[1].shape)
        Z_truth = griddata(data['points'][:, :2], data['point_data']['f_16'], (grids_container[0], grids_container[1]))


        err = np.abs(Z_truth + Z_pred.tensor.reshape(res, res).detach().numpy())

        plt.subplot(1, 3, 1)
        plt.pcolor(-Z_pred.tensor.reshape(res, res).detach())
        plt.colorbar()
        plt.subplot(1, 3, 2)
        plt.pcolor(Z_truth)
        plt.colorbar()
        plt.subplot(1, 3, 3)
        print('gggggg')
        plt.pcolor(err, vmin=0, vmax=0.001)
        plt.colorbar()
        plt.show()
        with open('parpoisson_plus_plot.txt', 'w') as f_:
            f_.write('x y truth pred e\n')
            for (x, y), tru, pre, e in zip(unrolled_pts[:, :2], Z_truth.reshape(-1, 1), -Z_pred.tensor.reshape(-1, 1), err.reshape(-1, 1)):
                f_.write('{} {} {} {} {}\n'.format(x.item(), y.item(), tru.item(), pre.item(), e.item()))


