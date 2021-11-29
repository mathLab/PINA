import numpy as np
import torch
import argparse
from pina.pinn import PINN
from pina.ppinn import ParametricPINN as pPINN
from pina.label_tensor import LabelTensor
from torch.nn import ReLU, Tanh, Softplus
from pina.adaptive_functions.adaptive_softplus import AdaptiveSoftplus
from problems.parametric_elliptic_optimal_control_alpha_variable import ParametricEllipticOptimalControl
from pina.multi_deep_feed_forward import MultiDeepFeedForward
from pina.deep_feed_forward import DeepFeedForward

alpha = 1

class myFeature(torch.nn.Module):
    """
    Feature: sin(x)
    """

    def __init__(self):
        super(myFeature, self).__init__()

    def forward(self, x):
        return (-x[:, 0]**2+1) * (-x[:, 1]**2+1)


class CustomMultiDFF(MultiDeepFeedForward):

    def __init__(self, dff_dict):
        super().__init__(dff_dict)

    def forward(self, x):
        out = self.uu(x)
        p = LabelTensor((out['u_param'] * x[:, 3]).reshape(-1, 1), ['p'])
        a = LabelTensor.hstack([out, p])
        return a

model = CustomMultiDFF(
        {
            'uu': {
                'input_variables': ['x1', 'x2', 'mu', 'alpha'],
                'output_variables': ['u_param', 'y'],
                'layers': [40, 40, 20],
                'func': Softplus,
                'extra_features': [myFeature()],
            },
            # 'u_param': {
            #     'input_variables': ['u', 'mu'],
            #     'output_variables': ['u_param'],
            #     'layers': [],
            #     'func': None
            # },
            # 'p': {
            # 'input_variables': ['u'],
            # 'output_variables': ['p'],
            # 'layers': [10],
            # 'func': None
            # },
        }
)


opc = ParametricEllipticOptimalControl(alpha)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run PINA")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-s", "-save", action="store_true")
    group.add_argument("-l", "-load", action="store_true")
    args = parser.parse_args()

    # model = DeepFeedForward(
    #     layers=[40, 40, 20],
    #     output_variables=['u_param', 'y', 'p'],
    #     input_variables=opc.input_variables+['mu', 'alpha'],
    #     func=Softplus,
    #     extra_features=[myFeature()]
    # )


    pinn = pPINN(
        opc,
        model,
        lr=0.002,
        error_norm='mse',
        regularizer=1e-8,
        lr_accelerate=None)

    if args.s:

        pinn.span_pts(30, 'grid', ['D1'])
        pinn.span_pts(50, 'grid', ['gamma1', 'gamma2', 'gamma3', 'gamma4'])
        pinn.train(10000, 20)
        # with open('ocp_wrong_history.txt', 'w') as file_:
        #     for i, losses in enumerate(pinn.history):
        #         file_.write('{} {}\n'.format(i, sum(losses).item()))

        pinn.save_state('pina.ocp')

    else:
        pinn.load_state('working.pina.ocp')
        pinn.load_state('pina.ocp')

        import matplotlib
        matplotlib.use('GTK3Agg')
        import matplotlib.pyplot as plt

        # res = 64
        # param = torch.tensor([[3., 1]])
        # pts_container = []
        # for mn, mx in [[-1, 1], [-1, 1]]:
        #     pts_container.append(np.linspace(mn, mx, res))
        # grids_container = np.meshgrid(*pts_container)
        # unrolled_pts = torch.tensor([t.flatten() for t in grids_container]).T
        # unrolled_pts = torch.cat([unrolled_pts, param.double().repeat(unrolled_pts.shape[0], 1).reshape(-1, 2)], axis=1)

        # unrolled_pts = LabelTensor(unrolled_pts, ['x1', 'x2', 'mu', 'alpha'])
        # Z_pred = pinn.model(unrolled_pts.tensor)
        # print(Z_pred.tensor.shape)

        # plt.subplot(2, 3, 1)
        # plt.pcolor(Z_pred['y'].reshape(res, res).detach())
        # plt.colorbar()
        # plt.subplot(2, 3, 2)
        # plt.pcolor(Z_pred['u_param'].reshape(res, res).detach())
        # plt.colorbar()
        # plt.subplot(2, 3, 3)
        # plt.pcolor(Z_pred['p'].reshape(res, res).detach())
        # plt.colorbar()
        # with open('ocp_mu3_a1_plot.txt', 'w') as f_:
        #     f_.write('x y u p ys\n')
        #     for (x, y), tru, pre, e in zip(unrolled_pts[:, :2],
        #                                    Z_pred['u_param'].reshape(-1, 1),
        #                                    Z_pred['p'].reshape(-1, 1),
        #                                    Z_pred['y'].reshape(-1, 1),
        #                                    ):
        #         f_.write('{} {} {} {} {}\n'.format(x.item(), y.item(), tru.item(), pre.item(), e.item()))


        # param = torch.tensor([[3.0, 0.01]])
        # unrolled_pts = torch.tensor([t.flatten() for t in grids_container]).T
        # unrolled_pts = torch.cat([unrolled_pts, param.double().repeat(unrolled_pts.shape[0], 1).reshape(-1, 2)], axis=1)
        # unrolled_pts = LabelTensor(unrolled_pts, ['x1', 'x2', 'mu', 'alpha'])
        # Z_pred = pinn.model(unrolled_pts.tensor)

        # plt.subplot(2, 3, 4)
        # plt.pcolor(Z_pred['y'].reshape(res, res).detach())
        # plt.colorbar()
        # plt.subplot(2, 3, 5)
        # plt.pcolor(Z_pred['u_param'].reshape(res, res).detach())
        # plt.colorbar()
        # plt.subplot(2, 3, 6)
        # plt.pcolor(Z_pred['p'].reshape(res, res).detach())
        # plt.colorbar()

        # plt.show()
        # with open('ocp_mu3_a0.01_plot.txt', 'w') as f_:
        #     f_.write('x y u p ys\n')
        #     for (x, y), tru, pre, e in zip(unrolled_pts[:, :2],
        #                                    Z_pred['u_param'].reshape(-1, 1),
        #                                    Z_pred['p'].reshape(-1, 1),
        #                                    Z_pred['y'].reshape(-1, 1),
        #                                    ):
        #         f_.write('{} {} {} {} {}\n'.format(x.item(), y.item(), tru.item(), pre.item(), e.item()))




        y = {}
        u = {}
        for alpha in [0.01, 0.1, 1]:
            y[alpha] = []
            u[alpha] = []
            for p in np.linspace(0.5, 3, 32):
                a = pinn.model(LabelTensor(torch.tensor([[0, 0, p, alpha]]).double(), ['x1', 'x2', 'mu', 'alpha']).tensor)
                y[alpha].append(a['y'].detach().numpy()[0])
                u[alpha].append(a['u_param'].detach().numpy()[0])



        plt.plot(np.linspace(0.5, 3, 32), u[1], label='u')
        plt.plot(np.linspace(0.5, 3, 32), u[0.01], label='u')
        plt.plot(np.linspace(0.5, 3, 32), u[0.1], label='u')
        plt.plot([1, 2, 3], [0.28, 0.56, 0.85], 'o', label='Truth values')
        plt.legend()
        plt.show()
        print(y[1])
        print(y[0.1])
        print(y[0.01])
        with open('elliptic_param_y.txt', 'w') as f_:
            f_.write('mu 1 01 001\n')
            for mu, y1, y01, y001 in zip(np.linspace(0.5, 3, 32), y[1], y[0.1], y[0.01]):
                f_.write('{} {} {} {}\n'.format(mu, y1, y01, y001))

        with open('elliptic_param_u.txt', 'w') as f_:
            f_.write('mu 1 01 001\n')
            for mu, y1, y01, y001 in zip(np.linspace(0.5, 3, 32), u[1], u[0.1], u[0.01]):
                f_.write('{} {} {} {}\n'.format(mu, y1, y01, y001))


        plt.plot(np.linspace(0.5, 3, 32), y, label='y')
        plt.plot([1, 2, 3], [0.062, 0.12, 0.19], 'o', label='Truth values')
        plt.legend()
        plt.show()


