# import argparse
# import numpy as np
# import torch
# from torch.nn import Softplus

# from pina import PINN, LabelTensor, Plotter
# from pina.model import MultiFeedForward
# from problems.parametric_elliptic_optimal_control_alpha_variable import (
#     ParametricEllipticOptimalControl)


# class myFeature(torch.nn.Module):
#     """
#     Feature: sin(x)
#     """

#     def __init__(self):
#         super(myFeature, self).__init__()

#     def forward(self, x):
#         t = (-x.extract(['x1'])**2+1) * (-x.extract(['x2'])**2+1)
#         return LabelTensor(t, ['k0'])


# class CustomMultiDFF(MultiFeedForward):

#     def __init__(self, dff_dict):
#         super().__init__(dff_dict)

#     def forward(self, x):
#         out = self.uu(x)
#         p = LabelTensor((out.extract(['u_param']) * x.extract(['alpha'])), ['p'])
#         return out.append(p)


# if __name__ == "__main__":

#     parser = argparse.ArgumentParser(description="Run PINA")
#     group = parser.add_mutually_exclusive_group(required=True)
#     group.add_argument("-s", "-save", action="store_true")
#     group.add_argument("-l", "-load", action="store_true")
#     args = parser.parse_args()

#     opc = ParametricEllipticOptimalControl()
#     model = CustomMultiDFF(
#         {
#             'uu': {
#                 'input_variables': ['x1', 'x2', 'mu', 'alpha'],
#                 'output_variables': ['u_param', 'y'],
#                 'layers': [40, 40, 20],
#                 'func': Softplus,
#                 'extra_features': [myFeature()],
#             },
#         }
#     )

#     pinn = PINN(
#         opc,
#         model,
#         lr=0.002,
#         error_norm='mse',
#         regularizer=1e-8)

#     if args.s:

#         pinn.span_pts(
#             {'variables': ['x1', 'x2'], 'mode': 'random', 'n': 100},
#             {'variables': ['mu', 'alpha'], 'mode': 'grid', 'n': 5},
#             locations=['D'])
#         pinn.span_pts(
#             {'variables': ['x1', 'x2'], 'mode': 'grid', 'n': 20},
#             {'variables': ['mu', 'alpha'], 'mode': 'grid', 'n': 5},
#             locations=['gamma1', 'gamma2', 'gamma3', 'gamma4'])

#         pinn.train(1000, 20)
#         pinn.save_state('pina.ocp')

#     else:
#         pinn.load_state('pina.ocp')
#         plotter = Plotter()
#         plotter.plot(pinn, components='y', fixed_variables={'alpha': 0.01, 'mu': 1.0})
#         plotter.plot(pinn, components='u_param', fixed_variables={'alpha': 0.01, 'mu': 1.0})
#         plotter.plot(pinn, components='p', fixed_variables={'alpha': 0.01, 'mu': 1.0})
raise NotImplementedError('not available problem at the moment...')