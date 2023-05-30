# import numpy as np
# import torch
# from pina.problem import Problem
# from pina.segment import Segment
# from pina.cube import Cube
# from pina.problem2d import Problem2D

# xmin, xmax, ymin, ymax = -1, 1, -1, 1

# class ParametricEllipticOptimalControl(Problem2D):

#     def __init__(self, alpha=1):

#         def term1(input_, param_, output_):
#             grad_p = self.grad(output_['p'], input_)
#             gradgrad_p_x1 = self.grad(grad_p['x1'], input_)
#             gradgrad_p_x2 = self.grad(grad_p['x2'], input_)
#             return output_['y'] - param_ - (gradgrad_p_x1['x1'] + gradgrad_p_x2['x2'])

#         def term2(input_, param_, output_):
#             grad_y = self.grad(output_['y'], input_)
#             gradgrad_y_x1 = self.grad(grad_y['x1'], input_)
#             gradgrad_y_x2 = self.grad(grad_y['x2'], input_)
#             return - (gradgrad_y_x1['x1'] + gradgrad_y_x2['x2']) - output_['u_param']

#         def term3(input_, param_, output_):
#             return output_['p'] - output_['u_param']*alpha


#         def term(input_, param_, output_):
#             return term1( input_, param_, output_)  +term2( input_, param_, output_) + term3( input_, param_, output_)

#         def nil_dirichlet(input_, param_, output_):
#             y_value = 0.0
#             p_value = 0.0
#             return torch.abs(output_['y'] - y_value) + torch.abs(output_['p'] - p_value)

#         self.conditions = {
#             'gamma1': {'location': Segment((xmin, ymin), (xmax, ymin)), 'func': nil_dirichlet},
#             'gamma2': {'location': Segment((xmax, ymin), (xmax, ymax)), 'func': nil_dirichlet},
#             'gamma3': {'location': Segment((xmax, ymax), (xmin, ymax)), 'func': nil_dirichlet},
#             'gamma4': {'location': Segment((xmin, ymax), (xmin, ymin)), 'func': nil_dirichlet},
#             'D1': {'location': Cube([[xmin, xmax], [ymin, ymax]]), 'func': term},
#             #'D2': {'location': Cube([[0, 1], [0, 1]]), 'func': term2},
#             #'D3': {'location': Cube([[0, 1], [0, 1]]), 'func': term3}
#         }

#         self.input_variables = ['x1', 'x2']
#         self.output_variables = ['u', 'p', 'y']
#         self.parameters = ['mu']
#         self.spatial_domain = Cube([[xmin, xmax], [xmin, xmax]])
#         self.parameter_domain = np.array([[0.5, 3]])
raise NotImplementedError('not available problem at the moment...')
