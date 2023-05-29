# import torch
# from pina.problem import Problem
# from pina.segment import Segment
# from pina.cube import Cube
# from pina.problem2d import Problem2D

# xmin, xmax, ymin, ymax = -1, 1, -1, 1

# class EllipticOptimalControl(Problem2D):

#     def __init__(self, alpha=1):

#         def term1(input_, output_):
#             grad_p = self.grad(output_.extract(['p']), input_)
#             gradgrad_p_x1 = self.grad(grad_p.extract(['x1']), input_)
#             gradgrad_p_x2 = self.grad(grad_p.extract(['x2']), input_)
#             yd = 2.0
#             return output_.extract(['y']) - yd - (gradgrad_p_x1.extract(['x1']) + gradgrad_p_x2.extract(['x2']))

#         def term2(input_, output_):
#             grad_y = self.grad(output_.extract(['y']), input_)
#             gradgrad_y_x1 = self.grad(grad_y.extract(['x1']), input_)
#             gradgrad_y_x2 = self.grad(grad_y.extract(['x2']), input_)
#             return - (gradgrad_y_x1.extract(['x1']) + gradgrad_y_x2.extract(['x2'])) - output_.extract(['u'])

#         def term3(input_, output_):
#             return output_.extract(['p']) - output_.extract(['u'])*alpha


#         def nil_dirichlet(input_, output_):
#             y_value = 0.0
#             p_value = 0.0
#             return torch.abs(output_.extract(['y']) - y_value) + torch.abs(output_.extract(['p']) - p_value)

#         self.conditions = {
#             'gamma1': {'location': Segment((xmin, ymin), (xmax, ymin)), 'func': nil_dirichlet},
#             'gamma2': {'location': Segment((xmax, ymin), (xmax, ymax)), 'func': nil_dirichlet},
#             'gamma3': {'location': Segment((xmax, ymax), (xmin, ymax)), 'func': nil_dirichlet},
#             'gamma4': {'location': Segment((xmin, ymax), (xmin, ymin)), 'func': nil_dirichlet},
#             'D1': {'location': Cube([[xmin, xmax], [ymin, ymax]]), 'func': [term1, term2, term3]},
#         }

#         self.input_variables = ['x1', 'x2']
#         self.output_variables = ['u', 'p', 'y']
#         self.spatial_domain = Cube([[xmin, xmax], [xmin, xmax]])

raise NotImplementedError('not available problem at the moment...')