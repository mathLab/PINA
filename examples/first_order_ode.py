import argparse
import torch

from torch.nn import ReLU, Tanh, Softplus, PReLU

from pina.problem import SpatialProblem, ParametricProblem
from pina.operators import nabla, grad, div
from pina.model import FeedForward, DeepONet
from pina import Condition, Span, LabelTensor, Plotter, PINN

import matplotlib
matplotlib.use('Qt5Agg')


class FirstOrderODE(SpatialProblem):

    x_rng = [0, 5]
    output_variables = ['y']
    spatial_domain = Span({'x': x_rng})

    def ode(input_, output_):
        y = output_
        x = input_
        return grad(y, x) + y - x

    def fixed(input_, output_):
        exp_value = 1.
        return output_ - exp_value

    def solution(self, input_):
        x = input_
        return x - 1.0 + 2*torch.exp(-x)

    conditions = {
        'bc': Condition(Span({'x': x_rng[0]}), fixed),
        'dd': Condition(Span({'x': x_rng}), ode),
    }
    truth_solution = solution


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run PINA")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-s", "-save", action="store_true")
    group.add_argument("-l", "-load", action="store_true")
    parser.add_argument("id_run", help="number of run", type=int)
    args = parser.parse_args()

    problem = FirstOrderODE()
    model = FeedForward(
        layers=[4]*2,
        output_variables=problem.output_variables,
        input_variables=problem.input_variables,
        func=Softplus,
    )

    pinn = PINN(problem, model, lr=0.03, error_norm='mse', regularizer=0)

    if args.s:

        pinn.span_pts(
            {'variables': ['x'], 'mode': 'grid', 'n': 1}, locations=['bc'])
        pinn.span_pts(
            {'variables': ['x'], 'mode': 'grid', 'n': 30}, locations=['dd'])
        Plotter().plot_samples(pinn, ['x'])
        pinn.train(1200, 50)
        pinn.save_state('pina.ode')

    else:
        pinn.load_state('pina.ode')
        plotter = Plotter()
        plotter.plot(pinn, components=['y'])
