import torch
from torch.nn import Softplus
from termcolor import colored
import argparse
from pina.problem import SpatialProblem, ParametricProblem, InverseProblem
from pina.operators import laplacian
from pina.model import FeedForward
from pina.equation import Equation, FixedValue, ParametricEquation
from pina import (Condition, CartesianDomain, PINN, LabelTensor,
        Plotter, Location, Trainer)

### initialize the Poisson spatial problem
x_min = -2
x_max = 2
y_min = -2
y_max = 2
data_output = torch.load('data/pinn_solution_0.5_0.5')
data_input = torch.load('data/pts_0.5_0.5')

class Poisson(SpatialProblem, InverseProblem):
    output_variables = ['u']
    spatial_domain = CartesianDomain({'x': [x_min, x_max], 'y': [y_min, y_max]})
    inferred_parameters = torch.nn.Parameter(torch.rand(2,
        requires_grad=True))
    inferred_domain = CartesianDomain({'mu1': [-1, 1], 'mu2': [-1, 1]})

    def laplace_equation(input_, output_, inferred_parameters):
        force_term = torch.exp(
                - 2*(input_.extract(['x']) - inferred_parameters[0])**2
                - 2*(input_.extract(['y']) - inferred_parameters[1])**2)
        delta_u = laplacian(output_, input_, components=['u'], d=['x', 'y'])
        return delta_u - force_term

    conditions = {
        'gamma1': Condition(location=CartesianDomain({'x': [x_min, x_max],
            'y':  y_max}),
            equation=FixedValue(0.0, components=['u'])),
        'gamma2': Condition(location=CartesianDomain({'x': [x_min, x_max], 'y': y_min
            }),
            equation=FixedValue(0.0, components=['u'])),
        'gamma3': Condition(location=CartesianDomain({'x':  x_max, 'y': [y_min, y_max]
            }),
            equation=FixedValue(0.0, components=['u'])),
        'gamma4': Condition(location=CartesianDomain({'x': x_min, 'y': [y_min, y_max]
            }),
            equation=FixedValue(0.0, components=['u'])),
        'D': Condition(location=CartesianDomain({'x': [x_min, x_max], 'y': [y_min, y_max]
            }),
        equation=ParametricEquation(laplace_equation)),
        'data': Condition(input_points=data_input.extract(['x', 'y']), output_points=data_output)
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Poisson problem')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-s', '-save', action='store_true')
    group.add_argument('-l', '-load', action='store_true')
    args = parser.parse_args()

    problem = Poisson()
    model = FeedForward(
        layers=[10, 10, 10],
        func=Softplus,
        output_dimensions=len(problem.output_variables),
        input_dimensions=len(problem.input_variables)
        )

    problem.discretise_domain(30, 'grid', locations=['D'], variables=['x', 'y'])
    problem.discretise_domain(30, 'grid', locations=['gamma1', 'gamma2',
        'gamma3', 'gamma4'], variables=['x', 'y'])

    tmp_dir = "tmp_poisson_inverse"
    max_epochs = 3000

    ### train the problem with PINN
    if args.s:
        pinn = PINN(problem, model, optimizer_kwargs={'lr':0.01})
        trainer = Trainer(solver=pinn, accelerator='cpu', max_epochs=max_epochs,
                default_root_dir=tmp_dir)
        trainer.train()

    if args.l:
        new_pinn = PINN.load_from_checkpoint(
            '{}/lightning_logs/version_0/checkpoints/epoch={}-step={}.ckpt'.format(
                tmp_dir, max_epochs-1, max_epochs),
            problem=problem, model=model)
        plotter = Plotter()

        plotter.plot(new_pinn)

