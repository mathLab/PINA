import torch
from torch.nn import Softplus
from termcolor import colored
import argparse
from lightning.pytorch.callbacks import Callback
from pina.problem import SpatialProblem, ParametricProblem, InverseProblem
from pina.operators import laplacian
from pina.model import FeedForward
from pina.equation import Equation, FixedValue, ParametricEquation
from pina import (Condition, CartesianDomain, PINN, LabelTensor,
        Plotter, Location, Trainer)

### Define ranges of variables
x_min = -2
x_max = 2
y_min = -2
y_max = 2
### Import data (input and output that we consider as reference)
data_output = torch.load('data/pinn_solution_0.5_0.5')
data_input = torch.load('data/pts_0.5_0.5')

### initialize the Poisson spatial problem
class Poisson(SpatialProblem, InverseProblem):
    '''
    Problem definition for the Poisson equation.
    '''
    output_variables = ['u']
    spatial_domain = CartesianDomain({'x': [x_min, x_max], 'y': [y_min, y_max]})
    # initialize the parameters to be evaluated as torch Parameters
#    unknown_parameters = torch.nn.Parameter(torch.rand(2,
#        requires_grad=True))
    # define the ranges for the parameters
    unknown_parameters_domain = CartesianDomain({'mu1': [-1, 1], 'mu2': [-1, 1]})

    def laplace_equation(input_, output_, unknown_variables):
        '''
        Laplace equation with a force term
        '''
        print('sono in laplace equation', unknown_variables)
        force_term = torch.exp(
                - 2*(input_.extract(['x']) - unknown_variables[0])**2
                - 2*(input_.extract(['y']) - unknown_variables[1])**2)
        delta_u = laplacian(output_, input_, components=['u'], d=['x', 'y'])
        return delta_u - force_term

    # define the conditions for the loss (boundary conditions, equation, data)
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
    # initialize the model of PINN (simple FeedForward network)
    model = FeedForward(
        layers=[20, 20, 20],
        func=Softplus,
        output_dimensions=len(problem.output_variables),
        input_dimensions=len(problem.input_variables)
        )
    # discretize the spatial domain: define the points were the conditions are
    # evaluated
    problem.discretise_domain(50, 'grid', locations=['D'], variables=['x', 'y'])
    problem.discretise_domain(50, 'grid', locations=['gamma1', 'gamma2',
        'gamma3', 'gamma4'], variables=['x', 'y'])

    # temporary directory for saving logs of training
    tmp_dir = "tmp_poisson_inverse"
    max_epochs = 5000

    ### train the problem with PINN
    if args.s:
        # class for plotting the solution every 100 epochs
        class MakePlot(Callback):
            def on_train_epoch_end(self, trainer, __):
                if trainer.current_epoch % 100 == 99:
                    pl = Plotter()
                    pl.plot(trainer, levels=120, cmap='jet', filename=f'epoch_{trainer.current_epoch}.pdf')
                    # save the torch parameters every 100 epochs
                    torch.save(pinn.problem.unknown_parameters, f'epoch_{trainer.current_epoch}')

        pinn = PINN(problem, model, optimizer_kwargs={'lr':0.01})
        # define the trainer for the solver
        trainer = Trainer(solver=pinn, accelerator='cpu', max_epochs=max_epochs,
                default_root_dir=tmp_dir, callbacks=[MakePlot()])
        trainer.train()

    if args.l:
        # load the trained model
        new_pinn = PINN.load_from_checkpoint(
            '{}/lightning_logs/version_0/checkpoints/epoch={}-step={}.ckpt'.format(
                tmp_dir, max_epochs-1, max_epochs),
            problem=problem, model=model)
        # plot the solution
        plotter = Plotter()
        plotter.plot(new_pinn)

