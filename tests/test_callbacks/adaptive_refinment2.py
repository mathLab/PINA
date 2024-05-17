from pina.callbacks import R3Refinement
import torch
#import pytest

from pina.problem import SpatialProblem
from pina.operators import laplacian
from pina.geometry import CartesianDomain
from pina import Condition, LabelTensor
from pina.solvers import PINN
from pina.trainer import Trainer
from pina.model import FeedForward
from pina.equation.equation import Equation
from pina.equation.equation_factory import FixedValue


def laplace_equation(input_, output_):
    force_term = (torch.sin(input_.extract(['x']) * torch.pi) *
                  torch.sin(input_.extract(['y']) * torch.pi))
    delta_u = laplacian(output_.extract(['u']), input_)
    return delta_u - force_term


my_laplace = Equation(laplace_equation)
in_ = LabelTensor(torch.tensor([[0., 1.]]), ['x', 'y'])
out_ = LabelTensor(torch.tensor([[0.]]), ['u'])


class Poisson(SpatialProblem):
    output_variables = ['u']
    spatial_domain = CartesianDomain({'x': [0, 1], 'y': [0, 1]})

    conditions = {
        'gamma1': Condition(
            location=CartesianDomain({'x': [0, 1], 'y':  1}),
            equation=FixedValue(0.0)),
        'gamma2': Condition(
            location=CartesianDomain({'x': [0, 1], 'y': 0}),
            equation=FixedValue(0.0)),
        'gamma3': Condition(
            location=CartesianDomain({'x':  1, 'y': [0, 1]}),
            equation=FixedValue(0.0)),
        'gamma4': Condition(
            location=CartesianDomain({'x': 0, 'y': [0, 1]}),
            equation=FixedValue(0.0)),
        'D': Condition(
            input_points=LabelTensor(torch.rand(size=(100, 2)), ['x', 'y']),
            equation=my_laplace),
        # 'data': Condition(
        #     input_points=in_,
        #     output_points=out_)
    }


# make the problem
poisson_problem = Poisson()
boundaries = ['gamma1', 'gamma2', 'gamma3', 'gamma4']
n = 13
poisson_problem.discretise_domain(n, 'grid', locations=boundaries)
model = FeedForward(len(poisson_problem.input_variables),
                    len(poisson_problem.output_variables))

# make the solver
solver = PINN(problem=poisson_problem, model=model)


def test_r3constructor():
    R3Refinement(sample_every=10)

def get_num_pts():
    num_pts = {}
    for location in poisson_problem.input_pts:
        num_pts[location] = len(poisson_problem.input_pts[location])
    return num_pts

def test_r3refinment_routine():
    # make the trainer
    
    trainer = Trainer(solver=solver,
                      callbacks=[R3Refinement(sample_every=1)],
                      accelerator='cpu',
                      max_epochs=5)
    
    print(f'Before: {get_num_pts()}')
    trainer.train()
    print(f'After: {get_num_pts()}')

def test_r3refinment_routine_double_precision():
    model = FeedForward(len(poisson_problem.input_variables),
                    len(poisson_problem.output_variables))
    solver = PINN(problem=poisson_problem, model=model)
    trainer = Trainer(solver=solver,
                      precision='64-true',
                      accelerator='cpu',
                      callbacks=[R3Refinement(sample_every=2)],
                      max_epochs=5)
    print(f'Before: {get_num_pts()}')
    trainer.train()
    print(f'After: {get_num_pts()}')




def main():
    print('Test on R3Refinement routine')
    test_r3refinment_routine()
    print('\n\nTest on R3Refinement routine with double precision')
    test_r3refinment_routine_double_precision()

main()
