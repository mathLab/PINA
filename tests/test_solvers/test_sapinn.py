import torch
import matplotlib.pyplot as plt

from pina.solvers.pinns.sapinn import SAPINN
from pina.solvers.pinns.pinn import PINN
from pina.operators import laplacian, grad
from pina.geometry import CartesianDomain
from pina import Condition, Trainer, Plotter
from pina.problem import SpatialProblem, TimeDependentProblem
from pina.model import FeedForward
from pina.equation.equation import Equation
from pina.equation.equation_factory import FixedValue

class Burgers(TimeDependentProblem, SpatialProblem):

    # define the burger equation
    def burger_equation(input_, output_):
        du = grad(output_, input_)
        ddu = grad(du, input_, components=['dudx'])
        return (
            du.extract(['dudt']) +
            output_.extract(['u'])*du.extract(['dudx']) -
            (0.01/torch.pi)*ddu.extract(['ddudxdx'])
        )

    # define initial condition
    def initial_condition(input_, output_):
        u_expected = -torch.sin(torch.pi*input_.extract(['x']))
        return output_.extract(['u']) - u_expected

    # assign output/ spatial and temporal variables
    output_variables = ['u']
    spatial_domain = CartesianDomain({'x': [-1, 1]})
    temporal_domain = CartesianDomain({'t': [0, 1]})

    # problem condition statement
    conditions = {
        'gamma1': Condition(location=CartesianDomain({'x': -1, 't': [0, 1]}), equation=FixedValue(0.)),
        'gamma2': Condition(location=CartesianDomain({'x':  1, 't': [0, 1]}), equation=FixedValue(0.)),
        't0': Condition(location=CartesianDomain({'x': [-1, 1], 't': 0}), equation=Equation(initial_condition)),
        'D': Condition(location=CartesianDomain({'x': [-1, 1], 't': [0, 1]}), equation=Equation(burger_equation)),
    }


problem = Burgers()
# discretizztion of the domain
problem.discretise_domain(10000, 'random', locations=['D'])
problem.discretise_domain(100, 'random', locations=['t0'])
problem.discretise_domain(200, 'random', locations=['gamma1', 'gamma2'])

# definining the model
model = FeedForward(
    layers=[20, 20, 20, 20, 20, 20, 20, 20],
    func=torch.nn.Tanh,
    output_dimensions=len(problem.output_variables),
    input_dimensions=len(problem.input_variables)
)

# inizialize sapinn
sapinn = SAPINN(
    problem,
    model
)


# Cration of the trainer
trainer = Trainer(
solver=sapinn,
max_epochs=10000,
accelerator='cpu',
enable_model_summary=False
)

# training
trainer.train()

# plot of the approximate solution
pl = Plotter()
pl.plot(sapinn)

# plot of weights
for key, value in sapinn.models[1].torchmodel.items():
    plt.scatter(problem.input_pts[key].extract('t').tensor.detach().numpy(), problem.input_pts[key].extract('x').tensor.detach().numpy(), c=value.forward().detach().numpy())
plt.colorbar()

plt.show()