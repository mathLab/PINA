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

"""def laplace_equation(input_, output_):
    force_term = (torch.sin(input_.extract(['x']) * torch.pi) *
                torch.sin(input_.extract(['y']) * torch.pi))
    delta_u = laplacian(output_.extract(['u']), input_)
    return delta_u - force_term

my_laplace = Equation(laplace_equation)"""

class Burgers(SpatialProblem,TimeDependentProblem):
    output_variables = ['u']
    spatial_domain = CartesianDomain({'x': [-1, 1]})
    temporal_domain = CartesianDomain({'t' : [0, 1]})

    def initial_condition(input_, output_):
        u_expected = - torch.sin(torch.pi * input_.extract('x'))
        return output_.extract('u') - u_expected

    def burger_equation(input_, output_):
        u_t = grad(output_, input_, components=['u'], d=['t'])
        u_x = grad(output_, input_, components=['u'], d=['x'])
        u_xx = laplacian(output_, input_, components=['u'], d=['x'])
        return u_t + output_.extract('u') * u_x - (0.01/torch.pi) * u_xx

    conditions = {
        'gamma1': Condition(
            location=CartesianDomain({'x': -1, 't':  [0, 1]}),
            equation=FixedValue(0.0)
        ),
        'gamma2': Condition(
            location=CartesianDomain({'x': 1, 't':  [0, 1]}),
            equation=FixedValue(0.0)
        ),
        't0' : Condition(
            location=CartesianDomain({'x' : [-1, 1], 't' : 0}),
            equation=Equation(initial_condition)
        ),
        'D': Condition(
            location=CartesianDomain({'x' : [-1, 1], 't' : [0, 1]}),
            equation=Equation(burger_equation))
    }


problem = Burgers()
# Discretizzazione del dominio
problem.discretise_domain(10000, 'random', locations=['D'])
problem.discretise_domain(100, 'random', locations=['t0'])
problem.discretise_domain(200, 'random', locations=['gamma1', 'gamma2'])

# Definizione del modello risolutivo
model = FeedForward(
    layers=[20, 20],
    func=torch.nn.Tanh,
    output_dimensions=len(problem.output_variables),
    input_dimensions=len(problem.input_variables)
)

# Inizializzazione SAPINN()
sapinn = SAPINN(
    problem,
    model
)


# Creaimo il trainer
trainer = Trainer(
solver=sapinn,
max_epochs=10000,
accelerator='cpu',
enable_model_summary=False
)


trainer.train()

pl = Plotter()

pl.plot(sapinn)

for key, value in sapinn.models[1].torchmodel.items():
    plt.scatter(problem.input_pts[key].extract('t').tensor.detach().numpy(), problem.input_pts[key].extract('x').tensor.detach().numpy(), c=value.forward().detach().numpy())
plt.colorbar()

plt.show()