import torch
import matplotlib.pyplot as plt

from pina.solvers.pinns.sapinn import SAPINN
from pina.operators import laplacian
from pina.geometry import CartesianDomain
from pina import Condition, LabelTensor
from pina.problem import SpatialProblem
from pina.model import FeedForward
from pina.equation.equation import Equation
from pina.equation.equation_factory import FixedValue

def laplace_equation(input_, output_):
    force_term = (torch.sin(input_.extract(['x']) * torch.pi) *
                torch.sin(input_.extract(['y']) * torch.pi))
    delta_u = laplacian(output_.extract(['u']), input_)
    return delta_u - force_term

my_laplace = Equation(laplace_equation)

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
            location=CartesianDomain({'x' : [0, 1], 'y' : [0, 1]}),
            equation=my_laplace)
    }

    def poisson_sol(self, pts):
        return -(torch.sin(pts.extract(['x']) * torch.pi) *
                 torch.sin(pts.extract(['y']) * torch.pi)) / (2 * torch.pi**2)

    truth_solution = poisson_sol


problem = Poisson()
# Discretizzazione del dominio
problem.discretise_domain(5, 'random', locations=['D'])
problem.discretise_domain(5, 'random', locations=['D', 'gamma1', 'gamma2', 'gamma3','gamma4'])

# Definizione del modello risolutivo
model = FeedForward(
    layers=[10, 10],
    func=torch.nn.Tanh,
    output_dimensions=len(problem.output_variables),
    input_dimensions=len(problem.input_variables)
)

# Inizializzazione SAPINN()
sapinn = SAPINN(
    problem,
    model,
    mask_type={"type" : "sigmoid", "coefficient": [100, 1, 1]}
)

print(sapinn.weights)