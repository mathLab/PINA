import torch

from .benchmark_problem import BenchmarkProblemInterface
from pina.problem import SpatialProblem, ParametricProblem
from pina.equation import Equation, FixedValue
from pina.operators import laplacian
from pina.geometry import CartesianDomain
from pina.condition import Condition


# simple poisson equation class

class PoissonEquation(Equation):

    def __init__(self, force_term, components=None, d=None):
        """
        Laplace Equation class. This class can be
        used to enforced a Laplace equation for a specific
        condition (force term set to zero).

        :param list(str) components: the name of the output
            variables to calculate the laplacian for. It should
            be a subset of the output labels. If ``None``,
            all the output variables are considered.
            Default is ``None``.
        :param list(str) d: the name of the input variables on
            which the laplacian is calculated. d should be a subset
            of the input labels. If ``None``, all the input variables
            are considered. Default is ``None``.
        """

        def equation(input_, output_):
            return laplacian(output_,
                             input_, 
                             components=components, d=d) - force_term(input_)

        super().__init__(equation)


################################################################################
class Poisson1DParametric(BenchmarkProblemInterface,
                           SpatialProblem,
                           ParametricProblem):
    r"""
    A one dimensional parametric Poisson problem. Given the input variables
    :math:`x, \mu` the problem computes the solution :math:`u(x, \mu)` by
    solving the following parametric partial differential equation:

    .. math::

        \begin{cases}
        \Delta u(x, \mu) = 2\pi\cos(\pi x)\cos(x + \mu) -
        (1+\pi^2)\sin(\pi x)\sin(x + \mu)  \quad x\in\Omega\times\mathbb{P} \\
        u(x, \mu) = 0 \quad x\in\partial\Omega\times\mathbb{P}
        \end{cases}

    The spatial domain is :math:`\Omega=[-1,1]`, while the parametric domain is
    :math:`\mathbb{P}=[-1, 1]`. The solution to the problem is analytical and
    reads :math:`sin(\pi x)sin(\mu + x)`. The problem can be solved by Physics
    Informed based Solvers, since is of type ``physics-informed``.
    """

    description = 'A one dimensional parametric Poisson problem.'
    problem_type = 'physics-informed'
    data_directory = ''
    input_variables = ['x', 'mu']
    output_variables = ['u']
    spatial_domain = CartesianDomain({'x': [-1, 1]})
    parameter_domain = CartesianDomain({'mu': [-1, 1]})

    def forcing(input_):
        x = input_.extract('x') 
        mu = input_.extract('mu')
        return (2 * torch.pi * torch.cos(mu + x) * torch.cos(torch.pi * x) - 
                (1 + torch.pi**2) * torch.sin(mu + x) * torch.sin(torch.pi * x))
    
    # problem condition statement
    conditions = {
        'gamma1': Condition(
            location=CartesianDomain({'x': -1, 'mu' : [-1, 1]}),
            equation=FixedValue(0)),
        'gamma2': Condition(
            location=CartesianDomain({'x':  1, 'mu' : [-1, 1]}),
            equation=FixedValue(0)),
        'D': Condition(
            location=CartesianDomain({'x': [-1, 1], 'mu' : [-1, 1]}),
            equation=PoissonEquation(forcing)),
    }