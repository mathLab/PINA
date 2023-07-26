""" Parametric elliptic optimal control problem with variable alpha"""

from pina import CartesianDomain, Condition
from pina.problem import SpatialProblem, ParametricProblem
from pina.operators import laplacian


class ParametricEllipticOptimalControl(SpatialProblem, ParametricProblem):
    """ Parametric elliptic optimal control problem with variable alpha"""

    xmin, xmax, ymin, ymax = -1, 1, -1, 1
    amin, amax = 0.0001, 1
    mumin, mumax = 0.5, 3
    mu_range = [mumin, mumax]
    a_range = [amin, amax]
    x_range = [xmin, xmax]
    y_range = [ymin, ymax]

    output_variables = ['u', 'p', 'y']
    spatial_domain = CartesianDomain({'x1': x_range, 'x2': y_range})
    parameter_domain = CartesianDomain({'mu': mu_range, 'alpha': a_range})

    def term1(self, input_, output_):
        """ Term 1 of the PDE."""
        laplace_p = laplacian(output_, input_, components=[
                              'p'], d=['x1', 'x2'])
        return output_.extract(['y']) - input_.extract(['mu']) - laplace_p

    def term2(self, input_, output_):
        """ Term 2 of the PDE."""
        laplace_y = laplacian(output_, input_, components=[
                              'y'], d=['x1', 'x2'])
        return - laplace_y - output_.extract(['u_param'])

    def state_dirichlet(self, input_, output_):
        """ Dirichlet boundary condition."""
        y_exp = 0.0
        return output_.extract(['y']) - y_exp

    def adj_dirichlet(self, input_, output_):
        """ Dirichlet boundary condition."""
        p_exp = 0.0
        return output_.extract(['p']) - p_exp

    conditions = {
        'gamma1': Condition(
            CartesianDomain({'x1': x_range, 'x2':  1, 'mu': mu_range, 'alpha': a_range}),
            [state_dirichlet, adj_dirichlet]),
        'gamma2': Condition(
            CartesianDomain({'x1': x_range, 'x2': -1, 'mu': mu_range, 'alpha': a_range}),
            [state_dirichlet, adj_dirichlet]),
        'gamma3': Condition(
            CartesianDomain({'x1':  1, 'x2': y_range, 'mu': mu_range, 'alpha': a_range}),
            [state_dirichlet, adj_dirichlet]),
        'gamma4': Condition(
            CartesianDomain({'x1': -1, 'x2': y_range, 'mu': mu_range, 'alpha': a_range}),
            [state_dirichlet, adj_dirichlet]),
        'D': Condition(
            CartesianDomain({'x1': x_range, 'x2': y_range,
                  'mu': mu_range, 'alpha': a_range}),
            [term1, term2]),
    }
