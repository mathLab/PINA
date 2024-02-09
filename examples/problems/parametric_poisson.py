""" Parametric Poisson problem. """

# ===================================================== #
#                                                       #
#  This script implements the two dimensional           #
#  Parametric Poisson problem. The ParametricPoisson    #
#  class is defined inheriting from SpatialProblem and  #
#  ParametricProblem. We  denote:                       #
#           u --> field variable                        #
#           x,y --> spatial variables                   #
#           mu1, mu2 --> parameter variables            #
#                                                       #
# ===================================================== #


from pina.geometry import CartesianDomain
from pina.problem import SpatialProblem, ParametricProblem
from pina.operators import laplacian
from pina.equation import FixedValue, Equation
from pina import Condition
import torch

# define the laplace equation
def laplace_equation(input_, output_):
    """ Laplace equation with a parametric force term."""
    force_term = torch.exp(
        -2 * (input_.extract(["x"]) - input_.extract(["mu1"])) ** 2
        - 2 * (input_.extract(["y"]) - input_.extract(["mu2"])) ** 2
    )
    return laplacian(output_.extract(["u"]), input_) - force_term

class ParametricPoisson(SpatialProblem, ParametricProblem):

    # assign output/ spatial and parameter variables
    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [-1, 1], "y": [-1, 1]})
    parameter_domain = CartesianDomain({"mu1": [-1, 1], "mu2": [-1, 1]})


    # problem condition statement
    conditions = {
        "gamma1": Condition(
            location=CartesianDomain(
                {"x": [-1, 1], "y": 1, "mu1": [-1, 1], "mu2": [-1, 1]}
            ),
            equation=FixedValue(0.0),
        ),
        "gamma2": Condition(
            location=CartesianDomain(
                {"x": [-1, 1], "y": -1, "mu1": [-1, 1], "mu2": [-1, 1]}
            ),
            equation=FixedValue(0.0),
        ),
        "gamma3": Condition(
            location=CartesianDomain(
                {"x": 1, "y": [-1, 1], "mu1": [-1, 1], "mu2": [-1, 1]}
            ),
            equation=FixedValue(0.0),
        ),
        "gamma4": Condition(
            location=CartesianDomain(
                {"x": -1, "y": [-1, 1], "mu1": [-1, 1], "mu2": [-1, 1]}
            ),
            equation=FixedValue(0.0),
        ),
        "D": Condition(
            location=CartesianDomain(
                {"x": [-1, 1], "y": [-1, 1], "mu1": [-1, 1], "mu2": [-1, 1]}
            ),
            equation=Equation(laplace_equation),
        ),
    }
