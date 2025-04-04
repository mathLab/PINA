#!/usr/bin/env python
# coding: utf-8

# # Tutorial: The `Equation` Class
# 
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mathLab/PINA/blob/master/tutorials/tutorial12/tutorial.ipynb)

# In this tutorial, we will show how to use the `Equation` Class in PINA. Specifically, we will see how use the Class and its inherited classes to enforce residuals minimization in PINNs.

# # Example: The Burgers 1D equation

# We will start implementing the viscous Burgers 1D problem Class, described as follows:
# 
# 
# $$
# \begin{equation}
# \begin{cases}
# \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} &= \nu \frac{\partial^2 u}{ \partial x^2}, \quad x\in(0,1), \quad t>0\\
# u(x,0) &= -\sin (\pi x)\\
# u(x,t) &= 0 \quad x = \pm 1\\
# \end{cases}
# \end{equation}
# $$
# 
# where we set $ \nu = \frac{0.01}{\pi}$.
# 
# In the class that models this problem we will see in action the `Equation` class and one of its inherited classes, the `FixedValue` class. 

# In[1]:


## routine needed to run the notebook on Google Colab
try:
    import google.colab

    IN_COLAB = True
except:
    IN_COLAB = False
if IN_COLAB:
    get_ipython().system('pip install "pina-mathlab"')

import torch

# useful imports
from pina import Condition
from pina.problem import SpatialProblem, TimeDependentProblem
from pina.equation import Equation, FixedValue
from pina.domain import CartesianDomain
from pina.operator import grad, laplacian


# In[2]:


# define the burger equation
def burger_equation(input_, output_):
    du = grad(output_, input_)
    ddu = grad(du, input_, components=["dudx"])
    return (
        du.extract(["dudt"])
        + output_.extract(["u"]) * du.extract(["dudx"])
        - (0.01 / torch.pi) * ddu.extract(["ddudxdx"])
    )


# define initial condition
def initial_condition(input_, output_):
    u_expected = -torch.sin(torch.pi * input_.extract(["x"]))
    return output_.extract(["u"]) - u_expected


class Burgers1D(TimeDependentProblem, SpatialProblem):

    # assign output/ spatial and temporal variables
    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [-1, 1]})
    temporal_domain = CartesianDomain({"t": [0, 1]})

    domains = {
        "bound_cond1": CartesianDomain({"x": -1, "t": [0, 1]}),
        "bound_cond2": CartesianDomain({"x": 1, "t": [0, 1]}),
        "time_cond": CartesianDomain({"x": [-1, 1], "t": 0}),
        "phys_cond": CartesianDomain({"x": [-1, 1], "t": [0, 1]}),
    }
    # problem condition statement
    conditions = {
        "bound_cond1": Condition(
            domain="bound_cond1", equation=FixedValue(0.0)
        ),
        "bound_cond2": Condition(
            domain="bound_cond2", equation=FixedValue(0.0)
        ),
        "time_cond": Condition(
            domain="time_cond", equation=Equation(initial_condition)
        ),
        "phys_cond": Condition(
            domain="phys_cond", equation=Equation(burger_equation)
        ),
    }


# 
# The `Equation` class takes as input a function (in this case it happens twice, with `initial_condition` and `burger_equation`) which computes a residual of an equation, such as a PDE. In a problem class such as the one above, the `Equation` class with such a given input is passed as a parameter in the specified `Condition`. 
# 
# The `FixedValue` class takes as input a value of same dimensions of the output functions; this class can be used to enforce a fixed value for a specific condition, e.g. Dirichlet boundary conditions, as it happens for instance in our example.
# 
# Once the equations are set as above in the problem conditions, the PINN solver will aim to minimize the residuals described in each equation in the training phase. 

# Available classes of equations include also:
# - `FixedGradient` and `FixedFlux`: they work analogously to `FixedValue` class, where we can require a constant value to be enforced, respectively, on the gradient of the solution or the divergence of the solution;
# - `Laplace`: it can be used to enforce the laplacian of the solution to be zero;
# - `SystemEquation`: we can enforce multiple conditions on the same subdomain through this class, passing a list of residual equations defined in the problem.
# 

# # Defining a new Equation class

# `Equation` classes can be also inherited to define a new class. As example, we can see how to rewrite the above problem introducing a new class `Burgers1D`; during the class call, we can pass the viscosity parameter $\nu$:

# In[3]:


class Burgers1DEquation(Equation):

    def __init__(self, nu=0.0):
        """
        Burgers1D class. This class can be
        used to enforce the solution u to solve the viscous Burgers 1D Equation.

        :param torch.float32 nu: the viscosity coefficient. Default value is set to 0.
        """
        self.nu = nu

        def equation(input_, output_):
            return (
                grad(output_, input_, d="t")
                + output_ * grad(output_, input_, d="x")
                - self.nu * laplacian(output_, input_, d="x")
            )

        super().__init__(equation)


# Now we can just pass the above class as input for the last condition, setting $\nu= \frac{0.01}{\pi}$:

# In[4]:


class Burgers1D(TimeDependentProblem, SpatialProblem):

    # define initial condition
    def initial_condition(input_, output_):
        u_expected = -torch.sin(torch.pi * input_.extract(["x"]))
        return output_.extract(["u"]) - u_expected

    # assign output/ spatial and temporal variables
    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [-1, 1]})
    temporal_domain = CartesianDomain({"t": [0, 1]})

    domains = {
        "bound_cond1": CartesianDomain({"x": -1, "t": [0, 1]}),
        "bound_cond2": CartesianDomain({"x": 1, "t": [0, 1]}),
        "time_cond": CartesianDomain({"x": [-1, 1], "t": 0}),
        "phys_cond": CartesianDomain({"x": [-1, 1], "t": [0, 1]}),
    }
    # problem condition statement
    conditions = {
        "bound_cond1": Condition(
            domain="bound_cond1", equation=FixedValue(0.0)
        ),
        "bound_cond2": Condition(
            domain="bound_cond2", equation=FixedValue(0.0)
        ),
        "time_cond": Condition(
            domain="time_cond", equation=Equation(initial_condition)
        ),
        "phys_cond": Condition(
            domain="phys_cond", equation=Burgers1DEquation(nu=0.01 / torch.pi)
        ),
    }


# # What's next?

# Congratulations on completing the `Equation` class tutorial of **PINA**! As we have seen, you can build new classes that inherit `Equation` to store more complex equations, as the Burgers 1D equation, only requiring to pass the characteristic coefficients of the problem. 
# From now on, you can:
# - define additional complex equation classes (e.g. `SchrodingerEquation`, `NavierStokeEquation`..)
# - define more `FixedOperator` (e.g. `FixedCurl`)
