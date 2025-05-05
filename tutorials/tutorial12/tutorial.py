#!/usr/bin/env python
# coding: utf-8

# # Tutorial: Introduction to PINA `Equation` class
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mathLab/PINA/blob/master/tutorials/tutorial12/tutorial.ipynb)
#
#
# In this tutorial, we will explore how to use the `Equation` class in **PINA**. We will focus on how to leverage this class, along with its inherited subclasses, to enforce residual minimization in **Physics-Informed Neural Networks (PINNs)**.
#
# By the end of this guide, you'll understand how to integrate physical laws and constraints directly into your model training, ensuring that the solution adheres to the underlying differential equations.
#
#
# ## Example: The Burgers 1D equation
# We will start implementing the viscous Burgers 1D problem Class, described as follows:
#
# $$
# \begin{equation}
# \begin{cases}
# \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} &= \nu \frac{\partial^2 u}{ \partial x^2}, \quad x\in(0,1), \quad t>0\\
# u(x,0) &= -\sin (\pi x), \quad x\in(0,1)\\
# u(x,t) &= 0, \quad x = \pm 1, \quad t>0\\
# \end{cases}
# \end{equation}
# $$
#
# where we set $ \nu = \frac{0.01}{\pi}$.
#
# In the class that models this problem we will see in action the `Equation` class and one of its inherited classes, the `FixedValue` class.

# In[ ]:


## routine needed to run the notebook on Google Colab
try:
    import google.colab

    IN_COLAB = True
except:
    IN_COLAB = False
if IN_COLAB:
    get_ipython().system('pip install "pina-mathlab[tutorial]"')

import torch

# useful imports
from pina import Condition
from pina.problem import SpatialProblem, TimeDependentProblem
from pina.equation import Equation, FixedValue
from pina.domain import CartesianDomain
from pina.operator import grad, fast_grad, laplacian


# Let's begin by defining the Burgers equation and its initial condition as Python functions. These functions will take the model's `input` (spatial and temporal coordinates) and `output` (predicted solution) as arguments. The goal is to compute the residuals for the Burgers equation, which we will minimize during training.

# In[2]:


# define the burger equation
def burger_equation(input_, output_):
    du = fast_grad(output_, input_, components=["u"], d=["x"])
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


# Above we use the `grad` operator from `pina.operator` to compute the gradient. In PINA each differential operator takes the following inputs:
# - `output_`: A tensor on which the operator is applied.
# - `input_`: A tensor with respect to which the operator is computed.
# - `components`: The names of the output variables for which the operator is evaluated.
# - `d`: The names of the variables with respect to which the operator is computed.
#
# Each differential operator has its **fast** version, which performs no internal checks on input and output tensors. For these methods, the user is always required to specify both ``components`` and ``d`` as lists of strings.
#
# Let's define now the problem!
#
# > **👉 Do you want to learn more on Problems? Check the dedicated [tutorial](https://mathlab.github.io/PINA/tutorial16/tutorial.html) to learn how to build a Problem from scratch.**

# In[ ]:


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


# The `Equation` class takes as input a function (in this case it happens twice, with `initial_condition` and `burger_equation`) which computes a residual of an equation, such as a PDE. In a problem class such as the one above, the `Equation` class with such a given input is passed as a parameter in the specified `Condition`.
#
# The `FixedValue` class takes as input a value of the same dimensions as the output functions. This class can be used to enforce a fixed value for a specific condition, such as Dirichlet boundary conditions, as demonstrated in our example.
#
# Once the equations are set as above in the problem conditions, the PINN solver will aim to minimize the residuals described in each equation during the training phase.
#
# ### Available classes of equations:
# - `FixedGradient` and `FixedFlux`: These work analogously to the `FixedValue` class, where we can enforce a constant value on the gradient or the divergence of the solution, respectively.
# - `Laplace`: This class can be used to enforce that the Laplacian of the solution is zero.
# - `SystemEquation`: This class allows you to enforce multiple conditions on the same subdomain by passing a list of residual equations defined in the problem.
#
# ## Defining a new Equation class
# `Equation` classes can also be inherited to define a new class. For example, we can define a new class `Burgers1D` to represent the Burgers equation. During the class call, we can pass the viscosity parameter $\nu$:
#
# ```python
# class Burgers1D(Equation):
#     def __init__(self, nu):
#         self.nu = nu
#
#     def equation(self, input_, output_):
#         ...
# ```
# In this case, the `Burgers1D` class will inherit from the `Equation` class and compute the residual of the Burgers equation. The viscosity parameter $\nu$ is passed when instantiating the class and used in the residual calculation. Let's see it in more details:

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


# ## What's Next?
#
# Congratulations on completing the `Equation` class tutorial of **PINA**! As we've seen, you can build new classes that inherit from `Equation` to store more complex equations, such as the 1D Burgers equation, by simply passing the characteristic coefficients of the problem.
#
# From here, you can:
#
# - **Define Additional Complex Equation Classes**: Create your own equation classes, such as `SchrodingerEquation`, `NavierStokesEquation`, etc.
# - **Define More `FixedOperator` Classes**: Implement operators like `FixedCurl`, `FixedDivergence`, and others for more advanced simulations.
# - **Integrate Custom Equations and Operators**: Combine your custom equations and operators into larger systems for more complex simulations.
# - **and many more!**: Explore for example different residual minimization techniques to improve the performance and accuracy of your models.
#
# For more resources and tutorials, check out the [PINA Documentation](https://mathlab.github.io/PINA/).
