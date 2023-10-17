#!/usr/bin/env python
# coding: utf-8

# # Tutorial 1: Physics Informed Neural Networks on PINA

# In this tutorial, we will demonstrate a typical use case of PINA on a toy problem. Specifically, the tutorial aims to introduce the following topics:
# 
# * Defining a PINA Problem,
# * Building a `pinn` object,
# * Sampling points in a domain
# 
# These are the three main steps needed **before** training a Physics Informed Neural Network (PINN). We will show each step in detail, and at the end, we will solve the problem.

# ## PINA Problem

# ### Initialize the `Problem` class

# Problem definition in the PINA framework is done by building a python `class`, which inherits from one or more problem classes (`SpatialProblem`, `TimeDependentProblem`, `ParametricProblem`) depending on the nature of the problem. Below is an example:
# #### Simple Ordinary Differential Equation
# Consider the following:
# 
# $$
# \begin{equation}
# \begin{cases}
# \frac{d}{dx}u(x) &=  u(x) \quad x\in(0,1)\\
# u(x=0) &= 1 \\
# \end{cases}
# \end{equation}
# $$
# 
# with the analytical solution $u(x) = e^x$. In this case, our ODE depends only on the spatial variable $x\in(0,1)$ , meaning that our `Problem` class is going to be inherited from the `SpatialProblem` class:
# 
# ```python
# from pina.problem import SpatialProblem
# from pina import CartesianProblem
# 
# class SimpleODE(SpatialProblem):
#     
#     output_variables = ['u']
#     spatial_domain = CartesianProblem({'x': [0, 1]})
# 
#     # other stuff ...
# ```
# 
# Notice that we define `output_variables` as a list of symbols, indicating the output variables of our equation (in this case only $u$). The `spatial_domain` variable indicates where the sample points are going to be sampled in the domain, in this case $x\in[0,1]$.

# What about if our equation is also time dependent? In this case, our `class` will inherit from both `SpatialProblem` and `TimeDependentProblem`:
# 

# In[1]:


from pina.problem import SpatialProblem, TimeDependentProblem
from pina import CartesianDomain

class TimeSpaceODE(SpatialProblem, TimeDependentProblem):
    
    output_variables = ['u']
    spatial_domain = CartesianDomain({'x': [0, 1]})
    temporal_domain = CartesianDomain({'t': [0, 1]})

    # other stuff ...


# where we have included the `temporal_domain` variable, indicating the time domain wanted for the solution.
# 
# In summary, using PINA, we can initialize a problem with a class which inherits from three base classes: `SpatialProblem`, `TimeDependentProblem`, `ParametricProblem`, depending on the type of problem we are considering. For reference:
# * `SpatialProblem` $\rightarrow$ a differential equation with spatial variable(s)
# * `TimeDependentProblem` $\rightarrow$ a time-dependent differential equation
# * `ParametricProblem` $\rightarrow$ a parametrized differential equation

# ### Write the `Problem` class
# 
# Once the `Problem` class is initialized, we need to represent the differential equation in PINA. In order to do this, we need to load the PINA operators from `pina.operators` module. Again, we'll consider Equation (1) and represent it in PINA:

# In[2]:


from pina.problem import SpatialProblem
from pina.operators import grad
from pina import Condition, CartesianDomain
from pina.equation.equation import Equation

import torch


class SimpleODE(SpatialProblem):

    output_variables = ['u']
    spatial_domain = CartesianDomain({'x': [0, 1]})

    # defining the ode equation
    def ode_equation(input_, output_):

        # computing the derivative
        u_x = grad(output_, input_, components=['u'], d=['x'])

        # extracting the u input variable
        u = output_.extract(['u'])

        # calculate the residual and return it
        return u_x - u

    # defining the initial condition
    def initial_condition(input_, output_):
        
        # setting the initial value
        value = 1.0

        # extracting the u input variable
        u = output_.extract(['u'])

        # calculate the residual and return it
        return u - value

    # conditions to hold
    conditions = {
        'x0': Condition(location=CartesianDomain({'x': 0.}), equation=Equation(initial_condition)),
        'D': Condition(location=CartesianDomain({'x': [0, 1]}), equation=Equation(ode_equation)),
    }

    # sampled points (see below)
    input_pts = None

    # defining the true solution
    def truth_solution(self, pts):
        return torch.exp(pts.extract(['x']))


# After we define the `Problem` class, we need to write different class methods, where each method is a function returning a residual. These functions are the ones minimized during PINN optimization, given the initial conditions. For example, in the domain $[0,1]$, the ODE equation (`ode_equation`) must be satisfied. We represent this by returning the difference between subtracting the variable `u` from its gradient (the residual), which we hope to minimize to 0. This is done for all conditions (`ode_equation`, `initial_condition`). 
# 
# Once we have defined the function, we need to tell the neural network where these methods are to be applied. To do so, we use the `Condition` class. In the `Condition` class, we pass the location points and the function we want minimized on those points (other possibilities are allowed, see the documentation for reference) as parameters.
# 
# Finally, it's possible to define a `truth_solution` function, which can be useful if we want to plot the results and see how the real solution compares to the expected (true) solution. Notice that the `truth_solution` function is a method of the `PINN` class, but is not mandatory for problem definition.
# 

# ## Build the `PINN` object

# The basic requirements for building a `PINN` model are a `Problem` and a model. We have just covered the `Problem` definition. For the model parameter, one can use either the default models provided in PINA or a custom model. We will not go into the details of model definition (see Tutorial2 and Tutorial3 for more details on model definition).

# In[3]:


from pina.model import FeedForward
from pina import PINN

# initialize the problem
problem = SimpleODE()

# build the model
model = FeedForward(
    layers=[10, 10],
    func=torch.nn.Tanh,
    output_dimensions=len(problem.output_variables),
    input_dimensions=len(problem.input_variables)
)

# create the PINN object
pinn = PINN(problem, model)


# Creating the `PINN` object is fairly simple. Different optional parameters include: optimizer, batch size, ... (see [documentation](https://mathlab.github.io/PINA/) for reference).

# ## Sample points in the domain 

# Once the `PINN` object is created, we need to generate the points for starting the optimization. To do so, we use the `sample` method of the `CartesianDomain` class. Below are three examples of sampling methods on the $[0,1]$ domain:

# In[4]:


# sampling 20 points in [0, 1] through discretization
pinn.problem.discretise_domain(n=20, mode='grid', variables=['x'])

# sampling 20 points in (0, 1) through latin hypercube samping
pinn.problem.discretise_domain(n=20, mode='latin', variables=['x'])

# sampling 20 points in (0, 1) randomly
pinn.problem.discretise_domain(n=20, mode='random', variables=['x'])


# ### Very simple training and plotting
# 
# Once we have defined the PINA model, created a network, and sampled points in the domain, we have everything necessary for training a PINN. To do so, we make use of the `Trainer` class.

# In[5]:


from pina import Trainer

# initialize trainer
trainer = Trainer(pinn)

# train the model
trainer.train()

