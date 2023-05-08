#!/usr/bin/env python
# coding: utf-8

# # Tutorial 1: Physics Informed Neural Networks on PINA

# In this tutorial we will show the typical use case of PINA on a toy problem. Specifically, the tutorial aims to introduce the following topics:
# 
# * Defining a PINA Problem,
# * Build a `pinn` object,
# * Sample points in the domain.
# 
# These are the three main steps needed **before** training a Physics Informed Neural Network (PINN). We will show in detailed each step, and at the end we will solve a very simple problem with PINA.

# ## PINA Problem

# ### Initialize the Problem class

# The problem definition in the PINA framework is done by building a phython `class`, inherited from one or more problem classes (`SpatialProblem`, `TimeDependentProblem`, `ParametricProblem`), depending on the nature of the problem treated. Let's see an example to better understand:
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
# with analytical solution $u(x) = e^x$. In this case we have that our ODE depends only on the spatial variable $x\in(0,1)$ , this means that our problem class is going to be inherited from `SpatialProblem` class:
# 
# ```python
# from pina.problem import SpatialProblem
# from pina import Span
# 
# class SimpleODE(SpatialProblem):
#     
#     output_variables = ['u']
#     spatial_domain = Span({'x': [0, 1]})
# 
#     # other stuff ...
# ```
# 
# Notice that we define `output_variables` as a list of symbols, indicating the output variables of our equation (in this case only $u$). The `spatial_domain` variable indicates where the sample points are going to be sampled in the domain, in this case $x\in(0,1)$.

# What about if we also have a time depencency in the equation? Well in that case our `class` will inherit from both `SpatialProblem` and `TimeDependentProblem`:
# ```python
# from pina.problem import SpatialProblem, TimeDependentProblem
# from pina import Span
# 
# class TimeSpaceODE(SpatialProblem, TimeDependentProblem):
#     
#     output_variables = ['u']
#     spatial_domain = Span({'x': [0, 1]})
#     temporal_domain = Span({'x': [0, 1]})
# 
#     # other stuff ...
# ```
# where we have included the `temporal_domain` variable indicating the time domain where we want the solution.
# 
# Summarizing, in PINA we can initialize a problem with a class which is inherited from three base classes: `SpatialProblem`, `TimeDependentProblem`, `ParametricProblem`, depending on the type of problem we are considering. For reference:
# * `SpatialProblem` $\rightarrow$ spatial variable(s) presented in the differential equation
# * `TimeDependentProblem` $\rightarrow$ time variable(s) presented in the differential equation
# * `ParametricProblem` $\rightarrow$ parameter(s) presented in the differential equation
# 

# ### Write the problem class
# 
# Once the problem class is initialized we need to write the differential equation in PINA language. For doing this we need to load the pina operators found in `pina.operators` module. Let's again consider the Equation (1) and try to write the PINA model class:

# In[ ]:


from pina.problem import SpatialProblem
from pina.operators import grad
from pina import Condition, Span

import torch


class SimpleODE(SpatialProblem):

    output_variables = ['u']
    spatial_domain = Span({'x': [0, 1]})

    # defining the ode equation
    def ode_equation(input_, output_):

        # computing the derivative
        u_x = grad(output_, input_, components=['u'], d=['x'])

        # extracting u input variable
        u = output_.extract(['u'])

        # calculate residual and return it
        return u_x - u

    # defining initial condition
    def initial_condition(input_, output_):
        
        # setting initial value
        value = 1.0

        # extracting u input variable
        u = output_.extract(['u'])

        # calculate residual and return it
        return u - value

    # Conditions to hold
    conditions = {
        'x0': Condition(location=Span({'x': 0.}), function=initial_condition),
        'D': Condition(location=Span({'x': [0, 1]}), function=ode_equation),
    }

    # defining true solution
    def truth_solution(self, pts):
        return torch.exp(pts.extract(['x']))


# After the defition of the Class we need to write different class methods, where each method is a function returning a residual. This functions are the ones minimized during the PINN optimization, for the different conditions. For example, in the domain $(0,1)$ the ODE equation (`ode_equation`) must be satisfied, so we write it by putting all the ODE equation on the right hand side, such that we return the zero residual. This is done for all the conditions  (`ode_equation`, `initial_condition`). 
# 
# Once we have defined the function we need to tell the network where these methods have to be applied. For doing this we use the class `Condition`. In `Condition` we pass the location points and the function to be minimized on those points (other possibilities are allowed, see the documentation for reference).
# 
# Finally, it's possible to defing the `truth_solution` function, which can be useful if we want to plot the results and see a comparison of real vs expected solution. Notice that `truth_solution` function is a method of the `PINN` class, but it is not mandatory for the problem definition.

# ## Build PINN object

# The basics requirements for building a PINN model are a problem and a model. We have already covered the problem definition. For the model one can use the default models provided in PINA or use a custom model. We will not go into the details of model definition, Tutorial2 and Tutorial3 treat the topic in detail.

# In[ ]:


from pina.model import FeedForward
from pina import PINN

# initialize the problem
problem = SimpleODE()

# build the model
model = FeedForward(
    layers=[10, 10],
    func=torch.nn.Tanh,
    output_variables=problem.output_variables,
    input_variables=problem.input_variables
)

# create the PINN object
pinn = PINN(problem, model)


# Creating the pinn object is fairly simple by using the `PINN` class, different optional inputs can be passed: optimizer, batch size, ... (see [documentation](https://mathlab.github.io/PINA/) for reference).

# ## Sample points in the domain 

# Once the `pinn` object is created, we need to generate the points for starting the optimization. For doing this we use the `span_pts` method of the `PINN` class.
# Let's see some methods to sample in $(0,1 )$.

# In[ ]:


# sampling 20 points in (0, 1) with discrite step
pinn.span_pts(20, 'grid', locations=['D'])

# sampling 20 points in (0, 1) with latin hypercube
pinn.span_pts(20, 'latin', locations=['D'])

# sampling 20 points in (0, 1) randomly
pinn.span_pts(20, 'random', locations=['D'])


# We can also use a dictionary for specific variables:

# In[ ]:


pinn.span_pts({'variables': ['x'], 'mode': 'grid', 'n': 20}, locations=['D'])


# We are going to use equispaced points for sampling. We need to sample in all the conditions domains. In our case we sample in `D` and `x0`.

# In[ ]:


# sampling for training
pinn.span_pts(1, 'random', locations=['x0'])
pinn.span_pts(20, 'grid', locations=['D'])


# ### Very simple training and plotting
# 
# Once we have defined the PINA model, created a network and sampled points in the domain, we have everything that is necessary for training a PINN. Here we show a very short training and some method for plotting the results.

# In[ ]:


# simple training 
final_loss = pinn.train(stop=3000, frequency_print=1000)


# After the training we have saved the final loss in `final_loss`, which we can inspect. By default PINA uses mean square error loss.

# In[ ]:


# inspecting final loss
final_loss


# By using the `Plotter` class from PINA we can also do some quatitative plots of the loss function. 

# In[ ]:


from pina.plotter import Plotter

# plotting the loss
plotter = Plotter()
plotter.plot_loss(pinn)


# We have a very smooth loss decreasing!
