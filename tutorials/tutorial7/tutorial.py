#!/usr/bin/env python
# coding: utf-8

# # Tutorial 7: Resolution of an inverse problem

# ### Problem definition

# This tutorial shows how to solve an inverse Poisson problem with Physics-Informed Neural Networks. The problem definition is that of a Poisson problem with homogeneous boundary conditions and it reads:
# \begin{equation}
# \begin{cases}\\
# \Delta u = \exp{-2(x-\mu_1)^2-2(y-\mu_2)^2} \text{ in } \Omega\, ,\\
# u = 0 \text{ on }\partial \Omega,\\
# u(\mu_1, \mu_2) = \text{ data},
# \end{cases}
# \end{equation}
# where $\Omega$ is a square domain $[-2, 2] \times [-2, 2]$, and $\partial \Omega=\Gamma_1 \cup \Gamma_2 \cup \Gamma_3 \cup \Gamma_4$ is the union of the boundaries of the domain.
#
# This kind of problem, namely the "inverse problem", has two main goals:
# - find the solution $u$ that satisfies the Poisson equation;
# - find the unknown parameters ($\mu_1$, $\mu_2$) that better fit some given data (third equation in the system above).
#
# In order to achieve both the goals we will need to define an `InverseProblem` in PINA.

# Let's start with useful imports.

# In[2]:


import matplotlib.pyplot as plt
import torch
from torch.nn import Softplus
from lightning.pytorch.callbacks import Callback
from pina.problem import SpatialProblem, InverseProblem
from pina.operators import laplacian
from pina.model import FeedForward
from pina.equation import Equation, FixedValue#, ParametricEquation
from pina import Condition, LabelTensor, Plotter, Trainer
from pina.geometry import CartesianDomain
from pina.solvers import PINN
from pina.callbacks import MetricTracker


# Then, we import the pre-saved data, for ($\mu_1$, $\mu_2$)=($0.5$, $0.5$). These two values are the optimal parameters that we want to find through the neural network training. In particular, we import the `input_points`(the spatial coordinates), and the `output_points` (the corresponding $u$ values evaluated at the `input_points`).

# In[ ]:


data_output = torch.load('data/pinn_solution_0.5_0.5')
data_input = torch.load('data/pts_0.5_0.5')


# Moreover, let's plot also the data points and the reference solution: this is the expected output of the neural network.

# In[ ]:


points = data_input.extract(['x', 'y']).detach().numpy()
truth = data_output.detach().numpy()
plt.scatter(points[:, 0], points[:, 1], c=truth)
plt.axis('equal')
plt.colorbar()
plt.show()


# Then, we initialize the Poisson problem, that is inherited from the `SpatialProblem` and from the `InverseProblem` classes. We here have to define all the variables, and the domain where our unknown parameters ($\mu_1$, $\mu_2$) belong.
# In the following class, the equation with the unknown parameters is inherited from the class `ParametricEquation`. Differently from the standard `Equation`, it takes as inputs also the unknown variables, that will be treated as parameters that the neural network optimizes during the training process.

# In[ ]:


### Define ranges of variables
x_min = -2
x_max = 2
y_min = -2
y_max = 2

class Poisson(SpatialProblem, InverseProblem):
    '''
    Problem definition for the Poisson equation.
    '''
    output_variables = ['u']
    spatial_domain = CartesianDomain({'x': [x_min, x_max], 'y': [y_min, y_max]})
    # define the ranges for the parameters
    unknown_parameter_domain = CartesianDomain({'mu1': [-1, 1], 'mu2': [-1, 1]})

    def laplace_equation(input_, output_, params_):
        '''
        Laplace equation with a force term.
        '''
        force_term = torch.exp(
                - 2*(input_.extract(['x']) - params_['mu1'])**2
                - 2*(input_.extract(['y']) - params_['mu2'])**2
                )
        delta_u = laplacian(output_, input_, components=['u'], d=['x', 'y'])

        return delta_u - force_term

    # define the conditions for the loss (boundary conditions, equation, data)
    conditions = {
        'gamma1': Condition(location=CartesianDomain({'x': [x_min, x_max],
            'y':  y_max}),
            equation=FixedValue(0.0, components=['u'])),
        'gamma2': Condition(location=CartesianDomain({'x': [x_min, x_max], 'y': y_min
            }),
            equation=FixedValue(0.0, components=['u'])),
        'gamma3': Condition(location=CartesianDomain({'x':  x_max, 'y': [y_min, y_max]
            }),
            equation=FixedValue(0.0, components=['u'])),
        'gamma4': Condition(location=CartesianDomain({'x': x_min, 'y': [y_min, y_max]
            }),
            equation=FixedValue(0.0, components=['u'])),
        'D': Condition(location=CartesianDomain({'x': [x_min, x_max], 'y': [y_min, y_max]
            }),
        equation=Equation(laplace_equation)),
#        'data': Condition(input_points=data_input.extract(['x', 'y']), output_points=data_output)
    }

problem = Poisson()


# Then, we define the model of the neural network we want to use, here a simple `FeedForward`.

# In[ ]:


model = FeedForward(
    layers=[20, 20, 20],
    func=Softplus,
    output_dimensions=len(problem.output_variables),
    input_dimensions=len(problem.input_variables)
    )


# After that, we discretize the spatial domain.

# In[ ]:


problem.discretise_domain(100, 'grid', locations=['D'], variables=['x', 'y'])
problem.discretise_domain(100, 'grid', locations=['gamma1', 'gamma2',
    'gamma3', 'gamma4'], variables=['x', 'y'])


# Here, we define a simple callback for the trainer, that will be used to save the parameters in a directory, such that we can then plot how their trend across the epochs.

# In[ ]:


# temporary directory for saving logs of training
tmp_dir = "tmp_poisson_inverse"

#class SaveParameters(Callback):
#    '''
#    Callback to save the parameters of the model every 100 epochs.
#    '''
#    def on_train_epoch_end(self, trainer, __):
#        if trainer.current_epoch % 100 == 99:
#            torch.save(pinn.problem.unknown_parameters, '{}/parameters_epoch{}'.format(tmp_dir, trainer.current_epoch))
#

# Then, we define the `PINN` object that we train the neural network.

# In[ ]:


### train the problem with PINN
max_epochs=5000
pinn = PINN(problem, model, optimizer_kwargs={'lr':0.001})
# define the trainer for the solver
trainer = Trainer(solver=pinn, accelerator='cpu', max_epochs=max_epochs)#,
    #    default_root_dir=tmp_dir)#, callbacks=[SaveParameters(), MetricTracker()])
trainer.train()


# We can finally plot the solution and see that it is really similar to the one given by the initial data.

# In[ ]:


plotter = Plotter()
plotter.plot(trainer)


# One can also see how the parameters vary during the training by reading the saved solution and plotting them. The plot shows that the parameters stabilize to their true value before reaching the epoch $1000$!

# In[ ]:


epochs_saved = range(99, max_epochs, 100)
parameters = torch.empty((int(max_epochs/100), 2))
for i, epoch in enumerate(epochs_saved):
    params_torch = torch.load('{}/parameters_epoch{}'.format(tmp_dir, epoch))
    parameters[i, :] = params_torch.data

# Plot parameters
plt.plot(epochs_saved, parameters[:, 0], label='mu1', marker='o')
plt.plot(epochs_saved, parameters[:, 1], label='mu2', marker='s')
plt.ylim(-1, 1)
plt.grid()
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Parameter value')
plt.show()

