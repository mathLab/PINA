#!/usr/bin/env python
# coding: utf-8

# # Tutorial: Resolution of an inverse problem
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mathLab/PINA/blob/master/tutorials/tutorial7/tutorial.ipynb)

# ### Introduction to the inverse problem

# This tutorial shows how to solve an inverse Poisson problem with Physics-Informed Neural Networks. The problem definition is that of a Poisson problem with homogeneous boundary conditions and it reads:
# \begin{equation}
# \begin{cases}
# \Delta u = e^{-2(x-\mu_1)^2-2(y-\mu_2)^2} \text{ in } \Omega\, ,\\
# u = 0 \text{ on }\partial \Omega,\\
# u(\mu_1, \mu_2) = \text{ data}
# \end{cases}
# \end{equation}
# where $\Omega$ is a square domain $[-2, 2] \times [-2, 2]$, and $\partial \Omega=\Gamma_1 \cup \Gamma_2 \cup \Gamma_3 \cup \Gamma_4$ is the union of the boundaries of the domain.
#
# This kind of problem, namely the "inverse problem", has two main goals:
# - find the solution $u$ that satisfies the Poisson equation;
# - find the unknown parameters ($\mu_1$, $\mu_2$) that better fit some given data (third equation in the system above).
#
# In order to achieve both goals we will need to define an `InverseProblem` in PINA.

# Let's start with useful imports.

# In[20]:


## routine needed to run the notebook on Google Colab
try:
    import google.colab

    IN_COLAB = True
except:
    IN_COLAB = False
if IN_COLAB:
    get_ipython().system('pip install "pina-mathlab"')
    # get the data
    get_ipython().system('mkdir "data"')
    get_ipython().system(
        'wget "https://github.com/mathLab/PINA/raw/refs/heads/master/tutorials/tutorial7/data/pinn_solution_0.5_0.5" -O "data/pinn_solution_0.5_0.5"'
    )
    get_ipython().system(
        'wget "https://github.com/mathLab/PINA/raw/refs/heads/master/tutorials/tutorial7/data/pts_0.5_0.5" -O "data/pts_0.5_0.5"'
    )

import matplotlib.pyplot as plt
import torch
import warnings

from pina import Condition, Trainer
from pina.problem import SpatialProblem, InverseProblem
from pina.operator import laplacian
from pina.model import FeedForward
from pina.equation import Equation, FixedValue
from pina.solver import PINN
from pina.domain import CartesianDomain
from pina.optim import TorchOptimizer
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import Callback

warnings.filterwarnings("ignore")
seed_everything(883)


# Then, we import the pre-saved data, for ($\mu_1$, $\mu_2$)=($0.5$, $0.5$). These two values are the optimal parameters that we want to find through the neural network training. In particular, we import the `input` points (the spatial coordinates), and the `target` points (the corresponding $u$ values evaluated at the `input`).

# In[21]:


data_output = torch.load(
    "data/pinn_solution_0.5_0.5", weights_only=False
).detach()
data_input = torch.load("data/pts_0.5_0.5", weights_only=False)


# Moreover, let's plot also the data points and the reference solution: this is the expected output of the neural network.

# In[22]:


points = data_input.extract(["x", "y"]).detach().numpy()
truth = data_output.detach().numpy()

plt.scatter(points[:, 0], points[:, 1], c=truth, s=8)
plt.axis("equal")
plt.colorbar()
plt.show()


# ### Inverse problem definition in PINA

# Then, we initialize the Poisson problem, that is inherited from the `SpatialProblem` and from the `InverseProblem` classes. We here have to define all the variables, and the domain where our unknown parameters ($\mu_1$, $\mu_2$) belong. Notice that the Laplace equation takes as inputs also the unknown variables, that will be treated as parameters that the neural network optimizes during the training process.

# In[23]:


def laplace_equation(input_, output_, params_):
    """
    Implementation of the laplace equation.

    :param LabelTensor input_: Input data of the problem.
    :param LabelTensor output_: Output data of the problem.
    :param dict params_: Parameters of the problem.
    :return: The residual of the laplace equation.
    :rtype: LabelTensor
    """
    force_term = torch.exp(
        -2 * (input_.extract(["x"]) - params_["mu1"]) ** 2
        - 2 * (input_.extract(["y"]) - params_["mu2"]) ** 2
    )
    delta_u = laplacian(output_, input_, components=["u"], d=["x", "y"])
    return delta_u - force_term


class Poisson(SpatialProblem, InverseProblem):
    r"""
    Implementation of the inverse 2-dimensional Poisson problem in the square
    domain :math:`[0, 1] \times [0, 1]`,
    with unknown parameter domain :math:`[-1, 1] \times [-1, 1]`.
    """

    output_variables = ["u"]
    x_min, x_max = -2, 2
    y_min, y_max = -2, 2
    spatial_domain = CartesianDomain({"x": [x_min, x_max], "y": [y_min, y_max]})
    unknown_parameter_domain = CartesianDomain({"mu1": [-1, 1], "mu2": [-1, 1]})

    domains = {
        "g1": CartesianDomain({"x": [x_min, x_max], "y": y_max}),
        "g2": CartesianDomain({"x": [x_min, x_max], "y": y_min}),
        "g3": CartesianDomain({"x": x_max, "y": [y_min, y_max]}),
        "g4": CartesianDomain({"x": x_min, "y": [y_min, y_max]}),
        "D": CartesianDomain({"x": [x_min, x_max], "y": [y_min, y_max]}),
    }

    conditions = {
        "g1": Condition(domain="g1", equation=FixedValue(0.0)),
        "g2": Condition(domain="g2", equation=FixedValue(0.0)),
        "g3": Condition(domain="g3", equation=FixedValue(0.0)),
        "g4": Condition(domain="g4", equation=FixedValue(0.0)),
        "D": Condition(domain="D", equation=Equation(laplace_equation)),
        "data": Condition(input=data_input, target=data_output),
    }


problem = Poisson()


# Then, we define the neural network model we want to use. Here we used a model which imposes hard constrains on the boundary conditions, as also done in the Wave tutorial!

# In[24]:


model = FeedForward(
    layers=[20, 20, 20],
    func=torch.nn.Softplus,
    output_dimensions=len(problem.output_variables),
    input_dimensions=len(problem.input_variables),
)


# After that, we discretize the spatial domain.

# In[25]:


problem.discretise_domain(20, "grid", domains=["D"])
problem.discretise_domain(
    1000,
    "random",
    domains=["g1", "g2", "g3", "g4"],
)


# Here, we define a simple callback for the trainer. We use this callback to save the parameters predicted by the neural network during the training. The parameters are saved every 100 epochs as `torch` tensors in a specified directory (`tmp_dir` in our case).
# The goal is to read the saved parameters after training and plot their trend across the epochs.

# In[26]:


# temporary directory for saving logs of training
tmp_dir = "tmp_poisson_inverse"


class SaveParameters(Callback):
    """
    Callback to save the parameters of the model every 100 epochs.
    """

    def on_train_epoch_end(self, trainer, __):
        if trainer.current_epoch % 100 == 99:
            torch.save(
                trainer.solver.problem.unknown_parameters,
                "{}/parameters_epoch{}".format(tmp_dir, trainer.current_epoch),
            )


# Then, we define the `PINN` object and train the solver using the `Trainer`.

# In[27]:


max_epochs = 1500
pinn = PINN(
    problem, model, optimizer=TorchOptimizer(torch.optim.Adam, lr=0.005)
)
# define the trainer for the solver
trainer = Trainer(
    solver=pinn,
    accelerator="cpu",
    max_epochs=max_epochs,
    default_root_dir=tmp_dir,
    enable_model_summary=False,
    callbacks=[SaveParameters()],
    train_size=1.0,
    val_size=0.0,
    test_size=0.0,
)
trainer.train()


# One can now see how the parameters vary during the training by reading the saved solution and plotting them. The plot shows that the parameters stabilize to their true value before reaching the epoch $1000$!

# In[28]:


epochs_saved = range(99, max_epochs, 100)
parameters = torch.empty((int(max_epochs / 100), 2))
for i, epoch in enumerate(epochs_saved):
    params_torch = torch.load(
        "{}/parameters_epoch{}".format(tmp_dir, epoch), weights_only=False
    )
    for e, var in enumerate(pinn.problem.unknown_variables):
        parameters[i, e] = params_torch[var].data

# Plot parameters
plt.close()
plt.plot(epochs_saved, parameters[:, 0], label="mu1", marker="o")
plt.plot(epochs_saved, parameters[:, 1], label="mu2", marker="s")
plt.ylim(-1, 1)
plt.grid()
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Parameter value")
plt.show()


# ## What's next?
#
# We have shown the basic usage PINNs in inverse problem modelling, further extensions include:
#
# 1. Train using different Physics Informed strategies
#
# 2. Try on more complex problems
#
# 3. Many more...
