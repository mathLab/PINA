#!/usr/bin/env python
# coding: utf-8

# # Tutorial: One dimensional Helmholtz equation using Periodic Boundary Conditions
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mathLab/PINA/blob/master/tutorials/tutorial9/tutorial.ipynb)
#
# This tutorial presents how to solve with Physics-Informed Neural Networks (PINNs)
# a one dimensional Helmholtz equation with periodic boundary conditions (PBC).
# We will train with standard PINN's training by augmenting the input with
# periodic expansion as presented in [*An expert’s guide to training
# physics-informed neural networks*](
# https://arxiv.org/abs/2308.08468).
#
# First of all, some useful imports.

# In[14]:


## routine needed to run the notebook on Google Colab
try:
    import google.colab

    IN_COLAB = True
except:
    IN_COLAB = False
if IN_COLAB:
    get_ipython().system('pip install "pina-mathlab"')

import torch
import matplotlib.pyplot as plt
import warnings

from pina import Condition, Trainer
from pina.problem import SpatialProblem
from pina.operator import laplacian
from pina.model import FeedForward
from pina.model.block import PeriodicBoundaryEmbedding  # The PBC module
from pina.solver import PINN
from pina.domain import CartesianDomain
from pina.equation import Equation
from pina.callback import MetricTracker

warnings.filterwarnings("ignore")


# ## The problem definition
#
# The one-dimensional Helmholtz problem is mathematically written as:
# $$
# \begin{cases}
# \frac{d^2}{dx^2}u(x) - \lambda u(x) -f(x) &=  0 \quad x\in(0,2)\\
# u^{(m)}(x=0) - u^{(m)}(x=2) &= 0 \quad m\in[0, 1, \cdots]\\
# \end{cases}
# $$
# In this case we are asking the solution to be $C^{\infty}$ periodic with
# period $2$, on the infinite domain $x\in(-\infty, \infty)$. Notice that the
# classical PINN would need infinite conditions to evaluate the PBC loss function,
# one for each derivative, which is of course infeasible...
# A possible solution, diverging from the original PINN formulation,
# is to use *coordinates augmentation*. In coordinates augmentation you seek for
# a coordinates transformation $v$ such that $x\rightarrow v(x)$ such that
# the periodicity condition $ u^{(m)}(x=0) - u^{(m)}(x=2) = 0 \quad m\in[0, 1, \cdots] $ is
# satisfied.
#
# For demonstration purposes, the problem specifics are $\lambda=-10\pi^2$,
# and $f(x)=-6\pi^2\sin(3\pi x)\cos(\pi x)$ which give a solution that can be
# computed analytically $u(x) = \sin(\pi x)\cos(3\pi x)$.

# In[15]:


def helmholtz_equation(input_, output_):
    x = input_.extract("x")
    u_xx = laplacian(output_, input_, components=["u"], d=["x"])
    f = (
        -6.0
        * torch.pi**2
        * torch.sin(3 * torch.pi * x)
        * torch.cos(torch.pi * x)
    )
    lambda_ = -10.0 * torch.pi**2
    return u_xx - lambda_ * output_ - f


class Helmholtz(SpatialProblem):
    output_variables = ["u"]
    spatial_domain = CartesianDomain({"x": [0, 2]})

    # here we write the problem conditions
    conditions = {
        "phys_cond": Condition(
            domain=spatial_domain, equation=Equation(helmholtz_equation)
        ),
    }

    def solution(self, pts):
        return torch.sin(torch.pi * pts) * torch.cos(3.0 * torch.pi * pts)


problem = Helmholtz()

# let's discretise the domain
problem.discretise_domain(200, "grid", domains=["phys_cond"])


# As usual, the Helmholtz problem is written in **PINA** code as a class.
# The equations are written as `conditions` that should be satisfied in the
# corresponding domains. The `solution`
# is the exact solution which will be compared with the predicted one. We used
# Latin Hypercube Sampling for choosing the collocation points.

# ## Solving the problem with a Periodic Network

# Any $\mathcal{C}^{\infty}$ periodic function
# $u : \mathbb{R} \rightarrow \mathbb{R}$ with period
# $L\in\mathbb{N}$ can be constructed by composition of an
# arbitrary smooth function $f : \mathbb{R}^n \rightarrow \mathbb{R}$ and a
# given smooth periodic function $v : \mathbb{R} \rightarrow \mathbb{R}^n$ with
# period $L$, that is $u(x) = f(v(x))$. The formulation is generalizable for
# arbitrary dimension, see [*A method for representing periodic functions and
# enforcing exactly periodic boundary conditions with
# deep neural networks*](https://arxiv.org/pdf/2007.07442).
#
# In our case, we rewrite
# $v(x) = \left[1, \cos\left(\frac{2\pi}{L} x\right),
# \sin\left(\frac{2\pi}{L} x\right)\right]$, i.e
# the coordinates augmentation, and $f(\cdot) = NN_{\theta}(\cdot)$ i.e. a neural
# network. The resulting neural network obtained by composing $f$ with $v$ gives
# the PINN approximate solution, that is
# $u(x) \approx u_{\theta}(x)=NN_{\theta}(v(x))$.
#
# In **PINA** this translates in using the `PeriodicBoundaryEmbedding` layer for $v$, and any
# `pina.model` for $NN_{\theta}$. Let's see it in action!
#

# In[16]:


# we encapsulate all modules in a torch.nn.Sequential container
model = torch.nn.Sequential(
    PeriodicBoundaryEmbedding(input_dimension=1, periods=2),
    FeedForward(
        input_dimensions=3,  # output of PeriodicBoundaryEmbedding = 3 * input_dimension
        output_dimensions=1,
        layers=[10, 10],
    ),
)


# As simple as that! Notice that in higher dimension you can specify different periods
# for all dimensions using a dictionary, e.g. `periods={'x':2, 'y':3, ...}`
# would indicate a periodicity of $2$ in $x$, $3$ in $y$, and so on...
#
# We will now solve the problem as usually with the `PINN` and `Trainer` class, then we will look at the losses using the `MetricTracker` callback from `pina.callback`.

# In[17]:


pinn = PINN(
    problem=problem,
    model=model,
)
trainer = Trainer(
    pinn,
    max_epochs=5000,
    accelerator="cpu",
    enable_model_summary=False,
    callbacks=[MetricTracker()],
    train_size=1.0,
    val_size=0.0,
    test_size=0.0,
)
trainer.train()


# In[18]:


# plot loss
trainer_metrics = trainer.callbacks[0].metrics
plt.plot(
    range(len(trainer_metrics["train_loss"])), trainer_metrics["train_loss"]
)
# plotting
plt.xlabel("epoch")
plt.ylabel("loss")
plt.yscale("log")


# We are going to plot the solution now!

# In[19]:


pts = pinn.problem.spatial_domain.sample(256, "grid", variables="x")
predicted_output = pinn.forward(pts).extract("u").tensor.detach()
true_output = pinn.problem.solution(pts)
plt.plot(pts.extract(["x"]), predicted_output, label="Neural Network solution")
plt.plot(pts.extract(["x"]), true_output, label="True solution")
plt.legend()


# Great, they overlap perfectly! This seems a good result, considering the simple neural network used to some this (complex) problem. We will now test the neural network on the domain $[-4, 4]$ without retraining. In principle the periodicity should be present since the $v$ function ensures the periodicity in $(-\infty, \infty)$.

# In[20]:


# plotting solution
with torch.no_grad():
    # Notice here we put [-4, 4]!!!
    new_domain = CartesianDomain({"x": [0, 4]})
    x = new_domain.sample(1000, mode="grid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # Plot 1
    axes[0].plot(x, problem.solution(x), label=r"$u(x)$", color="blue")
    axes[0].set_title(r"True solution $u(x)$")
    axes[0].legend(loc="upper right")
    # Plot 2
    axes[1].plot(x, pinn(x), label=r"$u_{\theta}(x)$", color="green")
    axes[1].set_title(r"PINN solution $u_{\theta}(x)$")
    axes[1].legend(loc="upper right")
    # Plot 3
    diff = torch.abs(problem.solution(x) - pinn(x))
    axes[2].plot(x, diff, label=r"$|u(x) - u_{\theta}(x)|$", color="red")
    axes[2].set_title(r"Absolute difference $|u(x) - u_{\theta}(x)|$")
    axes[2].legend(loc="upper right")
    # Adjust layout
    plt.tight_layout()
    # Show the plots
    plt.show()


# It is pretty clear that the network is periodic, with also the error following a periodic pattern. Obviously a longer training and a more expressive neural network could improve the results!
#
# ## What's next?
#
# Congratulations on completing the one dimensional Helmholtz tutorial of **PINA**! There are multiple directions you can go now:
#
# 1. Train the network for longer or with different layer sizes and assert the finaly accuracy
#
# 2. Apply the `PeriodicBoundaryEmbedding` layer for a time-dependent problem (see reference in the documentation)
#
# 3. Exploit extrafeature training ?
#
# 4. Many more...
