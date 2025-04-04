#!/usr/bin/env python
# coding: utf-8

# # Tutorial: Reduced order models (POD-NN and POD-RBF) for parametric problems
# 
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mathLab/PINA/blob/master/tutorials/tutorial9/tutorial.ipynb)

# The tutorial aims to show how to employ the **PINA** library in order to apply a reduced order modeling technique [1]. Such methodologies have several similarities with machine learning approaches, since the main goal consists in predicting the solution of differential equations (typically parametric PDEs) in a real-time fashion.
# 
# In particular we are going to use the Proper Orthogonal Decomposition with either Radial Basis Function Interpolation (POD-RBF) or Neural Network (POD-NN) [2]. Here we basically perform a dimensional reduction using the POD approach, approximating the parametric solution manifold (at the reduced space) using a regression technique (NN) and comparing it to an RBF interpolation. In this example, we use a simple multilayer perceptron, but the plenty of different architectures can be plugged as well.

# Let's start with the necessary imports.
# It's important to note the minimum PINA version to run this tutorial is the `0.1`.

# In[ ]:


## routine needed to run the notebook on Google Colab
try:
    import google.colab

    IN_COLAB = True
except:
    IN_COLAB = False
if IN_COLAB:
    get_ipython().system('pip install "pina-mathlab"')

get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np
import warnings

from pina import Trainer
from pina.model import FeedForward
from pina.solver import SupervisedSolver
from pina.optim import TorchOptimizer
from pina.problem.zoo import SupervisedProblem
from pina.model.block import PODBlock, RBFBlock

warnings.filterwarnings("ignore")


# We exploit the [Smithers](https://github.com/mathLab/Smithers) library to collect the parametric snapshots. In particular, we use the `NavierStokesDataset` class that contains a set of parametric solutions of the Navier-Stokes equations in a 2D L-shape domain. The parameter is the inflow velocity.
# The dataset is composed by 500 snapshots of the velocity (along $x$, $y$, and the magnitude) and pressure fields, and the corresponding parameter values.
# 
# To visually check the snapshots, let's plot also the data points and the reference solution: this is the expected output of our model.

# In[83]:


from smithers.dataset import NavierStokesDataset

dataset = NavierStokesDataset()

fig, axs = plt.subplots(1, 4, figsize=(14, 3))
for ax, p, u in zip(axs, dataset.params[:4], dataset.snapshots["mag(v)"][:4]):
    ax.tricontourf(dataset.triang, u, levels=16)
    ax.set_title(f"$\mu$ = {p[0]:.2f}")


# The *snapshots* - aka the numerical solutions computed for several parameters - and the corresponding parameters are the only data we need to train the model, in order to predict the solution for any new test parameter. To properly validate the accuracy, we will split the 500 snapshots into the training dataset (90% of the original data) and the testing one (the reamining 10%) inside the `Trainer`.
# 
# It is now time to define the problem!

# In[84]:


u = torch.tensor(dataset.snapshots["mag(v)"]).float()
p = torch.tensor(dataset.params).float()
problem = SupervisedProblem(input_=p, output_=u)


# We can then build a `POD-NN` model (using an MLP architecture as approximation) and compare it with a `POD-RBF` model (using a Radial Basis Function interpolation as approximation).

# ## POD-NN reduced order model

# Let's build the `PODNN` class

# In[85]:


class PODNN(torch.nn.Module):
    """
    Proper orthogonal decomposition with neural network model.
    """

    def __init__(self, pod_rank, layers, func):
        """ """
        super().__init__()

        self.pod = PODBlock(pod_rank)
        self.nn = FeedForward(
            input_dimensions=1,
            output_dimensions=pod_rank,
            layers=layers,
            func=func,
        )

    def forward(self, x):
        """
        Defines the computation performed at every call.

        :param x: The tensor to apply the forward pass.
        :type x: torch.Tensor
        :return: the output computed by the model.
        :rtype: torch.Tensor
        """
        coefficents = self.nn(x)
        return self.pod.expand(coefficents)

    def fit_pod(self, x):
        """
        Just call the :meth:`pina.model.layers.PODBlock.fit` method of the
        :attr:`pina.model.layers.PODBlock` attribute.
        """
        self.pod.fit(x)


# We highlight that the POD modes are directly computed by means of the singular value decomposition (computed over the input data), and not trained using the backpropagation approach. Only the weights of the MLP are actually trained during the optimization loop.

# In[86]:


pod_nn = PODNN(pod_rank=20, layers=[10, 10, 10], func=torch.nn.Tanh)
pod_nn_stokes = SupervisedSolver(
    problem=problem,
    model=pod_nn,
    optimizer=TorchOptimizer(torch.optim.Adam, lr=0.0001),
    use_lt=False,
)


# Before starting we need to fit the POD basis on the training dataset, this can be easily done in PINA as well:

# In[87]:


trainer = Trainer(
    solver=pod_nn_stokes,
    max_epochs=1000,
    batch_size=None,
    accelerator="cpu",
    train_size=0.9,
    val_size=0.0,
    test_size=0.1,
)

# fit the pod basis
trainer.data_module.setup("fit")  # set up the dataset
x_train = trainer.data_module.train_dataset.conditions_dict["data"][
    "target"
]  # extract data for training
pod_nn.fit_pod(x=x_train)

# now train
trainer.train()


# Done! Now that the computational expensive part is over, we can load in future the model to infer new parameters (simply loading the checkpoint file automatically created by `Lightning`) or test its performances. We measure the relative error for the training and test datasets, printing the mean one.

# In[ ]:


# extract train and test data
trainer.data_module.setup("test")  # set up the dataset
p_train = trainer.data_module.train_dataset.conditions_dict["data"]["input"]
u_train = trainer.data_module.train_dataset.conditions_dict["data"]["target"]
p_test = trainer.data_module.test_dataset.conditions_dict["data"]["input"]
u_test = trainer.data_module.test_dataset.conditions_dict["data"]["target"]

# compute statistics
u_test_nn = pod_nn_stokes(p_test)
u_train_nn = pod_nn_stokes(p_train)

relative_error_train = torch.norm(u_train_nn - u_train) / torch.norm(u_train)
relative_error_test = torch.norm(u_test_nn - u_test) / torch.norm(u_test)

print("Error summary for POD-NN model:")
print(f"  Train: {relative_error_train.item():e}")
print(f"  Test:  {relative_error_test.item():e}")


# ## POD-RBF reduced order model

# Then, we define the model we want to use, with the POD (`PODBlock`) and the RBF (`RBFBlock`) objects.

# In[89]:


class PODRBF(torch.nn.Module):
    """
    Proper orthogonal decomposition with Radial Basis Function interpolation model.
    """

    def __init__(self, pod_rank, rbf_kernel):
        """ """
        super().__init__()

        self.pod = PODBlock(pod_rank)
        self.rbf = RBFBlock(kernel=rbf_kernel)

    def forward(self, x):
        """
        Defines the computation performed at every call.

        :param x: The tensor to apply the forward pass.
        :type x: torch.Tensor
        :return: the output computed by the model.
        :rtype: torch.Tensor
        """
        coefficents = self.rbf(x)
        return self.pod.expand(coefficents)

    def fit(self, p, x):
        """
        Call the :meth:`pina.model.layers.PODBlock.fit` method of the
        :attr:`pina.model.layers.PODBlock` attribute to perform the POD,
        and the :meth:`pina.model.layers.RBFBlock.fit` method of the
        :attr:`pina.model.layers.RBFBlock` attribute to fit the interpolation.
        """
        self.pod.fit(x)
        self.rbf.fit(p, self.pod.reduce(x))


# We can then fit the model and ask it to predict the required field for unseen values of the parameters. Note that this model does not need a `Trainer` since it does not include any neural network or learnable parameters.

# In[90]:


pod_rbf = PODRBF(pod_rank=20, rbf_kernel="thin_plate_spline")
pod_rbf.fit(p_train, u_train)


# Compute errors

# In[91]:


u_test_rbf = pod_rbf(p_test)
u_train_rbf = pod_rbf(p_train)

relative_error_train = torch.norm(u_train_rbf - u_train) / torch.norm(u_train)
relative_error_test = torch.norm(u_test_rbf - u_test) / torch.norm(u_test)

print("Error summary for POD-RBF model:")
print(f"  Train: {relative_error_train.item():e}")
print(f"  Test:  {relative_error_test.item():e}")


# ## POD-RBF vs POD-NN

# We can of course also plot the solutions predicted by the `PODRBF` and by the `PODNN` model, comparing them to the original ones. We can note here, in the `PODNN` model and for low velocities, some differences, but improvements can be accomplished thanks to longer training.

# In[92]:


idx = torch.randint(0, len(u_test), (4,))
u_idx_rbf = pod_rbf(p_test[idx])
u_idx_nn = pod_nn_stokes(p_test[idx])


fig, axs = plt.subplots(4, 5, figsize=(14, 9))

relative_error_rbf = np.abs(u_test[idx] - u_idx_rbf.detach())
relative_error_rbf = np.where(
    u_test[idx] < 1e-7, 1e-7, relative_error_rbf / u_test[idx]
)

relative_error_nn = np.abs(u_test[idx] - u_idx_nn.detach())
relative_error_nn = np.where(
    u_test[idx] < 1e-7, 1e-7, relative_error_nn / u_test[idx]
)

for i, (idx_, rbf_, nn_, rbf_err_, nn_err_) in enumerate(
    zip(idx, u_idx_rbf, u_idx_nn, relative_error_rbf, relative_error_nn)
):

    axs[0, 0].set_title(f"Real Snapshots")
    axs[0, 1].set_title(f"POD-RBF")
    axs[0, 2].set_title(f"POD-NN")
    axs[0, 3].set_title(f"Error POD-RBF")
    axs[0, 4].set_title(f"Error POD-NN")

    cm = axs[i, 0].tricontourf(
        dataset.triang, rbf_.detach()
    )  # POD-RBF prediction
    plt.colorbar(cm, ax=axs[i, 0])

    cm = axs[i, 1].tricontourf(
        dataset.triang, nn_.detach()
    )  # POD-NN prediction
    plt.colorbar(cm, ax=axs[i, 1])

    cm = axs[i, 2].tricontourf(dataset.triang, u_test[idx_].flatten())  # Truth
    plt.colorbar(cm, ax=axs[i, 2])

    cm = axs[i, 3].tripcolor(
        dataset.triang, rbf_err_, norm=matplotlib.colors.LogNorm()
    )  # Error for POD-RBF
    plt.colorbar(cm, ax=axs[i, 3])

    cm = axs[i, 4].tripcolor(
        dataset.triang, nn_err_, norm=matplotlib.colors.LogNorm()
    )  # Error for POD-NN
    plt.colorbar(cm, ax=axs[i, 4])

plt.show()


# #### References
# 1. Rozza G., Stabile G., Ballarin F. (2022). Advanced Reduced Order Methods and Applications in Computational Fluid Dynamics, Society for Industrial and Applied Mathematics. 
# 2. Hesthaven, J. S., & Ubbiali, S. (2018). Non-intrusive reduced order modeling of nonlinear problems using neural networks. Journal of Computational Physics, 363, 55-78.
