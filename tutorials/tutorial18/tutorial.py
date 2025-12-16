#!/usr/bin/env python
# coding: utf-8

# # Tutorial: Introduction to Solver classes
# 
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mathLab/PINA/blob/master/tutorials/tutorial18/tutorial.ipynb)
# 
# In this tutorial, we will explore the Solver classes in PINA, that are the core components for optimizing models. Solvers are designed to manage and execute the optimization process, providing the flexibility to work with various types of neural networks and loss functions. We will show how to use this class to select and implement different solvers, such as Supervised Learning, Physics-Informed Neural Networks (PINNs), and Generative Learning solvers. By the end of this tutorial, you'll be equipped to easily choose and customize solvers for your own tasks, streamlining the model training process.
# 
# ## Introduction to Solvers
# 
# [`Solvers`](https://mathlab.github.io/PINA/_rst/_code.html#solvers) are versatile objects in PINA designed to manage the training and optimization of machine learning models. They handle key components of the learning process, including:
# 
# - Loss function minimization  
# - Model optimization (optimizer, schedulers)
# - Validation and testing workflows
# 
# PINA solvers are built on top of the [PyTorch Lightning `LightningModule`](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html), which provides a structured and scalable training framework. This allows solvers to leverage advanced features such as distributed training, early stopping, and logging — all with minimal setup.
# 
# ## Solvers Hierarchy: Single and MultiSolver
# 
# PINA provides two main abstract interfaces for solvers, depending on whether the training involves a single model or multiple models. These interfaces define the base functionality that all specific solver implementations inherit from.
# 
# ### 1. [`SingleSolverInterface`](https://mathlab.github.io/PINA/_rst/solver/solver_interface.html)
# 
# This is the abstract base class for solvers that train **a single model**, such as in standard supervised learning or physics-informed training. All specific solvers (e.g., `SupervisedSolver`, `PINN`) inherit from this interface.
# 
# **Arguments:**
# - `problem` – The problem to be solved.
# - `model` – The neural network model.
# - `optimizer` – Defaults to `torch.optim.Adam` if not provided.
# - `scheduler` – Defaults to `torch.optim.lr_scheduler.ConstantLR`.
# - `weighting` – Optional loss weighting schema., see [here](https://mathlab.github.io/PINA/_rst/_code.html#losses-and-weightings). We weight already for you!
# - `use_lt` – Whether to use LabelTensors as input.
# 
# ---
# 
# ### 2. [`MultiSolverInterface`](https://mathlab.github.io/PINA/_rst/solver/multi_solver_interface.html)
# 
# This is the abstract base class for solvers involving **multiple models**, such as in GAN architectures or ensemble training strategies. All multi-model solvers (e.g., `DeepEnsemblePINN`, `GAROM`) inherit from this interface.
# 
# **Arguments:**
# - `problem` – The problem to be solved.
# - `models` – The model or models used for training.
# - `optimizers` – Defaults to `torch.optim.Adam`.
# - `schedulers` – Defaults to `torch.optim.lr_scheduler.ConstantLR`.
# - `weightings` – Optional loss weighting schema, see [here](https://mathlab.github.io/PINA/_rst/_code.html#losses-and-weightings). We weight already for you!
# - `use_lt` – Whether to use LabelTensors as input.
# 
# ---
# 
# These base classes define the structure and behavior of solvers in PINA, allowing you to create customized training strategies while leveraging PyTorch Lightning's features under the hood. 
# 
# These classes are used to define the backbone, i.e. setting the problem, the model(s), the optimizer(s) and scheduler(s), but miss a key component the `optimization_cycle` method.
# 
# 
# ## Optimization Cycle
# The `optimization_cycle` method is the core function responsible for computing losses for **all conditions** in a given training batch. Each condition (e.g. initial condition, boundary condition, PDE residual) contributes its own loss, which is tracked and returned in a dictionary. This method should return a dictionary mapping **condition names** to their respective **scalar loss values**.
# 
# For supervised learning tasks, where each condition consists of an input-target pair, for example, the `optimization_cycle` may look like this:
# 
# ```python
# def optimization_cycle(self, batch):
#     """
#     The optimization cycle for Supervised solvers.
#     Computes loss for each condition in the batch.
#     """
#     condition_loss = {}
#     for condition_name, data in batch:
#         condition_loss[condition_name] = self.loss_data(
#             input=data["input"], target=data["target"]
#         )
#     return condition_loss
# ```
# In PINA, a **batch** is structured as a list of tuples, where each tuple corresponds to a specific training condition. Each tuple contains:
# 
# - The **name of the condition**
# - A **dictionary of data** associated with that condition
# 
# for example:
# 
# ```python
# batch = [
#     ("condition1", {"input": ..., "target": ...}),
#     ("condition2", {"input": ..., "equation": ...}),
#     ("condition3", {"input": ..., "target": ...}),
# ]
# ```
# 
# Fortunately, you don't need to implement the `optimization_cycle` yourself in most cases — PINA already provides default implementations tailored to common solver types. These implementations are available through the solver interfaces and cover various training strategies.
# 
# 1. [`PINNInterface`](https://mathlab.github.io/PINA/_rst/solver/physics_informed_solver/pinn_interface.html)  
#    Implements the optimization cycle for **physics-based solvers** (e.g., PDE residual minimization) as well as other useful methods to compute PDE residuals.  
#    ➤ [View method](https://mathlab.github.io/PINA/_rst/solver/physics_informed_solver/pinn_interface.html#pina.solver.physics_informed_solver.pinn_interface.PINNInterface.optimization_cycle)
# 
# 2. [`SupervisedSolverInterface`](https://mathlab.github.io/PINA/_rst/solver/supervised_solver/supervised_solver_interface.html)  
#    Defines the optimization cycle for **supervised learning tasks**, including traditional regression and classification.  
#    ➤ [View method](https://mathlab.github.io/PINA/_rst/solver/supervised_solver/supervised_solver_interface.html#pina.solver.supervised_solver.supervised_solver_interface.SupervisedSolverInterface.optimization_cycle)
# 
# 3. [`DeepEnsembleSolverInterface`](https://mathlab.github.io/PINA/_rst/solver/ensemble_solver/ensemble_solver_interface.html)  
#    Provides the optimization logic for **deep ensemble methods**, commonly used for uncertainty quantification or robustness.  
#    ➤ [View method](https://mathlab.github.io/PINA/_rst/solver/ensemble_solver/ensemble_solver_interface.html#pina.solver.ensemble_solver.ensemble_solver_interface.DeepEnsembleSolverInterface.optimization_cycle)
# 
# These ready-to-use implementations ensure that your solvers are properly structured and compatible with PINA’s training workflow. You can also inherit and override them to fit more specialized needs. They only require, the following arguments:
# **Arguments:**
# - `problem` – The problem to be solved.
# - `loss` - The loss to be minimized
# - `weightings` – Optional loss weighting schema.
# - `use_lt` – Whether to use LabelTensors as input.
# 
# ## Structure a Solver with Multiple Inheritance:
# 
# Thanks to PINA’s modular design, creating a custom solver is straightforward using **multiple inheritance**. You can combine different interfaces to define both the **optimization logic** and the **model structure**.
# 
# - **`PINN` Solver**
#   - Inherits from:  
#     - [`PINNInterface`](https://mathlab.github.io/PINA/_rst/solver/physics_informed_solver/pinn_interface.html) → physics-based optimization loop  
#     - [`SingleSolverInterface`](https://mathlab.github.io/PINA/_rst/solver/solver_interface.html) → training a single model
# 
# - **`SupervisedSolver`**
#   - Inherits from:  
#     - [`SupervisedSolverInterface`](https://mathlab.github.io/PINA/_rst/solver/supervised_solver/supervised_solver_interface.html) → data-driven optimization loop  
#     - [`SingleSolverInterface`](https://mathlab.github.io/PINA/_rst/solver/solver_interface.html) → training a single model
# 
# - **`GAROM`** (a variant of GAN)
#   - Inherits from:  
#     - [`SupervisedSolverInterface`](https://mathlab.github.io/PINA/_rst/solver/supervised_solver/supervised_solver_interface.html) → data-driven optimization loop  
#     - [`MultiSolverInterface`](https://mathlab.github.io/PINA/_rst/solver/multi_solver_interface.html) → training multiple models (e.g., generator and discriminator)
# 
# This structure promotes **code reuse** and **extensibility**, allowing you to quickly prototype new solver strategies by reusing core training and optimization logic.
# 
# ## Let's try to build some solvers!
# 
# We will now start building a simple supervised solver in PINA. Let's first import useful modules! 

# In[1]:


## routine needed to run the notebook on Google Colab
try:
    import google.colab

    IN_COLAB = True
except:
    IN_COLAB = False
if IN_COLAB:
    get_ipython().system('pip install "pina-mathlab[tutorial]"')

import warnings
import torch
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from pina import Trainer
from pina.solver import SingleSolverInterface, SupervisedSolverInterface
from pina.model import FeedForward
from pina.problem.zoo import SupervisedProblem


# Since we are using only one model for this task, we will inherit from two base classes:
# 
# - `SingleSolverInterface`: This ensures we are working with a single model.
# - `SupervisedSolverInterface`: This allows us to use supervised learning strategies for training the model.

# In[2]:


class MyFirstSolver(SupervisedSolverInterface, SingleSolverInterface):
    def __init__(
        self,
        problem,
        model,
        loss=None,
        optimizer=None,
        scheduler=None,
        weighting=None,
        use_lt=True,
    ):
        super().__init__(
            model=model,
            problem=problem,
            loss=loss,
            optimizer=optimizer,
            scheduler=scheduler,
            weighting=weighting,
            use_lt=use_lt,
        )


# By default, Python follows a specific method resolution order (MRO) when a class inherits from multiple parent classes. This means that the initialization (`__init__`) method is called based on the order of inheritance.
# 
# Since we inherit from `SupervisedSolverInterface` first, Python will call the `__init__` method from `SupervisedSolverInterface` (initialize `problem`, `loss`, `weighting` and `use_lt`) before calling the `__init__` method from `SingleSolverInterface` (initialize `model`, `optimizer`, `scheduler`). This allows us to customize the initialization process for our custom solver. 
# 
# We will learn a very simple problem, try to learn $y=\sin(x)$.

# In[3]:


# get the data
x = torch.linspace(0, torch.pi, 100).view(-1, 1)
y = torch.sin(x)
# build the problem
problem = SupervisedProblem(x, y)
# build the model
model = FeedForward(1, 1)


# If we now try to initialize the solver `MyFirstSolver` we will get the following error:
# 
# ```python
# ---------------------------------------------------------------------------
# TypeError                                 Traceback (most recent call last)
# Cell In[41], line 1
# ----> 1 MyFirstSolver(problem, model)
# 
# TypeError: Can't instantiate abstract class MyFirstSolver with abstract method loss_data
# ```
# 
# ### Data and Physics Loss
# The error above is because in PINA, all solvers must specify how to compute the loss during training. There are two main types of losses that can be computed, depending on the nature of the problem:
# 
# 1. **`loss_data`**: Computes the **data loss** between the model's output and the true solution. This is typically used in **supervised learning** setups, where we have ground truth data to compare the model's predictions. It expects some `input` (tensor, graph, ...) and a `target` (tensor, graph, ...)
#    
# 2. **`loss_phys`**: Computes the **physics loss** for **physics-informed solvers** (PINNs). This loss is based on the residuals of the governing equations that model physical systems, enforcing the equations during training. It expects some `samples` (`LabelTensor`) and an `equation` (`Equation`)
# 
# Therefore our implementation becomes:

# In[ ]:


class MyFirstSolver(SupervisedSolverInterface, SingleSolverInterface):
    def __init__(
        self,
        problem,
        model,
        loss=None,
        optimizer=None,
        scheduler=None,
        weighting=None,
        use_lt=True,
    ):
        super().__init__(
            model=model,
            problem=problem,
            loss=loss,
            optimizer=optimizer,
            scheduler=scheduler,
            weighting=weighting,
            use_lt=use_lt,
        )

    def loss_data(self, input, target):
        # self.loss stores the loss passed in the init
        network_output = self.forward(input)
        return self.loss(network_output, target)


# initialize (we use plain tensors!)
solver = MyFirstSolver(problem, model, use_lt=False)

# simple training
trainer = Trainer(
    solver, max_epochs=500, train_size=0.8, test_size=0.2, accelerator="cpu"
)
trainer.train()
_ = trainer.test()


# ## A Summary on Solvers
# 
# Solvers in PINA play a critical role in training and optimizing machine learning models, especially when working with complex problems like physics-informed neural networks (PINNs) or standard supervised learning. Here’s a quick recap of the key concepts we've covered:
# 
# 1. **Solver Interfaces**:
#    - **`SingleSolverInterface`**: For solvers using one model (e.g., a standard supervised solver or a single physics-informed model).
#    - **`MultiSolverInterface`**: For solvers using multiple models (e.g., Generative Adversarial Networks (GANs)).
# 
# 2. **Loss Functions**:
#    - **`loss_data`**: Computes the loss for supervised solvers, typically comparing the model's predictions to the true targets.
#    - **`loss_phys`**: Computes the physics loss for PINNs, typically using the residuals of a physical equation to enforce consistency with the physics of the system.
# 
# 3. **Custom Solver Implementation**:
#    - You can create custom solvers by inheriting from base classes such as `SingleSolverInterface`. The **`optimization_cycle`** method must be implemented to define how to compute the loss for each batch.
#    - `SupervisedSolverInterface`, `PINNInterface` already implement the `optimization_cycle` for you!
# 
# 
# By understanding and implementing solvers in PINA, you can build flexible, scalable models that can be optimized both with traditional supervised learning techniques and more specialized, physics-based methods.

# ## What's Next?
# 
# Congratulations on completing the tutorial on solver classes! Now that you have a solid foundation, here are a few directions you can explore:
# 
# 
# 1. **Physics Solvers**: Try to implement your own physics-based solver. Can you do it? This will involve creating a custom loss function that enforces the physics of a given problem insied `loss_phys`.
# 
# 2. **Multi-Model Solvers**: Take it to the next level by exploring multi-model solvers, such as GANs or ensemble-based solvers. You could implement and train models that combine the strengths of multiple neural networks.
# 
# 3. **...and many more!**: There are countless directions to further explore, try to look at our `solver` for example!
# 
# For more resources and tutorials, check out the [PINA Documentation](https://mathlab.github.io/PINA/).
