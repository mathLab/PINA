.. docmeta::
   :last_reviewed: 2026-06-24


Quickstart
==========

This guide gets you up and running with PINA in 5 minutes. By the end, you will have trained a Physics-Informed Neural Network (PINN) to solve the Poisson equation on a unit square.

Install
-------

.. code-block:: bash

   pip install pina-mathlab

Define a problem
----------------

Every PINA workflow starts by defining a :class:`~pina.problem.spatial_problem.SpatialProblem`.
A problem specifies the output variables, the computational domain, and the conditions
(PDE residual, boundary conditions, initial conditions, data) that the solver must satisfy.

.. code-block:: python

   from pina import Condition
   from pina.problem import SpatialProblem
   from pina.domain import CartesianDomain
   from pina.equation import Equation, FixedValue

   class PoissonProblem(SpatialProblem):
       output_variables = ["u"]

       domains = {
           "domain": CartesianDomain({"x": [0, 1], "y": [0, 1]}),
           "boundary": CartesianDomain({"x": [0, 1], "y": [0, 1]}),
       }

       conditions = {
           "domain": Condition(
               domain="domain",
               equation=Equation("d2(u,x) + d2(u,y) + sin(pi*x)*sin(pi*y) = 0"),
           ),
           "boundary": Condition(
               domain="boundary",
               equation=FixedValue(0.0),
           ),
       }

   problem = PoissonProblem()

Create a model
--------------

Choose a neural network architecture. For standard PINNs, a :class:`~pina.model.feed_forward.FeedForward` (MLP) is a solid starting point.

.. code-block:: python

   from pina.model import FeedForward

   model = FeedForward(
       input_dimensions=2,
       output_dimensions=1,
       inner_size=20,
       n_layers=3,
   )

Train with a solver
-------------------

The :class:`~pina.solver.physics_informed_solver.pinn.PINN` solver wraps the problem and model, and the :class:`~pina._src.core.trainer.Trainer` orchestrates the training loop.

.. code-block:: python

   from pina.solver import PINN
   from pina import Trainer

   pinn = PINN(problem=problem, model=model)
   trainer = Trainer(solver=pinn, max_epochs=1000)
   trainer.train()

Inspect results
---------------

After training, the model stores its solution in the solver. Evaluate at any point:

.. code-block:: python

   import torch

   x = torch.tensor([[0.5, 0.5]], requires_grad=True)
   u_pred = pinn(x)
   print(u_pred)

What's next?
------------

* Walk through the `Introductory Tutorial <tutorial17/tutorial.html>`_ for a deeper introduction.
* Explore the :doc:`API reference </_rst/_code>` for all available components.
* Read the :doc:`tutorials </_tutorial>` for domain-specific guides (Neural Operators, Supervised Learning, etc.).
