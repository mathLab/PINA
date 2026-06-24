:html_theme.sidebar_secondary.remove:

Welcome to PINA's documentation!
=======================================

.. raw:: html

   <div class="hero">

   <img src="_static/PINA_logo.png" alt="PINA" style="max-width: 320px;"/>

   <p class="tagline">A Unified Framework for Scientific Machine Learning</p>

   <a href="_installation.html" class="btn btn-primary" style="margin: 0.25rem;">Get Started</a>
   <a href="_quickstart.html" class="btn btn-primary" style="margin: 0.25rem;">Quickstart</a>
   <a href="https://github.com/mathLab/PINA" class="btn btn-primary" style="margin: 0.25rem;">View on GitHub</a>

   </div>

.. grid:: 1 1 1 3
   :gutter: 2

   .. grid-item-card:: 🧩 Modular Architecture
      :padding: 3
      :class-body: text-center

      Designed with composable abstractions — plug, replace, or extend components freely.

   .. grid-item-card:: 🚀 Scalable Performance
      :padding: 3
      :class-body: text-center

      Native multi-device training with minimal overhead for large-scale problems.

   .. grid-item-card:: ⚙️ Highly Flexible
      :padding: 3
      :class-body: text-center

      Full automation or granular control — PINA adapts to your workflow.

**PINA** is an open-source Python library designed to simplify and accelerate
the development of Scientific Machine Learning (SciML) solutions.
Built on top of `PyTorch <https://pytorch.org/>`_, `PyTorch Lightning <https://lightning.ai/docs/pytorch/stable/>`_,
and `PyTorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/>`_,
PINA provides an intuitive framework for defining, experimenting with,
and solving complex problems using Neural Networks,
Physics-Informed Neural Networks (PINNs), Neural Operators, and more.

.. tab-set::

   .. tab-item:: Physics-Informed NN

      .. code-block:: python

         from pina import Condition, Trainer
         from pina.problem import SpatialProblem
         from pina.solver import PINN
         from pina.model import FeedForward
         from pina.equation import Equation, FixedValue
         from pina.domain import CartesianDomain

         class PoissonProblem(SpatialProblem):
             output_variables = ["u"]
             domains = {"domain": CartesianDomain({"x": [0, 1], "y": [0, 1]}),
                        "boundary": CartesianDomain({"x": [0, 1], "y": [0, 1]})}
             conditions = {"boundary": Condition(domain="boundary",
                                                  equation=FixedValue(0)),
                           "domain": Condition(domain="domain",
                                                equation=Equation("d2(u,x)+d2(u,y)+sin(pi*x)*sin(pi*y)=0"))}

         problem = PoissonProblem()
         model = FeedForward(input_dimensions=2, output_dimensions=1)
         pinn = PINN(problem=problem, model=model)
         trainer = Trainer(solver=pinn, max_epochs=1000)
         trainer.train()

   .. tab-item:: Neural Operator

      .. code-block:: python

         from pina import Condition, Trainer
         from pina.problem import AbstractProblem
         from pina.solver import SupervisedSolver
         from pina.model import FNO

         class DarcyProblem(AbstractProblem):
             output_variables = ["u"]
             conditions = {"data": Condition(input=("x", "y"), target="u")}

         problem = DarcyProblem()
         fno = FNO(input_dimensions=3, output_dimensions=1, n_modes=[16, 16])
         solver = SupervisedSolver(problem=problem, model=fno)
         trainer = Trainer(solver=solver, max_epochs=500)
         trainer.train()

   .. tab-item:: Supervised

      .. code-block:: python

         from pina import Condition, Trainer
         from pina.problem import AbstractProblem
         from pina.solver import SupervisedSolver
         from pina.model import FeedForward

         class RegressionProblem(AbstractProblem):
             output_variables = ["y"]
             conditions = {"data": Condition(input="x", target="y")}

         problem = RegressionProblem()
         model = FeedForward(input_dimensions=1, output_dimensions=1)
         solver = SupervisedSolver(problem=problem, model=model)
         trainer = Trainer(solver=solver, max_epochs=200)
         trainer.train()

.. admonition:: What's New
   :class: tip

   * **Jan 2026**: Added Equivariant Graph Neural Operator and GAROM solvers.
   * **Oct 2025**: New Self-Adaptive PINN and CausalPINN solvers available.
   * **Jul 2025**: Major API overhaul with simplified :class:`~pina.condition.Condition` interface.

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: 📖 Installation
      :link: _installation.html
      :class-card: sd-shadow-sm

      Install PINA via pip, from source, or with extras.

   .. grid-item-card:: 🚀 Quickstart
      :link: _quickstart.html
      :class-card: sd-shadow-sm

      Get up and running in 5 minutes.

   .. grid-item-card:: 📚 API Reference
      :link: _rst/_code.html
      :class-card: sd-shadow-sm

      Complete API docs for problems, solvers, models, and more.

   .. grid-item-card:: 🎓 Tutorials
      :link: _tutorial.html
      :class-card: sd-shadow-sm

      Step-by-step tutorials covering PINNs, Neural Operators, and more.

   .. grid-item-card:: 📄 Cite PINA
      :link: _cite.html
      :class-card: sd-shadow-sm

      Citation and BibTeX for academic use.

   .. grid-item-card:: 🤝 Contributing
      :link: _contributing.html
      :class-card: sd-shadow-sm

      Guide for contributors, bug reports, and PRs.

   .. grid-item-card:: 👥 Team
      :link: _team.html
      :class-card: sd-shadow-sm

      Meet the PINA team and our funding sources.

   .. grid-item-card:: 📜 License
      :link: _LICENSE.html
      :class-card: sd-shadow-sm

      Open-source license information.

.. toctree::
   :hidden:
   :maxdepth: 1

   Quickstart <_quickstart>
   API <_rst/_code>
   Tutorials <_tutorial>
   Contributing <_contributing>
   Team & Foundings <_team.rst>
   Installing <_installation>
   Cite PINA <_cite.rst>
   License <_LICENSE.rst>
