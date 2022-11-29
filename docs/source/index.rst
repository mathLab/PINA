Welcome to PINA's documentation!
===================================================

PINA is a Python package providing an easy interface to deal with
physics-informed neural networks (PINN) for the approximation of (differential,
nonlinear, ...) functions. Based on Pytorch, PINA offers a simple and intuitive
way to formalize a specific problem and solve it using PINN.

Physics-informed neural network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PINN is a novel approach that involves neural networks to solve supervised
learning tasks while respecting any given law of physics described by general
nonlinear differential equations. Proposed in "Physics-informed neural
networks: A deep learning framework for solving forward and inverse problems
involving nonlinear partial differential equations", such framework aims to
solve problems in a continuous and nonlinear settings.


.. toctree::
   :maxdepth: 1
   :caption: Package Documentation:

   Installation <_rst/installation>
   API <_rst/code>
   Contributing <_rst/contributing>
   License <LICENSE.rst>

.. the following is demo content intended to showcase some of the features you can invoke in reStructuredText
.. this can be safely deleted or commented out
.. ........................................................................................

.. toctree::
    :maxdepth: 1
    :numbered:
    :caption: Tutorials:

    Poisson problem <_rst/tutorial2/tutorial.rst>

.. ........................................................................................

.. toctree::
    :maxdepth: 2
    :numbered:
    :caption: Download

.. ........................................................................................
