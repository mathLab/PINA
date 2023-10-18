Welcome to PINA's documentation!
===================================================

.. figure:: index_files/pina_logo.png
    :align: center
    :width: 150

|


PINA is a Python package providing an easy interface to deal with
physics-informed neural networks (PINN) for the approximation of (differential,
nonlinear, ...) functions. Based on Pytorch, PINA offers a simple and intuitive
way to formalize a specific problem and solve it using PINN. The approximated
solution of a differential equation can be implemented using PINA in a few lines
of code thanks to the intuitive and user-friendly interface.


.. figure:: index_files/API_color.png
    :alt: PINA application program interface
    :align: center
    :width: 500

|

Physics-informed neural network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PINN is a novel approach that involves neural networks to solve supervised
learning tasks while respecting any given law of physics described by general
nonlinear differential equations. Proposed in "Physics-informed neural
networks: A deep learning framework for solving forward and inverse problems
involving nonlinear partial differential equations", such framework aims to
solve problems in a continuous and nonlinear settings. :py:class:`pina.pinn.PINN`


.. toctree::
   :maxdepth: 2
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

    Getting start with PINA <_rst/tutorial1/tutorial.rst>
    Poisson problem <_rst/tutorial2/tutorial.rst>
    Wave equation <_rst/tutorial3/tutorial.rst>
    Continuous Convolutional Filter <_rst/tutorial4/tutorial.rst>
    Fourier Neural Operator <_rst/tutorial5/tutorial.rst>
    Geometry Usage <_rst/tutorial6/tutorial.rst>

.. ........................................................................................

.. toctree::
    :maxdepth: 2
    :numbered:
    :caption: Download

.. ........................................................................................
