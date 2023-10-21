Welcome to PINA's documentation!
===================================================

.. figure:: index_files/pina_logo.png
    :align: center
    :width: 150

|


Physics Informed Neural network for Advanced modeling (**PINA**) is
an open-source Python library providing an intuitive interface for
solving differential equations using PINNs, NOs or both together.
Based on `PyTorch <https://pytorch.org/>`_ and `PyTorchLightning <https://lightning.ai/docs/pytorch/stable/>`_,
PINA offers a simple and intuitive way to formalize a specific (differential) problem
and solve it using neural networks . The approximated solution of a differential equation
can be implemented using PINA in a few lines of code thanks to the intuitive and user-friendly interface.

`PyTorchLightning <https://lightning.ai/docs/pytorch/stable/>`_ as backhand is done to offer
professional AI researchers and machine learning engineers the possibility of using advancement
training strategies provided by the library, such as multiple device training, modern model compression techniques,
gradient accumulation, and so on. In addition, it provides the possibility to add arbitrary
self-contained routines (callbacks) to the training for easy extensions without the need to touch the
underlying code.

The high-level structure of the package is depicted in our API. The pipeline to solve differential equations
with PINA follows just five steps: problem definition, model selection, data generation, solver selection, and training.

.. figure:: index_files/API_color.png
    :alt: PINA application program interface
    :align: center
    :width: 500

|

Physics-informed neural network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`PINN <https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125>`_ is a novel approach that
involves neural networks to solve differential equations in an unsupervised manner, while respecting
any given law of physics described by general differential equations. Proposed in "*Physics-informed neural
networks: A deep learning framework for solving forward and inverse problems
involving nonlinear partial differential equations*", such framework aims to
solve problems in a continuous and nonlinear settings. 

Neural operator learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`Neural Operators <https://www.jmlr.org/papers/v24/21-1524.html>`_ is a novel approach involving neural networks
to learn differential operators using supervised learning strategies. By learning the differential operator, the
neural network is able to generalize across different instances of the differential equations (e.g. different forcing
terms), without the need of re-training. 



.. toctree::
   :maxdepth: 2
   :caption: Package Documentation:

   API <_rst/_code>
   Contributing <_rst/_contributing>
   License <_LICENSE.rst>

.. the following is demo content intended to showcase some of the features you can invoke in reStructuredText
.. this can be safely deleted or commented out
.. ........................................................................................

.. toctree::
    :maxdepth: 1
    :numbered:
    :caption: Getting Started:

    Installation <_rst/_installation>
    Tutorials <_rst/_tutorials>
