Code Documentation
==================
Welcome to PINA documentation! Here you can find the modules of the package divided in different sections.

PINA Features
--------------
.. toctree::
    :titlesonly:

    LabelTensor <label_tensor.rst>
    Condition <condition.rst>
    Plotter <plotter.rst>

Problem
--------------

.. toctree::
    :titlesonly:

    AbstractProblem <problem/abstractproblem.rst>
    SpatialProblem <problem/spatialproblem.rst>
    TimeDependentProblem <problem/timedepproblem.rst>
    ParametricProblem <problem/parametricproblem.rst>

Solvers
--------------

.. toctree::
    :titlesonly:
    
    SolverInterface <solvers/solver_interface.rst>
    PINN <solvers/pinn.rst>


Models
------------

.. toctree::
    :titlesonly:

    Network <models/network.rst>
    FeedForward <models/fnn.rst>
    MultiFeedForward <models/multifeedforward.rst>
    ResidualFeedForward <models/fnn_residual.rst>
    DeepONet <models/deeponet.rst>
    MIONet <models/mionet.rst>
    FNO <models/fno.rst>

Layers
-------------

.. toctree::
    :titlesonly:

    ContinuousConv <layers/convolution.rst>
    

Geometries
-----------------

.. toctree::
    :titlesonly:

    Location <geometry/location.rst>
    CartesianDomain <geometry/cartesian.rst>
    EllipsoidDomain <geometry/ellipsoid.rst>
    SimplexDomain <geometry/simplex.rst>


Loss
------

.. toctree::
    :titlesonly:

    LossInterface <loss/loss_interface.rst>
    LpLoss <loss/lploss.rst>
    PowerLoss <loss/powerloss.rst>