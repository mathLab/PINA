Code Documentation
==================
Welcome to PINA documentation! Here you can find the modules of the package divided in different sections.
The high-level structure of the package is depicted in our API. 

.. figure:: ../index_files/API_color.png
    :alt: PINA application program interface
    :align: center
    :width: 400


The pipeline to solve differential equations with PINA follows just five steps:

    1. Define the `Problem`_ the user aim to solve
    2. Generate data using built in `Geometries`_, or load high level simulation results as :doc:`LabelTensor <label_tensor>`
    3. Choose or build one or more `Models`_ to solve the problem
    4. Choose a solver across PINA available `Solvers`_, or build one using the :doc:`SolverInterface <solvers/solver_interface>`
    5. Train the model with the PINA :doc:`Trainer <solvers/solver_interface>`, enhance the train with `Callbacks_`

PINA Features
--------------
.. toctree::
    :titlesonly:

    LabelTensor <label_tensor.rst>
    Condition <condition.rst>
    Trainer <trainer.rst>
    Plotter <plotter.rst>


Solvers
--------------

.. toctree::
    :titlesonly:
    
    SolverInterface <solvers/solver_interface.rst>
    PINN <solvers/pinn.rst>
    Supervised solver <solvers/supervised.rst>
    GAROM <solvers/garom.rst>


Models
------------

.. toctree::
    :titlesonly:
    :maxdepth: 5

    Network <models/network.rst>
    KernelNeuralOperator <models/base_no.rst>
    FeedForward <models/fnn.rst>
    MultiFeedForward <models/multifeedforward.rst>
    ResidualFeedForward <models/fnn_residual.rst>
    DeepONet <models/deeponet.rst>
    MIONet <models/mionet.rst>
    FourierIntegralKernel <models/fourier_kernel.rst>
    FNO <models/fno.rst>
    AveragingNeuralOperator <models/avno.rst>
    LowRankNeuralOperator <models/lno.rst>

Layers
-------------

.. toctree::
    :titlesonly:

    Residual layer <layers/residual.rst>
    EnhancedLinear layer <layers/enhanced_linear.rst>
    Spectral convolution <layers/spectral.rst>
    Fourier layers <layers/fourier.rst>
    Averaging layer <layers/avno_layer.rst>
    Low Rank layer <layers/lowrank_layer.rst>
    Continuous convolution <layers/convolution.rst>
    Proper Orthogonal Decomposition <layers/pod.rst>
    Periodic Boundary Condition embeddings <layers/embedding.rst>

Equations and Operators
-------------------------

.. toctree::
    :titlesonly:
    
    Equations <equations.rst>
    Differential Operators <operators.rst>


Problem
--------------

.. toctree::
    :titlesonly:

    AbstractProblem <problem/abstractproblem.rst>
    SpatialProblem <problem/spatialproblem.rst>
    TimeDependentProblem <problem/timedepproblem.rst>
    ParametricProblem <problem/parametricproblem.rst>

Geometries
-----------------

.. toctree::
    :titlesonly:

    Location <geometry/location.rst>
    CartesianDomain <geometry/cartesian.rst>
    EllipsoidDomain <geometry/ellipsoid.rst>
    SimplexDomain <geometry/simplex.rst>

Geometry set operations
------------------------

.. toctree::
    :titlesonly:

    OperationInterface <geometry/operation_interface.rst>
    Union <geometry/union_domain.rst>
    Intersection <geometry/intersection_domain.rst>
    Difference <geometry/difference_domain.rst>
    Exclusion <geometry/exclusion_domain.rst>

Callbacks
--------------------

.. toctree::
    :titlesonly:

    Metric tracking <callbacks/processing_callbacks.rst>
    Optimizer callbacks <callbacks/optimizer_callbacks.rst>
    Adaptive Refinments <callbacks/adaptive_refinment_callbacks.rst>

Metrics and Losses
--------------------

.. toctree::
    :titlesonly:

    LossInterface <loss/loss_interface.rst>
    LpLoss <loss/lploss.rst>
    PowerLoss <loss/powerloss.rst>