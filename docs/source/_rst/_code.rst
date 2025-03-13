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
    2. Generate data using built in `Domains`_, or load high level simulation results as :doc:`LabelTensor <label_tensor>`
    3. Choose or build one or more `Models`_ to solve the problem
    4. Choose a solver across PINA available `Solvers`_, or build one using the :doc:`SolverInterface <solvers/solver_interface>`
    5. Train the model with the PINA :doc:`Trainer <solvers/solver_interface>`, enhance the train with `Callback`_


Training and Datamodules
-------------------------
.. toctree::
    :titlesonly:

    Trainer <trainer.rst>

Data Types
------------
.. toctree::
    :titlesonly:

    LabelTensor <label_tensor.rst>


Conditions
-------------
.. toctree::
    :titlesonly:

    ConditionInterface <condition/condition_interface.rst>
    Condition <condition/condition.rst>
    DataCondition <condition/data_condition.rst>
    DomainEquationCondition <condition/domain_equation_condition.rst>
    InputEquationCondition <condition/input_equation_condition.rst>
    InputTargetCondition <condition/input_target_condition.rst>

Solvers
--------------

.. toctree::
    :titlesonly:

    SolverInterface <solvers/solver_interface.rst>
    PINNInterface <solvers/basepinn.rst>
    PINN <solvers/pinn.rst>
    GPINN <solvers/gpinn.rst>
    CausalPINN <solvers/causalpinn.rst>
    CompetitivePINN <solvers/competitivepinn.rst>
    SAPINN <solvers/sapinn.rst>
    RBAPINN <solvers/rba_pinn.rst>
    Supervised solver <solvers/supervised.rst>
    ReducedOrderModelSolver <solvers/rom.rst>
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
    Spline <models/spline.rst>
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
    Periodic Boundary Condition Embedding <layers/pbc_embedding.rst>
    Fourier Feature Embedding <layers/fourier_embedding.rst>
    Radial Basis Function Interpolation <layers/rbf_layer.rst>

Adaptive Activation Functions
-------------------------------

.. toctree::
    :titlesonly:

    Adaptive Function Interface <adaptive_function/AdaptiveFunctionInterface.rst>
    Adaptive ReLU <adaptive_function/AdaptiveReLU.rst>
    Adaptive Sigmoid <adaptive_function/AdaptiveSigmoid.rst>
    Adaptive Tanh <adaptive_function/AdaptiveTanh.rst>
    Adaptive SiLU <adaptive_function/AdaptiveSiLU.rst>
    Adaptive Mish <adaptive_function/AdaptiveMish.rst>
    Adaptive ELU <adaptive_function/AdaptiveELU.rst>
    Adaptive CELU <adaptive_function/AdaptiveCELU.rst>
    Adaptive GELU <adaptive_function/AdaptiveGELU.rst>
    Adaptive Softmin <adaptive_function/AdaptiveSoftmin.rst>
    Adaptive Softmax <adaptive_function/AdaptiveSoftmax.rst>
    Adaptive SIREN <adaptive_function/AdaptiveSIREN.rst>
    Adaptive Exp <adaptive_function/AdaptiveExp.rst>


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

Domains
-----------------

.. toctree::
    :titlesonly:

    Domain <domain/domain.rst>
    CartesianDomain <domain/cartesian.rst>
    EllipsoidDomain <domain/ellipsoid.rst>
    SimplexDomain <domain/simplex.rst>

domain set operations
------------------------

.. toctree::
    :titlesonly:

    OperationInterface <domain/operation_interface.rst>
    Union <domain/union_domain.rst>
    Intersection <domain/intersection_domain.rst>
    Difference <domain/difference_domain.rst>
    Exclusion <domain/exclusion_domain.rst>

Callback
--------------------

.. toctree::
    :titlesonly:

    Processing callback <callback/processing_callback.rst>
    Optimizer callback <callback/optimizer_callback.rst>
    Refinment callback <callback/adaptive_refinment_callback.rst>
    Weighting callback <callback/linear_weight_update_callback.rst>

Metrics and Losses
--------------------

.. toctree::
    :titlesonly:

    LossInterface <loss/loss_interface.rst>
    LpLoss <loss/lploss.rst>
    PowerLoss <loss/powerloss.rst>
