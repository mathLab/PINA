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


Trainer, Dataset and Datamodule
--------------------------------
.. toctree::
    :titlesonly:

    Trainer <trainer.rst>
    Dataset <data/dataset.rst>
    DataModule <data/data_module.rst>

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

    FeedForward <model/feed_forward.rst>
    MultiFeedForward <model/multi_feed_forward.rst>
    ResidualFeedForward <model/residual_feed_forward.rst>
    Spline <model/spline.rst>
    DeepONet <model/deeponet.rst>
    MIONet <model/mionet.rst>
    KernelNeuralOperator <model/kernel_neural_operator.rst>
    FourierIntegralKernel <model/fourier_integral_kernel.rst>
    FNO <model/fourier_neural_operator.rst>
    AveragingNeuralOperator <model/average_neural_operator.rst>
    LowRankNeuralOperator <model/low_rank_neural_operator.rst>
    GraphNeuralOperator <model/>

Blocks
-------------

.. toctree::
    :titlesonly:

    Residual Block <model/block/residual.rst>
    EnhancedLinear Block <model/block/enhanced_linear.rst>
    Spectral Convolution Block <model/block/spectral.rst>
    Fourier Block <model/block/fourier_block.rst>
    Averaging Block <model/block/average_neural_operator_block.rst>
    Low Rank Block <model/block/low_rank_block.rst>
    Continuous Convolution Block <model/block/convolution.rst>


Reduction and Embeddings
--------------------------

.. toctree::
    :titlesonly:

    Proper Orthogonal Decomposition <model/block/pod_block.rst>
    Periodic Boundary Condition Embedding <model/block/pbc_embedding.rst>
    Fourier Feature Embedding <model/block/fourier_embedding.rst>
    Radial Basis Function Interpolation <model/block/rbf_block.rst>


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


Equations
-------------

.. toctree::
    :titlesonly:

    EquationInterface <equation.equation_interface.rst>
    Equation <equation.equation.rst>
    SystemEquation <equation.system_equation.rst>
    Equation Factory <equation.equation_factory.rst>


Differential Operators
-------------------------

.. toctree::
    :titlesonly:

    Equations <equations.rst>
    Differential Operators <operators.rst>


Problems
--------------

.. toctree::
    :titlesonly:

    AbstractProblem <problem/abstractproblem.rst>
    SpatialProblem <problem/spatialproblem.rst>
    TimeDependentProblem <problem/timedepproblem.rst>
    ParametricProblem <problem/parametricproblem.rst>

Geometrical Domains
---------------------

.. toctree::
    :titlesonly:

    Domain <domain/domain.rst>
    CartesianDomain <domain/cartesian.rst>
    EllipsoidDomain <domain/ellipsoid.rst>
    SimplexDomain <domain/simplex.rst>

Domain Operations
------------------

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

Losses and Weightings
---------------------

.. toctree::
    :titlesonly:

    LossInterface <loss/loss_interface.rst>
    LpLoss <loss/lploss.rst>
    PowerLoss <loss/powerloss.rst>
    WeightingInterface <loss/weighting_interface.rst>
    ScalarWeighting <loss/scalar_weighting.rst>
