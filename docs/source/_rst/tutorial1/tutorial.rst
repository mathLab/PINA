Tutorial 1: Physics Informed Neural Networks on PINA
====================================================

In this tutorial, we will demonstrate a typical use case of PINA on a
toy problem. Specifically, the tutorial aims to introduce the following
topics:

-  Defining a PINA Problem,
-  Building a ``pinn`` object,
-  Sampling points in a domain

These are the three main steps needed **before** training a Physics
Informed Neural Network (PINN). We will show each step in detail, and at
the end, we will solve the problem.

PINA Problem
------------

Initialize the ``Problem`` class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Problem definition in the PINA framework is done by building a python
``class``, which inherits from one or more problem classes
(``SpatialProblem``, ``TimeDependentProblem``, ``ParametricProblem``)
depending on the nature of the problem. Below is an example: #### Simple
Ordinary Differential Equation Consider the following:

.. math::


   \begin{equation}
   \begin{cases}
   \frac{d}{dx}u(x) &=  u(x) \quad x\in(0,1)\\
   u(x=0) &= 1 \\
   \end{cases}
   \end{equation}

with the analytical solution :math:`u(x) = e^x`. In this case, our ODE
depends only on the spatial variable :math:`x\in(0,1)` , meaning that
our ``Problem`` class is going to be inherited from the
``SpatialProblem`` class:

.. code:: python

    from pina.problem import SpatialProblem
    from pina import CartesianProblem

    class SimpleODE(SpatialProblem):
        
        output_variables = ['u']
        spatial_domain = CartesianProblem({'x': [0, 1]})

        # other stuff ...

Notice that we define ``output_variables`` as a list of symbols,
indicating the output variables of our equation (in this case only
:math:`u`). The ``spatial_domain`` variable indicates where the sample
points are going to be sampled in the domain, in this case
:math:`x\in[0,1]`.

What about if our equation is also time dependent? In this case, our
``class`` will inherit from both ``SpatialProblem`` and
``TimeDependentProblem``:

.. code:: ipython3

    from pina.problem import SpatialProblem, TimeDependentProblem
    from pina import CartesianDomain
    
    class TimeSpaceODE(SpatialProblem, TimeDependentProblem):
        
        output_variables = ['u']
        spatial_domain = CartesianDomain({'x': [0, 1]})
        temporal_domain = CartesianDomain({'t': [0, 1]})
    
        # other stuff ...

where we have included the ``temporal_domain`` variable, indicating the
time domain wanted for the solution.

In summary, using PINA, we can initialize a problem with a class which
inherits from three base classes: ``SpatialProblem``,
``TimeDependentProblem``, ``ParametricProblem``, depending on the type
of problem we are considering. For reference: \* ``SpatialProblem``
:math:`\rightarrow` a differential equation with spatial variable(s) \*
``TimeDependentProblem`` :math:`\rightarrow` a time-dependent
differential equation \* ``ParametricProblem`` :math:`\rightarrow` a
parametrized differential equation

Write the ``Problem`` class
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the ``Problem`` class is initialized, we need to represent the
differential equation in PINA. In order to do this, we need to load the
PINA operators from ``pina.operators`` module. Again, we'll consider
Equation (1) and represent it in PINA:

.. code:: ipython3

    from pina.problem import SpatialProblem
    from pina.operators import grad
    from pina import Condition, CartesianDomain
    from pina.equation.equation import Equation
    
    import torch
    
    
    class SimpleODE(SpatialProblem):
    
        output_variables = ['u']
        spatial_domain = CartesianDomain({'x': [0, 1]})
    
        # defining the ode equation
        def ode_equation(input_, output_):
    
            # computing the derivative
            u_x = grad(output_, input_, components=['u'], d=['x'])
    
            # extracting the u input variable
            u = output_.extract(['u'])
    
            # calculate the residual and return it
            return u_x - u
    
        # defining the initial condition
        def initial_condition(input_, output_):
            
            # setting the initial value
            value = 1.0
    
            # extracting the u input variable
            u = output_.extract(['u'])
    
            # calculate the residual and return it
            return u - value
    
        # conditions to hold
        conditions = {
            'x0': Condition(location=CartesianDomain({'x': 0.}), equation=Equation(initial_condition)),
            'D': Condition(location=CartesianDomain({'x': [0, 1]}), equation=Equation(ode_equation)),
        }
    
        # sampled points (see below)
        input_pts = None
    
        # defining the true solution
        def truth_solution(self, pts):
            return torch.exp(pts.extract(['x']))

After we define the ``Problem`` class, we need to write different class
methods, where each method is a function returning a residual. These
functions are the ones minimized during PINN optimization, given the
initial conditions. For example, in the domain :math:`[0,1]`, the ODE
equation (``ode_equation``) must be satisfied. We represent this by
returning the difference between subtracting the variable ``u`` from its
gradient (the residual), which we hope to minimize to 0. This is done
for all conditions (``ode_equation``, ``initial_condition``).

Once we have defined the function, we need to tell the neural network
where these methods are to be applied. To do so, we use the
``Condition`` class. In the ``Condition`` class, we pass the location
points and the function we want minimized on those points (other
possibilities are allowed, see the documentation for reference) as
parameters.

Finally, it's possible to define a ``truth_solution`` function, which
can be useful if we want to plot the results and see how the real
solution compares to the expected (true) solution. Notice that the
``truth_solution`` function is a method of the ``PINN`` class, but is
not mandatory for problem definition.

Build the ``PINN`` object
-------------------------

The basic requirements for building a ``PINN`` model are a ``Problem``
and a model. We have just covered the ``Problem`` definition. For the
model parameter, one can use either the default models provided in PINA
or a custom model. We will not go into the details of model definition
(see Tutorial2 and Tutorial3 for more details on model definition).

.. code:: ipython3

    from pina.model import FeedForward
    from pina import PINN
    
    # initialize the problem
    problem = SimpleODE()
    
    # build the model
    model = FeedForward(
        layers=[10, 10],
        func=torch.nn.Tanh,
        output_dimensions=len(problem.output_variables),
        input_dimensions=len(problem.input_variables)
    )
    
    # create the PINN object
    pinn = PINN(problem, model)

Creating the ``PINN`` object is fairly simple. Different optional
parameters include: optimizer, batch size, ... (see
`documentation <https://mathlab.github.io/PINA/>`__ for reference).

Sample points in the domain
---------------------------

Once the ``PINN`` object is created, we need to generate the points for
starting the optimization. To do so, we use the ``sample`` method of the
``CartesianDomain`` class. Below are three examples of sampling methods
on the :math:`[0,1]` domain:

.. code:: ipython3

    # sampling 20 points in [0, 1] through discretization
    pinn.problem.discretise_domain(n=20, mode='grid', variables=['x'])
    
    # sampling 20 points in (0, 1) through latin hypercube samping
    pinn.problem.discretise_domain(n=20, mode='latin', variables=['x'])
    
    # sampling 20 points in (0, 1) randomly
    pinn.problem.discretise_domain(n=20, mode='random', variables=['x'])

Very simple training and plotting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once we have defined the PINA model, created a network, and sampled
points in the domain, we have everything necessary for training a PINN.
To do so, we make use of the ``Trainer`` class.

.. code:: ipython3

    from pina import Trainer
    
    # initialize trainer
    trainer = Trainer(pinn)
    
    # train the model
    trainer.train()


.. parsed-literal::

    /u/n/ndemo/.local/lib/python3.9/site-packages/torch/cuda/__init__.py:546: UserWarning: Can't initialize NVML
      warnings.warn("Can't initialize NVML")
    GPU available: True (cuda), used: True
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    HPU available: False, using: 0 HPUs
    /u/n/ndemo/.local/lib/python3.9/site-packages/lightning/pytorch/loops/utilities.py:72: PossibleUserWarning: `max_epochs` was not set. Setting it to 1000 epochs. To train without an epoch limit, set `max_epochs=-1`.
      rank_zero_warn(
    2023-10-17 10:02:21.318700: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2023-10-17 10:02:21.345355: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2023-10-17 10:02:23.572602: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
    /opt/sissa/apps/intelpython/2022.0.2/intelpython/latest/lib/python3.9/site-packages/scipy/__init__.py:138: UserWarning: A NumPy version >=1.16.5 and <1.23.0 is required for this version of SciPy (detected version 1.26.0)
      warnings.warn(f"A NumPy version >={np_minversion} and <{np_maxversion} is required for this version of "
    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    
      | Name        | Type    | Params
    ----------------------------------------
    0 | _loss       | MSELoss | 0     
    1 | _neural_net | Network | 141   
    ----------------------------------------
    141       Trainable params
    0         Non-trainable params
    141       Total params
    0.001     Total estimated model params size (MB)



.. parsed-literal::

    Training: 0it [00:00, ?it/s]


.. parsed-literal::

    `Trainer.fit` stopped: `max_epochs=1000` reached.

