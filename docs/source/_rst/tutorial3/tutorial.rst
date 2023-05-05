Tutorial 3: resolution of wave equation with custom Network
===========================================================

The problem solution
~~~~~~~~~~~~~~~~~~~~

In this tutorial we present how to solve the wave equation using the
``SpatialProblem`` and ``TimeDependentProblem`` class, and the
``Network`` class for building custom **torch** networks.

The problem is written in the following form:

:raw-latex:`\begin{equation}
\begin{cases}
\Delta u(x,y,t) = \frac{\partial^2}{\partial t^2} u(x,y,t) \quad \text{in } D, \\\\
u(x, y, t=0) = \sin(\pi x)\sin(\pi y), \\\\
u(x, y, t) = 0 \quad \text{on } \Gamma_1 \cup \Gamma_2 \cup \Gamma_3 \cup \Gamma_4,
\end{cases}
\end{equation}`

where :math:`D` is a square domain :math:`[0,1]^2`, and
:math:`\Gamma_i`, with :math:`i=1,...,4`, are the boundaries of the
square, and the velocity in the standard wave equation is fixed to one.

First of all, some useful imports.

.. code:: ipython3

    import torch
    
    from pina.problem import SpatialProblem, TimeDependentProblem
    from pina.operators import nabla, grad
    from pina.model import Network
    from pina import Condition, Span, PINN, Plotter

Now, the wave problem is written in PINA code as a class, inheriting
from ``SpatialProblem`` and ``TimeDependentProblem`` since we deal with
spatial, and time dependent variables. The equations are written as
``conditions`` that should be satisfied in the corresponding domains.
``truth_solution`` is the exact solution which will be compared with the
predicted one.

.. code:: ipython3

    class Wave(TimeDependentProblem, SpatialProblem):
        output_variables = ['u']
        spatial_domain = Span({'x': [0, 1], 'y': [0, 1]})
        temporal_domain = Span({'t': [0, 1]})
    
        def wave_equation(input_, output_):
            u_t = grad(output_, input_, components=['u'], d=['t'])
            u_tt = grad(u_t, input_, components=['dudt'], d=['t'])
            nabla_u = nabla(output_, input_, components=['u'], d=['x', 'y'])
            return nabla_u - u_tt
    
        def nil_dirichlet(input_, output_):
            value = 0.0
            return output_.extract(['u']) - value
    
        def initial_condition(input_, output_):
            u_expected = (torch.sin(torch.pi*input_.extract(['x'])) *
                          torch.sin(torch.pi*input_.extract(['y'])))
            return output_.extract(['u']) - u_expected
    
        conditions = {
            'gamma1': Condition(location=Span({'x': [0, 1], 'y':  1, 't': [0, 1]}), function=nil_dirichlet),
            'gamma2': Condition(location=Span({'x': [0, 1], 'y': 0, 't': [0, 1]}), function=nil_dirichlet),
            'gamma3': Condition(location=Span({'x':  1, 'y': [0, 1], 't': [0, 1]}), function=nil_dirichlet),
            'gamma4': Condition(location=Span({'x': 0, 'y': [0, 1], 't': [0, 1]}), function=nil_dirichlet),
            't0': Condition(location=Span({'x': [0, 1], 'y': [0, 1], 't': 0}), function=initial_condition),
            'D': Condition(location=Span({'x': [0, 1], 'y': [0, 1], 't': [0, 1]}), function=wave_equation),
        }
    
        def wave_sol(self, pts):
            return (torch.sin(torch.pi*pts.extract(['x'])) *
                    torch.sin(torch.pi*pts.extract(['y'])) *
                    torch.cos(torch.sqrt(torch.tensor(2.))*torch.pi*pts.extract(['t'])))
    
        truth_solution = wave_sol
    
    problem = Wave()

After the problem, a **torch** model is needed to solve the PINN. With
the ``Network`` class the users can convert any **torch** model in a
**PINA** model which uses label tensors with a single line of code. We
will write a simple residual network using linear layers. Here we
implement a simple residual network composed by linear torch layers.

This neural network takes as input the coordinates (in this case
:math:`x`, :math:`y` and :math:`t`) and provides the unkwown field of
the Wave problem. The residual of the equations are evaluated at several
sampling points (which the user can manipulate using the method
``span_pts``) and the loss minimized by the neural network is the sum of
the residuals.

.. code:: ipython3

    class TorchNet(torch.nn.Module):
        
        def __init__(self):
            super().__init__()
             
            self.residual = torch.nn.Sequential(torch.nn.Linear(3, 24),
                                                torch.nn.Tanh(),
                                                torch.nn.Linear(24, 3))
            
            self.mlp = torch.nn.Sequential(torch.nn.Linear(3, 64),
                                           torch.nn.Tanh(),
                                           torch.nn.Linear(64, 1))
        def forward(self, x):
            residual_x = self.residual(x)
            return self.mlp(x + residual_x)
    
    # model definition
    model = Network(model = TorchNet(),
                    input_variables=problem.input_variables,
                    output_variables=problem.output_variables,
                    extra_features=None)

In this tutorial, the neural network is trained for 2000 epochs with a
learning rate of 0.001. These parameters can be modified as desired. We
highlight that the generation of the sampling points and the train is
here encapsulated within the function ``generate_samples_and_train``,
but only for saving some lines of code in the next cells; that function
is not mandatory in the **PINA** framework. The training takes
approximately one minute.

.. code:: ipython3

    def generate_samples_and_train(model, problem):
        # generate pinn object
        pinn = PINN(problem, model, lr=0.001)
    
        pinn.span_pts(1000, 'random', locations=['D','t0', 'gamma1', 'gamma2', 'gamma3', 'gamma4'])
        pinn.train(1500, 150)
        return pinn
    
    
    pinn = generate_samples_and_train(model, problem)


.. parsed-literal::

                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di t0initial_co Dwave_equati 
    [epoch 00000] 1.021557e-01 1.350026e-02 4.368403e-03 6.463497e-03 1.698729e-03 5.513944e-02 2.098533e-02 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di t0initial_co Dwave_equati 
    [epoch 00001] 8.096325e-02 7.543423e-03 2.978407e-03 7.128799e-03 2.084145e-03 3.967418e-02 2.155431e-02 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di t0initial_co Dwave_equati 
    [epoch 00150] 4.684930e-02 9.609548e-03 3.093602e-03 7.733506e-03 2.570329e-03 1.896760e-02 4.874712e-03 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di t0initial_co Dwave_equati 
    [epoch 00300] 3.519089e-02 6.642059e-03 2.865276e-03 6.399740e-03 2.900236e-03 1.244203e-02 3.941551e-03 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di t0initial_co Dwave_equati 
    [epoch 00450] 2.766160e-02 5.089254e-03 2.789679e-03 5.370538e-03 3.071685e-03 7.834940e-03 3.505504e-03 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di t0initial_co Dwave_equati 
    [epoch 00600] 2.361075e-02 4.279066e-03 2.785937e-03 4.689044e-03 3.101575e-03 5.907214e-03 2.847910e-03 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di t0initial_co Dwave_equati 
    [epoch 00750] 8.005206e-02 3.891625e-03 2.690672e-03 3.808867e-03 3.402538e-03 6.042966e-03 6.021538e-02 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di t0initial_co Dwave_equati 
    [epoch 00900] 1.892301e-02 3.592897e-03 2.639081e-03 3.797543e-03 2.988781e-03 3.860098e-03 2.044612e-03 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di t0initial_co Dwave_equati 
    [epoch 01050] 1.739456e-02 3.420912e-03 2.557583e-03 3.532733e-03 2.910482e-03 3.114843e-03 1.858010e-03 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di t0initial_co Dwave_equati 
    [epoch 01200] 1.663617e-02 3.213567e-03 2.571464e-03 3.355495e-03 2.749454e-03 3.247283e-03 1.498912e-03 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di t0initial_co Dwave_equati 
    [epoch 01350] 1.551488e-02 3.121611e-03 2.481438e-03 3.141828e-03 2.706321e-03 2.636140e-03 1.427544e-03 
    [epoch 01500] 1.497287e-02 2.974171e-03 2.475442e-03 2.979754e-03 2.593079e-03 2.723322e-03 1.227099e-03 


After the training is completed one can now plot some results using the
``Plotter`` class of **PINA**.

.. code:: ipython3

    plotter = Plotter()
    
    # plotting at fixed time t = 0.6
    plotter.plot(pinn, fixed_variables={'t': 0.6})




.. image:: tutorial_files/tutorial_12_0.png


We can also plot the pinn loss during the training to see the decrease.

.. code:: ipython3

    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(16, 6))
    plotter.plot_loss(pinn, label='Loss')
    
    plt.grid()
    plt.legend()
    plt.show()



.. image:: tutorial_files/tutorial_14_0.png


You can now trying improving the training by changing network, optimizer
and its parameters, changin the sampling points,or adding extra
features!
