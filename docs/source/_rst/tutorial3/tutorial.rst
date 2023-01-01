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
            'gamma1': Condition(Span({'x': [0, 1], 'y':  1, 't': [0, 1]}), nil_dirichlet),
            'gamma2': Condition(Span({'x': [0, 1], 'y': 0, 't': [0, 1]}), nil_dirichlet),
            'gamma3': Condition(Span({'x':  1, 'y': [0, 1], 't': [0, 1]}), nil_dirichlet),
            'gamma4': Condition(Span({'x': 0, 'y': [0, 1], 't': [0, 1]}), nil_dirichlet),
            't0': Condition(Span({'x': [0, 1], 'y': [0, 1], 't': 0}), initial_condition),
            'D': Condition(Span({'x': [0, 1], 'y': [0, 1], 't': [0, 1]}), wave_equation),
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
    [epoch 00000] 4.567502e-01 2.847714e-02 1.962997e-02 9.094939e-03 1.247287e-02 3.838658e-01 3.209481e-03 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di t0initial_co Dwave_equati 
    [epoch 00001] 4.184132e-01 1.914901e-02 2.436301e-02 8.384322e-03 1.077990e-02 3.530422e-01 2.694697e-03 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di t0initial_co Dwave_equati 
    [epoch 00150] 1.694410e-01 9.840883e-03 1.117415e-02 1.140828e-02 1.003646e-02 1.260622e-01 9.190784e-04 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di t0initial_co Dwave_equati 
    [epoch 00300] 1.666860e-01 9.847926e-03 1.122043e-02 1.142906e-02 9.706282e-03 1.237589e-01 7.233715e-04 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di t0initial_co Dwave_equati 
    [epoch 00450] 1.564735e-01 8.579318e-03 1.203290e-02 1.264551e-02 8.249855e-03 1.136869e-01 1.279038e-03 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di t0initial_co Dwave_equati 
    [epoch 00600] 1.281068e-01 5.976059e-03 1.463099e-02 1.191054e-02 7.087692e-03 8.658079e-02 1.920737e-03 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di t0initial_co Dwave_equati 
    [epoch 00750] 7.482838e-02 5.880896e-03 1.912235e-02 5.754319e-03 4.252454e-03 3.697925e-02 2.839110e-03 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di t0initial_co Dwave_equati 
    [epoch 00900] 3.109156e-02 2.877797e-03 5.560369e-03 3.611543e-03 3.818088e-03 1.117986e-02 4.043903e-03 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di t0initial_co Dwave_equati 
    [epoch 01050] 1.969596e-02 2.598281e-03 3.658714e-03 3.426491e-03 3.696677e-03 4.037755e-03 2.278043e-03 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di t0initial_co Dwave_equati 
    [epoch 01200] 1.625224e-02 2.496960e-03 3.069649e-03 3.198287e-03 3.420298e-03 2.728654e-03 1.338392e-03 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di t0initial_co Dwave_equati 
    [epoch 01350] 1.430180e-02 2.350929e-03 2.700139e-03 2.961276e-03 3.141905e-03 2.189825e-03 9.577314e-04 
    [epoch 01500] 1.293717e-02 2.182199e-03 2.440975e-03 2.706538e-03 2.904802e-03 1.891113e-03 8.115429e-04 


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
