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
u(x, y, t=0) = \sin(\pi x)\sin(\pi y)\cos(\sqrt{2}\pi), \\\\
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
                          torch.sin(torch.pi*input_.extract(['y'])) *
                          torch.cos(torch.sqrt(torch.tensor(2.))))
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
    
    # defining the problem
    problem = Wave()

After the problem, a **torch** model is needed to solve the PINN. With
the ``Network`` class the users can convert any **torch** model in a
**PINA** model which uses label tensors with a single line of code. We
will write a simple residual network using linear layers.

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
            
            self.residual = torch.nn.Sequential(torch.nn.Linear(3, 16),
                                                torch.nn.Tanh(),
                                                torch.nn.Linear(16, 3),
                                                torch.nn.Tanh())  
            
            self.mlp = torch.nn.Sequential(torch.nn.Linear(3, 24),
                                           torch.nn.Tanh(),
                                            torch.nn.Linear(24, 1))
        def forward(self, x):
            residual_x = self.residual(x)
            return self.mlp(x+residual_x)
        
    model = Network(model=TorchNet(),
                    input_variables=problem.input_variables, 
                    output_variables=problem.output_variables)

In this tutorial, the neural network is trained for 1500 epochs with a
learning rate of 0.008. These parameters can be modified as desired. We
highlight that the generation of the sampling points and the train is
here encapsulated within the function ``generate_samples_and_train``,
but only for saving some lines of code in the next cells; that function
is not mandatory in the **PINA** framework.

.. code:: ipython3

    def generate_samples_and_train(model, problem):
        pinn = PINN(problem, model, lr=0.008)
        pinn.span_pts(15, 'grid', locations=['D'])
        pinn.span_pts(20, 'grid', locations=['gamma1', 'gamma2', 'gamma3', 'gamma4'])
        pinn.span_pts(80, 'grid', locations=['t0'])
        pinn.train(1500, 250)
        return pinn
    
    
    pinn = generate_samples_and_train(model, problem)


.. parsed-literal::

                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di t0initial_co Dwave_equati 
    [epoch 00000] 3.696696e-01 2.948399e-02 1.348359e-01 1.414441e-01 2.550657e-02 2.666699e-02 1.173203e-02 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di t0initial_co Dwave_equati 
    [epoch 00001] 1.295608e-01 7.026880e-03 4.988546e-02 4.242127e-02 8.680532e-03 1.438522e-02 7.161420e-03 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di t0initial_co Dwave_equati 
    [epoch 00250] 4.090840e-03 2.548697e-04 1.971033e-04 3.415658e-04 1.860978e-04 3.089011e-03 2.219297e-05 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di t0initial_co Dwave_equati 
    [epoch 00500] 3.994201e-03 2.137229e-04 2.208564e-04 2.831228e-04 2.112072e-04 3.043587e-03 2.170474e-05 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di t0initial_co Dwave_equati 
    [epoch 00750] 3.892737e-03 1.960989e-04 2.171265e-04 2.855759e-04 2.186773e-04 2.940620e-03 3.463851e-05 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di t0initial_co Dwave_equati 
    [epoch 01000] 3.608999e-03 1.350552e-04 2.278226e-04 3.551233e-04 2.776817e-04 2.564327e-03 4.898960e-05 
                  sum          gamma1nil_di gamma2nil_di gamma3nil_di gamma4nil_di t0initial_co Dwave_equati 
    [epoch 01250] 3.472723e-03 1.203779e-04 2.665457e-04 3.634643e-04 3.228022e-04 2.377785e-03 2.174835e-05 
    [epoch 01500] 3.349001e-03 1.410526e-04 2.640025e-04 3.414921e-04 3.039106e-04 2.271722e-03 2.682119e-05 


After the training is completed one can now plot some results using the
``Plotter`` class of **PINA**.

.. code:: ipython3

    plotter = Plotter()
    
    # plotting at fixed time t = 0.5
    plotter.plot(pinn, fixed_variables={'t' : 0.5})




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

