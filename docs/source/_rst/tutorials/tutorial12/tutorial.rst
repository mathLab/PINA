Tutorial: The ``Equation`` Class
================================

In this tutorial, we will show how to use the ``Equation`` Class in
PINA. Specifically, we will see how use the Class and its inherited
classes to enforce residuals minimization in PINNs.

Example: The Burgers 1D equation
--------------------------------

We will start implementing the viscous Burgers 1D problem Class,
described as follows:

.. math::


   \begin{equation}
   \begin{cases}
   \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} &= \nu \frac{\partial^2 u}{ \partial x^2}, \quad x\in(0,1), \quad t>0\\
   u(x,0) &= -\sin (\pi x)\\
   u(x,t) &= 0 \quad x = \pm 1\\
   \end{cases}
   \end{equation}

where we set :math:`\nu = \frac{0.01}{\pi}` .

In the class that models this problem we will see in action the
``Equation`` class and one of its inherited classes, the ``FixedValue``
class.

.. code:: ipython3

    #useful imports
    from pina.problem import SpatialProblem, TimeDependentProblem
    from pina.equation import Equation, FixedValue, FixedGradient, FixedFlux
    from pina.geometry import CartesianDomain
    import torch
    from pina.operators import grad, laplacian
    from pina import Condition
    


.. code:: ipython3

    class Burgers1D(TimeDependentProblem, SpatialProblem):
    
        # define the burger equation
        def burger_equation(input_, output_):
            du = grad(output_, input_)
            ddu = grad(du, input_, components=['dudx'])
            return (
                du.extract(['dudt']) +
                output_.extract(['u'])*du.extract(['dudx']) -
                (0.01/torch.pi)*ddu.extract(['ddudxdx'])
            )
    
        # define initial condition
        def initial_condition(input_, output_):
            u_expected = -torch.sin(torch.pi*input_.extract(['x']))
            return output_.extract(['u']) - u_expected
    
        # assign output/ spatial and temporal variables
        output_variables = ['u']
        spatial_domain = CartesianDomain({'x': [-1, 1]})
        temporal_domain = CartesianDomain({'t': [0, 1]})
    
        # problem condition statement
        conditions = {
            'gamma1': Condition(location=CartesianDomain({'x': -1, 't': [0, 1]}), equation=FixedValue(0.)),
            'gamma2': Condition(location=CartesianDomain({'x':  1, 't': [0, 1]}), equation=FixedValue(0.)),
            't0': Condition(location=CartesianDomain({'x': [-1, 1], 't': 0}), equation=Equation(initial_condition)),
            'D': Condition(location=CartesianDomain({'x': [-1, 1], 't': [0, 1]}), equation=Equation(burger_equation)),
        }

The ``Equation`` class takes as input a function (in this case it
happens twice, with ``initial_condition`` and ``burger_equation``) which
computes a residual of an equation, such as a PDE. In a problem class
such as the one above, the ``Equation`` class with such a given input is
passed as a parameter in the specified ``Condition``.

The ``FixedValue`` class takes as input a value of same dimensions of
the output functions; this class can be used to enforced a fixed value
for a specific condition, e.g. Dirichlet boundary conditions, as it
happens for instance in our example.

Once the equations are set as above in the problem conditions, the PINN
solver will aim to minimize the residuals described in each equation in
the training phase.

Available classes of equations include also: - ``FixedGradient`` and
``FixedFlux``: they work analogously to ``FixedValue`` class, where we
can require a constant value to be enforced, respectively, on the
gradient of the solution or the divergence of the solution; -
``Laplace``: it can be used to enforce the laplacian of the solution to
be zero; - ``SystemEquation``: we can enforce multiple conditions on the
same subdomain through this class, passing a list of residual equations
defined in the problem.

Defining a new Equation class
-----------------------------

``Equation`` classes can be also inherited to define a new class. As
example, we can see how to rewrite the above problem introducing a new
class ``Burgers1D``; during the class call, we can pass the viscosity
parameter :math:`\nu`:

.. code:: ipython3

    class Burgers1DEquation(Equation):
        
        def __init__(self, nu = 0.):
            """
            Burgers1D class. This class can be
            used to enforce the solution u to solve the viscous Burgers 1D Equation.
            
            :param torch.float32 nu: the viscosity coefficient. Default value is set to 0.
            """
            self.nu = nu 
        
            def equation(input_, output_):
                    return grad(output_, input_, d='x') +\
                           output_*grad(output_, input_, d='t') -\
                           self.nu*laplacian(output_, input_, d='x')
    
                
            super().__init__(equation)

Now we can just pass the above class as input for the last condition,
setting :math:`\nu= \frac{0.01}{\pi}`:

.. code:: ipython3

    class Burgers1D(TimeDependentProblem, SpatialProblem):
    
        # define initial condition
        def initial_condition(input_, output_):
            u_expected = -torch.sin(torch.pi*input_.extract(['x']))
            return output_.extract(['u']) - u_expected
    
        # assign output/ spatial and temporal variables
        output_variables = ['u']
        spatial_domain = CartesianDomain({'x': [-1, 1]})
        temporal_domain = CartesianDomain({'t': [0, 1]})
    
        # problem condition statement
        conditions = {
            'gamma1': Condition(location=CartesianDomain({'x': -1, 't': [0, 1]}), equation=FixedValue(0.)),
            'gamma2': Condition(location=CartesianDomain({'x':  1, 't': [0, 1]}), equation=FixedValue(0.)),
            't0': Condition(location=CartesianDomain({'x': [-1, 1], 't': 0}), equation=Equation(initial_condition)),
            'D': Condition(location=CartesianDomain({'x': [-1, 1], 't': [0, 1]}), equation=Burgers1DEquation(0.01/torch.pi)),
        }

What’s next?
------------

Congratulations on completing the ``Equation`` class tutorial of
**PINA**! As we have seen, you can build new classes that inherits
``Equation`` to store more complex equations, as the Burgers 1D
equation, only requiring to pass the characteristic coefficients of the
problem. From now on, you can: - define additional complex equation
classes (e.g. ``SchrodingerEquation``, ``NavierStokeEquation``..) -
define more ``FixedOperator`` (e.g. ``FixedCurl``)
