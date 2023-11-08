Equations
==========
Equations are used in PINA to make easy the training. During problem definition
each `equation` passed to a `Condition` object must be an `Equation` or `SystemEquation`.
An `Equation` is simply a wrapper over callable python functions, while `SystemEquation` is
a wrapper arounf a list of callable python functions. We provide a wide rage of already implemented
equations to ease the code writing, such as `FixedValue`, `Laplace`, and many more.


.. currentmodule:: pina.equation.equation_interface
.. autoclass:: EquationInterface
   :members:
   :show-inheritance:

.. currentmodule:: pina.equation.equation
.. autoclass:: Equation
   :members:
   :show-inheritance:


.. currentmodule:: pina.equation.system_equation
.. autoclass:: SystemEquation
   :members:
   :show-inheritance:


.. currentmodule:: pina.equation.equation_factory
.. autoclass:: FixedValue
   :members:
   :show-inheritance:

.. autoclass:: FixedGradient
   :members:
   :show-inheritance:

.. autoclass:: FixedFlux
   :members:
   :show-inheritance:

.. autoclass:: Laplace
   :members:
   :show-inheritance: