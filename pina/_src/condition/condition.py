"""Module for the Condition class."""

from pina._src.condition.input_equation_condition import InputEquationCondition
from pina._src.condition.input_target_condition import InputTargetCondition
from pina._src.condition.time_series_condition import TimeSeriesCondition
from pina._src.condition.data_condition import DataCondition
from pina._src.condition.domain_equation_condition import (
    DomainEquationCondition,
)


class Condition:
    """
    The :class:`Condition` class is a core component of the PINA framework that
    provides a unified interface to define heterogeneous constraints that must
    be satisfied by a :class:`~pina.problem.base_problem.BaseProblem`.

    It encapsulates all types of constraints - physical, boundary, initial, or
    data-driven - that the solver must satisfy during training. The specific
    behavior is inferred from the arguments passed to the constructor.

    Multiple types of conditions can be used within the same problem, allowing
    for a high degree of flexibility in defining complex problems.

    The :class:`Condition` class behavior specializes internally based on the
    arguments provided during instantiation. Depending on the specified keyword
    arguments, the class automatically selects the appropriate internal
    implementation.

    Available `Condition` types:

    - :class:`~pina.condition.input_target_condition.InputTargetCondition`:
      represents a supervised condition defined by both ``input`` and ``target``
      data. The model is trained to reproduce the ``target`` values given the
      ``input``. Supported data types include :class:`torch.Tensor`,
      :class:`~pina.label_tensor.LabelTensor`, :class:`~pina.graph.Graph`, or
      :class:`~torch_geometric.data.Data`. The class automatically selects the
      appropriate implementation based on the types of ``input`` and ``target``.

    - :class:`~pina.condition.domain_equation_condition.DomainEquationCondition`
      : represents a general physics-informed condition defined by a ``domain``
      and an ``equation``. The model learns to minimize the equation residual
      through evaluations performed at points sampled from the specified domain.

    - :class:`~pina.condition.input_equation_condition.InputEquationCondition`:
      represents a general physics-informed condition defined by ``input``
      points and an ``equation``. The model learns to minimize the equation
      residual through evaluations performed at the provided ``input``.
      Supported data types for the ``input`` include :class:`~pina.graph.Graph`
      or :class:`~pina.label_tensor.LabelTensor`. The class automatically
      selects the appropriate implementation based on the types of the
      ``input``.

    - :class:`~pina.condition.time_series_condition.TimeSeriesCondition`:
      represents a condition designed for time series data, where the model is
      trained to capture temporal dependencies and dynamics. It is defined by an
      ``input`` tensor of shape ``[trajectories, time_steps, *features]``
      containing time series data. Supported data types for the ``input``
      include class:`~pina.label_tensor.LabelTensor` or :class:`torch.Tensor`.
      The class automatically selects the appropriate implementation based on
      the type of the ``input``.

    - :class:`~pina.condition.data_condition.DataCondition`: represents an
      unsupervised, data-driven condition defined by the ``input`` only.
      The model is trained using a custom unsupervised loss determined by the
      chosen :class:`~pina.solver.base_solver.BaseSolver`, while leveraging the
      provided data during training. Optional ``conditional_variables`` can be
      specified when the model depends on additional parameters.
      Supported data types include  :class:`~pina.label_tensor.LabelTensor`,
      :class:`torch.Tensor`, :class:`~torch_geometric.data.Data`, or
      :class:`~pina.graph.Graph`. The class automatically selects the
      appropriate implementation based on the type of the ``input``.

    .. note::

        The user should always instantiate :class:`Condition` directly, without
        manually creating subclass instances. Please refer to the specific
        :class:`Condition` classes for implementation details.

    :Example:

    >>> from pina import Condition

    >>> # Example of InputTargetCondition signature
    >>> condition = Condition(input=input, target=target)

    >>> # Example of DomainEquationCondition signature
    >>> condition = Condition(domain=domain, equation=equation)

    >>> # Example of InputEquationCondition signature
    >>> condition = Condition(input=input, equation=equation)

    >>> # Example of TimeSeriesCondition signature
    >>> condition = Condition(
    ...    input=input, n_windows=n_windows, unroll_length=unroll_length
    ... )

    >>> # Example of DataCondition signature
    >>> condition = Condition(input=data, conditional_variables=cond_vars)
    """

    # Internal specifications for condition types, used for dispatching
    # Each tuple contains: (condition class, required kwargs, optional kwargs)
    _SPECS = (
        (InputTargetCondition, {"input", "target"}, set()),
        (InputEquationCondition, {"input", "equation"}, set()),
        (DomainEquationCondition, {"domain", "equation"}, set()),
        (DataCondition, {"input"}, {"conditional_variables"}),
        (
            TimeSeriesCondition,
            {"input", "n_windows", "unroll_length"},
            {"randomize"},
        ),
    )

    # Compute the set of all available keyword arguments (optional + required)
    available_kwargs = sorted(set().union(*(rq | op for _, rq, op in _SPECS)))

    def __new__(cls, *args, **kwargs):
        """
        Instantiate the appropriate :class:`Condition` object based on the
        keyword arguments passed.

        :param tuple args: The positional arguments (should be empty).
        :param dict kwargs: The keyword arguments corresponding to the
            parameters of the specific :class:`Condition` type to instantiate.
        :raises ValueError: If unexpected positional arguments are provided.
        :raises ValueError: If the keyword arguments do not match any valid
            signature for the available condition types.
        :return: The appropriate :class:`Condition` object.
        :rtype: ConditionInterface
        """
        # Ensure no positional arguments are provided
        if args:
            raise ValueError(
                "Condition takes only keyword arguments. "
                f"Available arguments are: {cls.available_kwargs}."
            )

        # Iterate through the specifications to find a matching condition type
        for condition_cls, required, optional in cls._SPECS:

            # Find allowed keys for condition type
            allowed = required | optional

            # Check if the provided keys match the required and optional keys
            if required <= set(kwargs) <= allowed:
                return condition_cls(**kwargs)

        # If no valid signature is found, prepare a list of valid signatures
        valid_signatures = [
            sorted(required | optional) for _, required, optional in cls._SPECS
        ]

        # If no valid signature is found, raise an error
        raise ValueError(
            f"Invalid keyword arguments {sorted(set(kwargs))}. "
            f"Valid signatures are: {valid_signatures}."
        )
