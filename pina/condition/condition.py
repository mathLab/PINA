""" Condition module. """

from ..label_tensor import LabelTensor
from ..domain import DomainInterface
from ..equation.equation import Equation

from . import DomainOutputCondition, DomainEquationCondition


def dummy(a):
    """Dummy function for testing purposes."""
    return None


class Condition:
    """
    The class ``Condition`` is used to represent the constraints (physical
    equations, boundary conditions, etc.) that should be satisfied in the
    problem at hand. Condition objects are used to formulate the PINA :obj:`pina.problem.abstract_problem.AbstractProblem` object.
    Conditions can be specified in three ways:

        1. By specifying the input and output points of the condition; in such a
        case, the model is trained to produce the output points given the input
        points.

        2. By specifying the location and the equation of the condition; in such
        a case, the model is trained to minimize the equation residual by
        evaluating it at some samples of the location.

        3. By specifying the input points and the equation of the condition; in
        such a case, the model is trained to minimize the equation residual by
        evaluating it at the passed input points.

    Example::

    >>> example_domain = Span({'x': [0, 1], 'y': [0, 1]})
    >>> def example_dirichlet(input_, output_):
    >>>     value = 0.0
    >>>     return output_.extract(['u']) - value
    >>> example_input_pts = LabelTensor(
    >>>     torch.tensor([[0, 0, 0]]), ['x', 'y', 'z'])
    >>> example_output_pts = LabelTensor(torch.tensor([[1, 2]]), ['a', 'b'])
    >>>
    >>> Condition(
    >>>     input_points=example_input_pts,
    >>>     output_points=example_output_pts)
    >>> Condition(
    >>>     location=example_domain,
    >>>     equation=example_dirichlet)
    >>> Condition(
    >>>     input_points=example_input_pts,
    >>>     equation=example_dirichlet)

    """

    # def _dictvalue_isinstance(self, dict_, key_, class_):
    #     """Check if the value of a dictionary corresponding to `key` is an instance of `class_`."""
    #     if key_ not in dict_.keys():
    #         return True

    #     return isinstance(dict_[key_], class_)

    # def __init__(self, *args, **kwargs):
    #     """
    #     Constructor for the `Condition` class.
    #     """
    #     self.data_weight = kwargs.pop("data_weight", 1.0)

    #     if len(args) != 0:
    #         raise ValueError(
    #             f"Condition takes only the following keyword arguments: {Condition.__slots__}."
    #         )

    def __new__(cls, *args, **kwargs):

        if sorted(kwargs.keys()) == sorted(["input_points", "output_points"]):
            return DomainOutputCondition(
                domain=kwargs["input_points"],
                output_points=kwargs["output_points"]
            )
        elif sorted(kwargs.keys()) == sorted(["domain", "output_points"]):
            return DomainOutputCondition(**kwargs)
        elif sorted(kwargs.keys()) == sorted(["domain", "equation"]):
            return DomainEquationCondition(**kwargs)
        else:
            raise ValueError(f"Invalid keyword arguments {kwargs.keys()}.")
        
        if (
            sorted(kwargs.keys()) != sorted(["input_points", "output_points"])
            and sorted(kwargs.keys()) != sorted(["location", "equation"])
            and sorted(kwargs.keys()) != sorted(["input_points", "equation"])
        ):
            raise ValueError(f"Invalid keyword arguments {kwargs.keys()}.")

        if not self._dictvalue_isinstance(kwargs, "input_points", LabelTensor):
            raise TypeError("`input_points` must be a torch.Tensor.")
        if not self._dictvalue_isinstance(kwargs, "output_points", LabelTensor):
            raise TypeError("`output_points` must be a torch.Tensor.")
        if not self._dictvalue_isinstance(kwargs, "location", Location):
            raise TypeError("`location` must be a Location.")
        if not self._dictvalue_isinstance(kwargs, "equation", Equation):
            raise TypeError("`equation` must be a Equation.")

        for key, value in kwargs.items():
            setattr(self, key, value)
