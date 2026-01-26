"""Module for the InputEquationCondition class and its subclasses."""

from .condition_base import ConditionBase
from ..label_tensor import LabelTensor
from ..graph import Graph
from ..equation.equation_interface import EquationInterface
from ..condition.data_manager import _DataManager


class InputEquationCondition(ConditionBase):
    """
    The class :class:`InputEquationCondition` defines a condition based on
    ``input`` data and an ``equation``. This condition is typically used in
    physics-informed problems, where the model is trained to satisfy a given
    ``equation`` through the evaluation of the residual performed at the
    provided ``input``.

    :Example:

    >>> from pina import Condition, LabelTensor
    >>> from pina.equation import Equation
    >>> import torch

    >>> # Equation to be satisfied over the input points: # x^2 + y^2 - 1 = 0
    >>> def dummy_equation(pts):
    ...     return pts["x"]**2 + pts["y"]**2 - 1

    >>> pts = LabelTensor(torch.randn(100, 2), labels=["x", "y"])
    >>> condition = Condition(input=pts, equation=Equation(dummy_equation))
    """

    # Available input data types
    __fields__ = ["input", "equation"]
    _avail_input_cls = (LabelTensor, Graph)
    _avail_equation_cls = EquationInterface

    def __new__(cls, input, equation):
        """
        Check the types of ``input`` and ``equation`` and instantiate a class
        of :class:`InputEquationCondition` accordingly.

        :param input: The input data for the condition.
        :type input: LabelTensor | Graph |  list[Graph] | tuple[Graph]
        :param EquationInterface equation: The equation to be satisfied over the
            specified ``input`` data.
        :return: The subclass of InputEquationCondition.
        :rtype: pina.condition.input_equation_condition.
            InputTensorEquationCondition |
            pina.condition.input_equation_condition.InputGraphEquationCondition

        :raises ValueError: If input is not of type :class:`~pina.graph.Graph`
            or :class:`~pina.label_tensor.LabelTensor`.
        """

        # CHeck input type
        if not isinstance(input, cls._avail_input_cls):
            raise ValueError(
                "The input data object must be a LabelTensor or a Graph object."
            )

        # Check equation type
        if not isinstance(equation, cls._avail_equation_cls):
            raise ValueError(
                "The equation must be an instance of EquationInterface."
            )

        return super().__new__(cls)

    def store_data(self, **kwargs):
        """
        Store the input data in a :class:`_DataManager` object.
        :param dict kwargs: The keyword arguments containing the input data.
        """
        setattr(self, "equation", kwargs.pop("equation"))
        return _DataManager(**kwargs)

    @property
    def input(self):
        """
        Return the input data for the condition.

        :return: The input data.
        :rtype: LabelTensor | Graph | list[Graph] | tuple[Graph]
        """
        return self.data.input

    @property
    def equation(self):
        """
        Return the equation associated with this condition.

        :return: Equation associated with this condition.
        :rtype: EquationInterface
        """
        return self._equation

    @equation.setter
    def equation(self, value):
        """
        Set the equation associated with this condition.

        :param EquationInterface value: The equation to associate with this
            condition
        """
        if not isinstance(value, EquationInterface):
            raise TypeError(
                "The equation must be an instance of EquationInterface."
            )
        self._equation = value
