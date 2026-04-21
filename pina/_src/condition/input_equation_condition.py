"""Module for the Input-Equation Condition class."""

from pina._src.condition.base_condition import BaseCondition
from pina._src.core.label_tensor import LabelTensor
from pina._src.core.graph import Graph
from pina._src.equation.base_equation import BaseEquation
from pina._src.condition.data_manager import _DataManager
from pina._src.core.utils import check_consistency


class InputEquationCondition(BaseCondition):
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

    # Available fields, input and equation data types
    __fields__ = ["input", "equation"]
    _avail_input_cls = (LabelTensor, Graph)
    _avail_equation_cls = BaseEquation

    def __new__(cls, input, equation):
        """
        Check the types of ``input`` and ``equation`` and instantiate an
        instance of :class:`InputEquationCondition` accordingly.

        :param input: The input data associated with the condition.
        :type input: LabelTensor | Graph | list[Graph] | tuple[Graph]
        :param BaseEquation equation: The equation associated with the
            condition.
        :raises ValueError: If ``input`` is not an instance of
            :class:`~pina.label_tensor.LabelTensor`, or
            :class:`~pina.graph.Graph`, nor a list or tuple of
            :class:`~pina.graph.Graph`.
        :raises ValueError: If ``equation`` is not an instance of
            :class:`~pina.equation.base_equation.BaseEquation`.
        :return: A new instance of :class:`InputEquationCondition`.
        :rtype: InputEquationCondition
        """
        # Check input type - equation is checked in the setter
        if isinstance(input, (list, tuple)):
            check_consistency(input, Graph)
        else:
            check_consistency(input, cls._avail_input_cls)

        return super().__new__(cls)

    def store_data(self, **kwargs):
        """
        Store the input data in a dictionary-like structure.

        :param dict kwargs: The keyword arguments containing the data to be
            stored.
        :return: A dictionary-like structure containing the stored data.
        :rtype: _DataManager
        """
        # Save the equation as an attribute of the condition instance
        setattr(self, "equation", kwargs.pop("equation"))

        return _DataManager(**kwargs)

    @property
    def input(self):
        """
        The input data associated with the condition.

        :return: The input data.
        :rtype: LabelTensor | Graph | list[Graph] | tuple[Graph]
        """
        return self.data.input

    @property
    def equation(self):
        """
        The equation associated with the condition.

        :return: The equation.
        :rtype: BaseEquation
        """
        return self._equation

    @equation.setter
    def equation(self, value):
        """
        Set the equation associated with this condition.

        :param BaseEquation value: The equation to associate with the condition.
        :raises ValueError: If ``value`` is not an instance of
            :class:`~pina.equation.base_equation.BaseEquation`.
        """
        # Check consistency
        check_consistency(value, self._avail_equation_cls)
        self._equation = value
