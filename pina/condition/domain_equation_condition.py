"""Module for the DomainEquationCondition class."""

from .condition_base import ConditionBase
from ..domain import DomainInterface
from ..equation.equation_interface import EquationInterface


class DomainEquationCondition(ConditionBase):
    """
    The class :class:`DomainEquationCondition` defines a condition based on a
    ``domain`` and an ``equation``. This condition is typically used in
    physics-informed problems, where the model is trained to satisfy a given
    ``equation`` over a specified ``domain``. The ``domain`` is used to sample
    points where the ``equation`` residual is evaluated and minimized during
    training.

    :Example:

    >>> from pina.domain import CartesianDomain
    >>> from pina.equation import Equation
    >>> from pina import Condition

    >>> # Equation to be satisfied over the domain: # x^2 + y^2 - 1 = 0
    >>> def dummy_equation(pts):
    ...     return pts["x"]**2 + pts["y"]**2 - 1

    >>> domain = CartesianDomain({"x": [0, 1], "y": [0, 1]})
    >>> condition = Condition(domain=domain, equation=Equation(dummy_equation))
    """

    # Available slots
    __fields__ = ["domain", "equation"]

    _avail_domain_cls = DomainInterface
    _avail_equation_cls = EquationInterface

    def __new__(cls, domain, equation):
        """
        Check the types of ``domain`` and ``equation`` and instantiate an
        instance of :class:`DomainEquationCondition`.

        :return: An instance of :class:`DomainEquationCondition`.
        :rtype: pina.condition.domain_equation_condition.DomainEquationCondition
        :raises ValueError: If ``domain`` is not of type
        :class:`DomainInterface` or
            ``equation`` is not of type :class:`
        """
        if not isinstance(domain, cls._avail_domain_cls):
            raise ValueError(
                "The domain must be an instance of DomainInterface."
            )

        if not isinstance(equation, cls._avail_equation_cls):
            raise ValueError(
                "The equation must be an instance of EquationInterface."
            )

        return super().__new__(cls)

    def __len__(self):
        """
        Raise NotImplementedError since the number of points is determined by
        the domain sampling strategy.

        :raises NotImplementedError: Always raised since the number of points is
            determined by the domain sampling strategy.
        """
        raise NotImplementedError(
            "`__len__` method is not implemented for "
            "`DomainEquationCondition` since the number of points is "
            "determined by the domain sampling strategy."
        )

    def __getitem__(self, idx):
        """
        Raise NotImplementedError since data retrieval is not applicable.

        :param int idx: Index of the data point(s) to retrieve.
        :raises NotImplementedError: Always raised since data retrieval is not
            applicable for this condition.
        """
        raise NotImplementedError(
            "`__getitem__` method is not implemented for "
            "`DomainEquationCondition`"
        )

    def store_data(self, **kwargs):
        """
        Store data for the condition. No data is stored for this condition.

        :return: An empty dictionary since no data is stored.
        :rtype: dict
        """
        setattr(self, "domain", kwargs.get("domain"))
        setattr(self, "equation", kwargs.get("equation"))
