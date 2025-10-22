"""Module for the Base class for all set-operations."""

from copy import deepcopy
from abc import ABCMeta
from .operation_interface import OperationInterface
from .base_domain import BaseDomain
from ..utils import check_consistency, check_positive_integer


class BaseOperation(OperationInterface, metaclass=ABCMeta):
    """
    Base class for all set operation defined on geometric domains, implementing
    common functionality.

    All specific operation types should inherit from this class and implement
    the abstract methods defined in both the following interfaces:
    :class:`~pina.domain.operation_interface.OperationInterface`, and
    :class:`~pina.domain.domain_interface.DomainInterface`.

    This class is not meant to be instantiated directly.
    """

    def __init__(self, geometries):
        """
        Initialization of the :class:`OperationInterface` class.

        :param geometries: The list of domains on which to perform the set
            operation.
        :type geometries: list[BaseDomain] | tuple[BaseDomain]
        :raises TypeError: If geometries is neither a list nor a tuple.
        :raises ValueError: If geometries elements are not instances of
            :class:`~pina.domain.base_domain.BaseDomain`.
        :raises NotImplementedError: If the dimensions of the geometries are not
            consistent.
        """
        self.geometries = geometries

    def _validate_sampling(self, n, mode, variables):
        """
        Validate the sampling settings.

        :param int n: The number of samples to generate.
        :param str mode: The sampling method.
        :param variables: The list of variables to sample. If ``all``, all
            variables are sampled.
        :raises AssertionError: If ``n`` is not a positive integer.
        :raises ValueError: If the sampling mode is invalid.
        :raises ValueError: If ``variables`` is neither ``all``, a string, nor a
            list/tuple of strings.
        :raises ValueError: If any of the specified variables is unknown.
        :return: The validated list of variables to sample.
        :rtype: list[str]
        """
        # Validate n
        check_positive_integer(value=n, strict=True)

        # Validate mode
        if mode not in self.sample_modes:
            raise ValueError(
                f"Invalid sampling mode: {mode}. Available: {self.sample_modes}"
            )

        # Validate variables
        check_consistency(variables, str)
        if variables == "all":
            variables = self.variables
        elif isinstance(variables, str):
            variables = [variables]
        else:
            variables = list(dict.fromkeys(variables))

        # Check for unknown variables
        unknown = [v for v in variables if v not in self.variables]
        if unknown:
            raise ValueError(
                f"Unknown variable(s): {unknown}. Available: {self.variables}"
            )

        return sorted(variables)

    def update(self, domain):
        """
        Update the domain resulting from the operation.

        :param DomainInterface domain: The domain whose labels are to be merged
            into the current one.
        :raises NotImplementedError: If the geometries involved in the operation
            are of different types.
        :raises TypeError: If the passed domain is not of the same type of all
            the geometries involved in the operation.
        :return: A new domain instance with the merged labels.
        :rtype: BaseOperation
        """
        # Check all geometries are of the same type
        domain_type = type(self.geometries[0])
        if not all(isinstance(g, domain_type) for g in self.geometries):
            raise NotImplementedError(
                f"The {self.__class__.__name__} of geometries of different"
                " types does not support the update operation. All geometries"
                " must be of the same type."
            )

        # Check domain type consistency
        if not isinstance(domain, domain_type):
            raise TypeError(
                f"Cannot update the {self.__class__.__name__} of domains of"
                f" type {domain_type} with domain of type {type(domain)}."
            )

        # Update each geometry
        updated = deepcopy(self)
        updated.geometries = [geom.update(domain) for geom in self.geometries]

        return updated

    @property
    def sample_modes(self):
        """
        The list of available sampling modes.

        :return: The list of available sampling modes.
        :rtype: list[str]
        """
        return list(
            set.intersection(
                *map(set, [g.sample_modes for g in self.geometries])
            )
        )

    @property
    def variables(self):
        """
        The list of variables of the domain.

        :return: The list of variables of the domain.
        :rtype: list[str]
        """
        return sorted({v for g in self.geometries for v in g.variables})

    @property
    def domain_dict(self):
        """
        Returns a dictionary representation of the operation domain.

        :return: The dictionary representation of the operation domain.
        :rtype: dict
        """
        return {
            "type": self.__class__.__name__,
            "geometries": [geom.domain_dict for geom in self.geometries],
        }

    @property
    def geometries(self):
        """
        The domains on which to perform the set operation.

        :return: The domains on which to perform the set operation.
        :rtype: list[BaseDomain]
        """
        return self._geometries

    @property
    def range(self):
        """
        The range variables of each geometry.

        :return: The range variables of each geometry.
        :rtype: dict
        """
        return {f"geometry_{i}": g.range for i, g in enumerate(self.geometries)}

    @property
    def fixed(self):
        """
        The fixed variables of each geometry.

        :return: The fixed variables of each geometry.
        :rtype: dict
        """
        return {f"geometry_{i}": g.fixed for i, g in enumerate(self.geometries)}

    @geometries.setter
    def geometries(self, values):
        """
        Setter for the ``geometries`` property.

        :param values: The geometries to be set.
        :type values: list[BaseDomain] | tuple[BaseDomain]
        :raises TypeError: If values is neither a list nor a tuple.
        :raises ValueError: If values elements are not instances of
            :class:`~pina.domain.base_domain.BaseDomain`.
        :raises NotImplementedError: If the dimensions of the geometries are not
            consistent.
        """
        # Check geometries are list or tuple
        if not isinstance(values, (list, tuple)):
            raise TypeError(
                "geometries must be either a list or a tuple of BaseDomain."
            )

        # Check consistency
        check_consistency(values, (BaseDomain, BaseOperation))

        # Check geometries
        for v in values:
            if v.variables != values[0].variables:
                raise NotImplementedError(
                    f"The {self.__class__.__name__} of geometries living in "
                    "different ambient spaces is not well-defined. "
                    "All geometries must share the same dimensions and labels."
                )

        self._geometries = values
