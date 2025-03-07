"""
Module for the Collector class.
"""

from .graph import Graph
from .utils import check_consistency


class Collector:
    """
    Collector class for collecting data from the problem.
    """

    def __init__(self, problem):
        """ "
        Initialize the Collector class, by creating a hook between the collector
        and the problem and initializing the data collections.

        :param AbstractProblem problem: The problem to collect data from.
        """
        # creating a hook between collector and problem
        self.problem = problem

        # those variables are used for the dataloading
        self._data_collections = {name: {} for name in self.problem.conditions}
        self.conditions_name = dict(enumerate(self.problem.conditions))

        # variables used to check that all conditions are sampled
        self._is_conditions_ready = {
            name: False for name in self.problem.conditions
        }
        self.full = False

    @property
    def full(self):
        """
        Return True if all conditions are ready.
        """

        return all(self._is_conditions_ready.values())

    @full.setter
    def full(self, value):
        """
        Set the full property of the collector. Admit only boolean values.

        :param bool value: The value to set the full property to.
        """
        check_consistency(value, bool)
        self._full = value

    @property
    def data_collections(self):
        """
        Return the data collections, created by combining together all condition
        in the problem.

        :return: The data collections.
        :rtype: dict
        """

        return self._data_collections

    @property
    def problem(self):
        """
        Property that return the problem connected to the collector.

        :return: The problem connected to the collector.
        :rtype: AbstractProblem
        """
        return self._problem

    @problem.setter
    def problem(self, value):
        """
        Return the problem connected to the collector.

        return: The problem connected to the collector.
        rtype: AbstractProblem
        """

        self._problem = value

    def store_fixed_data(self):
        """
        Store inside data collections the fixed data of the problem. These comes
        from the conditions that do not require sampling.
        """

        # loop over all conditions
        for condition_name, condition in self.problem.conditions.items():
            # if the condition is not ready and domain is not attribute
            # of condition, we get and store the data
            if (not self._is_conditions_ready[condition_name]) and (
                not hasattr(condition, "domain")
            ):
                # get data
                keys = condition.__slots__
                values = [getattr(condition, name) for name in keys]
                values = [
                    value.data if isinstance(value, Graph) else value
                    for value in values
                ]
                self.data_collections[condition_name] = dict(zip(keys, values))
                # condition now is ready
                self._is_conditions_ready[condition_name] = True

    def store_sample_domains(self):
        """
        Store inside data collections the sampled data of the problem. These
        comes from the conditions that require sampling.
        """
        for condition_name in self.problem.conditions:
            condition = self.problem.conditions[condition_name]
            if not hasattr(condition, "domain"):
                continue

            samples = self.problem.discretised_domains[condition.domain]

            self.data_collections[condition_name] = {
                "input": samples,
                "equation": condition.equation,
            }
