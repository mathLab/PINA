"""
# TODO
"""

from .graph import Graph
from .utils import check_consistency


class Collector:

    def __init__(self, problem):
        # creating a hook between collector and problem
        self.problem = problem

        # those variables are used for the dataloading
        self._data_collections = {name: {} for name in self.problem.conditions}
        self.conditions_name = {
            i: name for i, name in enumerate(self.problem.conditions)
        }

        # variables used to check that all conditions are sampled
        self._is_conditions_ready = {
            name: False for name in self.problem.conditions
        }
        self.full = False

    @property
    def full(self):
        return all(self._is_conditions_ready.values())

    @full.setter
    def full(self, value):
        check_consistency(value, bool)
        self._full = value

    @property
    def data_collections(self):
        return self._data_collections

    @property
    def problem(self):
        return self._problem

    @problem.setter
    def problem(self, value):
        self._problem = value

    def store_fixed_data(self):
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
        # TODO: Add docstring
        """
        for condition_name in self.problem.conditions:
            condition = self.problem.conditions[condition_name]
            if not hasattr(condition, "domain"):
                continue

            samples = self.problem.discretised_domains[condition.domain]

            self.data_collections[condition_name] = {
                "input_points": samples,
                "equation": condition.equation,
            }
