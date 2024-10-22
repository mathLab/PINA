from sympy.strategies.branch import condition

from . import LabelTensor
from .utils import check_consistency, merge_tensors


class Collector:
    def __init__(self, problem):
        # creating a hook between collector and problem
        self.problem = problem

        # this variable is used to store the data in the form:
        # {'[condition_name]' : 
        #           {'input_points' : Tensor, 
        #            '[equation/output_points/conditional_variables]': Tensor}
        # }
        # those variables are used for the dataloading
        self._data_collections = {name: {} for name in self.problem.conditions}

        # variables used to check that all conditions are sampled
        self._is_conditions_ready = {
            name: False for name in self.problem.conditions}
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
            not hasattr(condition, "domain")):
                # get data
                keys = condition.__slots__
                values = [getattr(condition, name) for name in keys]
                self.data_collections[condition_name] = dict(zip(keys, values))
                # condition now is ready
                self._is_conditions_ready[condition_name] = True

    def store_sample_domains(self, n, mode, variables, sample_locations):
        # loop over all locations
        for loc in sample_locations:
            # get condition
            condition = self.problem.conditions[loc]
            keys = ["input_points", "equation"]
            # if the condition is not ready, we get and store the data
            if (not self._is_conditions_ready[loc]):
                # if it is the first time we sample
                if not self.data_collections[loc]:
                    already_sampled = []
                # if we have sampled the condition but not all variables
                else:
                    already_sampled = [
                        self.data_collections[loc]['input_points']]
            # if the condition is ready but we want to sample again
            else:
                self._is_conditions_ready[loc] = False
                already_sampled = []

            # get the samples
            samples = [
                          condition.domain.sample(n=n, mode=mode,
                                                  variables=variables)
                      ] + already_sampled
            pts = merge_tensors(samples)
            if (
                    set(pts.labels).issubset(
                        sorted(self.problem.input_variables))
            ):
                pts = pts.sort_labels()
                if sorted(pts.labels) == sorted(self.problem.input_variables):
                    self._is_conditions_ready[loc] = True
                values = [pts, condition.equation]
                self.data_collections[loc] = dict(zip(keys, values))
            else:
                raise RuntimeError(
                    'Try to sample variables which are not in problem defined in the problem')

    def add_points(self, new_points_dict):
        """
        Add input points to a sampled condition

        :param new_points_dict: Dictonary of input points (condition_name: LabelTensor)
        :raises RuntimeError: if at least one condition is not already sampled
        """
        for k, v in new_points_dict.items():
            if not self._is_conditions_ready[k]:
                raise RuntimeError(
                    'Cannot add points on a non sampled condition')
            self.data_collections[k]['input_points'] = self.data_collections[k][
                'input_points'].vstack(v)
