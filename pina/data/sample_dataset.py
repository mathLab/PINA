"""
Sample dataset module
"""
from copy import deepcopy
from .base_dataset import BaseDataset
from ..condition import InputPointsEquationCondition


class SamplePointDataset(BaseDataset):
    """
    This class extends the BaseDataset to handle physical datasets
    composed of only input points.
    """
    data_type = 'physics'
    __slots__ = InputPointsEquationCondition.__slots__

    def add_points(self, data_dict, condition_idx, batching_dim=0):
        data_dict = deepcopy(data_dict)
        data_dict.pop('equation')
        super().add_points(data_dict, condition_idx)

    def _init_from_problem(self, collector_dict):
        for name, data in collector_dict.items():
            keys = list(data.keys())
            if set(self.__slots__) == set(keys):
                data = deepcopy(data)
                data.pop('equation')
                self._populate_init_list(data)
                idx = [
                    key for key, val in
                    self.problem.collector.conditions_name.items()
                    if val == name
                ]
                self.conditions_idx.append(idx)
        self.initialize()
