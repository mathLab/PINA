"""
Sample dataset module
"""
from .base_dataset import BaseDataset
from ..condition.input_equation_condition import InputPointsEquationCondition


class SamplePointDataset(BaseDataset):
    """
    This class extends the BaseDataset to handle physical datasets
    composed of only input points.
    """
    data_type = 'physics'
    __slots__ = InputPointsEquationCondition.__slots__
