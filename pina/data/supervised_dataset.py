"""
Supervised dataset module
"""
from .base_dataset import BaseDataset


class SupervisedDataset(BaseDataset):
    """
    This class extends the BaseDataset to handle datasets that consist of input-output pairs.
    """
    data_type = 'supervised'
    __slots__ = ['input_points', 'output_points']
