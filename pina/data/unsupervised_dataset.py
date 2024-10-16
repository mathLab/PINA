"""
Unsupervised dataset module
"""
from .base_dataset import BaseDataset


class UnsupervisedDataset(BaseDataset):
    """
    This class extend BaseDataset class to handle unsupervised dataset,
    composed of input points and, optionally, conditional variables
    """
    data_type = 'unsupervised'
    __slots__ = ['input_points', 'conditional_variables']
