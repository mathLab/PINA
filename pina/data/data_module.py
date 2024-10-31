"""
This module provide basic data management functionalities
"""

import math
import torch
import logging
from pytorch_lightning import LightningDataModule
from .sample_dataset import SamplePointDataset
from .supervised_dataset import SupervisedDataset
from .unsupervised_dataset import UnsupervisedDataset
from .pina_dataloader import PinaDataLoader
from .pina_subset import PinaSubset


class PinaDataModule(LightningDataModule):
    """
    This class extend LightningDataModule, allowing proper creation and
    management of different types of Datasets defined in PINA
    """

    def __init__(self,
                 problem,
                 device,
                 train_size=.7,
                 test_size=.2,
                 val_size=.1,
                 predict_size=0.,
                 batch_size=None,
                 shuffle=True,
                 datasets=None):
        """
        Initialize the object, creating dataset based on input problem
        :param AbstractProblem problem: PINA problem
        :param device: Device used for training and testing
        :param train_size: number/percentage of elements in train split
        :param test_size: number/percentage of elements in test split
        :param eval_size: number/percentage of elements in evaluation split
        :param batch_size: batch size used for training
        :param datasets: list of datasets objects
        """
        logging.debug('Start initialization of Pina DataModule')
        logging.info('Start initialization of Pina DataModule')
        super().__init__()
        self.problem = problem
        self.device = device
        self.dataset_classes = [
            SupervisedDataset, UnsupervisedDataset, SamplePointDataset
        ]
        if datasets is None:
            self.datasets = None
        else:
            self.datasets = datasets

        self.split_length = []
        self.split_names = []
        self.loader_functions = {}
        self.batch_size = batch_size
        self.condition_names = problem.collector.conditions_name

        if train_size > 0:
            self.split_names.append('train')
            self.split_length.append(train_size)
            self.loader_functions['train_dataloader'] = lambda \
                x: PinaDataLoader(self.splits['train'], self.batch_size,
                                  self.condition_names)
        if test_size > 0:
            self.split_length.append(test_size)
            self.split_names.append('test')
            self.loader_functions['test_dataloader'] = lambda x: PinaDataLoader(
                self.splits['test'], self.batch_size, self.condition_names)
        if val_size > 0:
            self.split_length.append(val_size)
            self.split_names.append('val')
            self.loader_functions['val_dataloader'] = lambda x: PinaDataLoader(
                self.splits['val'], self.batch_size, self.condition_names)
        if predict_size > 0:
            self.split_length.append(predict_size)
            self.split_names.append('predict')
            self.loader_functions[
                'predict_dataloader'] = lambda x: PinaDataLoader(
                self.splits['predict'], self.batch_size, self.condition_names)
        self.splits = {k: {} for k in self.split_names}
        self.shuffle = shuffle

        for k, v in self.loader_functions.items():
            setattr(self, k, v.__get__(self, PinaDataModule))

    def prepare_data(self):
        if self.datasets is None:
            self._create_datasets()

    def setup(self, stage=None):
        """
        Perform the splitting of the dataset
        """
        logging.debug('Start setup of Pina DataModule obj')
        if self.datasets is None:
            self._create_datasets()
        if stage == 'fit' or stage is None:
            for dataset in self.datasets:
                if len(dataset) > 0:
                    splits = self.dataset_split(dataset,
                                                self.split_length,
                                                shuffle=self.shuffle)
                    for i in range(len(self.split_length)):
                        self.splits[self.split_names[i]][
                            dataset.data_type] = splits[i]
        elif stage == 'test':
            raise NotImplementedError("Testing pipeline not implemented yet")
        else:
            raise ValueError("stage must be either 'fit' or 'test'")

    @staticmethod
    def dataset_split(dataset, lengths, seed=None, shuffle=True):
        """
        Perform the splitting of the dataset
        :param dataset: dataset object we wanted to split
        :param lengths: lengths of elements in dataset
        :param seed: random seed
        :param shuffle: shuffle dataset
        :return: split dataset
        :rtype: PinaSubset
        """
        if sum(lengths) - 1 < 1e-3:
            len_dataset = len(dataset)
            lengths = [
                int(math.floor(len_dataset * length)) for length in lengths
            ]
            remainder = len(dataset) - sum(lengths)
            for i in range(remainder):
                lengths[i % len(lengths)] += 1
        elif sum(lengths) - 1 >= 1e-3:
            raise ValueError(f"Sum of lengths is {sum(lengths)} less than 1")

        if shuffle:
            if seed is not None:
                generator = torch.Generator()
                generator.manual_seed(seed)
                indices = torch.randperm(sum(lengths), generator=generator)
            else:
                indices = torch.randperm(sum(lengths))
            dataset.apply_shuffle(indices)

        offsets = [
            sum(lengths[:i]) if i > 0 else 0 for i in range(len(lengths))
        ]
        return [
            PinaSubset(dataset, slice(offset, offset + length))
            for offset, length in zip(offsets, lengths)
        ]

    def _create_datasets(self):
        """
        Create the dataset objects putting data 
        """
        logging.debug('Dataset creation in PinaDataModule obj')
        collector = self.problem.collector
        batching_dim = self.problem.batching_dimension
        datasets_slots = [i.__slots__ for i in self.dataset_classes]
        self.datasets = [
            dataset(device=self.device) for dataset in self.dataset_classes
        ]
        logging.debug('Filling datasets in PinaDataModule obj')
        for name, data in collector.data_collections.items():
            keys = list(data.keys())
            idx = [
                key for key, val in collector.conditions_name.items()
                if val == name
            ]
            for i, slot in enumerate(datasets_slots):
                if slot == keys:
                    self.datasets[i].add_points(data, idx[0], batching_dim)
                    continue
        datasets = []
        for dataset in self.datasets:
            if not dataset.empty:
                dataset.initialize()
                datasets.append(dataset)
        self.datasets = datasets
