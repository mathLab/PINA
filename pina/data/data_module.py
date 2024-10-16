"""
This module provide basic data management functionalities
"""

import math
import torch
from lightning import LightningDataModule
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
                 eval_size=.1,
                 batch_size=None,
                 shuffle=True,
                 datasets = None):
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
        super().__init__()
        dataset_classes = [SupervisedDataset, UnsupervisedDataset, SamplePointDataset]
        if datasets is None:
            self.datasets = [DatasetClass(problem, device) for DatasetClass in dataset_classes]
        else:
            self.datasets = datasets

        self.split_length = []
        self.split_names = []
        if train_size > 0:
            self.split_names.append('train')
            self.split_length.append(train_size)
        if test_size > 0:
            self.split_length.append(test_size)
            self.split_names.append('test')
        if eval_size > 0:
            self.split_length.append(eval_size)
            self.split_names.append('eval')

        self.batch_size = batch_size
        self.condition_names = None
        self.splits = {k: {} for k in self.split_names}
        self.shuffle = shuffle

    def setup(self, stage=None):
        """
        Perform the splitting of the dataset
        """
        self.extract_conditions()
        if stage == 'fit' or stage is None:
            for dataset in self.datasets:
                if len(dataset) > 0:
                    splits = self.dataset_split(dataset,
                                                self.split_length,
                                                shuffle=self.shuffle)
                    for i in range(len(self.split_length)):
                        self.splits[
                            self.split_names[i]][dataset.data_type] = splits[i]
        elif stage == 'test':
            raise NotImplementedError("Testing pipeline not implemented yet")
        else:
            raise ValueError("stage must be either 'fit' or 'test'")

    def extract_conditions(self):
        """
        Extract conditions from dataset and update condition indices
        """
        # Extract number of conditions
        n_conditions = 0
        for dataset in self.datasets:
            if n_conditions != 0:
                dataset.condition_names = {
                    key + n_conditions: value
                    for key, value in dataset.condition_names.items()
                }
            n_conditions += len(dataset.condition_names)

        self.condition_names = {
            key: value
            for dataset in self.datasets
            for key, value in dataset.condition_names.items()
        }



    def train_dataloader(self):
        """
        Return the training dataloader for the dataset
        :return: data loader
        :rtype: PinaDataLoader
        """
        return PinaDataLoader(self.splits['train'], self.batch_size,
                              self.condition_names)

    def test_dataloader(self):
        """
        Return the testing dataloader for the dataset
        :return: data loader
        :rtype: PinaDataLoader
        """
        return PinaDataLoader(self.splits['test'], self.batch_size,
                              self.condition_names)

    def eval_dataloader(self):
        """
        Return the evaluation dataloader for the dataset
        :return: data loader
        :rtype: PinaDataLoader
        """
        return PinaDataLoader(self.splits['eval'], self.batch_size,
                              self.condition_names)

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
            lengths = [
                int(math.floor(len(dataset) * length)) for length in lengths
            ]

            remainder = len(dataset) - sum(lengths)
            for i in range(remainder):
                lengths[i % len(lengths)] += 1
        elif sum(lengths) - 1 >= 1e-3:
            raise ValueError(f"Sum of lengths is {sum(lengths)} less than 1")

        if sum(lengths) != len(dataset):
            raise ValueError("Sum of lengths is not equal to dataset length")

        if shuffle:
            if seed is not None:
                generator = torch.Generator()
                generator.manual_seed(seed)
                indices = torch.randperm(sum(lengths), generator=generator).tolist()
            else:
                indices = torch.arange(sum(lengths)).tolist()
        else:
            indices = torch.arange(0, sum(lengths), 1, dtype=torch.uint8).tolist()
        offsets = [
            sum(lengths[:i]) if i > 0 else 0 for i in range(len(lengths))
        ]
        return [
            PinaSubset(dataset, indices[offset:offset + length])
            for offset, length in zip(offsets, lengths)
        ]
