"""
This module contains the PinaDataModule class, which extends the
LightningDataModule class to allow proper creation and management of
different types of Datasets defined in PINA.
"""

import warnings
from lightning.pytorch import LightningDataModule
import torch
from torch_geometric.data import Data
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from ..label_tensor import LabelTensor
from .dataset import PinaDatasetFactory, PinaTensorDataset
from ..collector import Collector


class DummyDataloader:
    """ "
    Dummy dataloader used when batch size is None. It callects all the data
    in self.dataset and returns it when it is called a single batch.
    """

    def __init__(self, dataset):
        """
        Preprare a dataloader object which will return the entire dataset
        in a single batch. Depending on the number of GPUs, the dataset we
        have the following cases:

        - **Distributed Environment** (multiple GPUs):
            - Divides the dataset across processes using the rank and world
                size.
            - Fetches only the portion of data corresponding to the current
                process.
        - **Non-Distributed Environment** (single GPU):
            - Fetches the entire dataset.

        :param dataset: The dataset object to be processed.
        :type dataset: PinaDataset

        .. note:: This data loader is used when the batch size is None.
        """

        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            if len(dataset) < world_size:
                raise RuntimeError(
                    "Dimension of the dataset smaller than world size."
                    " Increase the size of the partition or use a single GPU"
                )
            idx, i = [], rank
            while i < len(dataset):
                idx.append(i)
                i += world_size
            self.dataset = dataset.fetch_from_idx_list(idx)
        else:
            self.dataset = dataset.get_all_data()

    def __iter__(self):
        return self

    def __len__(self):
        return 1

    def __next__(self):
        return self.dataset


class Collator:
    """
    Class used to collate the batch
    """

    def __init__(
        self, max_conditions_lengths, automatic_batching, dataset=None
    ):
        """
        Initialize the object, setting the collate function based on whether
        automatic batching is enabled or not.

        :param dict max_conditions_lengths: dict containing the maximum number
            of data points to consider in a single batch for each condition.
        :param PinaDataset dataset: The dataset where the data is stored.
        """

        self.max_conditions_lengths = max_conditions_lengths
        # Set the collate function based on the batching strategy
        # collate_pina_dataloader is used when automatic batching is disabled
        # collate_torch_dataloader is used when automatic batching is enabled
        self.callable_function = (
            self._collate_torch_dataloader
            if automatic_batching
            else (self._collate_pina_dataloader)
        )
        self.dataset = dataset

        # Set the function which performs the actual collation
        if isinstance(self.dataset, PinaTensorDataset):
            # If the dataset is a PinaTensorDataset, use this collate function
            self._collate = self._collate_tensor_dataset
        else:
            # If the dataset is a PinaDataset, use this collate function
            self._collate = self._collate_graph_dataset

    def _collate_pina_dataloader(self, batch):
        """
        Function used to create a batch when automatic batching is disabled.

        :param list(int) batch: List of integers representing the indices of
            the data points to be fetched.
        :return: Dictionary containing the data points fetched from the dataset.
        :rtype: dict
        """
        # Call the fetch_from_idx_list method of the dataset
        return self.dataset.fetch_from_idx_list(batch)

    def _collate_torch_dataloader(self, batch):
        """
        Function used to collate the batch
        """
        batch_dict = {}
        if isinstance(batch, dict):
            return batch
        conditions_names = batch[0].keys()
        # Condition names
        for condition_name in conditions_names:
            single_cond_dict = {}
            condition_args = batch[0][condition_name].keys()
            for arg in condition_args:
                data_list = [
                    batch[idx][condition_name][arg]
                    for idx in range(
                        min(
                            len(batch),
                            self.max_conditions_lengths[condition_name],
                        )
                    )
                ]
                single_cond_dict[arg] = self._collate(data_list)

            batch_dict[condition_name] = single_cond_dict
        return batch_dict

    @staticmethod
    def _collate_tensor_dataset(data_list):
        """
        Function used to collate the data when the dataset is a
        `PinaTensorDataset`.

        :param data_list: List of `torch.Tensor` or `LabelTensor` to be
            collated.
        :type data_list: list(torch.Tensor) | list(LabelTensor)
        :raises RuntimeError: If the data is not a `torch.Tensor` or a
            `LabelTensor`.
        :return: Batch of data
        :rtype: dict
        """

        if isinstance(data_list[0], LabelTensor):
            return LabelTensor.stack(data_list)
        if isinstance(data_list[0], torch.Tensor):
            return torch.stack(data_list)
        raise RuntimeError("Data must be Tensors or LabelTensor ")

    def _collate_graph_dataset(self, data_list):
        """
        Function used to collate the data when the dataset is a
        `PinaGraphDataset`.

        :param data_list: List of `Data` or `Graph` to be collated.
        :type data_list: list(Data) | list(Graph)
        :raises RuntimeError: If the data is not a `Data` or a `Graph`.
        :return: Batch of data
        :rtype: dict
        """

        if isinstance(data_list[0], LabelTensor):
            return LabelTensor.cat(data_list)
        if isinstance(data_list[0], torch.Tensor):
            return torch.cat(data_list)
        if isinstance(data_list[0], Data):
            return self.dataset.create_batch(data_list)
        raise RuntimeError("Data must be Tensors or LabelTensor or pyG Data")

    def __call__(self, batch):
        """
        Call the function to collate the batch, defined in __init__.

        :param batch: list of indices or list of retrieved data
        :type batch: list(int) | list(dict)
        :return: Dictionary containing the data points fetched from the dataset,
            collated.
        :rtype: dict
        """

        return self.callable_function(batch)


class PinaSampler:
    """
    Class used to create the sampler instance.
    """

    def __new__(cls, dataset, shuffle):
        """
        Create the sampler instance, according to shuffle and whether the
        environment is distributed or not.

        :param PinaDataset dataset: The dataset to be sampled.
        :param bool shuffle: whether to shuffle the dataset or not before
            sampling.
        :return: The sampler instance.
        :rtype: torch.utils.data.Sampler
        """

        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            sampler = DistributedSampler(dataset, shuffle=shuffle)
        else:
            if shuffle:
                sampler = RandomSampler(dataset)
            else:
                sampler = SequentialSampler(dataset)
        return sampler


class PinaDataModule(LightningDataModule):
    """
    This class extend LightningDataModule, allowing proper creation and
    management of different types of Datasets defined in PINA
    """

    def __init__(
        self,
        problem,
        train_size=0.7,
        test_size=0.2,
        val_size=0.1,
        batch_size=None,
        shuffle=True,
        repeat=False,
        automatic_batching=None,
        num_workers=0,
        pin_memory=False,
    ):
        """
        Initialize the object, creating datasets based on the input problem.

        :param AbstractProblem problem: The problem containing the data on which
            to train/test the model.
        :param float train_size: Fraction or number of elements in the training
            split.
        :param float test_size: Fraction or number of elements in the test
            split.
        :param float val_size: Fraction or number of elements in the validation
            split.
        :param batch_size: The batch size used for training. If `None`, the
            entire dataset is used per batch.
        :type batch_size: int | None
        :param bool shuffle: Whether to shuffle the dataset before splitting.
        :param bool repeat: Whether to repeat the dataset indefinitely.
        :param automatic_batching: Whether to enable automatic batching.
        :param int num_workers: Number of worker threads for data loading.
            Default 0 (serial loading)
        :param bool pin_memory: Whether to use pinned memory for faster data
            transfer to GPU. (Default False)
        """
        super().__init__()

        # Store fixed attributes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.repeat = repeat
        self.automatic_batching = automatic_batching
        if batch_size is None and num_workers != 0:
            warnings.warn(
                "Setting num_workers when batch_size is None has no effect on "
                "the DataLoading process."
            )
            self.num_workers = 0
        else:
            self.num_workers = num_workers
        if batch_size is None and pin_memory:
            warnings.warn(
                "Setting pin_memory to True has no effect when "
                "batch_size is None."
            )
            self.pin_memory = False
        else:
            self.pin_memory = pin_memory

        # Collect data
        collector = Collector(problem)
        collector.store_fixed_data()
        collector.store_sample_domains()

        # Check if the splits are correct
        self._check_slit_sizes(train_size, test_size, val_size)

        # Split input data into subsets
        splits_dict = {}
        if train_size > 0:
            splits_dict["train"] = train_size
            self.train_dataset = None
        else:
            self.train_dataloader = super().train_dataloader
        if test_size > 0:
            splits_dict["test"] = test_size
            self.test_dataset = None
        else:
            self.test_dataloader = super().test_dataloader
        if val_size > 0:
            splits_dict["val"] = val_size
            self.val_dataset = None
        else:
            self.val_dataloader = super().val_dataloader

        self.collector_splits = self._create_splits(collector, splits_dict)
        self.transfer_batch_to_device = self._transfer_batch_to_device

    def setup(self, stage=None):
        """
        Perform the splitting of the dataset
        """
        if stage == "fit" or stage is None:
            self.train_dataset = PinaDatasetFactory(
                self.collector_splits["train"],
                max_conditions_lengths=self.find_max_conditions_lengths(
                    "train"
                ),
                automatic_batching=self.automatic_batching,
            )
            if "val" in self.collector_splits.keys():
                self.val_dataset = PinaDatasetFactory(
                    self.collector_splits["val"],
                    max_conditions_lengths=self.find_max_conditions_lengths(
                        "val"
                    ),
                    automatic_batching=self.automatic_batching,
                )
        elif stage == "test":
            self.test_dataset = PinaDatasetFactory(
                self.collector_splits["test"],
                max_conditions_lengths=self.find_max_conditions_lengths("test"),
                automatic_batching=self.automatic_batching,
            )
        else:
            raise ValueError("stage must be either 'fit' or 'test'.")

    @staticmethod
    def _split_condition(condition_dict, splits_dict):
        len_condition = len(condition_dict["input"])

        lengths = [
            int(len_condition * length) for length in splits_dict.values()
        ]

        remainder = len_condition - sum(lengths)
        for i in range(remainder):
            lengths[i % len(lengths)] += 1

        splits_dict = {
            k: max(1, v) for k, v in zip(splits_dict.keys(), lengths)
        }
        to_return_dict = {}
        offset = 0

        for stage, stage_len in splits_dict.items():
            to_return_dict[stage] = {
                k: v[offset : offset + stage_len]
                for k, v in condition_dict.items()
                if k != "equation"
                # Equations are NEVER dataloaded
            }
            if offset + stage_len >= len_condition:
                offset = len_condition - 1
                continue
            offset += stage_len
        return to_return_dict

    def _create_splits(self, collector, splits_dict):
        """
        Create the dataset objects putting data
        """

        # ----------- Auxiliary function ------------
        def _apply_shuffle(condition_dict, len_data):
            idx = torch.randperm(len_data)
            for k, v in condition_dict.items():
                if k == "equation":
                    continue
                if isinstance(v, list):
                    condition_dict[k] = [v[i] for i in idx]
                elif isinstance(v, LabelTensor):
                    condition_dict[k] = LabelTensor(v.tensor[idx], v.labels)
                elif isinstance(v, torch.Tensor):
                    condition_dict[k] = v[idx]
                else:
                    raise ValueError(f"Data type {type(v)} not supported")

        # ----------- End auxiliary function ------------

        split_names = list(splits_dict.keys())
        dataset_dict = {name: {} for name in split_names}
        for (
            condition_name,
            condition_dict,
        ) in collector.data_collections.items():
            len_data = len(condition_dict["input"])
            if self.shuffle:
                _apply_shuffle(condition_dict, len_data)
            for key, data in self._split_condition(
                condition_dict, splits_dict
            ).items():
                dataset_dict[key].update({condition_name: data})
        return dataset_dict

    def _create_dataloader(self, split, dataset):
        shuffle = self.shuffle if split == "train" else False
        # Suppress the warning about num_workers.
        # In many cases, especially for PINNs,
        # serial data loading can outperform parallel data loading.
        warnings.filterwarnings(
            "ignore",
            message=(
                "The '(train|val|test)_dataloader' does not have many workers "
                "which may be a bottleneck."
            ),
            module="lightning.pytorch.trainer.connectors.data_connector",
        )
        # Use custom batching (good if batch size is large)
        if self.batch_size is not None:
            sampler = PinaSampler(dataset, shuffle)
            if self.automatic_batching:
                collate = Collator(
                    self.find_max_conditions_lengths(split),
                    self.automatic_batching,
                    dataset=dataset,
                )
            else:
                collate = Collator(
                    None, self.automatic_batching, dataset=dataset
                )
            return DataLoader(
                dataset,
                self.batch_size,
                collate_fn=collate,
                sampler=sampler,
                num_workers=self.num_workers,
            )
        dataloader = DummyDataloader(dataset)
        dataloader.dataset = self._transfer_batch_to_device(
            dataloader.dataset, self.trainer.strategy.root_device, 0
        )
        self.transfer_batch_to_device = self._transfer_batch_to_device_dummy
        return dataloader

    def find_max_conditions_lengths(self, split):
        """
        Define the maximum length of the conditions.

        :param split:  The splits of the dataset.
        :type split: dict
        :return: The maximum length of the conditions.
        :rtype: dict
        """
        max_conditions_lengths = {}
        for k, v in self.collector_splits[split].items():
            if self.batch_size is None:
                max_conditions_lengths[k] = len(v["input"])
            elif self.repeat:
                max_conditions_lengths[k] = self.batch_size
            else:
                max_conditions_lengths[k] = min(
                    len(v["input"]), self.batch_size
                )
        return max_conditions_lengths

    def val_dataloader(self):
        """
        Create the validation dataloader
        """
        return self._create_dataloader("val", self.val_dataset)

    def train_dataloader(self):
        """
        Create the training dataloader

        :return: The training dataloader
        :rtype: DataLoader
        """
        return self._create_dataloader("train", self.train_dataset)

    def test_dataloader(self):
        """
        Create the testing dataloader

        :return: The testing dataloader
        :rtype: DataLoader
        """
        return self._create_dataloader("test", self.test_dataset)

    @staticmethod
    def _transfer_batch_to_device_dummy(batch, device, dataloader_idx):
        """
        Transfer the batch to the device. This method is called in the
        training loop and is used to transfer the batch to the device.
        This method is used when the batch size is None: batch has already
        been transferred to the device.

        :param list(tuple) batch: list of tuple where the first element of the
            tuple is the condition name and the second element is the data.
        :param device: device to which the batch is transferred
        :type device: torch.device
        :param dataloader_idx: index of the dataloader
        :type dataloader_idx: int
        :return: The batch transferred to the device.
        :rtype: list(tuple)
        """
        return batch

    def _transfer_batch_to_device(self, batch, device, dataloader_idx):
        """
        Transfer the batch to the device. This method is called in the
        training loop and is used to transfer the batch to the device.

        :param dict batch: The batch to be transferred to the device.
        :param device: The device to which the batch is transferred.
        :type device: torch.device
        :param int dataloader_idx: The index of the dataloader.
        :return: The batch transferred to the device.
        :rtype: list(tuple)
        """
        batch = [
            (
                k,
                super(LightningDataModule, self).transfer_batch_to_device(
                    v, device, dataloader_idx
                ),
            )
            for k, v in batch.items()
        ]

        return batch

    @staticmethod
    def _check_slit_sizes(train_size, test_size, val_size):
        """
        Check if the splits are correct
        """
        if train_size < 0 or test_size < 0 or val_size < 0:
            raise ValueError("The splits must be positive")
        if abs(train_size + test_size + val_size - 1) > 1e-6:
            raise ValueError("The sum of the splits must be 1")

    @property
    def input(self):
        """
        Return all the input points coming from all the datasets.

        :return: The input points for training.
        :rtype dict
        """
        to_return = {}
        if hasattr(self, "train_dataset") and self.train_dataset is not None:
            to_return["train"] = self.train_dataset.input
        if hasattr(self, "val_dataset") and self.val_dataset is not None:
            to_return["val"] = self.val_dataset.input
        if hasattr(self, "test_dataset") and self.test_dataset is not None:
            to_return["test"] = self.test_dataset.input
        return to_return
