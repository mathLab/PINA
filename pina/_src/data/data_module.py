"""
This module contains the PinaDataModule class, which extends the
LightningDataModule class to allow proper creation and management of
different types of Datasets defined in PINA.
"""

import warnings
from lightning.pytorch import LightningDataModule
import torch
from torch_geometric.data import Batch
from pina._src.data.creator import _Creator
from pina._src.core.graph import LabelBatch, Graph
from pina._src.data.aggregator import _Aggregator


class _ConditionSubset:
    """
    This class extends the :class:`torch.utils.data.Subset` class, allowing to
    fetch the data from the dataset based on a list of indices.
    """

    def __init__(self, condition, indices, automatic_batching):
        super().__init__()
        self.condition = condition
        self.indices = indices
        self.automatic_batching = automatic_batching

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Fetch the data from the dataset based on the list of indices.

        :param int idx: The index of the data to be fetched.
        :return: The data corresponding to the given index.
        :rtype: dict
        """
        idx = self.indices[idx]
        if not self.automatic_batching:
            return idx
        return self.condition[idx]

    def get_all_data(self):
        data = self.condition[self.indices]
        if "data" in data and isinstance(data["data"], list):
            batch_fn = (
                LabelBatch.from_data_list
                if isinstance(data["data"][0], Graph)
                else Batch.from_data_list
            )
            data["data"] = batch_fn(data["data"])
            data = {
                "input": data["data"],
                "target": data["data"].y,
            }
        return data


class PinaDataModule(LightningDataModule):
    """
    This class extends :class:`~lightning.pytorch.core.LightningDataModule`,
    allowing proper creation and management of different types of datasets
    defined in PINA.
    """

    def __init__(
        self,
        problem,
        train_size=0.7,
        test_size=0.2,
        val_size=0.1,
        batch_size=None,
        shuffle=True,
        batching_mode="common_batch_size",
        automatic_batching=None,
        num_workers=0,
        pin_memory=False,
    ):
        """
        Initialize the object and creating datasets based on the input problem.

        :param AbstractProblem problem: The problem containing the data on which
            to create the datasets and dataloaders.
        :param float train_size: Fraction of elements in the training split. It
            must be in the range [0, 1].
        :param float test_size: Fraction of elements in the test split. It must
            be in the range [0, 1].
        :param float val_size: Fraction of elements in the validation split. It
            must be in the range [0, 1].
        :param int batch_size: The batch size used for training. If ``None``,
            the entire dataset is returned in a single batch.
            Default is ``None``.
        :param bool shuffle: Whether to shuffle the dataset before splitting.
            Default ``True``.
        :param str batching_mode: The batching mode to use. Options are
            ``"common_batch_size"``, ``"proportional"``, and
            ``"separate_conditions"``. Default is ``"common_batch_size"``.
        :param automatic_batching: If ``True``, automatic PyTorch batching
            is performed, which consists of extracting one element at a time
            from the dataset and collating them into a batch. This is useful
            when the dataset is too large to fit into memory. On the other hand,
            if ``False``, the items are retrieved from the dataset all at once
            avoind the overhead of collating them into a batch and reducing the
            ``__getitem__`` calls to the dataset. This is useful when the
            dataset fits into memory. Avoid using automatic batching when
            ``batch_size`` is large. Default is ``False``.
        :param int num_workers: Number of worker threads for data loading.
            Default ``0`` (serial loading).
        :param bool pin_memory: Whether to use pinned memory for faster data
            transfer to GPU. Default ``False``.

        :raises ValueError: If at least one of the splits is negative.
        :raises ValueError: If the sum of the splits is different from 1.

        .. seealso::
            For more information on multi-process data loading, see:
            https://pytorch.org/docs/stable/data.html#multi-process-data-loading

            For details on memory pinning, see:
            https://pytorch.org/docs/stable/data.html#memory-pinning
        """
        super().__init__()

        self.problem = problem
        # Store fixed attributes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batching_mode = batching_mode
        self.automatic_batching = automatic_batching
        self.batching_mode = batching_mode

        # If batch size is None, num_workers has no effect
        if batch_size is None and num_workers != 0:
            warnings.warn(
                "Setting num_workers when batch_size is None has no effect on "
                "the DataLoading process."
            )
            self.num_workers = 0
        else:
            self.num_workers = num_workers

        # If batch size is None, pin_memory has no effect
        if batch_size is None and pin_memory:
            warnings.warn(
                "Setting pin_memory to True has no effect when "
                "batch_size is None."
            )
            self.pin_memory = False
        else:
            self.pin_memory = pin_memory
        self.problem.move_discretisation_into_conditions()
        self._check_slit_sizes(train_size, test_size, val_size)

        if train_size > 0:
            self.train_dataset = None
        else:
            # Use the super method to create the train dataloader which
            # raises NotImplementedError
            self.train_dataloader = super().train_dataloader
        if test_size > 0:
            self.test_dataset = None
        else:
            # Use the super method to create the train dataloader which
            # raises NotImplementedError
            self.test_dataloader = super().test_dataloader
        if val_size > 0:
            self.val_dataset = None
        else:
            # Use the super method to create the train dataloader which
            # raises NotImplementedError
            self.val_dataloader = super().val_dataloader

        self._create_condition_splits(problem, train_size, test_size, val_size)
        self.creator = _Creator(
            batching_mode=batching_mode,
            batch_size=batch_size,
            shuffle=shuffle,
            automatic_batching=automatic_batching,
            num_workers=num_workers,
            pin_memory=pin_memory,
            conditions=problem.conditions,
        )

    @staticmethod
    def _check_slit_sizes(train_size, test_size, val_size):
        """
        Check if the splits are correct. The splits sizes must be positive and
        the sum of the splits must be 1.

        :param float train_size: The size of the training split.
        :param float test_size: The size of the testing split.
        :param float val_size: The size of the validation split.

        :raises ValueError: If at least one of the splits is negative.
        :raises ValueError: If the sum of the splits is different
            from 1.
        """

        if train_size < 0 or test_size < 0 or val_size < 0:
            raise ValueError("The splits must be positive")
        if abs(train_size + test_size + val_size - 1) > 1e-6:
            raise ValueError("The sum of the splits must be 1")

    def _create_condition_splits(
        self, problem, train_size, test_size, val_size
    ):
        self.split_idxs = {}
        for condition_name, condition in problem.conditions.items():
            len_condition = len(condition)
            # Create the indices for shuffling and splitting
            indices = (
                torch.randperm(len_condition).tolist()
                if self.shuffle
                else list(range(len_condition))
            )

            # Determine split sizes
            train_end = int(train_size * len_condition)
            test_end = train_end + int(test_size * len_condition)

            # Split indices
            train_indices = indices[:train_end]
            test_indices = indices[train_end:test_end]
            val_indices = indices[test_end:]
            splits = {}
            splits["train"], splits["test"], splits["val"] = (
                train_indices,
                test_indices,
                val_indices,
            )
            self.split_idxs[condition_name] = splits

    def setup(self, stage=None):
        """
        Create the dataset objects for the given stage.
        If the stage is "fit", the training and validation datasets are created.
        If the stage is "test", the testing dataset is created.

        :param str stage: The stage for which to perform the dataset setup.

        :raises ValueError: If the stage is neither "fit" nor "test".
        """
        if stage in ("fit", None):
            self.train_datasets = {
                name: _ConditionSubset(
                    condition,
                    self.split_idxs[name]["train"],
                    automatic_batching=self.automatic_batching,
                )
                for name, condition in self.problem.conditions.items()
                if len(self.split_idxs[name]["train"]) > 0
            }

            self.val_datasets = {
                name: _ConditionSubset(
                    condition,
                    self.split_idxs[name]["val"],
                    automatic_batching=self.automatic_batching,
                )
                for name, condition in self.problem.conditions.items()
                if len(self.split_idxs[name]["val"]) > 0
            }

        if stage in ("test", None):
            self.test_datasets = {
                name: _ConditionSubset(
                    condition,
                    self.split_idxs[name]["test"],
                    automatic_batching=self.automatic_batching,
                )
                for name, condition in self.problem.conditions.items()
                if len(self.split_idxs[name]["test"]) > 0
            }
        if stage not in ("fit", "test", None):
            raise ValueError(
                f"Invalid stage {stage}. Stage must be either 'fit' or 'test'."
            )

    def train_dataloader(self):
        return _Aggregator(
            self.creator(self.train_datasets),
            batching_mode=self.batching_mode,
        )

    def val_dataloader(self):
        return _Aggregator(
            self.creator(self.val_datasets), batching_mode=self.batching_mode
        )

    def test_dataloader(self):
        return _Aggregator(
            self.creator(self.test_datasets),
            batching_mode=self.batching_mode,
        )

    @staticmethod
    def _transfer_batch_to_device_dummy(batch, device, dataloader_idx):
        """
        Transfer the batch to the device. This method is used when the batch
        size is None: batch has already been transferred to the device.

        :param list[tuple] batch: List of tuple where the first element of the
            tuple is the condition name and the second element is the data.
        :param torch.device device: Device to which the batch is transferred.
        :param int dataloader_idx: Index of the dataloader.
        :return: The batch transferred to the device.
        :rtype: list[tuple]
        """
        return batch

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """
        Transfer the batch to the device. This method is called in the
        training loop and is used to transfer the batch to the device.

        :param dict batch: The batch to be transferred to the device.
        :param torch.device device: The device to which the batch is
            transferred.
        :param int dataloader_idx: The index of the dataloader.
        :return: The batch transferred to the device.
        :rtype: list[tuple]
        """
        to_return = []
        for condition_name, condition in batch.items():
            to_return.append((condition_name, condition.to(device)))
        return to_return
