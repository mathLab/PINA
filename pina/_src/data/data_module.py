"""
Utilities for creating and managing datasets and dataloaders.

This module defines a custom extension of the Lighting DataModule used to handle
dataset splitting, batching, and dataloader creation for PINA conditions.
"""

import warnings
import torch
from lightning.pytorch import LightningDataModule
from pina._src.data.condition_subset import _ConditionSubset
from pina._src.data.aggregator import _Aggregator
from pina._src.data.creator import _Creator


class DataModule(LightningDataModule):
    """
    An extension of the Lightning data module for managing PINA condition
    datasets.

    The data module handles train/validation/test dataset splitting, condition
    subset creation, dataloader construction, and batching coordination across
    multiple conditions.

    Dataset splitting is performed independently for each condition, and the
    resulting subsets are wrapped into :class:`_ConditionSubset` objects.
    Dataloaders are then created and aggregated according to the selected
    batching strategy.

    :Example:

        >>> import torch
        >>> from pina import LabelTensor
        >>> from pina.condition import Condition
        >>> from pina.problem import BaseProblem
        >>> class MyProblem(BaseProblem):
        ...     def __init__(self):
        ...         super().__init__()
        ...         pts = LabelTensor(torch.randn(100, 2), labels=["x", "y"])
        ...         self.conditions = {"cond1": Condition(input=pts)}
        >>> problem = MyProblem()
        >>> dm = DataModule(problem, train_size=0.8, val_size=0.1,
        ...     test_size=0.1, batch_size=32, batching_mode="common_batch_size",
        ...     automatic_batching=False, shuffle=True, num_workers=0,
        ...     pin_memory=False)
        >>> dm.setup("fit")
        >>> list(dm.train_datasets.keys())
        ['cond1']
    """

    def __init__(
        self,
        problem,
        train_size,
        val_size,
        test_size,
        batch_size,
        batching_mode,
        automatic_batching,
        shuffle,
        num_workers,
        pin_memory,
    ):
        """
        Initialization of the :class:`DataModule` class.

        :param BaseProblem problem: The problem containing the conditions and
            sampled data used to construct datasets and dataloaders.
        :param float train_size: The fraction of samples assigned to the
            training split. Must belong to the interval ``[0, 1]``.
        :param float val_size: The fraction of samples assigned to the
            validation split. Must belong to the interval ``[0, 1]``.
        :param float test_size: The fraction of samples assigned to the test
            split. Must belong to the interval ``[0, 1]``.
        :param int batch_size: The number of samples per batch. If ``None``, the
            entire dataset is processed as a single batch.
        :param str batching_mode: The strategy used to aggregate batches across
            dataloaders. Available options are ``"common_batch_size"`` for
            uniform batch sizes across conditions, ``"proportional"`` for batch
            sizes proportional to dataset sizes, and ``"separate_conditions"``
            for iterating through each condition separately.
        :param bool automatic_batching: Whether PyTorch automatic batching
            should be enabled. If ``True``, dataset elements are retrieved
            individually and collated into batches by the dataloader.
            If ``False``, entire subsets are retrieved directly from the
            condition object.
        :param bool shuffle: Whether condition samples should be shuffled before
            splitting.
        :param int num_workers: The number of worker processes used by
            dataloaders.
        :param bool pin_memory: Whether pinned memory should be enabled during
            data loading.
        :raises UserWarning: If ``num_workers`` is set to non-default value
            while ``batch_size`` is None.
        :raises UserWarning: If ``pin_memory`` is set to ``True`` while
            ``batch_size`` is None.
        """
        super().__init__()

        # Initialize the attributes -- consistency checked in trainer
        self.problem = problem
        self.batch_size = batch_size
        self.batching_mode = batching_mode
        self.automatic_batching = automatic_batching
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # If batch size is None, num_workers has no effect
        if batch_size is None and num_workers != 0:
            warnings.warn("num_workers has no effect when batch_size is None.")
            self.num_workers = 0

        # If batch size is None, pin_memory has no effect
        if batch_size is None and pin_memory:
            warnings.warn("pin_memory has no effect when batch_size is None.")
            self.pin_memory = False

        # Move domain discretisation into conditions subsets
        self.problem.move_discretisation_into_conditions()

        # If no splits are defined, use the default dataloaders
        if train_size == 0:
            self.train_dataloader = super().train_dataloader
        if val_size == 0:
            self.val_dataloader = super().val_dataloader
        if test_size == 0:
            self.test_dataloader = super().test_dataloader

        # Otherwise, create the condition splits and initialize the creator
        self._create_condition_splits(train_size, test_size)
        self.creator = _Creator(
            batching_mode=self.batching_mode,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            automatic_batching=self.automatic_batching,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            conditions=self.problem.conditions,
        )

    def _create_condition_splits(self, train_size, test_size):
        """
        Create train/validation/test index splits for each condition.

        Samples belonging to each condition are optionally shuffled before being
        partitioned into train, validation, and test subsets according to the
        specified split fractions.

        :param float train_size: The fraction of samples assigned to the
            training split. Must belong to the interval ``[0, 1]``.
        :param float test_size: The fraction of samples assigned to the test
            split. Must belong to the interval ``[0, 1]``.
        """
        # Initialize the dictionary to store the split idx for each condition
        self.split_idxs = {}

        # Iterate through conditions and create the splits
        for condition_name, condition in self.problem.conditions.items():

            # Get the total number of samples for the current condition
            condition_length = len(condition)

            # Generate shuffled or sequential indices for the condition samples
            indices = (
                torch.randperm(condition_length).tolist()
                if self.shuffle
                else list(range(condition_length))
            )

            # Compute the split indices for train, validation, and test subsets
            train_end = int(train_size * condition_length)
            test_end = train_end + int(test_size * condition_length)

            # Store the computed split indices in the dictionary
            self.split_idxs[condition_name] = {
                "train": indices[:train_end],
                "test": indices[train_end:test_end],
                "val": indices[test_end:],
            }

    def setup(self, stage=None):
        """
        Create dataset subsets for the requested execution stage.

        Depending on the selected stage, it initializes the ``train_datasets``,
        the ``val_datasets``, or the ``test_datasets`` attributes. Each dataset
        is represented as a mapping between condition names and
        :class:`_ConditionSubset` instances.

        :param str stage: The execution stage. Available options are ``"fit"``
            for training/validation and ``"test"`` for testing. If ``None``, both
            training/validation and testing datasets are created.
            Default is ``None``.
        :raises ValueError: If the provided stage is invalid.
        """
        # Validate the stage argument
        if stage not in ("fit", "test", None):
            raise ValueError(
                f"Invalid stage. Got {stage}, expected either 'fit' or  'test'."
            )

        # Fit stage: create training and validation datasets
        if stage in ("fit", None):

            # Train dataset
            self.train_datasets = {
                name: _ConditionSubset(
                    condition,
                    self.split_idxs[name]["train"],
                    automatic_batching=self.automatic_batching,
                )
                for name, condition in self.problem.conditions.items()
                if len(self.split_idxs[name]["train"]) > 0
            }

            # Validation dataset
            self.val_datasets = {
                name: _ConditionSubset(
                    condition,
                    self.split_idxs[name]["val"],
                    automatic_batching=self.automatic_batching,
                )
                for name, condition in self.problem.conditions.items()
                if len(self.split_idxs[name]["val"]) > 0
            }

        # Test stage: create testing dataset
        if stage in ("test", None):

            # Test dataset
            self.test_datasets = {
                name: _ConditionSubset(
                    condition,
                    self.split_idxs[name]["test"],
                    automatic_batching=self.automatic_batching,
                )
                for name, condition in self.problem.conditions.items()
                if len(self.split_idxs[name]["test"]) > 0
            }

    def transfer_batch_to_device(self, batch, device, _):
        """
        Transfer a batch to the target device.

        The method transfers all condition batches contained in the aggregated
        batch dictionary to the specified device.

        :param dict batch: The mapping between the condition names and the
            condition batches.
        :param torch.device device: The target device.
        :param _: Placeholder argument, not used.
        :return: A list of tuples containing condition names and transferred
            batches.
        :rtype: list[tuple[str, Any]]
        """
        return [
            (condition_name, condition.to(device))
            for condition_name, condition in batch.items()
        ]

    def train_dataloader(self):
        """
        Create the aggregated train dataloader.

        :return: The aggregated dataloader coordinating all train condition
            dataloaders.
        :rtype: _Aggregator
        """
        return _Aggregator(
            self.creator(self.train_datasets),
            batching_mode=self.batching_mode,
        )

    def val_dataloader(self):
        """
        Create the aggregated validation dataloader.

        :return: The aggregated dataloader coordinating all validation condition
            dataloaders.
        :rtype: _Aggregator
        """
        return _Aggregator(
            self.creator(self.val_datasets), batching_mode=self.batching_mode
        )

    def test_dataloader(self):
        """
        Create the aggregated test dataloader.

        :return: The aggregated dataloader coordinating all test condition
            dataloaders.
        :rtype: _Aggregator
        """
        return _Aggregator(
            self.creator(self.test_datasets),
            batching_mode=self.batching_mode,
        )
