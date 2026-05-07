"""Module for creating dataloaders for multiple conditions."""

import torch
from torch.utils.data.distributed import DistributedSampler


class _Creator:
    """
    The class :class:`_Creator` is responsible for creating data loaders for
    multiple conditions based on specified batching strategies. It supports
    different batching strategies to accommodate various training requirements.
    """

    """
    Utility class for creating data loaders associated with multiple conditions.

    The class supports different batching strategies to adapt data loading
    behavior to specific training requirements
    """

    # Available batching modes
    _AVAIL_BATCHING_MODES = {
        "common_batch_size",
        "proportional",
        "separate_conditions",
    }

    def __init__(
        self,
        batching_mode,
        batch_size,
        shuffle,
        automatic_batching,
        num_workers,
        pin_memory,
        conditions,
    ):
        """
        Initialization of the :class:`_Creator` class.

        :param str batching_mode: The strategy used to aggregate batches across
            data loaders. Available options are ``"common_batch_size"`` for
            uniform batch sizes across conditions, ``"proportional"`` for batch
            sizes proportional to dataset sizes, and ``"separate_conditions"``
            for iterating through each condition separately.
        :param int batch_size: Batch size configuration used by the selected
            batching strategy. For ``"common_batch_size"``, the same batch size
            is assigned to all conditions. For ``"proportional"``, this value
            represents the total batch size distributed proportionally across
            conditions. For ``"separate_conditions"``, this value is applied
            independently to each condition and capped by the corresponding
            dataset size.
        :param bool shuffle: Whether samples should be shuffled during loading.
        :param bool automatic_batching: Whether automatic batching should be
            enabled in the data loaders.
        :param int num_workers: The number of worker processes used for data
            loading.
        :param bool pin_memory: Whether data loaders should pin memory.
        :param dict[str, BaseCondition] conditions: The mapping between
            condition names and condition objects responsible for data loader
            creation.
        :raises ValueError: If an invalid batching mode is provided.
        """
        # Check consistency
        if batching_mode not in self._AVAIL_BATCHING_MODES:
            raise ValueError(
                f"Invalid batching mode '{batching_mode}'. "
                f"Available options are: {self._AVAIL_BATCHING_MODES}"
            )

        # Initialize attributes
        self.batching_mode = batching_mode
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.automatic_batching = automatic_batching
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.conditions = conditions

    def __call__(self, datasets):
        """
        Create data loaders for all provided datasets.

        Batch sizes are computed according to the selected batching mode, and a
        dedicated data loader is created for each condition.

        :param dict[str, _ConditionSubset] datasets: The mapping between
            condition names and datasets.
        :return: The mapping between condition names and the corresponding
            data loaders.
        :rtype: dict[str, DataLoader]
        """
        # Compute batch sizes per condition based on batching_mode
        batch_sizes = self._compute_batch_sizes(datasets)
        dataloaders = {}

        # If common_batch_size mode, ensure all datasets have the same length
        if self.batching_mode == "common_batch_size":
            max_len = max(len(dataset) for dataset in datasets.values())

        # Iterate through datasets and create dataloaders
        for name, dataset in datasets.items():

            # If common_batch_size mode, set max_len for datasets
            if (
                self.batching_mode == "common_batch_size"
                and dataset.length != batch_sizes[name]
            ):
                dataset.max_len = max_len

            # Create dataloader for the current condition
            dataloaders[name] = self.conditions[name].create_dataloader(
                dataset=dataset,
                batch_size=batch_sizes[name],
                automatic_batching=self.automatic_batching,
                sampler=self._define_sampler(dataset, self.shuffle),
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )

        return dataloaders

    def _define_sampler(self, dataset, shuffle):
        """
        Define the sampling strategy for a dataset.

        Distributed training uses :class:`DistributedSampler`, while
        non-distributed execution uses either :class:`RandomSampler` or
        :class:`SequentialSampler` depending on ``shuffle``.

        :param _ConditionSubset dataset: The dataset associated with the
            sampler.
        :param bool shuffle: Whether samples should be shuffled during loading.
        :return: The configured sampler instance.
        :rtype: Sampler
        """
        # Distributed training case
        if torch.distributed.is_initialized():
            return DistributedSampler(dataset, shuffle=shuffle)

        # Non-distributed training case - shuffle True
        if shuffle:
            return torch.utils.data.RandomSampler(dataset)

        # Non-distributed training case - shuffle False
        return torch.utils.data.SequentialSampler(dataset)

    def _compute_batch_sizes(self, datasets):
        """
        Compute batch sizes for each dataset according to the selected batching
        mode.

        :param dict[str, _ConditionSubset] datasets: The mapping between
            condition names and datasets.
        :return: The mapping between condition names and computed batch sizes.
        :rtype: dict[str, int]
        """
        # Common batch size mode
        if self.batching_mode == "common_batch_size":

            # Compute batch size
            batch_size = (
                max(dataset.length for dataset in datasets.values())
                if self.batch_size is None
                else self.batch_size
            )

            return {
                name: min(batch_size, len(dataset))
                for name, dataset in datasets.items()
            }

        # Proportional batch size mode
        if self.batching_mode == "proportional":
            return self._compute_proportional_batch_sizes(datasets)

        # Separate conditions mode
        return {
            name: (
                len(dataset)
                if self.batch_size is None
                else min(self.batch_size, len(dataset))
            )
            for name, dataset in datasets.items()
        }

    def _compute_proportional_batch_sizes(self, datasets):
        """
        Compute batch sizes proportionally to dataset sizes.

        Each dataset receives a fraction of the total batch size proportional to
        its number of samples, while ensuring that each dataset contributes at
        least one sample.

        :param dict[str, _ConditionSubset] datasets: The mapping between
            condition names and datasets.
        :return: The mapping between condition names and proportional batch
            sizes.
        :rtype: dict[str, int]
        """
        # Compute the sizes of each dataset
        dataset_sizes = {
            name: len(dataset) for name, dataset in datasets.items()
        }

        # Determine the total number of elements across all datasets
        total_size = sum(dataset_sizes.values())

        # Compute the batch sizes
        batch_sizes = {
            name: max(1, int(self.batch_size * (size / total_size)))
            for name, size in dataset_sizes.items()
        }

        # Compute assigned batch size and difference with the total batch size
        assigned_batch_size = sum(batch_sizes.values())
        difference = self.batch_size - assigned_batch_size

        # If difference > 0, distribute to datasets with more than 1 sample
        if difference > 0:

            # Sort datasets by size in descending order
            sorted_datasets = sorted(
                dataset_sizes,
                key=lambda name: dataset_sizes[name],
                reverse=True,
            )

            # Distribute to datasets with more than 1 sample
            for name in sorted_datasets:

                # Stop distribution when the difference is fully allocated
                if difference == 0:
                    break

                # Distribute to datasets with more than 1 sample
                if dataset_sizes[name] > 1:
                    batch_sizes[name] += 1
                    difference -= 1

        # If difference < 0, reduce from datasets with more than 1 sample
        if difference < 0:

            # Sort batches by size in descending order
            sorted_batches = sorted(
                batch_sizes, key=lambda name: batch_sizes[name], reverse=True
            )

            # Reduce from datasets with more than 1 sample
            for name in sorted_batches:

                # Stop reduction when the difference is fully allocated
                if difference == 0:
                    break

                # Reduce from datasets with more than 1 sample
                if batch_sizes[name] > 1:
                    batch_sizes[name] -= 1
                    difference += 1

        return batch_sizes
