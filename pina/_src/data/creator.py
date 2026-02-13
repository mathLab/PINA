"""
Module defining the Creator class, responsible for creating dataloaders
for multiple conditions with various batching strategies.
"""

import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler


class _Creator:
    """
    The class :class:`_Creator` is responsible for creating dataloaders for
    multiple conditions based on specified batching strategies. It supports
    different batching modes to accommodate various training requirements.
    """

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

        :param batching_mode: The batching mode to use. Options are
            ``"common_batch_size"``, ``"proportional"``, and
            ``"separate_conditions"``.
        :type batching_mode: str
        :param batch_size: The batch size to use for dataloaders. If
            ``batching_mode`` is ``"proportional"``, this represents the total
            batch size across all conditions.
        :type batch_size: int | None
        :param shuffle: Whether to shuffle the data in the dataloaders.
        :type shuffle: bool
        :param automatic_batching: Whether to use automatic batching in the
            dataloaders.
        :type automatic_batching: bool
        :param num_workers: The number of worker processes to use for data
            loading.
        :type num_workers: int
        :param pin_memory: Whether to pin memory in the dataloaders.
        :type pin_memory: bool
        :param conditions: A dictionary mapping condition names to their
            respective condition objects.
        :type conditions: dict[str, Condition]
        """
        self.batching_mode = batching_mode
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.automatic_batching = automatic_batching
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.conditions = conditions

    def _define_sampler(self, dataset, shuffle):
        if torch.distributed.is_initialized():
            return DistributedSampler(dataset, shuffle=shuffle)
        if shuffle:
            return RandomSampler(dataset)
        return SequentialSampler(dataset)

    def _compute_batch_sizes(self, datasets):
        """
        Compute batch sizes for each condition based on the specified
        batching mode.

        :param datasets: A dictionary mapping condition names to their
            respective datasets.
        :type datasets: dict[str, Dataset]
        :return: A dictionary mapping condition names to their computed batch
            sizes.
        :rtype: dict[str, int]
        """
        batch_sizes = {}
        if self.batching_mode == "common_batch_size":
            for name in datasets.keys():
                if self.batch_size is None:
                    batch_sizes[name] = len(datasets[name])
                else:
                    batch_sizes[name] = min(
                        self.batch_size, len(datasets[name])
                    )
            return batch_sizes
        if self.batching_mode == "proportional":
            return self._compute_proportional_batch_sizes(datasets)
        if self.batching_mode == "separate_conditions":
            for name in datasets.keys():
                condition = self.conditions[name]
                if self.batch_size is None:
                    batch_sizes[name] = len(datasets[name])
                else:
                    batch_sizes[name] = min(
                        self.batch_size, len(datasets[name])
                    )
            return batch_sizes
        raise ValueError(f"Unknown batching mode: {self.batching_mode}")

    def _compute_proportional_batch_sizes(self, datasets):
        """
        Compute batch sizes for each condition proportionally based on the
        size of their datasets.
        :param datasets: A dictionary mapping condition names to their
            respective datasets.
        :type datasets: dict[str, Dataset]
        :return: A dictionary mapping condition names to their computed batch
            sizes.
        :rtype: dict[str, int]
        """
        # Compute number of elements per dataset
        elements_per_dataset = {
            dataset_name: len(dataset)
            for dataset_name, dataset in datasets.items()
        }
        # Compute the total number of elements
        total_elements = sum(el for el in elements_per_dataset.values())
        # Compute the portion of each dataset
        portion_per_dataset = {
            name: el / total_elements
            for name, el in elements_per_dataset.items()
        }
        # Compute batch size per dataset. Ensure at least 1 element per
        # dataset.
        batch_size_per_dataset = {
            name: max(1, int(portion * self.batch_size))
            for name, portion in portion_per_dataset.items()
        }
        # Adjust batch sizes to match the specified total batch size
        tot_el_per_batch = sum(el for el in batch_size_per_dataset.values())
        if self.batch_size > tot_el_per_batch:
            difference = self.batch_size - tot_el_per_batch
            while difference > 0:
                for k, v in batch_size_per_dataset.items():
                    if difference == 0:
                        break
                    if v > 1:
                        batch_size_per_dataset[k] += 1
                        difference -= 1
        if self.batch_size < tot_el_per_batch:
            difference = tot_el_per_batch - self.batch_size
            while difference > 0:
                for k, v in batch_size_per_dataset.items():
                    if difference == 0:
                        break
                    if v > 1:
                        batch_size_per_dataset[k] -= 1
                        difference -= 1
        return batch_size_per_dataset

    def __call__(self, datasets):
        """
        Create dataloaders for each condition based on the specified batching
        mode.
        :param datasets: A dictionary mapping condition names to their
            respective datasets.
        :type datasets: dict[str, Dataset]
        :return: A dictionary mapping condition names to their created
            dataloaders.
        :rtype: dict[str, DataLoader]
        """
        # Compute batch sizes per condition based on batching_mode
        batch_sizes = self._compute_batch_sizes(datasets)
        dataloaders = {}
        for name, dataset in datasets.items():
            dataloaders[name] = self.conditions[name].create_dataloader(
                dataset=dataset,
                batch_size=batch_sizes[name],
                automatic_batching=self.automatic_batching,
                sampler=self._define_sampler(dataset, self.shuffle),
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
        return dataloaders
