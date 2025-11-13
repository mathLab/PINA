"""DataLoader module for PinaDataset."""

import itertools
from functools import partial
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SequentialSampler


class DummyDataloader:
    """
    DataLoader that returns the entire dataset in a single batch.
    """

    def __init__(self, dataset):
        """
        Prepare a dataloader object that returns the entire dataset in a single
        batch. Depending on the number of GPUs, the dataset is managed
        as follows:

        - **Distributed Environment** (multiple GPUs): Divides dataset across
            processes using the rank and world size. Fetches only portion of
            data corresponding to the current process.
        - **Non-Distributed Environment** (single GPU): Fetches the entire
            dataset.

        :param PinaDataset dataset: The dataset object to be processed.

        .. note::
           This dataloader is used when the batch size is ``None``.
        """
        # Handle distributed environment
        if PinaSampler.is_distributed():
            # Get rank and world size
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
            # Ensure dataset is large enough
            if len(dataset) < world_size:
                raise RuntimeError(
                    "Dimension of the dataset smaller than world size."
                    " Increase the size of the partition or use a single GPU"
                )
            # Split dataset among processes
            idx, i = [], rank
            while i < len(dataset):
                idx.append(i)
                i += world_size
        else:
            idx = list(range(len(dataset)))

        self.dataset = dataset.getitem_from_list(idx)

    def __iter__(self):
        """
        Iterate over the dataloader.
        """
        return self

    def __len__(self):
        """
        Return the length of the dataloader, which is always 1.
        :return: The length of the dataloader.
        :rtype: int
        """
        return 1

    def __next__(self):
        """
        Return the entire dataset as a single batch.
        :return: The entire dataset.
        :rtype: dict
        """
        return self.dataset


class PinaSampler:
    """
    This class is used to create the sampler instance based on the shuffle
    parameter and the environment in which the code is running.
    """

    def __new__(cls, dataset, shuffle=True):
        """
        Instantiate and initialize the sampler.

        :param PinaDataset dataset: The dataset from which to sample.
        :return: The sampler instance.
        :rtype: :class:`torch.utils.data.Sampler`
        """

        if cls.is_distributed():
            sampler = DistributedSampler(dataset, shuffle=shuffle)
        else:
            if shuffle:
                sampler = torch.utils.data.RandomSampler(dataset)
            else:
                sampler = SequentialSampler(dataset)
        return sampler

    @staticmethod
    def is_distributed():
        """
        Check if the sampler is distributed.
        :return: True if the sampler is distributed, False otherwise.
        :rtype: bool
        """
        return (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        )


def _collect_items(batch):
    """
    Helper function to collect items from a batch of graph data samples.
    :param batch: List of graph data samples.
    """
    to_return = {name: [] for name in batch[0].keys()}
    for sample in batch:
        for k, v in sample.items():
            to_return[k].append(v)
    return to_return


def collate_fn_custom(batch, dataset):
    """
    Override the default collate function to handle datasets without automatic
    batching.
    :param batch: List of indices from the dataset.
    :param dataset: The PinaDataset instance (must be provided).
    """
    return dataset.getitem_from_list(batch)


def collate_fn_default(batch, stack_fn):
    """
    Default collate function that simply returns the batch as is.
    :param batch: List of data samples.
    """
    to_return = _collect_items(batch)
    return {k: stack_fn[k](v) for k, v in to_return.items()}


class PinaDataLoader:
    """
    Custom DataLoader for PinaDataset.
    """

    def __init__(
        self,
        dataset_dict,
        batch_size,
        num_workers=0,
        shuffle=False,
        common_batch_size=True,
        separate_conditions=False,
    ):
        self.dataset_dict = dataset_dict
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.separate_conditions = separate_conditions

        # Batch size None means we want to load the entire dataset in a single
        # batch
        if batch_size is None:
            batch_size_per_dataset = {
                split: None for split in dataset_dict.keys()
            }
        else:
            # Compute batch size per dataset
            if common_batch_size:  # all datasets have the same batch size
                # (the sum of the batch sizes is equal to
                # n_conditions * batch_size)
                batch_size_per_dataset = {
                    split: batch_size for split in dataset_dict.keys()
                }
            else:  # batch size proportional to dataset size (the sum of the
                # batch sizes is equal to the specified batch size)
                batch_size_per_dataset = self._compute_batch_size()

        # Creaete a dataloader per dataset
        self.dataloaders = {
            split: self._create_dataloader(
                dataset, batch_size_per_dataset[split]
            )
            for split, dataset in dataset_dict.items()
        }

    def _compute_batch_size(self):
        """
        Compute an appropriate batch size for the given dataset.
        """
        # Compute number of elements per dataset
        elements_per_dataset = {
            dataset_name: len(dataset)
            for dataset_name, dataset in self.dataset_dict.items()
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

    def _create_dataloader(self, dataset, batch_size):
        """
        Create the dataloader for the given dataset.
        """
        # If batch size is None, use DummyDataloader
        if batch_size is None or batch_size >= len(dataset):
            return DummyDataloader(dataset)

        # Determine the appropriate collate function
        if not dataset.automatic_batching:
            collate_fn = partial(collate_fn_custom, dataset=dataset)
        else:
            collate_fn = partial(collate_fn_default, stack_fn=dataset.stack_fn)

        # Create and return the dataloader
        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            sampler=PinaSampler(dataset, shuffle=self.shuffle),
        )

    def __len__(self):
        """
        Return the length of the dataloader.
        :return: The length of the dataloader.
        :rtype: int
        """
        # If separate conditions, return sum of lengths of all dataloaders
        # else, return max length among dataloaders
        if self.separate_conditions:
            return sum(len(dl) for dl in self.dataloaders.values())
        return max(len(dl) for dl in self.dataloaders.values())

    def __iter__(self):
        """
        Iterate over the dataloader.
        :return: Yields batches from the dataloader.
        :rtype: dict
        """
        if self.separate_conditions:
            for split, dl in self.dataloaders.items():
                for batch in dl:
                    yield {split: batch}
            return

        iterators = {
            split: itertools.cycle(dl) for split, dl in self.dataloaders.items()
        }

        for _ in range(len(self)):
            batch_dict = {}
            for split, it in iterators.items():

                # Iterate through each dataloader and get the next batch
                batch = next(it, None)
                # Check if batch is None (in case of uneven lengths)
                if batch is None:
                    return

                batch_dict[split] = batch
            yield batch_dict
