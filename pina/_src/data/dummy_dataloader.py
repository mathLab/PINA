"""
Module containing the ``DummyDataloader`` class
"""

import torch


class DummyDataloader:
    """
    A dummy dataloader that returns the entire dataset in a single batch. This
    is used when the batch size is ``None``. It supports both distributed and
    non-distributed environments. In a distributed environment, it divides the
    dataset across processes using the rank and world size, fetching only the
    portion of data corresponding to the current process. In a non-distributed
    environment, it fetches the entire dataset.
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
            self.dataset = dataset.fetch_from_idx_list(idx).to_batch()
        else:
            self.dataset = dataset.get_all_data().to_batch()

    def __iter__(self):
        return self

    def __len__(self):
        return 1

    def __next__(self):
        return self.dataset
