"""Module for the Single-Batch Data Loader class."""

import torch


class _SingleBatchDataLoader:
    """
    Data loader wrapper that returns the entire dataset as a single batch.

    This utility is intended for cases where mini-batching is disabled (e.g.
    ``batch_size=None``). The loader yields exactly one batch per iteration.

    In distributed environments, the dataset is automatically partitioned across
    processes according to the current rank and world size. Each process
    receives only its corresponding subset of data.

    In non-distributed environments, the full dataset is returned.
    """

    def __init__(self, dataset):
        """
        Initialization of the :class:`_SingleBatchDataLoader` class.

        In distributed training, the dataset indices are split across processes
        using the current rank and world size, so that each process receives
        only its corresponding subset of data.

        In non-distributed training, the full dataset is loaded.

        The resulting data is converted into a single batch and stored
        internally.

        :param dataset: Dataset object.
        :raises RuntimeError: If the dataset size is smaller than the number of
            distributed processes.
        """
        # Initialize the flag to track if the batch has been yielded
        self._has_yielded = False

        # Distributed training
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            # Get rank and world_size
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()

            # Raise runtime error if the dataset is smaller than the world size
            if len(dataset) < world_size:
                raise RuntimeError(
                    "Dataset size is smaller than the distributed world size. "
                    "Increase the dataset size or use a single GPU."
                )

            # Select dataset idx assigned to the current distributed process
            idx, i = [], rank
            while i < len(dataset):
                idx.append(i)
                i += world_size

            # Fetch the process-specific subset
            self.dataset = dataset.fetch_from_idx_list(idx).to_batch()

        # Non-distributed training
        else:
            self.dataset = dataset.get_all_data().to_batch()

    def __iter__(self):
        """
        Return the data loader iterator.

        :return: The data loader instance itself.
        :rtype: _SingleBatchDataLoader
        """
        # Reset the flag to yield the batch again if iterator is restarted
        self._has_yielded = False
        return self

    def __len__(self):
        """
        Return the number of batches produced by the data loader.

        Since the entire dataset is returned as a single batch, the length is
        always ``1``.

        :return: The number of batches.
        :rtype: int
        """
        return 1

    def __next__(self):
        """
        Return the next batch.

        :return: The dataset converted into a single batch.
        :rtype: _BatchManager
        """
        # Yield the batch only once per iteration
        if self._has_yielded:
            raise StopIteration

        # Set the flag to indicate that the batch has been yielded
        self._has_yielded = True

        return self.dataset
