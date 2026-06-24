"""Utility class for aggregating multiple dataloaders into a single iterable."""


class _Aggregator:
    """
    Aggregate multiple dataloaders into a unified iterable object.

    The aggregator combines batches produced by multiple dataloaders according
    to the selected batching strategy. It is primarily used to coordinate the
    iteration of multiple training conditions within a single training loop.

    :Example:

        >>> import torch
        >>> from torch.utils.data import DataLoader, TensorDataset
        >>> ds1 = TensorDataset(torch.randn(10, 2))
        >>> ds2 = TensorDataset(torch.randn(15, 2))
        >>> dl1 = DataLoader(ds1, batch_size=5)
        >>> dl2 = DataLoader(ds2, batch_size=5)
        >>> aggregator = _Aggregator(
        ...     {"cond1": dl1, "cond2": dl2},
        ...     batching_mode="common_batch_size"
        ... )
        >>> len(aggregator)
        3
    """

    def __init__(self, dataloaders, batching_mode):
        """
        Initialization of the :class:`_Aggregator` class.

        :param dict[str, DataLoader] dataloaders: The mapping between condition
            names and their corresponding dataloaders.
        :param str batching_mode: The strategy used to aggregate batches across
            dataloaders. Available options are ``"common_batch_size"`` for
            uniform batch sizes across conditions, ``"proportional"`` for batch
            sizes proportional to dataset sizes, and ``"separate_conditions"``
            for iterating through each condition separately.
        :raises NotImplementedError: If the selected batching mode is not yet
            implemented.
        """
        # Raise not implemented error for separate_conditions mode
        if batching_mode == "separate_conditions":
            raise NotImplementedError(
                "Batching mode 'separate_conditions' is not implemented yet."
            )

        # Initialize attributes
        self.dataloaders = dataloaders
        self.batching_mode = batching_mode

    def __len__(self):
        """
        Return the length of the aggregated dataloader. The length is determined
        by the number of iterations required to exhaust the dataloaders based on
        the selected batching mode.

        For ``"separate_conditions"``, the total number of iterations is the sum
        of the lengths of all dataloaders. For all other batching modes, the
        length corresponds to the maximum length among the aggregated
        dataloaders.

        :return: The length of the aggregated dataloader.
        :rtype: int
        """
        # Separate conditions case
        if self.batching_mode == "separate_conditions":
            return sum(len(dl) for dl in self.dataloaders.values())

        return max(len(dl) for dl in self.dataloaders.values())

    def __iter__(self):
        """
        Iterate over the aggregated dataloaders.

        At each iteration, a dictionary containing one batch per dataloader is
        yielded. If a dataloader is exhausted before the others, its iterator is
        restarted automatically to ensure continuous batch generation.

        :yield: The dictionary mapping each condition name to its batch.
        :rtype: Iterator[dict[str, Any]]
        """
        # Initialize iterators for each dataloader
        iterators = {name: iter(dl) for name, dl in self.dataloaders.items()}

        # Iterate until the maximum number of iterations is reached
        for _ in range(len(self)):
            batch = {}

            # Generate a batch for each dataloader
            for name, dataloader in self.dataloaders.items():

                # Attempt to get the next batch from the dataloader's iterator
                try:
                    batch[name] = next(iterators[name])

                # Restart the iterator if it is exhausted
                except StopIteration:
                    iterators[name] = iter(dataloader)
                    batch[name] = next(iterators[name])

            yield batch
