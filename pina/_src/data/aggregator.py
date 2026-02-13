"""
Aggregator for multiple dataloaders.
"""


class _Aggregator:
    """
    The class :class:`_Aggregator` is responsible for aggregating multiple
    dataloaders into a single iterable object. It supports different batching
    modes to accommodate various training requirements.
    """

    def __init__(self, dataloaders, batching_mode):
        """
        Initialization of the :class:`_Aggregator` class.

        :param dataloaders: A dictionary mapping condition names to their
            respective dataloaders.
        :type dataloaders: dict[str, DataLoader]
        :param batching_mode: The batching mode to use. Options are
            ``"common_batch_size"``, ``"proportional"``, and
            ``"separate_conditions"``.
        :type batching_mode: str
        """
        self.dataloaders = dataloaders
        self.batching_mode = batching_mode

    def __len__(self):
        """
        Return the length of the aggregated dataloader.

        :return: The length of the aggregated dataloader.
        :rtype: int
        """
        return max(len(dl) for dl in self.dataloaders.values())

    def __iter__(self):
        """
        Return an iterator over the aggregated dataloader.

        :return: An iterator over the aggregated dataloader.
        :rtype: iterator
        """
        if self.batching_mode == "separate_conditions":
            for name, dl in self.dataloaders.items():
                for batch in dl:
                    yield {name: batch}
            return
        iterators = {name: iter(dl) for name, dl in self.dataloaders.items()}
        for _ in range(len(self)):
            batch = {}
            for name, it in iterators.items():
                try:
                    batch[name] = next(it)
                except StopIteration:
                    iterators[name] = iter(self.dataloaders[name])
                    batch[name] = next(iterators[name])
            yield batch
