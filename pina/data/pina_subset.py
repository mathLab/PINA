class PinaSubset:
    """
    TODO
    """
    __slots__ = ['dataset', 'indices']

    def __init__(self, dataset, indices):
        """
        TODO
        """
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        """
        TODO
        """
        return len(self.indices)

    def __getattr__(self, name):
        return self.dataset.__getattribute__(name)
