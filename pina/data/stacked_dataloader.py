import torch
from math import ceil


class StackedDataLoader:
    def __init__(self, datasets, batch_size=32, shuffle=True):
        for d in datasets.values():
            if d.is_graph_dataset:
                raise ValueError("Each dataset must be a dictionary")
        self.chunks = {}
        self.total_length = 0
        self.indices = []

        self._init_chunks(datasets)
        self.indices = list(range(self.total_length))
        self.batch_size = batch_size
        self.shuffle = shuffle
        if self.shuffle:
            torch.random.manual_seed(42)
            self.indices = torch.randperm(self.total_length).tolist()
        self.datasets = datasets

    def _init_chunks(self, datasets):
        inc = 0
        total_length = 0
        for name, dataset in datasets.items():
            self.chunks[name] = {"start": inc, "end": inc + len(dataset)}
            inc += len(dataset)
        self.total_length = inc

    def __len__(self):
        return ceil(self.total_length / self.batch_size)

    def _build_batch_indices(self, batch_idx):
        start = batch_idx * self.batch_size
        end = min(start + self.batch_size, self.total_length)
        return self.indices[start:end]

    def __iter__(self):
        for batch_idx in range(len(self)):
            batch_indices = self._build_batch_indices(batch_idx)
            batch_data = {}
            for name, chunk in self.chunks.items():
                local_indices = [
                    idx - chunk["start"]
                    for idx in batch_indices
                    if chunk["start"] <= idx < chunk["end"]
                ]
                if local_indices:
                    batch_data[name] = self.datasets[name].getitem_from_list(
                        local_indices
                    )
            yield batch_data
