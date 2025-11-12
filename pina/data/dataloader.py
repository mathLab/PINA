from torch.utils.data import DataLoader
from functools import partial
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SequentialSampler
import torch


class DummyDataloader:

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
        print("Using DummyDataloader")
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
        else:
            idx = list(range(len(dataset)))

        self.dataset = dataset._getitem_from_list(idx)

    def __iter__(self):
        return self

    def __len__(self):
        return 1

    def __next__(self):
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

        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            sampler = DistributedSampler(dataset, shuffle=shuffle)
        else:
            if shuffle:
                sampler = torch.utils.data.RandomSampler(dataset)
            else:
                sampler = SequentialSampler(dataset)
        return sampler


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
    Override the default collate function to handle datasets without automatic batching.
    :param batch: List of indices from the dataset.
    :param dataset: The PinaDataset instance (must be provided).
    """
    return dataset._getitem_from_list(batch)


def collate_fn_default(batch, stack_fn):
    """
    Default collate function that simply returns the batch as is.
    :param batch: List of data samples.
    """
    print("Using default collate function")
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
        shuffle=False,
        num_workers=0,
        collate_fn=None,
        common_batch_size=True,
    ):
        self.dataset_dict = dataset_dict
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.collate_fn = collate_fn

        print(batch_size)

        if batch_size is None:
            batch_size_per_dataset = {
                split: None for split in dataset_dict.keys()
            }
        else:
            if common_batch_size:
                batch_size_per_dataset = {
                    split: batch_size for split in dataset_dict.keys()
                }
            else:
                batch_size_per_dataset = self._compute_batch_size()
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
        elements_per_dataset = {
            dataset_name: len(dataset)
            for dataset_name, dataset in self.dataset_dict.items()
        }
        total_elements = sum(el for el in elements_per_dataset.values())
        portion_per_dataset = {
            name: el / total_elements
            for name, el in elements_per_dataset.items()
        }
        batch_size_per_dataset = {
            name: max(1, int(portion * self.batch_size))
            for name, portion in portion_per_dataset.items()
        }
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
        print(batch_size)
        if batch_size is None:
            return DummyDataloader(dataset)

        if not dataset.automatic_batching:
            collate_fn = partial(collate_fn_custom, dataset=dataset)
        else:
            collate_fn = partial(collate_fn_default, stack_fn=dataset.stack_fn)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            sampler=PinaSampler(dataset, shuffle=self.shuffle),
        )

    def __len__(self):
        return max(len(dl) for dl in self.dataloaders.values())

    def __iter__(self):
        """
        Restituisce un iteratore che produce dizionari di batch.

        Itera per un numero di passi pari al dataloader più lungo (come da __len__)
        e fa ricominciare i dataloader più corti quando si esauriscono.
        """
        # 1. Crea un iteratore per ogni dataloader
        iterators = {split: iter(dl) for split, dl in self.dataloaders.items()}

        # 2. Itera per il numero di batch del dataloader più lungo
        for _ in range(len(self)):

            # 3. Prepara il dizionario di batch per questo step
            batch_dict = {}

            # 4. Ottieni il prossimo batch da ogni iteratore
            for split, it in iterators.items():
                try:
                    batch = next(it)
                except StopIteration:
                    # 5. Se un iteratore è esaurito, resettalo e prendi il primo batch
                    new_it = iter(self.dataloaders[split])
                    iterators[split] = new_it  # Salva il nuovo iteratore
                    batch = next(new_it)

                batch_dict[split] = batch

            # 6. Restituisci il dizionario di batch
            yield batch_dict
