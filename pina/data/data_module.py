
import torch
from lightning import LightningDataModule
from .sample_dataset import SamplePointDataset
from .data_dataset import DataPointDataset
from .unsupervised_dataset import UnsupervisedDataset
from .pina_dataloader import PinaDataLoader


class PinaDataModule(LightningDataModule):
    def __init__(self, problem, device, train_size=.7, test_size=.2, eval_size=.1, batch_size=None):
        self.sample_dataset = SamplePointDataset(problem, device)
        self.data_dataset = DataPointDataset(problem, device)
        self.unsupervised_dataset = UnsupervisedDataset(problem, device)
        self.split_length = []
        self.split_names = []
        if train_size > 0:
            self.split_names.append('train')
            self.split_length.append(train_size)
        if test_size > 0:
            self.split_length.append(test_size)
            self.split_names.append('test')
        if eval_size > 0:
            self.split_length.append(eval_size)
            self.split_names.append('eval')

        self.batch_size = batch_size
        self.condition_names = None
        self.splits = {k: {} for k in self.split_names}

    def setup(self, stage=None):
        self.extract_conditions()
        if stage == 'fit' or stage is None:
            if len(self.data_dataset) > 0:
                print(self.split_length)
                data_split = self.random_split(
                    self.data_dataset, self.split_length)
                for i in range(len(self.split_length)):
                    self.splits[self.split_names[i]]['supervised'] = data_split[i]
            if len(self.sample_dataset) > 0:
                sample_split = torch.utils.data.random_split(
                    self.sample_dataset, self.split_length)
                for i in range(len(self.split_length)):
                    self.splits[self.split_names[i]]['sample'] = sample_split[i]
            if len(self.unsupervised_dataset) > 0:
                unsupervised_split = torch.utils.data.random_split(
                    self.unsupervised_dataset, self.split_length)
                for i in range(len(self.split_length)):
                    self.splits[self.split_names[i]]['unsupervised'] = unsupervised_split[i]
        elif stage == 'test':
            raise NotImplementedError("Testing pipeline not implemented yet")
        else:
            raise ValueError("stage must be either 'fit' or 'test'")

    def extract_conditions(self):
        #Extract number of conditions
        n_data_conditions = len(self.data_dataset.condition_names)
        n_phys_conditions = len(self.sample_dataset.condition_names)

        #Increment indices in data condition and update names dict
        self.data_dataset.condition_names = {key + n_phys_conditions: value
                                          for key, value in self.data_dataset.condition_names.items()}
        self.unsupervised_dataset.condition_names = {key + n_phys_conditions + n_data_conditions: value
                                          for key, value in self.data_dataset.condition_names.items()}
        self.data_dataset.condition_indices += n_phys_conditions
        self.unsupervised_dataset.condition_indices += n_phys_conditions + n_data_conditions

        self.condition_names = {**self.data_dataset.condition_names,
                                **self.unsupervised_dataset.condition_names,
                                **self.sample_dataset.condition_names}

    def train_dataloader(self):
        return PinaDataLoader(self.splits['train'], self.batch_size, self.condition_names)

    def test_dataloader(self):
        return PinaDataLoader(self.test_splits, self.batch_size, self.condition_names)

    def eval_dataloader(self):
        return PinaDataLoader(self.test_splits, self.batch_size, self.condition_names)

    @staticmethod
    def random_split(dataset, lengths, seed=None) :
        import math
        """
        """
        if sum(lengths) - 1 < 1e-3:
            lengths = [int(math.floor(len(dataset)  * length)) for length in lengths]

            remainder = len(dataset) - sum(lengths)
            for i in range(remainder):
                lengths[i % len(lengths)] += 1
        elif sum(lengths) - 1 >= 1e-3:
            raise ValueError(f"Sum of lengths is {sum(lengths)} less than 1")

        if sum(lengths) != len(dataset):
            raise ValueError(
                "Sum of lengths is not equal to dataset length"
            )
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
            indices = torch.randperm(sum(lengths), generator=generator).tolist()
        else:
            indices = torch.randperm(sum(lengths)).tolist()
        print(lengths)
        offsets = [sum(lengths[:i]) if i > 0 else 0 for i in range(len(lengths))]
        for offset, length in zip(offsets, lengths):
            print(indices[offset : offset+length])
            print(offset, offset+length)
        return [
            PinaSubset(dataset, indices[offset : offset+length])
            for offset, length in zip(offsets, lengths)
        ]

class PinaSubset:
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[idx, self.indices]

