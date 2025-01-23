import logging
from lightning.pytorch import LightningDataModule
import math
import torch
from ..label_tensor import LabelTensor
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler, \
    RandomSampler
from torch.utils.data.distributed import DistributedSampler
from .dataset import PinaDatasetFactory

class DummyDataloader:
    def __init__(self, dataset, device):
        self.dataset = dataset.get_all_data()

    def __iter__(self):
        return self

    def __len__(self):
        return 1

    def __next__(self):
        return self.dataset

class Collator:
    def __init__(self, max_conditions_lengths, ):
        self.max_conditions_lengths = max_conditions_lengths
        self.callable_function = self._collate_custom_dataloader if \
            max_conditions_lengths is None else (
            self._collate_standard_dataloader)

    @staticmethod
    def _collate_custom_dataloader(batch):
        return batch[0]

    def _collate_standard_dataloader(self, batch):
        """
        Function used to collate the batch
        """
        batch_dict = {}
        if isinstance(batch, dict):
            return batch
        conditions_names = batch[0].keys()

        # Condition names
        for condition_name in conditions_names:
            single_cond_dict = {}
            condition_args = batch[0][condition_name].keys()
            for arg in condition_args:
                data_list = [batch[idx][condition_name][arg] for idx in range(
                    min(len(batch),
                    self.max_conditions_lengths[condition_name]))]
                if isinstance(data_list[0], LabelTensor):
                    single_cond_dict[arg] = LabelTensor.stack(data_list)
                elif isinstance(data_list[0], torch.Tensor):
                    single_cond_dict[arg] = torch.stack(data_list)
                else:
                    raise NotImplementedError(
                        f"Data type {type(data_list[0])} not supported")
            batch_dict[condition_name] = single_cond_dict
        return batch_dict

    def __call__(self, batch):
        return self.callable_function(batch)


class PinaBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size, shuffle, sampler=None):
        if sampler is None:
            if (torch.distributed.is_available() and
                    torch.distributed.is_initialized()):
                rank = torch.distributed.get_rank()
                world_size = torch.distributed.get_world_size()
                sampler = DistributedSampler(dataset, shuffle=shuffle,
                                             rank=rank, num_replicas=world_size)
            else:
                if shuffle:
                    sampler = RandomSampler(dataset)
                else:
                    sampler = SequentialSampler(dataset)
        super().__init__(sampler=sampler, batch_size=batch_size,
                         drop_last=False)

class PinaDataModule(LightningDataModule):
    """
    This class extend LightningDataModule, allowing proper creation and
    management of different types of Datasets defined in PINA
    """

    def __init__(self,
                 collector,
                 train_size=.7,
                 test_size=.2,
                 val_size=.1,
                 predict_size=0.,
                 batch_size=None,
                 shuffle=True,
                 repeat=False,
                 automatic_batching=False
                 ):
        """
        Initialize the object, creating dataset based on input problem
        :param Collector collector: PINA problem
        :param train_size: number/percentage of elements in train split
        :param test_size: number/percentage of elements in test split
        :param val_size: number/percentage of elements in evaluation split
        :param batch_size: batch size used for training
        """
        logging.debug('Start initialization of Pina DataModule')
        logging.info('Start initialization of Pina DataModule')
        super().__init__()
        self.automatic_batching = automatic_batching
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.repeat = repeat

        # Begin Data splitting
        splits_dict = {}
        if train_size > 0:
            splits_dict['train'] = train_size
            self.train_dataset = None
        else:
            self.train_dataloader = super().train_dataloader
        if test_size > 0:
            splits_dict['test'] = test_size
            self.test_dataset = None
        else:
            self.test_dataloader = super().test_dataloader
        if val_size > 0:
            splits_dict['val'] = val_size
            self.val_dataset = None
        else:
            self.val_dataloader = super().val_dataloader
        if predict_size > 0:
            splits_dict['predict'] = predict_size
            self.predict_dataset = None
        else:
            self.predict_dataloader = super().predict_dataloader
        self.collector_splits = self._create_splits(collector, splits_dict)

    def setup(self, stage=None):
        """
        Perform the splitting of the dataset
        """
        logging.debug('Start setup of Pina DataModule obj')
        if stage == 'fit' or stage is None:
            self.train_dataset = PinaDatasetFactory(
                self.collector_splits['train'],
                max_conditions_lengths=self.find_max_conditions_lengths(
                    'train'), automatic_batching=self.automatic_batching)
            if 'val' in self.collector_splits.keys():
                self.val_dataset = PinaDatasetFactory(
                    self.collector_splits['val'],
                    max_conditions_lengths=self.find_max_conditions_lengths(
                        'val'),  automatic_batching=self.automatic_batching
                )
        elif stage == 'test':
            self.test_dataset = PinaDatasetFactory(
                self.collector_splits['test'],
                max_conditions_lengths=self.find_max_conditions_lengths(
                    'test'), automatic_batching=self.automatic_batching
            )
        elif stage == 'predict':
            self.predict_dataset = PinaDatasetFactory(
                self.collector_splits['predict'],
                max_conditions_lengths=self.find_max_conditions_lengths(
                    'predict'), automatic_batching=self.automatic_batching
            )
        else:
            raise ValueError(
                "stage must be either 'fit' or 'test' or 'predict'."
            )

    @staticmethod
    def _split_condition(condition_dict, splits_dict):
        len_condition = len(condition_dict['input_points'])

        lengths = [
            int(math.floor(len_condition * length)) for length in
            splits_dict.values()
        ]

        remainder = len_condition - sum(lengths)
        for i in range(remainder):
            lengths[i % len(lengths)] += 1
        splits_dict = {k: v for k, v in zip(splits_dict.keys(), lengths)
                       }
        to_return_dict = {}
        offset = 0
        for stage, stage_len in splits_dict.items():
            to_return_dict[stage] = {k: v[offset:offset + stage_len]
                                     for k, v in condition_dict.items() if
                                     k != 'equation'
                                     # Equations are NEVER dataloaded
                                     }
            offset += stage_len
        return to_return_dict

    def _create_splits(self, collector, splits_dict):
        """
        Create the dataset objects putting data
        """

        # ----------- Auxiliary function ------------
        def _apply_shuffle(condition_dict, len_data):
            idx = torch.randperm(len_data)
            for k, v in condition_dict.items():
                if k == 'equation':
                    continue
                if isinstance(v, list):
                    condition_dict[k] = [v[i] for i in idx]
                elif isinstance(v, LabelTensor):
                    condition_dict[k] = LabelTensor(v.tensor[idx],
                                                    v.labels)
                elif isinstance(v, torch.Tensor):
                    condition_dict[k] = v[idx]
                else:
                    raise ValueError(f"Data type {type(v)} not supported")
        # ----------- End auxiliary function ------------

        logging.debug('Dataset creation in PinaDataModule obj')
        split_names = list(splits_dict.keys())
        dataset_dict = {name: {} for name in split_names}
        for condition_name, condition_dict in collector.data_collections.items():
            len_data = len(condition_dict['input_points'])
            if self.shuffle:
                _apply_shuffle(condition_dict, len_data)
            for key, data in self._split_condition(condition_dict,
                                                   splits_dict).items():
                dataset_dict[key].update({condition_name: data})
        return dataset_dict

    def find_max_conditions_lengths(self, split):
        max_conditions_lengths = {}
        for k, v in self.collector_splits[split].items():
            if self.batch_size is None:
                max_conditions_lengths[k] = len(v['input_points'])
            elif self.repeat:
                max_conditions_lengths[k] = self.batch_size
            else:
                max_conditions_lengths[k] = min(len(v['input_points']),
                                                self.batch_size)
        return max_conditions_lengths

    def val_dataloader(self):
        """
        Create the validation dataloader
        """
        # Use custom batching (good if batch size is large)
        if self.batch_size is not None:
            # Use default batching in torch DataLoader (good is batch size is small)
            if self.automatic_batching:
                collate = Collator(self.find_max_conditions_lengths('val'))
                return DataLoader(self.val_dataset, self.batch_size,
                          collate_fn=collate)
            collate = Collator(None)
            sampler = PinaBatchSampler(self.val_dataset, self.batch_size, shuffle=False)
            return DataLoader(self.val_dataset, sampler=sampler,
                          collate_fn=collate)
        dataloader = DummyDataloader(self.val_dataset, self.trainer.strategy.root_device)
        dataloader.dataset = self.transfer_batch_to_device(dataloader.dataset, self.trainer.strategy.root_device, 0)
        return dataloader

    def train_dataloader(self):
        """
        Create the training dataloader
        """
        # Use custom batching (good if batch size is large)
        if self.batch_size is not None:
            # Use default batching in torch DataLoader (good is batch size is small)
            if self.automatic_batching:
                collate = Collator(self.find_max_conditions_lengths('train'))
                return DataLoader(self.train_dataset, self.batch_size,
                                  collate_fn=collate)
            collate = Collator(None)
            sampler = PinaBatchSampler(self.train_dataset, self.batch_size,
                                           shuffle=False)
            return DataLoader(self.train_dataset, sampler=sampler,
                              collate_fn=collate)
        dataloader = DummyDataloader(self.train_dataset, self.trainer.strategy.root_device)
        dataloader.dataset = self.transfer_batch_to_device(dataloader.dataset, self.trainer.strategy.root_device, 0)
        return dataloader

    def test_dataloader(self):
        """
        Create the testing dataloader
        """
        raise NotImplementedError("Test dataloader not implemented")

    def predict_dataloader(self):
        """
        Create the prediction dataloader
        """
        raise NotImplementedError("Predict dataloader not implemented")

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """
        Transfer the batch to the device. This method is called in the
        training loop and is used to transfer the batch to the device.
        """
        if isinstance(batch, list):
            return batch
        batch = [
            (k, super(LightningDataModule, self).transfer_batch_to_device(v,
                                                                device,
                                                                dataloader_idx))
            for k, v in batch.items()
        ]

        return batch
