"""
This module provide basic data management functionalities
"""

import logging
from lightning.pytorch import LightningDataModule
import torch
import math
from .. import LabelTensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from functools import partial

class PinaDataset(Dataset):
    def __init__(self, conditions_dict):
        self.conditions_dict = conditions_dict
        print(conditions_dict.keys())
        self.length = self._get_max_len()

    def _get_max_len(self):
        max_len = 0
        for condition in self.conditions_dict.values():
            max_len = max(max_len, len(condition['input_points']))
        return max_len

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            k: {k_data: v[k_data][idx % len(v['input_points'])] for k_data
                in v.keys()} for k, v in self.conditions_dict.items()
        }

def collate_fn(batch, max_conditions_lengths):
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
                min(len(batch), max_conditions_lengths[condition_name]))]
            if isinstance(data_list[0], LabelTensor):
                single_cond_dict[arg] = LabelTensor.cat(data_list)
            elif isinstance(data_list[0], torch.Tensor):
                single_cond_dict[arg] = torch.stack(data_list)
            else:
                raise NotImplementedError(
                    f"Data type {type(data_list[0])} not supported")
        batch_dict[condition_name] = single_cond_dict
    return batch_dict


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
                 repeat=False):
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
            splits_dict['test'] = train_size
            self.test_dataset = None
        else:
            self.test_dataloader = super().test_dataloader
        if val_size > 0:
            splits_dict['val'] = train_size
            self.val_dataset = None
        else:
            self.val_dataloader = super().val_dataloader
        if predict_size > 0:
            splits_dict['predict'] = train_size
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
            self.train_dataset = PinaDataset(self.collector_splits['train'])
            if 'val' in self.collector_splits.keys():
                self.val_dataset = PinaDataset(self.collector_splits['val'])
        elif stage == 'test':
            self.test_dataset = PinaDataset(self.collector_splits['test'])
        elif stage == 'predict':
            self.predict_dataset = PinaDataset(self.collector_splits['predict'])
        else:
            raise ValueError("stage must be either 'fit' or 'test' or 'predict'.")

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
            to_return_dict[stage] = {k:v [offset:offset + stage_len]
                                     for k, v in condition_dict.items() if k != 'equation' # Equations are NEVER dataloaded
                                     }
            offset += stage_len
        return to_return_dict

    def _create_splits(self, collector, splits_dict):
        """
        Create the dataset objects putting data 
        """
        logging.debug('Dataset creation in PinaDataModule obj')
        split_names = list(splits_dict.keys())
        dataset_dict = {name: {} for name in split_names}
        for condition_name, condition_dict in collector.data_collections.items():
            len_data = len(condition_dict['input_points'])
            if self.shuffle:
                idx = torch.randperm(len_data)
                for k, v in condition_dict.items():
                    if k == 'equation':
                        continue
                    if isinstance(v, list):
                        condition_dict[k] = [v[i] for i in idx]
                    elif isinstance(v, (torch.Tensor, LabelTensor)):
                        condition_dict[k] = v[[idx]]
                    else:
                        raise ValueError(f"Data type {type(v)} not supported")
            for key, data in self._split_condition(condition_dict, splits_dict).items():
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
        max_conditions_lengths = self.find_max_conditions_lengths('val')
        collate_fn_val = partial(collate_fn, max_conditions_lengths = max_conditions_lengths)
        return DataLoader(self.val_dataset, self.batch_size,
                          collate_fn=collate_fn_val, shuffle=False # already shuffled in self._create_split
                          )

    def train_dataloader(self):
        """
        Create the training dataloader
        """
        max_conditions_lengths = self.find_max_conditions_lengths('train')
        collate_fn_train = partial(collate_fn, max_conditions_lengths = max_conditions_lengths)
        return DataLoader(self.train_dataset, self.batch_size,
                          collate_fn=collate_fn_train, shuffle=False # already shuffled in self._create_split
                          )

    def test_dataloader(self):
        """
        Create the testing dataloader
        """
        max_conditions_lengths = self.find_max_conditions_lengths('test')
        collate_fn_test = partial(collate_fn, max_conditions_lengths = max_conditions_lengths)
        return DataLoader(self.test_dataset, self.batch_size,
                          collate_fn=collate_fn_test, shuffle=False)

    def predict_dataloader(self):
        """
        Create the prediction dataloader
        """
        max_conditions_lengths = self.find_max_conditions_lengths('predict')
        collate_fn_predict = partial(collate_fn, max_conditions_lengths = max_conditions_lengths)
        return DataLoader(self.predict_dataset, self.batch_size,
                          collate_fn=collate_fn_predict, shuffle=False)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """
        Transfer the batch to the device. This method is called in the
        training loop and is used to transfer the batch to the device.
        """
        '''
        for i in batch.keys():
            for j in batch[i].keys():
                batch[i][j] = batch[i][j].to(device)
        '''
        batch = {k: super(LightningDataModule, self).transfer_batch_to_device(v, device, dataloader_idx)
                 for k, v in batch.items()}
        return batch


from ..label_tensor import LabelTensor
import torch

