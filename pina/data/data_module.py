"""
This module contains the PinaDataModule class, which extends the
LightningDataModule class to allow proper creation and management of
different types of Datasets defined in PINA.
"""

import warnings
from lightning.pytorch import LightningDataModule
import torch
from ..label_tensor import LabelTensor
from .dataset import PinaDatasetFactory
from .dataloader import PinaDataLoader


class PinaDataModule(LightningDataModule):
    """
    This class extends :class:`~lightning.pytorch.core.LightningDataModule`,
    allowing proper creation and management of different types of datasets
    defined in PINA.
    """

    def __init__(
        self,
        problem,
        train_size=0.7,
        test_size=0.2,
        val_size=0.1,
        batch_size=None,
        shuffle=True,
        common_batch_size=True,
        separate_conditions=False,
        automatic_batching=None,
        num_workers=0,
        pin_memory=False,
    ):
        """
        Initialize the object and creating datasets based on the input problem.

        :param AbstractProblem problem: The problem containing the data on which
            to create the datasets and dataloaders.
        :param float train_size: Fraction of elements in the training split. It
            must be in the range [0, 1].
        :param float test_size: Fraction of elements in the test split. It must
            be in the range [0, 1].
        :param float val_size: Fraction of elements in the validation split. It
            must be in the range [0, 1].
        :param int batch_size: The batch size used for training. If ``None``,
            the entire dataset is returned in a single batch.
            Default is ``None``.
        :param bool shuffle: Whether to shuffle the dataset before splitting.
            Default ``True``.
        :param bool common_batch_size: If ``True``, the same batch size is used
            for all conditions. If ``False``, each condition can have its own
            batch size, proportional to the size of the dataset in that
            condition. Default is ``True``.
        :param bool separate_conditions: If ``True``, dataloaders for each
            condition are iterated separately. Default is ``False``.
        :param automatic_batching: If ``True``, automatic PyTorch batching
            is performed, which consists of extracting one element at a time
            from the dataset and collating them into a batch. This is useful
            when the dataset is too large to fit into memory. On the other hand,
            if ``False``, the items are retrieved from the dataset all at once
            avoind the overhead of collating them into a batch and reducing the
            ``__getitem__`` calls to the dataset. This is useful when the
            dataset fits into memory. Avoid using automatic batching when
            ``batch_size`` is large. Default is ``False``.
        :param int num_workers: Number of worker threads for data loading.
            Default ``0`` (serial loading).
        :param bool pin_memory: Whether to use pinned memory for faster data
            transfer to GPU. Default ``False``.

        :raises ValueError: If at least one of the splits is negative.
        :raises ValueError: If the sum of the splits is different from 1.

        .. seealso::
            For more information on multi-process data loading, see:
            https://pytorch.org/docs/stable/data.html#multi-process-data-loading

            For details on memory pinning, see:
            https://pytorch.org/docs/stable/data.html#memory-pinning
        """
        super().__init__()

        # Store fixed attributes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.common_batch_size = common_batch_size
        self.separate_conditions = separate_conditions
        self.automatic_batching = automatic_batching

        # If batch size is None, num_workers has no effect
        if batch_size is None and num_workers != 0:
            warnings.warn(
                "Setting num_workers when batch_size is None has no effect on "
                "the DataLoading process."
            )
            self.num_workers = 0
        else:
            self.num_workers = num_workers

        # If batch size is None, pin_memory has no effect
        if batch_size is None and pin_memory:
            warnings.warn(
                "Setting pin_memory to True has no effect when "
                "batch_size is None."
            )
            self.pin_memory = False
        else:
            self.pin_memory = pin_memory

        # Collect data
        problem.collect_data()

        # Check if the splits are correct
        self._check_slit_sizes(train_size, test_size, val_size)

        # Split input data into subsets
        splits_dict = {}
        if train_size > 0:
            splits_dict["train"] = train_size
            self.train_dataset = None
        else:
            # Use the super method to create the train dataloader which
            # raises NotImplementedError
            self.train_dataloader = super().train_dataloader
        if test_size > 0:
            splits_dict["test"] = test_size
            self.test_dataset = None
        else:
            # Use the super method to create the train dataloader which
            # raises NotImplementedError
            self.test_dataloader = super().test_dataloader
        if val_size > 0:
            splits_dict["val"] = val_size
            self.val_dataset = None
        else:
            # Use the super method to create the train dataloader which
            # raises NotImplementedError
            self.val_dataloader = super().val_dataloader

        self.data_splits = self._create_splits(
            problem.collected_data, splits_dict
        )
        self.transfer_batch_to_device = self._transfer_batch_to_device

    def setup(self, stage=None):
        """
        Create the dataset objects for the given stage.
        If the stage is "fit", the training and validation datasets are created.
        If the stage is "test", the testing dataset is created.

        :param str stage: The stage for which to perform the dataset setup.

        :raises ValueError: If the stage is neither "fit" nor "test".
        """
        if stage == "fit" or stage is None:
            self.train_dataset = PinaDatasetFactory(
                self.data_splits["train"],
                automatic_batching=self.automatic_batching,
            )
            if "val" in self.data_splits.keys():
                self.val_dataset = PinaDatasetFactory(
                    self.data_splits["val"],
                    automatic_batching=self.automatic_batching,
                )
        elif stage == "test":
            self.test_dataset = PinaDatasetFactory(
                self.data_splits["test"],
                automatic_batching=self.automatic_batching,
            )
        else:
            raise ValueError("stage must be either 'fit' or 'test'.")

    @staticmethod
    def _split_condition(single_condition_dict, splits_dict):
        """
        Split the condition into different stages.

        :param dict single_condition_dict: The condition to be split.
        :param dict splits_dict: The dictionary containing the number of
            elements in each stage.
        :return: A dictionary containing the split condition.
        :rtype: dict
        """

        len_condition = len(single_condition_dict["input"])

        lengths = [
            int(len_condition * length) for length in splits_dict.values()
        ]

        remainder = len_condition - sum(lengths)
        for i in range(remainder):
            lengths[i % len(lengths)] += 1

        splits_dict = {
            k: max(1, v) for k, v in zip(splits_dict.keys(), lengths)
        }
        to_return_dict = {}
        offset = 0

        for stage, stage_len in splits_dict.items():
            to_return_dict[stage] = {
                k: v[offset : offset + stage_len]
                for k, v in single_condition_dict.items()
                if k != "equation"
                # Equations are NEVER dataloaded
            }
            if offset + stage_len >= len_condition:
                offset = len_condition - 1
                continue
            offset += stage_len
        return to_return_dict

    def _create_splits(self, collector, splits_dict):
        """
        Create the dataset objects putting data in the correct splits.

        :param Collector collector: The collector object containing the data.
        :param dict splits_dict: The dictionary containing the number of
            elements in each stage.
        :return: The dictionary containing the dataset objects.
        :rtype: dict
        """

        # ----------- Auxiliary function ------------
        def _apply_shuffle(condition_dict, len_data):
            idx = torch.randperm(len_data)
            for k, v in condition_dict.items():
                if k == "equation":
                    continue
                if isinstance(v, list):
                    condition_dict[k] = [v[i] for i in idx]
                elif isinstance(v, LabelTensor):
                    condition_dict[k] = LabelTensor(v.tensor[idx], v.labels)
                elif isinstance(v, torch.Tensor):
                    condition_dict[k] = v[idx]
                else:
                    raise ValueError(f"Data type {type(v)} not supported")

        # ----------- End auxiliary function ------------

        split_names = list(splits_dict.keys())
        dataset_dict = {name: {} for name in split_names}
        for (
            condition_name,
            condition_dict,
        ) in collector.items():
            len_data = len(condition_dict["input"])
            if self.shuffle:
                _apply_shuffle(condition_dict, len_data)
            for key, data in self._split_condition(
                condition_dict, splits_dict
            ).items():
                dataset_dict[key].update({condition_name: data})
        return dataset_dict

    def _create_dataloader(self, split, dataset):
        """ "
        Create the dataloader for the given split.

        :param str split: The split on which to create the dataloader.
        :param str dataset: The dataset to be used for the dataloader.
        :return: The dataloader for the given split.
        :rtype: torch.utils.data.DataLoader
        """
        # Suppress the warning about num_workers.
        # In many cases, especially for PINNs,
        # serial data loading can outperform parallel data loading.
        warnings.filterwarnings(
            "ignore",
            message=(
                "The '(train|val|test)_dataloader' does not have many workers "
                "which may be a bottleneck."
            ),
            module="lightning.pytorch.trainer.connectors.data_connector",
        )
        return PinaDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=None,
            common_batch_size=self.common_batch_size,
            separate_conditions=self.separate_conditions,
        )

    def val_dataloader(self):
        """
        Create the validation dataloader.

        :return: The validation dataloader
        :rtype: torch.utils.data.DataLoader
        """
        return self._create_dataloader("val", self.val_dataset)

    def train_dataloader(self):
        """
        Create the training dataloader

        :return: The training dataloader
        :rtype: torch.utils.data.DataLoader
        """
        return self._create_dataloader("train", self.train_dataset)

    def test_dataloader(self):
        """
        Create the testing dataloader

        :return: The testing dataloader
        :rtype: torch.utils.data.DataLoader
        """
        return self._create_dataloader("test", self.test_dataset)

    @staticmethod
    def _transfer_batch_to_device_dummy(batch, device, dataloader_idx):
        """
        Transfer the batch to the device. This method is used when the batch
        size is None: batch has already been transferred to the device.

        :param list[tuple] batch: List of tuple where the first element of the
            tuple is the condition name and the second element is the data.
        :param torch.device device: Device to which the batch is transferred.
        :param int dataloader_idx: Index of the dataloader.
        :return: The batch transferred to the device.
        :rtype: list[tuple]
        """

        return batch

    def _transfer_batch_to_device(self, batch, device, dataloader_idx):
        """
        Transfer the batch to the device. This method is called in the
        training loop and is used to transfer the batch to the device.

        :param dict batch: The batch to be transferred to the device.
        :param torch.device device: The device to which the batch is
            transferred.
        :param int dataloader_idx: The index of the dataloader.
        :return: The batch transferred to the device.
        :rtype: list[tuple]
        """

        batch = [
            (
                k,
                super(LightningDataModule, self).transfer_batch_to_device(
                    v, device, dataloader_idx
                ),
            )
            for k, v in batch.items()
        ]

        return batch

    @staticmethod
    def _check_slit_sizes(train_size, test_size, val_size):
        """
        Check if the splits are correct. The splits sizes must be positive and
        the sum of the splits must be 1.

        :param float train_size: The size of the training split.
        :param float test_size: The size of the testing split.
        :param float val_size: The size of the validation split.

        :raises ValueError: If at least one of the splits is negative.
        :raises ValueError: If the sum of the splits is different
            from 1.
        """

        if train_size < 0 or test_size < 0 or val_size < 0:
            raise ValueError("The splits must be positive")
        if abs(train_size + test_size + val_size - 1) > 1e-6:
            raise ValueError("The sum of the splits must be 1")

    @property
    def input(self):
        """
        Return all the input points coming from all the datasets.

        :return: The input points for training.
        :rtype: dict
        """

        to_return = {}
        if hasattr(self, "train_dataset") and self.train_dataset is not None:
            to_return["train"] = self.train_dataset.input
        if hasattr(self, "val_dataset") and self.val_dataset is not None:
            to_return["val"] = self.val_dataset.input
        if hasattr(self, "test_dataset") and self.test_dataset is not None:
            to_return["test"] = self.test_dataset.input
        return to_return
