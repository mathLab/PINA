import torch
import pytest
from pina.data import _Creator


"""
Note: this test intentionally avoids relying on the actual Condition and
DataLoader implementations in order to keep the test focused on the creator
logic itself and independent from the behavior of external classes. The full
pipeline is tested in the DataLoader tests, which ensures that the creator works
correctly when used in the intended context.
"""


# Define a dummy dataset for testing purposes
class DummyDataset:
    def __init__(self, data, length=None):
        self.data = data
        self.dataset_length = len(data) if length is None else length
        self.iterable_length = None

    def __len__(self):
        return len(self.data)


# Define a dummy dataloader for testing purposes
class DummyDataloader:
    def create_dataloader(
        self,
        dataset,
        batch_size,
        automatic_batching,
        sampler,
        num_workers,
        pin_memory,
    ):
        return {
            "dataset": dataset,
            "batch_size": batch_size,
            "automatic_batching": automatic_batching,
            "sampler": sampler,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }


# Create dataloaders for testing
dataloaders = {
    "dataset_1": DummyDataloader(),
    "dataset_2": DummyDataloader(),
}


@pytest.mark.parametrize("batching_mode", _Creator._AVAIL_BATCHING_MODES)
def test_constructor(batching_mode):

    _Creator(
        batching_mode=batching_mode,
        batch_size=4,
        shuffle=False,
        automatic_batching=True,
        num_workers=0,
        pin_memory=False,
        conditions=dataloaders,
    )

    # Should fail if an invalid batching mode is provided
    with pytest.raises(ValueError):
        _Creator(
            batching_mode="invalid_mode",
            batch_size=4,
            shuffle=False,
            automatic_batching=True,
            num_workers=0,
            pin_memory=False,
            conditions=dataloaders,
        )


@pytest.mark.parametrize(
    "batching_mode, batch_size, expected_batch_sizes, expected_max_len",
    [
        (
            "common_batch_size",
            2,
            {"dataset_1": 2, "dataset_2": 2},
            {"dataset_1": None, "dataset_2": 4},
        ),
        (
            "common_batch_size",
            None,
            {"dataset_1": 3, "dataset_2": 4},
            {"dataset_1": 4, "dataset_2": None},
        ),
        (
            "separate_conditions",
            2,
            {"dataset_1": 2, "dataset_2": 2},
            {"dataset_1": None, "dataset_2": None},
        ),
        (
            "separate_conditions",
            None,
            {"dataset_1": 3, "dataset_2": 4},
            {"dataset_1": None, "dataset_2": None},
        ),
        (
            "proportional",
            4,
            {"dataset_1": 1, "dataset_2": 3},
            {"dataset_1": None, "dataset_2": None},
        ),
    ],
)
@pytest.mark.parametrize("shuffle", [True, False])
def test_call(
    batching_mode,
    batch_size,
    expected_batch_sizes,
    expected_max_len,
    shuffle,
):

    # Create dummy datasets
    datasets = {
        "dataset_1": DummyDataset([1, 2, 3], length=2),
        "dataset_2": DummyDataset([10, 20, 30, 40], length=4),
    }

    # Initialize the creator
    creator = _Creator(
        batching_mode=batching_mode,
        batch_size=batch_size,
        shuffle=shuffle,
        automatic_batching=True,
        num_workers=0,
        pin_memory=False,
        conditions=dataloaders,
    )

    # Call the creator to create dataloaders
    created_loaders = creator(datasets)

    # Check that dataloaders are created for all conditions
    assert set(created_loaders.keys()) == set(datasets.keys())

    # Iterate over datasets
    for name in datasets:

        # Assert that the dataloader is created with the correct parameters
        assert created_loaders[name]["dataset"] is datasets[name]
        assert created_loaders[name]["batch_size"] == expected_batch_sizes[name]
        assert created_loaders[name]["automatic_batching"] is True
        assert created_loaders[name]["num_workers"] == 0
        assert created_loaders[name]["pin_memory"] is False
        assert datasets[name].iterable_length == expected_max_len[name]

        # Check that the correct sampler is used based on the shuffle parameter
        if shuffle:
            assert isinstance(
                created_loaders[name]["sampler"],
                torch.utils.data.RandomSampler,
            )
        else:
            assert isinstance(
                created_loaders[name]["sampler"],
                torch.utils.data.SequentialSampler,
            )
