import torch
import pytest
from pina.data import _SingleBatchDataLoader


# Initialize the test environment
size = 8
distributed_rank = 1
distributed_world_size = 3
full_data_value = "all"


# Helper functions for testing
def _distributed_idx(size):
    return list(range(distributed_rank, size, distributed_world_size))


# Helper function to set up the distributed environment for testing
def _setup_distributed_environment(monkeypatch, distribute):
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: distribute)

    if distribute:
        monkeypatch.setattr(
            torch.distributed, "get_rank", lambda: distributed_rank
        )
        monkeypatch.setattr(
            torch.distributed,
            "get_world_size",
            lambda: distributed_world_size,
        )


# Create a dummy data class for testing purposes
class DummyData:
    def __init__(self, value):
        self.value = value

    def to_batch(self):
        return self


# Create a dummy dataset class for testing purposes
class DummyDataset:
    def __init__(self, size=size):
        self.size = size
        self.fetched_indices = None
        self.get_all_data_called = False

    def __len__(self):
        return self.size

    def get_all_data(self):
        self.get_all_data_called = True
        return DummyData(full_data_value)

    def fetch_from_idx_list(self, idx):
        self.fetched_indices = idx
        return DummyData(idx)


@pytest.mark.parametrize("distribute", [True, False])
def test_constructor(monkeypatch, distribute):

    # Set up distributed mock environment
    _setup_distributed_environment(monkeypatch, distribute)

    # Create dataset and data loader
    dataset = DummyDataset()
    data_loader = _SingleBatchDataLoader(dataset)

    # Distributed case
    if distribute:
        expected_value = _distributed_idx(size)
        assert data_loader.dataset.value == expected_value
        assert dataset.fetched_indices == expected_value

    # Non-distributed case
    else:
        assert data_loader.dataset.value == full_data_value
        assert dataset.get_all_data_called is True

    # Verify that the data loader yields exactly one batch per iteration
    assert len(data_loader) == 1

    # Should fail if dataset is smaller than world size in distributed case
    if distribute:
        small_dataset = DummyDataset(size=distributed_world_size - 1)
        with pytest.raises(RuntimeError):
            _SingleBatchDataLoader(small_dataset)


@pytest.mark.parametrize("distribute", [True, False])
def test_iter(monkeypatch, distribute):

    # Set up distributed mock environment
    _setup_distributed_environment(monkeypatch, distribute)

    # Create dataset and data loader
    dataset = DummyDataset()
    data_loader = _SingleBatchDataLoader(dataset)

    # Iterate through the data loader
    batches = list(data_loader)

    # Expected value based on the distributed setting
    expected_value = _distributed_idx(size) if distribute else full_data_value

    # Verify iteration behavior
    assert len(batches) == 1
    assert batches[0].value == expected_value
