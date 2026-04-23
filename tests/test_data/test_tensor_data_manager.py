import torch
from pina import LabelTensor
from pina.data.manager import _DataManager, _TensorDataManager, _BatchManager


# Define data for testing
standard_tensor = torch.rand((10, 3))
label_tensor = LabelTensor(standard_tensor, labels=["a", "b", "c"])


def test_constructor():

    # Create data manager
    data_manager = _DataManager(standard=standard_tensor, labeled=label_tensor)

    # Check that the data manager is an instance of _TensorDataManager
    assert isinstance(data_manager, _TensorDataManager)

    # Check that the attributes are set correctly
    assert hasattr(data_manager, "standard")
    assert hasattr(data_manager, "labeled")

    # Check that the attributes have the correct types
    assert isinstance(data_manager.standard, torch.Tensor)
    assert isinstance(data_manager.labeled, LabelTensor)

    # Check that the values of the attributes are correct
    assert torch.equal(data_manager.standard, standard_tensor)
    assert torch.equal(data_manager.labeled, label_tensor)


def test_create_batch():

    # Create data manager
    data_manager = _DataManager(standard=standard_tensor, labeled=label_tensor)

    # Batch over indices
    idx = [0, 2]
    batch = _TensorDataManager.create_batch([data_manager[idx] for idx in idx])

    # Check that the batch is an instance of _BatchManager
    assert isinstance(batch, _BatchManager)

    # Check that the attributes are set correctly
    assert hasattr(batch, "standard")
    assert hasattr(batch, "labeled")

    # Check that the attributes have the correct types
    assert isinstance(batch.standard, torch.Tensor)
    assert isinstance(batch.labeled, LabelTensor)

    # Check that the values of the attributes are correct
    assert torch.equal(batch.standard, standard_tensor[idx])
    assert torch.equal(batch.labeled, label_tensor[idx])
