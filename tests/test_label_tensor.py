import torch
import pytest

from pina.label_tensor import LabelTensor
#import pina

data = torch.rand((20, 3))
labels_column = {
    1: {
        "name": "space",
        "dof": ['x', 'y', 'z']
    }
}
labels_row = {
    0: {
        "name": "samples",
        "dof": range(20)
    }
}
labels_all = labels_column | labels_row

@pytest.mark.parametrize("labels", [labels_column, labels_row, labels_all])
def test_constructor(labels):
    LabelTensor(data, labels)

def test_wrong_constructor():
    with pytest.raises(ValueError):
        LabelTensor(data, ['a', 'b'])

@pytest.mark.parametrize("labels", [labels_column, labels_all])
@pytest.mark.parametrize("labels_te", ['z', ['z'], {'space': ['z']}])
def test_extract_column(labels, labels_te):
    tensor = LabelTensor(data, labels)
    new = tensor.extract(labels_te)
    assert new.ndim == tensor.ndim
    assert new.shape[1] == 1
    assert new.shape[0] == 20
    assert torch.all(torch.isclose(data[:, 2].reshape(-1, 1), new))

@pytest.mark.parametrize("labels", [labels_row, labels_all])
@pytest.mark.parametrize("labels_te", [{'samples': [2]}])
def test_extract_row(labels, labels_te):
    tensor = LabelTensor(data, labels)
    new = tensor.extract(labels_te)
    assert new.ndim == tensor.ndim
    assert new.shape[1] == 3
    assert new.shape[0] == 1
    assert torch.all(torch.isclose(data[2].reshape(1, -1), new))

@pytest.mark.parametrize("labels_te", [
    {'samples': [2], 'space': ['z']},
    {'space': 'z', 'samples': 2}
])
def test_extract_2D(labels_te):
    labels = labels_all
    tensor = LabelTensor(data, labels)
    new = tensor.extract(labels_te)
    assert new.ndim == tensor.ndim
    assert new.shape[1] == 1
    assert new.shape[0] == 1
    assert torch.all(torch.isclose(data[2,2].reshape(1, 1), new))

def test_extract_3D():
    labels = labels_all
    data = torch.rand(20, 3, 4)
    labels = {
        1: {
            "name": "space",
            "dof": ['x', 'y', 'z']
        },
        2: {
            "name": "time",
            "dof": range(4)
        },
    }
    labels_te = {
        'space': ['x', 'z'],
        'time': range(1, 4)
    }

    tensor = LabelTensor(data, labels)
    new = tensor.extract(labels_te)
    assert new.ndim == tensor.ndim
    assert new.shape[0] == 20
    assert new.shape[1] == 2
    assert new.shape[2] == 3
    assert torch.all(torch.isclose(
        data[:, 0::2, 1:4].reshape(20, 2, 3),
        new
    ))