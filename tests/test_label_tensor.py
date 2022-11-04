import torch
import pytest

from pina import LabelTensor

data = torch.rand((20, 3))
labels = ['a', 'b', 'c']


def test_constructor():
    LabelTensor(data, labels)


def test_wrong_constructor():
    with pytest.raises(ValueError):
        LabelTensor(data, ['a', 'b'])


def test_labels():
    tensor = LabelTensor(data, labels)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.labels == labels
    with pytest.raises(ValueError):
        tensor.labels = labels[:-1]


def test_extract():
    label_to_extract = ['a', 'c']
    tensor = LabelTensor(data, labels)
    new = tensor.extract(label_to_extract)
    assert new.labels == label_to_extract
    assert new.shape[1] == len(label_to_extract)
    assert torch.all(torch.isclose(data[:, 0::2], new))


def test_extract_onelabel():
    label_to_extract = ['a']
    tensor = LabelTensor(data, labels)
    new = tensor.extract(label_to_extract)
    assert new.ndim == 2
    assert new.labels == label_to_extract
    assert new.shape[1] == len(label_to_extract)
    assert torch.all(torch.isclose(data[:, 0].reshape(-1, 1), new))


def test_wrong_extract():
    label_to_extract = ['a', 'cc']
    tensor = LabelTensor(data, labels)
    with pytest.raises(ValueError):
        tensor.extract(label_to_extract)


def test_extract_order():
    label_to_extract = ['c', 'a']
    tensor = LabelTensor(data, labels)
    new = tensor.extract(label_to_extract)
    expected = torch.cat(
        (data[:, 2].reshape(-1, 1), data[:, 0].reshape(-1, 1)),
        dim=1)
    print(expected)
    assert new.labels == label_to_extract
    assert new.shape[1] == len(label_to_extract)
    assert torch.all(torch.isclose(expected, new))


def test_merge():
    tensor = LabelTensor(data, labels)
    tensor_a = tensor.extract('a')
    tensor_b = tensor.extract('b')
    tensor_c = tensor.extract('c')

    tensor_bc = tensor_b.append(tensor_c)
    assert torch.allclose(tensor_bc, tensor.extract(['b', 'c']))

def test_merge():
    tensor = LabelTensor(data, labels)
    tensor_b = tensor.extract('b')
    tensor_c = tensor.extract('c')

    tensor_bc = tensor_b.append(tensor_c)
    assert torch.allclose(tensor_bc, tensor.extract(['b', 'c']))
