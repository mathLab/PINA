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
@pytest.mark.parametrize("labels_te", [2, [2], {'samples': [2]}])
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
    data = torch.rand((20, 3, 4))
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

# def test_labels():
#     tensor = LabelTensor(data, labels)
#     assert isinstance(tensor, torch.Tensor)
#     assert tensor.labels == labels
#     with pytest.raises(ValueError):
#         tensor.labels = labels[:-1]


# def test_extract():
#     label_to_extract = ['a', 'c']
#     tensor = LabelTensor(data, labels)
#     new = tensor.extract(label_to_extract)
#     assert new.labels == label_to_extract
#     assert new.shape[1] == len(label_to_extract)
#     assert torch.all(torch.isclose(data[:, 0::2], new))


# def test_extract_onelabel():
#     label_to_extract = ['a']
#     tensor = LabelTensor(data, labels)
#     new = tensor.extract(label_to_extract)
#     assert new.ndim == 2
#     assert new.labels == label_to_extract
#     assert new.shape[1] == len(label_to_extract)
#     assert torch.all(torch.isclose(data[:, 0].reshape(-1, 1), new))


# def test_wrong_extract():
#     label_to_extract = ['a', 'cc']
#     tensor = LabelTensor(data, labels)
#     with pytest.raises(ValueError):
#         tensor.extract(label_to_extract)


# def test_extract_order():
#     label_to_extract = ['c', 'a']
#     tensor = LabelTensor(data, labels)
#     new = tensor.extract(label_to_extract)
#     expected = torch.cat(
#         (data[:, 2].reshape(-1, 1), data[:, 0].reshape(-1, 1)),
#         dim=1)
#     assert new.labels == label_to_extract
#     assert new.shape[1] == len(label_to_extract)
#     assert torch.all(torch.isclose(expected, new))


# def test_merge():
#     tensor = LabelTensor(data, labels)
#     tensor_a = tensor.extract('a')
#     tensor_b = tensor.extract('b')
#     tensor_c = tensor.extract('c')

#     tensor_bc = tensor_b.append(tensor_c)
#     assert torch.allclose(tensor_bc, tensor.extract(['b', 'c']))


# def test_merge2():
#     tensor = LabelTensor(data, labels)
#     tensor_b = tensor.extract('b')
#     tensor_c = tensor.extract('c')

#     tensor_bc = tensor_b.append(tensor_c)
#     assert torch.allclose(tensor_bc, tensor.extract(['b', 'c']))


# def test_getitem():
#     tensor = LabelTensor(data, labels)
#     tensor_view = tensor['a']

#     assert tensor_view.labels == ['a']
#     assert torch.allclose(tensor_view.flatten(), data[:, 0])

#     tensor_view = tensor['a', 'c']

#     assert tensor_view.labels == ['a', 'c']
#     assert torch.allclose(tensor_view, data[:, 0::2])

# def test_getitem2():
#     tensor = LabelTensor(data, labels)
#     tensor_view = tensor[:5]
#     assert tensor_view.labels == labels
#     assert torch.allclose(tensor_view, data[:5])

#     idx = torch.randperm(tensor.shape[0])
#     tensor_view = tensor[idx]
#     assert tensor_view.labels == labels


# def test_slice():
#     tensor = LabelTensor(data, labels)
#     tensor_view = tensor[:5, :2]
#     assert tensor_view.labels == labels[:2]
#     assert torch.allclose(tensor_view, data[:5, :2])

#     tensor_view2 = tensor[3]
#     assert tensor_view2.labels == labels
#     assert torch.allclose(tensor_view2, data[3])

#     tensor_view3 = tensor[:, 2]
#     assert tensor_view3.labels == labels[2]
#     assert torch.allclose(tensor_view3, data[:, 2].reshape(-1, 1))
