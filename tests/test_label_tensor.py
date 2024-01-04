import torch
import pytest

from pina import LabelTensor

data_1D = torch.rand((20))
data_2D = torch.rand((20, 3))
data_3D = torch.rand((20, 3, 2))
data_ND = torch.rand([5]*10)

labels_20 = [f'dof{i}' for i in range(20)]
labels_3 = [f'dim{i}' for i in range(3)]
labels_2 = [f'channel{i}' for i in range(2)]
labels_5 = [f'output{i}' for i in range(5)]

# @pytest.mark.parametrize("rank", [1, 2, 10])
def test_constructor_1D():
    # Label any component
    LabelTensor(data_1D, labels=labels_20)

def test_constructor_2D():
    # Label the column
    LabelTensor(data_2D, labels=labels_3)
    # Label any component 2D
    LabelTensor(data_2D, labels=[labels_3, labels_20])
    LabelTensor(data_2D, labels={'D': labels_3, 'N': labels_20})

    with pytest.raises(ValueError):
        LabelTensor(data_2D, labels=labels_20)
        LabelTensor(data_2D, labels=[labels_20, labels_3]) 
        LabelTensor(data_2D, labels=[labels_3, labels_20[:-1]])
        LabelTensor(data_2D, labels=[labels_3[1:], labels_20])

def test_constructor_3D():
    LabelTensor(data_3D, labels=labels_2)
    LabelTensor(data_3D, labels=[labels_2, labels_3])
    LabelTensor(data_3D, labels=[labels_2, labels_3, labels_20])
    LabelTensor(data_3D, labels={
        'C': labels_2, 'D': labels_3, 'N': labels_20})
    with pytest.raises(ValueError):
        LabelTensor(data_3D, labels=labels_3)
        LabelTensor(data_3D, labels=[labels_3, labels_2])

def test_constructor_ND():
    LabelTensor(data_ND, labels=labels_5)
    LabelTensor(data_ND, labels=[labels_5]*10)
    LabelTensor(data_ND, labels={f'O{i}': labels_5 for i in range(10)})
    with pytest.raises(ValueError):
        LabelTensor(data_ND, labels=labels_20)

# def test_labels():
#     tensor = LabelTensor(data, labels)
#     assert isinstance(tensor, torch.Tensor)
#     assert tensor.labels == labels
#     with pytest.raises(ValueError):
#         tensor.labels = labels[:-1]


@pytest.mark.parametrize("tensor", [
    LabelTensor(data_2D, labels=[labels_3, labels_20]),
    LabelTensor(data_2D, labels=labels_3),
    LabelTensor(data_3D, labels=[labels_2, labels_3, labels_20]),
])
def test_extract_consistency(tensor):
    label_to_extract = ['dim0', 'dim2']
    new = tensor.extract_(label_to_extract)
    assert new.labels.keys() == tensor.labels.keys()
    if len(tensor.labels) == 2:
        assert new.labels[-2] == tensor.labels[-2]
    elif len(tensor.labels) == 3:
        assert new.labels[-3] == tensor.labels[-3]
    elif len(tensor.labels) == 1:
        assert new.labels[-1] == label_to_extract
    # assert new.shape[1] == len(label_to_extract)
    # assert torch.all(torch.isclose(data_2D[:, 0::2], new))


@pytest.mark.parametrize("tensor", [
    LabelTensor(data_2D, labels=[labels_3, labels_20]),
    LabelTensor(data_3D, labels=[labels_2, labels_3, labels_20]),
])
def test_tensor_consistency(tensor):
    tt = tensor.tensor
    mean = tt.mean()
    tt += 1.
    assert isinstance(tt, torch.Tensor)
    torch.testing.assert_close(tt, tensor.tensor)
    torch.testing.assert_close(tt.mean(), mean + 1.)

@pytest.mark.parametrize(("tensor1", "tensor2"), [
    (
        LabelTensor(data_2D, labels=labels_3),
        LabelTensor(data_2D, labels=labels_3),
    ),
    (
        LabelTensor(data_2D, labels=labels_3),
        LabelTensor(data_2D[:-2], labels=labels_3)
    ),
    # (
    #     LabelTensor(data_3D, labels=labels_2),
    #     LabelTensor(data_3D, labels=labels_2),
    # )
])
def test_append(tensor1, tensor2):
    tensor = tensor1.append(tensor2, component=labels_3)
    assert tensor.labels == tensor1.labels
    assert tensor.shape[0] == tensor1.shape[0] + tensor2.shape[0]
    assert torch.allclose(tensor, torch.cat((tensor1, tensor2), dim=0))

# @pytest.mark.parametrize(("tensor1", "tensor2"), [
#     (
#         LabelTensor(data_2D, labels=labels_3),
#         LabelTensor(data_2D.T, labels=labels_3),
#     ),
#     (
#         LabelTensor(data_2D, labels=labels_3),
#         LabelTensor(data_2D[:-2].T, labels=labels_3)
#     ),
#     # (
#     #     LabelTensor(data_3D, labels=labels_2),
#     #     LabelTensor(data_3D, labels=labels_2),
#     # )
# ])
# def test_append_transpose(tensor1, tensor2):
#     tensor = tensor1.append(tensor2, component=labels_3)
#     assert tensor.labels == tensor1.labels
#     assert tensor.shape[0] == tensor1.shape[0] + tensor2.shape[1]
#     assert torch.allclose(tensor, torch.cat((tensor1, tensor2.T), dim=0))

"""
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
    assert new.labels == label_to_extract
    assert new.shape[1] == len(label_to_extract)
    assert torch.all(torch.isclose(expected, new))

"""

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
