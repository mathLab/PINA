import torch

from pina.utils import merge_tensors
from pina.label_tensor import LabelTensor
from pina import LabelTensor
from pina.geometry import EllipsoidDomain, CartesianDomain
from pina.utils import check_consistency
import pytest
from pina.geometry import Location


def test_merge_tensors():
    tensor1 = LabelTensor(torch.rand((20, 3)), ['a', 'b', 'c'])
    tensor2 = LabelTensor(torch.zeros((20, 3)), ['d', 'e', 'f'])
    tensor3 = LabelTensor(torch.ones((30, 3)), ['g', 'h', 'i'])

    merged_tensor = merge_tensors((tensor1, tensor2, tensor3))
    assert tuple(merged_tensor.labels) == (
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i')
    assert merged_tensor.shape == (20*20*30, 9)
    assert torch.all(merged_tensor.extract(('d', 'e', 'f')) == 0)
    assert torch.all(merged_tensor.extract(('g', 'h', 'i')) == 1)


def test_check_consistency_correct():
    ellipsoid1 = EllipsoidDomain({'x': [1, 2], 'y': [-2, 1]})
    example_input_pts = LabelTensor(torch.tensor([[0, 0, 0]]), ['x', 'y', 'z'])
    
    check_consistency(example_input_pts, torch.Tensor)
    check_consistency(CartesianDomain, Location, subclass=True)
    check_consistency(ellipsoid1, Location)


def test_check_consistency_incorrect():
    ellipsoid1 = EllipsoidDomain({'x': [1, 2], 'y': [-2, 1]})
    example_input_pts = LabelTensor(torch.tensor([[0, 0, 0]]), ['x', 'y', 'z'])
    
    with pytest.raises(ValueError):
        check_consistency(example_input_pts, Location)
    with pytest.raises(ValueError):
        check_consistency(torch.Tensor, Location, subclass=True)
    with pytest.raises(ValueError):
        check_consistency(ellipsoid1, torch.Tensor)
