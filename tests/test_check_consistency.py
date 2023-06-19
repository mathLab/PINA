import torch
from pina import LabelTensor
from pina.geometry import Union, EllipsoidDomain, CartesianDomain
from pina.utils import check_consistency
import pytest
from pina.geometry import Location

cart1 = CartesianDomain({'x': [0, 1], 'y': [0, 1]})
ellipsoid1 = EllipsoidDomain({'x': [1, 2], 'y': [-2, 1]})
union = Union([cart1, ellipsoid1])
example_input_pts = LabelTensor(torch.tensor([[0, 0, 0]]), ['x', 'y', 'z'])
example_output_pts = LabelTensor(torch.tensor([[1, 2]]), ['a', 'b'])


def test_check_consistency_correct():
    check_consistency(example_input_pts, torch.Tensor)
    check_consistency(union, Location)
    check_consistency(ellipsoid1, Location)


def test_check_consistency_incorrect():
    with pytest.raises(ValueError):
        check_consistency(example_input_pts, Location)
    with pytest.raises(ValueError):
        check_consistency(union, torch.Tensor)
    with pytest.raises(ValueError):
        check_consistency(ellipsoid1, torch.Tensor)
