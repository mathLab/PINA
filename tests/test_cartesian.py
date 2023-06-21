import torch
import pytest

from pina import LabelTensor, Condition, CartesianDomain, PINN
from pina.problem import SpatialProblem
from pina.model import FeedForward
from pina.operators import nabla



def test_constructor():
    CartesianDomain({'x': [0, 1], 'y': [0, 1]})


def test_is_inside_check_border():
    pt_1 = LabelTensor(torch.tensor([[0.5, 0.5]]), ['x', 'y'])
    pt_2 = LabelTensor(torch.tensor([[1.0, 0.5]]), ['x', 'y'])
    pt_3 = LabelTensor(torch.tensor([[1.5, 0.5]]), ['x', 'y'])
    domain = CartesianDomain({'x': [0, 1], 'y': [0, 1]})
    for pt, exp_result in zip([pt_1, pt_2, pt_3], [True, True, False]):
        assert domain.is_inside(pt, check_border=True) == exp_result

def test_is_inside_not_check_border():
    pt_1 = LabelTensor(torch.tensor([[0.5, 0.5]]), ['x', 'y'])
    pt_2 = LabelTensor(torch.tensor([[1.0, 0.5]]), ['x', 'y'])
    pt_3 = LabelTensor(torch.tensor([[1.5, 0.5]]), ['x', 'y'])
    domain = CartesianDomain({'x': [0, 1], 'y': [0, 1]})
    for pt, exp_result in zip([pt_1, pt_2, pt_3], [True, False, False]):
        assert domain.is_inside(pt, check_border=False) == exp_result

def test_is_inside_fixed_variables():
    pt_1 = LabelTensor(torch.tensor([[0.5, 0.5]]), ['x', 'y'])
    pt_2 = LabelTensor(torch.tensor([[1.0, 0.5]]), ['x', 'y'])
    pt_3 = LabelTensor(torch.tensor([[1.0, 1.5]]), ['x', 'y'])
    domain = CartesianDomain({'x': 1, 'y': [0, 1]})
    for pt, exp_result in zip([pt_1, pt_2, pt_3], [False, True, False]):
        assert domain.is_inside(pt, check_border=False) == exp_result