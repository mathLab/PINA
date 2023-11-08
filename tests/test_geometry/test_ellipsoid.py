import torch
import pytest

from pina import LabelTensor
from pina.geometry import EllipsoidDomain


def test_constructor():
    EllipsoidDomain({'x': [0, 1], 'y': [0, 1]})
    EllipsoidDomain({'x': [0, 1], 'y': [0, 1]}, sample_surface=True)


def test_is_inside_sample_surface_false():
    domain = EllipsoidDomain({'x': [0, 1], 'y': [0, 1]}, sample_surface=False)
    pt_1 = LabelTensor(torch.tensor([[0.5, 0.5]]), ['x', 'y'])
    pt_2 = LabelTensor(torch.tensor([[1.0, 0.5]]), ['x', 'y'])
    pt_3 = LabelTensor(torch.tensor([[1.5, 0.5]]), ['x', 'y'])
    for pt, exp_result in zip([pt_1, pt_2, pt_3], [True, False, False]):
        assert domain.is_inside(pt) == exp_result
    for pt, exp_result in zip([pt_1, pt_2, pt_3], [True, True, False]):
        assert domain.is_inside(pt, check_border=True) == exp_result


def test_is_inside_sample_surface_true():
    domain = EllipsoidDomain({'x': [0, 1], 'y': [0, 1]}, sample_surface=True)
    pt_1 = LabelTensor(torch.tensor([[0.5, 0.5]]), ['x', 'y'])
    pt_2 = LabelTensor(torch.tensor([[1.0, 0.5]]), ['x', 'y'])
    pt_3 = LabelTensor(torch.tensor([[1.5, 0.5]]), ['x', 'y'])
    for pt, exp_result in zip([pt_1, pt_2, pt_3], [False, True, False]):
        assert domain.is_inside(pt) == exp_result
