import torch

from pina import LabelTensor
from pina.domain import CartesianDomain

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

def test_sampler_all_range():
    domain = CartesianDomain({'x': [2, 3], 'y': [0, 1]})
    sampled_points = domain.sample(n=100)
    assert sampled_points.shape == (100, 2)
    assert torch.max(sampled_points[:,0]) <=3
    assert torch.max(sampled_points[:,0]) >= 2
    assert torch.max(sampled_points[:,1]) <= 1
    assert torch.max(sampled_points[:,1]) >= 0
    sampled_points = domain.sample(n=100, mode='grid')
    assert sampled_points.shape == (100, 2)
    assert torch.max(sampled_points[:,0]) <=3
    assert torch.max(sampled_points[:,0]) >= 2
    assert torch.max(sampled_points[:,1]) <= 1
    assert torch.max(sampled_points[:,1]) >= 0

def test_sampler_range_fixed():
    domain = CartesianDomain({'x': [2, 3], 'y': 1})
    sampled_points = domain.sample(n=100)
    assert sampled_points.shape == (100, 2)
    assert torch.max(sampled_points[:, 0]) <= 3
    assert torch.max(sampled_points[:, 0]) >= 2
    assert torch.eq(sampled_points[:, 1].tensor, torch.ones(100)).all()
    sampled_points = domain.sample(n=100, mode='grid')
    assert sampled_points.shape == (100, 2)
    assert torch.max(sampled_points[:, 0]) <= 3
    assert torch.max(sampled_points[:, 0]) >= 2
    assert torch.eq(sampled_points[:, 1].tensor, torch.ones(100)).all()

def test_sampler_fixed():
    domain = CartesianDomain({'x': 2, 'y': 1})
    sampled_points = domain.sample(n=100)
    assert sampled_points.shape == (100, 2)
    assert torch.eq(sampled_points[:, 1].tensor, torch.ones(100)).all()
    assert torch.eq(sampled_points[:, 0].tensor, torch.ones(100)*2).all()