import torch

from pina import LabelTensor
from pina.geometry import Exclusion, EllipsoidDomain, CartesianDomain


def test_constructor_two_CartesianDomains():
    Exclusion([
        CartesianDomain({
            'x': [0, 2],
            'y': [0, 2]
        }),
        CartesianDomain({
            'x': [1, 3],
            'y': [1, 3]
        })
    ])


def test_constructor_two_3DCartesianDomain():
    Exclusion([
        CartesianDomain({
            'x': [0, 2],
            'y': [0, 2],
            'z': [0, 2]
        }),
        CartesianDomain({
            'x': [1, 3],
            'y': [1, 3],
            'z': [1, 3]
        })
    ])


def test_constructor_three_CartesianDomains():
    Exclusion([
        CartesianDomain({
            'x': [0, 2],
            'y': [0, 2]
        }),
        CartesianDomain({
            'x': [1, 3],
            'y': [1, 3]
        }),
        CartesianDomain({
            'x': [2, 4],
            'y': [2, 4]
        })
    ])


def test_is_inside_two_CartesianDomains():
    pt_1 = LabelTensor(torch.tensor([[0.5, 0.5]]), ['x', 'y'])
    pt_2 = LabelTensor(torch.tensor([[-1, -0.5]]), ['x', 'y'])
    domain = Exclusion([
        CartesianDomain({
            'x': [0, 2],
            'y': [0, 2]
        }),
        CartesianDomain({
            'x': [1, 3],
            'y': [1, 3]
        })
    ])
    assert domain.is_inside(pt_1) == True
    assert domain.is_inside(pt_2) == False


def test_is_inside_two_3DCartesianDomain():
    pt_1 = LabelTensor(torch.tensor([[0.5, 0.5, 0.5]]), ['x', 'y', 'z'])
    pt_2 = LabelTensor(torch.tensor([[-1, -0.5, -0.5]]), ['x', 'y', 'z'])
    domain = Exclusion([
        CartesianDomain({
            'x': [0, 2],
            'y': [0, 2],
            'z': [0, 2]
        }),
        CartesianDomain({
            'x': [1, 3],
            'y': [1, 3],
            'z': [1, 3]
        })
    ])
    assert domain.is_inside(pt_1) == True
    assert domain.is_inside(pt_2) == False


def test_sample():
    n = 100
    domain = Exclusion([
        EllipsoidDomain({
            'x': [-1, 1],
            'y': [-1, 1]
        }),
        CartesianDomain({
            'x': [0.3, 1.5],
            'y': [0.3, 1.5]
        })
    ])
    pts = domain.sample(n)
    assert isinstance(pts, LabelTensor)
    assert pts.shape[0] == n

    n = 105
    pts = domain.sample(n)
    assert isinstance(pts, LabelTensor)
    assert pts.shape[0] == n
