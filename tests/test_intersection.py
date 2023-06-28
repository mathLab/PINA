import torch

from pina import LabelTensor
from pina.geometry import Intersection, EllipsoidDomain, CartesianDomain


def test_constructor_two_CartesianDomains():
    Intersection([CartesianDomain({'x': [0, 2], 'y': [0, 2]}),
                  CartesianDomain({'x': [1, 3], 'y': [1, 3]})])


def test_constructor_two_3DCartesianDomain():
    Intersection([CartesianDomain({'x': [0, 2], 'y': [0, 2], 'z': [0, 2]}),
                  CartesianDomain({'x': [1, 3], 'y': [1, 3], 'z': [1, 3]})])


def test_constructor_three_CartesianDomains():
    Intersection([CartesianDomain({'x': [0, 2], 'y': [0, 2]}), CartesianDomain(
        {'x': [1, 3], 'y': [1, 3]}), CartesianDomain({'x': [2, 4], 'y': [2, 4]})])


def test_is_inside_two_CartesianDomains():
    pt_1 = LabelTensor(torch.tensor([[0.5, 0.5]]), ['x', 'y'])
    pt_2 = LabelTensor(torch.tensor([[-1, -0.5]]), ['x', 'y'])
    pt_3 = LabelTensor(torch.tensor([[1.5, 1.5]]), ['x', 'y'])

    domain = Intersection([CartesianDomain({'x': [0, 2], 'y': [0, 2]}),
                           CartesianDomain({'x': [1, 3], 'y': [1, 3]})])
    assert domain.is_inside(pt_1) == False
    assert domain.is_inside(pt_2) == False
    assert domain.is_inside(pt_3) == True


def test_is_inside_two_3DCartesianDomain():
    pt_1 = LabelTensor(torch.tensor([[0.5, 0.5, 0.5]]), ['x', 'y', 'z'])
    pt_2 = LabelTensor(torch.tensor([[-1, -0.5, -0.5]]), ['x', 'y', 'z'])
    pt_3 = LabelTensor(torch.tensor([[1.5, 1.5, 1.5]]), ['x', 'y', 'z'])
    domain = Intersection([CartesianDomain({'x': [0, 2], 'y': [0, 2], 'z': [
        0, 2]}), CartesianDomain({'x': [1, 3], 'y': [1, 3], 'z': [1, 3]})])
    assert domain.is_inside(pt_1) == False
    assert domain.is_inside(pt_2) == False
    assert domain.is_inside(pt_3) == True


def test_sample():
    n = 100
    domain = Intersection([EllipsoidDomain(
        {'x': [-1, 1], 'y': [-1, 1]}), CartesianDomain({'x': [-0.5, 0.5], 'y': [-0.5, 0.5]})])
    pts = domain.sample(n, type="intersection")
    assert isinstance(pts, LabelTensor)
    assert pts.shape[0] == n

    n = 105
    pts = domain.sample(n, type="intersection")
    assert isinstance(pts, LabelTensor)
    assert pts.shape[0] == n
