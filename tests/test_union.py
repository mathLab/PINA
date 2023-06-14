import torch
import pytest

from pina import LabelTensor, Union, EllipsoidDomain, CartesianDomain


def test_constructor_two_CartesianDomains():
    Union([CartesianDomain({'x': [0, 1], 'y': [0, 1]}),
           CartesianDomain({'x': [0.5, 2], 'y': [-1, 0.1]})])


def test_constructor_two_EllipsoidDomains():
    Union([EllipsoidDomain({'x': [-1, 1], 'y': [-1, 1], 'z': [-1, 1]}),
           EllipsoidDomain({'x': [-0.5, 0.5], 'y': [-0.5, 0.5], 'z': [-0.5, 0.5]})])


def test_constructor_EllipsoidDomain_CartesianDomain():
    Union([EllipsoidDomain({'x': [-1, 1], 'y': [-1, 1]}),
           CartesianDomain({'x': [-0.5, 0.5], 'y': [-0.5, 0.5]})])


def test_is_inside_two_CartesianDomains():
    pt_1 = LabelTensor(torch.tensor([[0.5, 0.5]]), ['x', 'y'])
    pt_2 = LabelTensor(torch.tensor([[-1, -1]]), ['x', 'y'])
    domain = Union([CartesianDomain({'x': [0, 1], 'y': [0, 1]}),
                    CartesianDomain({'x': [0.5, 2], 'y': [-1, 0.1]})])
    assert domain.is_inside(pt_1) == True
    assert domain.is_inside(pt_2) == False


def test_is_inside_two_EllipsoidDomains():
    pt_1 = LabelTensor(torch.tensor([[0.5, 0.5, 0.5]]), ['x', 'y', 'z'])
    pt_2 = LabelTensor(torch.tensor([[-1, -1, -1]]), ['x', 'y', 'z'])
    domain = Union([EllipsoidDomain({'x': [-1, 1], 'y': [-1, 1], 'z': [-1, 1]}),
                    EllipsoidDomain({'x': [-0.5, 0.5], 'y': [-0.5, 0.5], 'z': [-0.5, 0.5]})])
    assert domain.is_inside(pt_1) == True
    assert domain.is_inside(pt_2) == False


def test_is_inside_EllipsoidDomain_CartesianDomain():
    pt_1 = LabelTensor(torch.tensor([[0.5, 0.5]]), ['x', 'y'])
    pt_2 = LabelTensor(torch.tensor([[-1, -1]]), ['x', 'y'])
    domain = Union([EllipsoidDomain({'x': [-1, 1], 'y': [-1, 1], }),
                    CartesianDomain({'x': [0.6, 1.5], 'y': [-2, 0]})])
    assert domain.is_inside(pt_1) == True
    assert domain.is_inside(pt_2) == False
