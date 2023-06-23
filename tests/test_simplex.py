import torch
import pytest

from pina import LabelTensor
from pina.geometry import SimplexDomain


pt2D_1 = LabelTensor(torch.tensor([[0, 0]]), ['x', 'y'])
pt2D_2 = LabelTensor(torch.tensor([[1, 1]]), ['x', 'y'])
pt2D_3 = LabelTensor(torch.tensor([[2, 0]]), ['x', 'y'])
pt2D_4 = LabelTensor(torch.tensor([[1, .5]]), ['x', 'y'])
pt2D_5 = LabelTensor(torch.tensor([[1.2, .6]]), ['x', 'y'])
pt2D_6 = LabelTensor(torch.tensor([[0, 1]]), ['x', 'y'])
pt2D_7 = LabelTensor(torch.tensor([[1.01, 1]]), ['x', 'y'])
pt2D_8 = LabelTensor(torch.tensor([[2, 0.001]]), ['x', 'y'])
pt3D_1 = LabelTensor(torch.tensor([[1, 2, 3]]), ['x', 'y', 'z'])
pt3D_2 = LabelTensor(torch.tensor([[2, 2, 3]]), ['x', 'y', 'z'])
pt3D_3 = LabelTensor(torch.tensor([[1, 3, 3]]), ['x', 'y', 'z'])
pt3D_4 = LabelTensor(torch.tensor([[1, 2, 9]]), ['x', 'y', 'z'])
pt3D_5 = LabelTensor(torch.tensor([[1.25, 2.25, 4.5]]), ['x', 'y', 'z'])
pt3D_6 = LabelTensor(torch.tensor([[100, 100, 100]]), ['x', 'y', 'z'])
pt3D_7 = LabelTensor(torch.tensor([[1, 3, 3]]), ['x', 'y', 'z'])
pt3D_8 = LabelTensor(torch.tensor([[1, 2, 7.8]]), ['x', 'y', 'z'])
pt3D_9 = LabelTensor(torch.tensor([[0.8, 2.4, 6.6]]), ['x', 'y', 'z'])
pt3D_10 = LabelTensor(torch.tensor([[1, 2.5, 4]]), ['x', 'y', 'z'])
pts2D = [pt2D_1, pt2D_2, pt2D_3, pt2D_4, pt2D_5, pt2D_6, pt2D_7, pt2D_8]
pts3D = [pt3D_1, pt3D_2, pt3D_3, pt3D_4, pt3D_5, pt3D_6, pt3D_7, pt3D_8, pt3D_9, pt3D_10]
domain2D = SimplexDomain({'vertex1': [0, 0], 'vertex2': [1, 1], 'vertex3': [2, 0]}, labels=['x', 'y'])
domain3D = SimplexDomain({'vertex1': [1, 2, 3], 'vertex2': [2, 2, 3], 'vertex3': [1, 3, 3], 'vertex4': [1, 2, 9]}, labels=['x', 'y', 'z'])


def test_constructor():
    SimplexDomain({'vertex1': [0, 0], 'vertex2': [1, 1], 'vertex3': [2, 0]}, labels=['x', 'y'])
    SimplexDomain({'vertex1': [1, 2, 3], 'vertex2': [2, 2, 3], 'vertex3': [1, 3, 3], 'vertex4': [1, 2, 9]}, labels=['x', 'y', 'z'])

def test_is_inside_2D_check_border_true():
    domain = SimplexDomain({'vertex1': [0, 0], 'vertex2': [2, 2], 'vertex3': [4, 0]}, ['x', 'y'], sample_surface=True)
    #pt1 = LabelTensor(torch.tensor([[0, 0]]), ['x', 'y']) # TODO: ZERO BUG
    pt2 = LabelTensor(torch.tensor([[2, 2]]), ['x', 'y'])
    pt3 = LabelTensor(torch.tensor([[4, 0]]), ['x', 'y'])
    pt4 = LabelTensor(torch.tensor([[3, 1]]), ['x', 'y'])
    pt5 = LabelTensor(torch.tensor([[2, 1]]), ['x', 'y'])
    pt6 = LabelTensor(torch.tensor([[100, 100]]), ['x', 'y'])
    pts = [pt2, pt3, pt4, pt5, pt6]
    for pt, exp_result in zip(pts, [True, True, True, False, False]):
        assert domain.is_inside(point=pt, check_border=True) == exp_result

def test_is_inside_2D_check_border_false():
    domain = SimplexDomain({'vertex1': [0, 0], 'vertex2': [2, 2], 'vertex3': [4, 0]}, ['x', 'y'], sample_surface=True)
    # pt1 = LabelTensor(torch.tensor([[0, 0]]), ['x', 'y']) # TODO: ZERO BUG
    pt2 = LabelTensor(torch.tensor([[2, 2]]), ['x', 'y'])
    pt3 = LabelTensor(torch.tensor([[4, 0]]), ['x', 'y'])
    pt4 = LabelTensor(torch.tensor([[3, 1]]), ['x', 'y'])
    pt5 = LabelTensor(torch.tensor([[2, 1]]), ['x', 'y'])
    pt6 = LabelTensor(torch.tensor([[2.5, 1]]), ['x', 'y'])
    pt7 = LabelTensor(torch.tensor([[100, 100]]), ['x', 'y'])
    pts = [pt2, pt3, pt4, pt5, pt6, pt7]
    for pt, exp_result in zip(pts, [False, False, False, True, True, False]):
        assert domain.is_inside(point=pt, check_border=False) == exp_result

def test_is_inside_3D_check_border_true():
    domain = SimplexDomain({'vertex1': [0, 0, 0], 'vertex2': [2, 2, 0], 'vertex3': [4, 0, 0], 'vertex4': [0, 0, 20]}, ['x', 'y', 'z'], sample_surface=True)
    #pt1 = LabelTensor(torch.tensor([[0, 0, 0]]), ['x', 'y']) # TODO: ZERO BUG
    pt2 = LabelTensor(torch.tensor([[2, 2, 0]]), ['x', 'y', 'z'])
    pt3 = LabelTensor(torch.tensor([[4, 0, 0]]), ['x', 'y', 'z'])
    pt4 = LabelTensor(torch.tensor([[3, 1, 0]]), ['x', 'y', 'z'])
    pt5 = LabelTensor(torch.tensor([[2, 1, 0]]), ['x', 'y', 'z'])
    pt6 = LabelTensor(torch.tensor([[100, 100, 0]]), ['x', 'y', 'z'])
    pt7 = LabelTensor(torch.tensor([[0, 0, 19]]), ['x', 'y', 'z'])
    pt8 = LabelTensor(torch.tensor([[0, 0, 20]]), ['x', 'y', 'z'])
    pts = [pt2, pt3, pt4, pt5, pt6, pt7, pt8]
    for pt, exp_result in zip(pts, [True, True, True, False, False, True, True]):
        assert domain.is_inside(point=pt, check_border=True) == exp_result

def test_is_inside_3D_check_border_false():
    domain = SimplexDomain({'vertex1': [0, 0], 'vertex2': [2,2], 'vertex3': [4,0]}, ['x', 'y'], sample_surface=True)
    # pt1 = LabelTensor(torch.tensor([[0, 0]]), ['x', 'y']) # TODO: ZERO BUG
    pt2 = LabelTensor(torch.tensor([[2, 2]]), ['x', 'y'])
    pt3 = LabelTensor(torch.tensor([[4, 0]]), ['x', 'y'])
    pt4 = LabelTensor(torch.tensor([[3, 1]]), ['x', 'y'])
    pt5 = LabelTensor(torch.tensor([[2, 1]]), ['x', 'y'])
    pt6 = LabelTensor(torch.tensor([[2.5, 1]]), ['x', 'y'])
    pt7 = LabelTensor(torch.tensor([[100, 100]]), ['x', 'y'])
    pts = [pt2, pt3, pt4, pt5, pt6, pt7]
    for pt, exp_result in zip(pts, [False, False, False, True, True, False]):
        assert domain.is_inside(point=pt, check_border=False) == exp_result