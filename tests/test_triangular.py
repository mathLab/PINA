import torch
import pytest

from pina import LabelTensor
from pina.geometry import TriangleDomain


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
domain2D = TriangleDomain({'vertex1': [0, 0], 'vertex2': [1, 1], 'vertex3': [2, 0]})
domain3D = TriangleDomain({'vertex1': [1, 2, 3], 'vertex2': [2, 2, 3], 'vertex3': [1, 3, 3], 'vertex4': [1, 2, 9]})


def test_constructor():
    TriangleDomain({'vertex1': [0, 0], 'vertex2': [1, 1], 'vertex3': [2, 0]})
    TriangleDomain({'vertex1': [1, 2, 3], 'vertex2': [2, 2, 3], 'vertex3': [1, 3, 3], 'vertex4': [1, 2, 9]})

def test_is_inside_boundary_2D():
    for pt, exp_result in zip(pts2D, [True, True, True, True, True, False, False, False]):
        assert domain2D.is_inside(point=pt, check_border=True) == exp_result, f"Expected domain2D.is_inside({pt.extract(['x', 'y'])}) to be {exp_result}, but got {not exp_result}"

def test_is_inside_no_boundary_2D():
    for pt, exp_result in zip(pts2D, [False, False, False, True, True, False, False, False]):
        assert domain2D.is_inside(point=pt, check_border=False) == exp_result, f"Expected domain2D.is_inside({pt.extract(['x', 'y'])}) to be {exp_result}, but got {not exp_result}"

def test_is_inside_boundary_3D():
    for pt, exp_result in zip(pts3D, [True, True, True, True, True, False, True, True, False, True]):
        assert domain3D.is_inside(point=pt, check_border=True) == exp_result, f"Expected domain3D.is_inside({pt.extract(['x', 'y', 'z'])}) to be {exp_result}, but got {not exp_result}"

def test_is_inside_no_boundary_3D():
    for pt, exp_result in zip(pts3D, [False, False, False, False, True, False, False, False, False, True]):
        assert domain3D.is_inside(point=pt, check_border=False) == exp_result, f"Expected domain3D.is_inside({pt.extract(['x', 'y', 'z'])}) to be {exp_result}, but got {not exp_result}"