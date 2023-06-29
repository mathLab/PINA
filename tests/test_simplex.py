import torch
import pytest

from pina import LabelTensor
from pina.geometry import SimplexDomain

<<<<<<< HEAD

=======
>>>>>>> 3e1902a (Simplex Domain)
def test_constructor():
    SimplexDomain(
        [
            LabelTensor(torch.tensor([[0, 0]]), labels=["x", "y"]),
            LabelTensor(torch.tensor([[1, 1]]), labels=["x", "y"]),
            LabelTensor(torch.tensor([[0, 2]]), labels=["x", "y"]),
        ]
    )
    SimplexDomain(
        [
            LabelTensor(torch.tensor([[0, 0]]), labels=["x", "y"]),
            LabelTensor(torch.tensor([[1, 1]]), labels=["x", "y"]),
            LabelTensor(torch.tensor([[0, 2]]), labels=["x", "y"]),
        ],
        sample_surface=True,
    )
    with pytest.raises(ValueError):
        SimplexDomain(
            [
                LabelTensor(torch.tensor([[0, 0]]), labels=["x", "y"]),
                LabelTensor(torch.tensor([[1, 1]]), labels=["x", "z"]),
                LabelTensor(torch.tensor([[0, 2]]), labels=["x", "y"]),
            ]
        )
        SimplexDomain(
            [
                LabelTensor(torch.tensor([[0, 0]]), labels=["x", "y"]),
                [1, 1],
                LabelTensor(torch.tensor([[0, 2]]), labels=["x", "y"]),
            ]
        )


def test_is_inside_2D_check_border_true():
<<<<<<< HEAD
    domain = SimplexDomain(
        [
            LabelTensor(torch.tensor([[0, 0]]), labels=["x", "y"]),
            LabelTensor(torch.tensor([[2, 2]]), labels=["x", "y"]),
            LabelTensor(torch.tensor([[4, 0]]), labels=["x", "y"]),
        ],
        sample_surface=True,
    )
    pt1 = LabelTensor(torch.tensor([[0, 0]]), ["x", "y"])
    pt2 = LabelTensor(torch.tensor([[2, 2]]), ["x", "y"])
    pt3 = LabelTensor(torch.tensor([[4, 0]]), ["x", "y"])
    pt4 = LabelTensor(torch.tensor([[3, 1]]), ["x", "y"])
    pt5 = LabelTensor(torch.tensor([[2, 1]]), ["x", "y"])
    pt6 = LabelTensor(torch.tensor([[100, 100]]), ["x", "y"])
=======
    domain = SimplexDomain({'vertex1': [0, 0], 'vertex2': [2, 2], 'vertex3': [4, 0]}, ['x', 'y'], sample_surface=True)
    pt1 = LabelTensor(torch.tensor([[0, 0]]), ['x', 'y']) # TODO: ZERO BUG
    pt2 = LabelTensor(torch.tensor([[2, 2]]), ['x', 'y'])
    pt3 = LabelTensor(torch.tensor([[4, 0]]), ['x', 'y'])
    pt4 = LabelTensor(torch.tensor([[3, 1]]), ['x', 'y'])
    pt5 = LabelTensor(torch.tensor([[2, 1]]), ['x', 'y'])
    pt6 = LabelTensor(torch.tensor([[100, 100]]), ['x', 'y'])
>>>>>>> 3e1902a (Simplex Domain)
    pts = [pt1, pt2, pt3, pt4, pt5, pt6]
    for pt, exp_result in zip(pts, [True, True, True, True, False, False]):
        assert domain.is_inside(point=pt, check_border=True) == exp_result


def test_is_inside_2D_check_border_false():
<<<<<<< HEAD
    domain = SimplexDomain(
        [
            LabelTensor(torch.tensor([[0, 0]]), labels=["x", "y"]),
            LabelTensor(torch.tensor([[2, 2]]), labels=["x", "y"]),
            LabelTensor(torch.tensor([[4, 0]]), labels=["x", "y"]),
        ],
        sample_surface=False,
    )
    pt1 = LabelTensor(torch.tensor([[0, 0]]), ["x", "y"])
    pt2 = LabelTensor(torch.tensor([[2, 2]]), ["x", "y"])
    pt3 = LabelTensor(torch.tensor([[4, 0]]), ["x", "y"])
    pt4 = LabelTensor(torch.tensor([[3, 1]]), ["x", "y"])
    pt5 = LabelTensor(torch.tensor([[2, 1]]), ["x", "y"])
    pt6 = LabelTensor(torch.tensor([[2.5, 1]]), ["x", "y"])
    pt7 = LabelTensor(torch.tensor([[100, 100]]), ["x", "y"])
=======
    domain = SimplexDomain({'vertex1': [0, 0], 'vertex2': [2, 2], 'vertex3': [4, 0]}, ['x', 'y'], sample_surface=True)
    pt1 = LabelTensor(torch.tensor([[0, 0]]), ['x', 'y']) # TODO: ZERO BUG
    pt2 = LabelTensor(torch.tensor([[2, 2]]), ['x', 'y'])
    pt3 = LabelTensor(torch.tensor([[4, 0]]), ['x', 'y'])
    pt4 = LabelTensor(torch.tensor([[3, 1]]), ['x', 'y'])
    pt5 = LabelTensor(torch.tensor([[2, 1]]), ['x', 'y'])
    pt6 = LabelTensor(torch.tensor([[2.5, 1]]), ['x', 'y'])
    pt7 = LabelTensor(torch.tensor([[100, 100]]), ['x', 'y'])
>>>>>>> 3e1902a (Simplex Domain)
    pts = [pt1, pt2, pt3, pt4, pt5, pt6, pt7]
    for pt, exp_result in zip(pts, [False, False, False, False, True, True, False]):
        assert domain.is_inside(point=pt, check_border=False) == exp_result


def test_is_inside_3D_check_border_true():
<<<<<<< HEAD
    domain = SimplexDomain(
        [
            LabelTensor(torch.tensor([[0, 0, 0]]), labels=["x", "y", "z"]),
            LabelTensor(torch.tensor([[2, 2, 0]]), labels=["x", "y", "z"]),
            LabelTensor(torch.tensor([[4, 0, 0]]), labels=["x", "y", "z"]),
            LabelTensor(torch.tensor([[0, 0, 20]]), labels=["x", "y", "z"]),
        ],
        sample_surface=True,
    )
    pt1 = LabelTensor(torch.tensor([[0, 0, 0]]), ["x", "y", "z"])
    pt2 = LabelTensor(torch.tensor([[2, 2, 0]]), ["x", "y", "z"])
    pt3 = LabelTensor(torch.tensor([[4, 0, 0]]), ["x", "y", "z"])
    pt4 = LabelTensor(torch.tensor([[3, 1, 0]]), ["x", "y", "z"])
    pt5 = LabelTensor(torch.tensor([[2, 1, 0]]), ["x", "y", "z"])
    pt6 = LabelTensor(torch.tensor([[100, 100, 0]]), ["x", "y", "z"])
    pt7 = LabelTensor(torch.tensor([[0, 0, 19]]), ["x", "y", "z"])
    pt8 = LabelTensor(torch.tensor([[0, 0, 20]]), ["x", "y", "z"])
    pt9 = LabelTensor(torch.tensor([[2, 1, 1]]), ["x", "y", "z"])
    pts = [pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8, pt9]
    for pt, exp_result in zip(
        pts, [True, True, True, True, True, False, True, True, False]
    ):
=======
    domain = SimplexDomain({'vertex1': [0, 0, 0], 'vertex2': [2, 2, 0], 'vertex3': [4, 0, 0], 'vertex4': [0, 0, 20]}, ['x', 'y', 'z'], sample_surface=True)
    pt1 = LabelTensor(torch.tensor([[0, 0, 0]]), ['x', 'y', 'z']) # TODO: ZERO BUG
    pt2 = LabelTensor(torch.tensor([[2, 2, 0]]), ['x', 'y', 'z'])
    pt3 = LabelTensor(torch.tensor([[4, 0, 0]]), ['x', 'y', 'z'])
    pt4 = LabelTensor(torch.tensor([[3, 1, 0]]), ['x', 'y', 'z'])
    pt5 = LabelTensor(torch.tensor([[2, 1, 0]]), ['x', 'y', 'z'])
    pt6 = LabelTensor(torch.tensor([[100, 100, 0]]), ['x', 'y', 'z'])
    pt7 = LabelTensor(torch.tensor([[0, 0, 19]]), ['x', 'y', 'z'])
    pt8 = LabelTensor(torch.tensor([[0, 0, 20]]), ['x', 'y', 'z'])
    pt9 = LabelTensor(torch.tensor([[2, 1, 1]]), ['x', 'y', 'z'])
    pts = [pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8, pt9]
    for pt, exp_result in zip(pts, [True, True, True, True, True, False, True, True, False]):
>>>>>>> 3e1902a (Simplex Domain)
        assert domain.is_inside(point=pt, check_border=True) == exp_result


def test_is_inside_3D_check_border_false():
<<<<<<< HEAD
    domain = SimplexDomain(
        [
            LabelTensor(torch.tensor([[0, 0, 0]]), labels=["x", "y", "z"]),
            LabelTensor(torch.tensor([[2, 2, 0]]), labels=["x", "y", "z"]),
            LabelTensor(torch.tensor([[4, 0, 0]]), labels=["x", "y", "z"]),
            LabelTensor(torch.tensor([[0, 0, 20]]), labels=["x", "y", "z"]),
        ],
        sample_surface=False,
    )
    pt1 = LabelTensor(torch.tensor([[0, 0, 0]]), ["x", "y", "z"])
    pt2 = LabelTensor(torch.tensor([[3, 1, 0]]), ["x", "y", "z"])
    pt3 = LabelTensor(torch.tensor([[2, 1, 0]]), ["x", "y", "z"])
    pt4 = LabelTensor(torch.tensor([[100, 100, 0]]), ["x", "y", "z"])
    pt5 = LabelTensor(torch.tensor([[0, 0, 19]]), ["x", "y", "z"])
    pt6 = LabelTensor(torch.tensor([[0, 0, 20]]), ["x", "y", "z"])
    pt7 = LabelTensor(torch.tensor([[2, 1, 1]]), ["x", "y", "z"])
    pts = [pt1, pt2, pt3, pt4, pt5, pt6, pt7]
    for pt, exp_result in zip(pts, [False, False, False, False, False, False, True]):
        assert domain.is_inside(point=pt, check_border=False) == exp_result
=======
    domain = SimplexDomain({'vertex1': [0, 0, 0], 'vertex2': [2, 2, 0], 'vertex3': [4, 0, 0], 'vertex4': [0, 0, 20]}, ['x', 'y', 'z'], sample_surface=True)
    pt1 = LabelTensor(torch.tensor([[0, 0, 0]]), ['x', 'y', 'z']) # TODO: ZERO BUG
    pt2 = LabelTensor(torch.tensor([[2, 2, 0]]), ['x', 'y', 'z'])
    pt3 = LabelTensor(torch.tensor([[4, 0, 0]]), ['x', 'y', 'z'])
    pt4 = LabelTensor(torch.tensor([[3, 1, 0]]), ['x', 'y', 'z'])
    pt5 = LabelTensor(torch.tensor([[2, 1, 0]]), ['x', 'y', 'z'])
    pt6 = LabelTensor(torch.tensor([[100, 100, 0]]), ['x', 'y', 'z'])
    pt7 = LabelTensor(torch.tensor([[0, 0, 19]]), ['x', 'y', 'z'])
    pt8 = LabelTensor(torch.tensor([[0, 0, 20]]), ['x', 'y', 'z'])
    pt9 = LabelTensor(torch.tensor([[2, 1, 1]]), ['x', 'y', 'z'])
    pts = [pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8, pt9]
    for pt, exp_result in zip(pts, [False, False, False, False, False, False, False, False, True]):
        assert domain.is_inside(point=pt, check_border=False) == exp_result
>>>>>>> 3e1902a (Simplex Domain)
