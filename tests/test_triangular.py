import torch
import pytest

from pina import LabelTensor, Condition, TriangularDomain, PINN
from pina.problem import SpatialProblem
from pina.model import FeedForward
from pina.operators import nabla


def test_constructor():
    TriangularDomain({'vertex1': [0, 0], 'vertex2': [1, 1], 'vertex3': [2, 0]})


def test_is_inside_boundary():
    pt_1 = [0, 0]
    pt_2 = [1, 1]
    pt_3 = [2, 0]
    pt_4 = [1, .5]
    pt_5 = [1.2, .6]
    pt_6 = [0, 1]
    pt_7 = [1.01, 1]
    pt_8 = [2, 0.001]
    domain = TriangularDomain({'vertex1': [0, 0], 'vertex2': [1, 1], 'vertex3': [2, 0]})
    for pt, exp_result in zip([pt_1, pt_2, pt_3, pt_4, pt_5, pt_6, pt_7, pt_8], [True, True, True, True, True, False, False, False]):
        assert domain.is_inside(pt, True) == exp_result


def test_is_inside_no_boundary():
    pt_1 = [0, 0]
    pt_2 = [1, 1]
    pt_3 = [2, 0]
    pt_4 = [1, .5]
    pt_5 = [1.2, .6]
    pt_6 = [0, 1]
    pt_7 = [1.01, 1]
    pt_8 = [2, 0.001]
    domain = TriangularDomain({'vertex1': [0, 0], 'vertex2': [1, 1], 'vertex3': [2, 0]})
    for pt, exp_result in zip([pt_1, pt_2, pt_3, pt_4, pt_5, pt_6, pt_7, pt_8], [False, False, False, True, True, False, False, False]):
        assert domain.is_inside(pt, False) == exp_result