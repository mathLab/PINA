import torch
import pytest

from pina.problem.zoo import Poisson2DSquareProblem

def test_constructor():
    Poisson2DSquareProblem()