import pytest
import torch
from pina import Graph


def test_build_1():
    x = [torch.rand(10, 2) for _ in range(3)]
    pos = [torch.rand(10, 3) for _ in range(3)]
    graph = Graph(x=x, pos=pos, method='radius', r=.3)
    assert len(graph.data) == 3

def test_build_2():
    x = torch.rand(10, 2)
    pos = torch.rand(10, 3)
    graph = Graph(x=x, pos=pos, method='radius', r=.3)
    assert len(graph.data) == 1

def test_build_3():
    x = torch.rand(10, 2)
    pos = [torch.rand(10, 3) for _ in range(3)]
    graph = Graph(x=x, pos=pos, method='radius', r=.3)
    assert len(graph.data) == 3
