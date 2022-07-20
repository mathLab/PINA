import torch
import pytest

from pina import LabelTensor
from pina.operators import grad, div, nabla

def func_vec(x):
    return x**2

def func_scalar(x):
    return x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**3

data = torch.rand((20, 3), requires_grad=True)
inp = LabelTensor(data, ['x', 'y', 'mu'])
labels = ['a', 'b', 'c']
tensor_v = LabelTensor(func_vec(inp), labels)
tensor_s = LabelTensor(func_scalar(inp).reshape(-1, 1), labels[0])


def test_grad_scalar_output():
    grad_tensor_s = grad(tensor_s, inp)
    assert grad_tensor_s.shape == inp.shape
    assert grad_tensor_s.labels == [f'd{tensor_s.labels[0]}d{i}' for i in inp.labels]
    grad_tensor_s = grad(tensor_s, inp, d=['x', 'y'])
    assert grad_tensor_s.shape == (inp.shape[0], 2)
    assert grad_tensor_s.labels == [f'd{tensor_s.labels[0]}d{i}' for i in ['x', 'y']]

def test_grad_vector_output():
    grad_tensor_v = grad(tensor_v, inp)
    assert grad_tensor_v.shape == (20, 9)
    grad_tensor_v = grad(tensor_v, inp, d=['x', 'mu'])
    assert grad_tensor_v.shape == (inp.shape[0], 6)

def test_div_vector_output():
    grad_tensor_v = div(tensor_v, inp)
    assert grad_tensor_v.shape == (20, 1)
    grad_tensor_v = div(tensor_v, inp, components=['a', 'b'], d=['x', 'mu'])
    assert grad_tensor_v.shape == (inp.shape[0], 1)

def test_nabla_scalar_output():
    laplace_tensor_v = nabla(tensor_s, inp, components=['a'], d=['x', 'y'])
    assert laplace_tensor_v.shape == tensor_s.shape

def test_nabla_vector_output():
    laplace_tensor_v = nabla(tensor_v, inp)
    assert laplace_tensor_v.shape == tensor_v.shape
    laplace_tensor_v = nabla(tensor_v, inp, components=['a', 'b'], d=['x', 'y'])
    assert laplace_tensor_v.shape == tensor_v.extract(['a', 'b']).shape

