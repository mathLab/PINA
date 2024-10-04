import torch
import pytest

from pina import LabelTensor
from pina.operators import grad, div, laplacian


def func_vector(x):
    return x**2


def func_scalar(x):
    x_ = x.extract(['x'])
    y_ = x.extract(['y'])
    z_ = x.extract(['z'])
    return x_**2 + y_**2 + z_**2


data = torch.rand((20, 3))
inp = LabelTensor(data, ['x', 'y', 'mu']).requires_grad_(True)
labels = ['a', 'b', 'c']
tensor_v = LabelTensor(func_vec(inp), labels)
tensor_s = LabelTensor(func_scalar(inp).reshape(-1, 1), labels[0])


def test_grad_scalar_output():
    grad_tensor_s = grad(tensor_s, inp)
    true_val = 2*inp
    assert grad_tensor_s.shape == inp.shape
    assert grad_tensor_s.labels[grad_tensor_s.ndim-1]['dof'] == [
        f'd{tensor_s.labels[tensor_s.ndim-1]["dof"][0]}d{i}' for i in inp.labels[inp.ndim-1]['dof']
    ]
    assert torch.allclose(grad_tensor_s, true_val)

    grad_tensor_s = grad(tensor_s, inp, d=['x', 'y'])
    assert grad_tensor_s.shape == (20, 2)
    assert grad_tensor_s.labels[grad_tensor_s.ndim-1]['dof'] == [
        f'd{tensor_s.labels[tensor_s.ndim-1]["dof"][0]}d{i}' for i in ['x', 'y']
    ]
    assert torch.allclose(grad_tensor_s, true_val)

def test_grad_vector_output():
    grad_tensor_v = grad(tensor_v, inp)
    true_val = torch.cat(
        (2*inp.extract(['x']),
         torch.zeros_like(inp.extract(['y'])),
         torch.zeros_like(inp.extract(['z'])),
         torch.zeros_like(inp.extract(['x'])),
         2*inp.extract(['y']),
         torch.zeros_like(inp.extract(['z'])),
         torch.zeros_like(inp.extract(['x'])),
         torch.zeros_like(inp.extract(['y'])),
         2*inp.extract(['z'])
        ), dim=1
    )
    assert grad_tensor_v.shape == (20, 9)
    assert grad_tensor_v.labels == [
        f'd{j}d{i}' for j in tensor_v.labels for i in inp.labels
    ]
    assert torch.allclose(grad_tensor_v, true_val)

    grad_tensor_v = grad(tensor_v, inp, d=['x', 'y'])
    true_val = torch.cat(
        (2*inp.extract(['x']),
         torch.zeros_like(inp.extract(['y'])),
         torch.zeros_like(inp.extract(['x'])),
         2*inp.extract(['y']),
         torch.zeros_like(inp.extract(['x'])),
         torch.zeros_like(inp.extract(['y']))
        ), dim=1
    )
    assert grad_tensor_v.shape == (inp.shape[0], 6)
    assert grad_tensor_v.labels == [
        f'd{j}d{i}' for j in tensor_v.labels for i in ['x', 'y']
    ]
    assert torch.allclose(grad_tensor_v, true_val)

def test_div_vector_output():
    div_tensor_v = div(tensor_v, inp)
    true_val = 2*torch.sum(inp, dim=1).reshape(-1,1)
    assert div_tensor_v.shape == (20, 1)
    assert div_tensor_v.labels == [f'dadx+dbdy+dcdz']
    assert torch.allclose(div_tensor_v, true_val)

    div_tensor_v = div(tensor_v, inp, components=['a', 'b'], d=['x', 'y'])
    true_val = 2*torch.sum(inp.extract(['x', 'y']), dim=1).reshape(-1,1)
    assert div_tensor_v.shape == (inp.shape[0], 1)
    assert div_tensor_v.labels == [f'dadx+dbdy']
    assert torch.allclose(div_tensor_v, true_val)

def test_laplacian_scalar_output():
    laplace_tensor_s = laplacian(tensor_s, inp)
    true_val = 6*torch.ones_like(laplace_tensor_s)
    assert laplace_tensor_s.shape == tensor_s.shape
    assert laplace_tensor_s.labels == [f"dd{tensor_s.labels[0]}"]
    assert torch.allclose(laplace_tensor_s, true_val)

    laplace_tensor_s = laplacian(tensor_s, inp, components=['a'], d=['x', 'y'])
    true_val = 4*torch.ones_like(laplace_tensor_s)
    assert laplace_tensor_s.shape == tensor_s.shape
    assert laplace_tensor_s.labels == [f"dd{tensor_s.labels[0]}"]
    assert torch.allclose(laplace_tensor_s, true_val)

def test_laplacian_vector_output():
    laplace_tensor_v = laplacian(tensor_v, inp)
    true_val = 2*torch.ones_like(tensor_v)
    assert laplace_tensor_v.shape == tensor_v.shape
    assert laplace_tensor_v.labels == [
        f'dd{i}' for i in tensor_v.labels
    ]
    assert torch.allclose(laplace_tensor_v, true_val)

    laplace_tensor_v = laplacian(tensor_v,
                                 inp,
                                 components=['a', 'b'],
                                 d=['x', 'y'])
    true_val = 2*torch.ones_like(tensor_v.extract(['a', 'b']))
    assert laplace_tensor_v.shape == tensor_v.extract(['a', 'b']).shape
    assert laplace_tensor_v.labels == [
        f'dd{i}' for i in ['a', 'b']
    ]
    assert torch.allclose(laplace_tensor_v, true_val)
