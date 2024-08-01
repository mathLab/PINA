from pina.equation import SystemEquation
from pina.operators import grad, laplacian
from pina import LabelTensor
import torch
import pytest


def eq1(input_, output_):
    u_grad = grad(output_, input_)
    u1_xx = grad(u_grad, input_, components=['du1dx'], d=['x'])
    u2_xy = grad(u_grad, input_, components=['du2dx'], d=['y'])
    return torch.hstack([u1_xx, u2_xy])


def eq2(input_, output_):
    force_term = (torch.sin(input_.extract(['x']) * torch.pi) *
                  torch.sin(input_.extract(['y']) * torch.pi))
    delta_u = laplacian(output_.extract(['u1']), input_)
    return delta_u - force_term


def foo():
    pass


def test_constructor():
    SystemEquation([eq1, eq2])
    SystemEquation([eq1, eq2], reduction='sum')
    with pytest.raises(NotImplementedError):
        SystemEquation([eq1, eq2], reduction='foo')
    with pytest.raises(ValueError):
        SystemEquation(foo)


def test_residual():

    pts = LabelTensor(torch.rand(10, 2), labels=['x', 'y'])
    pts.requires_grad = True
    u = torch.pow(pts, 2)
    u.labels = ['u1', 'u2']

    eq_1 = SystemEquation([eq1, eq2], reduction='mean')
    res = eq_1.residual(pts, u)
    assert res.shape == torch.Size([10])

    eq_1 = SystemEquation([eq1, eq2], reduction='sum')
    res = eq_1.residual(pts, u)
    assert res.shape == torch.Size([10])

    eq_1 = SystemEquation([eq1, eq2], reduction=None)
    res = eq_1.residual(pts, u)
    assert res.shape == torch.Size([10, 3])

    eq_1 = SystemEquation([eq1, eq2])
    res = eq_1.residual(pts, u)
    assert res.shape == torch.Size([10, 3])
