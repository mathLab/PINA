from pina.equation import Equation
from pina.operators import grad, laplacian
from pina import LabelTensor
import torch
import pytest

def eq1(input_, output_):
    u_grad = grad(output_, input_)
    u1_xx = grad(u_grad, input_, components=['du1dx'], d=['x'])
    u2_xy = grad(u_grad, input_, components=['du2dx'], d=['y'])
    return torch.hstack([u1_xx , u2_xy])  

def eq2(input_, output_):
    force_term = (torch.sin(input_.extract(['x'])*torch.pi) *
                    torch.sin(input_.extract(['y'])*torch.pi))
    delta_u = laplacian(output_.extract(['u1']), input_)
    return delta_u - force_term

def foo():
    pass

def test_constructor():
    Equation(eq1)
    Equation(eq2)
    with pytest.raises(ValueError):
        Equation([1, 2, 4])
    with pytest.raises(ValueError):
        Equation(foo())

def test_residual():
    eq_1 = Equation(eq1)
    eq_2 = Equation(eq2)

    pts = LabelTensor(torch.rand(10, 2), labels=['x', 'y'])
    pts.requires_grad = True
    u = torch.pow(pts, 2)
    u.labels = ['u1', 'u2']

    eq_1_res = eq_1.residual(pts, u)
    eq_2_res = eq_2.residual(pts, u)

    assert eq_1_res.shape == torch.Size([10, 2])
    assert eq_2_res.shape == torch.Size([10, 1])
