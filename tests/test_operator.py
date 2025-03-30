import torch
import pytest

from pina import LabelTensor
from pina.operator import grad, div, laplacian, advection


def func_vector(x):
    return x**2


def func_scalar(x):
    x_ = x.extract(["x"])
    y_ = x.extract(["y"])
    z_ = x.extract(["z"])
    return x_**2 + y_**2 + z_**2


n = 20
data = torch.rand((n, 3), requires_grad=True)
inp = LabelTensor(data, ["x", "y", "z"])
labels = ["a", "b", "c"]
tensor_v = LabelTensor(func_vector(inp), labels)
tensor_s = LabelTensor(func_scalar(inp).reshape(-1, 1), labels[0])


def test_grad_scalar_output():
    grad_tensor_s = grad(output_=tensor_s, input_=inp)
    true_val = 2 * inp
    true_val.labels = inp.labels
    assert grad_tensor_s.shape == inp.shape
    assert grad_tensor_s.labels == [
        f"d{tensor_s.labels[0]}d{i}" for i in inp.labels
    ]
    assert torch.allclose(grad_tensor_s, true_val)

    grad_tensor_s = grad(output_=tensor_s, input_=inp, d=["x", "y"])
    assert grad_tensor_s.shape == (n, 2)
    assert grad_tensor_s.labels == [
        f"d{tensor_s.labels[0]}d{i}" for i in ["x", "y"]
    ]
    assert torch.allclose(grad_tensor_s, true_val.extract(["x", "y"]))


def test_grad_vector_output():
    grad_tensor_v = grad(output_=tensor_v, input_=inp)
    true_val = torch.cat(
        (
            2 * inp.extract(["x"]),
            torch.zeros_like(inp.extract(["y"])),
            torch.zeros_like(inp.extract(["z"])),
            torch.zeros_like(inp.extract(["x"])),
            2 * inp.extract(["y"]),
            torch.zeros_like(inp.extract(["z"])),
            torch.zeros_like(inp.extract(["x"])),
            torch.zeros_like(inp.extract(["y"])),
            2 * inp.extract(["z"]),
        ),
        dim=1,
    )
    assert grad_tensor_v.shape == (n, 9)
    assert grad_tensor_v.labels == [
        f"d{j}d{i}" for j in tensor_v.labels for i in inp.labels
    ]
    assert torch.allclose(grad_tensor_v, true_val)

    grad_tensor_v = grad(output_=tensor_v, input_=inp, d=["x", "y"])
    true_val = torch.cat(
        (
            2 * inp.extract(["x"]),
            torch.zeros_like(inp.extract(["y"])),
            torch.zeros_like(inp.extract(["x"])),
            2 * inp.extract(["y"]),
            torch.zeros_like(inp.extract(["x"])),
            torch.zeros_like(inp.extract(["y"])),
        ),
        dim=1,
    )
    assert grad_tensor_v.shape == (inp.shape[0], 6)
    assert grad_tensor_v.labels == [
        f"d{j}d{i}" for j in tensor_v.labels for i in ["x", "y"]
    ]
    assert torch.allclose(grad_tensor_v, true_val)


def test_div_vector_output():
    div_tensor_v = div(output_=tensor_v, input_=inp)
    true_val = 2 * torch.sum(inp, dim=1).reshape(-1, 1)
    assert div_tensor_v.shape == (n, 1)
    assert div_tensor_v.labels == [f"dadx+dbdy+dcdz"]
    assert torch.allclose(div_tensor_v, true_val)

    div_tensor_v = div(
        output_=tensor_v, input_=inp, components=["a", "b"], d=["x", "y"]
    )
    true_val = 2 * torch.sum(inp.extract(["x", "y"]), dim=1).reshape(-1, 1)
    assert div_tensor_v.shape == (inp.shape[0], 1)
    assert div_tensor_v.labels == [f"dadx+dbdy"]
    assert torch.allclose(div_tensor_v, true_val)


@pytest.mark.parametrize("method", ["std", "divgrad"])
def test_laplacian_scalar_output(method):
    laplace_tensor_s = laplacian(output_=tensor_s, input_=inp, method=method)
    true_val = 6 * torch.ones_like(laplace_tensor_s)
    assert laplace_tensor_s.shape == tensor_s.shape
    assert laplace_tensor_s.labels == [f"dd{tensor_s.labels[0]}"]
    assert torch.allclose(laplace_tensor_s, true_val)

    laplace_tensor_s = laplacian(
        output_=tensor_s,
        input_=inp,
        components=["a"],
        d=["x", "y"],
        method=method,
    )
    true_val = 4 * torch.ones_like(laplace_tensor_s)
    assert laplace_tensor_s.shape == tensor_s.shape
    assert laplace_tensor_s.labels == [f"dd{tensor_s.labels[0]}"]
    assert torch.allclose(laplace_tensor_s, true_val)


@pytest.mark.parametrize("method", ["std", "divgrad"])
def test_laplacian_vector_output(method):
    laplace_tensor_v = laplacian(output_=tensor_v, input_=inp, method=method)
    true_val = 2 * torch.ones_like(tensor_v)
    assert laplace_tensor_v.shape == tensor_v.shape
    assert laplace_tensor_v.labels == [f"dd{i}" for i in tensor_v.labels]
    assert torch.allclose(laplace_tensor_v, true_val)

    laplace_tensor_v = laplacian(
        output_=tensor_v,
        input_=inp,
        components=["a", "b"],
        d=["x", "y"],
        method=method,
    )
    true_val = 2 * torch.ones_like(tensor_v.extract(["a", "b"]))
    assert laplace_tensor_v.shape == tensor_v.extract(["a", "b"]).shape
    assert laplace_tensor_v.labels == [f"dd{i}" for i in ["a", "b"]]
    assert torch.allclose(laplace_tensor_v, true_val)


@pytest.mark.parametrize("method", ["std", "divgrad"])
def test_laplacian_vector_output2(method):
    x = torch.linspace(0, 1, 10, requires_grad=True).reshape(-1, 1)
    y = torch.linspace(3, 4, 10, requires_grad=True).reshape(-1, 1)
    input_ = LabelTensor(torch.cat((x, y), dim=1), labels=["x", "y"])

    # Construct two scalar functions:
    # u = x**2 + y**2
    # v = x**2 - y**2
    u = input_.extract("x") ** 2 + input_.extract("y") ** 2
    v = input_.extract("x") ** 2 - input_.extract("y") ** 2

    # Define a vector-valued function, whose components are u and v.
    f = LabelTensor(torch.cat((u, v), dim=1), labels=["u", "v"])

    # Compute the scalar laplacian of both u and v:
    # Lap(u) = [4, 4, 4, ..., 4]
    # Lap(v) = [0, 0, 0, ..., 0]
    lap_u = laplacian(
        output_=f.extract("u"), input_=input_, components=["u"], method=method
    )
    lap_v = laplacian(
        output_=f.extract("v"), input_=input_, components=["v"], method=method
    )

    # Compute the laplacian of f: the two columns should correspond
    # to the laplacians of u and v, respectively...
    lap_f = laplacian(
        output_=f, input_=input_, components=["u", "v"], method=method
    )

    assert torch.allclose(lap_f.extract("ddu"), lap_u)
    assert torch.allclose(lap_f.extract("ddv"), lap_v)


def test_advection():
    # Advection term
    adv_tensor = advection(output_=tensor_v, input_=inp, velocity_field="c")

    # True value
    velocity = tensor_v.extract(["c"])
    true_val = velocity * 2 * inp.extract(["x", "y"])

    assert adv_tensor.shape == (tensor_v.shape[0], tensor_v.shape[1] - 1)
    assert torch.allclose(adv_tensor, true_val)


@pytest.mark.parametrize("method", ["std", "divgrad"])
def test_label_format(method):
    # Testing the format of `components` or `d` in case of single str of length
    # greater than 1; e.g.: "aaa".
    # This test is conducted only for gradient and laplacian, since div is not
    # implemented for single components.
    inp.labels = ["xx", "yy", "zz"]
    tensor_v = LabelTensor(func_vector(inp), ["aa", "bbb", "c"])
    comp = tensor_v.labels[0]
    single_d = inp.labels[0]

    # Single component as string + list of d
    grad_tensor_v = grad(output_=tensor_v, input_=inp, components=comp, d=None)
    assert grad_tensor_v.labels == [f"d{comp}d{i}" for i in inp.labels]

    lap_tensor_v = laplacian(
        output_=tensor_v, input_=inp, components=comp, d=None, method=method
    )
    assert lap_tensor_v.labels == [f"dd{comp}"]

    # Single component as list + list of d
    grad_tensor_v = grad(
        output_=tensor_v, input_=inp, components=[comp], d=None
    )
    assert grad_tensor_v.labels == [f"d{comp}d{i}" for i in inp.labels]

    lap_tensor_v = laplacian(
        output_=tensor_v, input_=inp, components=[comp], d=None, method=method
    )
    assert lap_tensor_v.labels == [f"dd{comp}"]

    # List of components + single d as string
    grad_tensor_v = grad(
        output_=tensor_v, input_=inp, components=None, d=single_d
    )
    assert grad_tensor_v.labels == [f"d{i}d{single_d}" for i in tensor_v.labels]

    lap_tensor_v = laplacian(
        output_=tensor_v, input_=inp, components=None, d=single_d, method=method
    )
    assert lap_tensor_v.labels == [f"dd{i}" for i in tensor_v.labels]

    # List of components + single d as list
    grad_tensor_v = grad(
        output_=tensor_v, input_=inp, components=None, d=[single_d]
    )
    assert grad_tensor_v.labels == [f"d{i}d{single_d}" for i in tensor_v.labels]

    lap_tensor_v = laplacian(
        output_=tensor_v,
        input_=inp,
        components=None,
        d=[single_d],
        method=method,
    )
    assert lap_tensor_v.labels == [f"dd{i}" for i in tensor_v.labels]
