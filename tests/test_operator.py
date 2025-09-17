import torch
import pytest
from pina import LabelTensor
from pina.operator import grad, div, laplacian, advection


class Function(object):

    def __iter__(self):
        functions = [
            (
                getattr(self, f"{name}_input"),
                getattr(self, f"{name}"),
                getattr(self, f"{name}_grad"),
                getattr(self, f"{name}_div"),
                getattr(self, f"{name}_lap"),
            )
            for name in [
                "scalar_scalar",
                "scalar_vector",
                "vector_scalar",
                "vector_vector",
            ]
        ]
        return iter(functions)

    # Scalar to scalar function
    @staticmethod
    def scalar_scalar(x):
        return x**2

    @staticmethod
    def scalar_scalar_grad(x):
        return 2 * x

    @staticmethod
    def scalar_scalar_div(x):
        return 2 * x

    @staticmethod
    def scalar_scalar_lap(x):
        return 2 * torch.ones_like(x)

    @staticmethod
    def scalar_scalar_input():
        input_ = torch.rand((20, 1), requires_grad=True)
        return LabelTensor(input_, ["x"])

    # Scalar to vector function
    @staticmethod
    def scalar_vector(x):
        u = x**2
        v = x**3 + x
        return torch.cat((u, v), dim=-1)

    @staticmethod
    def scalar_vector_grad(x):
        u = 2 * x
        v = 3 * x**2 + 1
        return torch.cat((u, v), dim=-1)

    @staticmethod
    def scalar_vector_div(x):
        return ValueError

    @staticmethod
    def scalar_vector_lap(x):
        u = 2 * torch.ones_like(x)
        v = 6 * x
        return torch.cat((u, v), dim=-1)

    @staticmethod
    def scalar_vector_input():
        input_ = torch.rand((20, 1), requires_grad=True)
        return LabelTensor(input_, ["x"])

    # Vector to scalar function
    @staticmethod
    def vector_scalar(x):
        return torch.prod(x**2, dim=-1, keepdim=True)

    @staticmethod
    def vector_scalar_grad(x):
        return 2 * torch.prod(x**2, dim=-1, keepdim=True) / x

    @staticmethod
    def vector_scalar_div(x):
        return ValueError

    @staticmethod
    def vector_scalar_lap(x):
        return 2 * torch.sum(
            torch.prod(x**2, dim=-1, keepdim=True) / x**2,
            dim=-1,
            keepdim=True,
        )

    @staticmethod
    def vector_scalar_input():
        input_ = torch.rand((20, 2), requires_grad=True)
        return LabelTensor(input_, ["x", "yy"])

    # Vector to vector function
    @staticmethod
    def vector_vector(x):
        u = torch.prod(x**2, dim=-1, keepdim=True)
        v = torch.sum(x**2, dim=-1, keepdim=True)
        return torch.cat((u, v), dim=-1)

    @staticmethod
    def vector_vector_grad(x):
        u = 2 * torch.prod(x**2, dim=-1, keepdim=True) / x
        v = 2 * x
        return torch.cat((u, v), dim=-1)

    @staticmethod
    def vector_vector_div(x):
        u = 2 * torch.prod(x**2, dim=-1, keepdim=True) / x[..., 0]
        v = 2 * x[..., 1]
        return u + v

    @staticmethod
    def vector_vector_lap(x):
        u = torch.sum(
            2 * torch.prod(x**2, dim=-1, keepdim=True) / x**2,
            dim=-1,
            keepdim=True,
        )
        v = 2 * x.shape[-1] * torch.ones_like(u)
        return torch.cat((u, v), dim=-1)

    @staticmethod
    def vector_vector_input():
        input_ = torch.rand((20, 2), requires_grad=True)
        return LabelTensor(input_, ["x", "yy"])


@pytest.mark.parametrize(
    "f",
    Function(),
    ids=["scalar_scalar", "scalar_vector", "vector_scalar", "vector_vector"],
)
def test_gradient(f):

    # Unpack the function
    func_input, func, func_grad, _, _ = f

    # Define input and output
    input_ = func_input()
    output_ = func(input_)
    labels = [f"u{i}" for i in range(output_.shape[-1])]
    output_ = LabelTensor(output_, labels)

    # Compute the true gradient and the pina gradient
    pina_grad = grad(output_=output_, input_=input_)
    true_grad = func_grad(input_)

    # Check the shape and labels of the gradient
    n_components = len(output_.labels) * len(input_.labels)
    assert pina_grad.shape == (*output_.shape[:-1], n_components)
    assert pina_grad.labels == [
        f"d{c}d{i}" for c in output_.labels for i in input_.labels
    ]

    # Compare the values
    assert torch.allclose(pina_grad, true_grad)

    # Test if labels are handled correctly
    grad(output_=output_, input_=input_, components=output_.labels[0])
    grad(output_=output_, input_=input_, d=input_.labels[0])

    # Should fail if input not a LabelTensor
    with pytest.raises(TypeError):
        grad(output_=output_, input_=input_.tensor)

    # Should fail if output not a LabelTensor
    with pytest.raises(TypeError):
        grad(output_=output_.tensor, input_=input_)

    # Should fail for non-existent input labels
    with pytest.raises(RuntimeError):
        grad(output_=output_, input_=input_, d=["x", "y"])

    # Should fail for non-existent output labels
    with pytest.raises(RuntimeError):
        grad(output_=output_, input_=input_, components=["a", "b", "c"])


@pytest.mark.parametrize(
    "f",
    Function(),
    ids=["scalar_scalar", "scalar_vector", "vector_scalar", "vector_vector"],
)
def test_divergence(f):

    # Unpack the function
    func_input, func, _, func_div, _ = f

    # Define input and output
    input_ = func_input()
    output_ = func(input_)
    labels = [f"u{i}" for i in range(output_.shape[-1])]
    output_ = LabelTensor(output_, labels)

    # Scalar to vector or vector to scalar functions
    if func_div(input_) == ValueError:
        with pytest.raises(ValueError):
            div(output_=output_, input_=input_)

    # Scalar to scalar or vector to vector functions
    else:
        # Compute the true divergence and the pina divergence
        pina_div = div(output_=output_, input_=input_)
        true_div = func_div(input_)

        # Check the shape and labels of the divergence
        assert pina_div.shape == (*output_.shape[:-1], 1)
        tmp_labels = [
            f"d{c}d{d_}" for c, d_ in zip(output_.labels, input_.labels)
        ]
        assert pina_div.labels == ["+".join(tmp_labels)]

        # Compare the values
        assert torch.allclose(pina_div, true_div)

        # Test if labels are handled correctly. Performed in a single call to
        # avoid components and d having different lengths.
        div(
            output_=output_,
            input_=input_,
            components=output_.labels[0],
            d=input_.labels[0],
        )

        # Should fail if input not a LabelTensor
        with pytest.raises(TypeError):
            div(output_=output_, input_=input_.tensor)

        # Should fail if output not a LabelTensor
        with pytest.raises(TypeError):
            div(output_=output_.tensor, input_=input_)

        # Should fail for non-existent labels
        with pytest.raises(RuntimeError):
            div(output_=output_, input_=input_, d=["x", "y"])

        with pytest.raises(RuntimeError):
            div(output_=output_, input_=input_, components=["a", "b", "c"])


@pytest.mark.parametrize(
    "f",
    Function(),
    ids=["scalar_scalar", "scalar_vector", "vector_scalar", "vector_vector"],
)
@pytest.mark.parametrize("method", ["std", "divgrad"])
def test_laplacian(f, method):

    # Unpack the function
    func_input, func, _, _, func_lap = f

    # Define input and output
    input_ = func_input()
    output_ = func(input_)
    labels = [f"u{i}" for i in range(output_.shape[-1])]
    output_ = LabelTensor(output_, labels)

    # Compute the true laplacian and the pina laplacian
    pina_lap = laplacian(output_=output_, input_=input_, method=method)
    true_lap = func_lap(input_)

    # Check the shape and labels of the laplacian
    assert pina_lap.shape == output_.shape
    assert pina_lap.labels == [f"dd{l}" for l in output_.labels]

    # Compare the values
    assert torch.allclose(pina_lap, true_lap)

    # Test if labels are handled correctly
    laplacian(
        output_=output_,
        input_=input_,
        components=output_.labels[0],
        method=method,
    )
    laplacian(output_=output_, input_=input_, d=input_.labels[0], method=method)

    # Should fail if input not a LabelTensor
    with pytest.raises(TypeError):
        laplacian(output_=output_, input_=input_.tensor, method=method)

    # Should fail if output not a LabelTensor
    with pytest.raises(TypeError):
        laplacian(output_=output_.tensor, input_=input_, method=method)

    # Should fail for non-existent input labels
    with pytest.raises(RuntimeError):
        laplacian(output_=output_, input_=input_, d=["x", "y"], method=method)

    # Should fail for non-existent output labels
    with pytest.raises(RuntimeError):
        laplacian(
            output_=output_,
            input_=input_,
            components=["a", "b", "c"],
            method=method,
        )


def test_advection_scalar():

    # Define 3-dimensional input
    input_ = torch.rand((20, 3), requires_grad=True)
    input_ = LabelTensor(input_, ["x", "y", "z"])

    # Define 3-dimensional velocity field and quantity to be advected
    velocity = torch.rand((20, 3), requires_grad=True)
    field = torch.sum(input_**2, dim=-1, keepdim=True)

    # Combine velocity and field into a LabelTensor
    labels = ["ux", "uy", "uz", "c"]
    output_ = LabelTensor(torch.cat((velocity, field), dim=1), labels)

    # Compute the pina advection
    components = ["c"]
    pina_adv = advection(
        output_=output_,
        input_=input_,
        velocity_field=["ux", "uy", "uz"],
        components=components,
        d=["x", "y", "z"],
    )

    # Compute the true advection
    grads = 2 * input_
    true_adv = torch.sum(grads * velocity, dim=grads.ndim - 1, keepdim=True)

    # Check the shape, labels, and value of the advection
    assert pina_adv.shape == (*output_.shape[:-1], len(components))
    assert pina_adv.labels == ["adv_c"]
    assert torch.allclose(pina_adv, true_adv)

    # Should fail if input not a LabelTensor
    with pytest.raises(TypeError):
        advection(
            output_=output_,
            input_=input_.tensor,
            velocity_field=["ux", "uy", "uz"],
        )

    # Should fail if output not a LabelTensor
    with pytest.raises(TypeError):
        advection(
            output_=output_.tensor,
            input_=input_,
            velocity_field=["ux", "uy", "uz"],
        )

    # Should fail for non-existent input labels
    with pytest.raises(RuntimeError):
        advection(
            output_=output_,
            input_=input_,
            d=["x", "a"],
            velocity_field=["ux", "uy", "uz"],
        )

    # Should fail for non-existent output labels
    with pytest.raises(RuntimeError):
        advection(
            output_=output_,
            input_=input_,
            components=["a", "b", "c"],
            velocity_field=["ux", "uy", "uz"],
        )

    # Should fail if velocity_field labels are not present in the output labels
    with pytest.raises(RuntimeError):
        advection(
            output_=output_,
            input_=input_,
            velocity_field=["ux", "uy", "nonexistent"],
            components=["c"],
        )

    # Should fail if velocity_field dimensionality does not match input tensor
    with pytest.raises(RuntimeError):
        advection(
            output_=output_,
            input_=input_,
            velocity_field=["ux", "uy"],
            components=["c"],
        )


def test_advection_vector():

    # Define 3-dimensional input
    input_ = torch.rand((20, 3), requires_grad=True)
    input_ = LabelTensor(input_, ["x", "y", "z"])

    # Define 3-dimensional velocity field
    velocity = torch.rand((20, 3), requires_grad=True)

    # Define 2-dimensional field to be advected
    field_1 = torch.sum(input_**2, dim=-1, keepdim=True)
    field_2 = torch.sum(input_**3, dim=-1, keepdim=True)

    # Combine velocity and field into a LabelTensor
    labels = ["ux", "uy", "uz", "c1", "c2"]
    output_ = LabelTensor(
        torch.cat((velocity, field_1, field_2), dim=1), labels
    )

    # Compute the pina advection
    components = ["c1", "c2"]
    pina_adv = advection(
        output_=output_,
        input_=input_,
        velocity_field=["ux", "uy", "uz"],
        components=components,
        d=["x", "y", "z"],
    )

    # Compute the true gradients of the fields "c1", "c2"
    grads1 = 2 * input_
    grads2 = 3 * input_**2

    # Compute the true advection for each field
    true_adv1 = torch.sum(grads1 * velocity, dim=grads1.ndim - 1, keepdim=True)
    true_adv2 = torch.sum(grads2 * velocity, dim=grads2.ndim - 1, keepdim=True)
    true_adv = torch.cat((true_adv1, true_adv2), dim=-1)

    # Check the shape, labels, and value of the advection
    assert pina_adv.shape == (*output_.shape[:-1], len(components))
    assert pina_adv.labels == ["adv_c1", "adv_c2"]
    assert torch.allclose(pina_adv, true_adv)

    # Should fail if input not a LabelTensor
    with pytest.raises(TypeError):
        advection(
            output_=output_,
            input_=input_.tensor,
            velocity_field=["ux", "uy", "uz"],
        )

    # Should fail if output not a LabelTensor
    with pytest.raises(TypeError):
        advection(
            output_=output_.tensor,
            input_=input_,
            velocity_field=["ux", "uy", "uz"],
        )

    # Should fail for non-existent input labels
    with pytest.raises(RuntimeError):
        advection(
            output_=output_,
            input_=input_,
            d=["x", "a"],
            velocity_field=["ux", "uy", "uz"],
        )

    # Should fail for non-existent output labels
    with pytest.raises(RuntimeError):
        advection(
            output_=output_,
            input_=input_,
            components=["a", "b", "c"],
            velocity_field=["ux", "uy", "uz"],
        )

    # Should fail if velocity_field labels are not present in the output labels
    with pytest.raises(RuntimeError):
        advection(
            output_=output_,
            input_=input_,
            velocity_field=["ux", "uy", "nonexistent"],
            components=["c"],
        )

    # Should fail if velocity_field dimensionality does not match input tensor
    with pytest.raises(RuntimeError):
        advection(
            output_=output_,
            input_=input_,
            velocity_field=["ux", "uy"],
            components=["c"],
        )
