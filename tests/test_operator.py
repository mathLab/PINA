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
def test_laplacian(f):

    # Unpack the function
    func_input, func, _, _, func_lap = f

    # Define input and output
    input_ = func_input()
    output_ = func(input_)
    labels = [f"u{i}" for i in range(output_.shape[-1])]
    output_ = LabelTensor(output_, labels)

    # Compute the true laplacian and the pina laplacian
    pina_lap = laplacian(output_=output_, input_=input_)
    true_lap = func_lap(input_)

    # Check the shape and labels of the laplacian
    assert pina_lap.shape == output_.shape
    assert pina_lap.labels == [f"dd{l}" for l in output_.labels]

    # Compare the values
    assert torch.allclose(pina_lap, true_lap)

    # Test if labels are handled correctly
    laplacian(output_=output_, input_=input_, components=output_.labels[0])
    laplacian(output_=output_, input_=input_, d=input_.labels[0])

    # Should fail if input not a LabelTensor
    with pytest.raises(TypeError):
        laplacian(output_=output_, input_=input_.tensor)

    # Should fail if output not a LabelTensor
    with pytest.raises(TypeError):
        laplacian(output_=output_.tensor, input_=input_)

    # Should fail for non-existent input labels
    with pytest.raises(RuntimeError):
        laplacian(output_=output_, input_=input_, d=["x", "y"])

    # Should fail for non-existent output labels
    with pytest.raises(RuntimeError):
        laplacian(output_=output_, input_=input_, components=["a", "b", "c"])


def test_advection():

    # Define input and output
    input_ = torch.rand((20, 3), requires_grad=True)
    input_ = LabelTensor(input_, ["x", "y", "z"])
    output_ = LabelTensor(input_**2, ["u", "v", "c"])

    # Define the velocity field
    velocity = output_.extract(["c"])

    # Compute the true advection and the pina advection
    pina_advection = advection(
        output_=output_, input_=input_, velocity_field="c"
    )
    true_advection = velocity * 2 * input_.extract(["x", "y"])

    # Check the shape of the advection
    assert pina_advection.shape == (*output_.shape[:-1], output_.shape[-1] - 1)
    assert torch.allclose(pina_advection, true_advection)
