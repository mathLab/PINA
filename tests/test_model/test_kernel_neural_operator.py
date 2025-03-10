import torch
from pina.model import KernelNeuralOperator, FeedForward

input_dim = 2
output_dim = 4
embedding_dim = 24
batch_size = 10
numb = 256
data = torch.rand(size=(batch_size, numb, input_dim), requires_grad=True)
output_shape = torch.Size([batch_size, numb, output_dim])


lifting_operator = FeedForward(
    input_dimensions=input_dim, output_dimensions=embedding_dim
)
projection_operator = FeedForward(
    input_dimensions=embedding_dim, output_dimensions=output_dim
)
integral_kernels = torch.nn.Sequential(
    FeedForward(
        input_dimensions=embedding_dim, output_dimensions=embedding_dim
    ),
    FeedForward(
        input_dimensions=embedding_dim, output_dimensions=embedding_dim
    ),
)


def test_constructor():
    KernelNeuralOperator(
        lifting_operator=lifting_operator,
        integral_kernels=integral_kernels,
        projection_operator=projection_operator,
    )


def test_forward():
    operator = KernelNeuralOperator(
        lifting_operator=lifting_operator,
        integral_kernels=integral_kernels,
        projection_operator=projection_operator,
    )
    out = operator(data)
    assert out.shape == output_shape


def test_backward():
    operator = KernelNeuralOperator(
        lifting_operator=lifting_operator,
        integral_kernels=integral_kernels,
        projection_operator=projection_operator,
    )
    out = operator(data)
    loss = torch.nn.functional.mse_loss(out, torch.zeros_like(out))
    loss.backward()
    grad = data.grad
    assert grad.shape == data.shape
