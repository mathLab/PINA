import torch
import pytest

from pina.model.layers import PeriodicBoundaryEmbedding
from pina import LabelTensor

def check_same_columns(tensor):
    # Get the first column
    first_column = tensor[0]
    # Compare each column with the first column
    all_same = torch.allclose(tensor, first_column, rtol=0.)
    return all_same

def grad(u, x):
    """
    Compute the first derivative of u with respect to x.
    """
    return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                               create_graph=True, allow_unused=True,
                               retain_graph=True)[0]

def test_constructor():
    PeriodicBoundaryEmbedding(input_dimension=1, periods=2)
    PeriodicBoundaryEmbedding(input_dimension=1, periods={'x': 3, 'y' : 4})
    PeriodicBoundaryEmbedding(input_dimension=1, periods={0: 3, 1 : 4})
    PeriodicBoundaryEmbedding(input_dimension=1, periods=2, output_dimension=10)
    with pytest.raises(TypeError):
        PeriodicBoundaryEmbedding()
    with pytest.raises(ValueError):
        PeriodicBoundaryEmbedding(input_dimension=1., periods=1)
        PeriodicBoundaryEmbedding(input_dimension=1, periods=1, output_dimension=1.)
        PeriodicBoundaryEmbedding(input_dimension=1, periods={'x':'x'})
        PeriodicBoundaryEmbedding(input_dimension=1, periods={0:'x'})


@pytest.mark.parametrize("period", [1, 4, 10])
@pytest.mark.parametrize("input_dimension", [1, 2, 3])
def test_forward_same_period(input_dimension, period):
    func = torch.nn.Sequential(
        PeriodicBoundaryEmbedding(input_dimension=input_dimension,
                     output_dimension=60, periods=period),
        torch.nn.Tanh(),
        torch.nn.Linear(60, 60),
        torch.nn.Tanh(),
        torch.nn.Linear(60, 1)
    )
    # coordinates
    x = period * torch.tensor([[0.],[1.]])
    if input_dimension == 2:
        x = torch.cartesian_prod(x.flatten(),x.flatten())
    elif input_dimension == 3:
        x = torch.cartesian_prod(x.flatten(),x.flatten(),x.flatten())
    x.requires_grad = True
    # output
    f = func(x)
    assert check_same_columns(f)



def test_forward_same_period_labels():
    func = torch.nn.Sequential(
        PeriodicBoundaryEmbedding(input_dimension=2,
                     output_dimension=60, periods={'x':1, 'y':2}),
        torch.nn.Tanh(),
        torch.nn.Linear(60, 60),
        torch.nn.Tanh(),
        torch.nn.Linear(60, 1)
    )
    # coordinates
    tensor = torch.tensor([[0., 0.], [0., 2.], [1., 0.], [1., 2.]])
    with pytest.raises(RuntimeError):
        func(tensor)
    tensor = tensor.as_subclass(LabelTensor)
    tensor.labels = ['x', 'y']
    tensor.requires_grad = True
    # output
    f = func(tensor)
    assert check_same_columns(f)

def test_forward_same_period_index():
    func = torch.nn.Sequential(
        PeriodicBoundaryEmbedding(input_dimension=2,
                     output_dimension=60, periods={0:1, 1:2}),
        torch.nn.Tanh(),
        torch.nn.Linear(60, 60),
        torch.nn.Tanh(),
        torch.nn.Linear(60, 1)
    )
    # coordinates
    tensor = torch.tensor([[0., 0.], [0., 2.], [1., 0.], [1., 2.]])
    tensor.requires_grad = True
    # output
    f = func(tensor)
    assert check_same_columns(f)
    tensor = tensor.as_subclass(LabelTensor)
    tensor.labels = ['x', 'y']
    # output
    f = func(tensor)
    assert check_same_columns(f)