import torch
import pytest
from pina.model.block import OrthogonalBlock

torch.manual_seed(111)

list_matrices = [
    torch.randn(10, 3),
    torch.rand(100, 5),
    torch.randn(5, 5),
]

list_prohibited_matrices_dim0 = list_matrices[:-1]


@pytest.mark.parametrize("dim", [-1, 0, 1, None])
@pytest.mark.parametrize("requires_grad", [True, False, None])
def test_constructor(dim, requires_grad):
    if dim is None and requires_grad is None:
        block = OrthogonalBlock()
    elif dim is None:
        block = OrthogonalBlock(requires_grad=requires_grad)
    elif requires_grad is None:
        block = OrthogonalBlock(dim=dim)
    else:
        block = OrthogonalBlock(dim=dim, requires_grad=requires_grad)

    if dim is not None:
        assert block.dim == dim
    if requires_grad is not None:
        assert block.requires_grad == requires_grad


def test_wrong_constructor():
    with pytest.raises(IndexError):
        OrthogonalBlock(2)
    with pytest.raises(ValueError):
        OrthogonalBlock("a")


@pytest.mark.parametrize("V", list_matrices)
def test_forward(V):
    orth = OrthogonalBlock()
    orth_row = OrthogonalBlock(0)
    V_orth = orth(V)
    V_orth_row = orth_row(V.T)
    assert torch.allclose(V_orth.T @ V_orth, torch.eye(V.shape[1]), atol=1e-6)
    assert torch.allclose(
        V_orth_row @ V_orth_row.T, torch.eye(V.shape[1]), atol=1e-6
    )


@pytest.mark.parametrize("V", list_matrices)
def test_backward(V):
    orth = OrthogonalBlock(requires_grad=True)
    V_orth = orth(V)
    loss = V_orth.mean()
    loss.backward()


@pytest.mark.parametrize("V", list_matrices)
def test_wrong_backward(V):
    orth = OrthogonalBlock(requires_grad=False)
    V_orth = orth(V)
    loss = V_orth.mean()
    with pytest.raises(RuntimeError):
        loss.backward()


@pytest.mark.parametrize("V", list_prohibited_matrices_dim0)
def test_forward_prohibited(V):
    orth = OrthogonalBlock(0)
    with pytest.raises(Warning):
        V_orth = orth(V)
        assert V.shape[0] > V.shape[1]
