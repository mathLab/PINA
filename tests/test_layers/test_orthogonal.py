import torch
import pytest
from pina.model.layers.orthogonal import OrthogonalBlock

list_matrices = [
    torch.randn(10, 3),
    torch.rand(100, 5),
    torch.randn(5, 5),
    ]

list_prohibited_matrices_dim0 = list_matrices[:-1]

def test_constructor():
    orth = OrthogonalBlock(1) 
    orth = OrthogonalBlock(0) 
    orth = OrthogonalBlock()

@pytest.mark.parametrize("V", list_matrices)
def test_forward(V):
    orth = OrthogonalBlock()
    orth_row = OrthogonalBlock(0)
    V_orth = orth(V)
    V_orth_row = orth_row(V.T)
    assert torch.allclose(V_orth.T @ V_orth, torch.eye(V.shape[1]), atol=1e-6)
    assert torch.allclose(V_orth_row @ V_orth_row.T, torch.eye(V.shape[1]), atol=1e-6)

@pytest.mark.parametrize("V", list_prohibited_matrices_dim0)
def test_forward_prohibited(V):
    orth = OrthogonalBlock(0)
    with pytest.raises(Warning):
        V_orth = orth(V)
        assert V.shape[0] > V.shape[1]

