import pytest
import torch
from pina.model.block.message_passing import SchnetBlock

# Data for testing
x = torch.rand(10, 4)
pos = torch.rand(10, 3)
edge_index = torch.randint(0, 10, (2, 20))


@pytest.mark.parametrize("node_feature_dim", [1, 3])
def test_constructor(node_feature_dim):

    SchnetBlock(
        node_feature_dim=node_feature_dim,
        hidden_dim=64,
        n_message_layers=2,
        n_update_layers=2,
        n_radial_layers=2,
    )

    # Should fail if node_feature_dim is negative
    with pytest.raises(AssertionError):
        SchnetBlock(node_feature_dim=-1)

    # Should fail if hidden_dim is negative
    with pytest.raises(AssertionError):
        SchnetBlock(node_feature_dim=node_feature_dim, hidden_dim=-1)

    # Should fail if n_message_layers is negative
    with pytest.raises(AssertionError):
        SchnetBlock(node_feature_dim=node_feature_dim, n_message_layers=-1)

    # Should fail if n_update_layers is negative
    with pytest.raises(AssertionError):
        SchnetBlock(node_feature_dim=node_feature_dim, n_update_layers=-1)

    # Should fail if n_radial_layers is negative
    with pytest.raises(AssertionError):
        SchnetBlock(node_feature_dim=node_feature_dim, n_radial_layers=-1)


def test_forward():

    model = SchnetBlock(
        node_feature_dim=x.shape[1],
        hidden_dim=64,
        n_message_layers=2,
        n_update_layers=2,
        n_radial_layers=2,
    )

    output_ = model(edge_index=edge_index, x=x, pos=pos)
    assert output_.shape == x.shape


def test_backward():

    model = SchnetBlock(
        node_feature_dim=x.shape[1],
        hidden_dim=64,
        n_message_layers=2,
        n_update_layers=2,
        n_radial_layers=2,
    )

    output_ = model(
        edge_index=edge_index, x=x.requires_grad_(), pos=pos.requires_grad_()
    )

    loss = torch.mean(output_)
    loss.backward()
    assert x.grad.shape == x.shape


def test_invariance():

    # Graph to be fully connected and undirected
    edge_index = torch.combinations(torch.arange(x.shape[0]), r=2).T
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    # Random rotation (det(rotation) should be 1)
    rotation = torch.linalg.qr(torch.rand(pos.shape[-1], pos.shape[-1])).Q
    if torch.det(rotation) < 0:
        rotation[:, 0] *= -1

    # Random translation
    translation = torch.rand(1, pos.shape[-1])

    model = SchnetBlock(node_feature_dim=x.shape[1]).eval()

    out1 = model(edge_index=edge_index, x=x, pos=pos)
    out2 = model(edge_index=edge_index, x=x, pos=pos @ rotation.T + translation)

    assert torch.allclose(out1, out2, atol=1e-5)
